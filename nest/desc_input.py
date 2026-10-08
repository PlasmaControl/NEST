"""The DESC input NEST starts from: an equilibrium and, optionally, its external field.

Any DESC equilibrium works: fixed or free boundary, any NFP, stellarator symmetric
or not, with the rotational transform or the toroidal current prescribed. It can
be given as a file (an ``Equilibrium``, or an ``EquilibriaFamily`` whose last
member is used) or as an ``Equilibrium`` object. The external field, needed for
B = B_ext + B_plasma, is optional: a coil set or magnetic field, as a file or an
object.

``DESCInput`` gathers what the rest of NEST needs from DESC in one place: the
equilibrium, its resolution and size, the extent of the boundary, the geometry
series R(rho, theta, zeta), Z(rho, theta, zeta) for the JAX evaluators, and B_ext
at Cartesian points. It also checks NEST's assumptions on the input; the
functions of NEST accept either a ``DESCInput`` or a bare ``Equilibrium``
(``equilibrium`` below).
"""

import functools
import os

import numpy as np

import desc.io
from desc.equilibrium import Equilibrium
from desc.grid import LinearGrid, QuadratureGrid

from separable import Separable


def equilibrium(obj):
    """The DESC ``Equilibrium`` of a ``DESCInput`` (or ``obj`` itself)."""
    return obj.eq if isinstance(obj, DESCInput) else obj


def _load_equilibrium(path):
    eq = desc.io.load(path)
    if not isinstance(eq, Equilibrium):  # EquilibriaFamily (or a list of them)
        eq = eq[-1]
    if not isinstance(eq, Equilibrium):
        raise TypeError(f"{path} holds a {type(eq).__name__}, not a DESC equilibrium")
    return eq


class DESCInput:
    """A DESC equilibrium and, optionally, the external magnetic field.

    Parameters
    ----------
    eq : Equilibrium
    field : _MagneticField or None
        External field (e.g. a ``CoilSet``); needed only for B_ext.
    name : str
        Label for printing; defaults to the file name or "equilibrium".
    check : bool
        Run ``check()`` on construction.
    """

    def __init__(self, eq, field=None, name=None, check=True):
        if not isinstance(eq, Equilibrium):
            raise TypeError(f"eq must be a DESC Equilibrium, got {type(eq).__name__}")
        self.eq = eq
        self.field = field
        self.name = name or "equilibrium"
        if check:
            self.check()

    @classmethod
    def load(cls, path, field=None, name=None, check=True):
        """From an equilibrium file; ``field`` is a file, a field object or None."""
        eq = _load_equilibrium(path)
        if isinstance(field, (str, os.PathLike)):
            field = desc.io.load(field)
        return cls(eq, field, name or os.path.basename(str(path)), check)

    # ----------------------------------------------------------------- basics
    @property
    def NFP(self):
        return int(self.eq.NFP)

    @property
    def resolution(self):
        """(L, M, N) of the equilibrium."""
        return self.eq.L, self.eq.M, self.eq.N

    @property
    def prescribed_profile(self):
        """"iota" or "current": the profile DESC held fixed."""
        return "iota" if self.eq.iota is not None else "current"

    def compute(self, names, grid):
        """DESC quantities on ``grid`` as NumPy arrays."""
        names = [names] if isinstance(names, str) else list(names)
        return {k: np.asarray(v) for k, v in self.eq.compute(names, grid=grid).items()}

    @functools.cached_property
    def scalars(self):
        """Minor radius a, major radius R0, volume V [m, m, m^3] and <beta>_vol."""
        g = QuadratureGrid(L=2 * self.eq.L, M=2 * self.eq.M, N=2 * self.eq.N, NFP=self.NFP)
        d = self.compute(["a", "R0", "V", "<beta>_vol"], g)
        return {k: float(np.ravel(d[k])[0]) for k in ("a", "R0", "V", "<beta>_vol")}

    @property
    def a(self):
        return self.scalars["a"]

    @property
    def R0(self):
        return self.scalars["R0"]

    @property
    def volume(self):
        return self.scalars["V"]

    @functools.lru_cache(maxsize=4)
    def extent(self, M=None, N=None):
        """Bounds of the boundary surface: (R_min, R_max, Z_min, Z_max) [m].

        Sampled on rho = 1 with 4 M + 1 poloidal and 4 N + 1 toroidal points per
        period (defaults from the equilibrium resolution).
        """
        M = 4 * self.eq.M + 1 if M is None else M
        N = 4 * self.eq.N + 1 if N is None else N
        g = LinearGrid(rho=1.0, theta=M, zeta=N, NFP=self.NFP, sym=False)
        d = self.compute(["R", "Z"], g)
        return (float(d["R"].min()), float(d["R"].max()),
                float(d["Z"].min()), float(d["Z"].max()))

    @functools.cached_property
    def geometry(self):
        """R(rho, theta, zeta), Z(rho, theta, zeta) as one ``Separable`` (columns R, Z)."""
        eq = self.eq
        return Separable([eq.R_basis, eq.Z_basis], [eq.R_lmn, eq.Z_lmn], eq.NFP)

    # ---------------------------------------------------------------- checks
    def check(self):
        """Raise if the input breaks NEST's assumptions.

        - DESC's toroidal angle zeta equals the cylindrical angle phi (the inverse
          map X(rho, theta, zeta) = x solves only for rho, theta at zeta = phi);
        - the Jacobian sqrt(g) has one sign inside the plasma (nested, non-
          degenerate flux coordinates of the starting equilibrium).
        """
        g = LinearGrid(rho=np.array([0.5, 1.0]), M=4, N=4, NFP=self.NFP, sym=False)
        d = self.compute(["phi", "zeta"], g)
        err = np.max(np.abs(d["phi"] - d["zeta"]))
        if err > 1e-12:
            raise NotImplementedError(f"{self.name}: phi != zeta (max diff {err:.1e})")
        g = QuadratureGrid(L=self.eq.L, M=self.eq.M, N=self.eq.N, NFP=self.NFP)
        sqrt_g = self.compute("sqrt(g)", g)["sqrt(g)"]
        if not (np.all(sqrt_g > 0) or np.all(sqrt_g < 0)):
            raise ValueError(f"{self.name}: sqrt(g) changes sign (bad flux coordinates)")
        return self

    # --------------------------------------------------------- external field
    def B_ext(self, x, source_grid=None, chunk_size=None):
        """External field at Cartesian points x, shape (n, 3), in (B_x, B_y, B_z) [T]."""
        if self.field is None:
            raise ValueError(f"{self.name}: no external field was given")
        return np.asarray(self.field.compute_magnetic_field(
            np.asarray(x, dtype=float), basis="xyz", source_grid=source_grid,
            chunk_size=chunk_size))

    # ------------------------------------------------------------------ info
    def summary(self):
        """Dictionary of the main properties, for printing or logging."""
        L, M, N = self.resolution
        return {
            "name": self.name,
            "NFP": self.NFP,
            "L, M, N": (L, M, N),
            "sym": bool(self.eq.sym),
            "prescribed": self.prescribed_profile,
            "a [m]": self.a,
            "R0 [m]": self.R0,
            "V [m^3]": self.volume,
            "<beta>": self.scalars["<beta>_vol"],
            "external field": None if self.field is None else type(self.field).__name__,
        }

    def __repr__(self):
        s = self.summary()
        return (f"DESCInput({s['name']}: NFP={s['NFP']}, (L,M,N)={s['L, M, N']}, "
                f"{s['prescribed']} prescribed, a={s['a [m]']:.4g} m, "
                f"R0={s['R0 [m]']:.4g} m, <beta>={s['<beta>']:.3g}, "
                f"field={s['external field']})")
