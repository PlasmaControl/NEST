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

import jax
import jax.numpy as jnp
import numpy as np

import desc.io
from desc.equilibrium import Equilibrium
from desc.grid import Grid, LinearGrid, QuadratureGrid
from desc.magnetic_fields import SumMagneticField

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


def _load_field(field):
    """A field object from a file, a field object, or a list of those (summed)."""
    if isinstance(field, (list, tuple)):
        return SumMagneticField([_load_field(f) for f in field])
    if isinstance(field, (str, os.PathLike)):
        return desc.io.load(field)
    return field


def covariant_basis(RZ, rho, theta, zeta):
    """Tangent basis and Jacobian of the geometry series ``RZ`` (columns R, Z) at the points.

    Returns (X, E, sqrt_g): X (n, 2) = (R, Z), E (n, 3, 3) the rows e_rho = (R_r, 0, Z_r),
    e_theta = (R_t, 0, Z_t), e_zeta = (R_z, R, Z_z) in (R, phi, Z) components
    (phi = zeta), and sqrt(g) = e_rho . e_theta x e_zeta = R (R_t Z_r - R_r Z_t).
    Pure JAX.
    """
    X, Xr = RZ.evaluate(rho, theta, zeta), RZ.evaluate(rho, theta, zeta, dr=1)
    Xt, Xz = RZ.evaluate(rho, theta, zeta, dt=1), RZ.evaluate(rho, theta, zeta, dz=1)
    R, zero = X[:, 0], jnp.zeros_like(rho)
    E = jnp.stack([jnp.stack([Xr[:, 0], zero, Xr[:, 1]], -1),
                   jnp.stack([Xt[:, 0], zero, Xt[:, 1]], -1),
                   jnp.stack([Xz[:, 0], R, Xz[:, 1]], -1)], axis=1)
    return X, E, R * (Xt[:, 0] * Xr[:, 1] - Xr[:, 0] * Xt[:, 1])


def dual_basis(E, sqrt_g):
    """grad(rho), grad(theta), grad(zeta) as rows, (n, 3, 3): e_theta x e_zeta / sqrt(g) etc."""
    return jnp.stack([jnp.cross(E[:, 1], E[:, 2]), jnp.cross(E[:, 2], E[:, 0]),
                      jnp.cross(E[:, 0], E[:, 1])], axis=1) / sqrt_g[:, None, None]


def current_from_sqrtg(sJ, E, sqrt_g):
    """J = sum_i sqrt(g) J^i e_i / sqrt(g) in the components of E.

    ``sJ`` (n, ..., 3) the sqrt(g) J^i, ``E`` (n, 3, 3) the tangent basis as rows
    (``desc_input.covariant_basis``), ``sqrt_g`` (n,). Returns (n, ..., 3). For
    rho > 0 every term of the sum is O(rho) on its own, so there is no
    cancellation near the axis; at rho = 0 see ``_current``.
    """
    return jnp.einsum("n...i,nia->n...a", sJ, E) / sqrt_g.reshape((-1,) + (1,) * (sJ.ndim - 1))


@functools.partial(jax.jit, static_argnums=0)
def _basis_chunk(RZ, rho, theta, zeta):
    X, E, sqrt_g = covariant_basis(RZ, rho, theta, zeta)
    return jnp.concatenate([X, E.reshape(-1, 9), dual_basis(E, sqrt_g).reshape(-1, 9),
                            sqrt_g[:, None]], axis=1)


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
        """From an equilibrium file. ``field`` is a file, a field object, a list of those
        (their sum, ``SumMagneticField``: DESC cannot save a ``SumMagneticField``, so a
        field of several parts is kept as one file per part) or None."""
        eq = _load_equilibrium(path)
        return cls(eq, _load_field(field), name or os.path.basename(str(path)), check)

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

    def basis_vectors(self, rtz, chunk=50000):
        """Geometry at flux coordinates ``rtz`` (n, 3), NumPy, vectors in Cartesian
        components: E (n, 3, 3) tangent basis (rows e_rho, e_theta, e_zeta), G (n, 3,
        3) dual basis (rows grad(rho), grad(theta), grad(zeta)), sqrt_g (n,), x (n, 3)
        the positions and phi (n,) = zeta. From ``geometry`` (``covariant_basis``)."""
        from cartesian_grid import cylindrical_to_cartesian, evaluate_chunked

        rtz = np.asarray(rtz, dtype=float)
        out = evaluate_chunked(functools.partial(_basis_chunk, self.geometry), rtz, chunk)
        phi = rtz[:, 2]
        cart = lambda V: np.stack([cylindrical_to_cartesian(V[:, i], phi) for i in range(3)], 1)
        R, Z = out[:, 0], out[:, 1]
        return {"E": cart(out[:, 2:11].reshape(-1, 3, 3)), "G": cart(out[:, 11:20].reshape(-1, 3, 3)),
                "sqrt_g": out[:, 20], "phi": phi,
                "x": np.column_stack([R * np.cos(phi), R * np.sin(phi), Z])}

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

    # ------------------------------------------------- scattered points
    @functools.cached_property
    def _iota_fit(self):
        """Chebyshev fit (degree 24) of iota on 64 flux surfaces (current prescribed)."""
        r = (1 - np.cos(np.pi * (np.arange(64) + 0.5) / 64)) / 2
        g = LinearGrid(rho=r, M=self.eq.M_grid, N=self.eq.N_grid, NFP=self.NFP)
        return np.polynomial.Chebyshev.fit(r, g.compress(self.compute(["iota"], g)["iota"]),
                                           24, domain=[0, 1])

    def iota(self, rho):
        """(iota, d iota / d rho) at the radii ``rho``: from the profile when iota
        is prescribed, else from a fit of iota on flux surfaces."""
        rho = np.asarray(rho, dtype=float)
        if self.eq.iota is not None:
            g = Grid(np.column_stack([rho, 0 * rho, 0 * rho]), sort=False)
            return [np.asarray(self.eq.iota.compute(g, dr=i)) for i in range(2)]
        return [self._iota_fit(rho), self._iota_fit.deriv()(rho)]

    def compute_points(self, names, rtz, chunk=20000):
        """DESC quantities at scattered flux coordinates ``rtz`` (n, 3), in chunks.

        iota is passed in (``iota``): DESC would otherwise need surface
        integrals, which scattered points do not have. Vectors are in (R, phi, Z).
        """
        out = {k: [] for k in names}
        for i in range(0, len(rtz), chunk):
            x = np.asarray(rtz[i : i + chunk], dtype=float)
            iota, iota_r = self.iota(x[:, 0])
            d = self.eq.compute(names, grid=Grid(x, sort=False),
                                data={"iota": iota, "iota_r": iota_r})
            for k in names:
                out[k].append(np.asarray(d[k]))
        return {k: np.concatenate(v) for k, v in out.items()}

    # --------------------------------------------------------- external field
    def B_ext(self, x, source_grid=None, chunk_size=None, rtz=None):
        """External field at Cartesian points x, shape (n, 3), in (B_x, B_y, B_z) [T].

        ``rtz``: the flux coordinates of the points, if known; a field defined
        through them (``DESCExternalField.at_flux``) then needs no inverse map.
        """
        if self.field is None:
            raise ValueError(f"{self.name}: no external field was given")
        if rtz is not None and hasattr(self.field, "at_flux"):
            return np.asarray(self.field.at_flux(rtz, x))
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
