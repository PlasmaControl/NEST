"""A radial B-spline x Fourier basis for the a and b series of the current potential
(``boundary_potential.BoundaryPotential``, ``radial="spline"``).

    a, b = sum_{(j, m, n)} c_jmn B_j(rho^2) F_m(theta) F_n(NFP zeta)

with DESC's real Fourier functions F (cos for m >= 0, sin for m < 0; the same for n) and B_j clamped cubic
B-splines in s = rho^2 with breakpoints s_i = rho_i^2, the rho_i spaced ``coarse`` and ``dense`` within ``half``
of each centre (rational surfaces). m = 0 uses every spline; m != 0 drops the first three (value, slope or
curvature nonzero at s = 0), so every m != 0 term is O(s^3) = O(rho^6) at the axis (B_3 ~ s^3 on [0, s_1], the
others zero there) and a, b are bounded, C^5 scalars there (smooth for even |m| <= 6). The potential's structure (chi, the window f, h_0, I) is
unchanged, so div J = 0, J = 0 on the LCFS and the net current hold for any coefficients, as with
Fourier-Zernike a, b.

``reg="rhom"``: every spline for every m, as rho^q_m B_j(rho^2) scaled to unit maximum (q_m = |m| up to
``q_cap``, then capped with the parity of m; ``spline_pressure.q_of_m``): Zernike-type regularity for any
coefficients instead of dropping the three axis splines (``reg="drop3"``, the default).

``SplineBasis`` has the attributes the rest of NEST reads from a ``FourierZernikeBasis`` (modes as (l, m, n)
with l the spline index, L, M, N, NFP, sym, num_modes, spectral_indexing); its L is an effective radial
resolution, round(1 / dense), so grids sized by L (e.g. the uniform flux grid of ``objective.Fields``, 4 L radii)
put a few points in every fine knot interval. ``SplineSeparable`` is ``separable.Separable`` with B_j(rho^2)
as the radial factor (values and rho-derivatives up to 2, in JAX).
"""

import numpy as np
import jax.numpy as jnp

from desc.basis import fourier

from separable import Separable
from spline_pressure import (breakpoints, knots_s, q_of_m, radial_factor, radial_scale, rational_centres,
                             spline_keep)


def _bspline_all(t, x, k):
    """All degree-k B-splines on the knots t at x (JAX), (n, len(t) - k - 1); the last nonempty interval closed."""
    t = np.asarray(t, float)
    B = [((x >= t[i]) & (x < t[i + 1])).astype(x.dtype) for i in range(len(t) - 1)]
    last = int(np.flatnonzero(np.diff(t) > 0)[-1])
    B[last] = jnp.where(x == t[last + 1], 1.0, B[last])
    for d in range(1, k + 1):
        nb = []
        for i in range(len(t) - 1 - d):
            a = (x - t[i]) / (t[i + d] - t[i]) if t[i + d] > t[i] else 0.0
            b = (t[i + d + 1] - x) / (t[i + d + 1] - t[i + 1]) if t[i + d + 1] > t[i + 1] else 0.0
            nb.append(a * B[i] + b * B[i + 1])
        B = nb
    return jnp.stack(B[: len(t) - k - 1], axis=1)


def _deriv_matrix(t, k):
    """D (n_k x n_{k-1}) with d/ds sum c_j B_{j,k} = sum (D^T c)_i B_{i,k-1} on the same knots:
    B'_{j,k} = k (B_{j,k-1} / (t_{j+k} - t_j) - B_{j+1,k-1} / (t_{j+k+1} - t_{j+1}))."""
    t = np.asarray(t, float)
    n_k, n_k1 = len(t) - k - 1, len(t) - k
    D = np.zeros((n_k, n_k1))
    for j in range(n_k):
        d1, d2 = t[j + k] - t[j], t[j + k + 1] - t[j + 1]
        if d1 > 0:
            D[j, j] += k / d1
        if d2 > 0:
            D[j, j + 1] -= k / d2
    return D


def spline_radial(t, rho, dr=0):
    """B_j(rho^2) and its rho-derivatives (dr = 0, 1, 2) for all j, (n, n_s), JAX."""
    s = rho**2
    B = _bspline_all(t, s, 3)
    if dr == 0:
        return B
    D3 = jnp.asarray(_deriv_matrix(t, 3))
    B1 = _bspline_all(t, s, 2) @ D3.T  # dB/ds
    if dr == 1:
        return 2 * rho[:, None] * B1
    D2 = jnp.asarray(_deriv_matrix(t, 2))
    B2 = (_bspline_all(t, s, 1) @ D2.T) @ D3.T  # d2B/ds2
    if dr == 2:
        return 2 * B1 + 4 * (rho**2)[:, None] * B2
    raise ValueError(f"dr = {dr}: 0, 1 or 2")


class SplineBasis:
    """(j, m, n) modes of a radial-spline x Fourier series (module docstring)."""

    spectral_indexing = "spline"

    def __init__(self, bp_rho, M, N, NFP, sym=False, centres=(), m_tol=0.0, m_max=None, m_fz=None,
                 reg="drop3", q_cap=6, inside=None):
        self.bp_rho = np.asarray(bp_rho, float)
        self.t = tuple(float(v) for v in knots_s(self.bp_rho))
        self.n_s = len(self.t) - 4
        self.M, self.N, self.NFP = int(M), int(N), int(NFP)
        self.sym = "cos" if sym else False
        self.centres = list(centres)
        # truncation (spline_pressure.spline_keep): drop splines with rho_end^|m| < m_tol, and |m| > m_max
        self.m_tol, self.m_max = float(m_tol), (None if m_max is None or m_max < 0 else int(m_max))
        self.m_fz = int(m_fz) if m_fz else None
        self.reg, self.q_cap = str(reg), int(q_cap)
        self.inside = [tuple(float(v) for v in x) for x in inside] if inside else None
        if self.reg not in ("drop3", "rhom"):
            raise ValueError(f"reg {reg!r}: 'drop3' or 'rhom'")
        # effective L for grid sizing: 1 / the finest spacing, capped at 100 (spacing 0.01, the q = 2 band): FZ-matched
        # knots have spacings ~1e-3 at the edge, and 1 / that would blow up every grid sized by L; the radial
        # collocation is composite Gauss per knot interval anyway (mf_fixed), so each spline has its own nodes
        dense = max(float(np.diff(self.bp_rho).min()), 0.01)
        self.L = int(round(1.0 / dense))
        modes = []
        for m in range(-self.M, self.M + 1):
            for n in range(-self.N, self.N + 1):
                if self.sym and (m < 0) != (n < 0):  # cos-type: F_m F_n with the same parity
                    continue
                for j in spline_keep(self.t, m, self.m_tol, self.m_max, self.m_fz, self.reg, self.q_cap, self.inside):
                    modes.append((j, m, n))
        self.modes = np.asarray(modes, int)
        self.num_modes = len(self.modes)
        qs = range(0, self.q_cap + 1) if self.reg == "rhom" else [0]
        self._scale = radial_scale(self.t, qs) if self.reg == "rhom" else {0: np.ones(self.n_s)}

    def q(self, m):
        """Axis power of poloidal number m (0 for every m with reg="drop3")."""
        return q_of_m(m, self.q_cap) if self.reg == "rhom" else 0

    def radial(self, js, ms, rho, dr=0):
        """Radial functions of the (j, m) pairs (rho^q_m B_j(rho^2) x scale, or B_j(rho^2)) and their
        rho-derivatives (dr = 0, 1, 2), JAX, (n, len(js))."""
        js, ms = np.asarray(js, int), np.asarray(ms, int)
        B = [spline_radial(self.t, rho, d)[:, js] for d in range(dr + 1)]
        if self.reg == "drop3":
            return B[dr]
        qv = np.array([self.q(m) for m in ms], int)
        sc = jnp.asarray([self._scale[q][j] for q, j in zip(qv, js)])
        r = rho[:, None]
        g = [radial_factor(r, qv[None, :], d) for d in range(dr + 1)]
        if dr == 0:
            out = g[0] * B[0]
        elif dr == 1:
            out = g[1] * B[0] + g[0] * B[1]
        else:
            out = g[2] * B[0] + 2 * g[1] * B[1] + g[0] * B[2]
        return out * sc[None, :]

    @classmethod
    def from_knots(cls, eq, M, N, coarse=0.05, dense=0.01, half=0.08, helicities=("2/1",), sym=None, m_tol=0.0,
                   m_max=None, m_fz=None, fz_knots=None, bands=None, reg="drop3", q_cap=6, inside=None):
        centres = rational_centres(eq, helicities) if helicities else []
        bp = breakpoints(0.0, 1.0, coarse, dense, centres, half, fz_knots, bands)
        return cls(bp, M, N, eq.NFP, bool(eq.sym) if sym is None else sym, centres, m_tol, m_max, m_fz, reg, q_cap,
                   inside)

    def radial_columns(self, m, rho):
        """(j values, B_j(rho^2), d_rho B_j(rho^2)) of the radial functions of |m| (NumPy), the
        ``_fit_blocks`` analogue of Zernike's (l, Z_l^|m|, d_rho Z_l^|m|)."""
        js = np.unique(self.modes[np.abs(self.modes[:, 1]) == m, 0])
        r = jnp.asarray(rho)
        ms = np.full(len(js), m)
        return js, np.asarray(self.radial(js, ms, r, 0)), np.asarray(self.radial(js, ms, r, 1))

    def summary(self):
        return {"kind": "spline", "n_s": self.n_s, "M": self.M, "N": self.N, "num_modes": self.num_modes,
                "breakpoints": len(self.bp_rho), "min spacing": float(np.diff(self.bp_rho).min()),
                "centres": self.centres, "effective L": self.L, "m_tol": self.m_tol, "m_max": self.m_max,
                "m_fz": self.m_fz, "max spacing": float(np.diff(self.bp_rho).max()), "reg": self.reg,
                "q_cap": self.q_cap, "inside": self.inside}


class SplineSeparable(Separable):
    """``Separable`` with B_j(rho^2) F_m(theta) as the poloidal factor (lm = distinct (j, m))."""

    def __init__(self, bases, coefs, NFP):
        self.t = bases[0].t
        self.sbasis = bases[0]
        super().__init__(bases, coefs, NFP)

    def radial_table(self, rho, dr=0):
        """Radial factors of every (j, m) pair, (n_rho, n_lm) (``FactoredFactors``)."""
        return self.sbasis.radial(self.lm[:, 0], self.lm[:, 1], rho, dr)

    def poloidal(self, rho, theta, dr=0, dt=0):
        j, m = self.lm[:, 0], self.lm[:, 1]
        return self.sbasis.radial(j, m, rho, dr) * fourier(theta[:, None], m, dt=dt)


# ------------------------------------------------- Fourier-Zernike + band-localized splines
class CompositeBasis:
    """A Fourier-Zernike basis plus radial B-splines localized inside refined bands ("fz+spline").

    Modes: the FZ (l, m, n), then the splines (L + 1 + j, m, n) of ``sb`` (a ``SplineBasis`` built with
    ``inside`` = the bands: every kept spline has its support inside a band, so it and its first two
    derivatives vanish at the band edges and away from them). The FZ part keeps its spectral accuracy, its axis
    regularity and its edge refinement; the splines add local radial resolution around the resonant surfaces.
    The FZ modes keep their labels, so FZ states embed exactly (``meanforce_fixed.embed_truncate``)."""

    spectral_indexing = "fz+spline"

    def __init__(self, fz, sb):
        self.fz, self.sb = fz, sb
        self.L, self.M, self.N, self.NFP, self.sym = fz.L, fz.M, fz.N, fz.NFP, fz.sym
        self.l0 = int(fz.L) + 1
        sm = np.asarray(sb.modes, int).copy()
        sm[:, 0] += self.l0
        self.modes = np.concatenate([np.asarray(fz.modes, int), sm]) if len(sm) else np.asarray(fz.modes, int)
        self.num_modes, self.n_fz = len(self.modes), int(fz.num_modes)
        self.bp_rho = sb.bp_rho

    def make_separable(self, bases, coefs, NFP):
        return CompositeSeparable(bases, coefs, NFP)

    def radial(self, ls, ms, rho, dr=0):
        """Radial functions of (l, m) pairs (Zernike for l <= L, splines beyond), (n, len(ls)), JAX."""
        from desc.basis import zernike_radial
        ls, ms = np.asarray(ls, int), np.asarray(ms, int)
        fz, sp = np.flatnonzero(ls < self.l0), np.flatnonzero(ls >= self.l0)
        out = jnp.zeros((rho.shape[0], len(ls)))
        if len(fz):
            out = out.at[:, fz].set(zernike_radial(rho[:, None], ls[fz], np.abs(ms[fz]), dr=dr))
        if len(sp):
            out = out.at[:, sp].set(self.sb.radial(ls[sp] - self.l0, ms[sp], rho, dr))
        return out

    def radial_columns(self, m, rho):
        ls = np.unique(self.modes[np.abs(self.modes[:, 1]) == m, 0])
        r = jnp.asarray(rho)
        ms = np.full(len(ls), m)
        return ls, np.asarray(self.radial(ls, ms, r, 0)), np.asarray(self.radial(ls, ms, r, 1))

    def summary(self):
        return {"kind": "fz+spline", "FZ": [self.L, self.M, self.N], "n_fz": self.n_fz,
                "n_spline": self.num_modes - self.n_fz, "num_modes": self.num_modes, "spline": self.sb.summary()}


class CompositeSeparable(Separable):
    """``Separable`` over a ``CompositeBasis`` (Zernike or spline radial factor by l)."""

    def __init__(self, bases, coefs, NFP):
        self.cbasis = bases[0]
        super().__init__(bases, coefs, NFP)

    def radial_table(self, rho, dr=0):
        return self.cbasis.radial(self.lm[:, 0], self.lm[:, 1], rho, dr)

    def poloidal(self, rho, theta, dr=0, dt=0):
        m = self.lm[:, 1]
        return self.cbasis.radial(self.lm[:, 0], m, rho, dr) * fourier(theta[:, None], m, dt=dt)


def separable_for(bases, coefs, NFP):
    """The Separable class a basis needs (Zernike, spline or composite)."""
    b = bases[0]
    if hasattr(b, "make_separable"):
        return b.make_separable(bases, coefs, NFP)
    if getattr(b, "spectral_indexing", None) == "spline":
        return SplineSeparable(bases, coefs, NFP)
    return Separable(bases, coefs, NFP)


def composite_save_fields(b):
    """npz fields of a CompositeBasis (both parts)."""
    s = b.sb
    return dict(kind="fz+spline", fz_L=b.fz.L, fz_M=b.fz.M, fz_N=b.fz.N, fz_si=b.fz.spectral_indexing,
                bp_rho=s.bp_rho, sp_M=s.M, sp_N=s.N, NFP=b.NFP, sym=str(b.sym) if b.sym else "",
                centres=np.asarray(s.centres, float), m_tol=s.m_tol, m_max=-1 if s.m_max is None else s.m_max,
                m_fz=s.m_fz or 0, reg=s.reg, q_cap=s.q_cap, inside=np.asarray(s.inside or [], float).reshape(-1, 2))


def composite_from_fields(f):
    from desc.basis import FourierZernikeBasis
    sym = str(f["sym"]) if str(f["sym"]) else False
    fz = FourierZernikeBasis(int(f["fz_L"]), int(f["fz_M"]), int(f["fz_N"]), NFP=int(f["NFP"]), sym=sym,
                             spectral_indexing=str(f["fz_si"]))
    m_max = int(f["m_max"])
    sb = SplineBasis(f["bp_rho"], int(f["sp_M"]), int(f["sp_N"]), int(f["NFP"]), bool(sym), list(np.asarray(f["centres"])),
                     float(f["m_tol"]), None if m_max < 0 else m_max, int(f["m_fz"]) or None, str(f["reg"]),
                     int(f["q_cap"]), [tuple(x) for x in np.asarray(f["inside"])])
    return CompositeBasis(fz, sb)
