"""Pressure as radial B-splines x Fourier modes, with knots clustered at rational surfaces.

    p(rho, theta, zeta) = w(rho) Q,   w = (1 - rho^2)^(k+1)   (edge order k, as ``Pressure``)
    Q = sum_{(m, n), j} c_jmn B_j(rho^2) F_mn(theta, zeta),
    F_mn = cos(m theta - n NFP zeta)  (and sin(...) without stellarator symmetry)

B_j: clamped cubic B-splines in s = rho^2 with breakpoints s_i = rho_i^2; the rho_i have spacing
``coarse`` and ``dense`` within ``half`` of each centre (the rational surfaces of DESC's iota
for the chosen helicities, e.g. q = 2), so the basis can resolve a sharp p near an island
(a flat zone inside, steep gradients on the separatrix) where the Fourier-Zernike basis
spreads its radial resolution evenly.

Regularity at the axis: m = 0 uses every spline (functions of rho^2, smooth). For m >= 1 the
first three splines (value, slope or curvature nonzero at s = 0) are dropped: the others
vanish identically on [0, s_1] (s_1 the first interior breakpoint), so every term is smooth
at the axis. p = 0 on the boundary holds through w for any coefficients.

Regularity ``reg="rhom"`` (default ``"drop3"`` above): every spline is kept for every m and multiplied by
rho^q_m, q_m = |m| for |m| <= q_cap, else the largest q <= q_cap with the parity of m; each radial function
rho^q_m B_j(rho^2) is scaled to unit maximum. With q_m = |m| this is Zernike's regularity (rho^|m| e^{i m theta}
= (x + i y)^m times a smooth function of x^2 + y^2) for any coefficients, and the m = 1 terms keep the ~rho
behaviour of a shifted axis; capping q at q_cap leaves the series C^(q_cap - 2) at the axis (enough for J and
grad p) while avoiding rho^32-sized columns. ``radial_factor`` gives rho^q_m and its rho-derivatives.

Explicit knot bands (``bands`` = [(lo, hi, h), ...] in rho): breakpoints with spacing h on [lo, hi], replacing
the outer points there; used instead of the centres / half / dense bands when given.

Unknowns per (m, n): n_s splines (m = 0) or n_s - 3 (m >= 1); with symmetry and M, N the
Fourier resolution: n = n_s (N + 1) + (n_s - 3) M (2N + 1), i.e. linear in the number of
knots, against (M+1)(M+2)/2 (2N+1)/2 for Fourier-Zernike at L = M (``count``).

Interface of ``Pressure`` used by ``objective.Fields`` and ``meanforce_fixed``: ``n``, ``c``,
``with_c``, ``labels``, ``tensor(grid, geo, RZ)`` -> callable c -> (p, grad p in (R, phi, Z)),
``point_evaluator``, ``evaluate``, ``fit`` (via ``fit_cache``), ``save`` / ``load``.
"""

import functools

import jax
import jax.numpy as jnp
import numpy as np
from scipy.interpolate import BSpline

from flux_interp import _Pytree


# ------------------------------------------------------------------ knots
def fz_points(L, c=1.0):
    """Breakpoints on [0, 1] with the local spacing of Fourier-Zernike at radial resolution L, times c:
    h(rho) = c pi sqrt(1 - rho^2) / (L + 2) (the spacing of the zeros of the top Zernike radial function), i.e.
    evenly spaced in arcsin rho: rho_i = sin(i c pi / (L + 2)), the last one moved to 1."""
    d = c * np.pi / (L + 2)
    n = max(int(round(0.5 * np.pi / d)), 1)
    return np.sin(np.linspace(0.0, 0.5 * np.pi, n + 1))


def breakpoints(lo, hi, coarse, dense, centres, half, fz_knots=None, bands=None):
    """Breakpoints on [lo, hi]: spacing ``coarse`` (or, with ``fz_knots`` = (L, c), the spacing of Fourier-Zernike
    at L times c, ``fz_points``; with (L, c, rho_from), ``coarse`` up to rho_from and the FZ spacing beyond it: an
    edge refinement), and ``dense`` within ``half`` of each centre (points closer than dense / 2 merged; with
    fz_knots the outer points inside a dense band are dropped instead)."""
    if bands:  # explicit (lo, hi, spacing) bands
        return _banded(lo, hi, coarse, fz_knots, [(float(a), float(b), float(h)) for a, b, h in bands])
    bands = [(max(lo, c - half), min(hi, c + half)) for c in centres]
    bands = [(a, b) for a, b in bands if b > a]
    if fz_knots is None:
        outer = np.linspace(lo, hi, max(int(round((hi - lo) / coarse)), 1) + 1)
    else:
        L_, c_ = fz_knots[0], fz_knots[1]
        r0 = float(fz_knots[2]) if len(fz_knots) > 2 else lo
        outer = fz_points(L_, c_)
        outer = outer[(outer > r0) & (outer <= hi)]
        if r0 > lo:  # coarse below rho_from
            outer = np.concatenate([np.linspace(lo, r0, max(int(round((r0 - lo) / coarse)), 1) + 1), outer])
        else:
            outer = outer[outer >= lo]
        outer = np.unique(np.concatenate([[lo, hi], outer]))
        inside = np.zeros(len(outer), bool)
        for a, b in bands:  # the band's own points replace these (no near-duplicates at the band edges)
            inside |= (outer > a - dense / 2) & (outer < b + dense / 2)
        outer = outer[~inside | (outer == lo) | (outer == hi)]
    pts = [outer]
    for a, b in bands:
        pts.append(np.linspace(a, b, max(int(round((b - a) / dense)), 1) + 1))
    x = np.unique(np.concatenate(pts))
    if fz_knots is not None:
        return x
    keep = [x[0]]
    for v in x[1:]:
        if v - keep[-1] >= dense / 2:
            keep.append(v)
    keep[-1] = hi
    return np.asarray(keep)


def _banded(lo, hi, coarse, fz_knots, bands):
    """Outer breakpoints (``coarse`` spacing, or the FZ spacing of ``fz_knots``) with every band (a, b, h)
    replaced by uniform points of spacing h: outer points within h / 2 of a band are dropped."""
    if fz_knots is None:
        outer = np.linspace(lo, hi, max(int(round((hi - lo) / coarse)), 1) + 1)
    else:
        L_, c_ = fz_knots[0], fz_knots[1]
        r0 = float(fz_knots[2]) if len(fz_knots) > 2 else lo
        outer = fz_points(L_, c_)
        outer = outer[(outer > r0) & (outer <= hi)]
        if r0 > lo:
            outer = np.concatenate([np.linspace(lo, r0, max(int(round((r0 - lo) / coarse)), 1) + 1), outer])
        outer = np.unique(np.concatenate([[lo, hi], outer[outer >= lo]]))
    keep = np.ones(len(outer), bool)
    pts = []
    for a, b, h in bands:
        a, b = max(lo, a), min(hi, b)
        if b <= a:
            continue
        keep &= ~((outer > a - h / 2) & (outer < b + h / 2))
        pts.append(np.linspace(a, b, max(int(round((b - a) / h)), 1) + 1))
    keep[0] = keep[-1] = True
    x = np.unique(np.concatenate([outer[keep]] + pts))
    # an outer point left closer than half the smaller neighbouring spacing to a band end is merged
    out = [x[0]]
    for v in x[1:]:
        if v - out[-1] > 1e-9:
            out.append(v)
    return np.asarray(out)


def q_of_m(m, q_cap):
    """Axis power of poloidal number m under ``reg="rhom"``: |m| up to q_cap, then the largest
    q <= q_cap with the parity of |m|."""
    am = abs(int(m))
    if am <= q_cap:
        return am
    return int(q_cap - ((q_cap - am) % 2))


def radial_scale(t, qs, n=4001):
    """1 / max_rho rho^q B_j(rho^2) for every spline j and every q in ``qs``: dict q -> (n_s,)."""
    rho = np.linspace(0.0, 1.0, n)
    B = design(np.asarray(t, float), rho**2)
    out = {}
    for q in sorted(set(int(v) for v in qs)):
        mx = np.max(B * (rho**q)[:, None], axis=0)
        out[q] = 1.0 / np.where(mx > 0, mx, 1.0)
    return out


def radial_factor(rho, q, dr=0):
    """rho^q (dr = 0) or its rho-derivatives (dr = 1, 2), elementwise (q integer array >= 0), JAX-safe at 0."""
    q = jnp.asarray(q)
    if dr == 0:
        return rho**q
    if dr == 1:
        return jnp.where(q >= 1, q * rho ** jnp.maximum(q - 1, 0), 0.0)
    if dr == 2:
        return jnp.where(q >= 2, q * (q - 1) * rho ** jnp.maximum(q - 2, 0), 0.0)
    raise ValueError(dr)


def knots_s(bp_rho):
    """Clamped cubic knot vector in s = rho^2 from breakpoints in rho."""
    bs = np.asarray(bp_rho) ** 2
    return np.concatenate([[bs[0]] * 3, bs, [bs[-1]] * 3])


def design(t, x, nu=0):
    """B-spline basis (or its nu-th derivative) at x: (len(x), len(t) - 4)."""
    n = len(t) - 4
    return np.stack([np.nan_to_num(BSpline(t, np.eye(n)[i], 3, extrapolate=False)(x, nu)) for i in range(n)],
                    axis=1).astype(float)


def rational_centres(eq, helicities, n_rho=201):
    """rho where |iota| = NFP n / m for each "m/n" in ``helicities`` (q = m / n). iota from full
    flux surfaces (theta, zeta resolution of the equilibrium's grid): for a current-prescribed
    equilibrium it is a surface average, so a single-theta grid would be wrong."""
    from desc.grid import LinearGrid
    rho = np.linspace(0.01, 1, n_rho)
    g = LinearGrid(rho=rho, M=eq.M_grid, N=eq.N_grid, NFP=eq.NFP, sym=eq.sym)
    iota = np.abs(np.asarray(g.compress(eq.compute("iota", grid=g)["iota"])))
    out = []
    for h in helicities:
        m, n = (int(v) for v in h.split("/"))
        v = eq.NFP * n / m
        for i in np.flatnonzero(np.diff(np.sign(iota - v)) != 0):
            out.append(float(rho[i] + (v - iota[i]) / (iota[i + 1] - iota[i]) * (rho[i + 1] - rho[i])))
    return sorted(out)


def composite_gauss(bp_rho, per_interval=4, min_nodes=0):
    """Radial collocation nodes and weights on [0, 1]: ``per_interval`` Gauss-Legendre points in
    every breakpoint interval (so every spline has points under it), the intervals split further
    if fewer than ``min_nodes`` nodes would result (e.g. the Fourier-Zernike current's L // 2 + 1)."""
    bp = np.asarray(bp_rho, float)
    while (len(bp) - 1) * per_interval < min_nodes:
        bp = np.sort(np.concatenate([bp, 0.5 * (bp[1:] + bp[:-1])]))
    x, w = np.polynomial.legendre.leggauss(per_interval)
    a, b = bp[:-1, None], bp[1:, None]
    rho = (a + (b - a) * (x[None] + 1) / 2).ravel()
    wr = ((b - a) / 2 * w[None]).ravel()
    return rho, wr


def fz_first_zero(L, m):
    """rho of the first zero of the top Zernike radial function of |m| at resolution L (R_l^m = rho^|m|
    P_k^(|m|, 0)(1 - 2 rho^2), k = (L - |m|) / 2): FZ's m-th harmonic oscillates only beyond it. None if k < 1."""
    from scipy.special import roots_jacobi
    k = (L - abs(m)) // 2
    if k < 1:
        return None
    y = roots_jacobi(k, abs(m), 0)[0]
    return float(np.sqrt((1 - y.max()) / 2))


def spline_keep(t, m, m_tol=0.0, m_max=None, m_fz=None, reg="drop3", axis_m=6, inside=None):
    """Spline indices j used by poloidal number m: all for m = 0; for m != 0 the first three are dropped (axis
    regularity), and with truncation also every j whose support ends where rho^|m| < m_tol (rho_end = sqrt of the
    support's last knot in s), every m with |m| > m_max (empty), and with ``m_fz`` = L every j whose support ends
    before the first zero of Fourier-Zernike's top radial function of |m| at L (``fz_first_zero``; FZ's own
    radial reach of that harmonic; |m| > L - 2: rho^|m| < 1e-3 instead). With reg="rhom" the four splines that
    touch the axis are kept for |m| <= axis_m whatever the truncation: they carry rho^|m| (a + b s + c s^2 + d s^3)
    near the axis, the low-order content Zernike always has (e.g. the m = 1 ~ rho of a shifted axis)."""
    n_s = len(t) - 4
    if inside:  # band-localized splines only: support [t_j, t_{j+4}] inside one of the (lo, hi) bands (rho)
        def _in(j):
            rs, re = np.sqrt(t[j]), np.sqrt(t[j + 4])
            return any(lo - 1e-9 <= rs and re <= hi + 1e-9 for lo, hi in inside)
        cand = [j for j in spline_keep(t, m, m_tol, m_max, m_fz, reg, axis_m) if _in(j)]
        return cand
    if m == 0:
        return list(range(n_s))
    if m_max is not None and abs(m) > m_max:
        return []
    r1 = None
    if m_fz:
        r1 = fz_first_zero(m_fz, m)
        if r1 is None:
            r1 = 1e-3 ** (1.0 / abs(m))
    keep = []
    for j in range(3 if reg == "drop3" else 0, n_s):
        re = np.sqrt(t[j + 4])
        if reg == "rhom" and j <= 3 and abs(m) <= axis_m:
            keep.append(j)
            continue
        if m_tol > 0 and re ** abs(m) < m_tol:
            continue
        if r1 is not None and re <= r1:
            continue
        keep.append(j)
    return keep


def trunc_kw(m_tol=0.0, m_max=None, m_fz=None, fz_knots=None, bands=None, reg="drop3", q_cap=6):
    """fit keyword arguments of a truncation, empty when there is none (so untruncated fit caches keep their keys)."""
    kw = {}
    if m_tol:
        kw["m_tol"] = float(m_tol)
    if m_max is not None and m_max >= 0:
        kw["m_max"] = int(m_max)
    if m_fz:
        kw["m_fz"] = int(m_fz)
    if fz_knots:
        kw["fz_knots"] = (int(fz_knots[0]),) + tuple(float(v) for v in fz_knots[1:])
    if bands:
        kw["bands"] = tuple(tuple(float(v) for v in b) for b in bands)
    if reg != "drop3":
        kw["reg"], kw["q_cap"] = str(reg), int(q_cap)
    return kw


def modes(M, N, sym):
    """(m, n, kind) with kind 0 = cos(m theta - n NFP zeta), 1 = sin(...)."""
    out = [(0, n, 0) for n in range(0, N + 1)]
    out += [(m, n, 0) for m in range(1, M + 1) for n in range(-N, N + 1)]
    if not sym:
        out += [(0, n, 1) for n in range(1, N + 1)]
        out += [(m, n, 1) for m in range(1, M + 1) for n in range(-N, N + 1)]
    return np.asarray(out, int)


def count(n_s, M, N, sym=True):
    """Unknowns of the spline pressure (module docstring)."""
    md = modes(M, N, sym)
    return int(sum(n_s if m == 0 else n_s - 3 for m, _, _ in md))


def count_fz(L, M, N, sym=True):
    """Approximate unknowns of a Fourier-Zernike scalar (ANSI, l <= L, |m| <= M), for comparison:
    the (l, m) pairs times 2N + 1, halved with stellarator symmetry."""
    pairs = sum(1 for l in range(L + 1) for m in range(-min(l, M), min(l, M) + 1) if (l - abs(m)) % 2 == 0)
    return pairs * (2 * N + 1) // (2 if sym else 1)


# --------------------------------------------------------------- evaluators
class TensorSplinePressure(_Pytree):
    """p and grad p on a tensor grid (rho, theta, zeta)."""

    _dynamic = ("geo", "Bs", "Bds", "Phi", "Phit", "Phiz", "wk", "wkr", "jj", "mm", "G", "Gr", "sc")

    def __init__(self, sp, grid, geo):
        self.shape = tuple(grid.shape)
        rho = np.asarray(grid.rho)
        s = rho**2
        self.Bs = jnp.asarray(design(sp.t, s))  # (n_rho, n_s)
        self.Bds = jnp.asarray(design(sp.t, s, 1) * 2 * rho[:, None])  # d/drho
        th, ze = np.asarray(grid.theta), np.asarray(grid.zeta)
        T, Z = np.meshgrid(th, ze, indexing="ij")
        md = sp.modes
        arg = md[:, 0][:, None, None] * T[None] - (md[:, 1] * sp.NFP)[:, None, None] * Z[None]
        c, sn = np.cos(arg), np.sin(arg)
        cos_k = (md[:, 2] == 0)[:, None, None]
        F = np.where(cos_k, c, sn)
        dF = np.where(cos_k, -sn, c)  # d/d(arg)
        self.Phi = jnp.asarray(F.reshape(len(md), -1))
        self.Phit = jnp.asarray((md[:, 0][:, None, None] * dF).reshape(len(md), -1))
        self.Phiz = jnp.asarray((-(md[:, 1] * sp.NFP)[:, None, None] * dF).reshape(len(md), -1))
        k = sp.edge_order
        self.wk = jnp.asarray((1 - s) ** (k + 1))[:, None]
        self.wkr = jnp.asarray(-2 * (k + 1) * rho * (1 - s) ** k)[:, None]
        self.jj, self.mm = jnp.asarray(sp.j_idx), jnp.asarray(sp.mode_idx)
        self.n_s, self.n_modes = sp.n_s, len(md)
        r = jnp.asarray(rho)[:, None]
        self.G = radial_factor(r, sp.q_mode[None, :])  # (n_rho, n_modes): rho^q_m (ones for drop3)
        self.Gr = radial_factor(r, sp.q_mode[None, :], 1)
        self.sc = jnp.asarray(sp.scale)
        self.geo = geo

    def __call__(self, c):
        C = jnp.zeros((self.n_s, self.n_modes), dtype=c.dtype).at[self.jj, self.mm].set(c * self.sc,
                                                                                       unique_indices=True)
        Q0 = self.Bs @ C          # (n_rho, n_modes)
        Qr = Q0 * self.G
        Qdr = (self.Bds @ C) * self.G + Q0 * self.Gr
        Q = Qr @ self.Phi         # (n_rho, n_t n_z)
        Q_r = Qdr @ self.Phi
        Q_t = Qr @ self.Phit
        Q_z = Qr @ self.Phiz
        p, pr, pt, pz = self.wk * Q, self.wkr * Q + self.wk * Q_r, self.wk * Q_t, self.wk * Q_z
        sh = self.shape
        p, pr, pt, pz = (x.reshape(sh) for x in (p, pr, pt, pz))
        return p, self.geo.gradient(pr, pt, pz)


def _deboor_basis(t, x, k=3):
    """All B-spline basis values at x (n,) in JAX: (n, len(t) - k - 1), by the Cox-de Boor
    recursion (``t`` concrete: a tuple or NumPy array)."""
    t = np.asarray(t, float)
    n_b = len(t) - k - 1
    B = [((x >= t[i]) & (x < t[i + 1])).astype(x.dtype) for i in range(len(t) - 1)]
    # right end: the last nonempty interval is closed
    last = int(np.flatnonzero(np.diff(t) > 0)[-1])
    B[last] = jnp.where(x == t[last + 1], 1.0, B[last])
    for d in range(1, k + 1):
        nb = []
        for i in range(len(t) - 1 - d):
            a = jnp.where(t[i + d] > t[i], (x - t[i]) / jnp.where(t[i + d] > t[i], t[i + d] - t[i], 1.0), 0.0)
            b = jnp.where(t[i + d + 1] > t[i + 1],
                          (t[i + d + 1] - x) / jnp.where(t[i + d + 1] > t[i + 1], t[i + d + 1] - t[i + 1], 1.0), 0.0)
            nb.append(a * B[i] + b * B[i + 1])
        B = nb
    return jnp.stack(B[:n_b], axis=1)


@functools.partial(jax.jit, static_argnums=(0, 1, 2, 3))
def _p_and_grad(t, NFP, k, RZ, modes_, jj, mm, c, qm, sc, rho, theta, zeta):
    """(p, dp/dR, dp/(R dphi), dp/dZ) at the points, (n, 4); t (tuple), NFP, k, RZ static."""
    n_s = len(t) - 4
    s = rho**2
    Bs = _deboor_basis(t, s)
    # derivative: B'_j = 3 (B2_{j} / (t_{j+3} - t_j) - B2_{j+1} / (t_{j+4} - t_{j+1})), B2 on t
    B2 = _deboor_basis(t, s, k=2)  # (n, n_s + 1)
    tt = np.asarray(t, float)
    den1 = tt[3: 3 + n_s] - tt[:n_s]
    den2 = tt[4: 4 + n_s] - tt[1: 1 + n_s]
    Bd = 3 * (jnp.where(den1 > 0, B2[:, :n_s] / jnp.where(den1 > 0, den1, 1), 0)
              - jnp.where(den2 > 0, B2[:, 1: n_s + 1] / jnp.where(den2 > 0, den2, 1), 0))
    Bdr = Bd * 2 * rho[:, None]
    C = jnp.zeros((n_s, modes_.shape[0])).at[jj, mm].set(c * sc, unique_indices=True)
    arg = modes_[None, :, 0] * theta[:, None] - modes_[None, :, 1] * NFP * zeta[:, None]
    cos_k = modes_[None, :, 2] == 0
    F = jnp.where(cos_k, jnp.cos(arg), jnp.sin(arg))
    dF = jnp.where(cos_k, -jnp.sin(arg), jnp.cos(arg))
    G = radial_factor(rho[:, None], qm[None, :])
    Gr = radial_factor(rho[:, None], qm[None, :], 1)
    Q0 = Bs @ C
    Qm, Qmr = Q0 * G, (Bdr @ C) * G + Q0 * Gr
    Q = jnp.sum(Qm * F, 1)
    Qr = jnp.sum(Qmr * F, 1)
    Qt = jnp.sum(Qm * dF * modes_[None, :, 0], 1)
    Qz = jnp.sum(Qm * dF * (-modes_[None, :, 1] * NFP), 1)
    w = (1 - s) ** (k + 1)
    wr = -2 * (k + 1) * rho * (1 - s) ** k
    p, pr, pt, pz = w * Q, wr * Q + w * Qr, w * Qt, w * Qz
    R = RZ.evaluate(rho, theta, zeta)[:, 0]
    Rr, Zr = RZ.evaluate(rho, theta, zeta, dr=1).T
    Rt, Zt = RZ.evaluate(rho, theta, zeta, dt=1).T
    Rz, Zz = RZ.evaluate(rho, theta, zeta, dz=1).T
    sqrt_g = R * (Rt * Zr - Rr * Zt)
    g_R = -pr * R * Zt + pt * R * Zr
    g_phi = pr * (Zt * Rz - Rt * Zz) + pt * (Zz * Rr - Rz * Zr) + pz * (Zr * Rt - Rr * Zt)
    g_Z = pr * R * Rt - pt * R * Rr
    return jnp.stack([p, g_R / sqrt_g, g_phi / sqrt_g, g_Z / sqrt_g], axis=-1)


# ------------------------------------------------------------------ the class
class SplinePressure:
    """p = w(rho) sum c B_j(rho^2) F_mn (module docstring)."""

    def __init__(self, bp_rho, M, N, NFP, sym, c, eq, edge_order=2, m_tol=0.0, m_max=None, m_fz=None,
                 reg="drop3", q_cap=6):
        self.bp_rho = np.asarray(bp_rho, float)
        self.m_tol, self.m_max = float(m_tol), (None if m_max is None or m_max < 0 else int(m_max))
        self.m_fz = int(m_fz) if m_fz else None
        self.t = knots_s(self.bp_rho)
        self.n_s = len(self.t) - 4
        self.M, self.N, self.NFP, self.sym = int(M), int(N), int(NFP), bool(sym)
        self.modes = modes(self.M, self.N, self.sym)
        self.reg, self.q_cap = str(reg), int(q_cap)
        jj, mm = [], []
        for i, (m, _, _) in enumerate(self.modes):
            js = spline_keep(self.t, m, self.m_tol, self.m_max, self.m_fz, self.reg, self.q_cap)
            jj += list(js)
            mm += [i] * len(js)
        self.j_idx, self.mode_idx = np.asarray(jj, int), np.asarray(mm, int)
        if self.reg == "rhom":
            self.q_mode = np.array([q_of_m(m, self.q_cap) for m in self.modes[:, 0]], int)
            tab = radial_scale(self.t, self.q_mode)
            self.scale = np.array([tab[self.q_mode[i]][j] for j, i in zip(self.j_idx, self.mode_idx)])
        elif self.reg == "drop3":
            self.q_mode = np.zeros(len(self.modes), int)
            self.scale = np.ones(len(self.j_idx))
        else:
            raise ValueError(f"reg {reg!r}: 'drop3' or 'rhom'")
        self.c = np.zeros(len(jj)) if c is None else np.asarray(c, float)
        assert self.c.shape == (len(jj),), (self.c.shape, len(jj))
        self.eq, self.edge_order = eq, int(edge_order)

    @classmethod
    def fit(cls, eq, L=None, M=None, N=None, factors=(3, 3, 2.5), edge_order=2, sym=None, coarse=0.05,
            dense=0.01, half=0.08, helicities=("2/1",), centres=None, m_tol=0.0, m_max=None, m_fz=None,
            fz_knots=None, bands=None, reg="drop3", q_cap=6):
        """DESC's p(rho) in the m = n = 0 splines (least squares); the other modes start at
        zero. M, N default to factors x the equilibrium's (as ``Pressure.fit``); L is unused
        (the radial resolution is the knots')."""
        from desc.grid import LinearGrid
        M = int(round(factors[1] * eq.M)) if M is None else M
        N = int(round(factors[2] * eq.N)) if N is None else N
        sym = bool(eq.sym) if sym is None else sym
        if centres is None:
            centres = rational_centres(eq, helicities) if helicities else []
        bp = breakpoints(0.0, 1.0, coarse, dense, centres, half, fz_knots, bands)
        sp = cls(bp, M, N, eq.NFP, sym, None, eq, edge_order, m_tol, m_max, m_fz, reg, q_cap)
        rho = np.linspace(0, 1, 8 * sp.n_s + 64)
        p = np.asarray(eq.compute("p", grid=LinearGrid(rho=rho, NFP=eq.NFP))["p"])
        Bm = design(sp.t, rho**2) * ((1 - rho**2) ** (edge_order + 1))[:, None]
        i0 = np.flatnonzero((sp.modes[:, 0] == 0) & (sp.modes[:, 1] == 0) & (sp.modes[:, 2] == 0))[0]
        sel = np.flatnonzero(sp.mode_idx == i0)
        c = np.zeros(len(sp.c))
        c[sel] = np.linalg.lstsq(Bm[:, sp.j_idx[sel]] * sp.scale[sel][None, :], p, rcond=None)[0]
        sp.c = c
        sp.centres = list(centres)
        return sp

    # the interface of Fields
    @property
    def n(self):
        return len(self.c)

    def with_c(self, c, eq=None):
        out = SplinePressure(self.bp_rho, self.M, self.N, self.NFP, self.sym, c, self.eq if eq is None else eq,
                             self.edge_order, self.m_tol, self.m_max, self.m_fz, self.reg, self.q_cap)
        out.centres = getattr(self, "centres", [])
        return out

    def labels(self):
        """(kind, j, m, n) per unknown: kind 1 (pressure), j the spline index."""
        md = self.modes[self.mode_idx]
        return np.column_stack([np.ones(self.n, int), self.j_idx, md[:, 0], md[:, 1]])

    def tensor(self, grid, geo, RZ, factored=False):
        from flux_interp import TensorGeometry
        return TensorSplinePressure(self, grid, geo or TensorGeometry(RZ, grid))

    def point_evaluator(self, eq=None):
        from separable import Separable
        eq = self.eq if eq is None else eq
        RZ = Separable([eq.R_basis, eq.Z_basis], [eq.R_lmn, eq.Z_lmn], eq.NFP)
        return functools.partial(_p_and_grad, tuple(float(v) for v in self.t), self.NFP, self.edge_order, RZ,
                                 jnp.asarray(self.modes), jnp.asarray(self.j_idx), jnp.asarray(self.mode_idx),
                                 jnp.asarray(self.c), jnp.asarray(self.q_mode), jnp.asarray(self.scale))

    def evaluate(self, grid, chunk=50000):
        f = self.point_evaluator()
        nodes = np.asarray(grid.nodes)
        return np.concatenate([np.asarray(f(*(jnp.asarray(nodes[i: i + chunk, j]) for j in range(3))))
                               for i in range(0, len(nodes), chunk)])

    def summary(self):
        return {"kind": "spline", "n": self.n, "n_s": self.n_s, "M": self.M, "N": self.N, "sym": self.sym,
                "m_tol": self.m_tol, "m_max": self.m_max, "m_fz": self.m_fz, "reg": self.reg, "q_cap": self.q_cap,
                "breakpoints": len(self.bp_rho), "centres": getattr(self, "centres", []),
                "min spacing": float(np.diff(self.bp_rho).min()), "max spacing": float(np.diff(self.bp_rho).max())}

    def save(self, path):
        np.savez_compressed(path, c=self.c, bp_rho=self.bp_rho, M=self.M, N=self.N, NFP=self.NFP,
                            sym=self.sym, edge_order=self.edge_order, m_tol=self.m_tol,
                            m_max=-1 if self.m_max is None else self.m_max, m_fz=self.m_fz or 0,
                            reg=self.reg, q_cap=self.q_cap, centres=np.asarray(getattr(self, "centres", []), float))

    @classmethod
    def load(cls, path, eq):
        with np.load(path) as f:
            sp = cls(f["bp_rho"], int(f["M"]), int(f["N"]), int(f["NFP"]), bool(f["sym"]), f["c"], eq,
                     int(f["edge_order"]), float(f["m_tol"]) if "m_tol" in f else 0.0,
                     int(f["m_max"]) if "m_max" in f else None, int(f["m_fz"]) if "m_fz" in f else None,
                     str(f["reg"]) if "reg" in f else "drop3", int(f["q_cap"]) if "q_cap" in f else 6)
            sp.centres = list(np.asarray(f["centres"]))
        return sp
