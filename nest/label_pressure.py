"""Pressure as a profile of a label field, p = P(chi), so that its surfaces can form islands.

A Fourier-Zernike p (``pressure.py``) cannot make the closed contours of an island
chain of width ~0.01-0.1 in rho at the resolution it has. Here

    chi(rho, theta, zeta) = rho + sum_k sum_mn d_kmn Phi_k(rho) cos(m theta* - n NFP zeta),
    p = P(chi) = (1 - chi^2)^3 Q(chi^2),

with theta* = theta + lambda (DESC's straight-field-line angle of the anchor, fixed),
Phi_k cubic B-splines in rho times 1 - rho^2 (chi = 1 on the boundary, so p = 0 there
and p = O((1 - rho^2)^3) like the smooth-edge Fourier-Zernike p), and Q a cubic
spline in s = chi^2 (p even in chi: regular at the axis). Both have knots that are
dense in bands around the rational surfaces of DESC's iota (only their rough place
is needed). With d != 0 the contours of chi, hence of p, can close around O-points;
P can be flat across an island or keep loops inside it. The unknowns are
c = (q, d): the spline coefficients of Q, then d (``labels``).

The (m, n) = (0, 0) part of chi is left out: it is a change of P. Where P is flat chi
is undetermined: ``LabelDamping`` damps d (high (m, n) more).

Stellarator symmetry only (p even under (theta, zeta) -> (-theta, -zeta); theta* is
odd).
"""

import functools

import numpy as np
import jax
import jax.numpy as jnp
from scipy.interpolate import BSpline

from flux_interp import TensorGrid, _Pytree, separable_factors
from separable import Separable


# ---------------------------------------------------------------- splines
def breakpoints(lo, hi, coarse, dense, centres, half):
    """Breakpoints on [lo, hi]: spacing ``coarse``, and ``dense`` within ``half`` of
    each centre (points closer than dense / 2 merged)."""
    pts = [np.linspace(lo, hi, max(int(round((hi - lo) / coarse)), 1) + 1)]
    for c in centres:
        a, b = max(lo, c - half), min(hi, c + half)
        if b > a:
            pts.append(np.linspace(a, b, max(int(round((b - a) / dense)), 1) + 1))
    x = np.unique(np.concatenate(pts))
    keep = [x[0]]
    for v in x[1:]:
        if v - keep[-1] >= dense / 2:
            keep.append(v)
    keep[-1] = hi
    return np.asarray(keep)


def clamped(bp):
    """Cubic knot vector with the end breakpoints repeated 4 times."""
    return np.concatenate([[bp[0]] * 3, bp, [bp[-1]] * 3])


def design(t, x, nu=0):
    """B-spline basis (or its nu-th derivative) at x: (len(x), len(t) - 4)."""
    n = len(t) - 4
    return np.stack([BSpline(t, np.eye(n)[i], 3, extrapolate=False)(x, nu) for i in range(n)],
                    axis=1).astype(float)


def deboor(t, c, x):
    """The cubic spline (knots t, coefficients c) at x, in JAX (local: 4 terms)."""
    k = 3
    i = jnp.clip(jnp.searchsorted(t, x, side="right") - 1, k, len(t) - k - 2)
    d = [c[i - k + j] for j in range(k + 1)]
    for r in range(1, k + 1):
        for j in range(k, r - 1, -1):
            tl, tr = t[i + j - k], t[i + j + 1 - r]
            alpha = (x - tl) / (tr - tl)
            d[j] = (1 - alpha) * d[j - 1] + alpha * d[j]
    return d[k]


def rational_surfaces(eq, m_max=14, n_rho=1001):
    """rho, m, n of the rational surfaces iota = NFP n / m (m <= m_max) of DESC's iota."""
    from desc.grid import LinearGrid
    rho = np.linspace(0, 1, n_rho)
    iota = np.asarray(eq.compute("iota", grid=LinearGrid(rho=rho, NFP=eq.NFP))["iota"])
    iota = np.abs(iota)  # (DESC's sign follows the handedness)
    out = []
    for m in range(1, m_max + 1):
        for n in range(1, m_max + 1):
            if np.gcd(n, m) != 1:
                continue
            v = eq.NFP * n / m
            s = np.flatnonzero(np.diff(np.sign(iota - v)) != 0)
            for i in s:
                r = rho[i] + (v - iota[i]) / (iota[i + 1] - iota[i]) * (rho[i + 1] - rho[i])
                out.append((float(r), m, n))
    return sorted(out)


# ---------------------------------------------------------------- the pressure
class LabelPressure:
    """p = P(chi) (module docstring).

    Parameters
    ----------
    t_q : knots of Q(s), s in [0, 1]
    t_r : knots of the radial splines of chi, on [rho0, 1]
    modes : (n_modes, 2) int, the (m, n) of chi (n in units of NFP)
    c : (n_q + n_r * n_modes,) coefficients (q, then d in C order (k, mode))
    eq : the equilibrium (geometry and lambda)
    """

    edge_order = None

    def __init__(self, t_q, t_r, modes, c, eq, filt=(8.0, 2.0), chi_scale=1.0):
        if not eq.sym:
            raise NotImplementedError("LabelPressure: stellarator-symmetric equilibria only")
        self.t_q, self.t_r = np.asarray(t_q, float), np.asarray(t_r, float)
        self.modes = np.asarray(modes, int).reshape(-1, 2)
        self.c = np.asarray(c, float)
        self.eq = eq
        self.filt = tuple(filt)
        # chi - rho = chi_scale sum d Phi cos: the unknowns d scaled so that their Jacobian
        # columns are the size of the current's (``optimize_fb.py`` sets it on the first anchor)
        self.chi_scale = float(chi_scale)
        self.n_q = len(self.t_q) - 4
        self.n_r = len(self.t_r) - 4 - 2  # the first two splines dropped: chi C1 at rho0
        self.n_d = self.n_r * len(self.modes)
        assert len(self.c) == self.n, (len(self.c), self.n)

    @property
    def n(self):
        return self.n_q + self.n_d

    @classmethod
    def fit(cls, eq, M=16, N=4, rho0=0.15, coarse=0.04, dense=0.0125, half=0.04, m_max=14,
            filt=(8.0, 2.0), verbose=True):
        """Q fitted to DESC's p(rho) (chi = rho, d = 0); knots dense around the rationals."""
        rats = rational_surfaces(eq, m_max)
        centres = [r for r, _, _ in rats if rho0 < r < 1]
        t_r = clamped(breakpoints(rho0, 1.0, coarse, dense, centres, half))
        bp = breakpoints(0.0, 1.0, coarse, dense, centres, half)
        t_q = clamped(np.unique(np.clip(bp, 0, 1) ** 2))
        modes = [(0, n) for n in range(1, N + 1)] + [(m, n) for m in range(1, M + 1)
                                                      for n in range(-N, N + 1)]
        from desc.grid import LinearGrid
        rho = np.linspace(0, 1, 4001)[:-1]
        p = np.asarray(eq.compute("p", grid=LinearGrid(rho=rho, NFP=eq.NFP))["p"])
        A = (1 - rho**2)[:, None] ** 3 * design(t_q, rho**2)
        q = np.linalg.lstsq(A, p, rcond=None)[0]
        new = cls(t_q, t_r, modes, np.zeros(len(q) + (len(t_r) - 6) * len(modes)), eq, filt)
        new.c[: new.n_q] = q
        if verbose:
            err = np.abs(A @ q - p).max() / np.abs(p).max()
            print(f"  LabelPressure: Q {new.n_q} splines, chi {new.n_r} radial x {len(modes)} "
                  f"modes = {new.n_d} (M {M}, N {N}, rho >= {rho0}); rationals "
                  + ", ".join(f"{n * eq.NFP}/{m} at {r:.3f}" for r, m, n in rats if r > rho0)
                  + f"; fit of DESC p: max error {err:.1e} of p(0)", flush=True)
        return new

    def with_c(self, c, eq=None):
        return LabelPressure(self.t_q, self.t_r, self.modes, c, self.eq if eq is None else eq,
                             self.filt, self.chi_scale)

    def labels(self):
        """(kind, l, m, n) per unknown: kind 1 for q (m = n = 0), 2 for d."""
        q = np.column_stack([np.ones(self.n_q, int), np.arange(self.n_q), np.zeros((self.n_q, 2), int)])
        k, j = np.divmod(np.arange(self.n_d), len(self.modes))
        d = np.column_stack([np.full(self.n_d, 2), k, self.modes[j]])
        return np.concatenate([q, d])

    def damping(self):
        """Weight per unknown of ``LabelDamping``: 0 for q, 1 + (m/m_c)^4 + (n/n_c)^4 for d."""
        mc, nc = self.filt
        w = 1 + (self.modes[:, 0] / mc) ** 4 + (self.modes[:, 1] / nc) ** 4
        return np.concatenate([np.zeros(self.n_q), np.tile(w, self.n_r)])

    def scale(self):
        """max of (1 - rho^2) B_k over rho for the kept splines (Phi_k = (1 - rho^2) B_k / scale_k)."""
        t = self.t_r
        fine = np.linspace(t[0], 1, 2001)
        scale = ((1 - fine**2)[:, None] * np.nan_to_num(design(t, fine))).max(0)[2:]
        return np.where(scale > 0, scale, 1.0)

    def radial(self, rho):
        """Phi_k and dPhi_k/drho at rho: (len(rho), n_r) each; Phi = (1 - rho^2) B_k / max."""
        t = self.t_r
        scale = self.scale()
        r = np.asarray(rho, float)
        B, Br = (np.nan_to_num(design(t, r, nu))[:, 2:] for nu in (0, 1))
        inside = (r >= t[0])[:, None]
        w, wr = (1 - r**2)[:, None], (-2 * r)[:, None]
        return inside * w * B / scale, inside * (wr * B + w * Br) / scale

    def tensor(self, grid, geo, RZ=None, factored=False):
        return TensorLabelPressure(self, grid, geo)

    def save(self, path):
        np.savez_compressed(path, kind="label", t_q=self.t_q, t_r=self.t_r, modes=self.modes,
                            c=self.c, filt=np.asarray(self.filt), chi_scale=self.chi_scale)

    @classmethod
    def load(cls, path, eq, c=None):
        with np.load(path) as f:
            return cls(f["t_q"], f["t_r"], f["modes"], f["c"] if c is None else c, eq,
                       tuple(f["filt"]), float(f["chi_scale"]) if "chi_scale" in f else 1.0)

    # ------------------------------------------------------------ points
    def point_evaluator(self, eq=None):
        """Jitted f(rho, theta, zeta) -> (p, chi), arrays of points (plots, checks)."""
        eq = self.eq if eq is None else eq
        LAM = Separable([eq.L_basis], [eq.L_lmn], eq.NFP)
        tq, q = jnp.asarray(self.t_q), jnp.asarray(self.c[: self.n_q])
        D = self.chi_scale * self.c[self.n_q:].reshape(self.n_r, -1)
        m, n = jnp.asarray(self.modes[:, 0]), jnp.asarray(self.modes[:, 1])
        NFP = eq.NFP

        def f(rho, theta, zeta, Phi):
            ts = theta + LAM.evaluate(rho, theta, zeta)[:, 0]
            amp = Phi @ jnp.asarray(D)
            chi = rho + jnp.sum(amp * jnp.cos(m * ts[:, None] - n * NFP * zeta[:, None]), axis=1)
            s = jnp.clip(chi**2, 0, 1)
            return (1 - s) ** 3 * deboor(tq, q, s), chi

        f = jax.jit(f)
        return lambda rho, theta, zeta: f(jnp.asarray(rho), jnp.asarray(theta), jnp.asarray(zeta),
                                          jnp.asarray(self.radial(np.asarray(rho))[0]))

    def point_partials(self, eq=None):
        """Jitted f(rho, theta, zeta) -> (p, p_rho, p_theta, p_zeta, chi) at arrays of points,
        exact (JAX splines in rho too, jvp): the gradient follows with the metric."""
        eq = self.eq if eq is None else eq
        LAM = Separable([eq.L_basis], [eq.L_lmn], eq.NFP)
        tq, q, tr = jnp.asarray(self.t_q), jnp.asarray(self.c[: self.n_q]), jnp.asarray(self.t_r)
        D = self.chi_scale * self.c[self.n_q:].reshape(self.n_r, -1) / self.scale()[:, None]
        C = jnp.asarray(np.concatenate([np.zeros((2, D.shape[1])), D]))  # (n_r + 2, n_modes)
        m, n = jnp.asarray(self.modes[:, 0]), jnp.asarray(self.modes[:, 1])
        NFP, r0 = eq.NFP, float(self.t_r[0])

        def chi_p(rho, theta, zeta):
            ts = theta + LAM.evaluate(rho, theta, zeta)[:, 0]
            amp = jax.vmap(lambda c: deboor(tr, c, rho), out_axes=1)(C.T)  # (n_points, n_modes)
            amp = jnp.where((rho >= r0)[:, None], (1 - rho**2)[:, None] * amp, 0.0)
            chi = rho + jnp.sum(amp * jnp.cos(m * ts[:, None] - n * NFP * zeta[:, None]), axis=1)
            s = jnp.clip(chi**2, 0, 1)
            return (1 - s) ** 3 * deboor(tq, q, s), chi

        @jax.jit
        def f(rho, theta, zeta):
            one, zero = jnp.ones_like(rho), jnp.zeros_like(rho)
            (p, chi), (pr, _) = jax.jvp(chi_p, (rho, theta, zeta), (one, zero, zero))
            pt = jax.jvp(chi_p, (rho, theta, zeta), (zero, one, zero))[1][0]
            pz = jax.jvp(chi_p, (rho, theta, zeta), (zero, zero, one))[1][0]
            return p, pr, pt, pz, chi

        return f


class TensorLabelPressure(_Pytree):
    """p and grad p of a ``LabelPressure`` on a tensor grid (the ``TensorPressure`` interface):
    nonlinear in c; its derivative is taken by ``jax.jvp`` (``FreeBoundaryFields._tangent``)."""

    _dynamic = ("geo", "Phi", "Phir", "lam", "rho", "theta", "Zf", "tq", "chi_scale")

    def __init__(self, pre, grid, geo):
        eq = pre.eq
        self.grid, self.geo = grid, geo
        self.n_q, self.n_r, self.n_modes = pre.n_q, pre.n_r, len(pre.modes)
        self.M, self.N, self.NFP = int(pre.modes[:, 0].max()), int(np.abs(pre.modes[:, 1]).max()), eq.NFP
        m, n = pre.modes[:, 0], pre.modes[:, 1]
        self.dense_idx = (m * (2 * self.N + 1) + n + self.N).astype(np.int32)  # into (M + 1, 2N + 1)
        Phi, Phir = pre.radial(grid.rho)
        self.Phi, self.Phir = jnp.asarray(Phi), jnp.asarray(Phir)
        LAM = Separable([eq.L_basis], [eq.L_lmn], eq.NFP)
        f = separable_factors(LAM, grid)
        C = LAM.C
        # lambda, lambda_rho, lambda_theta, lambda_zeta: (4, n_r, n_t, n_z)
        self.lam = jnp.stack([f(C)[..., 0], f(C, dr=1)[..., 0], f(C, dt=1)[..., 0],
                              f(C, dz=1)[..., 0]])
        self.rho = jnp.asarray(grid.rho)[:, None, None]
        self.theta = jnp.asarray(grid.theta)[None, :, None]
        nn = np.arange(-self.N, self.N + 1)
        self.Zf = jnp.asarray(np.exp(-1j * np.outer(grid.zeta, nn) * eq.NFP))  # (n_z, 2N + 1)
        self.tq = jnp.asarray(pre.t_q)
        self.chi_scale = jnp.asarray(pre.chi_scale)

    def chi(self, d):
        """chi and its derivatives (rho, theta, zeta), each grid.shape."""
        K, Mp, Nn = self.n_r, self.M + 1, 2 * self.N + 1
        d = d * self.chi_scale
        Dd = jnp.zeros((K, Mp * Nn), d.dtype).at[:, self.dense_idx].set(d.reshape(K, -1))
        Dd = Dd.reshape(K, Mp, Nn)
        nn = jnp.arange(-self.N, self.N + 1)
        mm = jnp.arange(Mp)
        A, Ar = (jnp.einsum("rk,kmn->rmn", P, Dd) for P in (self.Phi, self.Phir))
        H, Hr = (jnp.einsum("rmn,zn->rzm", X, self.Zf) for X in (A, Ar))
        Hz = jnp.einsum("rmn,zn->rzm", A, self.Zf * (-1j * nn * self.NFP))
        lam, lr, lt, lz = self.lam
        E = jnp.exp(1j * (self.theta + lam)[..., None] * mm)  # (r, t, z, m)
        re = lambda X: jnp.real(jnp.einsum("rzm,rtzm->rtz", X, E))  # noqa: E731
        c0, cr, cz, cs = re(H), re(Hr), re(Hz), re(H * (1j * mm))
        return self.rho + c0, 1 + cr + cs * lr, cs * (1 + lt), cz + cs * lz

    def __call__(self, c):
        """(p, grad p in (R, phi, Z)), shapes grid.shape and grid.shape + (3,)."""
        q, d = c[: self.n_q], c[self.n_q :]
        chi, xr, xt, xz = self.chi(d)
        s = jnp.clip(chi**2, 0.0, 1.0)
        Q, Qs = jax.jvp(lambda s: deboor(self.tq, q, s), (s,), (jnp.ones_like(s),))
        w = 1 - s
        p = w**3 * Q
        dp = jnp.where(chi**2 < 1, 2 * chi * (w**3 * Qs - 3 * w**2 * Q), 0.0)  # dP/dchi
        return p, self.geo.gradient(dp * xr, dp * xt, dp * xz)

    def gram(self, c, idx, weights, sw, phi, batch=8):
        """Gram matrix of the columns (d/dc_i of the scaled F and p residuals, i in idx) at c.

        ``weights``: (quantity, weight / normalization) of the local objectives."""
        k = len(idx)
        E = jnp.zeros((k, len(c)), c.dtype).at[jnp.arange(k), jnp.asarray(idx)].set(1.0)
        return np.asarray(_gram(self, c, E, sw, phi, tuple((q, float(s)) for q, s in weights),
                                batch), dtype=float)


@jax.jit
def _chi_values(tp, d):
    return tp.chi(d)[0]


@jax.jit
def _p_values(tp, c):
    return tp(c)[0]


@functools.partial(jax.jit, static_argnums=(5, 6))
def _gram(tp, c, E, sw, phi, weights, batch):
    from objective import _cartesian

    def column(e):
        dp, dgp = jax.jvp(tp.__call__, (c,), (e,))[1]
        out = []
        for qty, s in weights:
            if qty == "F":
                out.append((-s * sw[:, None] * _cartesian(dgp.reshape(-1, 3), phi)).ravel())
            else:
                out.append(s * sw * dp.ravel())
        return jnp.concatenate(out).astype(jnp.float32)

    Phi = jax.lax.map(column, E, batch_size=batch)
    return jnp.matmul(Phi, Phi.T, precision=jax.lax.Precision.HIGHEST)


class LabelDamping:
    """w_d d -> 0 (``LabelPressure.damping``): damps chi where P does not determine it;
    rms over the d's, so ``weight`` is about the rms d it costs as much as F / norm."""

    name, quantity = "chi damping", "x_p"
    local_columns = False

    def __init__(self, weight, pressure):
        self.weight = float(weight)
        w = pressure.damping()
        # of chi - rho itself: chi_scale d
        self._w = jnp.asarray(pressure.chi_scale * w / np.sqrt(max(np.count_nonzero(w), 1)))
        self.target, self.normalization = jnp.asarray(0.0), 1.0
        self.dim_f = len(w)

    def build(self, fields):
        return self

    def compute(self, f):
        return self._w * f["x_p"]

    def compute_scaled_error(self, f):
        return self.weight * self.compute(f)

    def tangent(self, d, base):
        return self.compute(d)

    def compute_scaled_tangent(self, e):
        return self.weight * e
