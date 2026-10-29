"""Fixed-boundary, matrix-free version of ``meanforce``: the closest state to J_fit under a
bound on DESC's mean force, with the pressure eliminated by the weak Poisson equation, on
the equilibrium's own (fixed) boundary.

Unknowns: u (the current potential's coefficients) only. On the fixed geometry, with
w_i = dV_i / V, sw = sqrt(w):

    a   = sw (J x B)                         (3 n)
    P c = sw grad p(c)                       (3 n, linear in c, CONSTANT: fixed geometry)
    p*  = G^+ P^T a,   G = P^T P (k x k)     (weak Poisson; G built once, by radial slices)
    f   = a - P p* = sw F,   M = sum_i w_i |F_i| / norm
    g   = sw (J - J_fit) / rms J_fit

Nothing of size n x k is stored: P c is one evaluation of grad p, P^T x its transpose,
G^-1 by a Cholesky factor of G (+ rank_tol^2 mean(diag) jitter; cuSOLVER's eigh fails for
k > ~25k), with one refinement step per state. The metric's H_J is
accumulated by radial slices of the collocation grid (no n x n_u table). The trust-region
solver, the AL outer loop and the weights are ``meanforce``'s (``Solver``, ``Weights``): this
module only supplies an anchor (``FixedAnchor``) and rows (``FixedRows``) with the same
interface. There is no B.n constraint and no flux constraint (fixed boundary); the flux is
reported for monitoring only if a section is supplied (not implemented: None).
"""

import functools
import time

import jax
import jax.numpy as jnp
import numpy as np

from flux_interp import TensorGeometry, TensorGrid
from objective import _cartesian

import meanforce as mf


# ------------------------------------------------------------------ pieces
def _J(fields, u, I):
    return _cartesian(fields.J_col(u, I).reshape(-1, 3), fields.phi)


def _gradp(fields, c):
    _, gp = fields.p_col(c)
    return _cartesian(gp.reshape(-1, 3), fields.phi)


def _Pmul(fields, sw, c):
    """P c = sw grad p(c), flattened (3 n)."""
    return (sw[:, None] * _gradp(fields, c)).ravel()


def _PT(fields, sw, x, n_p):
    """P^T x (n_p): the adjoint of c -> P c."""
    return jax.linear_transpose(lambda c: _Pmul(fields, sw, c),
                                jax.ShapeDtypeStruct((n_p,), x.dtype))(x)[0]


def _project(fields, sw, x, Lg, n_p):
    """(I - P G^-1 P^T) x and the coefficients G^-1 P^T x (G = Lg Lg^T, Cholesky)."""
    c = jax.scipy.linalg.cho_solve((Lg, True), _PT(fields, sw, x, n_p))
    return x - _Pmul(fields, sw, c), c


@jax.jit
def _state_impl(fields, u, Jfit, sw, Lg, rmsJ, norm):
    n_p = Lg.shape[0]
    J = _J(fields, u, fields.I)
    B = fields.B_ext + fields.B_plasma(u)
    a = (sw[:, None] * jnp.cross(J, B)).ravel()
    f, c = _project(fields, sw, a, Lg, n_p)
    # one refinement step of the normal equations (cheap; tightens the Galerkin identity)
    f, dc = _project(fields, sw, f, Lg, n_p)
    c = c + dc
    n = J.shape[0]
    F = f.reshape(n, 3) / sw[:, None]
    w = sw**2
    absF = jnp.linalg.norm(F, axis=1)
    g = (sw[:, None] * (J - Jfit) / rmsJ).ravel()
    PTf = _PT(fields, sw, f, n_p)
    return {"J": J, "B": B, "f": f, "c": c, "F": F, "z": w[:, None] * F, "absF": absF,
            "M": jnp.sum(w * absF) / norm, "rmsF": jnp.sqrt(jnp.sum(w * absF**2)) / norm,
            "g": g, "PTf": PTf}


def _lin_impl(fields, v, aux):
    """v (n_u) -> (dg, dz): linear."""
    sw, J, B = aux["sw"], aux["J"], aux["B"]
    n_p = aux["Lg"].shape[0]
    dJ = _J(fields, v, 0.0)
    dB = fields.B_plasma(v, 0.0)
    da = (sw[:, None] * (jnp.cross(dJ, B) + jnp.cross(J, dB))).ravel()
    df, _ = _project(fields, sw, da, aux["Lg"], n_p)
    n = J.shape[0]
    dz = (sw[:, None] * df.reshape(n, 3)).ravel()
    dg = (sw[:, None] * dJ / aux["rmsJ"]).ravel()
    return dg, dz


_lin = jax.jit(_lin_impl)


def _linT_impl(fields, w, aux, n_u):
    x = jax.ShapeDtypeStruct((n_u,), jnp.float64)
    return jax.linear_transpose(lambda v: _lin_impl(fields, v, aux), x)(w)[0]


_linT = jax.jit(_linT_impl, static_argnums=(3,))


# ------------------------------------------------------ slice-wise Gram matrices
def _slices(fields, current, pressure):
    """Per radial slice of the collocation grid: (rows, J evaluator, p evaluator, phi)."""
    tg = fields.force.tgrid
    n_t, n_z = len(tg.theta), len(tg.zeta)
    m = n_t * n_z
    for i in range(len(tg.rho)):
        g = TensorGrid(tg.rho[i : i + 1], tg.theta, tg.zeta, tg.NFP)
        geo = TensorGeometry(current._RZ, g)
        yield slice(i * m, (i + 1) * m), g, geo


@jax.jit
def _cols_J(tc, E):
    """d(J in (R, phi, Z))/du at a slice for unit vectors E (k, n_u): (k, 3 m)."""
    return jax.vmap(lambda e: tc(e, 0.0).reshape(-1))(E)


@jax.jit
def _cols_gp(tp, E):
    return jax.vmap(lambda e: tp(e)[1].reshape(-1))(E)


def _gram(make_eval, cols_fn, n_cols, fields, current, pressure, scale, chunk, log, what):
    """sum over radial slices of (scale sw C)(scale sw C)^T, C the columns at the slice.

    Memory: H (n_pad^2) + one slice's columns C (n_pad x 3 m, preallocated, filled chunk by
    chunk with donation) + a (chunk x n_pad) temporary: H is updated in row blocks, never
    through a full-size C C^T temporary."""
    sw = np.asarray(fields.sw)
    n_pad = -(-n_cols // chunk) * chunk
    H = jnp.zeros((n_pad, n_pad))
    t0 = time.time()
    n_sl = len(fields.force.tgrid.rho)
    C = None
    for k, (sl, g, geo) in enumerate(_slices(fields, current, pressure)):
        ev = make_eval(g, geo)
        s3 = jnp.asarray(np.repeat(sw[sl], 3)) * scale
        if C is None or C.shape[1] != s3.shape[0]:
            C = jnp.zeros((n_pad, s3.shape[0]))
        for i in range(0, n_pad, chunk):
            C = _fill(cols_fn, ev, C, s3, i, chunk, n_cols)
        for r in range(0, n_pad, chunk):
            H = _rows_add(H, C, r, chunk)
        if k == 0 or (k + 1) % 8 == 0 or k + 1 == n_sl:
            log(f"    {what}: slice {k + 1}/{n_sl}, {time.time() - t0:.0f} s")
    del C
    if n_pad != n_cols:
        H = H[:n_cols, :n_cols]
    log(f"  {what} Gram ({n_cols} x {n_cols}) by radial slices in {time.time() - t0:.0f} s")
    return H


@functools.partial(jax.jit, static_argnums=(0, 5, 6), donate_argnums=(2,))
def _fill(cols_fn, ev, C, s3, i0, chunk, n):
    E = jax.nn.one_hot(i0 + jnp.arange(chunk), n, dtype=jnp.float64)  # rows >= n are zero
    return jax.lax.dynamic_update_slice_in_dim(C, cols_fn(ev, E) * s3[None], i0, 0)


@functools.partial(jax.jit, static_argnums=(3,), donate_argnums=(0,))
def _rows_add(H, C, r0, blk):
    Cr = jax.lax.dynamic_slice_in_dim(C, r0, blk, 0)
    Hr = jax.lax.dynamic_slice_in_dim(H, r0, blk, 0) + Cr @ C.T
    return jax.lax.dynamic_update_slice_in_dim(H, Hr, r0, 0)


def gram_J(fields, current, pressure, rmsJ, chunk=1024, log=print):
    from flux_interp import TensorCurrent
    return _gram(lambda g, geo: TensorCurrent(current.S, current._RZ, g, geo), _cols_J,
                 fields.n_u, fields, current, pressure, 1.0 / rmsJ, chunk, log, "H_J")


def gram_P(fields, current, pressure, chunk=1024, log=print):
    return _gram(lambda g, geo: pressure.tensor(g, geo, current._RZ), _cols_gp,
                 fields.n_p, fields, current, pressure, 1.0, chunk, log, "G = P^T P")


# ------------------------------------------------------------------ anchor
class FixedAnchor:
    """The fixed-boundary anchor with ``meanforce.Anchor``'s interface (n_u, n_xi = 0, n,
    state(z), galerkin(s), raw_lin/raw_lin_T are replaced by ``FixedRows``)."""

    def __init__(self, fields, current, pressure, u0, rmsJ, norm, G, rank_tol=1e-12,
                 log=print):
        self.fields, self.log = fields, log
        self.n_u, self.n_p, self.n_xi = fields.n_u, fields.n_p, 0
        self.n = self.n_u
        self.u0 = jnp.asarray(u0)
        self.rmsJ, self.norm = float(rmsJ), float(norm)
        self.n_pts = len(fields.phi)
        self.n_b = 1
        self.sw = jnp.asarray(fields.sw)
        # Cholesky (cuSOLVER's eigh fails above ~25k); G is full rank here (cond(P) ~1e5),
        # a jitter of rank_tol^2 x mean(diag) guards round-off; one refinement step in state()
        d = jnp.diag(G)
        Lg = jnp.linalg.cholesky(G + (rank_tol**2) * jnp.mean(d) * jnp.eye(G.shape[0]))
        if not bool(jnp.all(jnp.isfinite(jnp.diag(Lg)))):
            raise FloatingPointError("Cholesky of G = P^T P failed: pressure space rank-deficient")
        dl = np.asarray(jnp.diag(Lg))
        self.p_rank, self.p_cond = int(G.shape[0]), float((dl.max() / dl.min()) ** 2)
        self.Lg = Lg
        self.colnorm = jnp.sqrt(jnp.clip(jnp.diag(G), 1e-300))
        log(f"  pressure space: {self.p_rank} columns, Cholesky; cond(G) ~ {self.p_cond:.2e} "
            f"(diag ratio^2; cond(P) ~ {np.sqrt(self.p_cond):.2e})")
        self.Jfit = _J(fields, self.u0, fields.I)
        self.tables = None
        self.info = {}

    def state(self, z):
        u = jnp.asarray(z)[: self.n_u]
        s = _state_impl(self.fields, u, self.Jfit, self.sw, self.Lg, self.rmsJ, self.norm)
        s = dict(s)
        gal = jnp.max(jnp.abs(s["PTf"]) / (self.colnorm * jnp.linalg.norm(s["f"])))
        s.update({"z_vec": jnp.asarray(z), "xi": jnp.zeros(0), "sw": self.sw,
                  "galerkin": gal, "Bn_rms": 0.0, "Psi": 1.0, "rel": jnp.zeros(1),
                  "Jfit": self.Jfit})
        self.last_z, self.last_s = jnp.asarray(z), s
        return s

    def galerkin(self, s):
        return float(s["galerkin"])

    def aux(self, s):
        return {"sw": self.sw, "J": s["J"], "B": s["B"], "Lg": self.Lg, "rmsJ": self.rmsJ}


class FixedRows:
    """``meanforce.Rows`` for the fixed boundary: rows g, s_i z_i (majorizer of M) and the
    penalty-curvature row of c_F; the B.n and flux rows are absent."""

    def __init__(self, anchor, s, W, norm):
        self.A, self.s, self.W = anchor, s, W
        c = W.constraints(s)
        self.c = c
        w = np.asarray(anchor.sw) ** 2
        tau = W.tau_frac * W.eps * norm
        zn = jnp.linalg.norm(s["z"], axis=1)
        t = jnp.maximum(zn, jnp.asarray(w) * tau)
        d1F = W.d1(c[0], W.nu[0])
        self.d1 = (d1F, 0.0)
        self.sF = jnp.sqrt(d1F / (W.eps * norm * t))
        self.sF3 = jnp.repeat(self.sF, 3)
        aux = anchor.aux(s)
        f_ = anchor.fields
        n_u = anchor.n_u
        self.lin = lambda v: _lin(f_, v, aux)  # noqa: E731
        self.linT = lambda wg, wz: _linT(f_, (wg, wz), aux, n_u)  # noqa: E731
        n3 = anchor.n_pts * 3
        self.n3 = n3
        zhat = (s["z"] / jnp.maximum(zn, 1e-300)[:, None]).ravel()
        self.grad_cF = self.linT(jnp.zeros(n3), zhat / (W.eps * norm))
        self.aF = np.sqrt(W.d2_model(c[0], W.nu[0])) * self.grad_cF
        self.r = jnp.concatenate([s["g"], self.sF3 * s["z"].ravel(), jnp.zeros(1)])

    def J(self, v):
        dg, dz = self.lin(v)
        return jnp.concatenate([dg, self.sF3 * dz, jnp.stack([self.aF @ v])])

    def JT(self, w):
        n3 = self.n3
        out = self.linT(w[:n3], self.sF3 * w[n3 : 2 * n3])
        return out + w[2 * n3] * self.aF

    def grad(self):
        return self.JT(self.r)


class FixedRowsB(FixedRows):
    """FixedRows with the radial curvature of |z_i| reduced to beta / |z_i| (option 2):
    each force row is projected, dz -> (I - (1 - sqrt(beta)) zhat zhat^T) dz, at points above
    the smoothing floor, so the model of |z + dz| is |z| + dz_par + (beta dz_par^2 + |dz_perp|^2)
    / 2|z| (beta = 1: the majorizer; beta = 1/2: the most optimistic quadratic that never
    predicts |z| < 0). The residual is rescaled by 1/(1 - c) so that J^T r stays the exact
    gradient. Positive semidefinite: no z'' term, so no negative curvature."""

    beta = 0.5

    def __init__(self, anchor, s, W, norm):
        super().__init__(anchor, s, W, norm)
        w = anchor.sw ** 2
        tau = W.tau_frac * W.eps * norm
        zn = jnp.linalg.norm(s["z"], axis=1)
        act = zn > w * tau
        self.zh = s["z"] / jnp.maximum(zn, 1e-300)[:, None]
        self.cP = jnp.where(act, 1.0 - np.sqrt(self.beta), 0.0)
        zr = (s["z"] / (1.0 - self.cP)[:, None]).ravel()
        self.r = jnp.concatenate([s["g"], self.sF3 * zr, jnp.zeros(1)])

    def _P(self, x):
        x = x.reshape(-1, 3)
        return (x - self.cP[:, None] * self.zh * jnp.sum(self.zh * x, 1, keepdims=True)).ravel()

    def J(self, v):
        dg, dz = self.lin(v)
        return jnp.concatenate([dg, self.sF3 * self._P(dz), jnp.stack([self.aF @ v])])

    def JT(self, w):
        n3 = self.n3
        out = self.linT(w[:n3], self._P(self.sF3 * w[n3 : 2 * n3]))
        return out + w[2 * n3] * self.aF


def use_fixed_rows():
    """Make ``meanforce.Solver`` build ``FixedRows`` (it looks the class up by name)."""
    mf.Rows = FixedRows


def metric_L(HJ, reg=1e-10):
    """Lower Cholesky factor of H_J + reg mean(diag) I (on the device)."""
    n = HJ.shape[0]
    HJ = HJ.at[jnp.arange(n), jnp.arange(n)].add(reg * jnp.mean(jnp.diag(HJ)))
    return jnp.linalg.cholesky(HJ)


class FixedSolver(mf.Solver):
    """``meanforce.Solver`` with a complementarity-checked outer stop: a bound is accepted
    only when, besides feasibility and inner stationarity, the mean-force constraint is
    active whenever its multiplier is positive (|c_F| <= ``act_tol`` if nu_F > 0 before the
    update). ``meanforce.Solver.solve_alpha`` accepts any c_F <= tol: after an overshoot of
    nu_F the PHR inner optimum sits at c_F ~ -nu_F / rho (slack), and the sweep moved on
    from a feasible but non-optimal state (run fxL32, alpha 0.7: M = 0.972 eps)."""

    act_tol = 1e-2
    delta_reset = True  # Delta <- max(Delta, delta0) at the start of every outer iteration
    metric_refresh = None  # callable z -> (Metric, Delta): rebuild the TR metric per outer iteration
    mult_ls = False  # least-squares multiplier estimate at each bound's start; safeguard the updates

    def ls_multiplier(self, z):
        """nu_F minimizing |L^-1 (grad phi + nu grad c_F)| at z (stationarity of the
        Lagrangian in the metric norm; DESC's lsq-auglag initial multipliers), clipped >= 0."""
        s = self.A.state(z)
        rows = FixedRows(self.A, s, self.W, self.A.norm)
        gphi = rows.linT(s["g"], jnp.zeros(rows.n3))
        a, b = self.met.back_T(gphi), self.met.back_T(rows.grad_cF)
        return max(0.0, -float(a @ b) / max(float(b @ b), 1e-300))

    def solve_alpha(self, z, max_outer=25, tol=(1e-2, 2e-2, 1e-2), maxiter=40, rho_max=1e8,
                    gtol=1e-3, callback=None, stat_tol=1e-3, stat_window=10):
        W = self.W
        outer = []
        viol_prev = np.inf
        for k in range(max_outer):
            hist = []
            if self.mult_ls and k == 0:
                lam = self.ls_multiplier(z)
                self.log(f"  [multiplier] least-squares nu_F {lam:.3e} (was {W.nu[0]:.3e}) at the start of the bound")
                W.nu = np.array([lam, W.nu[1], W.nu[2]])
            # A multiplier/penalty update (or a new bound) changes the merit, so the radius
            # left by the previous inner solve is not a meaningful guess; never start an outer
            # iteration below delta0 (a too-large radius costs one or two rejected trials).
            if self.metric_refresh is not None:
                out = self.metric_refresh(z)
                if out is not None:
                    self.met, D = out
                    self.log(f"  [metric] refreshed at outer {k}; Delta {self.Delta} -> {D:.3e} (|L^-1 grad|)")
                    self.Delta = D
            elif self.delta_reset and self.Delta is not None and self.Delta < self.delta0:
                self.log(f"  [radius] reset Delta {self.Delta:.2e} -> {self.delta0:.2e} at outer {k}")
                self.Delta = self.delta0
            self._hist = hist  # the step model reads the last step's ratio (self-tuning beta)
            z, hist, status = self.inner(z, maxiter=maxiter, hist=hist, gtol=gtol,
                                         flat_tol=0.1 * stat_tol, flat_window=stat_window)
            s, f, phi, c = self.evaluate(z)
            # violation also counts slack with a positive multiplier (complementarity)
            slack = max(0.0, -c[0] - self.act_tol) if W.nu[0] > 0 else 0.0
            viol = max(max(c[0], 0.0), abs(c[2]), slack)
            phis = [h["phi"] for h in hist if h["accepted"]]
            flat = (len(phis) > stat_window
                    and abs(phis[-stat_window - 1] - phis[-1]) <= stat_tol * abs(phis[-1])
                    and hist[-1]["Delta_next"] > 1e-8)
            stationary = status == "stationary" or (flat and c[0] >= -0.01)
            active_ok = (W.nu[0] <= 0) or (c[0] >= -self.act_tol)
            nu_old = W.nu.copy()
            W.nu = np.array([W.d1(c[0], W.nu[0]), W.d1(c[1], W.nu[1]), W.nu[2] + W.rho * c[2]])
            if self.mult_ls:  # safeguard the first-order update by the least-squares estimate
                lam = self.ls_multiplier(z)
                if lam > 0 and not (0.25 * lam <= W.nu[0] <= 4.0 * lam):
                    clipped = float(np.clip(W.nu[0], 0.25 * lam, 4.0 * lam))
                    self.log(f"  [multiplier] update {W.nu[0]:.3e} clipped to {clipped:.3e} (LS {lam:.3e})")
                    W.nu[0] = clipped
            o = {"outer": k, "rho": W.rho, "nu_before": nu_old.tolist(), "nu": W.nu.tolist(),
                 "c": c.tolist(), "phi": phi, "M": float(s["M"]), "rmsF": float(s["rmsF"]),
                 "Bn_rms": 0.0, "Psi": 1.0, "viol": viol, "galerkin": self.A.galerkin(s),
                 "inner stop": status, "flat": bool(flat), "stationary": bool(stationary),
                 "active_ok": bool(active_ok), "inner": hist}
            outer.append(o)
            self.log("  outer {:2d} rho {:.1e}: M/eps {:.4f} phi {:.6e} nu_F {:.3e} -> {:.3e} "
                     "galerkin {:.1e} ({} inner, {}{}{})".format(
                         k, W.rho, 1 + c[0], phi, nu_old[0], W.nu[0], o["galerkin"], len(hist),
                         status, ", flat" if flat else "",
                         "" if active_ok else ", bound slack with nu_F > 0"))
            if callback is not None:
                callback(z, s, outer)
            if (c[0] <= tol[0] and abs(c[2]) <= tol[2] and k > 0 and stationary
                    and active_ok):
                break
            within = c[0] <= tol[0] and abs(c[2]) <= tol[2] and active_ok
            if viol > 0.25 * viol_prev and W.rho < rho_max and not within:
                W.rho *= 10.0
            viol_prev = viol
        return z, outer


# ----------------------------------------------- exact Hessian-vector model (opt-in)
def _hvp_impl(fields, v, aux, sc):
    """H v of the AL merit, exact up to the |z| smoothing: the GN part of g, the true
    curvature of sum |z_i| ((I - zhat zhat^T) / t_i on D v), the second-order term of the
    quadratic residual (zhat . z''[v, .]) and psi'' grad c (grad c . v). One forward and one
    adjoint annulus solve: the transpose carries all three weighted outputs at once."""
    sw, J, B = aux["sw"], aux["J"], aux["B"]
    n_p = aux["Lg"].shape[0]
    n = J.shape[0]
    dJv = _J(fields, v, 0.0)
    dBv = fields.B_plasma(v, 0.0)
    dav = (sw[:, None] * (jnp.cross(dJv, B) + jnp.cross(J, dBv))).ravel()
    dfv, _ = _project(fields, sw, dav, aux["Lg"], n_p)
    dzv = sw[:, None] * dfv.reshape(n, 3)
    dgv = (sw[:, None] * dJv / aux["rmsJ"]).ravel()
    zh, t, act = sc["zhat"], sc["t"], sc["act"]
    # d(z / t): (I - zhat zhat^T) / |z| where |z| > w tau, I / (w tau) where smoothed
    Hz = (dzv - act[:, None] * zh * jnp.sum(zh * dzv, 1, keepdims=True)) / t[:, None]
    q, _ = _project(fields, sw, (sw[:, None] * sc["zt"]).ravel(), aux["Lg"], n_p)
    k1 = sc["k1"]

    def Phi(w):
        dJ = _J(fields, w, 0.0)
        dB = fields.B_plasma(w, 0.0)
        da = (sw[:, None] * (jnp.cross(dJ, B) + jnp.cross(J, dB))).ravel()
        df, _ = _project(fields, sw, da, aux["Lg"], n_p)
        dz = (sw[:, None] * df.reshape(n, 3)).ravel()
        dg = (sw[:, None] * dJ / aux["rmsJ"]).ravel()
        a2 = (sw[:, None] * (jnp.cross(dJ, dBv) + jnp.cross(dJv, dB))).ravel()
        return dg, dz, a2

    x = jax.ShapeDtypeStruct(v.shape, v.dtype)
    out = jax.linear_transpose(Phi, x)((dgv, k1 * Hz.ravel(), k1 * q))[0]
    return out + sc["k2"] * sc["gc"] * jnp.dot(sc["gc"], v)


_hvp = jax.jit(_hvp_impl)


class FixedRowsH(FixedRows):
    """FixedRows plus the exact Hessian-vector product of L_A (``hvp``); the step model is
    then the second-order Taylor model of the plain merit (gradient unchanged: J^T r)."""

    def __init__(self, anchor, s, W, norm):
        super().__init__(anchor, s, W, norm)
        w = anchor.sw ** 2
        tau = W.tau_frac * W.eps * norm
        zn = jnp.linalg.norm(s["z"], axis=1)
        t = jnp.maximum(zn, w * tau)
        zhat = s["z"] / jnp.maximum(zn, 1e-300)[:, None]
        d1F = self.d1[0]
        self.sc = {"zhat": zhat, "t": t, "zt": s["z"] / t[:, None],
                   "act": (zn >= w * tau).astype(jnp.float64), "k1": d1F / (W.eps * norm),
                   "k2": W.d2(self.c[0], W.nu[0]), "gc": self.grad_cF}
        self._aux = anchor.aux(s)

    def hvp(self, v):
        return _hvp(self.A.fields, v, self._aux, self.sc)


def _tridiag_tr_indef(alpha, beta, gnorm, Delta):
    """min gnorm e1.h + h^T T h / 2, |h| <= Delta, T tridiagonal possibly indefinite (More-
    Sorensen on the eigendecomposition, with the hard case). Returns h, mu, hit, h_newton
    (the mu = 0 solution if T > 0, else the boundary solution)."""
    k = len(alpha)
    from scipy.linalg import eigh_tridiagonal  # O(k^2), vs O(k^3) for the dense eigh
    th, V = eigh_tridiagonal(np.asarray(alpha, float), np.asarray(beta[: k - 1], float))
    b = -gnorm * V[0]
    lmin = th.min()
    scale = max(abs(th).max(), 1e-300)

    def h(mu):
        return V @ (b / (th + mu))

    if lmin > 1e-14 * scale:
        h0 = h(0.0)
        if np.linalg.norm(h0) <= Delta:
            return h0, 0.0, False, h0
    else:
        h0 = None
    lo = max(0.0, -lmin) + 1e-14 * scale
    # hard case: no mu > -lmin reaches the boundary
    if np.linalg.norm(h(lo)) < Delta:
        hh = h(lo)
        e = V[:, np.argmin(th)]
        a2 = Delta**2 - hh @ hh
        tau = np.sqrt(max(a2, 0.0))
        return hh + tau * e, lo, True, (h0 if h0 is not None else hh + tau * e)
    hi = lo + np.linalg.norm(b) / Delta + scale
    for _ in range(200):
        mu = np.sqrt(lo * hi) if lo > 0 else 0.5 * (lo + hi)
        nh = np.linalg.norm(h(mu))
        if abs(nh - Delta) <= 1e-8 * Delta:
            break
        if nh > Delta:
            lo = mu
        else:
            hi = mu
    hb = h(mu)
    return hb, mu, True, (h0 if h0 is not None else hb)


LAST_H = {}
DIAG_H = []  # one entry per trust-region subproblem (accepted or rejected)


def lanczos_tr_step_H(Hv, grad, f0, met, Delta, tol=1e-2, maxiter=100):
    """GLTR on the exact second-order model m(y) = f0 + g.y + y^T B y / 2, B = L^-1 H L^-T
    (possibly indefinite). Same return signature as ``pointwise.lanczos_tr_step``; the 'share'
    is the model decrease of the unconstrained Krylov minimizer relative to f0 (clipped)."""
    Bv = lambda v: met.back_T(Hv(met.back(v)))  # noqa: E731
    g = met.back_T(grad)
    gnorm = float(jnp.linalg.norm(g))
    Q = [g / gnorm]
    alpha, beta = [], []
    q_prev, b_prev = jnp.zeros_like(g), 0.0
    for k in range(1, maxiter + 1):
        wv = Bv(Q[-1]) - b_prev * q_prev
        a_k = float(Q[-1] @ wv)
        wv = wv - a_k * Q[-1]
        Qm = jnp.stack(Q)
        wv = wv - Qm.T @ (Qm @ wv)
        b_k = float(jnp.linalg.norm(wv))
        alpha.append(a_k)
        beta.append(b_k)
        h, mu, hit, h0 = _tridiag_tr_indef(np.array(alpha), np.array(beta), gnorm, Delta)
        if b_k * abs(h[-1]) <= tol * gnorm or b_k <= 1e-14 * gnorm:
            break
        q_prev, b_prev = Q[-1], b_k
        Q.append(wv / b_k)
    Qm = jnp.stack(Q[: len(alpha)])
    y = Qm.T @ jnp.asarray(h)
    Tm = np.diag(alpha) + np.diag(beta[:-1], 1) + np.diag(beta[:-1], -1)
    model = lambda hh: f0 + gnorm * hh[0] + 0.5 * hh @ Tm @ hh  # noqa: E731
    m_step, m_n = model(h), model(h0)
    share = float(np.clip((f0 - m_n) / max(abs(f0), 1e-300), 0.0, 1.0))
    th = np.linalg.eigvalsh(Tm)
    LAST_H.clear()
    LAST_H.update({"lanczos": len(alpha), "theta_min": float(th[0]), "theta_max": float(th[-1]),
                   "res": float(beta[-1] * abs(h[-1]) / gnorm), "converged": bool(beta[-1] * abs(h[-1]) <= tol * gnorm),
                   "mu": float(mu), "hit": bool(hit), "yBy": float(h @ Tm @ h), "|y|": float(np.linalg.norm(h)),
                   "near_hard": bool(th[0] < 0 and abs(mu + th[0]) <= 1e-6 * max(abs(th[0]), 1e-300))})
    return y, float(m_step), hit, len(alpha), share, gnorm, mu


def use_exact_hvp(hybrid=False, far=0.1, fallback=5, **kw):
    """Backward-compatible wrapper: exact HVP (or the hybrid GN/HVP switch)."""
    use_models("hybrid" if hybrid else "hvp", far=far, fallback=fallback, **kw)


def use_models(mode, far=0.1, fallback=5, beta=0.5, solver=None, refresh=None, refresh_lz=None,
               refresh_dM=None, beta_adapt=False, beta_min=1.0 / 32, beta_hi=1.5, beta_lo=0.5, log=print,
               lz_stop="both"):
    """Install the step model for ``meanforce.Solver`` (GLTR) and, optionally, a state-
    triggered rebuild of the trust-region metric.

    mode: "gn" (majorizer rows), "beta" (FixedRowsB, radial curvature beta/|z|), "hvp"
    (exact second-order model, may be indefinite), "hybrid" (GN while |c_F| > ``far`` or for
    ``fallback`` subproblems after an exact-model step that made the solver cut the radius;
    exact model otherwise).

    refresh(rows) -> (L, Delta): rebuild the metric at the rows' state, before a subproblem,
      * when the previous subproblem hit the Lanczos cap (its step is truncated), or
      * rent-or-buy: when the Lanczos time spent above the first post-rebuild count,
        accumulated since the rebuild, exceeds the cost of one rebuild (METRIC_BUILD_T);
      * optional extra triggers: >= refresh_lz x the post-rebuild count, |dM/M| > refresh_dM.
    The Metric object is updated in place (the solver and the step share it); the radius is
    mapped to the new metric by the ratio of |L^-1 grad|.

    beta_adapt (mode "beta"): after each step, beta <- beta / 2 (>= beta_min) when the step was
    interior with ratio > beta_hi (model too conservative), beta <- min(1, 2 beta) when the
    ratio was < beta_lo or the step was rejected; it takes effect at the next accepted state
    (the rows are rebuilt there).

    lz_stop (models gn / beta): "both" (pointwise.lanczos_tr_step: stop when the trust-region
    step AND the unconstrained Gauss-Newton minimizer have converged; the latter only feeds the
    GN share and converges slowly in the soft directions) or "tr" (lanczos_tr_step_H on
    H = J^T J: stop on the trust-region residual alone; the share is the Krylov estimate)."""
    import types
    rows_cls = {"gn": FixedRows, "hvp": FixedRowsH, "hybrid": FixedRowsH}.get(mode)
    if mode == "beta":
        rows_cls = type("FixedRowsB_%g" % beta, (FixedRowsB,), {"beta": float(beta)})
    mf.Rows = rows_cls
    gn_step = mf.pw.lanczos_tr_step
    shim = types.SimpleNamespace(**{k: getattr(mf.pw, k) for k in dir(mf.pw) if not k.startswith("__")})
    st = {"prev_Delta": None, "prev_model": None, "gn_left": 0, "met_id": None, "k_ref": None,
          "M_ref": None, "last_k": None, "extra_t": 0.0, "n_hist": 0}

    def step(J, JT, r, met, Delta, tol=1e-2, maxiter=100):
        rows = J.__self__
        f0 = 0.5 * float(r @ r)
        cF = float(rows.c[0])
        M_now = float(rows.s["M"])
        # self-tuning beta from the solver's last step (applies at the next accepted state)
        hist = getattr(solver, "_hist", None) if solver is not None else None
        if mode == "beta" and beta_adapt and hist is not None and len(hist) > st["n_hist"]:
            h = hist[-1]
            b_old = rows_cls.beta
            if (not h["accepted"]) or h["ratio"] < beta_lo:
                rows_cls.beta = min(1.0, 2.0 * b_old)
            elif h["ratio"] > beta_hi and not h["hit"]:
                rows_cls.beta = max(beta_min, 0.5 * b_old)
            if rows_cls.beta != b_old:
                log(f"      [beta] {b_old:g} -> {rows_cls.beta:g} (last step ratio {h['ratio']:+.2f}, "
                    f"{'hit' if h['hit'] else 'interior'}, {'accepted' if h['accepted'] else 'rejected'})")
        if hist is not None:
            st["n_hist"] = len(hist)
        if id(met) != st["met_id"]:  # a new metric (outer-iteration rebuild): restart the trigger
            st.update(met_id=id(met), k_ref=None, M_ref=M_now, last_k=None, extra_t=0.0)
        if refresh is not None and st["last_k"] is not None and st["k_ref"] is not None:
            why = None
            if st["last_k"] >= maxiter:
                why = f"Lanczos hit the cap ({maxiter})"
            elif st["extra_t"] >= METRIC_BUILD_T[0]:
                why = (f"extra Lanczos time {st['extra_t']:.0f} s since the rebuild >= one rebuild "
                       f"({METRIC_BUILD_T[0]:.0f} s)")
            elif refresh_lz is not None and st["last_k"] >= refresh_lz * st["k_ref"]:
                why = f"Lanczos {st['last_k']} >= {refresh_lz:g} x {st['k_ref']}"
            elif refresh_dM is not None and abs(M_now / st["M_ref"] - 1.0) > refresh_dM:
                why = f"M moved {M_now / st['M_ref'] - 1.0:+.0%} since the rebuild"
            if why:
                D_old = float(jnp.linalg.norm(met.back_T(rows.JT(r))))  # |L_old^-1 grad| at this state
                L_new, D_new = refresh(rows)
                met.L = L_new
                # keep the radius in the same relative units (multiples of the gradient length):
                # the trust region's growth is preserved across the rebuild
                D_map = max(D_new, Delta * D_new / max(D_old, 1e-300))
                if solver is not None:
                    solver.Delta = D_map
                log(f"      [metric] rebuilt in the inner loop ({why}); Delta {Delta:.3e} -> {D_map:.3e} "
                    f"(|L^-1 g| {D_old:.3e} -> {D_new:.3e})")
                Delta = D_map
                st.update(k_ref=None, M_ref=M_now, last_k=None, prev_Delta=None, extra_t=0.0)
        model, why = ("gn" if mode in ("gn", "beta") else "hvp"), mode
        if mode == "hybrid":
            why = "near bound"
            if st["prev_model"] == "hvp" and st["prev_Delta"] is not None and Delta < 0.99 * st["prev_Delta"]:
                st["gn_left"] = fallback
            if abs(cF) > far:
                model, why = "gn", f"|c_F| {abs(cF):.2f} > {far}"
            elif st["gn_left"] > 0:
                model, why = "gn", f"fallback ({st['gn_left']} left)"
                st["gn_left"] -= 1
        st["prev_Delta"], st["prev_model"] = float(Delta), model
        t1 = time.time()
        if model == "gn" and lz_stop == "tr":
            out = lanczos_tr_step_H(lambda v: JT(J(v)), JT(r), f0, met, Delta, tol, maxiter)
            d = {"model": mode if mode == "beta" else "gn", "why": why, "lanczos": int(out[3]),
                 "Delta": float(Delta), "f0": f0, "c_F": cF, "t_sub": time.time() - t1,
                 "theta_min": LAST_H["theta_min"], "theta_max": LAST_H["theta_max"], "res": LAST_H["res"]}
            tag = "beta" if mode == "beta" else "gn "
            print(f"      [{tag}] lanczos {d['lanczos']} (TR stop; theta [{d['theta_min']:.2e}, {d['theta_max']:.2e}], "
                  f"hit {bool(out[2])}) t {d['t_sub']:.1f}s", flush=True)
        elif model == "gn":
            out = gn_step(J, JT, r, met, Delta, tol, maxiter)
            d = {"model": mode if mode == "beta" else "gn", "why": why, "lanczos": int(out[3]),
                 "Delta": float(Delta), "f0": f0, "c_F": cF, "t_sub": time.time() - t1}
            tag = "beta" if mode == "beta" else "gn "
            print(f"      [{tag}] lanczos {d['lanczos']} ({why}) t {d['t_sub']:.1f}s", flush=True)
        else:
            out = lanczos_tr_step_H(rows.hvp, rows.JT(r), f0, met, Delta, tol, maxiter)
            d = dict(LAST_H, model="hvp", why=why, Delta=float(Delta), f0=f0, c_F=cF, t_sub=time.time() - t1)
            Jy = J(met.back(out[0]))  # GN curvature along the step taken (one extra product)
            d["yGNy"] = float(Jy @ Jy)
            d["curv_ratio"] = d["yBy"] / max(d["yGNy"], 1e-300)
            print(f"      [hvp] lanczos {d['lanczos']} theta_min {d['theta_min']:+.2e} theta_max {d['theta_max']:.2e} "
                  f"mu {d['mu']:.2e} hit {d['hit']} near_hard {d['near_hard']} yHy/yGNy {d['curv_ratio']:.3f} "
                  f"t {d['t_sub']:.1f}s", flush=True)
        if mode == "beta":
            d["beta"] = float(rows_cls.beta)
        DIAG_H.append(d)
        k_now = max(int(out[3]), 1)
        st["last_k"] = k_now
        if st["k_ref"] is None:
            st["k_ref"] = k_now
        else:
            st["extra_t"] += max(0, k_now - st["k_ref"]) * (d["t_sub"] / k_now)
        return out

    shim.lanczos_tr_step = step
    mf.pw = shim


def embed_index(lo, hi):
    """Positions in ``hi.u`` of every coefficient of ``lo.u`` (two BoundaryPotentials whose
    bases are nested: chi by (m, n), then a and b by FourierZernike (l, m, n)). Raises if a
    low-resolution mode has no high-resolution counterpart."""
    pos = {}
    for i, mn in enumerate(map(tuple, hi.S.chi_modes)):
        pos[("chi",) + mn] = i
    K = hi.S.K
    for j, lmn in enumerate(map(tuple, hi.basis.modes)):
        pos[("a",) + lmn] = hi.S.n_chi + j
        pos[("b",) + lmn] = hi.S.n_chi + K + j
    keys = [("chi",) + tuple(mn) for mn in lo.S.chi_modes]
    keys += [("a",) + tuple(lmn) for lmn in lo.basis.modes] + [("b",) + tuple(lmn) for lmn in lo.basis.modes]
    missing = [k for k in keys if k not in pos]
    if missing:
        raise ValueError(f"{len(missing)} low-resolution modes missing, e.g. {missing[:3]}")
    assert lo.n_free == len(keys) and hi.n_free == hi.S.n_chi + 2 * K
    return np.array([pos[k] for k in keys])


def embed(lo, hi, u_lo, idx=None):
    idx = embed_index(lo, hi) if idx is None else idx
    z = np.zeros(hi.n_free)
    z[idx] = np.asarray(u_lo)
    return z


# ------------------------------------------- local force preconditioner (NEST-main style)
METRIC_BUILD_T = [130.0]  # seconds of the last force-metric build (rent-or-buy rebuild rule)
@jax.jit
def _cols_JxB(ev, E):
    """sw-less columns d(J x B0)/du at a slice, B0 frozen, both in (R, phi, Z): (k, 3 m)."""
    tc, Bl = ev
    return jax.vmap(lambda e: jnp.cross(tc(e, 0.0).reshape(-1, 3), Bl).reshape(-1))(E)


@functools.partial(jax.jit, static_argnums=(4,), donate_argnums=(0,))
def _blk_add(H, X, Y, w, blk, r0):
    """H[r0:r0+blk] += (X[r0:r0+blk] * w) @ Y^T (row block, donated). The product runs in
    X's dtype (FP32 when the slice columns are FP32) and is accumulated in H's dtype."""
    Xr = jax.lax.dynamic_slice_in_dim(X, r0, blk, 0) * w[None].astype(X.dtype)
    Hr = jax.lax.dynamic_slice_in_dim(H, r0, blk, 0) + (Xr @ Y.T).astype(H.dtype)
    return jax.lax.dynamic_update_slice_in_dim(H, Hr, r0, 0)


@functools.partial(jax.jit, static_argnums=(4,), donate_argnums=(0,))
def _blk_add_tf32(H, X, Y, w, blk, r0):
    """_blk_add with both matmul inputs rounded to TF32 (emulates GPU tensor-core FP32)."""
    Xr = _tf32_round(jax.lax.dynamic_slice_in_dim(X, r0, blk, 0) * w[None].astype(jnp.float32))
    Hr = jax.lax.dynamic_slice_in_dim(H, r0, blk, 0) + (Xr @ _tf32_round(Y).T).astype(H.dtype)
    return jax.lax.dynamic_update_slice_in_dim(H, Hr, r0, 0)


def _tf32_round(x):
    """Round FP32 values to TF32 (10 explicit mantissa bits; round to nearest)."""
    b = jax.lax.bitcast_convert_type(x.astype(jnp.float32), jnp.uint32)
    b = (b + jnp.uint32(0x1000)) & jnp.uint32(0xFFFFE000)
    return jax.lax.bitcast_convert_type(b, jnp.float32)


def gram_F_local(fields, current, pressure, B, q, Lg, chunk=1024, log=print, precision="fp64"):
    """Gauss-Newton Gram of the LOCAL part of the weighted force rows q * Pi(sw (dJ x B0)):
    the J0 x dB (Biot-Savart) part is dropped, the pressure projection Pi = I - P G^-1 P^T
    is kept exactly. With C = [sw (dJ_j x B0)], P = [sw grad p_k], Q = diag(q^2) (per point):
        H = C^T Q C - C^T P G^-1 P^T Q C - (.)^T + C^T P G^-1 (P^T Q P) G^-1 P^T C,
    all four blocks accumulated over radial slices (points are local)."""
    from flux_interp import TensorCurrent
    sw = np.asarray(fields.sw)
    n_u, n_p = fields.n_u, fields.n_p
    Bc = np.asarray(_cartesian(jnp.asarray(B), -jnp.asarray(fields.phi)))  # to (R, phi, Z)
    q2 = np.asarray(q) ** 2
    nu_pad, np_pad = -(-n_u // chunk) * chunk, -(-n_p // chunk) * chunk
    CQC, CP = jnp.zeros((nu_pad, nu_pad)), jnp.zeros((nu_pad, np_pad))
    CQP, PQP = jnp.zeros((nu_pad, np_pad)), jnp.zeros((np_pad, np_pad))
    t0 = time.time()
    n_sl = len(fields.force.tgrid.rho)
    for k, (sl, g, geo) in enumerate(_slices(fields, current, pressure)):
        tc = TensorCurrent(current.S, current._RZ, g, geo)
        tp = pressure.tensor(g, geo, current._RZ)
        s3 = jnp.asarray(np.repeat(sw[sl], 3))
        w3 = jnp.asarray(np.repeat(q2[sl], 3))
        one = jnp.ones_like(w3)
        C = jnp.zeros((nu_pad, s3.shape[0]))
        for i in range(0, nu_pad, chunk):
            C = _fill(_cols_JxB, (tc, jnp.asarray(Bc[sl])), C, s3, i, chunk, n_u)
        Pm = jnp.zeros((np_pad, s3.shape[0]))
        for i in range(0, np_pad, chunk):
            Pm = _fill(_cols_gp, tp, Pm, s3, i, chunk, n_p)
        if precision in ("fp32", "tf32"):  # products in single precision, FP64 accumulators
            C, Pm = C.astype(jnp.float32), Pm.astype(jnp.float32)
        add = _blk_add_tf32 if precision == "tf32" else _blk_add
        for r in range(0, nu_pad, chunk):
            CQC = add(CQC, C, C, w3, chunk, r)
            CP = add(CP, C, Pm, one, chunk, r)
            CQP = add(CQP, C, Pm, w3, chunk, r)
        for r in range(0, np_pad, chunk):
            PQP = add(PQP, Pm, Pm, w3, chunk, r)
        del C, Pm
        if k == 0 or (k + 1) % 8 == 0 or k + 1 == n_sl:
            log(f"    H_F(local): slice {k + 1}/{n_sl}, {time.time() - t0:.0f} s")
    CQC, CP, CQP, PQP = CQC[:n_u, :n_u], CP[:n_u, :n_p], CQP[:n_u, :n_p], PQP[:n_p, :n_p]
    # release each block as soon as it is used (peak ~ CQC + CP + one n_p x n_u + one n_u^2)
    Y = jax.scipy.linalg.cho_solve((Lg, True), CQP.T)  # G^-1 P^T Q C (n_p, n_u)
    del CQP
    T2 = CP @ Y
    del Y
    X = jax.scipy.linalg.cho_solve((Lg, True), CP.T)   # G^-1 P^T C   (n_p, n_u)
    del CP
    H = CQC - T2
    del CQC
    H = H - T2.T
    del T2
    H = H + X.T @ (PQP @ X)
    del X, PQP
    H = 0.5 * (H + H.T)
    log(f"  H_F(local) Gram ({n_u} x {n_u}) in {time.time() - t0:.0f} s")
    return H


def force_metric(anchor, current, pressure, HJ_or_none, s, W, norm, rows=None, reg=1e-10, chunk=1024,
                 log=print, HJ_builder=None, precision="fp64"):
    """Metric L = chol(H_J + H_F(local) + a_F a_F^T) at state s and the current weights W:
    the Gauss-Newton matrix of the rows, minus only the J0 x dB part of the force rows."""
    t_b = time.time()
    rows = rows or FixedRows(anchor, s, W, norm)
    q = rows.sF * anchor.sw  # row weight per point: s_F,i sw_i (dz = sw df)
    HF = gram_F_local(anchor.fields, current, pressure, s["B"], q, anchor.Lg, chunk=chunk, log=log,
                      precision=precision)
    HJ = HJ_or_none if HJ_or_none is not None else HJ_builder()
    H = HJ + HF + jnp.outer(rows.aF, rows.aF)
    tr = (float(jnp.trace(HJ)), float(jnp.trace(HF)))
    L = metric_L(H, reg)
    jax.block_until_ready(L)
    METRIC_BUILD_T[0] = time.time() - t_b
    log(f"  metric: trace H_J {tr[0]:.3e}, trace H_F(local) {tr[1]:.3e}")
    return L, tr


# ------------------------------------------------ lean (low-memory) metric builds
# Peak device memory ~ 1 n_u^2 + 2 n_u n_p + 2 n_p^2 (+ one slice of columns) instead of
# ~5 n_u^2 + resident copies: no padded n_u^2 arrays (exact tail blocks), the projected Gram
# assembled in row blocks on CQC's buffer, triangular solves and Cholesky on donated buffers,
# H_J kept in host memory and added in row blocks.
def _blocks(n, chunk):
    """(start, size) row blocks covering [0, n): full chunks and one exact tail."""
    return [(r, min(chunk, n - r)) for r in range(0, n, chunk)]


@functools.partial(jax.jit, static_argnums=(3,), donate_argnums=(0,))
def _rows_add_lean(H, C, r0, blk):
    Cr = jax.lax.dynamic_slice_in_dim(C, r0, blk, 0)
    Hr = jax.lax.dynamic_slice_in_dim(H, r0, blk, 0) + Cr @ C.T
    return jax.lax.dynamic_update_slice_in_dim(H, Hr, r0, 0)


def _gram_lean(make_eval, cols_fn, n_cols, fields, current, scale, chunk, log, what):
    """``_gram`` without padding: H (n x n) + one slice of columns."""
    sw = np.asarray(fields.sw)
    H = jnp.zeros((n_cols, n_cols))
    t0 = time.time()
    n_sl = len(fields.force.tgrid.rho)
    C = None
    bl = _blocks(n_cols, chunk)
    for k, (sl, g, geo) in enumerate(_slices(fields, current, None)):
        ev = make_eval(g, geo)
        s3 = jnp.asarray(np.repeat(sw[sl], 3)) * scale
        if C is None or C.shape[1] != s3.shape[0]:
            C = jnp.zeros((n_cols, s3.shape[0]))
        for i, b in bl:
            C = _fill(cols_fn, ev, C, s3, i, b, n_cols)
        for r, b in bl:
            H = _rows_add_lean(H, C, r, b)
        if k == 0 or (k + 1) % 8 == 0 or k + 1 == n_sl:
            log(f"    {what}: slice {k + 1}/{n_sl}, {time.time() - t0:.0f} s")
    del C
    log(f"  {what} Gram ({n_cols} x {n_cols}) by radial slices (lean) in {time.time() - t0:.0f} s")
    return H


def gram_J_lean(fields, current, pressure, rmsJ, chunk=1024, log=print):
    from flux_interp import TensorCurrent
    return _gram_lean(lambda g, geo: TensorCurrent(current.S, current._RZ, g, geo), _cols_J,
                      fields.n_u, fields, current, 1.0 / rmsJ, chunk, log, "H_J")


@functools.partial(jax.jit, donate_argnums=(1,))
def _rsolve_T(Lg, X):
    """X Lg^-T (X: (n, n_p)), on X's buffer."""
    return jax.lax.linalg.triangular_solve(Lg, X, left_side=False, lower=True, transpose_a=True)


@functools.partial(jax.jit, donate_argnums=(1,))
def _lsolve(Lg, X):
    """Lg^-1 X, on X's buffer."""
    return jax.lax.linalg.triangular_solve(Lg, X, left_side=True, lower=True, transpose_a=False)


@functools.partial(jax.jit, static_argnums=(4,), donate_argnums=(0,))
def _D_rows(Bt, At, S, r0, blk):
    """Bt[r] -= 0.5 At[r] S (row block, donated)."""
    Br = jax.lax.dynamic_slice_in_dim(Bt, r0, blk, 0) - 0.5 * (jax.lax.dynamic_slice_in_dim(At, r0, blk, 0) @ S)
    return jax.lax.dynamic_update_slice_in_dim(Bt, Br, r0, 0)


@functools.partial(jax.jit, static_argnums=(4,), donate_argnums=(0,))
def _H_rows(H, At, Dt, r0, blk):
    """H[r] -= At[r] Dt^T + Dt[r] At^T (row block, donated)."""
    Ar = jax.lax.dynamic_slice_in_dim(At, r0, blk, 0)
    Dr = jax.lax.dynamic_slice_in_dim(Dt, r0, blk, 0)
    Hr = jax.lax.dynamic_slice_in_dim(H, r0, blk, 0) - Ar @ Dt.T - Dr @ At.T
    return jax.lax.dynamic_update_slice_in_dim(H, Hr, r0, 0)


@functools.partial(jax.jit, static_argnums=(4,), donate_argnums=(0,))
def _add_rows(H, Hr_host, a, r0, blk):
    """H[r] += Hr_host + a[r] a^T (row block; Hr_host: the block of a host matrix)."""
    ar = jax.lax.dynamic_slice_in_dim(a, r0, blk, 0)
    Hr = jax.lax.dynamic_slice_in_dim(H, r0, blk, 0) + Hr_host + ar[:, None] * a[None, :]
    return jax.lax.dynamic_update_slice_in_dim(H, Hr, r0, 0)


@functools.partial(jax.jit, donate_argnums=(0,))
def _chol_donated(H, reg):
    n = H.shape[0]
    H = H.at[jnp.arange(n), jnp.arange(n)].add(reg * jnp.mean(jnp.diag(H)))
    return jnp.linalg.cholesky(H)


def metric_L_lean(H, reg=1e-10):
    """``metric_L`` on H's buffer (H is consumed)."""
    return _chol_donated(H, reg)


def gram_F_local_lean(fields, current, pressure, B, q, Lg, chunk=1024, log=print):
    """``gram_F_local`` (FP64) with H = CQC - At Dt^T - Dt At^T, At = (C^T P) Lg^-T,
    Dt = (C^T Q P) Lg^-T - At S / 2, S = Lg^-1 (P^T Q P) Lg^-T: algebraically the same matrix."""
    from flux_interp import TensorCurrent
    sw = np.asarray(fields.sw)
    n_u, n_p = fields.n_u, fields.n_p
    Bc = np.asarray(_cartesian(jnp.asarray(B), -jnp.asarray(fields.phi)))
    q2 = np.asarray(q) ** 2
    CQC, CP = jnp.zeros((n_u, n_u)), jnp.zeros((n_u, n_p))
    CQP, PQP = jnp.zeros((n_u, n_p)), jnp.zeros((n_p, n_p))
    bu, bp = _blocks(n_u, chunk), _blocks(n_p, chunk)
    t0 = time.time()
    n_sl = len(fields.force.tgrid.rho)
    for k, (sl, g, geo) in enumerate(_slices(fields, current, pressure)):
        tc = TensorCurrent(current.S, current._RZ, g, geo)
        tp = pressure.tensor(g, geo, current._RZ)
        s3 = jnp.asarray(np.repeat(sw[sl], 3))
        w3 = jnp.asarray(np.repeat(q2[sl], 3))
        one = jnp.ones_like(w3)
        C = jnp.zeros((n_u, s3.shape[0]))
        for i, b in bu:
            C = _fill(_cols_JxB, (tc, jnp.asarray(Bc[sl])), C, s3, i, b, n_u)
        Pm = jnp.zeros((n_p, s3.shape[0]))
        for i, b in bp:
            Pm = _fill(_cols_gp, tp, Pm, s3, i, b, n_p)
        for r, b in bu:
            CQC = _blk_add(CQC, C, C, w3, b, r)
            CP = _blk_add(CP, C, Pm, one, b, r)
            CQP = _blk_add(CQP, C, Pm, w3, b, r)
        for r, b in bp:
            PQP = _blk_add(PQP, Pm, Pm, w3, b, r)
        del C, Pm
        if k == 0 or (k + 1) % 8 == 0 or k + 1 == n_sl:
            log(f"    H_F(local, lean): slice {k + 1}/{n_sl}, {time.time() - t0:.0f} s")
    At = _rsolve_T(Lg, CP)    # (C^T P) Lg^-T, on CP's buffer
    Dt = _rsolve_T(Lg, CQP)   # (C^T Q P) Lg^-T, on CQP's buffer
    S = _rsolve_T(Lg, _lsolve(Lg, PQP))  # Lg^-1 PQP Lg^-T
    for r, b in bu:
        Dt = _D_rows(Dt, At, S, r, b)
    del S
    H = CQC
    del CQC
    for r, b in bu:
        H = _H_rows(H, At, Dt, r, b)
    del At, Dt
    log(f"  H_F(local) Gram ({n_u} x {n_u}, lean) in {time.time() - t0:.0f} s")
    return H


def force_metric_lean(anchor, current, pressure, HJ_host, s, W, norm, rows=None, reg=1e-10,
                      chunk=1024, log=print, return_H=False):
    """``force_metric`` with the lean build; HJ_host: H_J as a host (numpy) array."""
    t_b = time.time()
    rows = rows or FixedRows(anchor, s, W, norm)
    q = rows.sF * anchor.sw
    H = gram_F_local_lean(anchor.fields, current, pressure, s["B"], q, anchor.Lg, chunk=chunk, log=log)
    trF = float(jnp.trace(H))
    trJ = float(np.trace(HJ_host))
    aF = jnp.asarray(rows.aF)
    for r, b in _blocks(H.shape[0], chunk):
        H = _add_rows(H, jnp.asarray(HJ_host[r : r + b]), aF, r, b)
    if return_H:
        return H, (trJ, trF)
    L = metric_L_lean(H, reg)
    jax.block_until_ready(L)
    METRIC_BUILD_T[0] = time.time() - t_b
    log(f"  metric: trace H_J {trJ:.3e}, trace H_F(local) {trF:.3e} (lean build)")
    return L, (trJ, trF)


def embed_truncate(lo, hi, u_lo):
    """``embed`` for bases that are not nested (e.g. lower N in ``hi``): the coefficients of
    lo's modes that hi lacks are dropped. Returns (u_hi, dropped |u| fraction, n dropped)."""
    pos = {}
    for i, mn in enumerate(map(tuple, hi.S.chi_modes)):
        pos[("chi",) + mn] = i
    K = hi.S.K
    for j, lmn in enumerate(map(tuple, hi.basis.modes)):
        pos[("a",) + lmn] = hi.S.n_chi + j
        pos[("b",) + lmn] = hi.S.n_chi + K + j
    keys = [("chi",) + tuple(mn) for mn in lo.S.chi_modes]
    keys += [("a",) + tuple(lmn) for lmn in lo.basis.modes] + [("b",) + tuple(lmn) for lmn in lo.basis.modes]
    u_lo = np.asarray(u_lo)
    z = np.zeros(hi.n_free)
    keep = np.array([k in pos for k in keys])
    z[[pos[k] for k, f in zip(keys, keep) if f]] = u_lo[keep]
    frac = float(np.linalg.norm(u_lo[~keep]) / max(np.linalg.norm(u_lo), 1e-300))
    return z, frac, int((~keep).sum())
