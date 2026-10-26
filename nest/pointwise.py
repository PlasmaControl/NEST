"""Pointwise force bound: min |J - J_fit|^2 subject to |F(x_i)| / norm <= eps at every point.

Unknowns x = (u, c): the potential's coefficients (J = curl T) and the pressure's
(p = (1 - rho^2) sum_j c_j phi_j, p = 0 on the LCFS). No Poisson equation is solved;
p enters only through the constraints. With w_i = dV_i / V and sw = sqrt(w):

    g(u)    = sw (J(u) - J0) / rms J0                       (3 n rows)
    c_i(x)  = |F_i|^2 / (eps norm)^2 - 1 <= 0               (one per point)
    Phi     = |g|^2 / 2

Augmented Lagrangian (Powell-Hestenes-Rockafellar), multipliers lam_i >= 0, penalty rho:

    L = |g|^2 / 2 + (rho / 2) sum_i w_i [max(0, c_i + lam_i / rho)^2 - (lam_i / rho)^2]
      = |r|^2 / 2 + const,     r = (g, sqrt(rho) sw_i m_i),   m_i = max(0, c_i + lam_i / rho)

minimized over x for fixed (lam, rho) by a trust-region Gauss-Newton method, then
lam_i <- max(0, lam_i + rho c_i), and rho x 10 when the largest violation did not fall
by 4. The trust region is measured in the physical metric M = blockdiag(H_J, H_p),
H_J = sum_i w_i dJ_i . dJ_i / rms J0^2 (= g's own Gram matrix), H_p = sum_i w_i dp_i dp_i
/ rms p0^2: |y| = |L^T d|, L L^T = M, is the rms change of J and p.

Two models give F and its linearization (``DenseModel``: from column tables, B_plasma =
G u; ``FieldsModel``: matrix-free, one annulus solve per product), and two step solvers:
``steihaug_step`` (truncated CG, matrix-free) and ``qr_tr_step`` (QR of the stacked
Jacobian [J L^-T | r], DESC's exact trust-region rule; the QR is any callable, e.g.
jaxmg's distributed one). This module has no DESC dependency (the dense solver runs in
an environment with a newer JAX).
"""

import functools
import time

import jax
import jax.numpy as jnp
import numpy as np


# ------------------------------------------------------------------- models
class DenseModel:
    """F, J and their derivatives from column tables (dict of arrays, ``pointwise_tables``):

    A (n_u, n, 3), J_I (n, 3)   J = sum_k u_k A_k + J_I            (Cartesian, at the points)
    G (n_u, n, 3), B0 (n, 3)    B = sum_k u_k G_k + B0              (B0 = B_ext + B_I)
    gP (n_p, n, 3), P (n_p, n)  grad p = sum_j c_j gP_j, p = sum_j c_j P_j
    """

    def __init__(self, T):
        self.T = {k: jnp.asarray(v) for k, v in T.items() if k in
                  ("A", "J_I", "G", "B0", "gP", "P")}
        self.n_u, self.n = T["A"].shape[:2]
        self.n_p = T["gP"].shape[0]
        self.dim = self.n_u + self.n_p
        self.kind, self.data = "dense", self.T

    def base(self, x):
        return _dense_base(x, self.T, self.n_u)

    def tangent(self, v, base):
        return _dense_tangent(v, base, self.T, self.n_u)


@jax.jit
def _dense_base_impl(u, c, T):
    J = jnp.einsum("k,knc->nc", u, T["A"]) + T["J_I"]
    B = jnp.einsum("k,knc->nc", u, T["G"]) + T["B0"]
    gp = jnp.einsum("j,jnc->nc", c, T["gP"])
    p = c @ T["P"]
    return {"J": J, "B": B, "F": jnp.cross(J, B) - gp, "p": p, "grad_p": gp}


def _dense_base(x, T, n_u):
    return _dense_base_impl(x[:n_u], x[n_u:], T)


@jax.jit
def _dense_tangent_impl(du, dc, J, B, T):
    dJ = jnp.einsum("k,knc->nc", du, T["A"])
    dB = jnp.einsum("k,knc->nc", du, T["G"])
    dF = jnp.cross(dJ, B) + jnp.cross(J, dB) - jnp.einsum("j,jnc->nc", dc, T["gP"])
    return {"J": dJ, "F": dF}


def _dense_tangent(v, base, T, n_u):
    return _dense_tangent_impl(v[:n_u], v[n_u:], base["J"], base["B"], T)


class FieldsModel:
    """F and its linearization from ``objective.Fields`` (matrix-free: each base and each
    tangent costs one annulus solve). ``fields.x0`` sets nothing here."""

    def __init__(self, fields):
        self.fields = fields
        self.n_u, self.n_p = fields.n_u, fields.n_p
        self.dim = self.n_u + self.n_p
        self.n = len(fields.phi)
        self.kind, self.data = "fields", fields

    def base(self, x):
        return _fields_base(x, self.fields)

    def tangent(self, v, base):
        return _fields_tangent(v, base["J"], base["B"], self.fields)


@jax.jit
def _fields_base(x, fields):
    f = fields._compute(x)
    return {"J": f["J"], "B": f["B"], "F": f["F"], "p": f["p"], "grad_p": f["grad_p"]}


@jax.jit
def _fields_tangent(v, J, B, fields):
    d = fields._tangent(v, J, B)
    return {"J": d["J"], "F": d["F"]}


# ------------------------------------------------------- the AL least squares
class Problem:
    """The pointwise-bound AL problem on a model: residual, linearization, constraints.

    ``P``: dict with w (n,), norm, rmsJ0, J0 (n, 3) (the target current at the points).
    """

    def __init__(self, model, P):
        self.model = model
        self.w = jnp.asarray(P["w"])
        self.sw = jnp.sqrt(self.w)
        self.norm, self.rmsJ0 = float(P["norm"]), float(P["rmsJ0"])
        self.J0 = jnp.asarray(P["J0"])

    def constraints(self, base, eps):
        """c_i = |F_i|^2 / (eps norm)^2 - 1."""
        return jnp.sum(base["F"] ** 2, 1) / (eps * self.norm) ** 2 - 1.0

    def residual(self, x, lam, rho, eps, delta=0.0):
        """r = (g, sqrt(rho) sw m), m and its derivative dm/ds, c, the base fields.

        m = max(0, s) with s = c + lam / rho, or with ``delta`` > 0 the smoothed
        m = (s + sqrt(s^2 + delta^2)) / 2 (a C-infinity plus function: the Gauss-Newton
        model stays accurate when points enter or leave the active set)."""
        b = self.model.base(x)
        r, m, dm, c = _residual(b["J"], b["F"], self.J0, self.sw, lam, rho, eps * self.norm,
                                self.rmsJ0, delta)
        return r, (m, dm), c, b

    def linear(self, base, m, lam, rho, eps):
        """v -> J_r v and its transpose w -> J_r^T w at the base fields (jitted, with the
        model's arrays as inputs); ``m`` = (m, dm/ds) of ``residual``."""
        kind, data = self.model.kind, self.model.data
        aux = (base["J"], base["B"], base["F"], m[1], self.sw, jnp.sqrt(rho),
               eps * self.norm, self.rmsJ0)
        n = self.model.dim
        return ((lambda v: _jv(kind, n, v, aux, data)), (lambda wr: _jtw(kind, n, wr, aux, data)))

    def phi(self, base):
        g = self.sw[:, None] * (base["J"] - self.J0) / self.rmsJ0
        return 0.5 * float(jnp.sum(g**2))


def plus(s, delta):
    """max(0, s) (delta = 0) or (s + sqrt(s^2 + delta^2)) / 2, and its derivative."""
    if delta == 0:
        return jnp.maximum(0.0, s), (s > 0).astype(s.dtype)
    q = jnp.sqrt(s * s + delta * delta)
    return 0.5 * (s + q), 0.5 * (1.0 + s / q)


@functools.partial(jax.jit, static_argnums=8)
def _residual(J, F, J0, sw, lam, rho, en, rmsJ0, delta):
    g = (sw[:, None] * (J - J0) / rmsJ0).ravel()
    c = jnp.sum(F**2, 1) / en**2 - 1.0
    m, dm = plus(c + lam / rho, delta)
    return jnp.concatenate([g, jnp.sqrt(rho) * sw * m]), m, dm, c


def _tangent_of(kind, v, J, B, data):
    if kind == "dense":
        n_u = data["A"].shape[0]
        return _dense_tangent_impl(v[:n_u], v[n_u:], J, B, data)
    d = data._tangent(v, J, B)
    return {"J": d["J"], "F": d["F"]}


def _lin_rows(kind, v, aux, data):
    J, B, F, dm, sw, srho, en, rmsJ0 = aux
    d = _tangent_of(kind, v, J, B, data)
    dg = (sw[:, None] * d["J"] / rmsJ0).ravel()
    dc = 2.0 * jnp.sum(F * d["F"], 1) / en**2
    return jnp.concatenate([dg, srho * sw * dm * dc])


@functools.partial(jax.jit, static_argnums=(0, 1))
def _jv(kind, n, v, aux, data):
    return _lin_rows(kind, v, aux, data)


@functools.partial(jax.jit, static_argnums=(0, 1))
def _jtw(kind, n, wr, aux, data):
    x = jax.ShapeDtypeStruct((n,), wr.dtype)
    return jax.linear_transpose(lambda v: _lin_rows(kind, v, aux, data), x)(wr)[0]


# ------------------------------------------------------------ physical metric
def metric(T, reg=1e-10):
    """L with L L^T = M = blockdiag(H_J, H_p) (+ reg x mean diag per block), and H_J, the
    exact Gram matrix of g (g = sw A (u - u0) / rms J0)."""
    w = np.asarray(T["w"])
    sw = np.sqrt(w)
    A = np.asarray(T["A"])
    n_u = A.shape[0]
    Aw = (sw[None, :, None] * A).reshape(n_u, -1) / float(T["rmsJ0"])
    HJ = Aw @ Aw.T
    Pw = sw[None, :] * np.asarray(T["P"]) / float(T["rmsp0"])
    Hp = Pw @ Pw.T
    blocks = []
    for H in (HJ, Hp):
        Hr = H + reg * np.mean(np.diag(H)) * np.eye(len(H))
        blocks.append(np.linalg.cholesky(Hr))
    n = n_u + len(Hp)
    L = np.zeros((n, n))
    L[:n_u, :n_u], L[n_u:, n_u:] = blocks
    return L, HJ


class Metric:
    """d = L^-T y and its transpose, with L lower triangular (dense)."""

    def __init__(self, L):
        self.L = jnp.asarray(L)

    def back(self, y):  # d = L^-T y
        return jax.scipy.linalg.solve_triangular(self.L, y, lower=True, trans="T")

    def back_T(self, d):  # L^-1 d (the transpose of back)
        return jax.scipy.linalg.solve_triangular(self.L, d, lower=True)


# ------------------------------------------------- matrix-free trust region
def steihaug_step(J, JT, r, met, Delta, tol=1e-2, maxiter=100, gn_extra=True):
    """Truncated CG (Steihaug-Toint) for min |r + J L^-T y|^2 / 2 s.t. |y| <= Delta.

    Returns y, |r + J d|^2 / 2 at the step (the model value), whether it hit the boundary,
    the CG iterations and the Gauss-Newton share estimate: CG is continued past the
    boundary (``gn_extra``) to the unconstrained minimum of the model, so the share
    1 - min model / f is known even for a boundary step."""
    f = 0.5 * float(r @ r)
    Bv = lambda v: met.back_T(JT(J(met.back(v))))  # noqa: E731
    g = met.back_T(JT(r))
    z = jnp.zeros_like(g)
    res = -g
    p = res
    rr = float(res @ res)
    r0 = np.sqrt(rr)
    step, hit, k_step, its = None, False, 0, 0
    for k in range(1, maxiter + 1):
        its = k
        Bp = Bv(p)
        kap = float(p @ Bp)
        if kap <= 0:
            break
        alpha = rr / kap
        z_new = z + alpha * p
        if step is None and float(jnp.linalg.norm(z_new)) >= Delta:
            # the boundary: tau >= 0 with |z + tau p| = Delta
            a, b, cc = float(p @ p), 2 * float(z @ p), float(z @ z) - Delta**2
            tau = (-b + np.sqrt(b * b - 4 * a * cc)) / (2 * a)
            step, hit, k_step = z + tau * p, True, k
            if not gn_extra:
                break
        z = z_new
        res = res - alpha * Bp
        rr_new = float(res @ res)
        if np.sqrt(rr_new) <= tol * r0:
            break
        p = res + (rr_new / rr) * p
        rr = rr_new
    if step is None:
        step, k_step = z, its
    model = lambda y: 0.5 * float(jnp.sum((r + J(met.back(y))) ** 2))  # noqa: E731
    m_step = model(step)
    m_gn = model(z) if hit else m_step
    share = max(0.0, 1.0 - m_gn / f) if f > 0 else 0.0
    return step, m_step, hit, k_step, its, share, float(jnp.linalg.norm(g))


def _tridiag_tr(alpha, beta, gnorm, Delta):
    """min_h gnorm e1.h + h.T h / 2 s.t. |h| <= Delta for the Lanczos tridiagonal T
    (alpha diagonal, beta off-diagonal): mu = 0 when the minimum fits, else the secular
    equation |h(mu)| = Delta by Newton (T is positive semidefinite here). Returns h, mu,
    hit and the mu = 0 solution."""
    k = len(alpha)
    Tm = np.diag(alpha) + np.diag(beta[: k - 1], 1) + np.diag(beta[: k - 1], -1)
    th, V = np.linalg.eigh(Tm)
    b = -gnorm * V[0]  # V^T (-gnorm e1)
    th0 = np.maximum(th, 1e-14 * max(th.max(), 1e-300))

    def h(mu):
        return V @ (b / (th0 + mu))

    h0 = h(0.0)
    if np.linalg.norm(h0) <= Delta:
        return h0, 0.0, False, h0
    lo, hi, mu = 0.0, np.inf, max(1e-14 * th.max(), abs(b).max() / Delta - th0.min())
    mu = max(mu, 0.0)
    for _ in range(100):
        hm = h(mu)
        nh = np.linalg.norm(hm)
        ph = 1.0 / Delta - 1.0 / nh  # Newton on 1/|h| (well behaved)
        if abs(nh - Delta) <= 1e-6 * Delta:
            break
        if nh > Delta:
            lo = mu
        else:
            hi = mu
        dn = -np.sum(b**2 / (th0 + mu) ** 3) / nh  # d|h|/dmu
        mu_new = mu - ph / (dn / nh**2)
        mu = mu_new if (lo < mu_new < hi) else (0.5 * (lo + hi) if np.isfinite(hi) else 2 * mu + 1e-300)
    return h(mu), mu, True, h0


def lanczos_tr_step(J, JT, r, met, Delta, tol=1e-2, maxiter=100):
    """Matrix-free trust-region step (generalized Lanczos, GLTR): the subproblem
    min |r + J L^-T y|^2 / 2 s.t. |y| <= Delta solved exactly in the Krylov space of
    B = L^-1 J^T J L^-T and g = L^-1 J^T r (Lanczos with full reorthogonalization). As the
    QR trust region, mu = 0 (the Gauss-Newton step) when it fits, else mu on the boundary,
    here restricted to the Krylov space. Stops when |(B + mu) y + g| <= tol |g| for the
    trust-region solution (estimated as beta_k |e_k . h|) and for the Gauss-Newton one
    (whose model value gives the GN share). Returns y, the model value at y, hit, the
    iterations, the GN share, |g| and mu."""
    f = 0.5 * float(r @ r)
    Bv = lambda v: met.back_T(JT(J(met.back(v))))  # noqa: E731
    g = met.back_T(JT(r))
    gnorm = float(jnp.linalg.norm(g))
    Q = [g / gnorm]
    alpha, beta = [], []
    q_prev, b_prev = jnp.zeros_like(g), 0.0
    for k in range(1, maxiter + 1):
        wv = Bv(Q[-1]) - b_prev * q_prev
        a_k = float(Q[-1] @ wv)
        wv = wv - a_k * Q[-1]
        Qm = jnp.stack(Q)
        wv = wv - Qm.T @ (Qm @ wv)  # full reorthogonalization
        b_k = float(jnp.linalg.norm(wv))
        alpha.append(a_k)
        beta.append(b_k)
        h, mu, hit, h0 = _tridiag_tr(np.array(alpha), np.array(beta), gnorm, Delta)
        res_tr, res_gn = b_k * abs(h[-1]), b_k * abs(h0[-1])
        if (res_tr <= tol * gnorm and res_gn <= tol * gnorm) or b_k <= 1e-14 * gnorm:
            break
        q_prev, b_prev = Q[-1], b_k
        Q.append(wv / b_k)
    Qm = jnp.stack(Q[: len(alpha)])
    y = Qm.T @ jnp.asarray(h)
    Tm = np.diag(alpha) + np.diag(beta[:-1], 1) + np.diag(beta[:-1], -1)
    model = lambda hh: f + gnorm * hh[0] + 0.5 * hh @ Tm @ hh  # noqa: E731
    m_step, m_gn = model(h), model(h0)
    share = max(0.0, 1.0 - m_gn / f) if f > 0 else 0.0
    return y, float(m_step), hit, len(alpha), share, gnorm, mu


# ------------------------------------------------------- exact-QR trust region
def _damped_qr(R, c, s):
    """y minimizing |R y + c|^2 + s^2 |y|^2 by the QR of [R; s I]."""
    n = R.shape[1]
    Rt = jnp.linalg.qr(jnp.vstack([R, s * jnp.eye(n)]), mode="r")
    Qtb = jax.scipy.linalg.solve_triangular(Rt.T, -(R.T @ c), lower=True)
    return jax.scipy.linalg.solve_triangular(Rt, Qtb, lower=False)


def qr_tr_step(R, c, Delta, mu_prev=None):
    """DESC's exact trust-region step from R, c (= Q^T r) of the stacked Jacobian in the
    metric variables: the undamped Gauss-Newton step when it fits (mu = 0), else mu on the
    boundary |y(mu)| = Delta by safeguarded Newton on the secular equation (Cholesky of
    R^T R + mu I), with a log bisection by damped QR below 1e-10 lambda_max (as the
    lm-trust branch's ``_tr_step``). Returns y, mu, whether the boundary was hit."""
    n = R.shape[0]
    H = R.T @ R
    gs = R.T @ c
    v = jnp.ones(n) / np.sqrt(n)  # lambda_max of H by power iteration (no dense eigensolver)
    for _ in range(40):
        v = H @ v
        v = v / jnp.linalg.norm(v)
    lam = float(v @ (H @ v))
    dr = jnp.diag(R)
    dri = jnp.where(jnp.abs(dr) <= 1e-14 * jnp.max(jnp.abs(dr)), 0.0, 1.0 / jnp.where(dr == 0, 1.0, dr))
    yN = -dri * jax.scipy.linalg.solve_triangular(R * dri[:, None], c, lower=False,
                                                   unit_diagonal=True)
    nN = float(jnp.linalg.norm(yN))
    if np.isfinite(nN) and nN <= Delta:
        return yN, 0.0, False
    ii = jnp.arange(n)

    def solve(m):
        L = jnp.linalg.cholesky(H.at[ii, ii].add(m))
        return L, -jax.scipy.linalg.cho_solve((L, True), gs)

    mu_chol = 1e-10 * lam
    L, y = solve(mu_chol)
    ny = float(jnp.linalg.norm(y))
    if np.isfinite(ny) and ny <= Delta:  # root in (0, mu_chol): log bisection by QR
        lo, hi, yh = 1e-12 * mu_chol, mu_chol, y
        for _ in range(12):
            mid = np.sqrt(lo * hi)
            ym = _damped_qr(R, c, np.sqrt(mid))
            nm = float(jnp.linalg.norm(ym))
            if np.isfinite(nm) and nm <= Delta:
                hi, yh = mid, ym
                if nm >= 0.95 * Delta:
                    break
            else:
                lo = mid
        return yh, hi, True
    mu = max(mu_prev or mu_chol, mu_chol)
    lo, hi = mu_chol, np.inf
    for _ in range(30):
        L, y = solve(mu)
        ny = float(jnp.linalg.norm(y))
        if not np.isfinite(ny):
            lo = mu
            mu = np.sqrt(mu * hi) if np.isfinite(hi) else 10 * mu
            continue
        ph = ny - Delta
        if abs(ph) <= 0.05 * Delta:
            break
        lo, hi = (mu, hi) if ph > 0 else (lo, mu)
        q = jax.scipy.linalg.solve_triangular(L, y, lower=True)
        mu_new = mu + (ny / float(jnp.linalg.norm(q))) ** 2 * ph / Delta
        mu = mu_new if (np.isfinite(mu_new) and lo < mu_new < hi) else (
            np.sqrt(lo * hi) if np.isfinite(hi) else 10 * mu)
    return y, mu, True


def stacked_rows(T, base, x, x0, m, rho, eps, norm, RgLt, met_Lt, n_u, rows=None):
    """The stacked Jacobian rows [J_r L^-T | r] of the dense model at x, compressed:

        g rows      R_g L_u^-T (n_u rows), right-hand side R_g (u - u0)   (|g|^2 exactly)
        c rows      sqrt(rho) sw_i 2 F_i . dF_i/dx / (eps norm)^2 L^-T, rhs sqrt(rho) sw_i m_i

    for the active points (m_i > 0), or for the subset ``rows`` of the stacked row index
    (one process's block). ``RgLt``: (R_g L_u^-T, R_g); ``met_Lt``: L^-1 (dense), so that
    C L^-T = (L^-1 C^T)^T."""
    Rg_y, Rg = RgLt
    m, dm = m
    act = np.flatnonzero(np.asarray(dm) > 1e-14)
    n_rows = n_u + len(act)
    sel = np.arange(n_rows) if rows is None else rows
    out = []
    gsel = sel[sel < n_u]
    if len(gsel):
        rhs = Rg[gsel] @ (x[:n_u] - x0[:n_u])
        blk = jnp.concatenate([Rg_y[gsel], jnp.zeros((len(gsel), met_Lt.shape[0] - n_u)),
                               rhs[:, None]], 1)
        out.append(blk)
    csel = act[sel[sel >= n_u] - n_u]
    if len(csel):
        out.append(_c_rows(T, base["J"], base["B"], base["F"], m, jnp.asarray(csel), rho,
                           eps * norm, met_Lt, dm))
    return jnp.concatenate(out, 0) if out else jnp.zeros((0, met_Lt.shape[0] + 1))


@jax.jit
def _c_rows(T, J, B, F, m, idx, rho, en, Linv, dm=None):
    Ji, Bi, Fi = J[idx], B[idx], F[idx]
    sw = jnp.sqrt(T["w"][idx])
    s = jnp.sqrt(rho) * sw * 2.0 / en**2 * (1.0 if dm is None else dm[idx])
    # F . (A_k x B) = A_k . (B x F);  F . (J x G_k) = G_k . (F x J);  F . (-gP_j)
    Cu = (jnp.einsum("kic,ic->ik", T["A"][:, idx], jnp.cross(Bi, Fi))
          + jnp.einsum("kic,ic->ik", T["G"][:, idx], jnp.cross(Fi, Ji)))
    Cc = -jnp.einsum("jic,ic->ij", T["gP"][:, idx], Fi)
    C = s[:, None] * jnp.concatenate([Cu, Cc], 1)
    CLt = (Linv @ C.T).T
    return jnp.concatenate([CLt, (jnp.sqrt(rho) * sw * m[idx])[:, None]], 1)


# ------------------------------------------------------------------ Hessian
def constraint_curvature(T, x, eps, norm, weights, chunk=2048):
    """For the dense model at x, with per-point weights mu_i (zero where not wanted):

        Gc = sum_i mu_i grad c_i grad c_i^T
        Wc = sum_i mu_i hess c_i,   hess c_i = (2 / (eps norm)^2) (D_i^T D_i + F_i . d^2 F_i)

    D_i = dF_i/dx (3 x n): d F_i / du_k = A_k,i x B_i + J_i x G_k,i, d F_i / dc_j = -gP_j,i;
    d^2 F_i / du_k du_l = A_k,i x G_l,i + A_l,i x G_k,i (only the u-u block)."""
    Td = {k: jnp.asarray(v) for k, v in T.items() if k in ("A", "J_I", "G", "B0", "gP", "P")}
    n_u, n_p = Td["A"].shape[0], Td["gP"].shape[0]
    n = n_u + n_p
    b = _dense_base(jnp.asarray(x), Td, n_u)
    en2 = (eps * norm) ** 2
    mu = np.asarray(weights)
    idx_all = np.flatnonzero(mu != 0)
    Gc = jnp.zeros((n, n))
    Wc = jnp.zeros((n, n))
    for i0 in range(0, len(idx_all), chunk):
        idx = jnp.asarray(idx_all[i0 : i0 + chunk])
        Fi, Ji, Bi = b["F"][idx], b["J"][idx], b["B"][idx]
        mi = jnp.asarray(mu)[idx]
        A, G, gP = Td["A"][:, idx], Td["G"][:, idx], Td["gP"][:, idx]
        Du = jnp.cross(A, Bi[None]) + jnp.cross(Ji[None], G)  # (n_u, chunk, 3)
        D = jnp.concatenate([Du, -gP], 0)  # (n, chunk, 3)
        gc = 2.0 / en2 * jnp.einsum("kic,ic->ik", D, Fi)  # grad c_i, (chunk, n)
        Gc = Gc + (gc * mi[:, None]).T @ gc
        Dw = D * jnp.sqrt(jnp.abs(mi))[None, :, None]
        Dm = Dw.reshape(n, -1)
        sgn = jnp.repeat(jnp.sign(mi), 3)
        Wc = Wc + 2.0 / en2 * (Dm * sgn[None]) @ Dm.T
        K = jnp.einsum("kic,lic->kl", A * mi[None, :, None], jnp.cross(G, Fi[None]))
        Wc = Wc.at[:n_u, :n_u].add(2.0 / en2 * (K + K.T))
    return np.asarray(Gc), np.asarray(Wc)


def hessians(T, x, lam, rho, eps, norm, HJ, delta=1e-6, lam_tol=1e-6, c_tol=1e-3):
    """Exact Hessians at x: of the AL the inner solver minimizes (with the smoothed plus
    function m of ``delta``) and of the Lagrangian Phi + sum_i lam_i w_i c_i.

        H_AL  = H_J + sum_i rho w_i [(m'^2 + m m'') grad c_i grad c_i^T + m m' hess c_i]
        GN_AL = H_J + sum_i rho w_i m'^2 grad c_i grad c_i^T        (its Gauss-Newton part)
        H_Lag = H_J + sum_i w_i lam_i hess c_i

    The active set (for the tangent space) is the points on the bound (c_i > -c_tol) with
    lam_i > lam_tol max lam (smoothing leaves a tiny positive lam everywhere):
    ``grad_active`` = sum over them of w_i grad c_i grad c_i^T."""
    Td = {k: jnp.asarray(v) for k, v in T.items() if k in ("A", "J_I", "G", "B0", "gP", "P")}
    n_u = T["A"].shape[0]
    b = _dense_base(jnp.asarray(x), Td, n_u)
    w = np.asarray(T["w"])
    lam = np.asarray(lam)
    c = np.asarray(jnp.sum(b["F"] ** 2, 1)) / (eps * norm) ** 2 - 1.0
    s = c + lam / rho
    q = np.sqrt(s * s + delta * delta)
    m, m1, m2 = 0.5 * (s + q), 0.5 * (1 + s / q), 0.5 * delta**2 / q**3
    Gs, Ws = constraint_curvature(T, x, eps, norm, rho * w * (m1**2 + m * m2))
    Gn, _ = constraint_curvature(T, x, eps, norm, rho * w * m1**2)
    _, Wm = constraint_curvature(T, x, eps, norm, rho * w * m * m1)
    act = ((lam > lam_tol * lam.max()) & (c > -c_tol)).astype(float) if lam.max() > 0 \
        else np.zeros_like(lam)
    Ga, _ = constraint_curvature(T, x, eps, norm, w * act)
    _, Wl = constraint_curvature(T, x, eps, norm, w * lam)
    n = Ga.shape[0]
    HJn = np.zeros((n, n))
    HJn[:n_u, :n_u] = HJ
    return {"H_AL": HJn + Gs + Wm, "GN_AL": HJn + Gn, "H_Lag": HJn + Wl,
            "grad_active": Ga, "c": c, "m": m, "active": act}


# ----------------------------------------------------------------- AL driver
class ALSolver:
    """Augmented-Lagrangian continuation over alpha with a trust-region inner solver.

    ``step``: "steihaug" (matrix-free) or "qr" (dense, needs ``qr_fn`` and the tables).
    ``qr_fn(rows_fn, n_rows, n_cols) -> R_aug`` returns the (n + 1) x (n + 1) triangular
    factor of the stacked matrix whose rows ``rows_fn(sel)`` gives (the distributed QR
    builds each process's block)."""

    def __init__(self, prob, met, x0, step="steihaug", T=None, qr_fn=None, Rg=None, log=print,
                 cg_tol=1e-2, cg_maxiter=100):
        self.prob, self.met, self.x0 = prob, met, jnp.asarray(x0)
        self.step, self.T, self.qr_fn, self.log = step, T, qr_fn, log
        self.cg_tol, self.cg_maxiter = cg_tol, cg_maxiter
        self.Delta, self.mu = None, None
        if step == "qr":
            n_u = prob.model.n_u
            Lu = np.asarray(met.L)[:n_u, :n_u]
            self.Rg = np.asarray(Rg)
            Rg_y = np.linalg.solve(Lu, self.Rg.T).T  # R_g L_u^-T
            self.RgLt = (jnp.asarray(Rg_y), jnp.asarray(self.Rg))
            self.Linv = jnp.asarray(np.linalg.inv(np.asarray(met.L)))
            self.Td = {k: jnp.asarray(v) for k, v in T.items()
                       if k in ("A", "G", "gP", "w")}

    def inner(self, x, lam, rho, eps, maxiter=40, ftol=1e-12, gn_tol=1e-8, gtol=1e-3, hist=None,
              delta=0.0):
        """Minimize the AL at fixed (lam, rho) by trust-region steps; history per step."""
        prob, met = self.prob, self.met
        r, m, c, base = prob.residual(x, lam, rho, eps, delta)
        f = 0.5 * float(r @ r)
        hist = [] if hist is None else hist
        g_first = None
        if self.Delta is None:  # first radius: 0.1 x |L^T (x0 scale)|, refined by the rule
            self.Delta = 1e-2 * float(jnp.linalg.norm(met.L.T @ x)) or 1.0
        for it in range(maxiter):
            t0 = time.time()
            if self.step == "gltr":
                J, JT = prob.linear(base, m, lam, rho, eps)
                y, m_step, hit, k_cg, share, gn, mu_k = lanczos_tr_step(
                    J, JT, r, met, self.Delta, self.cg_tol, self.cg_maxiter)
                mu = None
                extra = {"lanczos": k_cg, "mu": mu_k}
            elif self.step == "steihaug":
                J, JT = prob.linear(base, m, lam, rho, eps)
                y, m_step, hit, k_cg, k_all, share, gn = steihaug_step(
                    J, JT, r, met, self.Delta, self.cg_tol, self.cg_maxiter)
                mu = None
                extra = {"cg": k_cg, "cg_gn": k_all}
            else:
                n_u = prob.model.n_u
                act = np.flatnonzero(np.asarray(m[1]) > 1e-14)
                n_rows = n_u + len(act)

                def rows_fn(sel, base=base, m=m):
                    return stacked_rows(self.Td, base, x, self.x0, m, rho, eps, prob.norm,
                                        self.RgLt, self.Linv, n_u, rows=sel)

                Ra = self.qr_fn(rows_fn, n_rows, prob.model.dim + 1)
                n = prob.model.dim
                R, cc, rp = Ra[:n, :n], Ra[:n, n], float(Ra[n, n])
                f_lin = 0.5 * (float(cc @ cc) + rp**2)
                share = float(cc @ cc) / (float(cc @ cc) + rp**2)
                gn = float(jnp.linalg.norm(R.T @ cc))
                y, mu, hit = qr_tr_step(R, cc, self.Delta, self.mu)
                rl = R @ y + cc
                m_step = f - f_lin + 0.5 * (float(rl @ rl) + rp**2)
                extra = {"rows": n_rows, "mu": mu, "f_lin/f": f_lin / f}
            d = met.back(y)
            ny = float(jnp.linalg.norm(y))
            r_new, m_new, c_new, base_new = prob.residual(x + d, lam, rho, eps, delta)
            f_new = 0.5 * float(r_new @ r_new)
            pred = f - m_step
            ratio = (f - f_new) / pred if pred > 0 else 0.0
            if not np.isfinite(ratio) or ratio < 0.25:
                self.Delta = 0.25 * ny
            elif ratio > 0.75 and hit:
                self.Delta = max(2.0 * ny, self.Delta)
            accepted = f_new < f and np.isfinite(f_new)
            g_first = gn if g_first is None else g_first
            h = {"f": f, "f_new": f_new, "ratio": ratio, "|grad|": gn, "delta": delta, "Delta": self.Delta, "|y|": ny,
                 "hit": hit, "GN share": share, "accepted": accepted,
                 "max c": float(jnp.max(c_new if accepted else c)), "time": time.time() - t0,
                 **extra}
            hist.append(h)
            self.log("    it {:3d} f {:.6e} -> {:.6e} ratio {:+.2f} |y| {:.2e} Delta {:.2e} "
                     "|g| {:.1e} GN {:.2e}{} {:.1f}s".format(it, f, f_new, ratio, ny, self.Delta, gn, share,
                                                  " hit" if hit else "", h["time"]))
            if accepted:
                dec = (f - f_new) / f
                x, r, m, c, base, f = x + d, r_new, m_new, c_new, base_new, f_new
                if mu is not None:
                    self.mu = mu
                if dec < ftol or share < gn_tol or gn < gtol * g_first:
                    break
            elif self.Delta < 1e-14 * (1 + float(jnp.linalg.norm(met.L.T @ x))):
                break
        return x, hist

    def solve_alpha(self, x, eps, lam, rho, max_outer=25, viol_tol=2e-2, maxiter=40,
                    rho_max=1e12, delta0=0.0, delta_min=1e-6):
        """AL outer loop at one bound eps: returns x, lam, rho and the outer history.
        Smoothing: delta_k = max(delta0 10^-k, delta_min) at outer iteration k (0: none);
        the multipliers are lam <- rho m(c + lam / rho), i.e. max(0, lam + rho c) at delta 0."""
        prob = self.prob
        outer = []
        viol_prev = np.inf
        for k in range(max_outer):
            delta = max(delta0 * 0.1**k, delta_min) if delta0 > 0 else 0.0
            x, hist = self.inner(x, lam, rho, eps, maxiter=maxiter, delta=delta)
            base = prob.model.base(x)
            c = prob.constraints(base, eps)
            viol = float(jnp.max(jnp.maximum(c, 0.0)))
            lam = rho * plus(c + lam / rho, delta)[0]
            comp = float(jnp.sum(prob.w * lam * jnp.abs(c)))
            Fn = jnp.linalg.norm(base["F"], axis=1) / prob.norm
            o = {"outer": k, "rho": rho, "delta": delta, "max|F|/eps": float(jnp.max(Fn)) / eps,
                 "viol": viol, "active": float(jnp.mean(c > -1e-2)), "phi": prob.phi(base),
                 "sum w lam": float(jnp.sum(prob.w * lam)), "complementarity": comp,
                 "inner": hist}
            outer.append(o)
            self.log("  outer {:2d} rho {:.1e}: max|F|/eps {:.4f} viol {:.2e} active {:.3f} "
                     "phi {:.6e} sum w lam {:.3e} ({} inner)".format(
                         k, rho, o["max|F|/eps"], viol, o["active"], o["phi"],
                         o["sum w lam"], len(hist)))
            if viol <= viol_tol and delta <= delta_min * 1.0001:
                break
            if viol > 0.25 * viol_prev and rho < rho_max:
                rho *= 10.0
            viol_prev = viol
        return x, lam, rho, outer


# -------------------------------------------------------------- diagnostics
RHO_BINS = (0.0, 0.5, 0.8, 0.9, 0.95, 1.0)


def state_metrics(T, base, x, lam, eps, base0=None):
    """Scalars of a state (NumPy): the force bound, distance to the start, B, p, the
    multipliers and the front's slope -(2 / eps) sum_i w_i lam_i (1 + c_i)."""
    w = np.asarray(T["w"])
    avg = lambda q, msk=1.0: float(np.sum(w * q * msk) / np.sum(w * msk))  # noqa: E731
    J, B, F, p, gp = (np.asarray(base[k]) for k in ("J", "B", "F", "p", "grad_p"))
    norm, rmsJ0, rmsp0 = float(T["norm"]), float(T["rmsJ0"]), float(T["rmsp0"])
    Fn = np.linalg.norm(F, axis=1) / norm
    c = Fn**2 / eps**2 - 1.0
    lam = np.asarray(lam)
    J0 = np.asarray(T["J0"])
    B0 = np.asarray(base0["B"]) if base0 is not None else B
    p0 = np.asarray(base0["p"]) if base0 is not None else p
    b0 = B0 / np.linalg.norm(B0, axis=1, keepdims=True)
    dJ = J - J0
    dpar = np.sum(dJ * b0, 1)
    B2 = np.sum(B**2, 1)
    gr = np.asarray(T["grad_rho"])
    Bgr = np.abs(np.sum(B * gr, 1)) / np.sqrt(B2 * np.sum(gr**2, 1))
    rho = np.asarray(T["rtz"])[:, 0]
    Bgp = np.abs(np.sum(B * gp, 1))
    act = c > -2e-2
    out = {
        "eps": float(eps),
        "max|F|": float(Fn.max()), "p99|F|": float(np.percentile(Fn, 99)),
        "rms|F|": float(np.sqrt(avg(Fn**2))), "<|F|>": avg(Fn),
        "max|F|/eps": float(Fn.max() / eps),
        "|J-J0|": float(np.sqrt(avg(np.sum(dJ**2, 1)))) / rmsJ0,
        "J-J0 along B": float(np.sqrt(avg(dpar**2))) / rmsJ0,
        "J-J0 across B": float(np.sqrt(max(avg(np.sum(dJ**2, 1) - dpar**2), 0.0))) / rmsJ0,
        "B-B0": float(np.sqrt(avg(np.sum((B - B0) ** 2, 1)) / avg(B0 ** 2 @ np.ones(3)))),
        "p-p0": float(np.sqrt(avg((p - p0) ** 2))) / rmsp0,
        "p-p_desc": float(np.sqrt(avg((p - np.asarray(T["p_desc"])) ** 2))) / rmsp0,
        "<p>/<p0>": avg(p) / avg(p0) if avg(p0) else np.nan,
        "min p/max p0": float(p.min() / np.max(p0)),
        "<|B.grad p|>/<|B||grad p|>": avg(Bgp) / avg(np.sqrt(B2 * np.sum(gp**2, 1))),
        "max |B.grad p|/(|B| eps norm)": float(np.max(Bgp / np.sqrt(B2)) / (eps * norm)),
        "B.grad rho": avg(Bgr), "B.grad rho, rho>0.8": avg(Bgr, rho > 0.8),
        "active fraction": float(np.mean(act)), "active volume": avg(act.astype(float)),
        "sum w lam": float(np.sum(w * lam)), "max lam": float(lam.max()),
        "slope dPhi/deps": float(-2.0 / eps * np.sum(w * lam * (1.0 + c))),
        "Phi": 0.5 * float(np.sum(w * np.sum(dJ**2, 1))) / rmsJ0**2,
        "active by rho": {f"{a:g}-{b:g}": avg(act.astype(float), (rho > a) & (rho <= b))
                          for a, b in zip(RHO_BINS[:-1], RHO_BINS[1:])},
        "max|F| by rho": {f"{a:g}-{b:g}": float(Fn[(rho > a) & (rho <= b)].max())
                          for a, b in zip(RHO_BINS[:-1], RHO_BINS[1:])},
    }
    return out, {"Fn": Fn, "c": c, "p": p, "dJ": np.linalg.norm(dJ, axis=1) / rmsJ0,
                 "dJpar": dpar / rmsJ0, "dp": (p - p0) / rmsp0}
