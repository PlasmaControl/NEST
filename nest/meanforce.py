"""Closest free-boundary state to J_fit under a bound on DESC's mean force, with the
pressure eliminated by the weak Poisson equation (docs/mean_force_poisson_plan.md).

Unknowns z = (u, xi): the current potential's coefficients and the boundary displacement
[m] of a ``FreeBoundaryFields`` anchor. With w_i = dV_i / V on the moved geometry and
sw = sqrt(w):

    a      = sw (J x B)                                   (3 n)
    P      = sw grad phi_j  (the pressure space, j kept)   (3 n x k),  P = Q R
    p*     = R^-1 Q^T a          (weighted least squares: the weak Poisson equation)
    f      = (I - Q Q^T) a = sw F,   F = J x B - grad p*
    M      = sum_i w_i |F_i| / norm                        (DESC's <|F|>_vol / norm)
    g      = sw (J - J_fit) / rms J_fit,   J_fit = J(u0) carried with the coordinates
    rel_j  = B.n / |B| on the boundary grid,  Bn_rms = sqrt(mean rel^2)  (DESC's plain mean)

    min |g|^2 / 2   s.t.   c_F = M / eps - 1 <= 0,  c_B = (Bn_rms / beta)^2 - 1 <= 0,
                           c_P = Psi / Psi_t - 1 = 0

by an augmented Lagrangian (multipliers nu, one penalty rho):

    L_A = |g|^2 / 2 + psi(c_F, nu_F) + psi(c_B, nu_B) + nu_P c_P + rho c_P^2 / 2,
    psi(c, nu) = (rho / 2) (max(0, c + nu / rho)^2 - (nu / rho)^2).

Inner problem: trust-region steps on L_A from a sum-of-squares model at the iterate x_k,
whose rows (``rows``) are

    g                                             (Gauss-Newton, exact)
    s_i z_i,  z_i = w_i F_i,  s_i^2 = psi'_F / (eps norm t_i),  t_i = max(|z_i|, w_i tau)
                                                  (the majorizer |z| <= (|z|^2 + t^2) / 2t of M:
                                                  every surrogate decrease is a true decrease,
                                                  and its gradient is psi'_F grad c_F exactly)
    s_B rel_j,  s_B^2 = psi'_B / (beta^2 n_b)      (c_B + 1 is a sum of squares)
    sqrt(rho) (c_P + nu_P / rho)                  (the equality, exact)
    sqrt(psi''_F) grad c_F . d,  sqrt(psi''_B) grad c_B . d   (the penalty's own curvature)

psi' = max(0, nu + rho c) and psi'' = rho [c + nu / rho > 0] are re-evaluated at every
accepted step (they are the AL term's derivatives there), the multipliers nu are held
within an inner solve. Every ratio and acceptance uses the plain L_A (plain M).

The linearization of f includes the projector's derivative in xi (variable projection,
Golub-Pereyra): with t = da - dP p*,

    df = (I - Q Q^T) t - Q R^-T (dP^T f)

where dP p* and dP^T f are derivatives of the pressure field alone (no field solve).
Steps: ``pointwise.lanczos_tr_step`` (GLTR, matrix-free: one annulus solve per product)
or ``pointwise.qr_tr_step`` on the QR of the stacked Jacobian in the metric variables
(u columns from this anchor's tables, xi columns by Jacobian-vector products).
"""

import json
import time

import jax
import jax.numpy as jnp
import numpy as np

from free_boundary import _with_geo
from objective import _cartesian
import pointwise as pw


# ------------------------------------------------------------- local pieces
def _sw(fields, xi):
    """sqrt(dV / V) at the collocation points on the geometry moved by xi."""
    gc, _ = fields.mg_col.at(xi)
    dV = jnp.abs(gc.sqrt_g).ravel() * fields.w_vol.ravel()
    return jnp.sqrt(dV / jnp.sum(dV))


def _gradp(fields, c, xi):
    """grad p (Cartesian, (n, 3)) of the pressure coefficients c on the moved geometry."""
    gc, _ = fields.mg_col.at(xi)
    _, gp = _with_geo(fields.p_col, gc)(c)
    return _cartesian(gp.reshape(-1, 3), fields.phi)


def _Jloc(fields, u, xi):
    """J (Cartesian) at the collocation points, exact in xi (no field solve)."""
    gc, _ = fields.mg_col.at(xi)
    return _cartesian(_with_geo(fields.J_col, gc)(u, fields.I).reshape(-1, 3), fields.phi)


@jax.jit
def _pcols_chunk(fields, E, xi, sw):
    return jax.vmap(lambda c: sw[:, None] * _gradp(fields, c, xi))(E)


@jax.jit
def _A_chunk(fields, E, xi):
    """dJ/du_k (Cartesian) for the unit vectors E (J is affine in u)."""
    z = jnp.zeros(E.shape[1])
    return jax.vmap(lambda e: jax.jvp(lambda u: _Jloc(fields, u, xi), (z,), (e,))[1])(E)


def _PTf(fields, xi, f, n_p):
    """P(xi)^T f (n_p): the adjoint of c -> sw grad p(c) at f."""
    lin = lambda c: (_sw(fields, xi)[:, None] * _gradp(fields, c, xi)).ravel()  # noqa: E731
    return jax.vjp(lin, jnp.zeros(n_p))[1](f)[0]


@jax.jit
def _fb_base(fields, xf):
    base, f = fields._base(xf)
    keep = {k: f[k] for k in ("J", "B", "Bn", "B_bnd", "Psi")}
    return base, keep


@jax.jit
def _reduce(P, a):
    """Thin QR of P, p* (in P's columns), f = (I - Q Q^T) a."""
    Q, R = jnp.linalg.qr(P)
    qa = Q.T @ a
    c = jax.scipy.linalg.solve_triangular(R, qa, lower=False)
    return Q, R, c, a - Q @ qa


@jax.jit
def _scalars(fields, u0, xi, sw, J, B, f, Bn, B_bnd, Psi, rmsJ, norm):
    """Everything of a state that the merit and the rows need, from the fields and f."""
    n = J.shape[0]
    F = f.reshape(n, 3) / sw[:, None]
    w = sw**2
    z = w[:, None] * F
    absF = jnp.linalg.norm(F, axis=1)
    M = jnp.sum(w * absF) / norm
    rmsF = jnp.sqrt(jnp.sum(w * absF**2)) / norm
    Jfit = _Jloc(fields, u0, xi)
    g = (sw[:, None] * (J - Jfit) / rmsJ).ravel()
    rel = Bn / jnp.linalg.norm(B_bnd, axis=1)
    return {"F": F, "z": z, "absF": absF, "M": M, "rmsF": rmsF, "g": g, "Jfit": Jfit,
            "rel": rel, "Bn_rms": jnp.sqrt(jnp.mean(rel**2)), "Psi": Psi[0]}


# ------------------------------------------------------------ the anchor
class Anchor:
    """One anchor geometry: the reduced Poisson pressure, the AL rows and their
    linearization on a ``FreeBoundaryFields`` (x = (u, c_p, xi); c_p is not used,
    the pressure is p*(u, xi)).

    ``u0``: the fit (J_fit = J(u0), carried with the coordinates); ``rmsJ``: rms J_fit
    (held across anchors); ``norm``: DESC's <|grad |B|^2 / 2 mu0|>_vol; ``chunk``:
    unit vectors per batch of the column builds."""

    def __init__(self, fields, u0, rmsJ, norm, chunk=64, rank_tol=1e-12, log=print,
                 tan_chunk=None):
        self.fields, self.log = fields, log
        self.n_u, self.n_p, self.n_xi = fields.n_u, fields.n_p, fields.n_xi
        self.n = self.n_u + self.n_xi
        self.u0 = jnp.asarray(u0)
        self.rmsJ, self.norm, self.chunk = float(rmsJ), float(norm), int(chunk)
        # vectors per batch of field-solve products (tables, xi columns): the annulus'
        # memory per product grows with its size, the local columns' does not
        self.tan_chunk = int(tan_chunk or chunk)
        self.n_pts = len(fields.phi)
        self.n_b = fields.n_bnd
        # the pressure columns kept: rank check at xi = 0
        xi0 = jnp.zeros(self.n_xi)
        P = self._pcols(xi0, _sw(fields, xi0), np.arange(self.n_p))
        R = jnp.linalg.qr(P, mode="r")
        d = np.abs(np.asarray(jnp.diag(R)))
        self.keep = np.flatnonzero(d > rank_tol * d.max())
        self.keep_j = jnp.asarray(self.keep)
        del P, R
        if len(self.keep) < self.n_p:
            log(f"  pressure columns: {self.n_p - len(self.keep)} of {self.n_p} dropped "
                f"(|R_jj| < {rank_tol:g} max)")
        self.tables = None

    # ---------------------------------------------------------- columns
    def _pcols(self, xi, sw, keep):
        E = np.eye(self.n_p)
        out = []
        for i in range(0, len(keep), self.chunk):
            out.append(_pcols_chunk(self.fields, jnp.asarray(E[keep[i : i + self.chunk]]), xi, sw)
                       .reshape(min(self.chunk, len(keep) - i), -1))
        return jnp.concatenate(out, 0).T  # (3 n, k)

    def A_table(self, xi=None):
        """dJ/du (n_u, n, 3) on this anchor (local)."""
        xi = jnp.zeros(self.n_xi) if xi is None else xi
        E = np.eye(self.n_u)
        return jnp.concatenate([_A_chunk(self.fields, jnp.asarray(E[i : i + self.chunk]), xi)
                                for i in range(0, self.n_u, self.chunk)], 0)

    def build_tables(self, x_u):
        """u columns of J, B, B.n, B on the boundary and Psi at xi = 0 (n_u field
        solves, in batches): B is linear in u at fixed geometry."""
        t0 = time.time()
        xf = jnp.concatenate([jnp.asarray(x_u), jnp.zeros(self.n_p + self.n_xi)])
        base, _ = _fb_base(self.fields, xf)
        E = np.eye(self.n_u)
        keys = ("J", "B", "Bn", "B_bnd", "Psi")
        parts = {k: [] for k in keys}
        for i in range(0, self.n_u, self.tan_chunk):
            e = np.zeros((min(self.tan_chunk, self.n_u - i), self.fields.n))
            e[:, : self.n_u] = E[i : i + self.tan_chunk]
            d = _tan_batch(self.fields, jnp.asarray(e), base)
            for k in keys:
                parts[k].append(d[k])
        self.tables = {k: jnp.concatenate(v, 0) for k, v in parts.items()}
        self.log(f"  tables: {self.n_u} field solves in {time.time() - t0:.0f} s")
        return time.time() - t0

    # ------------------------------------------------------------ state
    def state(self, z):
        """The fields, p*, f and the scalars of z = (u, xi)."""
        u, xi = z[: self.n_u], z[self.n_u :]
        xf = jnp.concatenate([u, jnp.zeros(self.n_p), xi])
        base, ff = _fb_base(self.fields, xf)
        sw = _sw_j(self.fields, xi)
        P = self._pcols(xi, sw, self.keep)
        a = (sw[:, None] * jnp.cross(ff["J"], ff["B"])).ravel()
        Q, R, c, f = _reduce(P, a)
        gal = jnp.max(jnp.abs(P.T @ f) / (jnp.linalg.norm(P, axis=0) * jnp.linalg.norm(f)))
        del P
        s = _scalars(self.fields, self.u0, xi, sw, ff["J"], ff["B"], f, ff["Bn"], ff["B_bnd"],
                     ff["Psi"], self.rmsJ, self.norm)
        cfull = jnp.zeros(self.n_p).at[self.keep_j].set(c)
        s.update({"z_vec": z, "xi": xi, "base": base, "sw": sw, "J": ff["J"], "B": ff["B"],
                  "Bn": ff["Bn"], "B_bnd": ff["B_bnd"], "Q": Q, "R": R, "c": cfull, "f": f,
                  "galerkin": gal})
        return s

    def galerkin(self, s):
        """max_j |P_j^T f| / (|P_j| |f|) (zero for the exact Poisson pressure)."""
        return float(s["galerkin"])

    # ------------------------------------------------------ linearization
    def raw_lin(self, s):
        """v -> (dg, dz, drel, dPsi) at the state s: linear (jitted, transposable)."""
        aux = _aux(s, self)
        return lambda v: _raw_lin(self.fields, v, aux, self.n_u, self.n_p)

    def raw_lin_T(self, s):
        aux = _aux(s, self)
        n = self.n
        return lambda w: _raw_lin_T(self.fields, w, aux, self.n_u, self.n_p, n)


def _aux(s, A):
    return {"base": s["base"], "xi": s["xi"], "sw": s["sw"], "J": s["J"], "B": s["B"],
            "Bn": s["Bn"], "B_bnd": s["B_bnd"], "Q": s["Q"], "R": s["R"], "c": s["c"],
            "f": s["f"], "g": s["g"], "Jfit": s["Jfit"], "u0": A.u0, "keep": A.keep_j,
            "rmsJ": A.rmsJ}


_sw_j = jax.jit(_sw)


@jax.jit
def _tan_batch(fields, V, base):
    d = jax.vmap(lambda v: fields._tangent(v, base))(V)
    return {k: d[k] for k in ("J", "B", "Bn", "B_bnd", "Psi")}


def _raw_lin_impl(fields, v, aux, n_u, n_p):
    du, dxi = v[:n_u], v[n_u:]
    xi, sw, J, B = aux["xi"], aux["sw"], aux["J"], aux["B"]
    Q, R, c, f, keep = aux["Q"], aux["R"], aux["c"], aux["f"], aux["keep"]
    n = J.shape[0]
    vf = jnp.concatenate([du, jnp.zeros(n_p, v.dtype), dxi])
    d = fields._tangent(vf, aux["base"])
    dsw = jax.jvp(lambda x: _sw(fields, x), (xi,), (dxi,))[1]
    da = (dsw[:, None] * jnp.cross(J, B)
          + sw[:, None] * (jnp.cross(d["J"], B) + jnp.cross(J, d["B"]))).ravel()
    dPp = jax.jvp(lambda x: (_sw(fields, x)[:, None] * _gradp(fields, c, x)).ravel(),
                  (xi,), (dxi,))[1]
    dPTf = jax.jvp(lambda x: _PTf(fields, x, f, n_p), (xi,), (dxi,))[1][keep]
    t = da - dPp
    df = t - Q @ (Q.T @ t) - Q @ jax.scipy.linalg.solve_triangular(R, dPTf, lower=False,
                                                                     trans="T")
    fn = f.reshape(n, 3)
    dz = (dsw[:, None] * fn + sw[:, None] * df.reshape(n, 3)).ravel()
    dJfit = jax.jvp(lambda x: _Jloc(fields, aux["u0"], x), (xi,), (dxi,))[1]
    dg = ((dsw[:, None] * (J - aux["Jfit"]) + sw[:, None] * (d["J"] - dJfit))
          / aux["rmsJ"]).ravel()
    Bb = aux["B_bnd"]
    nb = jnp.linalg.norm(Bb, axis=1)
    drel = d["Bn"] / nb - aux["Bn"] * jnp.sum(Bb * d["B_bnd"], 1) / nb**3
    return dg, dz, drel, d["Psi"]


def _raw_lin(fields, v, aux, n_u, n_p):
    return _raw_lin_c(fields, v, aux, n_u, n_p)


_raw_lin_c = jax.jit(_raw_lin_impl, static_argnums=(3, 4))


def _raw_lin_T(fields, w, aux, n_u, n_p, n):
    return _raw_lin_T_c(fields, w, aux, n_u, n_p, n)


def _rlt(fields, w, aux, n_u, n_p, n):
    x = jax.ShapeDtypeStruct((n,), jnp.float64)
    return jax.linear_transpose(lambda v: _raw_lin_impl(fields, v, aux, n_u, n_p), x)(w)[0]


_raw_lin_T_c = jax.jit(_rlt, static_argnums=(3, 4, 5))


_xi_cols = jax.jit(lambda fields, V, aux, n_u, n_p: jax.vmap(  # noqa: E731
    lambda v: _raw_lin_impl(fields, v, aux, n_u, n_p))(V), static_argnums=(3, 4))


# ------------------------------------------------------------ AL problem
class Weights:
    """The AL's scalar parts at an iterate: multipliers nu, penalty rho, bounds, and
    the derivatives psi', psi'' that weight the rows."""

    def __init__(self, eps, beta, Psi_t, nu=(0.0, 0.0, 0.0), rho=1.0, tau_frac=1e-3,
                 band=0.05):
        self.eps, self.beta, self.Psi_t = float(eps), float(beta), float(Psi_t)
        self.nu = np.array(nu, float)
        self.rho = float(rho)
        self.tau_frac = tau_frac
        self.band = band  # the model's penalty curvature also within this of the bound

    def constraints(self, s):
        return np.array([float(s["M"]) / self.eps - 1.0,
                         (float(s["Bn_rms"]) / self.beta) ** 2 - 1.0,
                         float(s["Psi"]) / self.Psi_t - 1.0])

    def psi(self, c, nu):
        r = self.rho
        return 0.5 * r * (max(0.0, c + nu / r) ** 2 - (nu / r) ** 2)

    def d1(self, c, nu):
        return max(0.0, nu + self.rho * c)

    def d2_model(self, c, nu):
        """The penalty curvature the step model uses: rho also when the constraint is
        inactive but within ``band`` of its bound (c + nu / rho > -band). The exact
        psi'' is 0 there, so a model without it cannot see a step that crosses the
        bound (a bound sitting at c = 0 with nu = 0 made GLTR's radius collapse);
        with it the model is conservative near the bound, and its gradient is unchanged."""
        return self.rho if c + nu / self.rho > -self.band else 0.0

    def d2(self, c, nu):
        return self.rho if c + nu / self.rho > 0 else 0.0

    def merit(self, s):
        c = self.constraints(s)
        phi = 0.5 * float(jnp.sum(s["g"] ** 2))
        return (phi + self.psi(c[0], self.nu[0]) + self.psi(c[1], self.nu[1])
                + self.nu[2] * c[2] + 0.5 * self.rho * c[2] ** 2), phi, c


class Rows:
    """The sum-of-squares model of L_A at a state: r, v -> J v, w -> J^T w (in x)."""

    def __init__(self, anchor, s, W, norm):
        self.A, self.s, self.W = anchor, s, W
        c = W.constraints(s)
        self.c = c
        w = np.asarray(s["sw"]) ** 2
        tau = W.tau_frac * W.eps * norm
        zn = jnp.linalg.norm(s["z"], axis=1)
        t = jnp.maximum(zn, jnp.asarray(w) * tau)
        d1F, d1B = W.d1(c[0], W.nu[0]), W.d1(c[1], W.nu[1])
        self.d1 = (d1F, d1B)
        self.sF = jnp.sqrt(d1F / (W.eps * norm * t))
        self.sB = np.sqrt(d1B / (W.beta**2 * anchor.n_b))
        self.sP = np.sqrt(W.rho) / W.Psi_t
        self.lin, self.linT = anchor.raw_lin(s), anchor.raw_lin_T(s)
        n3 = anchor.n_pts * 3
        self.n3, self.nb = n3, anchor.n_b
        # gradients of c_F and c_B (one transpose each) for the penalty-curvature rows
        zhat = (s["z"] / jnp.maximum(zn, 1e-300)[:, None]).ravel()
        zeros_g = jnp.zeros(n3)
        self.grad_cF = self.linT((zeros_g, zhat / (W.eps * norm), jnp.zeros(self.nb),
                                  jnp.zeros(1)))
        self.grad_cB = self.linT((zeros_g, jnp.zeros(n3),
                                  2.0 * s["rel"] / (W.beta**2 * anchor.n_b), jnp.zeros(1)))
        self.aF = np.sqrt(W.d2_model(c[0], W.nu[0])) * self.grad_cF
        self.aB = np.sqrt(W.d2_model(c[1], W.nu[1])) * self.grad_cB
        sF3 = jnp.repeat(self.sF, 3)
        self.sF3 = sF3
        self.r = jnp.concatenate([s["g"], sF3 * s["z"].ravel(), self.sB * s["rel"],
                                  jnp.asarray([np.sqrt(W.rho) * (c[2] + W.nu[2] / W.rho)]),
                                  jnp.zeros(2)])

    def J(self, v):
        dg, dz, drel, dP = self.lin(v)
        return jnp.concatenate([dg, self.sF3 * dz, self.sB * drel, self.sP * dP,
                                jnp.stack([self.aF @ v, self.aB @ v])])

    def JT(self, w):
        n3, nb = self.n3, self.nb
        wg, wz, wb, wp, wa = (w[:n3], w[n3 : 2 * n3], w[2 * n3 : 2 * n3 + nb],
                              w[2 * n3 + nb : 2 * n3 + nb + 1], w[2 * n3 + nb + 1 :])
        out = self.linT((wg, self.sF3 * wz, self.sB * wb, self.sP * wp))
        return out + wa[0] * self.aF + wa[1] * self.aB

    def grad(self):
        """grad L_A = J^T r (the model's gradient is the true one)."""
        return self.JT(self.r)

    def dense(self, L_u, LxiT_inv):
        """The stacked Jacobian in the metric variables, [J L^-T | r] (rows x (n + 1)):
        u columns from the anchor's tables (B linear in u at the anchor geometry), xi
        columns by Jacobian-vector products at the state (exact). ``L_u``: the metric's
        u block (lower triangular); ``LxiT_inv``: L_xi^-T."""
        A, s = self.A, self.s
        T = A.tables
        n_u, n3 = A.n_u, self.n3
        sw = s["sw"]
        Au = T["J"]
        # u columns, each (n_u, rows)
        dg = (sw[None, :, None] * Au / A.rmsJ).reshape(n_u, -1)
        da = (sw[None, :, None] * (jnp.cross(Au, s["B"][None]) + jnp.cross(s["J"][None], T["B"])))
        da = da.reshape(n_u, -1)
        Q = s["Q"]
        da = da - (da @ Q) @ Q.T
        dz = (sw[None, :, None] * da.reshape(n_u, -1, 3)).reshape(n_u, -1)
        Bb = s["B_bnd"]
        nb = jnp.linalg.norm(Bb, axis=1)
        drel = T["Bn"] / nb - s["Bn"] * jnp.einsum("kjc,jc->kj", T["B_bnd"], Bb) / nb**3
        dP = T["Psi"].reshape(n_u, 1)
        Cu = jnp.concatenate([dg, self.sF3[None] * dz, self.sB * drel, self.sP * dP,
                              jnp.stack([self.aF[:n_u], self.aB[:n_u]], 1)], 1)
        del dg, da, dz
        Cu = jax.scipy.linalg.solve_triangular(L_u, Cu, lower=True)  # rows of (J L^-T)^T for u
        # xi columns, in the metric variables: J L_xi^-T e_k = J (L_xi^-T)[:, k]
        LxT = LxiT_inv
        cols = []
        aux = _aux(s, A)
        for i in range(0, A.n_xi, A.tan_chunk):
            V = jnp.zeros((min(A.tan_chunk, A.n_xi - i), A.n)).at[:, n_u:].set(
                LxT[:, i : i + A.tan_chunk].T)
            dg, dz_, drel_, dP_ = _xi_cols(A.fields, V, aux, n_u, A.n_p)
            cols.append(jnp.concatenate([dg, self.sF3[None] * dz_, self.sB * drel_,
                                         self.sP * dP_.reshape(-1, 1),
                                         jnp.stack([V @ self.aF, V @ self.aB], 1)], 1))
        Cx = jnp.concatenate(cols, 0)
        return jnp.concatenate([Cu, Cx, self.r[None]], 0).T


# --------------------------------------------------------------- metric
def metric(A_u, sw, rmsJ, G_xi, kappa, reg=1e-10):
    """L = blockdiag(L_u, L_xi), L L^T = blockdiag(H_J, kappa G_xi): |L^T d| is the rms
    change of J / rms J_fit and sqrt(kappa) x the rms boundary displacement [m]."""
    n_u = A_u.shape[0]
    Aw = (sw[None, :, None] * A_u).reshape(n_u, -1) / rmsJ
    HJ = np.asarray(Aw @ Aw.T)
    blocks = []
    for H in (HJ, kappa * np.asarray(G_xi)):
        Hr = H + reg * np.mean(np.diag(H)) * np.eye(len(H))
        blocks.append(np.linalg.cholesky(Hr))
    n = n_u + len(G_xi)
    L = np.zeros((n, n))
    L[:n_u, :n_u], L[n_u:, n_u:] = blocks
    return L, HJ


# ------------------------------------------------------------- the solver
class Solver:
    """AL outer loop with trust-region inner steps (``step``: "gltr" or "qr"), across
    anchors. ``reanchor(z) -> anchor, z`` absorbs xi into a new geometry; it is called
    after an accepted step whose boundary displacement exceeds ``max_disp`` (``dmax``:
    xi -> largest displacement [m])."""

    def __init__(self, anchor, met, W, norm, step="gltr", dmax=None, max_disp=2e-3,
                 reanchor=None, log=print, cg_tol=1e-2, cg_maxiter=100, stall_dir=None,
                 run_name="run"):
        self.A, self.met, self.W, self.norm = anchor, met, W, float(norm)
        self.step, self.log = step, log
        self.dmax, self.max_disp, self.reanchor = dmax, max_disp, reanchor
        self.cg_tol, self.cg_maxiter = cg_tol, cg_maxiter
        self.Delta, self.mu = None, None
        self.Linv = None
        self.stall_dir, self.run_name = stall_dir, run_name
        self.stalls, self.anchors = [], []
        self.on_step = None  # callback(hist) after every trust-region step
        self.t_tables = []

    def _metric_inv(self):
        if self.Linv is None:
            import scipy.linalg
            L = np.asarray(self.met.L)
            n_u = self.A.n_u
            Lx = L[n_u:, n_u:]
            self.Linv = (jnp.asarray(L[:n_u, :n_u]),
                         jnp.asarray(scipy.linalg.solve_triangular(Lx.T, np.eye(len(Lx)),
                                                                   lower=False)))
        return self.Linv

    def evaluate(self, z):
        s = self.A.state(z)
        f, phi, c = self.W.merit(s)
        return s, f, phi, c

    def inner(self, z, maxiter=40, ftol=1e-12, gn_tol=1e-8, gtol=1e-3, hist=None,
              flat_tol=None, flat_window=10):
        """Trust-region steps on L_A at fixed multipliers. Stops on its own tests
        ("stationary", also when L_A changed by less than ``flat_tol`` relative over the
        last ``flat_window`` accepted steps), when the radius vanishes ("radius"), at the
        step budget ("budget") or for a re-anchoring ("reanchor")."""
        met, W = self.met, self.W
        s, f, phi, c = self.evaluate(z)
        hist = [] if hist is None else hist
        g_first = None
        if self.Delta is None:
            self.Delta = 1e-2
        tiny_run, rej_run = 0, 0
        rows, qr_cache = None, None
        for it in range(maxiter):
            t0 = time.time()
            if rows is None:
                rows = Rows(self.A, s, W, self.norm)
                qr_cache = None
            r = rows.r
            f_model0 = 0.5 * float(r @ r)
            if f_model0 == 0.0:  # nothing to reduce (at J_fit with the bounds inactive)
                return z, hist, "stationary"
            extra = {}
            if self.step == "gltr":
                y, m_step, hit, k_cg, share, gn, mu_k = pw.lanczos_tr_step(
                    rows.J, rows.JT, r, met, self.Delta, self.cg_tol, self.cg_maxiter)
                mu = None
                extra = {"lanczos": k_cg, "mu": float(mu_k)}
            else:
                if qr_cache is None:
                    t1 = time.time()
                    Lu, Lx = self._metric_inv()
                    C = rows.dense(Lu, Lx)
                    Ra = jnp.linalg.qr(C, mode="r")
                    del C
                    n = self.A.n
                    R, cc, rp = Ra[:n, :n], Ra[:n, n], float(Ra[n, n])
                    qr_cache = (R, cc, rp, time.time() - t1)
                R, cc, rp, t_qr = qr_cache
                f_lin = 0.5 * (float(cc @ cc) + rp**2)
                share = float(cc @ cc) / max(float(cc @ cc) + rp**2, 1e-300)
                gn = float(jnp.linalg.norm(R.T @ cc))
                y, mu, hit = pw.qr_tr_step(R, cc, self.Delta, self.mu)
                rl = R @ y + cc
                guard = False
                if mu == 0.0 and not float(rl @ rl) < float(cc @ cc):
                    # the undamped Gauss-Newton step of a near-singular R is unreliable
                    # (its model does not decrease): a damped step, mu = 1e-8 |R|_F^2
                    guard = True
                    mu = 1e-8 * float(jnp.sum(R * R))
                    y = pw._damped_qr(R, cc, np.sqrt(mu))
                    ny_ = float(jnp.linalg.norm(y))
                    if ny_ > self.Delta:
                        y, hit = y * (self.Delta / ny_), True
                    rl = R @ y + cc
                m_step = f_model0 - f_lin + 0.5 * (float(rl @ rl) + rp**2)
                extra = {"mu": float(mu), "t_qr": t_qr, "guard": guard}
            d = met.back(y)
            ny = float(jnp.linalg.norm(y))
            z_new = z + d
            s_new, f_new, phi_new, c_new = self.evaluate(z_new)
            pred = f_model0 - m_step
            ratio = (f - f_new) / pred if pred > 0 else 0.0
            Delta_old = self.Delta
            if not np.isfinite(ratio) or ratio < 0.25:
                self.Delta = 0.25 * ny
            elif ratio > 0.75 and hit:
                self.Delta = max(2.0 * ny, self.Delta)
            accepted = bool(np.isfinite(f_new) and f_new < f)
            g_first = gn if g_first is None else g_first
            xi_new = np.asarray(s_new["xi"] if accepted else s["xi"])
            dm = self.dmax(xi_new) if self.dmax else float(np.abs(xi_new).max())
            h = {"f": f, "f_new": f_new, "ratio": ratio, "|grad|": gn, "Delta": Delta_old,
                 "Delta_next": self.Delta, "|y|": ny, "hit": bool(hit), "GN share": share,
                 "accepted": accepted, "phi": phi_new if accepted else phi,
                 "M": float((s_new if accepted else s)["M"]),
                 "rmsF": float((s_new if accepted else s)["rmsF"]),
                 "Bn_rms": float((s_new if accepted else s)["Bn_rms"]),
                 "c": (c_new if accepted else c).tolist(), "nu": W.nu.tolist(), "rho": W.rho,
                 "psi'": list(rows.d1), "max|D|": dm,
                 "galerkin": float((s_new if accepted else s)["galerkin"]),
                 "time": time.time() - t0, **extra}
            hist.append(h)
            if self.on_step is not None:
                self.on_step(hist)
            self.log("    it {:3d} f {:.6e} -> {:.6e} ratio {:+.2f} |y| {:.2e} Delta {:.2e} "
                     "|g| {:.1e} GN {:.3f}{} M/eps {:.4f} Bn/beta {:.3f} dPsi {:+.1e} "
                     "|D| {:.2e} {:.1f}s".format(
                         it, f, f_new, ratio, ny, self.Delta, gn, share, " hit" if hit else "",
                         1 + h["c"][0], np.sqrt(max(1 + h["c"][1], 0)), h["c"][2], dm,
                         h["time"]))
            # stall detector
            tiny_run = tiny_run + 1 if (self.Delta < 1e-8 and share > 0.9) else 0
            rej_run = 0 if accepted else rej_run + 1
            if (tiny_run >= 5 or rej_run >= 3) and len(self.stalls) < 3:
                self.log("    stall detector fired: running diagnostics")
                try:
                    self.stalls.append(self.diagnose(s, rows, y, d, tag=f"{len(self.stalls)}"))
                except Exception as e:  # the diagnostic must not stop the run
                    self.log(f"    diagnostic failed: {e!r}")
                tiny_run, rej_run = 0, 0
            if accepted:
                dec = (f - f_new) / max(abs(f), 1e-300)
                z, s, f, phi, c = z_new, s_new, f_new, phi_new, c_new
                rows = None
                if mu is not None:
                    self.mu = mu
                if dm > self.max_disp and self.reanchor is not None:
                    h["reanchor"] = True
                    return z, hist, "reanchor"
                if dec < ftol or share < gn_tol or gn < gtol * g_first:
                    return z, hist, "stationary"
                fs = [e["f_new"] for e in hist if e["accepted"]]
                if (flat_tol is not None and len(fs) > flat_window
                        and abs(fs[-flat_window - 1] - fs[-1]) <= flat_tol * abs(fs[-1])):
                    return z, hist, "stationary"
            elif self.Delta < 1e-14:
                return z, hist, "radius"
        return z, hist, "budget"

    def solve_alpha(self, z, max_outer=25, tol=(1e-2, 2e-2, 1e-2), maxiter=40, rho_max=1e8,
                    gtol=1e-3, callback=None, stat_tol=1e-3, stat_window=10):
        """AL outer loop at one bound eps (``self.W.eps``). Converged when the three
        constraints are met to ``tol`` and the inner solve was stationary: it stopped on
        its own tests, or |J - J_fit|^2 / 2 changed by less than ``stat_tol`` (relative)
        over its last ``stat_window`` accepted steps."""
        W = self.W
        outer = []
        viol_prev = np.inf
        for k in range(max_outer):
            hist = []
            while True:  # one budget of maxiter steps across the re-anchorings
                z, hist, status = self.inner(z, maxiter=maxiter - len(hist), hist=hist,
                                             gtol=gtol, flat_tol=0.1 * stat_tol,
                                             flat_window=stat_window)
                if status != "reanchor" or len(hist) >= maxiter:
                    break
                t0 = time.time()
                self.A, z = self.reanchor(z, self.A)
                self.anchors.append({"outer": k, "step": len(hist), "time": time.time() - t0,
                                     **getattr(self.A, "info", {})})
                self.Linv = None  # (the metric is kept: same unknowns)
                self.log(f"  re-anchored in {time.time() - t0:.0f} s")
            s, f, phi, c = self.evaluate(z)
            viol = max(max(c[0], 0.0), max(c[1], 0.0) / 2.0, abs(c[2]))
            phis = [h["phi"] for h in hist if h["accepted"]]
            flat = (len(phis) > stat_window
                    and abs(phis[-stat_window - 1] - phis[-1]) <= stat_tol * abs(phis[-1]))
            stationary = status == "stationary" or flat
            nu_old = W.nu.copy()
            W.nu = np.array([W.d1(c[0], W.nu[0]), W.d1(c[1], W.nu[1]), W.nu[2] + W.rho * c[2]])
            o = {"outer": k, "rho": W.rho, "nu_before": nu_old.tolist(), "nu": W.nu.tolist(),
                 "c": c.tolist(), "phi": phi, "M": float(s["M"]), "rmsF": float(s["rmsF"]),
                 "Bn_rms": float(s["Bn_rms"]), "Psi": float(s["Psi"]), "viol": viol,
                 "galerkin": self.A.galerkin(s), "inner stop": status, "flat": bool(flat),
                 "stationary": bool(stationary), "inner": hist}
            outer.append(o)
            self.log("  outer {:2d} rho {:.1e}: M/eps {:.4f} Bn/beta {:.4f} dPsi {:+.2e} "
                     "phi {:.6e} nu {} galerkin {:.1e} ({} inner, {}{})".format(
                         k, W.rho, 1 + c[0], np.sqrt(max(1 + c[1], 0)), c[2], phi,
                         np.array2string(W.nu, precision=3), o["galerkin"], len(hist), status,
                         ", flat" if flat else ""))
            if callback is not None:
                callback(z, s, outer)
            if c[0] <= tol[0] and c[1] <= tol[1] and abs(c[2]) <= tol[2] and k > 0 and stationary:
                break
            if viol > 0.25 * viol_prev and W.rho < rho_max:
                W.rho *= 10.0
            viol_prev = viol
        return z, outer

    # -------------------------------------------------------- diagnostics
    def diagnose(self, s, rows, y, d, tag="0"):
        """The stall diagnostic (plan section 8): Jacobian against central differences
        (u, xi, all; the f rows separately), the adjoint test, the Galerkin identity, the
        models along the step against the truth, the IRLS weights, xi against the limit."""
        A, W, met = self.A, self.W, self.met
        z = s["z_vec"]
        rng = np.random.default_rng(int(tag) if tag.isdigit() else 0)
        out = {"Delta": self.Delta, "|y|": float(jnp.linalg.norm(y))}

        def frozen_rows(zz):
            """r at zz with the weights of ``rows`` (the rows' own function)."""
            ss = A.state(zz)
            dzv = zz - z
            return jnp.concatenate([ss["g"], rows.sF3 * ss["z"].ravel(), rows.sB * ss["rel"],
                                    jnp.asarray([rows.sP * float(ss["Psi"])
                                                 + np.sqrt(W.rho) * (W.nu[2] / W.rho - 1.0)]),
                                    jnp.stack([rows.aF @ dzv, rows.aB @ dzv])])

        n3, nb = rows.n3, rows.nb
        blocks = {"g": slice(0, n3), "f": slice(n3, 2 * n3), "Bn": slice(2 * n3, 2 * n3 + nb),
                  "Psi": slice(2 * n3 + nb, 2 * n3 + nb + 1)}
        dirs = {"step": np.asarray(d)}
        for j in range(3):
            v = rng.normal(size=A.n)
            dirs[f"rand{j}"] = v
        vu = rng.normal(size=A.n)
        vu[A.n_u :] = 0
        vx = rng.normal(size=A.n)
        vx[: A.n_u] = 0
        dirs["u only"], dirs["xi only"] = vu, vx
        jac = {}
        for name, v in dirs.items():
            v = jnp.asarray(v)
            v = v / jnp.linalg.norm(met.L.T @ v)  # unit length in the metric
            Jv = rows.J(v)
            res = {}
            for hexp in (2, 4, 6, 8):
                hh = 10.0 ** (-hexp)
                fd = (frozen_rows(z + hh * v) - frozen_rows(z - hh * v)) / (2 * hh)
                res[f"1e-{hexp}"] = {k: float(jnp.linalg.norm(fd[sl] - Jv[sl])
                                              / max(float(jnp.linalg.norm(Jv[sl])), 1e-300))
                                     for k, sl in blocks.items()}
            jac[name] = res
        out["jacobian vs FD (rel, per block)"] = jac
        v = jnp.asarray(rng.normal(size=A.n))
        wv = jnp.asarray(rng.normal(size=len(rows.r)))
        a1, a2 = float(wv @ rows.J(v)), float(rows.JT(wv) @ v)
        out["adjoint"] = abs(a1 - a2) / max(abs(a1), 1e-300)
        out["galerkin"] = A.galerkin(s)
        # the models along the step direction
        dn = d / max(float(jnp.linalg.norm(met.L.T @ d)), 1e-300)
        f0 = W.merit(s)[0]
        gL = rows.grad()
        hgrad = 1e-6

        def gradL(zz):
            return Rows(A, A.state(zz), W, self.norm).grad()

        Hd = (gradL(z + hgrad * dn) - gradL(z - hgrad * dn)) / (2 * hgrad)
        dHd = float(dn @ Hd)
        Jd = rows.J(dn)
        r = rows.r
        along = []
        for k in range(-4, 5):
            t = max(self.Delta, 1e-12) * 10.0**k
            ft = W.merit(A.state(z + t * dn))[0]
            gn_m = f0 + (0.5 * float(jnp.sum((r + t * Jd) ** 2)) - 0.5 * float(r @ r))
            so_m = f0 + t * float(gL @ dn) + 0.5 * t * t * dHd
            along.append({"t": t, "true": ft, "GN model": gn_m, "2nd-order model": so_m})
        out["along step"] = along
        out["d'Hd"], out["d'(J'J)d"] = dHd, float(Jd @ Jd)
        absF = np.asarray(s["absF"])
        tau = W.tau_frac * W.eps * self.norm
        sF = np.asarray(rows.sF)
        out["weights"] = {"frac |F| <= tau": float(np.mean(absF <= tau)),
                          "frac |F| <= 10 tau": float(np.mean(absF <= 10 * tau)),
                          "sF min/median/max": [float(sF.min()), float(np.median(sF)),
                                                float(sF.max())], "psi'": list(rows.d1)}
        xi = np.asarray(s["xi"])
        out["xi"] = {"max|D|": self.dmax(xi) if self.dmax else float(np.abs(xi).max()),
                     "limit": self.max_disp}
        if self.stall_dir:
            p = f"{self.stall_dir}/stall_{self.run_name}_{tag}.json"
            with open(p, "w") as fh:
                json.dump(out, fh, indent=1, default=float)
            np.savez(p[:-5] + ".npz", z=np.asarray(z))
            self.log(f"    stall diagnostic saved: {p}")
        return out


# ------------------------------------------------------------ Hessian
def hessian_lanczos(A, W, z, met, norm, k=60, active=None, h=1e-6, log=print):
    """Extreme eigenvalues of the Hessian of the Lagrangian |g|^2/2 + nu_F c_F + nu_B c_B
    + nu_P c_P in the metric variables, reduced to the tangent space of the active
    constraints, by Lanczos with Hessian-vector products from central differences of
    the exact gradient (each: two states and two transposes). Returns the Ritz values
    and the Gauss-Newton part's on the same space."""
    W0 = Weights(W.eps, W.beta, W.Psi_t, nu=W.nu, rho=W.rho, tau_frac=W.tau_frac)

    def grad_lag(zz):
        s = A.state(zz)
        c = W0.constraints(s)
        lin_T = A.raw_lin_T(s)
        zn = jnp.linalg.norm(s["z"], axis=1)
        zhat = (s["z"] / jnp.maximum(zn, 1e-300)[:, None]).ravel()
        n3 = A.n_pts * 3
        return lin_T((s["g"], W0.nu[0] * zhat / (W0.eps * norm),
                      W0.nu[1] * 2.0 * s["rel"] / (W0.beta**2 * A.n_b),
                      jnp.asarray([W0.nu[2] / W0.Psi_t]))), s, c

    g0, s0, c0 = grad_lag(z)
    lin_T = A.raw_lin_T(s0)
    zn = jnp.linalg.norm(s0["z"], axis=1)
    zhat = (s0["z"] / jnp.maximum(zn, 1e-300)[:, None]).ravel()
    n3 = A.n_pts * 3
    gc = [lin_T((jnp.zeros(n3), zhat / (W0.eps * norm), jnp.zeros(A.n_b), jnp.zeros(1))),
          lin_T((jnp.zeros(n3), jnp.zeros(n3), 2.0 * s0["rel"] / (W0.beta**2 * A.n_b),
                 jnp.zeros(1))),
          lin_T((jnp.zeros(n3), jnp.zeros(n3), jnp.zeros(A.n_b), jnp.asarray([1.0 / W0.Psi_t])))]
    if active is None:
        active = [c0[0] > -1e-2, c0[1] > -2e-2, True]
    # constraint gradients in the metric variables: L^-1 grad c
    Gm = [met.back_T(g) for g, a in zip(gc, active) if a]
    if Gm:
        Qc, _ = jnp.linalg.qr(jnp.stack(Gm, 1))
    else:
        Qc = jnp.zeros((A.n, 0))
    proj = lambda v: v - Qc @ (Qc.T @ v)  # noqa: E731
    lin = A.raw_lin(s0)

    def hv(y):
        d = met.back(y)
        gp = grad_lag(z + h * d)[0]
        gm = grad_lag(z - h * d)[0]
        return met.back_T((gp - gm) / (2 * h))

    def gnv(y):
        d = met.back(y)
        dg = lin(d)[0]
        return met.back_T(lin_T((dg, jnp.zeros(n3), jnp.zeros(A.n_b), jnp.zeros(1))))

    out = {}
    for name, op in (("H_Lag", hv), ("GN (|g|^2 part)", gnv)):
        rng = np.random.default_rng(1)
        q = proj(jnp.asarray(rng.normal(size=A.n)))
        q = q / jnp.linalg.norm(q)
        Qs, al, be = [q], [], []
        for j in range(k):
            wv = proj(op(Qs[-1]))
            a_j = float(Qs[-1] @ wv)
            Qm = jnp.stack(Qs)
            wv = wv - Qm.T @ (Qm @ wv)
            wv = wv - Qm.T @ (Qm @ wv)
            b_j = float(jnp.linalg.norm(wv))
            al.append(a_j)
            be.append(b_j)
            if b_j < 1e-12:
                break
            Qs.append(wv / b_j)
        Tm = np.diag(al) + np.diag(be[: len(al) - 1], 1) + np.diag(be[: len(al) - 1], -1)
        ev = np.linalg.eigvalsh(Tm)
        out[name] = {"ritz": ev.tolist(), "min": float(ev.min()), "max": float(ev.max()),
                     "negative": int(np.sum(ev < -1e-8 * abs(ev).max()))}
        log(f"  {name}: Ritz values min {ev.min():.3e}, max {ev.max():.3e}, "
            f"{out[name]['negative']} negative of {len(ev)}")
    out["active"] = [bool(a) for a in active]
    out["c"] = c0.tolist()
    return out


# ------------------------------------------------------------ profiles
@jax.jit
def iota_profile(fields, xi, B):
    """The surface-averaged rotational transform <sqrt(g) B^theta> / <sqrt(g) B^zeta> on
    each collocation radius (the flux coordinates of the geometry moved by xi; B
    Cartesian at the points). Exact for nested surfaces, an average otherwise."""
    gc, _ = fields.mg_col.at(xi)
    shape = gc.R.shape
    ph = fields.phi.reshape(shape)
    Bx, By, Bz = (B[:, i].reshape(shape) for i in range(3))
    BR, Bp = Bx * jnp.cos(ph) + By * jnp.sin(ph), -Bx * jnp.sin(ph) + By * jnp.cos(ph)
    Rr, Zr, Rt, Zt = gc.Xr[..., 0], gc.Xr[..., 1], gc.Xt[..., 0], gc.Xt[..., 1]
    Rz, Zz = gc.Xz[..., 0], gc.Xz[..., 1]
    gBt = BR * gc.R * Zr + Bp * (Zz * Rr - Rz * Zr) - Bz * gc.R * Rr
    gBz = Bp * (Zr * Rt - Rr * Zt)
    return jnp.sum(gBt, (1, 2)) / jnp.sum(gBz, (1, 2))
