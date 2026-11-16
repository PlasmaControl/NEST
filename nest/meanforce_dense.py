"""Dense (column-table) version of the fixed-boundary mean-force problem, for the exact-QR
trust region on several GPUs (jaxmg / cuSOLVERMp) and for a controlled GLTR-vs-QR comparison.

With the boundary fixed, everything the solver needs is LINEAR in the unknowns and constant:

    J(u)        = u A + J_I               A  (n_u, 3n)   J per unit coefficient (Cartesian)
    B_plasma(u) = u G + B_I               G  (n_u, 3n)   one annulus solve per coefficient
    grad p(c)   = c gP                    gP (n_p, 3n)

so the tables built once by ``export_tables`` (NEST environment, DESC + annulus) reproduce the
matrix-free ``meanforce_fixed`` state and linearization exactly, for every u, with no
re-anchoring. ``DenseAnchor`` is a drop-in ``FixedAnchor`` (same state dict); ``install``
swaps ``meanforce_fixed``'s linearization for the table one, so ``FixedRows`` / ``FixedRowsB``
(the beta model), ``FixedSolver`` (AL outer loop, LS multiplier, complementarity stop) and the
radius rules are the production code, unchanged. Only the subproblem solver differs:

    step "qr"    exact trust-region step from the QR of the stacked rows in the metric
                 variables, [R_g Ly ; a_F Ly ; F-rows Ly | r] (R_g^T R_g = H_J compresses the
                 3n misfit rows to n_u), factored by jaxmg.qr over the processes' row blocks
    step "gltr"  the Lanczos trust region of ``meanforce_fixed`` on the same rows and metric

The metric is the production one: L = chol(H_J + H_F(local) + a_F a_F^T) (``dense_force_metric``,
rebuilt per outer iteration like ``mf_fixed --precond force``) or chol(H_J). Exact linear
constraints (pinned current / flux, ``Ct``) are handled by an explicit basis Z of the feasible
subspace in the metric variables: Ly = L^-T Z, d = Ly y, |y| = the metric length of d.

This module imports no DESC: ``import_solver()`` stubs the NEST modules that need DESC
(``flux_interp``, ``objective``, ``free_boundary``) before importing meanforce / meanforce_fixed,
whose dense paths never touch them. Run in the jaxmg environment (``envs/nest_mg``).
"""

import functools
import json
import os
import sys
import time
import types

import jax
import jax.numpy as jnp
import numpy as np

TABLE_KEYS = ("A", "J_I", "G", "B_I", "B_ext", "gP", "sw", "gw", "Lg", "colnorm", "HJ", "Ct", "u0", "z0", "rho")


# ------------------------------------------------------------ DESC-free import
def import_solver():
    """meanforce, meanforce_fixed and pointwise without DESC (stubs for the modules only
    their matrix-free / annulus paths use)."""
    for name, attrs in (("flux_interp", ("TensorGeometry", "TensorGrid", "TensorCurrent")),
                        ("objective", ("_cartesian", "Fields")),
                        ("free_boundary", ("_with_geo",))):
        if name in sys.modules:
            continue
        mod = types.ModuleType(name)
        for a in attrs:
            setattr(mod, a, _missing(name, a))
        sys.modules[name] = mod
    import meanforce as mf
    import meanforce_fixed as mff
    import pointwise as pw
    return mf, mff, pw


def _missing(mod, name):
    def f(*a, **k):
        raise RuntimeError(f"{mod}.{name} is not available in the dense (DESC-free) environment")
    return f


# ---------------------------------------------------------------- the tables
def export_tables(path, fields, cur, pre, anchor, z0, HJ, Ct, meta, log=print, g_batch=16):
    """Write the tables (one .npy per array, memory-mappable) and meta.json to the directory
    ``path``, then check the table state and linearization against the matrix-free ones at
    z0 and at a random u (returns the check)."""
    import meanforce_fixed as mff
    import pointwise_tables as pt
    os.makedirs(path, exist_ok=True)
    t0 = time.time()
    T = pt.build(fields, cur, pre, with_G=True, g_batch=g_batch)
    n = len(fields.phi)
    out = {"A": T["A"].reshape(fields.n_u, 3 * n), "J_I": T["J_I"], "G": T["G"].reshape(fields.n_u, 3 * n),
           "B_I": T["B_I"], "B_ext": T["B_ext"], "gP": T["gP"].reshape(fields.n_p, 3 * n),
           "sw": np.asarray(anchor.sw), "gw": np.asarray(anchor.gw), "Lg": np.asarray(anchor.Lg),
           "colnorm": np.asarray(anchor.colnorm), "HJ": np.asarray(HJ),
           "Ct": np.zeros((fields.n_u, 0)) if Ct is None else np.asarray(Ct),
           "u0": np.asarray(anchor.u0), "z0": np.asarray(z0), "rho": np.asarray(T["rtz"])[:, 0]}
    for k, v in out.items():
        np.save(os.path.join(path, k + ".npy"), np.ascontiguousarray(v))
    meta = dict(meta, n_u=int(fields.n_u), n_p=int(fields.n_p), n=int(n), rmsJ=float(anchor.rmsJ),
                norm=float(anchor.norm), t_build=time.time() - t0)
    log(f"[dense] tables: n_u {fields.n_u}, n_p {fields.n_p}, {n} points; A, G {out['A'].nbytes / 1e9:.1f} GB "
        f"each, gP {out['gP'].nbytes / 1e9:.1f} GB; built in {time.time() - t0:.0f} s")
    # check against the matrix-free state and linearization
    Td = tables_to_device({k: out[k] for k in out})
    D = DenseAnchor(Td, meta)
    rng = np.random.default_rng(0)
    chk = {}
    for name, zz in (("z0", jnp.asarray(z0)), ("random", jnp.asarray(z0) * (1 + 1e-2 * rng.normal(size=len(z0))))):
        s_m, s_d = anchor.state(zz), D.state(zz)
        chk[name] = {k: float(jnp.linalg.norm(s_d[k] - s_m[k]) / max(float(jnp.linalg.norm(s_m[k])), 1e-300))
                     for k in ("J", "B", "f", "g", "c")}
        chk[name]["M"] = abs(float(s_d["M"]) / float(s_m["M"]) - 1.0)
    s_m = anchor.state(jnp.asarray(z0))
    v = jnp.asarray(rng.normal(size=fields.n_u)) * 1e-3
    dg_m, dz_m = mff._lin(fields, v, anchor.aux(s_m))
    dg_d, dz_d = lin(Td, v, D.aux(D.state(jnp.asarray(z0))))
    chk["lin"] = {"dg": float(jnp.linalg.norm(dg_d - dg_m) / jnp.linalg.norm(dg_m)),
                  "dz": float(jnp.linalg.norm(dz_d - dz_m) / jnp.linalg.norm(dz_m))}
    meta["check"] = chk
    with open(os.path.join(path, "meta.json"), "w") as fh:
        json.dump(meta, fh, indent=1, default=float)
    log(f"[dense] tables vs matrix-free (relative): {json.dumps(chk)}")
    return chk


def load_tables(path, keys=TABLE_KEYS):
    with open(os.path.join(path, "meta.json")) as fh:
        meta = json.load(fh)
    T = {k: np.load(os.path.join(path, k + ".npy"), mmap_mode="r") for k in keys}
    return T, meta


def tables_to_device(T):
    return {k: jnp.asarray(np.asarray(v)) for k, v in T.items()}


# -------------------------------------------------------------- state / linear
def _J(T, u):
    return (u @ T["A"]).reshape(-1, 3) + T["J_I"]


def _project(T, x):
    """(I - P G^-1 P^T) x and G^-1 P^T x, with P c = sw3 (c gP)."""
    sw3 = jnp.repeat(T["sw"], 3)
    PTx = T["gP"] @ (sw3 * x)
    c = jax.scipy.linalg.cho_solve((T["Lg"], True), PTx)
    return x - sw3 * (c @ T["gP"]), c


@jax.jit
def _state_impl(T, u, Jfit, rmsJ, norm):
    sw, gw = T["sw"], T["gw"]
    J = _J(T, u)
    B = T["B_ext"] + T["B_I"] + (u @ T["G"]).reshape(-1, 3)
    a = (sw[:, None] * jnp.cross(J, B)).ravel()
    f, c = _project(T, a)
    f, dc = _project(T, f)
    c = c + dc
    n = J.shape[0]
    F = f.reshape(n, 3) / sw[:, None]
    w = sw**2
    absF = jnp.linalg.norm(F, axis=1)
    g = ((sw * gw)[:, None] * (J - Jfit) / rmsJ).ravel()
    PTf = T["gP"] @ (jnp.repeat(sw, 3) * f)
    return {"J": J, "B": B, "f": f, "c": c, "F": F, "z": w[:, None] * F, "absF": absF,
            "M": jnp.sum(w * absF) / norm, "rmsF": jnp.sqrt(jnp.sum(w * absF**2)) / norm,
            "g": g, "PTf": PTf, "dJ": jnp.sqrt(jnp.sum(w[:, None] * (J - Jfit) ** 2)) / rmsJ}


def _lin_impl(T, v, aux):
    """v (n_u) -> (dg, dz), as ``meanforce_fixed._lin_impl``."""
    sw, J, B = T["sw"], aux["J"], aux["B"]
    dJ = (v @ T["A"]).reshape(-1, 3)
    dB = (v @ T["G"]).reshape(-1, 3)
    da = (sw[:, None] * (jnp.cross(dJ, B) + jnp.cross(J, dB))).ravel()
    df, _ = _project(T, da)
    n = J.shape[0]
    dz = (sw[:, None] * df.reshape(n, 3)).ravel()
    dg = ((sw * T["gw"])[:, None] * dJ / aux["rmsJ"]).ravel()
    return dg, dz


lin = jax.jit(_lin_impl)


def _linT_impl(T, w, aux, n_u):
    x = jax.ShapeDtypeStruct((n_u,), jnp.float64)
    return jax.linear_transpose(lambda v: _lin_impl(T, v, aux), x)(w)[0]


linT = jax.jit(_linT_impl, static_argnums=(3,))


def install(mff):
    """Point meanforce_fixed's linearization (used by FixedRows / FixedRowsB through
    ``anchor.fields``) at the tables."""
    mff._lin, mff._linT = lin, linT


class DenseAnchor:
    """``meanforce_fixed.FixedAnchor`` from the tables (``fields`` is the table dict)."""

    def __init__(self, T, meta):
        self.fields = T
        self.n_u, self.n_p, self.n_xi = int(meta["n_u"]), int(meta["n_p"]), 0
        self.n, self.n_pts, self.n_b = self.n_u, int(meta["n"]), 1
        self.rmsJ, self.norm = float(meta["rmsJ"]), float(meta["norm"])
        self.sw, self.gw, self.Lg, self.colnorm = T["sw"], T["gw"], T["Lg"], T["colnorm"]
        self.u0 = T["u0"]
        self.Jfit = _J(T, self.u0)
        self.pnorm, self.tables, self.info = None, None, {}
        dl = np.asarray(jnp.diag(self.Lg))
        self.p_rank, self.p_cond = int(self.n_p), float((dl.max() / dl.min()) ** 2)

    def state(self, z):
        z = jnp.asarray(z)
        s = dict(_state_impl(self.fields, z[: self.n_u], self.Jfit, self.rmsJ, self.norm))
        gal = jnp.max(jnp.abs(s["PTf"]) / (self.colnorm * jnp.linalg.norm(s["f"])))
        s.update({"z_vec": z, "xi": jnp.zeros(0), "sw": self.sw, "galerkin": gal, "Bn_rms": 0.0,
                  "Psi": 1.0, "rel": jnp.zeros(1), "Jfit": self.Jfit})
        self.last_z, self.last_s = z, s
        return s

    def galerkin(self, s):
        return float(s["galerkin"])

    def aux(self, s):
        return {"sw": self.sw, "J": s["J"], "B": s["B"], "Lg": self.Lg, "rmsJ": self.rmsJ, "gw": self.gw}


# ---------------------------------------------------------------- metric
class DenseMetric:
    """d = Ly y, Ly = L^-T Z (Z: orthonormal basis of the feasible subspace in the metric
    variables, the complement of range(L^-1 C^T)); |y| is the metric length of d."""

    def __init__(self, L, Ct=None):
        self.L = jnp.asarray(L)
        n = self.L.shape[0]
        Linv_T = jax.scipy.linalg.solve_triangular(self.L, jnp.eye(n), lower=True, trans="T")
        self.rank = 0
        if Ct is not None and Ct.shape[1] > 0:
            Wc = jax.scipy.linalg.solve_triangular(self.L, jnp.asarray(Ct), lower=True)
            Q, R = jnp.linalg.qr(Wc, mode="complete")
            d = np.abs(np.asarray(jnp.diag(R)))
            self.rank = int((d > 1e-12 * d.max()).sum())
            Linv_T = Linv_T @ Q[:, self.rank:]
        self.Ly = Linv_T
        self.n_free = int(self.Ly.shape[1])

    def back(self, y):
        return self.Ly @ y

    def back_T(self, d):
        return self.Ly.T @ d


def metric_L(H, reg=1e-10):
    n = H.shape[0]
    H = H.at[jnp.arange(n), jnp.arange(n)].add(reg * jnp.mean(jnp.diag(H)))
    return jnp.linalg.cholesky(H)


@jax.jit
def _K_block(T, E, B, q3):
    """Columns q Pi(sw (dJ x B0)) for the unit vectors E (the local force rows, no J0 x dB)."""
    sw = T["sw"]

    def one(e):
        dJ = (e @ T["A"]).reshape(-1, 3)
        da = (sw[:, None] * jnp.cross(dJ, B)).ravel()
        df, _ = _project(T, da)
        return q3 * jnp.repeat(sw, 3) * df
    return jax.vmap(one)(E)


@jax.jit
def _F_rows_block(T, Vt, aux, sF3, cP=None, zh=None):
    """The force rows sF3 P_beta(dz) of FixedRows / FixedRowsB for the b directions Vt (b, n_u), as
    (b, 3n), by matrix products on the tables (a vmap of the single-vector linearization made XLA
    copy A and gP transposed: 28 + 13 GB at 24/24/8)."""
    sw = T["sw"]
    b, n = Vt.shape[0], sw.shape[0]
    sw3 = jnp.repeat(sw, 3)
    dJ = (Vt @ T["A"]).reshape(b, n, 3)
    dB = (Vt @ T["G"]).reshape(b, n, 3)
    da = (sw[None, :, None] * (jnp.cross(dJ, aux["B"][None]) + jnp.cross(aux["J"][None], dB))).reshape(b, 3 * n)
    X = sw3[None] * da
    PTx = jax.lax.dot_general(X, T["gP"], (((1,), (1,)), ((), ())))  # (b, n_p), no gP^T copy
    c = jax.scipy.linalg.cho_solve((T["Lg"], True), PTx.T).T
    df = da - sw3[None] * (c @ T["gP"])
    dz = (sw3[None] * df).reshape(b, n, 3)
    if cP is not None:  # beta model: dz -> dz - cP zhat (zhat . dz)
        dz = dz - cP[None, :, None] * zh[None] * jnp.sum(zh[None] * dz, -1, keepdims=True)
    return sF3[None] * dz.reshape(b, 3 * n)


@functools.partial(jax.jit, donate_argnums=(0,))
def _put(X, B, r0, c0):
    """X[r0:, c0:] = B in place (X donated)."""
    return jax.lax.dynamic_update_slice(X, B.astype(X.dtype), (r0, c0))


@functools.partial(jax.jit, donate_argnums=(0,))
def _set_rows(X, blk, i):
    return jax.lax.dynamic_update_slice_in_dim(X, blk, i, 0)


@functools.partial(jax.jit, static_argnums=(1, 2))
def _gram_rows(K, i, b):
    """K[i:i+b] K^T, contracting over the points (axis 1 of both): no transposed copy of K."""
    Kb = jax.lax.dynamic_slice_in_dim(K, i, b, 0)
    return jax.lax.dot_general(Kb, K, (((1,), (1,)), ((), ())))


def dense_force_metric(T, rows, s, HJ, blk=128, log=print):
    """chol(H_J + H_F(local) + a_F a_F^T), H_F(local) = K^T K with K the local force rows
    (``meanforce_fixed.gram_F_local`` from the tables). Returns L and (tr H_J, tr H_F)."""
    t0 = time.time()
    n_u = HJ.shape[0]
    q3 = jnp.repeat(rows.sF, 3)
    n3 = q3.shape[0]
    # K (n_u x 3n) filled in place (no list + concatenate: that held K twice), then H_F row blocks
    # by a contraction over the points (no K.T copy: at 24/24/8 K alone is 28 GB)
    K = jnp.zeros((n_u, n3))
    for i in range(0, n_u, blk):
        E = jnp.eye(min(blk, n_u - i), n_u, k=i)
        K = _set_rows(K, _K_block(T, E, s["B"], q3), i)
    HF = jnp.zeros((n_u, n_u))
    for i in range(0, n_u, blk):
        HF = _set_rows(HF, _gram_rows(K, i, min(blk, n_u - i)), i)
    del K
    HF = 0.5 * (HF + HF.T)
    H = HJ + HF + jnp.outer(rows.aF, rows.aF)
    tr = (float(jnp.trace(HJ)), float(jnp.trace(HF)))
    L = metric_L(H)
    jax.block_until_ready(L)
    log(f"  [dense metric] trace H_J {tr[0]:.3e}, trace H_F(local) {tr[1]:.3e}, {time.time() - t0:.0f} s")
    return L, tr


# ------------------------------------------------------------ the stacked QR
def make_qr_factor(T, HJ, met_ref, kind="jaxmg", tile=512, blk=256, log=print):
    """solver.qr_factor(rows) -> (R, c, r_perp, seconds): the triangular factor of the stacked
    rows [R_g Ly ; a_F Ly ; F Ly | r] (n_u + 1 + 3n rows, n_free + 1 columns), R_g = chol(H_J)^T.
    Each process builds its own row block (``kind`` "jaxmg"); "jnp" factors on one device.
    ``met_ref``: a one-element list holding the current DenseMetric (it is rebuilt per outer)."""
    Rg = jnp.linalg.cholesky(HJ + 1e-14 * jnp.mean(jnp.diag(HJ)) * jnp.eye(HJ.shape[0])).T
    n_u = HJ.shape[0]
    if kind == "jaxmg":
        import jaxmg
        from jax.experimental import multihost_utils
        from jax.sharding import NamedSharding, PartitionSpec as P
        npr, pid = jax.process_count(), jax.process_index()
        mesh = jax.make_mesh((npr, 1), ("pr", "pc"))
    else:
        npr, pid = 1, 0

    def stacked_block(rows, sel, Ly, out):
        """Write the rows ``sel`` (a contiguous range of global indices) of [R_g Ly ; a_F Ly ; F Ly | r]
        into ``out`` (rows from 0), in place (donated updates: an un-jitted .at[].set copied the whole
        block once per column batch)."""
        n_free = Ly.shape[1]
        s0, s1 = int(sel[0]), int(sel[-1]) + 1
        u = rows.s["z_vec"][:n_u]
        n3 = rows.n3
        g0, g1 = max(s0, 0), min(s1, n_u)  # misfit rows R_g
        if g1 > g0:
            Rg_blk = Rg[g0:g1]
            out = _put(out, Rg_blk @ Ly, g0 - s0, 0)
            out = _put(out, (Rg_blk @ (u - T["u0"]))[:, None], g0 - s0, n_free)
        if s0 <= n_u < s1:  # the penalty-curvature row a_F
            out = _put(out, (rows.aF @ Ly)[None], n_u - s0, 0)
        f0, f1 = max(s0, n_u + 1) - n_u - 1, min(s1, n_u + 1 + n3) - n_u - 1  # force rows
        if f1 > f0:
            o = f0 + n_u + 1 - s0
            aux = {"J": rows.s["J"], "B": rows.s["B"]}
            cP, zh = getattr(rows, "cP", None), getattr(rows, "zh", None)
            for j in range(0, n_free, blk):
                Fj = _F_rows_block(T, Ly[:, j: j + blk].T, aux, rows.sF3, cP, zh)  # (b, 3n)
                out = _put(out, Fj[:, f0:f1].T, o, j)
                del Fj
            out = _put(out, rows.r[n3 + f0: n3 + f1][:, None], o, n_free)
        return out

    def qr_factor(rows):
        t0 = time.time()
        Ly = met_ref[0].Ly
        n_free = Ly.shape[1]
        n_rows = n_u + 1 + rows.n3
        n_cols = n_free + 1
        if kind == "jnp":
            Ra = jnp.linalg.qr(stacked_block(rows, np.arange(n_rows), Ly, jnp.zeros((n_rows, n_cols))), mode="r")
        else:
            nc = -(-n_cols // npr) * npr
            m = -(-max(n_rows, nc) // npr) * npr
            mb = m // npr
            sel = np.arange(pid * mb, min((pid + 1) * mb, n_rows))
            local = jnp.zeros((mb, nc))
            if len(sel):
                local = stacked_block(rows, sel, Ly, local)
            Ag = jax.make_array_from_process_local_data(NamedSharding(mesh, P("pr", None)), local, (m, nc))
            _, Ra = jaxmg.qr(Ag, tile, mesh=mesh, donate=True)
            if npr > 1:
                Ra = multihost_utils.process_allgather(Ra, tiled=True)
            Ra = jnp.asarray(np.asarray(Ra)[:n_cols, :n_cols])
        Ra = Ra[:n_cols, :n_cols]
        R, cc, rp = Ra[:n_free, :n_free], Ra[:n_free, n_free], float(Ra[n_free, n_free])
        return R, cc, abs(rp), time.time() - t0

    return qr_factor


# ------------------------------------------------- stable exact trust-region step (SVD of R)
def make_svd_tr_step(rcond=1e-12, log=print):
    """A drop-in for ``pointwise.qr_tr_step(R, c, Delta, mu_prev)``: the exact trust-region step of
    min |R y + c| s.t. |y| <= Delta from the SVD R = U S V^T (cached per R, so rejected trials reuse
    it). y(mu) = -V S (S^2 + mu)^-1 U^T c, so no R^T R is formed (``qr_tr_step``'s Cholesky of
    R^T R + mu I squares cond(R), ~1e8 here, and its undamped step y_N = -R^-1 c is garbage in
    the near-null directions: the QR runs' radius collapsed to 1e-12 at alpha 0.1). Singular
    values below rcond x the largest are dropped from the Gauss-Newton step (truncated SVD);
    mu on the boundary by Newton on 1/|y(mu)| - 1/Delta (monotone, safeguarded)."""
    cache = {"id": None}

    def step(R, c, Delta, mu_prev=None):
        if cache["id"] != id(R):
            U, s, Vt = jnp.linalg.svd(R, full_matrices=False)
            cache.update(id=id(R), s=np.asarray(s), g=np.asarray(U.T @ c), V=Vt.T,
                         keep=np.asarray(s) > rcond * float(s[0]))
            cache["R"] = R  # hold the reference so id(R) stays unique while cached
        s, g, V, keep = cache["s"], cache["g"], cache["V"], cache["keep"]

        def coef(mu):
            return np.where(keep | (mu > 0), -s * g / (s**2 + mu), 0.0)

        yN = coef(0.0)
        nN = float(np.linalg.norm(yN))
        if nN <= Delta:
            return V @ jnp.asarray(yN), 0.0, False
        mu = max(float(mu_prev or 0.0), 1e-16 * float(s[0]) ** 2)
        lo, hi = 0.0, np.inf
        for _ in range(100):
            y = coef(mu)
            ny = float(np.linalg.norm(y))
            if abs(ny - Delta) <= 1e-6 * Delta:
                break
            if ny > Delta:
                lo = mu
            else:
                hi = mu
            dy = float(np.sum(s**2 * g**2 * (-2) / (s**2 + mu) ** 3))  # d|y|^2/dmu
            # Newton on phi(mu) = 1/|y| - 1/Delta: mu <- mu - phi / phi', phi' = -(d|y|^2/dmu) / (2 |y|^3)
            new = mu + (1 / ny - 1 / Delta) / (dy / (2 * ny**3))
            mu = new if (lo < new < hi) else (np.sqrt(lo * hi) if np.isfinite(hi) and lo > 0 else
                                               (2 * mu if not np.isfinite(hi) else 0.5 * (lo + hi)))
        return V @ jnp.asarray(coef(mu)), float(mu), True

    return step
