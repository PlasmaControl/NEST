"""Where the stiff and soft modes of the preconditioned step model come from (test 2).

Two Lanczos runs on the preconditioned model Hessian L^-1 B L^-T in the same force metric
L L^T = H_J + H_F,loc + a_F a_F^T:
  full  - the production rows (exact force linearization: dJ x B0 + J0 x dB, pressure projection);
  local - the same rows, weights, beta projection and penalty row with J0 x dB removed, i.e. the
          operator the metric was built from (up to the beta projection, H_J and the jitter).
If the local spectrum is O(1) while the full one spans [3.5e-8, 7.5e5], the spread comes from
the dropped Biot-Savart term. For the top and bottom Ritz vectors of the full run it reports,
for d = L^-T y and dJ = J(d):
  perp2  = <w |dJ x B0|^2> / <w |dJ|^2 |B0|^2>   (0: dJ parallel to B0; 2/3: isotropic)
  R_loc  = |s_F . Pi(sw dJ x B0)|, R_nl = |s_F . Pi(sw J0 x dB)|, R_tot = |s_F . Pi(sw (both))|
  R_g    = |dg| (misfit rows), R_a = |a_F . d| (penalty row)
  theta_chk = (R_tot^2 + R_g^2 + R_a^2) / (R_loc^2 + R_g^2 + R_a^2)   (beta = 1 Rayleigh check)
  rho_c  = w |dJ|^2 - weighted mean radius of the mode.
Smoke test (CPU, 8/8/6, a/20, from J_fit at alpha 0.7): local spectrum [0.50, 0.998]; full
[2.7e-2, 2.9e2]; top modes R_nl/R_loc 14-20, bottom modes R_nl ~ R_loc (cancelling).
"""
import time

import jax
import jax.numpy as jnp
import numpy as np
from scipy.linalg import eigh_tridiagonal

import meanforce_fixed as mff
import subproblem_anatomy as SA


def _lin_local_impl(fields, v, aux):
    """_lin_impl without the J0 x dB term."""
    sw, J, B = aux["sw"], aux["J"], aux["B"]
    n_p = aux["Lg"].shape[0]
    dJ = mff._J(fields, v, 0.0)
    da = (sw[:, None] * jnp.cross(dJ, B)).ravel()
    df, _ = mff._project(fields, sw, da, aux["Lg"], n_p)
    n = J.shape[0]
    dz = (sw[:, None] * df.reshape(n, 3)).ravel()
    dg = (sw[:, None] * dJ / aux["rmsJ"]).ravel()
    return dg, dz


_lin_local = jax.jit(_lin_local_impl)


def _linT_local_impl(fields, w, aux, n_u):
    x = jax.ShapeDtypeStruct((n_u,), jnp.float64)
    return jax.linear_transpose(lambda v: _lin_local_impl(fields, v, aux), x)(w)[0]


_linT_local = jax.jit(_linT_local_impl, static_argnums=(3,))


def _parts_impl(fields, v, aux):
    sw, J, B = aux["sw"], aux["J"], aux["B"]
    n_p = aux["Lg"].shape[0]
    n = J.shape[0]
    dJ = mff._J(fields, v, 0.0)
    dB = fields.B_plasma(v, 0.0)
    cl, cn = jnp.cross(dJ, B), jnp.cross(J, dB)
    fl, _ = mff._project(fields, sw, (sw[:, None] * cl).ravel(), aux["Lg"], n_p)
    fn, _ = mff._project(fields, sw, (sw[:, None] * cn).ravel(), aux["Lg"], n_p)
    zl = (sw[:, None] * fl.reshape(n, 3)).ravel()
    zn = (sw[:, None] * fn.reshape(n, 3)).ravel()
    w = sw**2
    dJ2 = jnp.sum(dJ**2, 1)
    perp2 = jnp.sum(w * jnp.sum(cl**2, 1)) / jnp.sum(w * dJ2 * jnp.sum(B**2, 1))
    dg = (sw[:, None] * dJ / aux["rmsJ"]).ravel()
    return zl, zn, perp2, jnp.linalg.norm(dg), w * dJ2


_parts = jax.jit(_parts_impl)


def run(rows, A, met, s, K_full, K_loc, nvec, log, out_path=None):
    fields, aux, n3, n_u = A.fields, A.aux(s), rows.n3, A.n_u
    P = rows._P if hasattr(rows, "_P") else (lambda x: x)

    def J_loc(v):
        dg, dz = _lin_local(fields, v, aux)
        return jnp.concatenate([dg, rows.sF3 * P(dz), jnp.stack([rows.aF @ v])])

    def JT_loc(w):
        out = _linT_local(fields, (w[:n3], P(rows.sF3 * w[n3 : 2 * n3])), aux, n_u)
        return out + w[2 * n3] * rows.aF

    grad = rows.JT(rows.r)
    g = met.back_T(grad)
    rec = {}
    # radial slices of the collocation points (for the mode centroids)
    tg = fields.force.tgrid
    m_sl = len(tg.theta) * len(tg.zeta)
    rho_pts = np.repeat(np.asarray(tg.rho), m_sl) if len(tg.rho) * m_sl == A.n_pts else None
    for name, Hv, K in (("full", lambda v: rows.JT(rows.J(v)), K_full), ("local", lambda v: JT_loc(J_loc(v)), K_loc)):
        Bv = lambda v, Hv=Hv: met.back_T(Hv(met.back(v)))  # noqa: E731
        t0 = time.time()
        log(f"[stiff] {name} model: Lanczos to {K}")
        Q, al, be, gn = SA.lanczos(Bv, g, K, log, every=100)
        k = len(al)
        th, V = eigh_tridiagonal(al, be[: k - 1])
        res = be[k - 1] * np.abs(V[-1, :])  # Ritz residuals |B y - theta y|
        gc = gn * V[0, :]
        edges = [0, 1e-6, 1e-4, 1e-2, 1, 1e2, 1e4, np.inf]
        cnt = [int(np.sum((th >= lo) & (th < hi))) for lo, hi in zip(edges[:-1], edges[1:])]
        gsh = [float(np.sum(gc[(th >= lo) & (th < hi)] ** 2) / gn**2) for lo, hi in zip(edges[:-1], edges[1:])]
        log(f"[stiff] {name}: K {k}, {time.time() - t0:.0f} s; theta [{th[0]:.3e}, {th[-1]:.3e}], kappa {th[-1] / max(th[0], 1e-300):.3e}")
        log(f"[stiff] {name}: Ritz counts in theta bands [0,1e-6,1e-4,1e-2,1,1e2,1e4,inf): {cnt}")
        log(f"[stiff] {name}: gradient share per band: {['%.2e' % x for x in gsh]}")
        rec[f"{name}_theta"], rec[f"{name}_res"], rec[f"{name}_gc"] = th, res, gc
        rec[f"{name}_alpha"], rec[f"{name}_beta"] = al, be
        if name == "full":
            idx = list(range(k - 1, k - 1 - nvec, -1)) + list(range(min(nvec, k)))
            rows_out = []
            log("[stiff]   i    theta      Ritz res   perp2   R_loc      R_nl       R_nl/R_loc R_g        R_a        theta_chk  rho_c")
            for i in idx:
                y = Q.T @ jnp.asarray(V[:, i])
                d = met.back(y)
                zl, zn, perp2, Rg, e = _parts(fields, d, aux)
                Rl = float(jnp.linalg.norm(rows.sF3 * zl))
                Rn = float(jnp.linalg.norm(rows.sF3 * zn))
                Rt = float(jnp.linalg.norm(rows.sF3 * (zl + zn)))
                Ra = float(abs(rows.aF @ d))
                Rg = float(Rg)
                chk = (Rt**2 + Rg**2 + Ra**2) / max(Rl**2 + Rg**2 + Ra**2, 1e-300)
                rc = float(np.sum(rho_pts * np.asarray(e)) / np.sum(np.asarray(e))) if rho_pts is not None else np.nan
                row = [i, th[i], res[i], float(perp2), Rl, Rn, Rn / max(Rl, 1e-300), Rg, Ra, chk, rc]
                rows_out.append(row)
                log("[stiff] " + f"{i:4d} " + " ".join(f"{x:.3e}" for x in row[1:]))
            rec["modes"] = np.array(rows_out)
            rec["modes_cols"] = np.array(["i", "theta", "ritz_res", "perp2", "R_loc", "R_nl", "R_nl/R_loc", "R_g", "R_a", "theta_chk", "rho_c"])
        del Q
    if out_path:
        np.savez(out_path, **rec)
        log(f"[stiff] saved {out_path}")
    return rec
