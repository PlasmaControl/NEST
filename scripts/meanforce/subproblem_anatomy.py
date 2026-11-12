"""Anatomy of one trust-region subproblem: every Lanczos tolerance from a single Krylov run.

GLTR iterates for a fixed gradient g and operator B = L^-1 H L^-T live in one nested family of
Krylov spaces, independent of the radius Delta. So one Lanczos run to K iterations (full
reorthogonalization, the basis kept on the device) gives, for every Delta and every k <= K, the
trust-region solution h_k(Delta) of the tridiagonal model T_k, its residual estimate
beta_k |e_k . h_k|, its model decrease, and the step y_k = Q_k^T h_k. For each Delta we record the
first k at which the residual drops below each tolerance (the step the production rule would
return at that tolerance), evaluate the true merit / M / misfit there, and compare the step with
the most converged one. The Ritz decomposition at K (theta_i, v_i = Q^T V e_i) shows where the
gradient and each step live in curvature: band shares of |g_i|^2, of |w_i|^2 (w = V^T h) and of
the model decrease -(g_i w_i + theta_i w_i^2 / 2).
For the unconstrained (interior) solution the model-decrease shortfall at residual r is exactly
r^T B^-1 r / 2, so a relative tolerance eta is only "enough" when eta^2 x (the condition number
the residual sees) << 1; the table reports the measured shortfall per tolerance.
"""
import functools
import time

import jax
import jax.numpy as jnp
import numpy as np
from scipy.linalg import eigh_tridiagonal

import meanforce_fixed as mff

TOLS = (1e-1, 3e-2, 1e-2, 3e-3, 1e-3, 1e-4, 1e-5, 1e-6)


@functools.partial(jax.jit, donate_argnums=0)
def _put(Q, k, v):
    return Q.at[k].set(v)


@jax.jit
def _reorth(Q, w):
    w = w - Q.T @ (Q @ w)
    return w - Q.T @ (Q @ w)


def lanczos(Bv, g, kmax, log, every=100, on_check=None):
    n = g.shape[0]
    gnorm = float(jnp.linalg.norm(g))
    Q = jnp.zeros((kmax + 1, n), dtype=g.dtype)
    Q = _put(Q, 0, g / gnorm)
    alpha, beta = [], []
    q_prev, b_prev = jnp.zeros_like(g), 0.0
    t0 = time.time()
    for k in range(kmax):
        q = Q[k]
        w = Bv(q) - b_prev * q_prev
        a_k = float(q @ w)
        w = _reorth(Q, w - a_k * q)
        b_k = float(jnp.linalg.norm(w))
        alpha.append(a_k)
        beta.append(b_k)
        if (k + 1) % every == 0 or k + 1 == kmax:
            th = eigh_tridiagonal(np.array(alpha), np.array(beta[:-1]), eigvals_only=True)
            msg = f"  [anatomy] Lanczos {k + 1}: Ritz [{th[0]:.3e}, {th[-1]:.3e}] (kappa {th[-1] / max(th[0], 1e-300):.2e}); {time.time() - t0:.0f} s"
            if on_check is not None:
                msg += "; " + on_check(np.array(alpha), np.array(beta), gnorm)
            log(msg)
        if b_k <= 1e-14 * gnorm:
            log(f"  [anatomy] Krylov space exhausted at {k + 1}")
            break
        q_prev, b_prev = q, b_k
        Q = _put(Q, k + 1, w / b_k)
    k = len(alpha)
    return Q[:k], np.array(alpha), np.array(beta), gnorm


def tr_at(alpha, beta, gnorm, Delta, k):
    h, mu, hit, h0 = mff._tridiag_tr_indef(alpha[:k], beta[:k], gnorm, Delta)
    T = np.diag(alpha[:k]) + np.diag(beta[: k - 1], 1) + np.diag(beta[: k - 1], -1)
    dec = -(gnorm * h[0] + 0.5 * h @ T @ h)
    return h, float(mu), bool(hit), float(beta[k - 1] * abs(h[-1]) / gnorm), float(dec)


def run(Hv, grad, f0, met, kmax, delta_mults, evaluate, base, log, out_path=None, keep=()):
    """Hv: H v (the step model's Hessian); grad: model gradient; met: metric (back, back_T);
    evaluate(d) -> (f, M_over_eps, phi); base = (f, M/eps, phi) at the state."""
    Bv = lambda v: met.back_T(Hv(met.back(v)))  # noqa: E731
    g = met.back_T(grad)
    gnorm0 = float(jnp.linalg.norm(g))
    deltas = [m * gnorm0 for m in delta_mults]

    def on_check(al, be, gn):
        parts = []
        for m, D in zip(delta_mults, deltas):
            _, _, hit, res, dec = tr_at(al, be, gn, D, len(al))
            parts.append(f"Delta {m:g}|g|: res {res:.1e} dec/f0 {dec / f0:.3e} hit {int(hit)}")
        return "; ".join(parts)

    log(f"[anatomy] |L^-1 grad| {gnorm0:.4e}, f0 {f0:.6e}; radii {', '.join(f'{d:.3e}' for d in deltas)}; K {kmax}")
    Q, alpha, beta, gnorm = lanczos(Bv, g, kmax, log, on_check=on_check)
    K = len(alpha)
    th, V = eigh_tridiagonal(alpha, beta[: K - 1])
    gc = gnorm * V[0, :]  # gradient components on the Ritz vectors
    rel = th / th.max()
    edges = [1.0, 1e-1, 1e-2, 1e-3, 1e-4, 1e-5, 1e-6, 0.0]
    band = np.digitize(rel, edges[::-1][1:-1])  # 0 = smallest band ... 6 = top decade
    names = ["<1e-6", "1e-6..1e-5", "1e-5..1e-4", "1e-4..1e-3", "1e-3..1e-2", "1e-2..1e-1", "1e-1..1"]
    gshare = np.array([np.sum(gc[band == b] ** 2) for b in range(7)]) / gnorm**2
    log(f"[anatomy] Ritz spectrum at K = {K}: theta [{th[0]:.3e}, {th[-1]:.3e}], kappa {th[-1] / max(th[0], 1e-300):.3e} "
        f"(so cond of the preconditioned Jacobian ~{np.sqrt(th[-1] / max(th[0], 1e-300)):.2e}); "
        f"Ritz values per band (theta/theta_max): " + ", ".join(f"{nm}: {int(np.sum(band == b))}" for b, nm in enumerate(names)))
    log("[anatomy] gradient |g_i|^2 share per band: " + ", ".join(f"{nm}: {gshare[b]:.3e}" for b, nm in enumerate(names)))
    ks = np.unique(np.concatenate([np.geomspace(2, K, 60).astype(int), [K]]))
    rec = {"alpha": alpha, "beta": beta, "theta": th, "gc": gc, "ks": ks, "deltas": np.array(deltas),
           "delta_mults": np.array(delta_mults), "gnorm": gnorm, "f0": f0, "tols": np.array(TOLS)}
    base_f, base_M, base_phi = base
    for m, D in zip(delta_mults, deltas):
        curve = np.array([tr_at(alpha, beta, gnorm, D, int(k))[3:] for k in ks])  # res, dec
        rec[f"D{m:g}_res"], rec[f"D{m:g}_dec"] = curve[:, 0], curve[:, 1]
        hK, muK, hitK, resK, decK = tr_at(alpha, beta, gnorm, D, K)
        if m in keep:
            rec.setdefault("steps", {})[m] = Q[:K].T @ jnp.asarray(hK)
        log(f"[anatomy] Delta {m:g} |g| = {D:.3e}: converged (K = {K}) res {resK:.2e}, model dec/f0 {decK / f0:.4e}, "
            f"|y| {np.linalg.norm(hK):.3e}, hit {hitK}, mu {muK:.3e}")
        rows = []
        for tol in TOLS:
            hitk = np.where(curve[:, 0] <= tol)[0]
            if not len(hitk):
                log(f"    tol {tol:.0e}: not reached by K = {K}")
                continue
            k = int(ks[hitk[0]])
            # refine to the exact first k between the schedule points
            lo = int(ks[hitk[0] - 1]) if hitk[0] > 0 else 1
            for kk in range(lo + 1, k + 1):
                if tr_at(alpha, beta, gnorm, D, kk)[3] <= tol:
                    k = kk
                    break
            h, mu, hit, res, dec = tr_at(alpha, beta, gnorm, D, k)
            hp = np.zeros(K)
            hp[:k] = h
            cos = float(hp @ hK / (np.linalg.norm(hp) * np.linalg.norm(hK)))
            rdiff = float(np.linalg.norm(hp - hK) / np.linalg.norm(hK))
            w, wK = V.T @ hp, V.T @ hK
            nshare = np.array([np.sum(w[band == b] ** 2) for b in range(7)]) / max(np.sum(w**2), 1e-300)
            dshare = -(gc * w + 0.5 * th * w**2)
            dshare = np.array([np.sum(dshare[band == b]) for b in range(7)]) / max(dec, 1e-300)
            y = Q[:k].T @ jnp.asarray(h)
            f_new, M_new, phi_new = evaluate(met.back(y))
            act = base_f - f_new
            e = {"tol": tol, "k": k, "res": res, "pred": dec, "act": act, "ratio": act / dec if dec > 0 else np.nan,
                 "|y|": float(np.linalg.norm(h)), "hit": hit, "cos_to_conv": cos, "reldiff_to_conv": rdiff,
                 "dec_shortfall": (decK - dec) / decK, "M/eps": M_new, "phi": phi_new,
                 "norm_share": nshare, "dec_share": dshare}
            rows.append(e)
            log(f"    tol {tol:.0e}: k {k:4d} |y| {e['|y|']:.3e} hit {int(hit)} pred/f0 {dec / f0:.4e} act/f0 {act / f0:.4e} "
                f"ratio {e['ratio']:.3f} shortfall {e['dec_shortfall']:.2e} cos {cos:.4f} |y-y*|/|y*| {rdiff:.3f} "
                f"M/eps {base_M:.4f}->{M_new:.4f} phi {base_phi:.5f}->{phi_new:.5f}")
            log("      step norm^2 share by band: " + ", ".join(f"{nm} {nshare[b]:.2e}" for b, nm in enumerate(names) if nshare[b] > 1e-4)
                + " | model-decrease share: " + ", ".join(f"{nm} {dshare[b]:.2e}" for b, nm in enumerate(names) if abs(dshare[b]) > 1e-4))
        for key in ("tol", "k", "res", "pred", "act", "ratio", "|y|", "cos_to_conv", "reldiff_to_conv", "dec_shortfall", "M/eps", "phi"):
            rec[f"D{m:g}_{key}"] = np.array([r[key] for r in rows], dtype=float)
        rec[f"D{m:g}_norm_share"] = np.array([r["norm_share"] for r in rows])
        rec[f"D{m:g}_dec_share"] = np.array([r["dec_share"] for r in rows])
        if out_path:
            np.savez(out_path, **{k: v for k, v in rec.items() if k != "steps"})
    return rec
