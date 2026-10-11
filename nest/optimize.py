"""Step 9 of NEST: Gauss-Newton trust-region minimization of an ``ObjectiveFunction``.

    min_x f(x) = ||r(x)||^2

The step follows DESC's least-squares trust region (desc.optimize.least_squares
lsqtr, unbounded case) and calls DESC's own pieces:
  column scaling             the rule of compute_jac_scale: 1 / column norms of J,
                               kept monotone (max with the previous scale)
  trust_region_step_exact_cho  exact subproblem from J^T J (default): one n x n matrix;
                               squares the condition number kappa, which is measured
                               (eigvalsh) and refused if kappa^2 > 1e-3 / eps
  trust_region_step_exact_qr   exact subproblem from J = Q R (does not square kappa)
  update_tr_radius             radius update; the LM parameter alpha is rescaled by
                               old / new radius as in DESC
Two departures from lsqtr (as in fftq):
  - a step is accepted if the reduction ratio rho > 0.1 (DESC: any reduction); a
    rejected step is retried with the same factorization and a smaller radius;
  - reductions in cancellation-free form, pred = -(2 g.s + s^T B s) and act =
    -(2 r.dr + |dr|^2), and the run stops ("reduction_at_rounding") once |pred| is
    within 100x of the rounding noise of f, eps |r| (|r| + |J s|): nothing is left
    to gain. A negative pred (DESC's root-find can stop short, then rescales the
    step to the radius) is a rejected step, not a stop.

``points_per_block`` (default None: J materialized, (m, n)): J is never formed; the
rows are taken in blocks of points (``ObjectiveFunction.blocks``) and reduced to J^T J
and J^T r (cho) or to a QR factor R and z = Q^T r (qr, TSQR), as DESC's chunk sizes
trade memory for time. One JSON line per trial step is printed (prefix "GN").
"""

import json
import time

import jax
import jax.numpy as jnp
import numpy as np
from desc.backend import qr
from desc.optimize.tr_subproblems import (trust_region_step_exact_cho,
                                          trust_region_step_exact_qr, update_tr_radius)
from desc.optimize.utils import solve_triangular_regularized


def _scale(colnorm, m, prev_inv=None):
    """compute_jac_scale from column norms: 1 / max(norm, prev), 1 for null columns."""
    n = colnorm.shape[0]
    inv = jnp.where(colnorm < jnp.finfo(colnorm.dtype).eps * max(m, n), 1.0, colnorm)
    inv = inv if prev_inv is None else jnp.maximum(inv, prev_inv)
    return 1 / inv, inv


@jax.jit
def _acc_normal(A, b, J, r):
    return A + J.T @ J, b + J.T @ r


@jax.jit
def _acc_qr(R, z, J, r):
    Q, R2 = qr(jnp.concatenate([R, J]), mode="economic")
    return R2, Q.T @ jnp.concatenate([z, r])


def _reduce(obj, x, n, method, points_per_block):
    """(r, A = J^T J, b = J^T r) or (r, R, z) at x, and the column norms of J."""
    if points_per_block is None:
        r, J = obj.residual(x), obj.jacobian(x)
        if method == "qr":
            Q, R = qr(J, mode="economic")
            return r, (R, Q.T @ r), jnp.linalg.norm(J, axis=0)
        return r, (J.T @ J, J.T @ r), jnp.linalg.norm(J, axis=0)
    rs = []
    if method == "qr":
        R, z = jnp.zeros((0, n)), jnp.zeros(0)
        for rb, Jb in obj.blocks(x, points_per_block):
            rs.append(rb)
            if R.shape[0] < n:
                Q, R = qr(jnp.concatenate([R, Jb]), mode="economic")
                z = Q.T @ jnp.concatenate([z, rb])
            else:
                R, z = _acc_qr(R, z, Jb, rb)
        return jnp.concatenate(rs), (R, z), jnp.linalg.norm(R, axis=0)
    A, b = jnp.zeros((n, n)), jnp.zeros(n)
    for rb, Jb in obj.blocks(x, points_per_block):
        rs.append(rb)
        A, b = _acc_normal(A, b, Jb, rb)
    return jnp.concatenate(rs), (A, b), jnp.sqrt(jnp.maximum(jnp.diag(A), 0))


def gauss_newton(obj, x0, maxiter=100, tr_method="cho", points_per_block=None, ftol=None,
                 initial_trust_radius="scipy", max_retries=20, log_path=None, checkpoint=None,
                 callback=None, verbose=True):
    """Minimize ``obj`` from x0. Returns a dict: x, history, stop_reason, wall_s.

    Parameters
    ----------
    obj : ObjectiveFunction
    tr_method : {"cho", "qr", "auto"}
        "auto": cho while kappa(J_h)^2 <= 1e-3 / eps, else qr.
    points_per_block : int or None
        Row chunking (see the module docstring).
    ftol : float or None
        Also stop when the accepted relative reduction of f is below ftol.
    log_path, checkpoint : files for the JSON lines and for x (.npy, every step).
    callback : f(k, x, accepted) -> dict of extra values for the JSON line.
    """
    T0, eps = time.time(), float(np.finfo(float).eps)
    x = jnp.asarray(x0, dtype=jnp.float64)
    n = int(x.size)
    def emit(rec):
        rec["t"] = round(time.time() - T0, 2)
        line = json.dumps(rec)
        if verbose:
            print("GN " + line, flush=True)
        if log_path:
            with open(log_path, "a") as fh:
                fh.write(line + "\n")

    def factor(x, prev_inv):
        meth = "qr" if tr_method == "qr" else "cho"
        r, red, colnorm = _reduce(obj, x, n, meth, points_per_block)
        scale, inv = _scale(colnorm, int(r.size), prev_inv)
        kappa = None
        if meth == "cho":
            A, b = red
            B_h = scale[:, None] * A * scale[None, :]
            ev = np.asarray(jnp.linalg.eigvalsh(B_h))
            kappa = float(np.sqrt(max(ev[-1], 0) / max(ev[0], np.finfo(float).tiny)))
            if kappa**2 > 1e-3 / eps:
                if tr_method == "cho":
                    raise RuntimeError(f"kappa(J_h) = {kappa:.2e}: cho squares it beyond "
                                       "1e-3 / eps; use tr_method='qr' or 'auto'")
                meth = "qr"
                r, red, colnorm = _reduce(obj, x, n, "qr", points_per_block)
            else:
                return r, ("cho", B_h, scale * b), scale, inv, kappa
        R, z = red
        R_h = R * scale[None, :]
        return r, ("qr", R_h, z, solve_triangular_regularized(R_h, -z)), scale, inv, kappa

    t = time.time()
    r, fac, scale, inv, kappa = factor(x, None)
    t_jac = time.time() - t
    Delta = float(jnp.linalg.norm(x * inv)) if initial_trust_radius == "scipy" \
        else float(initial_trust_radius)
    Delta = Delta if Delta > 0 else 1.0
    alpha, hist, reason, k = 0.0, [], None, 0
    while k < maxiter and reason is None:
        rn = float(jnp.linalg.norm(r))
        accepted = False
        for retry in range(max_retries):
            if fac[0] == "cho":
                _, B_h, g_h = fac
                s_h, hit, alpha = trust_region_step_exact_cho(g_h, B_h, Delta, alpha)
                gs, sBs = float(g_h @ s_h), float(s_h @ (B_h @ s_h))
            else:
                _, R_h, z, p_newton = fac
                s_h, hit, alpha = trust_region_step_exact_qr(p_newton, z, R_h, Delta, alpha)
                Js = R_h @ s_h
                gs, sBs = float(z @ Js), float(Js @ Js)
            pred = -(2 * gs + sBs)
            d = scale * s_h
            r_new = obj.residual(x + d)
            dr = r_new - r
            act = -float(2 * (r @ dr) + dr @ dr)
            rho = act / pred if pred > 0 else -np.inf
            s_norm, old = float(jnp.linalg.norm(s_h)), Delta
            Delta = float(update_tr_radius(Delta, act, pred, s_norm, bool(hit))[0])
            alpha = float(alpha) * old / Delta if Delta > 0 else float(alpha)
            noise = eps * (rn + np.sqrt(max(sBs, 0))) ** 2
            rec = {"k": k, "retry": retry, "f": rn**2, "f_new": float(r_new @ r_new),
                   "pred": pred, "act": act, "rho": rho, "accepted": bool(rho > 0.1),
                   "Delta_used": old, "Delta_next": Delta, "hit_boundary": bool(hit),
                   "alpha": alpha, "step_norm": float(jnp.linalg.norm(d)),
                   "method": fac[0], "kappa_J_h": kappa, "jac_s": round(t_jac, 2)}
            t_jac = 0.0
            if rho > 0.1:
                accepted, f_old = True, rn**2
                x, r = x + d, r_new
            if callback is not None:
                rec.update(callback(k, x, accepted) or {})
            hist.append(rec)
            emit(rec)
            if abs(pred) <= 100 * noise:
                reason = "reduction_at_rounding"
                break
            if accepted:
                if ftol is not None and act / f_old < ftol:
                    reason = "ftol"
                break
        else:
            reason = "max_retries"
        if checkpoint:
            np.save(checkpoint, np.asarray(x))
        k += 1
        if reason is None and accepted:
            t = time.time()
            r, fac, scale, inv, kappa = factor(x, inv)
            t_jac = time.time() - t
    return {"x": x, "history": hist, "stop_reason": reason or "maxiter",
            "wall_s": time.time() - T0}


def continuation(obj, x0, weights, verbose=True, **kw):
    """Minimize ``obj`` for each set of term weights in turn, warm-started.

    ``weights``: list of weight tuples (one per term), e.g. for the combined objective
    [FieldDistance, ForceBalance], [(1, 1 / lam) for lam in lambdas] from the largest
    lambda (field-distance dominated) down. Returns one ``gauss_newton`` result each.
    """
    out, x = [], jnp.asarray(x0)
    for w in weights:
        obj.set_weights(w)
        if verbose:
            print(f"continuation: weights {tuple(float(v) for v in w)}; start "
                  f"{obj.summary(x)}", flush=True)
        res = gauss_newton(obj, x, verbose=verbose, **kw)
        res["weights"], res["summary"] = tuple(w), obj.summary(res["x"])
        if verbose:
            print(f"continuation: end {res['summary']}, {res['stop_reason']}", flush=True)
        out.append(res)
        x = res["x"]
    return out
