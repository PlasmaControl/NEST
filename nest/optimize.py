"""Optimization of the objective: matrix-free Levenberg-Marquardt.

minimize f(x) = |r(x)|^2 / 2 over the free unknowns (``free`` mask), r from
``Objective``. Each iteration:

  1. g = J^T r (one ``vjp``)
  2. the damped Gauss-Newton step: (J^T J + lam D) d = -g on the free unknowns,
     solved by preconditioned conjugate gradients with J v and J^T w only
     (``jacobian.normal_matvec``); D is the estimated diag(J^T J) (column
     scaling, also the Jacobi preconditioner), floored at a fraction of its mean
  3. the step length: r is exactly quadratic in x, so
         r(x + a d) = r + a J d + a^2 q,   q = r(x + d) - r - J d,
     and |r(x + a d)|^2 is a quartic in a, minimized exactly (roots of a cubic)
  4. accept if f decreases (lam /= 3), else reject (lam *= 4)

``history`` keeps, per iteration, the objective, the step, lam, the CG work and
``Objective.report``.
"""

import time

import jax
import jax.numpy as jnp
import numpy as np


def pcg(matvec, b, Minv, tol=1e-2, maxiter=50):
    """Preconditioned conjugate gradients for A x = b (A symmetric positive).

    Minv: an array (diagonal preconditioner) or a function r -> M^-1 r. Stops
    when |r| <= tol |b|. Returns x and the number of iterations.
    """
    apply = Minv if callable(Minv) else (lambda r: Minv * r)
    x = jnp.zeros_like(b)
    r = b
    z = apply(r)
    p = z
    rz = jnp.vdot(r, z)
    nb = float(jnp.linalg.norm(b))
    for k in range(1, maxiter + 1):
        Ap = matvec(p)
        alpha = rz / jnp.vdot(p, Ap)
        x = x + alpha * p
        r = r - alpha * Ap
        if float(jnp.linalg.norm(r)) <= tol * nb:
            return x, k
        z = apply(r)
        rz_new = jnp.vdot(r, z)
        p = z + (rz_new / rz) * p
        rz = rz_new
    return x, maxiter


def quartic_step(r, Jd, q, a_max=10.0):
    """argmin over a in (0, a_max] of |r + a Jd + a^2 q|^2 and the minimum.

    Exact along the line (r is quadratic in x), so a long step is safe; with
    truncated CG the best step is often well above 1."""
    # |r + a J + a^2 q|^2 = c0 + c1 a + c2 a^2 + c3 a^3 + c4 a^4
    rr, rJ, rq = float(jnp.vdot(r, r)), float(jnp.vdot(r, Jd)), float(jnp.vdot(r, q))
    JJ, Jq, qq = float(jnp.vdot(Jd, Jd)), float(jnp.vdot(Jd, q)), float(jnp.vdot(q, q))
    c = np.array([rr, 2 * rJ, JJ + 2 * rq, 2 * Jq, qq])
    roots = np.roots(np.polyder(c[::-1]))  # stationary points
    cand = [a_max] + [z.real for z in roots if abs(z.imag) < 1e-12 and 0 < z.real <= a_max]
    vals = [np.polyval(c[::-1], a) for a in cand]
    i = int(np.argmin(vals))
    return cand[i], vals[i]


def _linearized(obj, x):
    """r(x), v -> J v and w -> J^T w at x (``linearize`` of the objective if it
    has one, else its jvp and vjp)."""
    if hasattr(obj, "linearize"):
        return obj.linearize(x)
    return obj.r(x), (lambda v: obj.jvp(x, v)), (lambda w: obj.vjp(x, w))


def _diag(J, JT, n, mask, n_probes, seed=0):
    """Hutchinson estimate of diag(J^T J) on the free unknowns."""
    key = jax.random.PRNGKey(seed)
    d = jnp.zeros(n)
    for _ in range(n_probes):
        key, sub = jax.random.split(key)
        z = jax.random.rademacher(sub, (n,), dtype=jnp.float64) * mask
        d = d + z * JT(J(z))
    return jnp.abs(d) / n_probes * mask


def levenberg_marquardt(obj, x0=None, free=None, maxiter=10, lam=1e-2, cg_tol=1e-2,
                        cg_maxiter=50, n_probes=8, d_floor=1e-6, gtol=1e-6, verbose=True,
                        callback=None, time_limit=None, precond=None, ftol=None):
    """Minimize |r(x)|^2 / 2 from x0 over the unknowns with ``free`` True.

    Each iteration linearizes r once (``_linearized``); the CG products, the
    gradient and the quartic step use it. D (diag(J^T J), Hutchinson) is
    estimated at the start. Stops after ``maxiter`` iterations, or when
    |J^T r| < gtol |J^T r|_0, when f decreased by less than a fraction ``ftol``
    per iteration over the last 5 (stagnation), or when the next iteration would end after
    ``time_limit`` seconds from the start. ``callback(x, history)`` is called after
    every accepted step. ``precond``: None for the Jacobi preconditioner (D from
    Hutchinson), or a block preconditioner (``precondition.BlockPreconditioner``,
    called as precond(r, lam); D is the diagonal of its blocks). Returns x and
    the history (a list of dicts).
    """
    x = obj.x0 if x0 is None else jnp.asarray(x0)
    mask = jnp.ones(obj.n) if free is None else jnp.asarray(free, float)
    t0 = time.time()
    r, J, JT = _linearized(obj, x)
    f = 0.5 * float(jnp.vdot(r, r))
    D = _diag(J, JT, obj.n, mask, n_probes) if precond is None else precond.diag * mask
    D = jnp.where(mask > 0, jnp.maximum(D, d_floor * float(D[mask > 0].mean())), 1.0)
    history = [{"it": 0, "f": f, "lam": lam, "cg": 0, "step": 0.0, "time": time.time() - t0,
                **obj.report(x)}]
    if verbose:
        _log(history[-1])
    g0 = None
    for it in range(1, maxiter + 1):
        t1 = time.time()
        if time_limit is not None and len(history) > 1 and \
                t1 - t0 + history[-1]["time"] > time_limit:
            break
        g = mask * JT(r)
        g0 = float(jnp.linalg.norm(g)) if g0 is None else g0
        if float(jnp.linalg.norm(g)) < gtol * g0:
            break
        while True:
            def mv(v, lam=lam):
                v = v * mask
                return mask * JT(J(v)) + lam * D * v
            Minv = (1.0 / (D * (1 + lam)) if precond is None
                    else (lambda r, lam=lam: mask * precond(mask * r, lam)))
            d, n_cg = pcg(mv, -g, Minv, cg_tol, cg_maxiter)
            Jd = J(d)
            q = obj.r(x + d) - r - Jd
            a, f2 = quartic_step(r, Jd, q)
            f_new = 0.5 * f2
            if f_new < f:
                x = x + a * d
                lam = max(lam / 3, 1e-8)
                break
            lam *= 4
            if verbose:
                print(f"  rejected: f(a = {a:.3g}) = {f_new:.4e} >= {f:.4e}, cg {n_cg}, "
                      f"g.d = {float(jnp.vdot(g, d)):.3e}; lam -> {lam:.1e}", flush=True)
            if lam > 1e8:
                return x, history
        r, J, JT = _linearized(obj, x)
        f = 0.5 * float(jnp.vdot(r, r))
        history.append({"it": it, "f": f, "lam": lam, "cg": n_cg, "step": a,
                        "|d|/|x|": float(jnp.linalg.norm(a * d) / jnp.linalg.norm(x)),
                        "time": time.time() - t1, **obj.report(x)})
        if verbose:
            _log(history[-1])
        if callback is not None:
            callback(x, history)
        if ftol is not None and it >= 5:
            f5 = history[-6]["f"]
            if (f5 - f) / f < 5 * ftol:
                if verbose:
                    print(f"  stop: f decreased by {(f5 - f) / f:.1e} over 5 iterations "
                          f"(ftol {ftol:g} per iteration)", flush=True)
                break
    return x, history


def _log(h):
    head = ["it", "scalar", "<|F|>/norm"]
    tail = ["lam", "cg", "step", "|d|/|x|", "time"]
    keys = head + [k for k in h if k not in head + tail + ["f"]] + tail
    print("  ".join(f"{k} {h[k]:.3e}" if isinstance(h.get(k), float) else f"{k} {h[k]}"
                    for k in keys if k in h), flush=True)
