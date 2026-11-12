"""Where a quadratic model of M stops holding along a step, point by point.

Along z + t d the force is exactly quadratic per point, F_i(t) = F0_i + t f1_i + t^2 f2_i
(J linear, B affine, pressure a fixed projection), so three state evaluations (t = 0, 1, -1)
give M(t) = sum w_i |F_i(t)| / norm for every t. Compared at each t:
  M      exact,
  M_lin  the same without the quadratic force term f2 (|F0 + t f1|; what a linearized force sees),
  T2     the exact second-order Taylor expansion of M at t = 0 (what the exact-Hessian model sees).
The Taylor error M - T2 is split into the norm's own nonlinearity (M_lin - T2_lin) and the rest
(the quadratic force term dJ x dB), and its concentration over points is reported: if a few
points with small |F0| relative to their change carry the error, the limit is local (near-kink);
if it is spread, it is the global curvature of the valley.
Also: the weighted distribution of |F_i| / mean|F| at the state.
"""
import numpy as np
import jax.numpy as jnp


def fdist(absF, w, log, tag=""):
    absF, w = np.asarray(absF), np.asarray(w)
    mbar = float(np.sum(w * absF) / np.sum(w))
    r = absF / mbar
    out = {"mean|F|": mbar}
    for c in (1e-1, 1e-2, 1e-3, 1e-4):
        out[f"weight frac |F|/mean < {c:g}"] = float(np.sum(w[r < c]) / np.sum(w))
        out[f"point frac |F|/mean < {c:g}"] = float(np.mean(r < c))
    o = np.argsort(r)
    cw = np.cumsum(w[o]) / np.sum(w)
    out["weighted quantiles of |F|/mean (1, 5, 25, 50, 75, 95, 99%)"] = [
        float(r[o][np.searchsorted(cw, q)]) for q in (0.01, 0.05, 0.25, 0.5, 0.75, 0.95, 0.99)]
    log(f"[line] {tag} |F| distribution: " + "; ".join(
        f"{k} {v:.3e}" if not isinstance(v, list) else f"{k} " + " ".join(f"{x:.3g}" for x in v)
        for k, v in out.items()))
    return out


def run(A, z, d, norm, eps, log, ts, tag="", pred=None):
    """A: FixedAnchor; d: step in u (met.back(y)); ts: t grid (t = 1 is the step itself)."""
    s0, sp, sm = A.state(z), A.state(z + d), A.state(z - d)
    F0, Fp, Fm = (np.asarray(s["F"]) for s in (s0, sp, sm))
    f1, f2 = 0.5 * (Fp - Fm), 0.5 * (Fp + Fm) - F0
    w = np.asarray(A.sw) ** 2
    n0 = np.linalg.norm(F0, axis=1)
    zh = F0 / np.maximum(n0, 1e-300)[:, None]
    a1 = np.sum(zh * f1, 1)
    perp2 = np.sum(f1 * f1, 1) - a1**2
    a2 = np.sum(zh * f2, 1)
    # per-point Taylor coefficients of |F_i(t)| to second order, with and without f2
    c1, c2_lin, c2 = a1, perp2 / (2 * n0), perp2 / (2 * n0) + a2
    scale_i = n0 / np.maximum(np.linalg.norm(f1, axis=1), 1e-300)  # t at which the linear change equals |F0|
    M0 = float(np.sum(w * n0) / norm)
    rows = []
    for t in ts:
        Ft = F0 + t * f1 + t * t * f2
        Fl = F0 + t * f1
        Mi = w * np.linalg.norm(Ft, axis=1)
        Mli = w * np.linalg.norm(Fl, axis=1)
        T2i = w * (n0 + t * c1 + t * t * c2)
        T2li = w * (n0 + t * c1 + t * t * c2_lin)
        e = Mi - T2i
        e_norm = Mli - T2li
        ae = np.abs(e)
        o = np.argsort(-ae)
        tot = max(float(ae.sum()), 1e-300)
        k1, k01 = max(1, len(ae) // 100), max(1, len(ae) // 1000)
        r = {"t": float(t), "M/eps": float(Mi.sum() / norm / eps), "M_lin/eps": float(Mli.sum() / norm / eps),
             "T2/eps": float(T2i.sum() / norm / eps), "(M-T2)/eps": float(e.sum() / norm / eps),
             "norm part/eps": float(e_norm.sum() / norm / eps), "f2 part/eps": float((e - e_norm).sum() / norm / eps),
             "top 0.1% share of |M-T2|": float(ae[o[:k01]].sum() / tot), "top 1% share": float(ae[o[:k1]].sum() / tot),
             "median t-scale of top 1% points": float(np.median(scale_i[o[:k1]])),
             "median |F0|/mean of top 1% points": float(np.median(n0[o[:k1]] / (np.sum(w * n0) / np.sum(w))))}
        rows.append(r)
    log(f"[line] {tag}: M0/eps {M0 / eps:.4f}; median per-point t-scale |F0|/|f1| {np.median(scale_i):.3e} "
        f"(weighted 1% quantile {np.quantile(scale_i, 0.01):.3e})")
    for r in rows:
        log("[line] {} t {:.3f}: M/eps {:.5f} T2 {:.5f} M_lin {:.5f} | M-T2 {:+.2e} = norm {:+.2e} + f2 {:+.2e} | "
            "top0.1% {:.2f} top1% {:.2f} (t-scale {:.2e}, |F0|/mean {:.2e})".format(
                tag, r["t"], r["M/eps"], r["T2/eps"], r["M_lin/eps"], r["(M-T2)/eps"], r["norm part/eps"],
                r["f2 part/eps"], r["top 0.1% share of |M-T2|"], r["top 1% share"],
                r["median t-scale of top 1% points"], r["median |F0|/mean of top 1% points"]))
    return {"rows": rows, "fdist": fdist(n0, w, log, tag), "M0/eps": M0 / eps}


def accel_scan(A, z, y_d, met, rows, norm, eps, log, ts, fracs=(0.1, 0.3, 1.0, None), tag="", tol=1e-6, maxiter=400):
    """Geodesic acceleration along the step d = met.back(y_d), and an exact scan of M along the
    curved path z + t d + t^2 a_h.

    q(d, d) = f2 is exact (the force is quadratic along a line). In row space the second-order term
    of the force rows is sF (w f2) (IRLS weights frozen at z; the misfit rows are linear), and
    a_h = -(J^T J)^-1 J^T q solves the Gauss-Newton least squares |J a + q| in the same metric,
    by Lanczos with trust radius frac |y_d| (None: unconstrained). Along the curved path F is a
    quartic in t per point: five state evaluations (t = 0, .5, 1, 1.5, 2) give it exactly
    (checked at t = 0.75)."""
    import meanforce_fixed as mff
    d = met.back(y_d)
    s0, sp, sm = A.state(z), A.state(z + d), A.state(z - d)
    F0 = np.asarray(s0["F"])
    f2 = 0.5 * (np.asarray(sp["F"]) + np.asarray(sm["F"])) - F0
    w = np.asarray(A.sw) ** 2
    n3 = rows.n3
    q = jnp.concatenate([jnp.zeros(n3), rows.sF3 * jnp.asarray((w[:, None] * f2).ravel()), jnp.zeros(1)])
    gq = rows.JT(q)
    Hv = lambda v: rows.JT(rows.J(v))  # noqa: E731
    ny = float(jnp.linalg.norm(y_d))
    out = {"|y_d|": ny, "variants": []}
    base = run_path(A, z, d, None, norm, eps, ts)
    log(f"[accel] {tag} straight line: min M/eps {base['min']:.5f} at t {base['t_min']:.2f} (M0 {base['M0']:.5f})")
    out["straight"] = base
    for fr in fracs:
        Dlt = 1e30 if fr is None else fr * ny
        ya, m_step, hit, k, share, gn, mu = mff.lanczos_tr_step_H(Hv, gq, 0.5 * float(q @ q), met, Dlt, tol, maxiter)
        a_h = met.back(ya)
        red = 1.0 - float(jnp.linalg.norm(rows.J(a_h) + q)) / max(float(jnp.linalg.norm(q)), 1e-300)
        r = run_path(A, z, d, a_h, norm, eps, ts)
        v = {"frac": fr, "lanczos": int(k), "hit": bool(hit), "|a_h|/|y_d| (metric)": float(jnp.linalg.norm(ya)) / ny,
             "|J a + q|/|q| reduction": red, **r}
        out["variants"].append(v)
        log(f"[accel] {tag} radius {'inf' if fr is None else f'{fr:g}|y_d|'}: lanczos {k}, hit {hit}, |a_h|/|d| {v['|a_h|/|y_d| (metric)']:.3f}, "
            f"q removed {red:.3f}; min M/eps {r['min']:.5f} at t {r['t_min']:.2f}; fit check {r['fit err']:.1e}; "
            + " ".join(f"t{t:g}:{m:.4f}" for t, m in zip(r["ts"][::5], r["M/eps"][::5])))
    return out


def run_path(A, z, d, a_h, norm, eps, ts):
    nodes = np.array([0.0, 0.5, 1.0, 1.5, 2.0])
    path = (lambda t: z + t * d) if a_h is None else (lambda t: z + t * d + t * t * a_h)
    Fs = [np.asarray(A.state(path(t))["F"]) for t in nodes]
    V = np.vander(nodes, 5, increasing=True)  # F(t) = sum_k c_k t^k
    C = np.linalg.solve(V, np.stack([F.reshape(-1) for F in Fs]))  # (5, 3n)
    w = np.asarray(A.sw) ** 2
    Mt = []
    for t in ts:
        Ft = (np.array([t**k for k in range(5)]) @ C).reshape(-1, 3)
        Mt.append(float(np.sum(w * np.linalg.norm(Ft, axis=1)) / norm / eps))
    Fchk = np.asarray(A.state(path(0.75))["F"]).reshape(-1)
    Ffit = np.array([0.75**k for k in range(5)]) @ C
    err = float(np.linalg.norm(Ffit - Fchk) / max(np.linalg.norm(Fchk), 1e-300))
    i = int(np.argmin(Mt))
    return {"ts": [float(t) for t in ts], "M/eps": Mt, "min": Mt[i], "t_min": float(ts[i]), "M0": Mt[0], "fit err": err}


def _merit_grad(A, W, norm, met, z):
    """Merit L_A, M/eps, phi and the exact gradient at z: |L^-1 g| (the solver's metric) and |g| (coefficients)."""
    import meanforce_fixed as mff
    s = A.state(z)
    f, phi, c = W.merit(s)
    rows = mff.FixedRows(A, s, W, norm)
    g = rows.JT(rows.r)
    return {"f": float(f), "M/eps": 1.0 + float(c[0]), "phi": float(phi),
            "|g|_met": float(jnp.linalg.norm(met.back_T(g))), "|g|": float(jnp.linalg.norm(g))}, g


def _line_merit(A, W, norm, z, d, ts):
    """Exact merit along z + t d: F and the misfit g are quadratic / linear in t per point."""
    s0, sp, sm = A.state(z), A.state(z + d), A.state(z - d)
    F0 = np.asarray(s0["F"]); f1 = 0.5 * (np.asarray(sp["F"]) - np.asarray(sm["F"])); f2 = 0.5 * (np.asarray(sp["F"]) + np.asarray(sm["F"])) - F0
    g0 = np.asarray(s0["g"]); g1 = 0.5 * (np.asarray(sp["g"]) - np.asarray(sm["g"]))
    w = np.asarray(A.sw) ** 2
    out = []
    for t in ts:
        M = float(np.sum(w * np.linalg.norm(F0 + t * f1 + t * t * f2, axis=1)) / norm)
        phi = 0.5 * float(np.sum((g0 + t * g1) ** 2))
        c = M / W.eps - 1.0
        nu, rho = W.nu[0], W.rho
        psi = 0.5 * rho * (max(0.0, c + nu / rho) ** 2 - (nu / rho) ** 2)
        out.append((float(t), phi + psi, 1.0 + c, phi))
    return out


def chord_study(A, solver, W, norm, met, z0, K, log, n_chord=11):
    """K production inner steps from z0 at fixed (nu, rho), keeping every accepted iterate; then
    (a) along each chord between consecutive iterates: merit, M/eps and the exact gradient norm at n_chord points;
    (b) at each iterate, the 'drop to the floor' probe: the exact line minimum of the merit along the metric
        steepest-descent direction, its distance (metric units), merit gain and the gradient norm there."""
    Z, info = [z0], []
    prev_on = solver.on_step

    def on_step(hist):
        h = hist[-1]
        if h["accepted"]:
            Z.append(A.last_z)
            info.append({k: h[k] for k in ("ratio", "|y|", "hit", "Delta", "|grad|", "f", "f_new")})
    solver.on_step = on_step
    z, hist, status = solver.inner(z0, maxiter=K)
    solver.on_step = prev_on
    log(f"[chord] {len(Z) - 1} accepted steps of {len(hist)} (inner stop: {status})")
    res = {"steps": info, "chords": [], "probes": []}
    for k in range(len(Z)):
        e0, g = _merit_grad(A, W, norm, met, Z[k])
        # (b) probe along the metric steepest descent, unit metric length per t
        gm = met.back_T(g)
        d = met.back(-gm / jnp.linalg.norm(gm))
        ts = np.concatenate([[0.0], np.geomspace(1e-8, 1e6, 141)])
        lm = _line_merit(A, W, norm, Z[k], d, ts)
        i = int(np.argmin([x[1] for x in lm]))
        e1, _ = _merit_grad(A, W, norm, met, Z[k] + lm[i][0] * d)
        pr = {"iterate": k, "|g|_met": e0["|g|_met"], "|g|": e0["|g|"], "M/eps": e0["M/eps"], "t* (metric distance)": lm[i][0],
              "merit gain/f": (e0["f"] - lm[i][1]) / max(abs(e0["f"]), 1e-300), "|g|_met at t*": e1["|g|_met"],
              "|g| at t*": e1["|g|"], "M/eps at t*": e1["M/eps"]}
        # wall-Newton probe: unconstrained minimizer of the GN model in the first kk Krylov vectors of g (stiff directions first)
        import meanforce_fixed as mff
        s_k = A.state(Z[k])
        rows_k = mff.FixedRows(A, s_k, W, norm)
        Hv = lambda v: rows_k.JT(rows_k.J(v))  # noqa: E731
        for kk in (5, 20):
            yk, m_step, hit, nk, share, gn, mu = mff.lanczos_tr_step_H(Hv, g, 0.5 * float(rows_k.r @ rows_k.r), met, 1e30, 1e-14, kk)
            ek, _ = _merit_grad(A, W, norm, met, Z[k] + met.back(yk))
            pr[f"krylov{kk}"] = {"distance": float(jnp.linalg.norm(yk)), "|g|_met": ek["|g|_met"], "|g|": ek["|g|"],
                                  "merit gain/f": (e0["f"] - ek["f"]) / max(abs(e0["f"]), 1e-300), "M/eps": ek["M/eps"]}
            log(f"[probe] it {k}: wall-Newton (Krylov {kk}): |g|_met {e0['|g|_met']:.3e} -> {ek['|g|_met']:.3e} "
                f"(x{ek['|g|_met'] / max(e0['|g|_met'], 1e-300):.2e}) at metric distance {pr[f'krylov{kk}']['distance']:.3e}; "
                f"merit gain {pr[f'krylov{kk}']['merit gain/f']:.2e}; M/eps -> {ek['M/eps']:.5f}")
        res["probes"].append(pr)
        log(f"[probe] it {k}: |g|_met {pr['|g|_met']:.3e} -> {pr['|g|_met at t*']:.3e} (x{pr['|g|_met at t*'] / max(pr['|g|_met'], 1e-300):.2e}) "
            f"at metric distance {pr['t* (metric distance)']:.3e}; merit gain {pr['merit gain/f']:.2e}; M/eps {pr['M/eps']:.5f} -> {pr['M/eps at t*']:.5f}")
        if k + 1 < len(Z):
            dz = Z[k + 1] - Z[k]
            pts = []
            for t in np.linspace(0.0, 1.0, n_chord):
                e, _ = _merit_grad(A, W, norm, met, Z[k] + t * dz)
                pts.append({"t": float(t), **e})
            gs = [p["|g|_met"] for p in pts]
            j = int(np.argmin(gs))
            res["chords"].append({"from": k, "points": pts, "step": info[k]})
            log(f"[chord] {k}->{k + 1} (|y| {info[k]['|y|']:.2e}, hit {int(info[k]['hit'])}, ratio {info[k]['ratio']:+.2f}): |g|_met "
                + " ".join(f"{x:.2e}" for x in gs) + f" | min at t {pts[j]['t']:.1f}: x{gs[j] / max(gs[0], gs[-1]):.2e} of the larger end; "
                f"M/eps {pts[0]['M/eps']:.5f} -> {pts[-1]['M/eps']:.5f}")
    return res
