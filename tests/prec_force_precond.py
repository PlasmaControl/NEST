"""Measurement script (not a pass/fail gate): FP32 and TF32 force-preconditioner Gram
builds against FP64, in a near-active and in increasingly violated regimes (nu 5 ... 5e6).
Writes one CSV row per regime and precision (PREC_OUT, default prec_force_precond.csv).
Result: reduced precision loses positive definiteness once trace H_F / trace H_J exceeds ~1e8
(TF32 already at 2.7e4), so the driver builds in FP64. Setup follows test_hvp_fixed."""
import os, sys, time
import numpy as np
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, "nest")); sys.path.insert(0, os.path.join(ROOT, "tests"))
import test_meanforce_fixed as T  # noqa: E402  (builds the tiny model)
import jax, jax.numpy as jnp  # noqa: E402
import meanforce as mf, meanforce_fixed as mff  # noqa: E402


FAC = tuple(float(x) for x in os.environ.get("PREC_FAC", "0.5 0.5 0.5").split())


def main():
    t0 = time.time()
    import fit_cache
    from boundary_potential import BoundaryPotential
    from desc_input import DESCInput
    from force import ForceResidual
    from free_boundary import _chunked, reference
    from objective import Fields
    from optimize_fb import annulus
    from pressure import Pressure
    inp = DESCInput.load(os.path.join(T.D, "NCSX_fb_L16M16N12.h5"), field=os.path.join(T.D, "NCSX_coils.h5"))
    eq = inp.eq
    cur = fit_cache.fit(BoundaryPotential, inp, factors=FAC)
    pre = fit_cache.fit(Pressure, inp, factors=FAC, edge_order=2)
    b = cur.basis
    res = tuple(int(round(1.5 * (u + v))) for u, v in ((b.L, eq.L), (b.M, eq.M), (b.N, eq.N)))
    ref = reference(inp, res, current=cur)
    norm = ref["norm"]; M_desc = ref["F_desc_vol"] / norm
    box = annulus(inp, inp.a / T.FRAC, 0.02); box.locate_plasma_planes(inp, sym=bool(eq.sym), verbose=False)
    fr = ForceResidual(inp, box, resolution=res); fr.norm, fr.F_desc_vol = norm, ref["F_desc_vol"]
    if "B_ext" not in vars(fr):
        fr.B_ext = _chunked(inp.B_ext, fr.x)
    fields = Fields(fr, cur, pre, dtype=jnp.float64, cyl_stencil=True, mode_chunk=16, cyl_fused=True, cyl_rblock=32, cyl_order=8)
    sw = jnp.asarray(fields.sw); u0 = jnp.asarray(cur.u)
    Jfit = mff._J(fields, u0, fields.I); rmsJ = float(jnp.sqrt(jnp.sum(sw[:, None] ** 2 * Jfit**2)))
    G = mff.gram_P(fields, cur, pre, chunk=128, log=lambda *a: None)
    A = mff.FixedAnchor(fields, cur, pre, u0, rmsJ, norm, G, log=lambda *a: None)
    import pointwise as pw
    rng = np.random.default_rng(1)
    z = u0 * (1 + 0.01 * jnp.asarray(rng.normal(size=A.n_u)))
    s = A.state(z)
    HJ = mff.gram_J(fields, cur, pre, rmsJ, chunk=128, log=lambda *a: None)
    print(f"n_u {A.n_u}, n_p {A.n_p}, points {A.n_pts}")
    rows_out = []
    for regime, eps_fac, nu in [("near-active", 0.99, 0.01)] + [(f"violated nu {nu_:g}", 0.2, nu_) for nu_ in (5.0, 5e2, 5e4, 5e6)]:
        W = mf.Weights(eps_fac * float(s["M"]), 1.0, 1.0, nu=(nu, 0.0, 0.0), rho=1.0)
        rows = mff.FixedRows(A, s, W, norm)
        q = rows.sF * A.sw
        H = {}
        for prec in ("fp64", "fp32", "tf32"):
            t1 = time.time()
            H[prec] = mff.gram_F_local(fields, cur, pre, s["B"], q, A.Lg, chunk=128, log=lambda *a: None, precision=prec)
            H[prec + "_t"] = time.time() - t1
        aa = jnp.outer(rows.aF, rows.aF)
        Mfull = {p: HJ + H[p] + aa for p in ("fp64", "fp32", "tf32")}
        L64 = mff.metric_L(Mfull["fp64"])
        r = rows.r; f0 = 0.5 * float(r @ r); grad = rows.JT(r)
        Hv = lambda v: rows.JT(rows.J(v))
        print(f"== {regime}: trace H_J {float(jnp.trace(HJ)):.3e}, trace H_F {float(jnp.trace(H['fp64'])):.3e}")
        for p in ("fp64", "fp32", "tf32"):
            relF = float(jnp.linalg.norm(H[p] - H["fp64"]) / jnp.linalg.norm(H["fp64"]))
            Lp = mff.metric_L(Mfull[p])
            pd = bool(jnp.all(jnp.isfinite(jnp.diag(Lp))))
            X = jax.scipy.linalg.solve_triangular(L64, Mfull[p], lower=True)
            E = jax.scipy.linalg.solve_triangular(L64, X.T, lower=True)
            ev = np.linalg.eigvalsh(np.asarray(0.5 * (E + E.T)))
            msg = (f"  {p}: rel|H_F - H_F64| {relF:.2e}; eig(L64^-1 M L64^-T) [{ev.min():.6g}, {ev.max():.6g}]; "
                   f"PD {pd}; build {H[p + '_t']:.1f}s (CPU)")
            if pd:
                met = pw.Metric(Lp)
                Dl = 1e3 * float(jnp.linalg.norm(met.back_T(grad)))
                y, m, hit, k, *_ = mff.lanczos_tr_step_H(Hv, grad, f0, met, Dl, 1e-2, 600)
                d = dict(mff.LAST_H)
                msg += (f"; Lanczos {k} conv {d['converged']} theta [{d['theta_min']:.2e}, {d['theta_max']:.2e}] "
                        f"pred {f0 - m:.6e}")
            print(msg, flush=True)
            rows_out.append({"n_u": A.n_u, "regime": regime, "trace_HF_over_HJ": float(jnp.trace(H["fp64"]) / jnp.trace(HJ)),
                             "precision": p, "rel_err_HF": relF, "eig_min": float(ev.min()), "eig_max": float(ev.max()),
                             "PD": pd, "lanczos": (int(k) if pd else None)})
    import csv
    out = os.environ.get("PREC_OUT", "prec_force_precond.csv")
    with open(out, "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=list(rows_out[0])); w.writeheader(); w.writerows(rows_out)
    print(f"done ({time.time() - t0:.0f} s) -> {out}")


if __name__ == "__main__":
    main()
