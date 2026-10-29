"""Gates of the local force preconditioner (meanforce_fixed.gram_F_local, metric
chol(H_J + H_F_loc + a a^T); driver --precond force) on a tiny NCSX model (CPU), in a strongly
violated bound with a large multiplier (the alpha = 0.05 regime).
F1: the slice-built Gram matrix equals the dense Gram of the explicit local rows q Pi(sw (dJ x B0)).
F2: Lanczos iterations for the same Gauss-Newton trust-region subproblem with the old metric
chol(H_J) against the force metric (informational; the force metric must need fewer).
Setup follows test_hvp_fixed."""
import os, sys, time
import numpy as np
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, "nest")); sys.path.insert(0, os.path.join(ROOT, "tests"))
import test_meanforce_fixed as T  # noqa: E402  (builds the tiny model)
import jax, jax.numpy as jnp  # noqa: E402
import meanforce as mf, meanforce_fixed as mff  # noqa: E402


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
    cur = fit_cache.fit(BoundaryPotential, inp, factors=T.FAC)
    pre = fit_cache.fit(Pressure, inp, factors=T.FAC, edge_order=2)
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
    rng = np.random.default_rng(1)
    z = u0 * (1 + 0.01 * jnp.asarray(rng.normal(size=A.n_u)))
    s = A.state(z)
    # a strongly violated bound with a large multiplier (the alpha = 0.05 regime)
    W = mf.Weights(0.2 * float(s["M"]), 1.0, 1.0, nu=(5.0, 0.0, 0.0), rho=1.0)
    rows = mff.FixedRows(A, s, W, norm)
    ok = {}
    # F1: slice Gram == dense Gram of the explicit local rows q * Pi(sw (dJ x B0))
    q = rows.sF * A.sw
    HF = mff.gram_F_local(fields, cur, pre, s["B"], q, A.Lg, chunk=128, log=lambda *a: None)
    n = A.n_pts
    def loc(v):
        dJ = mff._J(fields, v, 0.0)
        da = (sw[:, None] * jnp.cross(dJ, s["B"])).ravel()
        df, _ = mff._project(fields, sw, da, A.Lg, A.n_p)
        return jnp.repeat(q, 3) * df  # row = s_F dz = s_F sw df = q df
    D = jax.vmap(loc)(jnp.eye(A.n_u))
    Hd = D @ D.T
    ok["F1 local Gram vs dense (rel Frobenius)"] = float(jnp.linalg.norm(HF - Hd) / jnp.linalg.norm(Hd))
    # F2: Lanczos iterations for the same GN subproblem, old metric chol(H_J) vs chol(H_J + H_F + aa^T)
    HJ = mff.gram_J(fields, cur, pre, rmsJ, chunk=128, log=lambda *a: None)
    import pointwise as pw
    Lold = mff.metric_L(HJ)
    Lnew, tr = mff.force_metric(A, cur, pre, HJ, s, W, norm, rows=rows, chunk=128, log=lambda *a: None)
    r = rows.r; f0 = 0.5 * float(r @ r); grad = rows.JT(r)
    Hv = lambda v: rows.JT(rows.J(v))
    info = {}
    for name, L in (("old", Lold), ("new", Lnew)):
        met = pw.Metric(L)
        Dl = 1e3 * float(jnp.linalg.norm(met.back_T(grad)))  # effectively unconstrained
        y, m, hit, k, *_ = mff.lanczos_tr_step_H(Hv, grad, f0, met, Dl, 1e-2, 600)
        d = dict(mff.LAST_H)
        info[name] = (k, d["converged"], d["theta_min"], d["theta_max"], f0 - m)
        print(f"  (info) {name} metric: lanczos {k} conv {d['converged']} theta [{d['theta_min']:.2e}, {d['theta_max']:.2e}] pred {f0 - m:.4e}")
    print(f"  (info) trace H_J {tr[0]:.3e}, trace H_F(local) {tr[1]:.3e}")
    ok["F2 new metric needs fewer Lanczos (k_new/k_old)"] = info["new"][0] / info["old"][0]
    for k_, v_ in ok.items():
        print(f"  {k_}: {v_:.3e}")
    bad = [k_ for k_, v_ in ok.items() if (k_.startswith("F1") and v_ > 1e-10) or (k_.startswith("F2") and v_ >= 0.5)]
    print("ALL PASSED" if not bad else f"FAILED: {', '.join(bad)}", f"({time.time() - t0:.0f} s)")


if __name__ == "__main__":
    main()
