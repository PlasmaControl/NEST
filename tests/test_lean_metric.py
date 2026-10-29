"""Lean metric builds (meanforce_fixed *_lean: gram_J_lean, gram_F_local_lean,
force_metric_lean, metric_L_lean) against the reference builds, on a tiny NCSX model (CPU).
Chunk 100 does not divide n_u or n_p, so the exact tail blocks are exercised.
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
    rel = lambda a, b: float(jnp.linalg.norm(a - b) / jnp.linalg.norm(b))
    ch = 100
    print(f"  n_u {A.n_u}, n_p {A.n_p}, chunk {ch}")
    HJ = mff.gram_J(fields, cur, pre, rmsJ, chunk=128, log=lambda *a: None)
    HJl = mff.gram_J_lean(fields, cur, pre, rmsJ, chunk=ch, log=lambda *a: None)
    ok["L1 H_J lean vs reference"] = rel(HJl, HJ)
    q = rows.sF * A.sw
    HF = mff.gram_F_local(fields, cur, pre, s["B"], q, A.Lg, chunk=128, log=lambda *a: None)
    HFl = mff.gram_F_local_lean(fields, cur, pre, s["B"], q, A.Lg, chunk=ch, log=lambda *a: None)
    ok["L2 H_F(local) lean vs reference"] = rel(HFl, HF)
    ok["L2b H_F lean symmetry (rel)"] = rel(HFl, HFl.T) if True else 0.0
    Hold = HJ + HF + jnp.outer(rows.aF, rows.aF)
    Hl, tr = mff.force_metric_lean(A, cur, pre, np.asarray(HJ), s, W, norm, rows=rows, chunk=ch,
                                   log=lambda *a: None, return_H=True)
    ok["L3 metric matrix lean vs reference"] = rel(Hl, Hold)
    Lold, _ = mff.force_metric(A, cur, pre, HJ, s, W, norm, rows=rows, chunk=128, log=lambda *a: None)
    Ll, _ = mff.force_metric_lean(A, cur, pre, np.asarray(HJ), s, W, norm, rows=rows, chunk=ch, log=lambda *a: None)
    ok["L4 Cholesky factor lean vs reference"] = rel(Ll, Lold)
    v = jnp.asarray(np.random.default_rng(2).normal(size=A.n_u))
    import pointwise as pw
    ok["L5 L^-1 v lean vs reference"] = rel(pw.Metric(Ll).back_T(v), pw.Metric(Lold).back_T(v))
    print(f"  (info) trace H_J {tr[0]:.3e}, trace H_F {tr[1]:.3e}")
    for k_, v_ in ok.items():
        print(f"  {k_}: {v_:.3e}")
    bad = [k_ for k_, v_ in ok.items() if v_ > 1e-10]
    print("ALL PASSED" if not bad else f"FAILED: {', '.join(bad)}", f"({time.time() - t0:.0f} s)")


if __name__ == "__main__":
    main()
