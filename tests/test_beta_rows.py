"""Gates of the beta model rows (meanforce_fixed.FixedRowsB) on a tiny NCSX model (CPU).
The beta model keeps the majorizer's gradient but scales the curvature of the force rows along
the active direction by beta. B1: gradient J^T r equals the majorizer rows' exact gradient.
B2: beta = 1 reproduces the majorizer rows' J v. B3: the force-row curvature equals
sF^2 (|dz_perp|^2 + beta dz_par^2) above the floor. B4: adjoint <J v, w> = <v, J^T w>.
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
    W = mf.Weights(0.2 * float(s["M"]), 1.0, 1.0, nu=(5.0, 0.0, 0.0), rho=1.0)
    R1 = mff.FixedRows(A, s, W, norm)
    ok = {}
    RB1 = type("B1", (mff.FixedRowsB,), {"beta": 1.0})(A, s, W, norm)
    RB = type("Bh", (mff.FixedRowsB,), {"beta": 0.5})(A, s, W, norm)
    v = jnp.asarray(rng.normal(size=A.n_u)) * float(jnp.linalg.norm(u0)) / np.sqrt(A.n_u)
    ok["B1 gradient J^T r equals the majorizer rows' (exact) gradient"] = float(jnp.linalg.norm(RB.JT(RB.r) - R1.JT(R1.r)) / jnp.linalg.norm(R1.JT(R1.r)))
    ok["B2 beta = 1 reproduces the majorizer rows J v"] = float(jnp.linalg.norm(RB1.J(v) - R1.J(v)) / jnp.linalg.norm(R1.J(v)))
    # B3 curvature: |J_B v|^2 force part == sum sF^2 (|dz_perp|^2 + beta dz_par^2) above the floor
    dg, dz = R1.lin(v)
    dz = dz.reshape(-1, 3); zh = RB.zh; par = jnp.sum(zh * dz, 1)
    perp2 = jnp.sum(dz * dz, 1) - par**2
    bet = jnp.where(RB.cP > 0, 0.5, 1.0)
    want = jnp.sum(R1.sF**2 * (perp2 + bet * par**2))
    n3 = R1.n3
    got = float(jnp.sum(RB.J(v)[n3 : 2 * n3] ** 2))
    ok["B3 force-row curvature = sF^2 (|dz_perp|^2 + beta dz_par^2)"] = abs(got - float(want)) / float(want)
    # B4 adjoint: <J_B v, w> = <v, J_B^T w>
    w = jnp.asarray(rng.normal(size=RB.r.shape[0]))
    ok["B4 adjoint"] = abs(float(RB.J(v) @ w - v @ RB.JT(w))) / abs(float(RB.J(v) @ w))
    print("  (info) active fraction", float(jnp.mean(RB.cP > 0)))
    for k_, v_ in ok.items():
        print(f"  {k_}: {v_:.3e}")
    bad = [k_ for k_, v_ in ok.items() if v_ > 1e-10]
    print("ALL PASSED" if not bad else f"FAILED: {', '.join(bad)}", f"({time.time() - t0:.0f} s)")


if __name__ == "__main__":
    main()
