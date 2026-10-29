"""Gates of the exact HVP (meanforce_fixed.FixedRowsH): H v against central differences of
the exact gradient J^T r (nu, rho fixed), symmetry u.Hv = v.Hu, cost against J v + J^T w, and
the indefinite tridiagonal trust-region solve against a dense reference."""
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
    # a bound that is active: eps slightly below M, with a multiplier
    W = mf.Weights(0.97 * float(s["M"]), 1.0, 1.0, nu=(0.3, 0.0, 0.0), rho=1.0)
    rows = mff.FixedRowsH(A, s, W, norm)
    scale = float(jnp.linalg.norm(u0)) / np.sqrt(A.n_u)
    v = jnp.asarray(rng.normal(size=A.n_u)) * scale
    w = jnp.asarray(rng.normal(size=A.n_u)) * scale
    Hv, Hw = rows.hvp(v), rows.hvp(w)
    ok = {}
    for h in (1e-5, 1e-6, 1e-7, 1e-8):
        gp = mff.FixedRowsH(A, A.state(z + h * v), W, norm).grad()
        gm = mff.FixedRowsH(A, A.state(z - h * v), W, norm).grad()
        ok[f"H1 Hv vs FD of grad (h={h:g})"] = float(jnp.linalg.norm(Hv - (gp - gm) / (2 * h)) / jnp.linalg.norm(Hv))
    ok["H2 symmetry"] = abs(float(w @ Hv - v @ Hw)) / max(abs(float(w @ Hv)), 1e-300)
    # GN part alone, to see how much the true curvature differs from it
    Jv = rows.J(v); gn_vHv = float(Jv @ Jv)
    ok["(info) v.Hv / v.(J^T J)v"] = float(v @ Hv) / gn_vHv
    for _ in range(2):
        t1 = time.time(); jax.block_until_ready(rows.hvp(v)); th = time.time() - t1
        t1 = time.time(); jax.block_until_ready(rows.JT(rows.J(v))); tg = time.time() - t1
    ok["(info) t_hvp / t_(J^T J v)"] = th / tg
    # H3 indefinite tridiagonal TR vs dense reference (random T with negative eigenvalues)
    errs = []
    for trial in range(20):
        k = 12
        al = rng.normal(size=k); be = np.abs(rng.normal(size=k)); gnorm = abs(rng.normal()) + 0.1
        Delta = 10 ** rng.uniform(-1, 1)
        hh, mu, hit, _ = mff._tridiag_tr_indef(al, be, gnorm, Delta)
        Tm = np.diag(al) + np.diag(be[:-1], 1) + np.diag(be[:-1], -1)
        e1 = np.zeros(k); e1[0] = gnorm
        m = lambda x: e1 @ x + 0.5 * x @ Tm @ x  # noqa: E731
        best = min(m(x / max(1, np.linalg.norm(x) / Delta)) for x in rng.normal(size=(20000, k)) * Delta)
        errs.append(max(0.0, m(hh) - best) / max(abs(best), 1e-12) if np.linalg.norm(hh) <= Delta * (1 + 1e-6) else 1.0)
    ok["H3 indefinite TR (model above sampled best, max rel)"] = max(errs)
    print("\n".join(f"  {k}: {val:.3e}" for k, val in ok.items()))
    h1 = [v for k, v in ok.items() if k.startswith("H1")]
    ok["H1 best FD agreement"] = min(h1)
    ok["H1 O(h^2) slope (err ratio per decade, expect ~100)"] = h1[0] / h1[1]
    bad = [k for k, val in ok.items() if k == "H1 best FD agreement" and val > 1e-6 or k.startswith("H1 O") and not (30 < val < 300) or k.startswith("H2") and val > 1e-9 or k.startswith("H3") and val > 1e-9]
    print("FAILED: " + ", ".join(bad) if bad else "ALL PASSED", f"({time.time() - t0:.0f} s)")


if __name__ == "__main__":
    main()
