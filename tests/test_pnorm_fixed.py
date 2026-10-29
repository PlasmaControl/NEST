"""Gates of the p-norm force bound (FixedAnchor.set_pnorm; FixedRows / FixedRowsB with per-point
weights kappa_i = (|F_i| / F_p)^(p-1) / W_int and beta = p - 1) on a tiny NCSX model (CPU).
P1: p = 1 over all points with the model beta reproduces the mean bound's FixedRowsB exactly
    (M_p = M, same residual, same J v).
P2: grad c_F . v against central differences of M_p (p = 8, interior rho < 0.9).
P3: J^T r . v against central differences of the merit phi + psi(c_F) (the exact gradient).
    P2 / P3 tolerance 1e-4 (2e-5 measured). The same check for the mean bound is printed for
    reference; it is ~1e-3 here because |F| has kinks where F passes through zero.
P4: force-row curvature |J_F v|^2 = sum sF^2 (|dz_perp|^2 + (p - 1) dz_par^2) above the floor.
P5: adjoint <J v, w> = <v, J^T w>.
P6-P8: the composite (M_p^q + (r M_edge)^q)^(1/q) bound: gradient and merit gradient vs FD,
    curvature with beta = p - 1 inside and the model beta on the edge, adjoint.
Setup follows test_hvp_fixed.
"""
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
    norm = ref["norm"]
    box = annulus(inp, inp.a / T.FRAC, 0.02); box.locate_plasma_planes(inp, sym=bool(eq.sym), verbose=False)
    fr = ForceResidual(inp, box, resolution=res); fr.norm, fr.F_desc_vol = norm, ref["F_desc_vol"]
    if "B_ext" not in vars(fr):
        fr.B_ext = _chunked(inp.B_ext, fr.x)
    fields = Fields(fr, cur, pre, dtype=jnp.float64, cyl_stencil=True, mode_chunk=16, cyl_fused=True, cyl_rblock=32, cyl_order=8)
    sw = jnp.asarray(fields.sw); u0 = jnp.asarray(cur.u)
    Jfit = mff._J(fields, u0, fields.I); rmsJ = float(jnp.sqrt(jnp.sum(sw[:, None] ** 2 * Jfit**2)))
    G = mff.gram_P(fields, cur, pre, chunk=128, log=lambda *a: None)
    A = mff.FixedAnchor(fields, cur, pre, u0, rmsJ, norm, G, log=lambda *a: None)
    assert len(fr.rtz) == A.n_pts, (len(fr.rtz), A.n_pts)
    rho_pts = np.asarray(fr.rtz[:, 0])
    rng = np.random.default_rng(1)
    z = u0 * (1 + 0.01 * jnp.asarray(rng.normal(size=A.n_u)))
    v = jnp.asarray(rng.normal(size=A.n_u)) * float(jnp.linalg.norm(u0)) / np.sqrt(A.n_u)
    ok = {}

    # P1: p = 1, all points, model beta 0.5 == the mean bound's FixedRowsB(0.5)
    s = A.state(z)
    W = mf.Weights(0.2 * float(s["M"]), 1.0, 1.0, nu=(5.0, 0.0, 0.0), rho=1.0)
    RBh = type("Bh", (mff.FixedRowsB,), {"beta": 0.5})
    R0 = RBh(A, s, W, norm)
    A.set_pnorm(1.0, np.ones(A.n_pts, bool), exact_beta=False)
    s1 = A.state(z)
    R1 = RBh(A, s1, W, norm)
    ok["P1 p = 1: M_p = M"] = abs(float(s1["M"]) - float(s["M"])) / float(s["M"])
    ok["P1 p = 1: residual"] = float(jnp.linalg.norm(R1.r - R0.r) / jnp.linalg.norm(R0.r))
    ok["P1 p = 1: J v"] = float(jnp.linalg.norm(R1.J(v) - R0.J(v)) / jnp.linalg.norm(R0.J(v)))

    # reference (info): the same finite-difference checks for the mean bound at this state
    A.pnorm = None
    s = A.state(z)
    W = mf.Weights(0.2 * float(s["M"]), 1.0, 1.0, nu=(5.0, 0.0, 0.0), rho=1.0)
    R = RBh(A, s, W, norm)
    h = 1e-5
    Mm = lambda zz: float(A.state(zz)["M"])  # noqa: E731
    fd = (Mm(z + h * v) - Mm(z - h * v)) / (2 * h) / W.eps
    ref_g = abs(float(R.grad_cF @ v) - fd) / abs(fd)
    meritm = lambda zz: float(W.merit(A.state(zz))[0])  # noqa: E731
    fdm = (meritm(z + h * v) - meritm(z - h * v)) / (2 * h)
    ref_m = abs(float(R.grad() @ v) - fdm) / abs(fdm)
    print(f"  (info) mean bound at the same state: grad c_F vs FD {ref_g:.3e}, merit gradient vs FD {ref_m:.3e}")

    # P2-P5: p = 8 on the interior, exact beta = p - 1
    p = 8.0
    A.set_pnorm(p, rho_pts < 0.9, exact_beta=True)
    s = A.state(z)
    W = mf.Weights(0.2 * float(s["M"]), 1.0, 1.0, nu=(5.0, 0.0, 0.0), rho=1.0)
    R = RBh(A, s, W, norm)
    assert abs(R.beta_used - (p - 1)) < 1e-15
    h = 1e-5
    Mp = lambda zz: float(A.state(zz)["M"])  # noqa: E731
    fd = (Mp(z + h * v) - Mp(z - h * v)) / (2 * h) / W.eps
    ok["P2 grad c_F . v vs FD of M_p"] = abs(float(R.grad_cF @ v) - fd) / abs(fd)
    merit = lambda zz: float(W.merit(A.state(zz))[0])  # noqa: E731
    fdm = (merit(z + h * v) - merit(z - h * v)) / (2 * h)
    ok["P3 J^T r . v vs FD of the merit"] = abs(float(R.grad() @ v) - fdm) / abs(fdm)
    dg, dz = R.lin(v)
    dz = dz.reshape(-1, 3); zh = R.zh; par = jnp.sum(zh * dz, 1)
    perp2 = jnp.sum(dz * dz, 1) - par**2
    bet = jnp.where(R.cP != 0, p - 1.0, 1.0)
    want = float(jnp.sum(R.sF**2 * (perp2 + bet * par**2)))
    got = float(jnp.sum(R.J(v)[R.n3 : 2 * R.n3] ** 2))
    ok["P4 force-row curvature = sF^2 (|dz_perp|^2 + (p-1) dz_par^2)"] = abs(got - want) / want
    w = jnp.asarray(rng.normal(size=R.r.shape[0]))
    ok["P5 adjoint"] = abs(float(R.J(v) @ w - v @ R.JT(w))) / abs(float(R.J(v) @ w))
    print(f"  (info) p {p:g}: M_p {float(s['M']):.4e}, interior max {float(s['M_max_int']):.4e} "
          f"(M_p / max {float(s['M']) / float(s['M_max_int']):.3f}), mean {float(s['M_mean']):.4e}, "
          f"edge mean {float(s['M_edge']):.4e}; interior points {int(np.sum(rho_pts < 0.9))} of {A.n_pts}")
    kap = np.asarray(R.kappa)
    print(f"  (info) kappa: max {kap.max():.3e}, points with kappa > 1e-3 max: {int(np.sum(kap > 1e-3 * kap.max()))}")
    # P6-P8: composite interior p-norm + edge mean, r chosen so both terms are comparable
    r_e = float(s["M_p"]) / float(s["M_edge"])
    A.set_pnorm(p, rho_pts < 0.9, exact_beta=True, edge_ratio=r_e, q=16.0)
    s = A.state(z)
    W = mf.Weights(0.2 * float(s["M"]), 1.0, 1.0, nu=(5.0, 0.0, 0.0), rho=1.0)
    R = RBh(A, s, W, norm)
    MC = lambda zz: float(A.state(zz)["M"])  # noqa: E731
    fd = (MC(z + h * v) - MC(z - h * v)) / (2 * h) / W.eps
    ok["P6 composite: grad c_F . v vs FD"] = abs(float(R.grad_cF @ v) - fd) / abs(fd)
    fdm = (merit(z + h * v) - merit(z - h * v)) / (2 * h)
    ok["P6 composite: J^T r . v vs FD of the merit"] = abs(float(R.grad() @ v) - fdm) / abs(fdm)
    dg, dz = R.lin(v)
    dz = dz.reshape(-1, 3); zh = R.zh; par = jnp.sum(zh * dz, 1)
    perp2 = jnp.sum(dz * dz, 1) - par**2
    inner = jnp.asarray(rho_pts < 0.9)
    bet = jnp.where(R.cP != 0, jnp.where(inner, p - 1.0, 0.5), 1.0)
    want = float(jnp.sum(R.sF**2 * (perp2 + bet * par**2)))
    got = float(jnp.sum(R.J(v)[R.n3 : 2 * R.n3] ** 2))
    ok["P7 composite: curvature (interior p - 1, edge model beta)"] = abs(got - want) / want
    w = jnp.asarray(rng.normal(size=R.r.shape[0]))
    ok["P8 composite: adjoint"] = abs(float(R.J(v) @ w - v @ R.JT(w))) / abs(float(R.J(v) @ w))
    print(f"  (info) composite: M {float(s['M']):.4e} (M_p {float(s['M_p']):.4e}, r M_edge {r_e * float(s['M_edge']):.4e}; "
          f"2^(1/16) x max = {2 ** (1 / 16) * max(float(s['M_p']), r_e * float(s['M_edge'])):.4e})")
    for k_, v_ in ok.items():
        print(f"  {k_}: {v_:.3e}")
    tol = {"P2": 1e-4, "P3": 1e-4, "P6": 1e-4}  # the p-norm is smooth; the mean reference (|F| kinks) is info only
    bad = [k_ for k_, v_ in ok.items() if v_ > tol.get(k_[:2], 1e-10)]
    print("ALL PASSED" if not bad else f"FAILED: {', '.join(bad)}", f"({time.time() - t0:.0f} s)")


if __name__ == "__main__":
    main()
