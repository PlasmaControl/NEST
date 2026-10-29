"""Gates of the exact linear equality constraints (meanforce_fixed.ProjectedMetric /
LinearConstraints: the m = 0 modes of a, i.e. the current enclosed by every DESC surface in
every cross-section, and the toroidally averaged toroidal flux) on the tiny NCSX model (CPU).
C1: no constraints: ProjectedMetric == Metric (back, back_T).
C2: every step d = back(y) satisfies the constraints: |d[pinned]| / |d| and
    |a_psi . d| / (|a_psi| |d|) at rounding.
C3: adjoint <back(y), w> = <y, back_T(w)>.
C4: enclosed current I(rho, zeta) = 2 pi <T_theta>_theta: equals I at rho = 1; unchanged by
    any change of the unpinned coefficients; changed by the pinned ones; equal to the direct
    quadrature of sqrt(g) J^zeta over the cross-section inside the surface.
C5: the flux is affine in u: Psi(u + v) - Psi(u) = a_psi . v; Psi(J_fit) vs DESC's Psi (info).
C6: a GLTR step (pointwise.lanczos_tr_step) with the projected metric keeps both constraints
    and decreases the model.
C7: restore() maps a perturbed state onto the constraint set (pins exact, flux to rounding).
"""
import os, sys, time
import numpy as np
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, "nest")); sys.path.insert(0, os.path.join(ROOT, "tests"))
import test_meanforce_fixed as T  # noqa: E402  (builds the tiny model)
import jax, jax.numpy as jnp  # noqa: E402
import meanforce as mf, meanforce_fixed as mff  # noqa: E402
import pointwise as pw  # noqa: E402


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
    HJ = mff.gram_J(fields, cur, pre, rmsJ, chunk=128, log=lambda *a: None)
    L = mff.metric_L(HJ)
    rng = np.random.default_rng(3)
    n = A.n_u
    ok = {}
    LC = mff.LinearConstraints(cur, fields, fr, u0, pin_current=True, pin_flux=True, log=print)
    idx = LC.idx
    print(f"n_u {n}, pinned {len(idx)}, rank {mff.ProjectedMetric(L, LC.Ct).rank}; setup {time.time() - t0:.0f} s")

    # C1
    m0, m1 = pw.Metric(L), mff.ProjectedMetric(L, np.zeros((n, 0)))
    y = jnp.asarray(rng.normal(size=n)); w = jnp.asarray(rng.normal(size=n))
    ok["C1 no constraints: back"] = float(jnp.linalg.norm(m1.back(y) - m0.back(y)) / jnp.linalg.norm(m0.back(y)))
    ok["C1 no constraints: back_T"] = float(jnp.linalg.norm(m1.back_T(w) - m0.back_T(w)) / jnp.linalg.norm(m0.back_T(w)))

    # C2, C3
    met = LC.metric(L)
    d = met.back(y)
    ok["C2 pins |d[idx]|/|d|"] = float(jnp.linalg.norm(d[idx]) / jnp.linalg.norm(d))
    ok["C2 flux |a.d|/(|a||d|)"] = abs(float(LC.a_psi @ d)) / float(jnp.linalg.norm(LC.a_psi) * jnp.linalg.norm(d))
    ok["C3 adjoint"] = abs(float(met.back(y) @ w) - float(y @ met.back_T(w))) / abs(float(y @ met.back_T(w)))

    # C4 enclosed current
    rho = np.array([0.3, 0.6, 0.9, 1.0]); zeta = np.array([0.0, 0.37, 1.1])
    uz = np.asarray(u0) * (1 + 0.05 * rng.normal(size=n))
    I0 = mff.enclosed_current(cur, uz, rho, zeta)
    ok["C4 I(rho=1) = I"] = float(np.abs(I0[-1] - cur.I).max() / abs(cur.I))
    free = np.setdiff1d(np.arange(n), idx)
    u1 = uz.copy(); u1[free] += 0.2 * np.abs(uz).max() * rng.normal(size=len(free))
    ok["C4 unpinned change: dI/I"] = float(np.abs(mff.enclosed_current(cur, u1, rho, zeta) - I0).max() / abs(cur.I))
    u2 = uz.copy(); u2[idx] += 0.2 * np.abs(uz[idx]).max() * rng.normal(size=len(idx))
    dI2 = float(np.abs(mff.enclosed_current(cur, u2, rho[:-1], zeta) - I0[:-1]).max() / abs(cur.I))
    ok["C4 pinned change moves I (want > 1e-3; reported as 1/x)"] = 1.0 / max(dI2, 1e-300) * 1e-3
    # direct quadrature at (rho 0.6, zeta 0.37)
    rq, wq = np.polynomial.legendre.leggauss(48); rq, wq = 0.3 * (rq + 1), 0.3 * wq
    th = 2 * np.pi * np.arange(96) / 96
    Rg, Tg = np.meshgrid(rq, th, indexing="ij")
    nodes = np.column_stack([Rg.ravel(), Tg.ravel(), np.full(Rg.size, 0.37)])
    Jc = np.asarray(cur.with_u(uz).evaluate(nodes))          # (J_R, J_phi, J_Z)
    gq = inp.compute_points(["R", "sqrt(g)"], nodes)
    Jzeta = Jc[:, 1] / gq["R"]                                # J . grad(zeta), zeta = phi
    Iq = float(np.sum((gq["sqrt(g)"] * Jzeta).reshape(Rg.shape) * wq[:, None]) * 2 * np.pi / len(th))
    ok["C4 direct quadrature vs 2 pi <T_theta>"] = abs(Iq - I0[1, 1]) / abs(cur.I)

    # C5 flux affine
    v = jnp.asarray(rng.normal(size=n)) * float(jnp.linalg.norm(u0)) / np.sqrt(n)
    dpsi = float(LC.psi(u0 + v)) - float(LC.psi(u0))
    ok["C5 flux affine"] = abs(dpsi - float(LC.a_psi @ v)) / abs(dpsi)
    print(f"info: Psi_bar(J_fit) {LC.psi_ref:.6e} Wb, DESC Psi {float(eq.Psi):.6e} Wb, ratio {LC.psi_ref / float(eq.Psi):.5f}")

    # C7 restore (and the state for C6)
    zp = u0 * (1 + 0.02 * jnp.asarray(rng.normal(size=n)))
    zr = LC.restore(zp, L)
    vr = LC.violation(zr)
    ok["C7 restore pins"] = vr["pin max|du|/max|u_ref|"]
    ok["C7 restore flux (affine)"] = abs(vr["dPsi/Psi"])
    ok["C7 restore flux (direct)"] = abs(float(LC.psi(zr)) - LC.psi_ref) / abs(LC.psi_ref)

    # C6 GLTR step with the projected metric
    s = A.state(zr)
    W = mf.Weights(0.5 * float(s["M"]), 1.0, 1.0, nu=(5.0, 0.0, 0.0), rho=1.0)
    rows = mff.FixedRows(A, s, W, norm)
    D = float(jnp.linalg.norm(met.back_T(rows.JT(rows.r))))
    yy, m_step, hit, k, share, gn, mu = pw.lanczos_tr_step(rows.J, rows.JT, rows.r, met, 0.5 * D, 1e-6, 200)
    dd = met.back(yy)
    f0 = 0.5 * float(rows.r @ rows.r)
    ok["C6 step pins |d[idx]|/|d|"] = float(jnp.linalg.norm(dd[idx]) / jnp.linalg.norm(dd))
    ok["C6 step flux dPsi/(|a||d|)"] = abs(float(LC.psi(zr + dd)) - float(LC.psi(zr))) / float(jnp.linalg.norm(LC.a_psi) * jnp.linalg.norm(dd))
    ok["C6 model decrease (want > 0; reported as 1/x)"] = 1e-6 / max((f0 - m_step) / f0, 1e-300)
    print(f"C6: Lanczos {k}, hit {hit}, model decrease {(f0 - m_step) / f0:.3e} of f0")

    tol = {k: 1e-11 for k in ok}
    tol["C4 direct quadrature vs 2 pi <T_theta>"] = 1e-6
    tol["C4 pinned change moves I (want > 1e-3; reported as 1/x)"] = 1.0
    tol["C6 model decrease (want > 0; reported as 1/x)"] = 1.0
    tol["C5 flux affine"] = 1e-9
    tol["C7 restore flux (direct)"] = 1e-9
    tol["C6 step flux dPsi/(|a||d|)"] = 1e-9
    bad = []
    for kk, vv in ok.items():
        flag = "ok" if vv <= tol[kk] else "FAIL"
        print(f"{flag:4s} {kk}: {vv:.3e} (tol {tol[kk]:.0e})")
        if flag == "FAIL":
            bad.append(kk)
    print("ALL PASSED" if not bad else f"FAILED: {bad}", f"({time.time() - t0:.0f} s)")
    return 0 if not bad else 1


if __name__ == "__main__":
    sys.exit(main())
