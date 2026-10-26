"""Closest free-boundary state to J_fit under a bound on DESC's mean force, the pressure
from the weak Poisson equation (``nest/meanforce.py``; docs/mean_force_poisson_plan.md).

Per bound eps = alpha <|F|>_DESC / norm: the AL outer loop with GLTR or QR trust-region
steps, re-anchoring the geometry when the boundary has moved by --max-disp. Saves per
alpha <out>_a<alpha>.npz (z = (u, xi), p*, the geometry <out>_a<alpha>_eq.h5) and
<out>.json (gates, start metrics, every step and outer iteration, anchors, the final
B.n check by direct Biot-Savart, the Hessian's extreme eigenvalues on the tangent
space). The stall detector's diagnostics go to <out>_stall_*.json.

run (Stellar AI): sbatch --qos=gpu-medium --time=24:00:00 run_sai.sbatch \
    scripts/meanforce/mf_run.py --method qr --alphas 0.7 --out results/meanforce/qr_a40
"""

import argparse
import gc
import json
import os
import sys
import time

import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.join(ROOT, "nest"))
sys.path.insert(0, os.path.join(ROOT, "scripts"))

from desc import set_device  # noqa: E402

set_device(os.environ.get("NEST_DEVICE", "cpu"))
import jax  # noqa: E402
import jax.numpy as jnp  # noqa: E402

jax.config.update("jax_enable_x64", True)

import fit_cache  # noqa: E402
import meanforce as mf  # noqa: E402
import pointwise as pw  # noqa: E402
from boundary_potential import BoundaryPotential  # noqa: E402
from cylindrical_grid import CylindricalGrid  # noqa: E402
from desc.grid import LinearGrid  # noqa: E402
from desc_input import DESCInput  # noqa: E402
from force import ForceResidual  # noqa: E402
from free_boundary import Displacement, FreeBoundaryFields, reference, release  # noqa: E402
from optimize_fb import annulus, boundary_check, rebase  # noqa: E402
from pressure import Pressure  # noqa: E402

D = os.path.join(ROOT, "desc_input", "vmec_free_boundary", "NCSX")


def peak():
    try:
        return jax.devices()[0].memory_stats()["peak_bytes_in_use"] / 1e9
    except Exception:
        return float("nan")


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--eq", default=os.path.join(D, "NCSX_fb_L16M16N12.h5"))
    ap.add_argument("--coils", default=os.path.join(D, "NCSX_coils.h5"))
    ap.add_argument("--factors", type=float, nargs=3, default=(1, 1, 1),
                    help="NEST basis (T and p) / the equilibrium's L, M, N")
    ap.add_argument("--colloc", type=float, default=1.5)
    ap.add_argument("--frac", type=float, default=40, help="annulus spacing h = a / frac")
    ap.add_argument("--N-phi", type=int, default=None, help="planes (default: arc spacing h)")
    ap.add_argument("--pad", type=float, default=0.02)
    ap.add_argument("--mode-chunk", type=int, default=16)
    ap.add_argument("--rblock", type=int, default=32)
    ap.add_argument("--interp-order", type=int, default=8)
    ap.add_argument("--xi-M", type=int, default=16)
    ap.add_argument("--xi-N", type=int, default=12)
    ap.add_argument("--method", default="gltr", choices=["gltr", "qr"])
    ap.add_argument("--alphas", type=float, nargs="+", default=[0.7])
    ap.add_argument("--beta", type=float, default=4.53e-3, help="B.n / |B| rms bound (DESC's)")
    ap.add_argument("--max-disp", type=float, default=2e-3, help="re-anchor beyond this [m]")
    ap.add_argument("--rho0", type=float, default=1.0)
    ap.add_argument("--band", type=float, default=0.05,
                    help="the step model keeps a bound's penalty curvature within this of it")
    ap.add_argument("--maxiter", type=int, default=40, help="trust-region steps per inner solve")
    ap.add_argument("--max-outer", type=int, default=25)
    ap.add_argument("--gtol", type=float, default=1e-3)
    ap.add_argument("--cg", type=int, default=200, help="GLTR: Lanczos iterations at most")
    ap.add_argument("--stat-tol", type=float, default=1e-3,
                    help="outer convergence also needs |J - J_fit|^2 flat to this (relative) over "
                         "the last 10 accepted inner steps, or the inner solve's own stop")
    ap.add_argument("--retries", type=int, default=3,
                    help="a bound not met: go to the geometric mean of it and the last one met, then "
                         "retry it; at most this many retries in the sweep, then stop")
    ap.add_argument("--cg-tol", type=float, default=1e-2)
    ap.add_argument("--chunk", type=int, default=64, help="unit vectors per batch (columns)")
    ap.add_argument("--hessian-k", type=int, default=40, help="Lanczos steps (0: none)")
    ap.add_argument("--kappa", type=float, default=None,
                    help="metric weight of xi (default: probed so the columns match u's)")
    ap.add_argument("--psi-target", default="start", choices=["start", "desc"])
    ap.add_argument("--out", default=os.path.join(ROOT, "results", "meanforce", "run"))
    a = ap.parse_args()
    os.makedirs(os.path.dirname(a.out) or ".", exist_ok=True)
    t0 = time.time()
    run = {"args": vars(a), "gates": {}, "alphas": [], "anchors": []}

    def dump():
        with open(a.out + ".json", "w") as fh:
            json.dump(run, fh, indent=1, default=float)

    inp0 = DESCInput.load(a.eq, field=a.coils)
    eq0 = inp0.eq
    cur0 = fit_cache.fit(BoundaryPotential, inp0, factors=tuple(a.factors))
    pre0 = fit_cache.fit(Pressure, inp0, factors=tuple(a.factors), edge_order=2)
    b = cur0.basis
    res = tuple(int(round(a.colloc * (u + v))) for u, v in
                ((b.L, eq0.L), (b.M, eq0.M), (b.N, eq0.N)))
    ref = reference(inp0, res, current=cur0)
    norm = ref["norm"]
    M_desc = ref["F_desc_vol"] / norm
    print(f"{inp0}\nbasis {b.L}/{b.M}/{b.N}, collocation resolution {res} "
          f"{tuple(ref['shape'])}; DESC <|F|>_vol / norm {M_desc:.4e}; {time.time() - t0:.0f} s",
          flush=True)

    # gate: DESC's own |F| integrated on NEST's collocation points (plain volume integral)
    fr_r = ForceResidual(inp0, None, resolution=res)
    tg = fr_r.tgrid
    dF = inp0.compute_tensor(["|F|"], tg.rho, tg.theta, tg.zeta)
    Fd = np.asarray(dF["|F|"]).ravel()
    M_pts = float(np.sum(fr_r.dV * Fd) / np.sum(fr_r.dV)) / norm
    run["gates"]["DESC mean on NEST points"] = {"NEST points": M_pts, "DESC quadrature": M_desc,
                                                "rel. diff": M_pts / M_desc - 1}
    print(f"gate: DESC's |F| on NEST's points {M_pts:.4e} vs DESC's quadrature {M_desc:.4e} "
          f"({100 * (M_pts / M_desc - 1):+.2f}%)", flush=True)
    del fr_r, dF

    grid0 = annulus(inp0, inp0.a / a.frac, a.pad, a.N_phi)
    state = {"solver": None, "eq": eq0, "k": 0}

    def make(eq_k, x_u):
        t1 = time.time()
        inp = DESCInput(eq_k, inp0.field, name=f"anchor {state['k']}", check=False)
        box = CylindricalGrid(grid0.r_axis, grid0.z_axis, grid0.NFP, grid0.N_phi)
        box.locate_plasma_planes(inp, sym=bool(eq0.sym), verbose=state["k"] == 0)
        fr = ForceResidual(inp, box, resolution=res)
        fr.norm, fr.F_desc_vol = norm, ref["F_desc_vol"]
        cur, pre = rebase(cur0, pre0, eq_k, np.concatenate([np.asarray(x_u), pre0.c]))
        disp = Displacement(eq_k, a.xi_M, a.xi_N)
        fields = FreeBoundaryFields(fr, cur, pre, disp, ref, dtype=jnp.float64, verbose=state["k"] == 0,
                                    cyl_stencil=True, cyl_solver=state["solver"],
                                    mode_chunk=a.mode_chunk, cyl_fused=True, cyl_rblock=a.rblock,
                                    cyl_order=a.interp_order or None)
        state["solver"] = fields.solver
        A = mf.Anchor(fields, cur0.u, 1.0, norm, chunk=a.chunk)
        A.info = {"anchor": state["k"], "setup [s]": time.time() - t1}
        if a.method == "qr":
            A.info["tables [s]"] = A.build_tables(jnp.asarray(x_u))
        A.extra = (fields, fr, box, disp, inp, cur, pre)
        g = LinearGrid(rho=1.0, M=2 * eq_k.M, N=2 * eq_k.N, NFP=eq_k.NFP)
        A.Fb = np.asarray(disp.basis.evaluate(g))
        print(f"  anchor {state['k']}: {box}, n_u {A.n_u}, n_p {A.n_p} ({len(A.keep)} kept), "
              f"n_xi {A.n_xi}, points {A.n_pts:,}, boundary {A.n_b}; "
              f"{time.time() - t1:.0f} s, GPU peak {peak():.1f} GB", flush=True)
        return A

    A = make(eq0, cur0.u)
    n_u = A.n_u
    z0 = jnp.concatenate([jnp.asarray(cur0.u), jnp.zeros(A.n_xi)])
    s0 = A.state(z0)
    rmsJ = float(jnp.sqrt(jnp.sum(s0["sw"][:, None] ** 2 * s0["Jfit"] ** 2)))
    A.rmsJ = rmsJ
    s0 = A.state(z0)
    Psi_desc = float(eq0.Psi)
    Psi_t = float(s0["Psi"]) if a.psi_target == "start" else Psi_desc
    io0 = np.asarray(mf.iota_profile(A.fields, jnp.zeros(A.n_xi), s0["B"]))
    start = {"M (Poisson p)": float(s0["M"]), "M/M_desc": float(s0["M"]) / M_desc,
             "rms|F|": float(s0["rmsF"]), "Bn/|B| rms": float(s0["Bn_rms"]),
             "Bn/|B| rms (DESC)": 4.527e-3, "Psi": float(s0["Psi"]), "Psi DESC": Psi_desc,
             "rms J_fit": rmsJ, "galerkin": float(s0["galerkin"]),
             "iota": io0.tolist(), "rho": np.asarray(A.extra[1].tgrid.rho).tolist()}
    run["start"] = start
    print(f"start: M {start['M (Poisson p)']:.4e} = {start['M/M_desc']:.3f} M_DESC, rms|F| "
          f"{start['rms|F|']:.3e}, B.n/|B| rms {start['Bn/|B| rms']:.3e} (DESC 4.53e-3), Psi "
          f"{start['Psi']:.6f} (DESC {Psi_desc:.6f}), Galerkin {start['galerkin']:.1e}",
          flush=True)
    bn0 = boundary_check(A.extra[4], A.extra[2], A.extra[5], A.extra[6],
                         np.concatenate([np.asarray(cur0.u), np.asarray(s0["c"])]), "truncated")
    run["gates"]["B.n start, direct Biot-Savart"] = bn0
    print(f"gate: B.n/|B| of the start by direct Biot-Savart: rms {bn0['Bn/|B| rms']:.3e} "
          f"(area-weighted), max {bn0['Bn/|B| max']:.3e}; annulus {start['Bn/|B| rms']:.3e}",
          flush=True)
    dump()

    # the metric: H_J and the boundary displacement's rms, weighted by kappa
    Fb = A.Fb
    G_xi = Fb.T @ Fb / Fb.shape[0]
    A_u = A.A_table()
    L1, HJ = mf.metric(A_u, np.asarray(s0["sw"]), rmsJ, G_xi, 1.0)
    W = mf.Weights(a.alphas[0] * M_desc, a.beta, Psi_t, rho=a.rho0, band=a.band)
    if a.kappa is None:  # mean diag of J^T J in the metric variables: xi's equal to u's
        rows = mf.Rows(A, s0, W, norm)
        rng = np.random.default_rng(0)
        Lu, Lx = L1[:n_u, :n_u], L1[n_u:, n_u:]
        su = sx = 0.0
        for _ in range(16):
            d = np.zeros(A.n)
            d[:n_u] = np.linalg.solve(Lu.T, rng.normal(size=n_u))
            su += float(jnp.sum(rows.J(jnp.asarray(d)) ** 2)) / n_u / 16
            d = np.zeros(A.n)
            d[n_u:] = np.linalg.solve(Lx.T, rng.normal(size=A.n_xi))
            sx += float(jnp.sum(rows.J(jnp.asarray(d)) ** 2)) / A.n_xi / 16
        kappa = sx / su
        del rows
    else:
        kappa = a.kappa
    L, _ = mf.metric(A_u, np.asarray(s0["sw"]), rmsJ, G_xi, kappa)
    met = pw.Metric(L)
    run["metric"] = {"kappa": kappa, "HJ diag mean": float(np.mean(np.diag(HJ)))}
    print(f"metric: kappa {kappa:.3e} (|y| = rms dJ / rms J_fit (+) sqrt(kappa) rms xi [m])",
          flush=True)
    del A_u

    def dmax_of(Aa):
        return lambda xi: float(np.abs(Aa.Fb @ np.asarray(xi)).max())

    def reanchor(z, Aold):
        xi = np.asarray(z[n_u:])
        s = Aold.state(z)
        io = np.asarray(mf.iota_profile(Aold.fields, jnp.asarray(xi), s["B"]))
        disp = Aold.extra[3]
        eq_new = disp.apply(xi)
        rec = {"anchor": state["k"], "max|D| [m]": dmax_of(Aold)(xi), "M before": float(s["M"]),
               "Bn before": float(s["Bn_rms"]), "Psi before": float(s["Psi"]), "iota": io.tolist()}
        release(*Aold.extra[:3])
        Aold.tables = None
        del Aold.extra, s
        jax.clear_caches()
        gc.collect()
        state["k"] += 1
        state["eq"] = eq_new
        An = make(eq_new, z[:n_u])
        An.rmsJ = rmsJ
        zn = jnp.concatenate([z[:n_u], jnp.zeros(An.n_xi)])
        sn = An.state(zn)
        rec.update({"M after": float(sn["M"]), "Bn after": float(sn["Bn_rms"]),
                    "Psi after": float(sn["Psi"]), **An.info})
        run["anchors"].append(rec)
        solver.dmax = dmax_of(An)
        print(f"  re-anchor {state['k']}: |D| {rec['max|D| [m]'] * 1e3:.2f} mm; M {rec['M before']:.4e}"
              f" -> {rec['M after']:.4e}, B.n {rec['Bn before']:.3e} -> {rec['Bn after']:.3e}, "
              f"Psi {rec['Psi before']:.6f} -> {rec['Psi after']:.6f}", flush=True)
        dump()
        return An, zn

    name = os.path.basename(a.out)
    solver = mf.Solver(A, met, W, norm, step=a.method, dmax=dmax_of(A), max_disp=a.max_disp,
                       reanchor=reanchor, cg_tol=a.cg_tol, cg_maxiter=a.cg,
                       stall_dir=os.path.dirname(a.out) or ".", run_name=name)
    z = z0
    queue, last_ok, retries = list(a.alphas), None, 0
    while queue:
        alpha = queue.pop(0)
        t1 = time.time()
        W.eps = alpha * M_desc
        print(f"alpha {alpha}: eps {W.eps:.4e}", flush=True)
        entry = {"alpha": alpha, "eps": W.eps}
        run["alphas"].append(entry)

        def cb(z, s, outer, entry=entry):
            entry["outer"] = outer
            entry["time"] = time.time() - t1
            entry.pop("inner (current outer)", None)
            dump()

        def on_step(hist, entry=entry):
            entry["inner (current outer)"] = hist
            dump()

        solver.on_step = on_step

        z, outer = solver.solve_alpha(z, max_outer=a.max_outer, maxiter=a.maxiter, gtol=a.gtol,
                                      callback=cb, stat_tol=a.stat_tol)
        A = solver.A
        s = A.state(z)
        f, phi, c = W.merit(s)
        xi = np.asarray(z[n_u:])
        io = np.asarray(mf.iota_profile(A.fields, jnp.asarray(xi), s["B"]))
        entry.update({"outer": outer, "time": time.time() - t1, "phi": phi, "c": c.tolist(),
                      "M": float(s["M"]), "M/M_desc": float(s["M"]) / M_desc,
                      "rms|F|": float(s["rmsF"]), "Bn/|B| rms": float(s["Bn_rms"]),
                      "Psi": float(s["Psi"]), "nu": W.nu.tolist(), "rho": W.rho,
                      "|J-J_fit|/rms": float(np.sqrt(2 * phi)), "iota": io.tolist(),
                      "max|D| in anchor [m]": dmax_of(A)(xi), "anchors": state["k"],
                      "stalls": solver.stalls, "galerkin": float(s["galerkin"]),
                      "slope dPhi/deps": -W.nu[0] / W.eps,
                      "feasible": bool(c[0] <= 1e-2 and c[1] <= 2e-2 and abs(c[2]) <= 1e-2),
                      "stationary": bool(outer[-1]["stationary"])})
        eq_f = A.extra[3].apply(xi)
        tag = f"{a.out}_a{alpha:g}"
        eq_f.save(tag + "_eq.h5")
        np.savez(tag + ".npz", z=np.asarray(z), u=np.asarray(z[:n_u]), xi=xi,
                 c_p=np.asarray(s["c"]), nu=W.nu, rho=W.rho, eps=W.eps, anchor=state["k"])
        # B.n by direct Biot-Savart on the final geometry (the annulus' check, plan section 5)
        inp_f = DESCInput(eq_f, inp0.field, name="final", check=False)
        cur_f, pre_f = rebase(cur0, pre0, eq_f, np.concatenate([np.asarray(z[:n_u]),
                                                                np.asarray(s["c"])]))
        bn = boundary_check(inp_f, A.extra[2], cur_f, pre_f,
                            np.concatenate([np.asarray(z[:n_u]), np.asarray(s["c"])]),
                            "truncated")
        entry["B.n direct"] = bn
        entry["B.n annulus - direct"] = float(s["Bn_rms"]) - bn["Bn/|B| rms"]
        print(f"alpha {alpha} done in {time.time() - t1:.0f} s: |J - J_fit| / rms "
              f"{entry['|J-J_fit|/rms']:.4e}, M/M_desc {entry['M/M_desc']:.4f}, B.n/|B| rms "
              f"{entry['Bn/|B| rms']:.3e} (direct {bn['Bn/|B| rms']:.3e}), Psi/Psi_t "
              f"{1 + c[2]:.5f}, nu {W.nu}, rho {W.rho:.1e}, anchors {state['k']}", flush=True)
        dump()
        if a.hessian_k > 0 and entry["feasible"]:
            t2 = time.time()
            entry["hessian"] = mf.hessian_lanczos(A, W, z, met, norm, k=a.hessian_k)
            entry["hessian"]["time"] = time.time() - t2
            dump()
        if entry["feasible"]:
            last_ok = alpha
        else:
            print(f"alpha {alpha} not met (c = {c})", flush=True)
            if last_ok is None or retries >= a.retries:
                print("sweep stopped", flush=True)
                break
            retries += 1
            # a smaller step from the last bound met, then the missed bound again, then on
            queue = [float(np.sqrt(last_ok * alpha)), alpha] + queue
            print(f"  retry at alpha {queue[0]:.4g}", flush=True)
    print(f"saved {a.out}.json; total {time.time() - t0:.0f} s, GPU peak {peak():.1f} GB",
          flush=True)


if __name__ == "__main__":
    main()
