"""Diagnostics of fixed-boundary mean-force states of the perturbed DSHAPE tokamak (mf_fixed.py with
--pert-eps), rebuilt from the run's own JSON record (same equilibrium, coils, perturbation, basis,
pressure basis, annulus and collocation grid).

For the start (J_fit) and each state:

  1. force and parallel pressure gradient at the collocation points, EXACT (p and grad p from the
     pressure basis, B = B_ext + eps B_pert + B_plasma(u) from the run's annulus solve; never by
     finite differences of p): <|F|>/norm, <|b . grad p|> / (p(0) / a) and <|b . grad p|> / <|grad p|>
     in rho bands, with fine bands around q = 2;
  2. Poincare section at phi = 0 (field table on (R, phi, Z), RK4; poincare.py's tracer), lines seeded
     across a band around q = 2 and uniformly elsewhere; crossings mapped to DESC's (rho, theta*);
  3. the 2/1 island width (rho extent of the lines trapped in the chain: their 2 theta* stays within
     2 pi when unwrapped), and the pressure through the island's O- and X-points (flattening: the
     drop of p across the island along the O-point line relative to the X-point line).

Writes <out>.json (numbers) and <out>_sections.npz (rho, theta* per line, trapped flags).

    python scripts/meanforce/dshape_fixed_diag.py --run out/base.json \
        --state a0.1=out/base_a0.1.npz --state a0.01=out/base_a0.01.npz --out out/base_diag
"""

import argparse
import json
import os
import sys
import time

import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.join(ROOT, "nest"))
sys.path.insert(0, os.path.join(ROOT, "scripts"))
sys.path.insert(0, os.path.join(ROOT, "scripts", "meanforce"))
sys.path.insert(0, os.path.join(ROOT, "desc_input", "vmec_free_boundary"))
from desc import set_device  # noqa: E402

set_device(os.environ.get("NEST_DEVICE", "cpu"))
import jax  # noqa: E402
import jax.numpy as jnp  # noqa: E402

jax.config.update("jax_enable_x64", True)

import fit_cache  # noqa: E402
import poincare as pc  # noqa: E402
from boundary_potential import BoundaryPotential  # noqa: E402
from desc_input import DESCInput  # noqa: E402
from force import ForceResidual  # noqa: E402
from free_boundary import _chunked, reference  # noqa: E402
from grid import map_to_flux  # noqa: E402
from objective import Fields, _cartesian  # noqa: E402
from optimize_fb import annulus  # noqa: E402
from perturbation import Perturbation  # noqa: E402
from poincare_fixed import table  # noqa: E402
from pressure import Pressure  # noqa: E402
from separable import Separable  # noqa: E402
from spline_pressure import SplinePressure, rational_centres, trunc_kw  # noqa: E402
from island_width import width  # noqa: E402
from resonant_profiles import peak_width, resonant_profiles, resonant_setup  # noqa: E402


class PerturbedInput:
    """``DESCInput`` whose B_ext includes eps B_pert (for poincare_fixed.table)."""

    def __init__(self, inp, pert, eps):
        self.inp, self.pert, self.eps = inp, pert, eps

    def B_ext(self, x):
        b = self.inp.B_ext(x)
        return b if not self.eps else b + self.eps * self.pert(x)


def setup(ra, frac=None):
    """The run's equilibrium, fits, collocation grid and Fields (as mf_fixed.py builds them)."""
    inp = DESCInput.load(ra["eq"], field=ra["coils"])
    eq = inp.eq
    kw = ({"L": ra["LMN"][0], "M": ra["LMN"][1], "N": ra["LMN"][2]} if ra.get("LMN") else
          {"factors": tuple(ra["factors"])})
    if ra.get("current", "fz") == "spline":
        cur = fit_cache.fit(BoundaryPotential, inp, M=ra["LMN"][1], N=ra["LMN"][2], radial="spline",
                            knots=tuple(ra["j_knots"]), helicities=tuple(ra["p_rationals"]),
                            **trunc_kw(ra.get("spl_mtol", 0.0), ra.get("spl_mmax"), ra.get("spl_mfz"),
                                      ra.get("knots_fz")))
    else:
        cur = fit_cache.fit(BoundaryPotential, inp, **kw)
    if ra.get("pressure", "fz") == "spline":
        pM, pN = ra["p_MN"] if ra.get("p_MN") else (cur.basis.M, cur.basis.N)
        pre = fit_cache.fit(SplinePressure, inp, M=pM, N=pN, edge_order=2, coarse=ra["p_knots"][0],
                            dense=ra["p_knots"][1], half=ra["p_knots"][2], helicities=tuple(ra["p_rationals"]),
                            **trunc_kw(ra.get("spl_mtol", 0.0), ra.get("spl_mmax"), ra.get("spl_mfz"),
                                      ra.get("knots_fz")))
    else:
        pre = fit_cache.fit(Pressure, inp, edge_order=2, **kw)
    b = cur.basis
    res = tuple(int(round(ra["colloc"] * (u + v))) for u, v in ((b.L, eq.L), (b.M, eq.M), (b.N, eq.N)))
    ref = reference(inp, res, current=cur)
    box = annulus(inp, inp.a / (frac or ra["frac"]), ra["pad"], ra.get("N_phi"))
    box.locate_plasma_planes(inp, sym=bool(eq.sym), verbose=False)
    rho_nodes = None
    bps = (([pre.bp_rho] if ra.get("pressure", "fz") == "spline" else [])
           + ([cur.basis.bp_rho] if ra.get("current", "fz") == "spline" else []))
    if bps:
        from spline_pressure import composite_gauss
        rho_nodes = composite_gauss(np.unique(np.concatenate(bps)), ra.get("p_colloc", 4), min_nodes=res[0] // 2 + 1)
    fr = ForceResidual(inp, box, resolution=res, rho_nodes=rho_nodes)
    fr.norm, fr.F_desc_vol = ref["norm"], ref["F_desc_vol"]
    if "B_ext" not in vars(fr):
        fr.B_ext = _chunked(inp.B_ext, fr.x)
    pert = Perturbation(inp.field, *ra.get("pert_mn", (2, -1)))
    eps = float(ra.get("pert_eps", 0.0) or 0.0)
    if eps:
        fr.B_ext = np.asarray(fr.B_ext) + eps * np.asarray(pert(np.asarray(fr.x)))
    fields = Fields(fr, cur, pre, dtype=jnp.float64, cyl_stencil=True, mode_chunk=ra.get("mode_chunk", 16),
                    cyl_fused=True, cyl_rblock=ra.get("rblock", 32), cyl_order=ra.get("interp_order", 8) or None)
    return inp, cur, pre, fr, fields, ref, pert, eps


def band_stats(rho, w, vals, edges):
    out = []
    for lo, hi in zip(edges[:-1], edges[1:]):
        m = (rho >= lo) & (rho < hi)
        if not m.any():
            continue
        out.append({"rho": [float(lo), float(hi)], "volume": float(w[m].sum()),
                    **{k: float(np.sum(w[m] * v[m]) / w[m].sum()) for k, v in vals.items()}})
    return out


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--run", required=True, help="the mf_fixed.py run record (.json)")
    ap.add_argument("--state", action="append", default=[], help="name=path.npz (u, c_p); repeatable")
    ap.add_argument("--skip-start", action="store_true")
    ap.add_argument("--frac", type=float, default=None, help="annulus a/frac for the evaluation (default: the run's)")
    ap.add_argument("--band", type=float, default=0.12, help="seed band half width around q = 2")
    ap.add_argument("--lines", type=int, default=48, help="lines in the q = 2 band (x 2 start angles)")
    ap.add_argument("--uniform", type=int, default=16, help="extra lines spread over rho")
    ap.add_argument("--periods", type=int, default=400)
    ap.add_argument("--steps", type=int, default=128)
    ap.add_argument("--grid", type=int, nargs=3, default=(201, 64, 241), help="table nR nphi nZ")
    ap.add_argument("--no-poincare", action="store_true")
    ap.add_argument("--fine", action="store_true",
                    help="also evaluate J, B, grad p (the solver's own operators) on --fine-n radii across q = 2 +- "
                         "--fine-half, between the collocation radii: off-node force, s21 and b21 profiles")
    ap.add_argument("--fine-n", type=int, default=121)
    ap.add_argument("--fine-half", type=float, default=0.12)
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    t0 = time.time()
    ra = json.load(open(a.run))["args"]
    inp, cur, pre, fr, fields, ref, pert, eps = setup(ra, a.frac)
    eq = inp.eq
    rq = rational_centres(eq, ["2/1"])
    rs2 = rq[0] if rq else 0.6
    from desc.grid import LinearGrid
    p_axis = float(np.asarray(eq.compute("p", grid=LinearGrid(rho=np.array([0.0]), NFP=eq.NFP))["p"])[0])
    norm = ref["norm"]
    print(f"{inp}; eps {eps:g} A; q = 2 at rho {rq}; p(0) {p_axis:.4e} Pa; setup {time.time() - t0:.0f} s", flush=True)

    states = {} if a.skip_start else {"start": (np.asarray(cur.u), np.asarray(pre.c))}
    for spec in a.state:
        name, path = spec.split("=", 1) if "=" in spec else (os.path.basename(spec), spec)
        st = np.load(path)
        states[name] = (np.asarray(st["u"]), np.asarray(st["c_p"]) if "c_p" in st else np.asarray(pre.c))

    rho_pts = np.asarray(fr.rtz[:, 0])
    w = np.asarray(fields.sw) ** 2
    t1 = time.time()
    rs = resonant_setup(inp, fr)
    fine = None
    if a.fine:  # a second evaluation grid: fine radii across q = 2, the collocation theta / zeta
        tg = fr.tgrid
        rf = np.linspace(rs2 - a.fine_half, rs2 + a.fine_half, a.fine_n)
        res_f = (2, (len(tg.theta) - 1) // 2, (len(tg.zeta) - 1) if fr.sym else (len(tg.zeta) - 1) // 2)
        fr_f = ForceResidual(inp, fr.box, resolution=res_f, rho_nodes=(rf, np.full(len(rf), rf[1] - rf[0])))
        assert len(fr_f.tgrid.theta) == len(tg.theta) and len(fr_f.tgrid.zeta) == len(tg.zeta)
        fr_f.norm, fr_f.F_desc_vol = fr.norm, fr.F_desc_vol
        fr_f.B_ext = np.asarray(_chunked(inp.B_ext, fr_f.x))
        if eps:
            fr_f.B_ext = fr_f.B_ext + eps * np.asarray(pert(np.asarray(fr_f.x)))
        fields_f = Fields(fr_f, cur, pre, dtype=jnp.float64, cyl_stencil=True, mode_chunk=ra.get("mode_chunk", 16),
                          cyl_fused=True, cyl_rblock=ra.get("rblock", 32), cyl_order=ra.get("interp_order", 8) or None)
        fine = {"fr": fr_f, "fields": fields_f, "rs": resonant_setup(inp, fr_f), "rho": rf,
                "dV": np.asarray(fr_f.dV).reshape(len(rf), -1)}
        print(f"fine grid: {a.fine_n} radii in [{rf[0]:.3f}, {rf[-1]:.3f}] (spacing {rf[1] - rf[0]:.4f}; collocation "
              f"radii there: {int(np.sum(np.abs(np.asarray(tg.rho) - rs2) <= a.fine_half))}), {fr_f.x.shape[0]} points; "
              f"{time.time() - t1:.0f} s", flush=True)
    Bp_eps = eps * np.asarray(pert(np.asarray(fr.x))) if eps else None
    prof = {"rho": rs["rho"]}
    if fine is not None:
        prof["fine_rho"] = fine["rho"]
    drive = {}
    if Bp_eps is not None:  # the perturbation's own resonant field (the drive, no plasma response)
        Bref = np.asarray(fields.B_ext)  # coils + eps B_pert at the points
        d_all = resonant_profiles(rs, Bref)["b21"]
        d_coils = resonant_profiles(rs, Bref - Bp_eps)["b21"]
        prof["drive_b21"], prof["coils_b21"] = d_all - d_coils, d_coils
        drive = {"b21 at q2 (eps B_pert)": float(np.interp(rs2, rs["rho"], d_all - d_coils))}
    print(f"resonant setup (lambda, grad rho, drive) {time.time() - t1:.0f} s; drive {drive}", flush=True)
    edges = np.unique(np.concatenate([np.linspace(0, 1, 11), rs2 + np.linspace(-0.1, 0.1, 11)]).clip(0, 1))
    report = {"run": a.run, "eps": eps, "q2 rho": rq, "p(0)": p_axis, "a": float(inp.a), "states": {}}
    report["drive"] = drive

    # seeds for the Poincare section: a band around q = 2 at theta = 0 and pi/2, plus uniform
    RZ = Separable([eq.R_basis, eq.Z_basis], [eq.R_lmn, eq.Z_lmn], eq.NFP)
    rb = np.linspace(max(rs2 - a.band, 0.02), min(rs2 + a.band, 0.99), a.lines)
    seeds = np.concatenate([np.column_stack([rb, np.zeros_like(rb)]), np.column_stack([rb, np.full_like(rb, np.pi / 2)]),
                            np.column_stack([np.linspace(0.05, 0.98, a.uniform), np.zeros(a.uniform)])])
    st0 = np.asarray(RZ.evaluate(jnp.asarray(seeds[:, 0]), jnp.asarray(seeds[:, 1]), jnp.zeros(len(seeds))))
    ra_, za_ = fields.cyl_grid.r_axis, fields.cyl_grid.z_axis
    Rs = np.linspace(ra_.a + 1e-4, ra_.b - 1e-4, a.grid[0])
    Zs = np.linspace(za_.a + 1e-4, za_.b - 1e-4, a.grid[2])
    phis = 2 * np.pi / eq.NFP * np.arange(a.grid[1]) / a.grid[1]
    pinp = PerturbedInput(inp, pert, eps)
    sec_out = {}
    for name, (u, c) in states.items():
        t1 = time.time()
        uj, cj = jnp.asarray(u), jnp.asarray(c)
        J = np.asarray(_cartesian(fields.J_col(uj, fields.I).reshape(-1, 3), fields.phi))
        B = np.asarray(fields.B_ext + fields.B_plasma(uj))
        p, gp = fields.p_col(cj)
        gp = np.asarray(_cartesian(gp.reshape(-1, 3), fields.phi))
        F = np.cross(J, B) - gp
        absB = np.linalg.norm(B, axis=1)
        bgp = np.abs(np.sum(B * gp, 1)) / absB
        vals = {"|F|/norm": np.linalg.norm(F, axis=1) / norm, "|b.grad p|/(p0/a)": bgp / (p_axis / inp.a),
                "|grad p|/(p0/a)": np.linalg.norm(gp, axis=1) / (p_axis / inp.a)}
        rep = {"pressure": "the fitted DESC p (start)" if name == "start" else "the state's Galerkin p*",
               "M": float(np.sum(w * vals["|F|/norm"])), "<|b.grad p|>/(p0/a)": float(np.sum(w * vals["|b.grad p|/(p0/a)"])),
               "<|b.grad p|>/<|grad p|>": float(np.sum(w * bgp) / np.sum(w * np.linalg.norm(gp, axis=1))),
               "bands": band_stats(rho_pts, w, vals, edges)}
        near = np.abs(rho_pts - rs2) < 0.05
        rep["q2 band |b.grad p|/(p0/a)"] = float(np.sum(w[near] * vals["|b.grad p|/(p0/a)"][near]) / w[near].sum())
        rep["q2 band |F|/norm"] = float(np.sum(w[near] * vals["|F|/norm"][near]) / w[near].sum())
        rp = resonant_profiles(rs, B, J)
        prof[f"{name}_b21"], prof[f"{name}_s21"], prof[f"{name}_s00"] = rp["b21"], rp["s21"], rp["s00"]
        nr = len(rs["rho"])
        Fnode = (w * vals["|F|/norm"]).reshape(nr, -1).sum(1) / w.reshape(nr, -1).sum(1)  # <|F|>(rho) at the nodes
        prof[f"{name}_node_F"] = Fnode
        if fine is not None:
            ff = fine["fields"]
            Jf = np.asarray(_cartesian(ff.J_col(uj, ff.I).reshape(-1, 3), ff.phi))
            Bf = np.asarray(ff.B_ext + ff.B_plasma(uj))
            _, gpf = ff.p_col(cj)
            gpf = np.asarray(_cartesian(gpf.reshape(-1, 3), ff.phi))
            Ff = np.linalg.norm(np.cross(Jf, Bf) - gpf, axis=1).reshape(len(fine["rho"]), -1) / norm
            Ffine = (fine["dV"] * Ff).sum(1) / fine["dV"].sum(1)
            rpf = resonant_profiles(fine["rs"], Bf, Jf)
            prof[f"{name}_fine_F"], prof[f"{name}_fine_s21"], prof[f"{name}_fine_b21"] = Ffine, rpf["s21"], rpf["b21"]
            inb = np.abs(rs["rho"] - rs2) <= a.fine_half
            node_in = Fnode[inb]
            interp_nodes = np.interp(fine["rho"], rs["rho"], Fnode)  # what the nodes alone would suggest
            rep["fine"] = {
                "<|F|> fine band mean / node band mean": float(Ffine.mean() / node_in.mean()) if inb.any() else None,
                "max over fine radii of <|F|>fine / <|F|>interpolated from nodes": float(np.max(Ffine / interp_nodes)),
                "<|F|> fine at q2": float(np.interp(rs2, fine["rho"], Ffine)),
                "s21 fine peak": peak_width(fine["rho"], rpf["s21"], rs2, band=a.fine_half),
                "b21 fine at q2": float(np.interp(rs2, fine["rho"], rpf["b21"])),
                "max |s21| fine / max |s21| nodes (band)": float(np.abs(rpf["s21"]).max() / np.abs(rp["s21"][inb]).max())
                if inb.any() else None}
            print(f"{name}: fine grid: <|F|> fine/node band {rep['fine']['<|F|> fine band mean / node band mean']:.3f}, "
                  f"max fine/interp {rep['fine']['max over fine radii of <|F|>fine / <|F|>interpolated from nodes']:.3f}, "
                  f"s21 fine peak {rep['fine']['s21 fine peak']}", flush=True)
        b_q = float(np.interp(rs2, rs["rho"], rp["b21"]))
        rep["resonant"] = {"b21 at q2": b_q,
                           "b21 / drive at q2": b_q / drive["b21 at q2 (eps B_pert)"] if drive else None,
                           "s21 peak": peak_width(rs["rho"], rp["s21"], rs2),
                           "s21 at q2": float(np.interp(rs2, rs["rho"], rp["s21"])),
                           "radial nodes within 0.05 of q2": int(np.sum(np.abs(rs["rho"] - rs2) < 0.05))}
        print(f"{name}: b21(q=2) {b_q:+.3e}" + (f" ({rep['resonant']['b21 / drive at q2']:+.3f} x drive)" if drive else "")
              + f"; s21 peak {rep['resonant']['s21 peak']}", flush=True)
        print(f"{name}: M {rep['M']:.3e}, <|b.grad p|>/(p0/a) {rep['<|b.grad p|>/(p0/a)']:.3e} (q = 2 band "
              f"{rep['q2 band |b.grad p|/(p0/a)']:.3e}), <|b.grad p|>/<|grad p|> {rep['<|b.grad p|>/<|grad p|>']:.3e}; "
              f"{time.time() - t1:.0f} s", flush=True)
        if not a.no_poincare:
            t1 = time.time()
            Bt = table(fields, pinp, u, Rs, phis, Zs)
            Rc, Zc, _, _ = pc.trace(Rs, phis, Zs, Bt, st0[:, 0], st0[:, 1], eq.NFP, a.periods, a.steps)
            pts = np.column_stack([Rc.ravel(), Zc.ravel()])
            ok = np.all(np.isfinite(pts), 1)
            flux = np.full((len(pts), 3), np.nan)
            flux[ok] = np.asarray(map_to_flux(inp, np.column_stack([pts[ok, 0], np.zeros(ok.sum()), pts[ok, 1]])))
            inside = np.isfinite(flux[:, 0])
            lam = np.full(len(pts), np.nan)
            lam[inside] = np.asarray(inp.compute_points(["lambda"], flux[inside])["lambda"]).ravel()
            r = flux[:, 0].reshape(Rc.shape)
            ts = np.mod(flux[:, 1] + lam, 2 * np.pi).reshape(Rc.shape)
            wd, tr = width(r, ts, rho_c=rs2, band=0.06)
            rep["2/1 width (rho)"] = wd
            rep["trapped lines"] = int(tr.sum())
            rep["lost lines"] = int(np.isnan(Rc[-1]).sum())
            sec_out[f"{name}_rho"], sec_out[f"{name}_ts"], sec_out[f"{name}_trapped"] = r, ts, tr
            # p through the O- and X-points (flattening)
            if tr.any():
                al = 2 * ts[:, tr]
                thO = float(np.angle(np.nanmean(np.exp(1j * al))) / 2) % np.pi
                pev = jax.jit(pre.with_c(c).point_evaluator(eq))
                rr = np.linspace(max(rs2 - a.band, 0.02), min(rs2 + a.band, 0.99), 161)
                pox = {}
                for lab, th_ in (("O", thO), ("X", thO + np.pi / 2)):
                    q = np.column_stack([rr, np.full(len(rr), th_), np.zeros(len(rr))])
                    for _ in range(4):
                        lm = np.asarray(inp.compute_points(["lambda"], q)["lambda"]).ravel()
                        q[:, 1] = th_ - lm
                    pox[lab] = np.asarray(pev(*(jnp.asarray(q[:, i]) for i in range(3))))[:, 0]
                inside_w = np.abs(rr - np.mean(r[:, tr][np.isfinite(r[:, tr])])) < max(wd / 2, 1e-3)
                dO = float(np.ptp(pox["O"][inside_w])) if inside_w.any() else float("nan")
                dX = float(np.ptp(pox["X"][inside_w])) if inside_w.any() else float("nan")
                rep["p drop across the island: O / X line"] = dO / dX if dX else float("nan")
                rep["O-point theta*"] = thO
                sec_out[f"{name}_pO"], sec_out[f"{name}_pX"], sec_out["rr"] = pox["O"], pox["X"], rr
            print(f"{name}: 2/1 width {rep['2/1 width (rho)']:.4f} ({rep['trapped lines']} trapped, "
                  f"{rep['lost lines']} lost); flattening O/X {rep.get('p drop across the island: O / X line', float('nan')):.3f}; "
                  f"{time.time() - t1:.0f} s", flush=True)
        report["states"][name] = rep
        with open(a.out + ".json", "w") as fh:
            json.dump(report, fh, indent=1, default=float)
    np.savez(a.out + "_resonant.npz", **prof)
    if sec_out:
        np.savez(a.out + "_sections.npz", seeds=seeds, **sec_out)
    print(f"saved {a.out}.json; total {time.time() - t0:.0f} s")


if __name__ == "__main__":
    main()
