"""NEST's force error at x0 against DESC's, on NEST's collocation grid.

At the start x0 (the BoundaryPotential and Pressure fits of DESC), with
B_ext = B_DESC - B_plasma(x0) on the box, so that B(x0) = B_DESC at the points:

  F_NEST = J_fit x B_DESC - grad p_fit,    F_DESC = J_DESC x B_DESC - grad p_DESC,

both on the same collocation points and weights. With that B_ext, F_NEST at x0
does not depend on the box, so none is built (``--frac``: build one and use B_ext
+ B_plasma(x0) anyway, as a check). F_NEST is split as
(``ForceResidual.decompose``) F_DESC + (J_fit - J_DESC) x B_DESC
- (grad p_fit - grad p_DESC) (+ the field term, zero here). <|F|> / norm overall
and by rho, and the relative rms errors of the fits of J and p by rho.

run: sbatch run_cpu.sbatch scripts/gs_x0_force.py [--factors 2 2 1] [--fit-target T|J]
    (a few minutes; with --frac 20 the box adds ~10 min)
"""

import argparse
import json
import os
import sys
import time

import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, "nest"))

from desc import set_device  # noqa: E402

set_device(os.environ.get("NEST_DEVICE", "cpu"))

import jax  # noqa: E402

import fit_cache  # noqa: E402
from boundary_potential import BoundaryPotential  # noqa: E402
from desc_input import DESCInput  # noqa: E402
from force import ForceResidual  # noqa: E402
from grid import CartesianGrid  # noqa: E402
from cartesian_grid import cylindrical_to_cartesian  # noqa: E402
from objective import Fields  # noqa: E402
from pressure import Pressure  # noqa: E402

jax.config.update("jax_enable_x64", True)
TOK = os.path.join(ROOT, "desc_input", "tokamak_free_boundary", "tokamak_L16M16")
BINS = (0.0, 0.5, 0.8, 0.9, 0.95, 1.0)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--eq", default=TOK + ".h5")
    ap.add_argument("--coils", default=TOK + "_field.h5")
    ap.add_argument("--factors", type=float, nargs=3, default=(2, 2, 1))
    ap.add_argument("--frac", type=int, default=None,
                    help="also build the box h = a/frac (not needed: B(x0) = B_DESC)")
    ap.add_argument("--colloc", type=float, default=2)
    ap.add_argument("--kernel", default="truncated", choices=["hockney", "truncated"])
    ap.add_argument("--fit-target", default="T", choices=["T", "J"],
                    help="fit DESC's potential T or its current J (BoundaryPotential.fit)")
    ap.add_argument("--edge-weights", type=float, nargs="+", default=None,
                    help="also fit with the rows at rho > --edge-rho weighted by each of "
                         "these (no box), and print their force by rho")
    ap.add_argument("--edge-rho", type=float, default=0.95)
    ap.add_argument("--out", default=os.path.join(ROOT, "results", "gs", "x0_force_f2_a20"))
    a = ap.parse_args()
    inp = DESCInput.load(a.eq, field=a.coils)
    eq = inp.eq
    t_fit = time.time()
    cur = fit_cache.fit(BoundaryPotential, inp, factors=tuple(a.factors), target=a.fit_target)
    pre = fit_cache.fit(Pressure, inp, factors=tuple(a.factors), edge_order=2)
    t_fit = time.time() - t_fit
    b = cur.basis
    res = tuple(int(round(a.colloc * (u + v))) for u, v in ((b.L, eq.L), (b.M, eq.M), (b.N, eq.N)))
    t0 = time.time()
    desc = None
    if a.frac is None:  # B = B_DESC at the points; J and grad p of the fits pointwise
        fr = ForceResidual(inp, None, resolution=res)
        desc = {k: np.asarray(v) for k, v in fr.desc_fields().items()}
        J = cylindrical_to_cartesian(fr._series(cur.point_evaluator(eq), fr.rtz), fr.phi)
        pg = fr._series(pre.point_evaluator(eq), fr.rtz)
        gp = cylindrical_to_cartesian(pg[:, 1:], fr.phi)
        out = {"J": np.asarray(J), "B": desc["B"], "grad_p": np.asarray(gp),
               "p": np.asarray(pg[:, 0])}
        out["F"] = np.cross(out["J"], out["B"]) - out["grad_p"]
        where = "no box (B = B_DESC)"
    else:
        box = CartesianGrid.periodic_box(inp, inp.a / a.frac, odd=bool(eq.sym)).locate_plasma(inp)
        fr = ForceResidual(inp, box, resolution=res)
        fields = Fields(fr, cur, pre, kernel=a.kernel)
        # the same B_ext as optimize_qa.py --bext desc
        fields.B_ext = fields.desc["B"] - fields.compute(fields.x0)["B_plasma"]
        out = {k: np.asarray(v) for k, v in fields.compute(fields.x0).items()}
        where = f"box h = a/{a.frac} {box.shape}"
    desc = desc if desc is not None else {k: np.asarray(v) for k, v in fr.desc_fields().items()}
    parts, resid, _ = fr.decompose(out, desc)
    print(f"{inp}\npotential L, M, N = {b.L}, {b.M}, {b.N} ({cur.n_free} unknowns), pressure "
          f"{pre.basis.L}, {pre.basis.M}, {pre.basis.N}; collocation {fr.tgrid.shape} "
          f"({len(fr.rtz):,} points); {where}; fit of {a.fit_target}; fits {t_fit:.0f} s, "
          f"fields {time.time() - t0:.0f} s")
    rms = lambda v: float(np.sqrt(np.sum(fr.dV * np.sum(np.reshape(v, (len(fr.dV), -1)) ** 2, 1))  # noqa: E731
                                  / np.sum(fr.dV)))
    print(f"B(x0) - B_DESC at the points: {rms(out['B'] - desc['B']) / rms(desc['B']):.1e} (rms, "
          f"relative)\n")

    rows = {"NEST F at x0": fr.metrics(out["F"])}
    rows.update({k: fr.metrics(v) for k, v in parts.items()})
    rows["(decomposition residual)"] = fr.metrics(resid)
    head = f"{'<|F|> / norm':42s} {'all':>9s} " + " ".join(f"{f'{x:g}-{y:g}':>9s}"
                                                          for x, y in zip(BINS[:-1], BINS[1:]))
    print(head)
    for k, m in rows.items():
        print(f"{k:42s} {m['mean']:9.2e} " + " ".join(f"{v:9.2e}" for v in m["bins"].values()))
    print(f"{'DESC on its own quadrature grid':42s} {fr.F_desc_vol / fr.norm:9.2e}")

    # the fits themselves: relative rms error of J and p by rho
    rho = fr.rtz[:, 0]
    fit = {}
    for k, nest, ref in (("J", out["J"], desc["J"]), ("p", out["p"], desc["p"])):
        r_all = rms(ref)
        fit[k] = {"all": rms(nest - ref) / r_all}
        for x, y in zip(BINS[:-1], BINS[1:]):
            m = (rho > x) & (rho <= y)
            e = np.reshape(nest - ref, (len(rho), -1))[m]
            fit[k][f"{x:g}-{y:g}"] = float(np.sqrt(np.sum(fr.dV[m] * np.sum(e**2, 1)) /
                                                   np.sum(fr.dV[m]))) / r_all
    print("\nfit error, rms |X_fit - X_DESC| (in the bin) / rms |X_DESC| (whole plasma):")
    for k, v in fit.items():
        print(f"{k:42s} " + " ".join(f"{x:9.2e}" for x in v.values()))
    scan = {}
    if a.edge_weights:  # the same comparison for edge-weighted fits (B = B_DESC, pointwise)
        print(f"\nedge-weighted fits (rows at rho > {a.edge_rho} x w): <|F|> / norm at x0 and "
              f"rms J error / rms J_DESC by rho")
        print(head)
        Bd = desc["B"]
        for w_e in a.edge_weights:
            t1 = time.time()
            c = BoundaryPotential.fit(eq, factors=tuple(a.factors), target=a.fit_target,
                                      edge_weight=w_e, edge_rho=a.edge_rho)
            Jw = np.asarray(cylindrical_to_cartesian(fr._series(c.point_evaluator(eq), fr.rtz),
                                                     fr.phi))
            m = fr.metrics(np.cross(Jw, Bd) - out["grad_p"])
            jerr = {}
            for x, y in zip(BINS[:-1], BINS[1:]):
                k = (rho > x) & (rho <= y)
                e = Jw[k] - desc["J"][k]
                jerr[f"{x:g}-{y:g}"] = float(np.sqrt(np.sum(fr.dV[k] * np.sum(e**2, 1)) /
                                                     np.sum(fr.dV[k]))) / rms(desc["J"])
            scan[w_e] = {"force": m, "J": jerr}
            print(f"{'w = %g: F' % w_e:42s} {m['mean']:9.2e} "
                  + " ".join(f"{v:9.2e}" for v in m["bins"].values())
                  + f"   ({time.time() - t1:.0f} s)")
            print(f"{'        J error':42s} {'':9s} " + " ".join(f"{v:9.2e}" for v in jerr.values()))
    with open(a.out + ".json", "w") as fo:
        json.dump({"args": vars(a), "force": rows, "fit": fit, "edge_scan": scan,
                   "desc_own_grid": fr.F_desc_vol / fr.norm}, fo, indent=1)
    print(f"saved {a.out}.json")


if __name__ == "__main__":
    main()
