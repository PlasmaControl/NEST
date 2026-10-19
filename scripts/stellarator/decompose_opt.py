"""What the force residual of an optimized x is made of.

F = J x B - grad p splits along b = B / |B| into
  F_par  = -(b . grad p)        no current can change it: p not constant on field lines
  F_perp = F - F_par b          J x B - grad p_perp: what the current balances
each <|.|>_vol / norm, overall and by radius, at the start (the fits) and at
x of ``optimize_qa.py`` (--x npz), with the optimization's setup.

run (GPU): sbatch run.sbatch scripts/stellarator/decompose_opt.py --x results/optTrunc_iter.npz [--frac 10]
"""

import argparse
import os
import sys

import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.join(ROOT, "nest"))

from desc import set_device  # noqa: E402

set_device(os.environ.get("NEST_DEVICE", "cpu"))

import jax  # noqa: E402

import fit_cache  # noqa: E402
from boundary_potential import BoundaryPotential  # noqa: E402
from desc_input import DESCInput  # noqa: E402
from force import ForceResidual  # noqa: E402
from grid import CartesianGrid  # noqa: E402
from objective import Fields  # noqa: E402
from pressure import Pressure  # noqa: E402

jax.config.update("jax_enable_x64", True)
EQ = os.path.join(ROOT, "desc_input", "precise_QA_free_boundary_smooth_J",
                  "precise_QA_free_boundary_smooth_J_edgeJ_r0.3.h5")
COILS = os.path.join(ROOT, "desc_input", "precise_QA_free_boundary", "precise_QA_helical_coils.h5")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--x", required=True)
    ap.add_argument("--frac", type=int, default=10)
    ap.add_argument("--kernel", default="truncated")
    ap.add_argument("--colloc", type=float, default=2)
    ap.add_argument("--factors", type=float, nargs=3, default=(3, 3, 2.5))
    a = ap.parse_args()
    inp = DESCInput.load(EQ, field=COILS)
    eq = inp.eq
    cur = fit_cache.fit(BoundaryPotential, inp, factors=tuple(a.factors))
    pre = fit_cache.fit(Pressure, inp, factors=tuple(a.factors), edge_order=2)
    box = CartesianGrid.periodic_box(inp, inp.a / a.frac, odd=bool(eq.sym)).locate_plasma(inp)
    b = cur.basis
    res = tuple(int(round(a.colloc * (u + v))) for u, v in ((b.L, eq.L), (b.M, eq.M), (b.N, eq.N)))
    fr = ForceResidual(inp, box, resolution=res)
    fields = Fields(fr, cur, pre, kernel=a.kernel)
    x_opt = np.load(a.x)["x"]
    desc = fr.desc_fields()
    print(f"{inp}\nbox {box.shape}, collocation {fr.tgrid.shape}, kernel {a.kernel}; "
          f"DESC <|F|>/norm {fr.F_desc_vol / fr.norm:.2e}")
    rows = {}
    for name, x in (("start", fields.x0), ("optimized", x_opt)):
        f = {k: np.asarray(v) for k, v in fields.compute(x).items()}
        B, F, gp = f["B"], f["F"], f["grad_p"]
        bh = B / np.linalg.norm(B, axis=1)[:, None]
        par = np.sum(F * bh, 1)
        perp = F - par[:, None] * bh
        rows[name + ": F"] = fr.metrics(F)
        rows[name + ": F_par = -b.grad p"] = fr.metrics(par[:, None])
        rows[name + ": F_perp"] = fr.metrics(perp)
        # the same split with DESC's B: is p constant on DESC's own field lines?
        bd = desc["B"] / np.linalg.norm(desc["B"], axis=1)[:, None]
        rows[name + ": b_desc.grad p"] = fr.metrics(np.sum(gp * bd, 1)[:, None])
        # how far NEST's B is from DESC's
        dB = np.linalg.norm(B - desc["B"], axis=1) / np.linalg.norm(desc["B"], axis=1)
        rows[name + ": |B - B_desc| / |B_desc| (x norm)"] = fr.metrics(dB[:, None] * fr.norm)
    # how much the pressure moved (the optimization may free it)
    f0, f1 = (fields.compute(x) for x in (fields.x0, x_opt))
    p0, p1 = np.asarray(f0["p"]), np.asarray(f1["p"])
    w = fr.dV / fr.dV.sum()
    rms = lambda q: float(np.sqrt(np.sum(w * q**2)))
    print(f"pressure: <p> ratio {np.sum(w * p1) / np.sum(w * p0):.6f}, rms |p - p0| / rms p0 "
          f"{rms(p1 - p0) / rms(p0):.2e}, max |p - p0| / max p0 {np.abs(p1 - p0).max() / p0.max():.2e}, "
          f"p(axis-most point) {p1[np.argmin(fr.rtz[:, 0])] / p0[np.argmin(fr.rtz[:, 0])]:.4f} x p0")
    bins = list(next(iter(rows.values()))["bins"])
    print(f"{'<|.|>_vol / norm':44s}" + "".join(f"{k:>10s}" for k in bins + ["all"]))
    for k, m in rows.items():
        print(f"{k:44s}" + "".join(f"{m['bins'][q]:10.2e}" for q in bins) + f"{m['mean']:10.2e}")


if __name__ == "__main__":
    main()
