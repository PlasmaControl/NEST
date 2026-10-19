"""Where does an optimized x lose on the check grid: collocation or J on the box?

For x0 and the optimized x of ``optimize_qa.py`` (``--out`` npz), <|F|> / norm
by radius on the collocation grid and on the finer check grid, each with
  interp   J on the box interpolated from the flux grid (as optimized; ``Fields``)
  fine     same, flux grid 1.5 x finer in each direction
  exact    J evaluated exactly at the box points (``ForceResidual.evaluate``)

run (GPU): sbatch --mem=64G run.sbatch scripts/stellarator/diagnose_opt.py results/optA4 [--colloc 4] [--check 6]
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
from flux_interp import TensorGrid  # noqa: E402
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
    ap.add_argument("out")
    ap.add_argument("--frac", type=int, default=10)
    ap.add_argument("--colloc", type=float, default=4)
    ap.add_argument("--check", type=float, default=6)
    ap.add_argument("--factors", type=float, nargs=3, default=(3, 3, 2.5))
    a = ap.parse_args()
    d = np.load(a.out + ".npz")
    inp = DESCInput.load(EQ, field=COILS)
    eq = inp.eq
    cur = fit_cache.fit(BoundaryPotential, inp, factors=tuple(a.factors))
    pre = fit_cache.fit(Pressure, inp, factors=tuple(a.factors), edge_order=2)
    box = CartesianGrid.periodic_box(inp, inp.a / a.frac, odd=bool(eq.sym)).locate_plasma(inp)
    res = lambda f: tuple(int(round(f * v)) for v in (eq.L, eq.M, eq.N))
    b = cur.basis
    fine = TensorGrid.uniform(6 * b.L, 2 * (9 * b.M // 2), 6 * b.N, eq.NFP, sym=bool(eq.sym))
    rows = {}
    for gname, f in (("colloc", a.colloc), ("check", a.check)):
        fr = ForceResidual(inp, box, resolution=res(f))
        fields = {"interp": Fields(fr, cur, pre)}
        for name in ("x0", "x"):
            x = d[name]
            u, c_p = fields["interp"].split(x)
            cur_x = cur.with_u(np.asarray(u))
            pre_x = Pressure(pre.basis, np.asarray(c_p), eq, pre.edge_order)
            rows[(gname, name, "interp")] = fr.metrics(np.asarray(fields["interp"].compute(x)["F"]))
            rows[(gname, name, "exact")] = fr.metrics(fr.evaluate(cur_x, pre_x)["F"])
        del fields
        ff = Fields(fr, cur, pre, flux_grid=fine)
        for name in ("x0", "x"):
            rows[(gname, name, "fine")] = fr.metrics(np.asarray(ff.compute(d[name])["F"]))
        del ff
        print(f"{gname} grid {fr.tgrid.shape} done; DESC {fr.F_desc_vol / fr.norm:.2e}", flush=True)
    bins = list(next(iter(rows.values()))["bins"])
    print(f"<|F|> / norm by rho (flux grid fine {fine.shape})")
    print(f"{'':24s}" + "".join(f"{k:>10s}" for k in bins + ["all"]))
    for key, m in rows.items():
        print(f"{' '.join(key):24s}" + "".join(f"{m['bins'][k]:10.2e}" for k in bins)
              + f"{m['mean']:10.2e}")


if __name__ == "__main__":
    main()
