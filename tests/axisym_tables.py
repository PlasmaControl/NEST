"""axisym_field.plasma_field: the exact tables against the direct basis evaluation.

Same quadrature, so the two must agree to round-off.  For each run (its fitted basis and
final coefficients): targets from the axis to rho = 0.999, max |B_tables - B_direct| over
max |B_poloidal|, and the times; with --full, the tables' time on a 160 x 160 grid.

run: sbatch run.sbatch tests/axisym_tables.py --runs results/gs/cylD_a160 results/gs/cyl3D_a160 --full
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
from axisym_field import plasma_field  # noqa: E402
from boundary_potential import BoundaryPotential  # noqa: E402
from desc_input import DESCInput  # noqa: E402

jax.config.update("jax_enable_x64", True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--runs", nargs="+", required=True)
    ap.add_argument("--full", action="store_true")
    a = ap.parse_args()
    for run in a.runs:
        args = json.load(open(run + "_iter.json"))["args"]
        inp = DESCInput.load(args["eq"], field=args["coils"])
        ep = args.get("edge_power", 2)
        cur = fit_cache.fit(BoundaryPotential, inp, factors=tuple(args["factors"]),
                            **({"edge_power": ep} if ep != 2 else {}))
        u = np.load(run + "_iter.npz")["x"][: cur.n_free]
        rho = np.array([0.0, 0.05, 0.3, 0.6, 0.9, 0.97, 0.99, 0.999])
        th = 2 * np.pi * (np.arange(8) + 0.13) / 8
        R_, T_ = np.meshgrid(rho, th, indexing="ij")
        rtz = np.column_stack([R_.ravel(), T_.ravel(), np.zeros(R_.size)])
        print(f"{run}: basis L {cur.basis.L} M {cur.basis.M}, edge power {cur.edge_power}, "
              f"{len(rtz)} targets", flush=True)
        t0 = time.time()
        Bd = plasma_field(cur, rtz, u=u, method="direct")
        td = time.time() - t0
        plasma_field(cur, rtz[:32], u=u)  # compile
        t0 = time.time()
        Bt = plasma_field(cur, rtz, u=u)
        tt = time.time() - t0
        scale = np.max(np.abs(Bd[:, [0, 2]]))
        err = np.abs(Bt - Bd)[:, [0, 2]].max(1).reshape(len(rho), -1).max(1) / scale
        print("  max |B_tables - B_direct| / max|B_pol| by rho: "
              + " ".join(f"{r:g}: {e:.1e}" for r, e in zip(rho, err)), flush=True)
        print(f"  direct {td:.1f} s ({len(rtz) / td:.1f} targets/s), tables {tt:.1f} s "
              f"({len(rtz) / tt:.1f} targets/s), x{td / tt:.0f}", flush=True)
        if a.full:
            r1 = (np.arange(160) + 0.5) / 160
            R_, T_ = np.meshgrid(r1, 2 * np.pi * np.arange(160) / 160, indexing="ij")
            rf = np.column_stack([R_.ravel(), T_.ravel(), np.zeros(R_.size)])
            t0 = time.time()
            plasma_field(cur, rf, u=u)
            t1 = time.time() - t0
            print(f"  tables on 160 x 160 = {len(rf)} targets: {t1:.1f} s ({len(rf) / t1:.0f} targets/s)",
                  flush=True)


if __name__ == "__main__":
    main()
