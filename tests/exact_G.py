"""axisym_field.plasma_field_matrix (the exact dense G) against plasma_field.

For each run (its basis, collocation grid and final u): build G on the collocation points,
time it, and compare G u + B_I with plasma_field(u) on every point (same quadrature: the
two must agree to round-off); also the run's own box field at u, for scale.

run: sbatch run.sbatch tests/exact_G.py --runs results/gs/cylD_a160 results/gs/cyl3D_a160
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
from axisym_field import plasma_field, plasma_field_matrix  # noqa: E402
from boundary_potential import BoundaryPotential  # noqa: E402
from desc_input import DESCInput  # noqa: E402
from force import ForceResidual  # noqa: E402

jax.config.update("jax_enable_x64", True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--runs", nargs="+", required=True)
    ap.add_argument("--chunk", type=int, default=32)
    a = ap.parse_args()
    for run in a.runs:
        args = json.load(open(run + "_iter.json"))["args"]
        inp = DESCInput.load(args["eq"], field=args["coils"])
        eq = inp.eq
        ep = args.get("edge_power", 2)
        cur = fit_cache.fit(BoundaryPotential, inp, factors=tuple(args["factors"]),
                            **({"edge_power": ep} if ep != 2 else {}))
        b = cur.basis
        res = tuple(int(round(args["colloc"] * (p + q))) for p, q in ((b.L, eq.L), (b.M, eq.M), (b.N, eq.N)))
        fr = ForceResidual(inp, None, resolution=res)
        rtz = np.asarray(fr.rtz)
        u = np.load(run + "_iter.npz")["x"][: cur.n_free]
        print(f"{run}: basis L {b.L} M {b.M}, n_u {cur.n_free}, {len(rtz)} collocation points", flush=True)
        t0 = time.time()
        G = plasma_field_matrix(cur, rtz, chunk=a.chunk, verbose=True)
        tG = time.time() - t0
        B_I = plasma_field(cur, rtz, u=np.zeros(cur.n_free))
        t0 = time.time()
        B = plasma_field(cur, rtz, u=u)
        tB = time.time() - t0
        BG = np.einsum("k,knc->nc", u, G) + B_I
        rms = np.sqrt(np.mean(np.sum(B**2, 1)))
        e = np.abs(BG - B)
        print(f"  G {G.shape} built in {tG:.0f} s (one plasma_field: {tB:.0f} s); "
              f"|G u + B_I - plasma_field(u)|: max {e.max() / np.abs(B).max():.1e} of max |B|, "
              f"rms {np.sqrt(np.mean(np.sum(e**2, 1))) / rms:.1e} of rms |B|", flush=True)
        del G


if __name__ == "__main__":
    main()
