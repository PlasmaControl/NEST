"""Solve driver: the combined objective ||r_D||^2 + lambda^-1 ||r_F||^2 by lambda continuation.

usage: python run_solve.py [EQ.h5] [--coils COILS.h5 [MORE.h5 ...]] [--basis L M N] [--frac 10]
           [--kernel vico_greengard] [--decades -2 2] [--nlam 5] [--dtol 1e-3]
           [--maxiter 100] [--points-per-block P] [--cache-dir DIR] [--eval-factor 4]
           [--out DIR]

On Stellar / Della: sbatch run.sbatch nest/run_solve.py ARGS (run_cpu.sbatch for CPU).

1. DESCInput with the coils; J = curl T (``BoundaryPotential.fit``) and p (``Pressure.fit``,
   smooth edge k = 2) on the solve basis L, M, N (default the equilibrium's).
2. Box h = a / frac; ``Fields`` on QuadratureGrid(2 L, 2 M, 2 N) (D_B cached in
   --cache-dir if given); the pressure is replaced by the force-optimal one.
3. Objective FieldDistance + lambda^-1 ForceBalance. lambda_ref = ||r_F(x_0)||^2 / dtol^2
   (the force term equals a field distance dtol at the start); nlam values over
   --decades, from the largest (field-distance dominated) down, each warm-started.
4. Every end state is checked on an independent grid, QuadratureGrid(f L, f M, f N)
   with f = --eval-factor, by ``ForceResidual.evaluate`` of its series (B by one
   Biot-Savart application, no D_B): a force or field distance that is small only at
   the collocation points shows up there.
Every step prints progress lines (D_B columns, one GN line per trial step with the
device memory); out/run_solve.json (settings, per-lambda summaries, solve and evaluation
grids) and out/x_<k>.npy are written after each lambda.
"""

import argparse
import json
import os
import time

import numpy as np
from desc import set_device

# DESC uses the CPU unless set_device('gpu') runs before any other DESC import
set_device(os.environ.get("NEST_DEVICE", "cpu"))

import jax.numpy as jnp  # noqa: E402
from desc.grid import QuadratureGrid  # noqa: E402

from boundary_potential import BoundaryPotential  # noqa: E402
from desc_input import DESCInput  # noqa: E402
from fields import Fields  # noqa: E402
from force import ForceResidual  # noqa: E402
from grid import CartesianGrid, device_memory  # noqa: E402
from objectives import FieldDistance, ForceBalance, ObjectiveFunction  # noqa: E402
from optimize import continuation  # noqa: E402
from pressure import Pressure  # noqa: E402

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
FB = os.path.join(ROOT, "desc_input", "precise_QA_free_boundary_smooth_J")

p = argparse.ArgumentParser()
p.add_argument("eq", nargs="?", default=os.path.join(
    FB, "precise_QA_free_boundary_smooth_J_edgeJ_r0.3.h5"))
p.add_argument("--coils", nargs="+", default=[os.path.join(
    ROOT, "desc_input", "precise_QA_free_boundary", "precise_QA_helical_coils.h5")],
    help="external field file(s); several are summed (e.g. DSHAPE's TF and PF parts)")
p.add_argument("--basis", type=int, nargs=3, default=None, help="L M N of J and p")
p.add_argument("--frac", type=float, default=10, help="box spacing h = a / frac")
p.add_argument("--kernel", default="vico_greengard")
p.add_argument("--decades", type=float, nargs=2, default=(-2, 2))
p.add_argument("--nlam", type=int, default=5)
p.add_argument("--dtol", type=float, default=1e-3)
p.add_argument("--maxiter", type=int, default=100)
p.add_argument("--points-per-block", type=int, default=None, help="row chunking of the GN step")
p.add_argument("--col-chunk", type=int, default=4, help="D_B columns per batch")
p.add_argument("--cache-dir", default=None, help="D_B cache (resumable)")
p.add_argument("--eval-factor", type=float, default=4)
p.add_argument("--out", default="solve_out")
args = p.parse_args()

t0 = time.time()
log = lambda msg: print(f"[{time.time() - t0:7.1f}s] {msg}", flush=True)
os.makedirs(args.out, exist_ok=True)

inp = DESCInput.load(args.eq, field=args.coils if len(args.coils) > 1 else args.coils[0])
L, M, N = args.basis or inp.resolution
log(f"{inp}; solve basis {L}, {M}, {N}; device memory {device_memory()}")
current = BoundaryPotential.fit(inp.eq, L, M, N)
pressure = Pressure.fit(inp.eq, L, M, N, edge_order=2)
box = CartesianGrid.periodic_box(inp, inp.a / args.frac).locate_plasma(inp, verbose=True)
log(f"box {box.shape}, h = a/{args.frac:g}, {len(box.inside):,} inside points")
F = Fields(inp, current, pressure, box, kernel=args.kernel, col_chunk=args.col_chunk,
           cache_dir=args.cache_dir)
log(f"Fields ready; device memory {device_memory()}")
x0 = F.fit_pressure(F.initial_state())
obj = ObjectiveFunction(F, [FieldDistance(F), ForceBalance(F)])
s0 = obj.summary(x0)
lam_ref = s0["force"] ** 2 / args.dtol**2
lams = lam_ref * 10.0 ** np.linspace(args.decades[1], args.decades[0], args.nlam)
log(f"start {s0}; lambda_ref {lam_ref:.3e}")

# independent evaluation grid: ForceResidual of the end states' series
f = args.eval_factor
geval = QuadratureGrid(L=int(f * L), M=int(f * M), N=int(f * N), NFP=inp.eq.NFP)
ev = ForceResidual(inp, box, geval, kernel=args.kernel, dtype=np.float64)
B_desc_ev = ev.desc_fields(["B"])["B"]


def evaluate(x):
    out = ev.evaluate(*F.objects(x))
    m = ev.metrics(out["F"])
    dB = np.linalg.norm(out["B"] - B_desc_ev, axis=1)
    return {"<|F|>/F_0": m["mean"], "rms|F|/F_0": m["rms"], "bins": m["bins"],
            "field_distance": float(np.sqrt(ev.average(dB**2) / ev.average(
                np.sum(B_desc_ev**2, axis=1))))}


record = {"args": vars(args), "solve_grid": [F.grid.L, F.grid.M, F.grid.N, len(F.x)],
          "eval_grid": [geval.L, geval.M, geval.N, len(ev.x)], "n": F.n,
          "box": list(box.shape), "inside": int(len(box.inside)), "lambda_ref": lam_ref,
          "start": {"solve": s0, "eval": evaluate(x0)}, "runs": []}
x = jnp.asarray(x0)
mem = lambda k, x, accepted: {"mem": device_memory()}
for k, lam in enumerate(lams):
    res = continuation(obj, x, [(1.0, 1.0 / lam)], maxiter=args.maxiter,
                       points_per_block=args.points_per_block, callback=mem)[0]
    x = res["x"]
    np.save(os.path.join(args.out, f"x_{k}.npy"), np.asarray(x))
    record["runs"].append({"lambda": float(lam), "lambda/ref": float(lam / lam_ref),
                           "stop": res["stop_reason"], "steps": len(res["history"]),
                           "wall_s": res["wall_s"], "solve": res["summary"],
                           "eval": evaluate(x)})
    log(f"lambda/ref {lam / lam_ref:.2e}: {record['runs'][-1]}")
    with open(os.path.join(args.out, "run_solve.json"), "w") as fh:
        json.dump(record, fh, indent=1)
log("done")
