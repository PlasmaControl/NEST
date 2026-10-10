"""Run a NEST optimization on the smooth-current precise_QA in its coils.

Starts from the DESC equilibrium (``BoundaryPotential`` and ``Pressure`` fits),
minimizes the chosen weighted objectives with the matrix-free Levenberg-Marquardt
of ``optimize``, and saves the coefficients, the history and the force error by
radius before and after to ``--out`` (npz) and the history to ``--out`` .json.

run (GPU): sbatch --qos=gpu-short --time=03:00:00 run.sbatch scripts/optimize_qa.py \
    [--frac 10] [--kernel hockney|truncated] [--free current|all] [--maxiter 20] \
    [--cg 100] [--precond jacobi|block] [--time-limit s] [--w-force 1] [--w-B 0] [--w-J 0] [--w-p 0] [--factors 3 3 2.5] \
    [--out results/opt]
"""

import argparse
import gc
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
import jax.numpy as jnp  # noqa: E402

import fit_cache  # noqa: E402
from boundary_potential import BoundaryPotential  # noqa: E402
from desc_input import DESCInput  # noqa: E402
from force import ForceResidual  # noqa: E402
from grid import CartesianGrid  # noqa: E402
from objective import (CurrentError, FieldError, Fields, ForceBalance,  # noqa: E402
                       ObjectiveFunction, PressureEnergy, PressureError)
from optimize import levenberg_marquardt  # noqa: E402
from precondition import BlockPreconditioner, mode_groups, mode_labels  # noqa: E402
from pressure import Pressure  # noqa: E402

jax.config.update("jax_enable_x64", True)
EQ = os.path.join(ROOT, "desc_input", "precise_QA_free_boundary_smooth_J",
                  "precise_QA_free_boundary_smooth_J_edgeJ_r0.3.h5")
COILS = os.path.join(ROOT, "desc_input", "precise_QA_free_boundary", "precise_QA_helical_coils.h5")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--frac", type=int, default=10)
    ap.add_argument("--kernel", default="truncated", choices=["hockney", "truncated"])
    ap.add_argument("--free", default="current", choices=["current", "all"])
    ap.add_argument("--maxiter", type=int, default=20)
    ap.add_argument("--cg", type=int, default=100)
    ap.add_argument("--cg-tol", type=float, default=1e-2)
    ap.add_argument("--start", default=None,
                    help="npz with x (from a previous run) to start from instead of the fits")
    ap.add_argument("--ftol", type=float, default=2e-4,
                    help="stop when f decreases by less than this per iteration (over 5)")
    ap.add_argument("--precond", default="block", choices=["jacobi", "block"],
                    help="block: (|m|, |n|) blocks of the local columns (precondition.py)")
    ap.add_argument("--time-limit", type=float, default=None,
                    help="seconds for the iterations (leave time for the check)")
    ap.add_argument("--factors", type=float, nargs=3, default=(3, 3, 2.5))
    ap.add_argument("--colloc", type=float, default=2,
                    help="collocation resolution / resolution of F (potential + equilibrium L, M, N)")
    ap.add_argument("--check", type=float, default=3,
                    help="independent check grid resolution / resolution of F")
    ap.add_argument("--check-frac", type=int, default=None,
                    help="box of the independent check, h = a / check-frac (default 2 frac, "
                         "at most a/20 unless frac is finer)")
    ap.add_argument("--check-kernel", default="truncated", choices=["hockney", "truncated"])
    for k in ("force", "B", "J", "p", "W"):
        ap.add_argument(f"--w-{k}", type=float, default=1.0 if k == "force" else 0.0)
    ap.add_argument("--out", default=os.path.join(ROOT, "results", "opt"))
    a = ap.parse_args()
    os.makedirs(os.path.dirname(a.out), exist_ok=True)

    t0 = time.time()
    inp = DESCInput.load(EQ, field=COILS)
    cur = fit_cache.fit(BoundaryPotential, inp, factors=tuple(a.factors))
    pre = fit_cache.fit(Pressure, inp, factors=tuple(a.factors), edge_order=2)
    sym = bool(inp.eq.sym)
    box = CartesianGrid.periodic_box(inp, inp.a / a.frac, odd=sym).locate_plasma(inp)
    eq = inp.eq
    # F = J x B - grad p has about the resolution of the potential plus the equilibrium's
    b = cur.basis
    res = lambda f: tuple(int(round(f * (u + v))) for u, v in ((b.L, eq.L), (b.M, eq.M), (b.N, eq.N)))
    fr = ForceResidual(inp, box, resolution=res(a.colloc))
    fields = Fields(fr, cur, pre, kernel=a.kernel)
    weights = {"force": a.w_force, "B": a.w_B, "J": a.w_J, "p": a.w_p, "W": a.w_W}
    classes = {"force": ForceBalance, "B": FieldError, "J": CurrentError, "p": PressureError,
               "W": PressureEnergy}
    obj = ObjectiveFunction([classes[k](w) for k, w in weights.items() if w > 0], fields).build()
    free = np.ones(obj.dim_x, bool) if a.free == "all" else np.arange(obj.dim_x) < fields.n_u
    print(f"{inp}\nbox {box.shape}, collocation {fr.tgrid.shape}, unknowns {obj.dim_x:,} "
          f"({int(free.sum()):,} free), residual {obj.dim_f:,}; objectives {weights}; "
          f"kernel {a.kernel}; setup {time.time() - t0:.0f} s; "
          f"{obj.dim_f / int(free.sum()):.1f} residuals per free unknown", flush=True)
    m0 = fr.metrics(np.asarray(fields.compute(obj.x0)["F"]))
    def save(x, hist):  # every iteration, so that a time-out keeps the last x
        np.savez(a.out + "_iter.npz", x=np.asarray(x), x0=np.asarray(obj.x0), free=free)
        with open(a.out + "_iter.json", "w") as f:
            json.dump({"args": vars(a), "history": hist}, f, indent=1)

    xs = obj.x0 if a.start is None else jnp.asarray(np.load(a.start)["x"])
    P = None
    if a.precond == "block":  # the local blocks at the start
        P = BlockPreconditioner(obj, xs, mode_groups(mode_labels(fields, cur, pre), free),
                                structure=cur.S)
    x, hist = levenberg_marquardt(obj, x0=xs, free=free, maxiter=a.maxiter, cg_maxiter=a.cg,
                                  cg_tol=a.cg_tol, callback=save, time_limit=a.time_limit,
                                  precond=P, ftol=a.ftol)
    m1 = fr.metrics(np.asarray(fields.compute(x)["F"]))
    x0, (u, c_p) = obj.x0, fields.split(x)
    print("force error <|F|> / norm by rho, before -> after:")
    for k in m0["bins"]:
        print(f"   {k:10s} {m0['bins'][k]:.2e} -> {m1['bins'][k]:.2e}")
    print(f"   all        {m0['mean']:.2e} -> {m1['mean']:.2e} (DESC {fr.F_desc_vol / fr.norm:.2e})")
    # independent check: F on a finer collocation grid and with B_plasma from a
    # finer box (and the truncated kernel): the optimum can otherwise fit the
    # discretization error of B on the optimization box (J on the box as in the
    # objective: the same as evaluating it exactly there, scripts/diagnose_opt.py)
    # free the optimization's device arrays first: the jitted functions keep the
    # objective (a static argument) in their caches
    del obj, fields, P
    jax.clear_caches()
    gc.collect()
    cf = a.check_frac or min(2 * a.frac, max(a.frac, 20))  # a/30 is beyond one A100
    cbox = CartesianGrid.periodic_box(inp, inp.a / cf, odd=sym).locate_plasma(inp)
    chk = ForceResidual(inp, cbox, resolution=res(a.check))
    fc = Fields(chk, cur, pre, kernel=a.check_kernel)
    out = {name: chk.metrics(np.asarray(fc.compute(xx)["F"])) for name, xx in (("start", x0), ("end", x))}
    print(f"check: grid {chk.tgrid.shape}, box h = a/{cf} {cbox.shape}, kernel {a.check_kernel}; "
          f"<|F|> / norm by rho, start -> end:")
    for k in out["start"]["bins"]:
        print(f"   {k:10s} {out['start']['bins'][k]:.2e} -> {out['end']['bins'][k]:.2e}")
    print(f"   all        {out['start']['mean']:.2e} -> {out['end']['mean']:.2e}")
    np.savez(a.out + ".npz", x=np.asarray(x), x0=np.asarray(x0), u=np.asarray(u),
             c_p=np.asarray(c_p), free=free, rho_bins=list(m0["bins"]),
             before=list(m0["bins"].values()), after=list(m1["bins"].values()))
    with open(a.out + ".json", "w") as f:
        json.dump({"args": vars(a), "history": hist, "before": m0, "after": m1, "check": out,
                   "desc_force_error": fr.F_desc_vol / fr.norm}, f, indent=1)
    print(f"saved {a.out}.npz/.json; total {time.time() - t0:.0f} s")


if __name__ == "__main__":
    main()
