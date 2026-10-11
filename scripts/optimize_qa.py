"""Run a NEST optimization on the smooth-current precise_QA in its coils.

Starts from the DESC equilibrium (``BoundaryPotential`` and ``Pressure`` fits),
minimizes the chosen weighted objectives with the matrix-free Levenberg-Marquardt
of ``optimize``, and saves the coefficients, the history and the force error by
radius before and after to ``--out`` (npz) and the history to ``--out`` .json.

Default: the current and the pressure free, the stored energy held and p >= 0,
the truncated kernel, the block preconditioner; ``--refine 20`` continues on a
finer box.

run (GPU): sbatch --qos=gpu-stellar --time=03:00:00 run.sbatch scripts/optimize_qa.py \
    [--frac 10] [--refine 20] [--kernel hockney|truncated] [--free current|all] \
    [--maxiter 100] [--cg 100] [--precond jacobi|block] [--time-limit s] [--start x.npz] \
    [--w-force 1] [--w-B 0] [--w-J 0] [--w-p 0] [--w-W 10] [--w-pos 10] [--factors 3 3 2.5] \
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
                       ObjectiveFunction, PressureEnergy, PressureError,
                       PressurePositivity)
from optimize import levenberg_marquardt  # noqa: E402
from precondition import BlockPreconditioner, mode_groups, mode_labels  # noqa: E402
from pressure import Pressure  # noqa: E402

jax.config.update("jax_enable_x64", True)
EQ = os.path.join(ROOT, "desc_input", "precise_QA_free_boundary_smooth_J",
                  "precise_QA_free_boundary_smooth_J_edgeJ_r0.3.h5")
COILS = os.path.join(ROOT, "desc_input", "precise_QA_free_boundary", "precise_QA_helical_coils.h5")


def check(inp, box, cur, pre, resolution, kernel, xs, n_rho=8):
    """<|F|>_vol / norm (overall and by radius) of each x in xs on the collocation
    grid of ``resolution``, evaluated n_rho radii at a time (the memory of a
    fraction of the grid); returns the metrics and the grid's shape."""
    full = ForceResidual(inp, box, resolution=resolution, rho_slice=slice(0, 1))
    L = resolution[0]
    n_r = L // 2 + 1
    F = {k: [] for k in xs}
    dV, rho = [], []
    base = None  # the box side (flux grid, box interpolation, kernel), built once
    for i in range(0, n_r, n_rho):
        fr = ForceResidual(inp, box, resolution=resolution, rho_slice=slice(i, i + n_rho))
        fields = Fields(fr, cur, pre, kernel=kernel, share=base)
        base = fields if base is None else base
        for k, x in xs.items():
            F[k].append(np.linalg.norm(np.asarray(fields.compute(x)["F"]), axis=1) / fr.norm)
        dV.append(fr.dV)
        rho.append(fr.rtz[:, 0])
        del fields, fr
        gc.collect()
    del base
    jax.clear_caches()
    gc.collect()
    dV, rho = np.concatenate(dV), np.concatenate(rho)
    edges = (0.0, 0.5, 0.8, 0.9, 0.95, 1.0)
    avg = lambda f, m: float(np.sum(f * dV * m) / np.sum(dV * m))
    out = {}
    for k in xs:
        f = np.concatenate(F[k])
        out[k] = {"mean": avg(f, 1.0), "bins": {f"{a:g}-{b:g}": avg(f, (rho > a) & (rho <= b))
                                               for a, b in zip(edges[:-1], edges[1:])}}
    return out, (n_r,) + full.tgrid.shape[1:]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--eq", default=EQ, help="DESC equilibrium file (default: smooth-current precise_QA)")
    ap.add_argument("--coils", default=COILS, help="its coils (DESC file)")
    ap.add_argument("--frac", type=int, default=10)
    ap.add_argument("--refine", type=int, nargs="*", default=[],
                    help="continue on finer boxes, h = a / refine (e.g. --refine 20)")
    ap.add_argument("--kernel", default="truncated", choices=["hockney", "truncated"])
    ap.add_argument("--free", default="all", choices=["current", "all"],
                    help="all: the current and the pressure (with W and pos held by default)")
    ap.add_argument("--maxiter", type=int, default=100, help="per box")
    ap.add_argument("--cg", type=int, default=100)
    ap.add_argument("--cg-tol", type=float, default=1e-2)
    ap.add_argument("--start", default=None,
                    help="npz with x (from a previous run) to start from instead of the fits")
    ap.add_argument("--ftol", type=float, default=2e-4,
                    help="stop when f decreases by less than this per iteration (over 5)")
    ap.add_argument("--precond", default="block", choices=["jacobi", "block"],
                    help="block: (|m|, |n|) blocks of the local columns (precondition.py)")
    ap.add_argument("--time-limit", type=float, default=None,
                    help="seconds for the iterations of each box (leave time for the check)")
    ap.add_argument("--factors", type=float, nargs=3, default=(3, 3, 2.5))
    ap.add_argument("--colloc", type=float, default=2,
                    help="collocation resolution / resolution of F (potential + equilibrium L, M, N)")
    ap.add_argument("--check", type=float, default=3,
                    help="independent check grid resolution / resolution of F")
    ap.add_argument("--check-frac", type=int, default=None,
                    help="box of the independent check, h = a / check-frac (default 2 x the last "
                         "box, at most a/20 unless that is finer)")
    ap.add_argument("--check-kernel", default="truncated", choices=["hockney", "truncated"])
    for k in ("force", "B", "J", "p", "W", "pos"):
        ap.add_argument(f"--w-{k}", type=float, default=1.0 if k == "force" else None)
    ap.add_argument("--fix-I", action="store_true",
                    help="hold the enclosed toroidal current I(rho) (its zeta average) at the "
                         "start's: the a_l00 coefficients fixed")
    ap.add_argument("--p-scale", type=float, default=1.0,
                    help="pressure x this: the start's c_p and the stored-energy target (a beta scan)")
    ap.add_argument("--out", default=os.path.join(ROOT, "results", "opt"))
    a = ap.parse_args()
    os.makedirs(os.path.dirname(a.out), exist_ok=True)
    # with the pressure free: its volume average held and p >= 0 (``PressureEnergy``,
    # ``PressurePositivity``); without them F also drops by deflating the plasma
    for k in ("W", "pos"):
        if getattr(a, f"w_{k}") is None:
            setattr(a, f"w_{k}", 10.0 if a.free == "all" else 0.0)
    for k in ("B", "J", "p"):
        if getattr(a, f"w_{k}") is None:
            setattr(a, f"w_{k}", 0.0)

    t0 = time.time()
    inp = DESCInput.load(a.eq, field=a.coils)
    cur = fit_cache.fit(BoundaryPotential, inp, factors=tuple(a.factors))
    pre = fit_cache.fit(Pressure, inp, factors=tuple(a.factors), edge_order=2)
    sym = bool(inp.eq.sym)
    eq = inp.eq
    # F = J x B - grad p has about the resolution of the potential plus the equilibrium's
    b = cur.basis
    res = lambda f: tuple(int(round(f * (u + v))) for u, v in ((b.L, eq.L), (b.M, eq.M), (b.N, eq.N)))
    weights = {"force": a.w_force, "B": a.w_B, "J": a.w_J, "p": a.w_p, "W": a.w_W,
               "pos": a.w_pos}
    classes = {"force": ForceBalance, "B": FieldError, "J": CurrentError, "p": PressureError,
               "W": PressureEnergy, "pos": PressurePositivity}
    x = None
    if a.start is not None:  # its pressure rescaled from the scale it was run at
        st = np.load(a.start)
        x = jnp.asarray(st["x"])
        s0 = float(st["p_scale"]) if "p_scale" in st else 1.0
        if a.p_scale != s0:
            x = x.at[cur.n_free :].multiply(a.p_scale / s0)
    hist, m0, m1 = [], None, None
    for stage, frac in enumerate([a.frac] + list(a.refine)):
        box = CartesianGrid.periodic_box(inp, inp.a / frac, odd=sym).locate_plasma(inp)
        fr = ForceResidual(inp, box, resolution=res(a.colloc))
        fields = Fields(fr, cur, pre, kernel=a.kernel)
        obj = ObjectiveFunction([(PressureEnergy(w, scale=a.p_scale) if k == "W" else classes[k](w))
                                 for k, w in weights.items() if w > 0], fields).build()
        free = np.ones(obj.dim_x, bool) if a.free == "all" else np.arange(obj.dim_x) < fields.n_u
        if a.fix_I:  # 2pi <T_theta>_(theta, zeta) = I h_0 + 2pi rho^2 f <a>: only a_l00 enter
            md = cur.basis.modes
            fix = cur.S.n_chi + np.nonzero((md[:, 1] == 0) & (md[:, 2] == 0))[0]
            free[fix] = False
        x0 = obj.x0
        if x is None:
            x = x0
            if a.p_scale != 1.0:  # the start's pressure scaled (once, before the first box)
                x = x.at[fields.n_u :].multiply(a.p_scale)
        if a.fix_I:  # DESC's I(rho) (the fits'), also from a start that changed it
            x = x.at[fix].set(x0[fix])
        print(f"{inp}\n[box {stage + 1}] h = a/{frac} {box.shape}, collocation {fr.tgrid.shape}, "
              f"unknowns {obj.dim_x:,} ({int(free.sum()):,} free), residual {obj.dim_f:,}; "
              f"objectives {weights}; kernel {a.kernel}; setup {time.time() - t0:.0f} s; "
              f"{obj.dim_f / int(free.sum()):.1f} residuals per free unknown", flush=True)
        if m0 is None:
            m0 = fr.metrics(np.asarray(fields.compute(x0)["F"]))

        def save(x, h, frac=frac):  # every iteration, so that a time-out keeps the last x
            np.savez(a.out + "_iter.npz", x=np.asarray(x), x0=np.asarray(x0), free=free,
                     p_scale=a.p_scale)
            with open(a.out + "_iter.json", "w") as f:
                json.dump({"args": vars(a), "box": frac, "history": hist + h}, f, indent=1)

        P = None
        if a.precond == "block":  # the local blocks at the start of this box
            P = BlockPreconditioner(obj, x, mode_groups(mode_labels(fields, cur, pre), free),
                                    structure=cur.S)
        x, h = levenberg_marquardt(obj, x0=x, free=free, maxiter=a.maxiter, cg_maxiter=a.cg,
                                   cg_tol=a.cg_tol, callback=save, time_limit=a.time_limit,
                                   precond=P, ftol=a.ftol)
        hist += [dict(e, box=frac) for e in h]
        m1 = fr.metrics(np.asarray(fields.compute(x)["F"]))
        print(f"force error <|F|> / norm by rho on box {stage + 1} (h = a/{frac}), start -> end:")
        for k in m0["bins"]:
            print(f"   {k:10s} {m0['bins'][k]:.2e} -> {m1['bins'][k]:.2e}")
        print(f"   all        {m0['mean']:.2e} -> {m1['mean']:.2e} (DESC {fr.F_desc_vol / fr.norm:.2e})",
              flush=True)
        desc_err = fr.F_desc_vol / fr.norm
        # free the device arrays of this box: the jitted functions keep the
        # objective (a static argument) in their caches
        del obj, fields, P, fr
        jax.clear_caches()
        gc.collect()
    u, c_p = x[: cur.n_free], x[cur.n_free :]
    # independent check: F on a finer collocation grid and with B_plasma from a
    # finer box (and the truncated kernel): the optimum can otherwise fit the
    # discretization error of B on the optimization box (J on the box as in the
    # objective: the same as evaluating it exactly there, scripts/diagnose_opt.py)
    last = ([a.frac] + list(a.refine))[-1]
    cf = a.check_frac or min(2 * last, max(last, 20))  # a/30 is beyond one A100
    cbox = CartesianGrid.periodic_box(inp, inp.a / cf, odd=sym).locate_plasma(inp)
    # saved first: the check is the most memory-hungry step
    np.savez(a.out + ".npz", x=np.asarray(x), x0=np.asarray(x0), u=np.asarray(u),
             c_p=np.asarray(c_p), free=free, p_scale=a.p_scale)
    out, shape = check(inp, cbox, cur, pre, res(a.check), a.check_kernel,
                       {"start": x0, "end": x})
    print(f"check: grid {shape}, box h = a/{cf} {cbox.shape}, kernel {a.check_kernel}; "
          f"<|F|> / norm by rho, start -> end:")
    for k in out["start"]["bins"]:
        print(f"   {k:10s} {out['start']['bins'][k]:.2e} -> {out['end']['bins'][k]:.2e}")
    print(f"   all        {out['start']['mean']:.2e} -> {out['end']['mean']:.2e}")
    np.savez(a.out + ".npz", x=np.asarray(x), x0=np.asarray(x0), u=np.asarray(u),
             c_p=np.asarray(c_p), free=free, p_scale=a.p_scale, rho_bins=list(m0["bins"]),
             before=list(m0["bins"].values()), after=list(m1["bins"].values()))
    with open(a.out + ".json", "w") as f:
        json.dump({"args": vars(a), "history": hist, "before": m0, "after": m1, "check": out,
                   "desc_force_error": desc_err}, f, indent=1)
    print(f"saved {a.out}.npz/.json; total {time.time() - t0:.0f} s")


if __name__ == "__main__":
    main()
