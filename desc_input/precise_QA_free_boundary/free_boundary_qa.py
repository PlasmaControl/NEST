"""Finite-beta free-boundary DESC equilibrium used as the fixed boundary for NEMO.

Starting point: the DESC ``precise_QA`` vacuum equilibrium (NFP=2) and the helical
coil set designed for it (``tests/inputs/precise_QA_helical_coils.h5``).

Steps
-----
1. Add finite pressure p = p0 (1 - rho^2)^2 and prescribe the rotational transform
   iota = iota0 - diota rho^2 by continuation from the vacuum state, solved as a
   fixed-boundary problem. iota is prescribed (not the current) so that the
   resonant surface iota = 1/2 stays where we want it after the free-boundary
   solve: with NFP=2 and stellarator symmetry only n = 2k harmonics exist, so
   iota = 2/4 is the lowest-order rational at which NEMO can open islands.
   Defaults put it at rho ~ 0.55 with low shear (wide islands) and keep every other
   n = 2k, m <= 12 rational out of the plasma.
2. Free-boundary solve in the (fixed) coil field: minimize ``BoundaryError``
   (B.n = 0 and [[B^2/2mu0 + p]] = 0 on the LCFS, plasma field from virtual casing)
   subject to force balance, using the proximal projection optimizer and a
   multigrid release of boundary modes.
3. Final tight force-balance solve and diagnostics.

Outputs (in --outdir): the equilibrium .h5, a copy of the coil set, diagnostics JSON
and plots.
"""

import argparse
import json
import os

parser = argparse.ArgumentParser()
parser.add_argument("--outdir", default="output")
parser.add_argument("--device", default="gpu")
parser.add_argument("--beta-target", type=float, default=0.01, help="<beta>_vol")
parser.add_argument("--iota0", type=float, default=0.52, help="iota on axis")
parser.add_argument("--diota", type=float, default=0.066, help="iota0 - iota(rho=1)")
parser.add_argument(
    "--k-list", type=int, nargs="*", default=[2, 4, 8, 12],
    help="empty with --resume: only the final solve and outputs",
)
parser.add_argument("--maxiter", type=int, default=150, help="last stage")
parser.add_argument(
    "--maxiter-intermediate", type=int, default=25,
    help="stages before the last one only need to move the low-order modes",
)
parser.add_argument(
    "--inner-ftol", type=float, default=1e-6,
    help="ftol of the equilibrium sub-solve; loose values make the outer "
    "objective noisy and stall the trust region",
)
parser.add_argument("--jac-chunk-size", type=int, default=16, help="limits GPU memory")
parser.add_argument(
    "--resume", default=None,
    help="checkpoint .h5 in outdir to continue the free-boundary stages from "
    "(stages 0-1 are skipped; pass the remaining --k-list)",
)
parser.add_argument("--LMN", type=int, nargs=3, default=[12, 12, 12], help="L M N")
args = parser.parse_args()

from desc import set_device  # noqa: E402

set_device(args.device)

import jax  # noqa: E402

os.makedirs(args.outdir, exist_ok=True)
jax.config.update("jax_compilation_cache_dir", os.path.join(args.outdir, ".jax-cache"))
jax.config.update("jax_persistent_cache_min_entry_size_bytes", -1)
jax.config.update("jax_persistent_cache_min_compile_time_secs", 0)

import matplotlib  # noqa: E402

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

import desc  # noqa: E402
import desc.examples  # noqa: E402
import desc.io  # noqa: E402
from desc.grid import LinearGrid  # noqa: E402
from desc.objectives import (  # noqa: E402
    BoundaryError,
    FixBoundaryR,
    FixBoundaryZ,
    FixIota,
    FixPressure,
    FixPsi,
    ForceBalance,
    ObjectiveFunction,
)
from desc.plotting import plot_2d, plot_comparison  # noqa: E402
from desc.profiles import PowerSeriesProfile  # noqa: E402
from fb_tools import log, pad, summary  # noqa: E402

DESC_ROOT = os.path.dirname(os.path.dirname(desc.__file__))
COIL_FILE = os.path.join(DESC_ROOT, "tests", "inputs", "precise_QA_helical_coils.h5")
mu0 = 4e-7 * np.pi


# ---------------------------------------------------------------------------
# 0. starting vacuum equilibrium and coils
# ---------------------------------------------------------------------------
eq_vac = desc.examples.get("precise_QA")
coils = desc.io.load(COIL_FILE)
fixb_file = os.path.join(args.outdir, "fixed_boundary_finite_beta.h5")
if args.resume is not None:
    eq = desc.io.load(args.resume)
    eq_fixb = desc.io.load(fixb_file)
    p0 = float(eq_fixb.pressure.params[0])
    with open(os.path.join(args.outdir, "diagnostics.json")) as f:
        diags = json.load(f)
    diags.pop("history", None)
    log(f"resuming from {args.resume}: L={eq.L} M={eq.M} N={eq.N}")
else:
    eq = eq_vac.copy()
    L, M, N = args.LMN
    if (L, M, N) != (eq.L, eq.M, eq.N):
        eq.change_resolution(L=L, M=M, N=N, L_grid=2 * L, M_grid=2 * M, N_grid=2 * N)
        eq.solve(ftol=1e-8, xtol=1e-12, gtol=1e-12, maxiter=200, verbose=1)
    log(f"eq: L={eq.L} M={eq.M} N={eq.N} NFP={eq.NFP} Psi={eq.Psi}; coils: {len(coils)}")
    diags = {"vacuum": summary(eq, coils, "vacuum (fixed boundary)")}

    # ---------------------------------------------------------------------------
    # 1. finite beta + prescribed iota, fixed boundary, by continuation
    # ---------------------------------------------------------------------------
    B0 = float(eq.compute("<|B|>_vol")["<|B|>_vol"])
    # <(1-rho^2)^2>_vol ~ 1/3, so p0 = 3 * beta * B^2/(2 mu0) to leading order
    p0 = 3 * args.beta_target * B0**2 / (2 * mu0)
    # vacuum iota of precise_QA is flat, ~0.42: ramp from it to the target profile
    iota_vac = float(eq.compute("iota", grid=LinearGrid(rho=[0.5]))["iota"][0])
    eq.pressure = PowerSeriesProfile([0.0, 0, 0, 0, 0], modes=[0, 2, 4, 6, 8])
    eq.iota = PowerSeriesProfile([iota_vac, 0, 0, 0, 0], modes=[0, 2, 4, 6, 8])
    for frac in [0.25, 0.5, 0.75, 1.0]:
        iota0 = iota_vac + frac * (args.iota0 - iota_vac)
        # profiles are resized to eq.L, so only set the leading even coefficients
        eq.pressure.params = pad(frac * p0 * np.array([1.0, -2.0, 1.0]), eq.pressure)
        eq.iota.params = pad(np.array([iota0, -frac * args.diota]), eq.iota)
        log(f"continuation step {frac}: p0={frac * p0:.4e} Pa, iota0={iota0:.4f}")
        eq.solve(ftol=1e-6, xtol=1e-10, gtol=1e-10, maxiter=100, verbose=1)
    eq.solve(ftol=1e-8, xtol=1e-12, gtol=1e-12, maxiter=200, verbose=1)
    eq.save(fixb_file)
    diags["fixed_boundary_finite_beta"] = summary(eq, coils, "finite beta (fixed boundary)")
    eq_fixb = eq.copy()

# ---------------------------------------------------------------------------
# 2. free-boundary solve in the coil field
# ---------------------------------------------------------------------------
constraints = (
    ForceBalance(eq=eq),
    FixPressure(eq=eq),
    FixIota(eq=eq),
    FixPsi(eq=eq),
)
objective = ObjectiveFunction(
    BoundaryError(eq=eq, field=coils, field_fixed=True),
    jac_chunk_size=args.jac_chunk_size,
)
history = []
for k in args.k_list:
    R_modes = eq.surface.R_basis.modes[np.max(np.abs(eq.surface.R_basis.modes), 1) > k]
    Z_modes = eq.surface.Z_basis.modes[np.max(np.abs(eq.surface.Z_basis.modes), 1) > k]
    bdry_constraints = (
        FixBoundaryR(eq=eq, modes=R_modes),
        FixBoundaryZ(eq=eq, modes=Z_modes),
    )
    log(f"free boundary: releasing boundary modes with max(|m|,|n|) <= {k}")
    eq, out = eq.optimize(
        objective,
        constraints + bdry_constraints,
        optimizer="proximal-lsq-exact",
        maxiter=args.maxiter if k == args.k_list[-1] else args.maxiter_intermediate,
        ftol=1e-6,
        xtol=1e-10,
        gtol=1e-10,
        verbose=3,
        options={
            "solve_options": {
                "ftol": args.inner_ftol, "xtol": 1e-12, "gtol": 1e-12, "maxiter": 100
            }
        },
        copy=False,
    )
    history.append(
        {"k": k, "nfev": int(out["nfev"]), "cost": float(out["cost"])}
        | {"message": str(out["message"])}
    )
    diags[f"free_boundary_k{k}"] = summary(eq, coils, f"free boundary k={k}")
    eq.save(os.path.join(args.outdir, f"free_boundary_k{k}.h5"))  # checkpoint
    with open(os.path.join(args.outdir, "diagnostics.json"), "w") as f:
        json.dump(diags | {"history": history}, f, indent=2)

# final tight equilibrium solve with the free-boundary LCFS held fixed
eq.solve(ftol=1e-9, xtol=1e-12, gtol=1e-12, maxiter=300, verbose=1)
diags["final"] = summary(eq, coils, "final")
diags["history"] = history
diags["params"] = vars(args) | {"p0 [Pa]": p0, "coil_file": COIL_FILE}

# ---------------------------------------------------------------------------
# 3. save
# ---------------------------------------------------------------------------
eq.save(os.path.join(args.outdir, "precise_QA_free_boundary.h5"))
coils.save(os.path.join(args.outdir, "precise_QA_helical_coils.h5"))
with open(os.path.join(args.outdir, "diagnostics.json"), "w") as f:
    json.dump(diags, f, indent=2)

fig, ax = plot_comparison(
    [eq_vac, eq_fixb, eq],
    labels=["vacuum (fixed bdry)", "finite beta (fixed bdry)", "finite beta (free bdry)"],
    theta=0,
)
fig.savefig(os.path.join(args.outdir, "surfaces_comparison.png"), dpi=150)
for name, e in [("fixed", eq_fixb), ("free", eq)]:
    fig, ax = plot_2d(e, "B*n", field=coils)
    fig.savefig(os.path.join(args.outdir, f"Bn_{name}_boundary.png"), dpi=150)
    plt.close("all")
log("done")
