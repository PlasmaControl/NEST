"""Free-boundary DESC equilibrium in the fixed coils, starting from the fixed-boundary
re-solve on the VMEC LCFS (``solve_desc.py``).

The NEST inputs ``<case>_L{L}M{M}N{N}.h5`` keep the VMEC LCFS fixed. In DESC's own
solution that surface is not exactly a free boundary: B.n/|B| is 2-4e-3 rms in the
coils. Here the LCFS is solved for instead, as for precise_QA
(``../precise_QA_free_boundary/free_boundary_qa.py``):

1. the start at the free-boundary resolution ``--fb-LMN`` (re-solved at fixed
   boundary if that differs from the start's; a 16/16/12 proximal solve does not
   fit in a 40 GB A100);
2. free boundary: minimize ``BoundaryError`` (B.n = 0 and [[B^2/2 mu0 + p]] = 0,
   plasma field by virtual casing) in the coils ``<case>_coils.h5``, with p, the
   current (or iota) and Psi fixed and force balance kept by ``proximal-lsq-exact``,
   releasing LCFS modes max(|m|, |n|) <= k for each k of ``--k-list`` (never up to
   the equilibrium resolution); the higher modes keep their VMEC values;
3. a tight fixed-boundary solve on the free-boundary LCFS at the final resolution
   ``--LMN`` (default: the start's).

For jobs under an hour: ``--no-final`` with the first k's, then ``--resume
<outdir>/free_boundary_last.h5 --k-list <the rest>``, and finally ``--resume ...
--k-list`` (empty: only step 3).

Outputs: ``<case>/<case>_fb_L{L}M{M}N{N}.h5`` and its ``_diagnostics.json`` (force
error, B.n/|B|, pressure jump, beta, iota and rational surfaces at each stage);
checkpoints in ``--outdir`` (default ``<case>/fb``).

Usage: python free_boundary_desc.py CASE [--start file.h5] [--fb-LMN L M N] [--LMN L M N]
           [--k-list 2 4 6] [--device gpu] [--jac-chunk-size 16]
run (GPU): sbatch --time=00:59:00 run_della.sbatch \
    desc_input/vmec_free_boundary/free_boundary_desc.py NCSX
"""

import argparse
import json
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, "..", "precise_QA_free_boundary"))

# the fixed-boundary NEST input of each case and the resolution of its free-boundary stage
CASES = {
    "NCSX": {"start": "NCSX_L16M16N12.h5", "fb_LMN": [12, 12, 8]},
    "QH_5beta": {"start": "QH_5beta_L16M16N12.h5", "fb_LMN": [12, 12, 10]},
    "CTH_like": {"start": "CTH_like_L12M12N8.h5", "fb_LMN": [12, 12, 8]},
}

parser = argparse.ArgumentParser()
parser.add_argument("case", choices=list(CASES))
parser.add_argument("--start", default=None, help="fixed-boundary equilibrium (default: the NEST input)")
parser.add_argument("--fb-LMN", type=int, nargs=3, default=None,
                    help="resolution of the free-boundary stage (default per case)")
parser.add_argument("--LMN", type=int, nargs=3, default=None,
                    help="final resolution (default: the start's)")
parser.add_argument("--k-list", type=int, nargs="*", default=[2, 4, 6])
parser.add_argument("--maxiter", type=int, default=100, help="last free-boundary stage")
parser.add_argument("--maxiter-intermediate", type=int, default=40)
parser.add_argument("--inner-ftol", type=float, default=1e-8,
                    help="of the equilibrium sub-solve; loose values stall the trust region")
parser.add_argument("--final-maxiter", type=int, default=300)
parser.add_argument("--jac-chunk-size", type=int, default=16, help="limits GPU memory")
parser.add_argument("--final-jac-chunk-size", type=int, default=200,
                    help="of the step-3 solve (16 16 12 needs one on a 40 GB A100)")
parser.add_argument("--resume", default=None, help="checkpoint h5 to continue from (skips step 1)")
parser.add_argument("--no-final", action="store_true", help="skip step 3 (more jobs follow)")
parser.add_argument("--outdir", default=None)
parser.add_argument("--device", default="gpu")
args = parser.parse_args()

from desc import set_device  # noqa: E402

set_device(args.device)

import jax  # noqa: E402

folder = os.path.join(HERE, args.case)
outdir = args.outdir or os.path.join(folder, "fb")
os.makedirs(outdir, exist_ok=True)
jax.config.update("jax_compilation_cache_dir", os.path.join(outdir, ".jax-cache"))
jax.config.update("jax_persistent_cache_min_entry_size_bytes", -1)
jax.config.update("jax_persistent_cache_min_compile_time_secs", 0)

import numpy as np  # noqa: E402

import desc.io  # noqa: E402
from desc.objectives import (  # noqa: E402
    BoundaryError,
    FixBoundaryR,
    FixBoundaryZ,
    FixCurrent,
    FixIota,
    FixPressure,
    FixPsi,
    ForceBalance,
    ObjectiveFunction,
)
from fb_tools import log, summary  # noqa: E402

case = CASES[args.case]
coils = desc.io.load(os.path.join(folder, f"{args.case}_coils.h5"))
diag_file = os.path.join(outdir, "diagnostics.json")
start_file = args.start or os.path.join(folder, case["start"])
start = desc.io.load(start_file)
L, M, N = args.LMN or (start.L, start.M, start.N)
fL, fM, fN = args.fb_LMN or case["fb_LMN"]

if args.resume is not None:
    eq = desc.io.load(args.resume)
    with open(diag_file) as f:
        diags = json.load(f)
    log(f"resuming from {args.resume}: L={eq.L} M={eq.M} N={eq.N}")
else:
    # -----------------------------------------------------------------------
    # 1. the fixed-boundary start at the free-boundary resolution
    # -----------------------------------------------------------------------
    eq = start
    diags = {"params": vars(args) | {"start": start_file},
             "start": summary(eq, coils, f"start {os.path.basename(start_file)} (fixed boundary)")}
    if (fL, fM, fN) != (eq.L, eq.M, eq.N):
        eq.change_resolution(L=fL, M=fM, N=fN, L_grid=2 * fL, M_grid=2 * fM, N_grid=2 * fN)
        eq.solve(ftol=1e-8, xtol=1e-12, gtol=1e-12, maxiter=200, verbose=1)
        diags["fixed_boundary_fb_resolution"] = summary(
            eq, coils, f"fixed boundary at L M N = {fL} {fM} {fN}")
    with open(diag_file, "w") as f:
        json.dump(diags, f, indent=2)

# ---------------------------------------------------------------------------
# 2. free boundary in the coils
# ---------------------------------------------------------------------------
if args.k_list:
    if max(args.k_list) >= min(eq.M, eq.N):
        raise ValueError(f"k = {max(args.k_list)} releases LCFS modes up to the resolution "
                         f"(M, N = {eq.M}, {eq.N}): the force error rises 10-100x")
    profile = FixIota(eq=eq) if eq.iota is not None else FixCurrent(eq=eq)
    constraints = (ForceBalance(eq=eq), FixPressure(eq=eq), profile, FixPsi(eq=eq))
    objective = ObjectiveFunction(BoundaryError(eq=eq, field=coils, field_fixed=True),
                                  jac_chunk_size=args.jac_chunk_size)
    history = diags.setdefault("history", [])
    for k in args.k_list:
        modes = eq.surface.R_basis.modes
        fixR = FixBoundaryR(eq=eq, modes=modes[np.max(np.abs(modes), 1) > k])
        modes = eq.surface.Z_basis.modes
        fixZ = FixBoundaryZ(eq=eq, modes=modes[np.max(np.abs(modes), 1) > k])
        log(f"free boundary: releasing LCFS modes with max(|m|,|n|) <= {k}")
        eq, out = eq.optimize(
            objective,
            constraints + (fixR, fixZ),
            optimizer="proximal-lsq-exact",
            maxiter=args.maxiter if k == args.k_list[-1] else args.maxiter_intermediate,
            ftol=1e-6,
            xtol=1e-10,
            gtol=1e-10,
            verbose=3,
            options={"solve_options": {"ftol": args.inner_ftol, "xtol": 1e-12, "gtol": 1e-12,
                                       "maxiter": 100}},
            copy=False,
        )
        history.append({"k": k, "nfev": int(out["nfev"]), "cost": float(out["cost"]),
                        "message": str(out["message"])})
        diags[f"free_boundary_k{k}"] = summary(eq, coils, f"free boundary k={k}")
        eq.save(os.path.join(outdir, f"free_boundary_k{k}.h5"))
        eq.save(os.path.join(outdir, "free_boundary_last.h5"))  # for --resume
        with open(diag_file, "w") as f:
            json.dump(diags, f, indent=2)

# ---------------------------------------------------------------------------
# 3. tight solve on the free-boundary LCFS at the final resolution, save
# ---------------------------------------------------------------------------
if args.no_final:
    log("done (free-boundary stage)")
    sys.exit()
if (L, M, N) != (eq.L, eq.M, eq.N):
    eq.change_resolution(L=L, M=M, N=N, L_grid=2 * L, M_grid=2 * M, N_grid=2 * N)
eq.solve(objective=ObjectiveFunction(ForceBalance(eq), jac_chunk_size=args.final_jac_chunk_size),
         ftol=1e-9, xtol=1e-12, gtol=1e-12, maxiter=args.final_maxiter, verbose=1)
tag = f"{args.case}_fb_L{L}M{M}N{N}"
diags["final"] = summary(eq, coils, tag)
eq.save(os.path.join(folder, f"{tag}.h5"))
for path in (diag_file, os.path.join(folder, f"{tag}_diagnostics.json")):
    with open(path, "w") as f:
        json.dump(diags, f, indent=2)
log(f"done: {tag}.h5")
