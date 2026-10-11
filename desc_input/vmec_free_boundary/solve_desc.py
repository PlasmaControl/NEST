"""DESC equilibrium on the LCFS of a VMEC free-boundary run, checked in its coils.

Starts from ``<case>_vmec.h5`` (the wout fitted by ``check_vmec_free_boundary.py``),
keeps the VMEC LCFS, replaces the spline profiles of the fit by the VMEC pressure
and net toroidal current as even power series in rho (``vmec_profiles.py``: the
fitted current spline makes iota spike at the axis), raises the resolution and
solves force balance as a fixed-boundary problem. The result is
saved as ``<case>_L{L}M{M}N{N}.h5`` with the same diagnostics as the precise_QA
inputs (force error, B.n/|B| and pressure jump in the coils, beta, iota, rational
surfaces) in ``<case>_L{L}M{M}N{N}_diagnostics.json``.

Usage: python solve_desc.py CASE --LMN 12 12 8 [--device cpu] [--jac-chunk-size n]
"""

import argparse
import json
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, "..", "precise_QA_free_boundary"))
sys.path.insert(0, HERE)

parser = argparse.ArgumentParser()
parser.add_argument("case")
parser.add_argument("--LMN", type=int, nargs=3, default=[12, 12, 8])
parser.add_argument("--device", default="cpu")
parser.add_argument("--maxiter", type=int, default=100)
parser.add_argument("--jac-chunk-size", type=int, default=None,
                    help="limits GPU memory (16 16 12 needs one on a 40 GB A100)")
args = parser.parse_args()

from desc import set_device  # noqa: E402

set_device(args.device)

import desc.io  # noqa: E402
from desc.objectives import ForceBalance, ObjectiveFunction  # noqa: E402
from fb_tools import log, summary  # noqa: E402
from vmec_profiles import vmec_profiles  # noqa: E402

WOUT = {
    "NCSX": "wout_ncsx_c09r00_free.nc",
    "QH_5beta": "wout_LBDQH5_coils.nc",
    "CTH_like": "wout_cth_like_free_bdy.nc",
}

folder = os.path.join(HERE, args.case)
eq = desc.io.load(os.path.join(folder, f"{args.case}_vmec.h5"))
coils = desc.io.load(os.path.join(folder, f"{args.case}_coils.h5"))
diags = {"vmec_fit": summary(eq, coils, f"{args.case} VMEC fit")}

pressure, current, info = vmec_profiles(os.path.join(folder, WOUT[args.case]))
log(f"{args.case}: VMEC profiles {info}")
eq.pressure, eq.current = pressure, current
diags["profiles"] = info

L, M, N = args.LMN
eq.change_resolution(L=L, M=M, N=N, L_grid=2 * L, M_grid=2 * M, N_grid=2 * N)
log(f"{args.case}: fixed-boundary solve at L M N = {L} {M} {N}")
objective = ObjectiveFunction(
    ForceBalance(eq), jac_chunk_size=args.jac_chunk_size or "auto"
)
eq.solve(objective=objective, ftol=1e-10, xtol=1e-12, gtol=1e-12,
         maxiter=args.maxiter, verbose=2)
tag = f"{args.case}_L{L}M{M}N{N}"
eq.save(os.path.join(folder, f"{tag}.h5"))
diags["desc"] = summary(eq, coils, tag)
with open(os.path.join(folder, f"{tag}_diagnostics.json"), "w") as f:
    json.dump(diags, f, indent=2)
