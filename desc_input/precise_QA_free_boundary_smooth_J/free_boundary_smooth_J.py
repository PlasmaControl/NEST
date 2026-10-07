"""Step 1: consistent free-boundary solve with the smooth-J profiles and a smooth LCFS.

Starts from ``precise_QA_free_boundary_smooth_J_bdry_k5.h5``: smooth toroidal current
(dI/ds ~ (1-s)^2 at the edge) and p ~ (1-rho^2)^3 prescribed, LCFS limited to
max(|m|, |n|) <= 5. That LCFS was obtained by trimming, so it is no longer the
free-boundary surface in the coil field. Here BoundaryError (B.n = 0 and
[[B^2/2mu0 + p]] = 0, plasma field by virtual casing) is minimized again in the field
of the 16 helical coils with:

- the current, pressure and Psi held fixed (iota is free to move slightly);
- force balance kept by the proximal projection at L = M = N = 12 (at 16/16/14 the
  proximal Jacobians do not fit in a 40 GB A100), then the equilibrium is re-solved
  at its original resolution with the new LCFS fixed;
- only LCFS modes with max(|m|, |n|) <= KMAX released (default 5), so the boundary
  cannot develop the short-wavelength wiggles that drive the residual edge current.

Afterwards it reports the free-boundary residuals, iota, and the edge current density:
surface-averaged, pointwise maximum, and the (4, 2) harmonic.

usage: python free_boundary_smooth_J.py EQ.h5 COILS.h5 OUTDIR JAC_CHUNK [KMAX [MAXITER]]
"""

import json
import os
import sys

from desc import set_device

set_device("gpu")

import desc.io  # noqa: E402
from desc.objectives import (  # noqa: E402
    BoundaryError,
    FixCurrent,
    FixPressure,
    FixPsi,
    ForceBalance,
    ObjectiveFunction,
)
from edge_tools import edge_current  # noqa: E402
from edge_tools import show as _show  # noqa: E402
from fb_tools import free_boundary_stages, log, summary  # noqa: E402


def show(label, e):
    _show(label, e, log)

eq = desc.io.load(sys.argv[1])
coils = desc.io.load(sys.argv[2])
outdir = sys.argv[3]
chunk = int(sys.argv[4])
kmax = int(sys.argv[5]) if len(sys.argv) > 5 else 5
maxiter = int(sys.argv[6]) if len(sys.argv) > 6 else 60
os.makedirs(outdir, exist_ok=True)
diag_file = os.path.join(outdir, "diagnostics.json")
diags = {"start": summary(eq, coils, "start: trimmed LCFS (k <= 5)")}
diags["start"]["edge current"] = edge_current(eq)
show("start edge current", diags["start"]["edge current"])

# free-boundary solve at L = M = N = 12, final solve back at the original resolution
L0, M0, N0 = eq.L, eq.M, eq.N
eq.change_resolution(L=12, M=12, N=12, L_grid=24, M_grid=24, N_grid=24)
eq.solve(objective=ObjectiveFunction(ForceBalance(eq=eq), jac_chunk_size=chunk),
         ftol=1e-10, xtol=1e-12, gtol=1e-12, maxiter=200, verbose=1)
log(f"resolution {L0},{M0},{N0} -> 12,12,12 for the free-boundary solve")

constraints = (
    ForceBalance(eq=eq, jac_chunk_size=chunk),
    FixCurrent(eq=eq),
    FixPressure(eq=eq),
    FixPsi(eq=eq),
)
objective = ObjectiveFunction(BoundaryError(eq=eq, field=coils, field_fixed=True),
                              jac_chunk_size=32)
eq = free_boundary_stages(
    eq, objective, constraints, [kmax], maxiter, 1e-9, coils, diags,
    "free_boundary", outdir, diag_file,
)
eq.change_resolution(L=L0, M=M0, N=N0, L_grid=2 * L0, M_grid=2 * M0, N_grid=2 * N0)
log(f"back to L, M, N = {L0}, {M0}, {N0} with the free-boundary LCFS fixed")
eq.solve(
    objective=ObjectiveFunction(ForceBalance(eq=eq), jac_chunk_size=chunk),
    ftol=1e-10, xtol=1e-12, gtol=1e-12, maxiter=300, verbose=1,
)
diags["final"] = summary(eq, coils, "final")
diags["final"]["edge current"] = edge_current(eq)
show("final edge current", diags["final"]["edge current"])
eq.save(os.path.join(outdir, "precise_QA_free_boundary_smooth_J_fb_k5.h5"))
with open(diag_file, "w") as f:
    json.dump(diags, f, indent=2)
log("done")
