"""Steps 2 and 3: put iota = 1/2 back at rho = 0.55, and penalize the edge current
directly in the free-boundary solve.

Starts from ``precise_QA_free_boundary_smooth_J_fb_k5.h5`` (smooth current prescribed,
free-boundary LCFS with max(|m|, |n|) <= 5, iota = 1/2 at rho = 0.50). Repeats twice,
at L = M = N = 12:

a. scale the prescribed current (same shape, so dI/ds still vanishes at the edge) by
   a secant search until iota = 1/2 sits at RHO_TARGET, fixed boundary;
b. free-boundary solve in the coil field minimizing
       BoundaryError  +  w * J_parallel / J_ref on rho = 0.98 and 1,
   current, pressure and Psi fixed, LCFS modes max(|m|, |n|) <= 5 released.
   w is set so that the edge-current term carries RATIO times the cost of the
   boundary error at the start of the first pass.

Then the equilibrium is re-solved at its original resolution (L = M = 16, N = 14) with
the LCFS fixed and the results are reported.

usage: python fb_iota_edgeJ.py EQ.h5 COILS.h5 OUTDIR JAC_CHUNK RATIO [RHO_TARGET]
"""

import json
import os
import sys

from desc import set_device

set_device("gpu")

import numpy as np  # noqa: E402

import desc.io  # noqa: E402
from desc.grid import LinearGrid  # noqa: E402
from desc.objectives import (  # noqa: E402
    BoundaryError,
    FixCurrent,
    FixPressure,
    FixPsi,
    ForceBalance,
    ObjectiveFromUser,
    ObjectiveFunction,
)
from edge_tools import J_REF, edge_current, rho_of_iota  # noqa: E402
from edge_tools import show as _show  # noqa: E402
from fb_tools import free_boundary_stages, log, summary  # noqa: E402

eq = desc.io.load(sys.argv[1])
coils = desc.io.load(sys.argv[2])
outdir = sys.argv[3]
chunk = int(sys.argv[4])
ratio = float(sys.argv[5])
rho_target = float(sys.argv[6]) if len(sys.argv) > 6 else 0.55
os.makedirs(outdir, exist_ok=True)
diag_file = os.path.join(outdir, "diagnostics.json")
KMAX = 5


def show(label, e):
    _show(label, e, log)


def solve_fixed(tight=False):
    eq.solve(objective=ObjectiveFunction(ForceBalance(eq=eq), jac_chunk_size=chunk),
             ftol=1e-10 if tight else 1e-9, xtol=1e-12, gtol=1e-12,
             maxiter=300 if tight else 150, verbose=1)


def rescale_current():
    """Secant search on the current scale so that iota = 1/2 is at rho_target."""
    c0 = np.array(eq.current.params)

    def residual(scale):
        eq.current.params = scale * c0
        solve_fixed()
        r = rho_of_iota(eq)
        log(f"  current scale {scale:.5f}: iota = 1/2 at rho = {r}")
        return (r[0] if r else 0.0) - rho_target

    s_prev, f_prev = 1.0, residual(1.0)
    s = 1.03 if f_prev < 0 else 0.97  # larger current raises core iota (pushes out)
    for _ in range(6):
        f = residual(s)
        if abs(f) < 3e-3:
            break
        s, s_prev, f_prev = s - f * (s - s_prev) / (f - f_prev), s, f
    return s


def edge_term(weight):
    grid = LinearGrid(rho=np.array([0.98, 1.0]), M=2 * eq.M, N=2 * eq.N,
                      NFP=eq.NFP, sym=eq.sym)

    def edge_jpar(grid, data):
        return data["J*B"] / data["|B|"] / J_REF

    return ObjectiveFromUser(fun=edge_jpar, thing=eq, grid=grid, weight=weight,
                             name="edge J_parallel")


diags = {"start": summary(eq, coils, "start (fb_k5)")}
diags["start"]["edge current"] = edge_current(eq)
show("start edge current", diags["start"]["edge current"])

L0, M0, N0 = eq.L, eq.M, eq.N
eq.change_resolution(L=12, M=12, N=12, L_grid=24, M_grid=24, N_grid=24)
solve_fixed()
weight = None
for it in range(2):
    scale = rescale_current()
    diags[f"pass{it}_rescaled"] = summary(eq, coils, f"pass {it}: current x {scale:.5f}")
    if weight is None:  # calibrate the edge-current weight once
        bdry = ObjectiveFunction(BoundaryError(eq=eq, field=coils, field_fixed=True))
        bdry.build(verbose=0)
        unit = ObjectiveFunction(edge_term(1.0))
        unit.build(verbose=0)
        cost_b = float(np.sum(bdry.compute_scaled_error(bdry.x(eq)) ** 2))
        cost_j = float(np.sum(unit.compute_scaled_error(unit.x(eq)) ** 2))
        weight = float(np.sqrt(ratio * cost_b / cost_j))
        log(f"edge term weight {weight:.3e} (cost ratio {ratio}; boundary cost "
            f"{cost_b:.3e}, unit edge cost {cost_j:.3e})")
    objective = ObjectiveFunction(
        (BoundaryError(eq=eq, field=coils, field_fixed=True), edge_term(weight)),
        jac_chunk_size=32,
    )
    constraints = (ForceBalance(eq=eq, jac_chunk_size=chunk), FixCurrent(eq=eq),
                   FixPressure(eq=eq), FixPsi(eq=eq))
    eq = free_boundary_stages(
        eq, objective, constraints, [KMAX], 60 if it == 0 else 30, 1e-9, coils,
        diags, f"pass{it}_free_boundary", outdir, diag_file,
    )
    log(f"pass {it}: iota = 1/2 at rho = {rho_of_iota(eq)}")

eq.change_resolution(L=L0, M=M0, N=N0, L_grid=2 * L0, M_grid=2 * M0, N_grid=2 * N0)
log(f"back to L, M, N = {L0}, {M0}, {N0} with the LCFS fixed")
solve_fixed(tight=True)
diags["final"] = summary(eq, coils, "final")
diags["final"]["edge current"] = edge_current(eq)
diags["final"]["iota = 1/2 at rho"] = rho_of_iota(eq)
diags["params"] = {"edge weight": weight, "cost ratio": ratio, "rho target": rho_target,
                   "current coefficients": np.asarray(eq.current.params).tolist()}
show("final edge current", diags["final"]["edge current"])
eq.save(os.path.join(outdir, f"precise_QA_free_boundary_smooth_J_edgeJ_r{ratio:g}.h5"))
with open(diag_file, "w") as f:
    json.dump(diags, f, indent=2)
log("done")
