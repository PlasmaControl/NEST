"""Re-solve a free-boundary result at higher resolution with its LCFS held fixed.

Releasing boundary modes close to the equilibrium resolution lowers the boundary
error but raises the force error (the interior cannot follow the boundary). Raising
L, M, N afterwards and re-solving force balance with the free-boundary LCFS fixed
recovers the force error while keeping the boundary (and its B.n error) unchanged.

usage: python refine_resolution.py EQ.h5 FIELD.h5 L M N OUT.h5 [JAC_CHUNK_SIZE]
"""

import json
import sys

from desc import set_device

set_device("gpu")

import desc.io  # noqa: E402
from desc.objectives import ForceBalance, ObjectiveFunction  # noqa: E402
from fb_tools import log, summary  # noqa: E402

eq = desc.io.load(sys.argv[1])
field = desc.io.load(sys.argv[2])
L, M, N = (int(v) for v in sys.argv[3:6])
out = sys.argv[6]
chunk = int(sys.argv[7]) if len(sys.argv) > 7 else None  # limits GPU memory

before = summary(eq, field, f"before (L,M,N = {eq.L},{eq.M},{eq.N})")
eq.change_resolution(L=L, M=M, N=N, L_grid=2 * L, M_grid=2 * M, N_grid=2 * N)
eq.solve(
    objective=ObjectiveFunction(ForceBalance(eq=eq), jac_chunk_size=chunk),
    ftol=1e-10, xtol=1e-12, gtol=1e-12, maxiter=300, verbose=1,
)
after = summary(eq, field, f"after (L,M,N = {L},{M},{N})")
eq.save(out)
with open(out.replace(".h5", "_diagnostics.json"), "w") as f:
    json.dump({"before": before, "after": after}, f, indent=2)
log("done")
