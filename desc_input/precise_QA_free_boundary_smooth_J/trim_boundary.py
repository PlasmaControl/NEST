"""Method 1 against the residual edge current: low-pass the LCFS and re-solve.

The zero-average J_parallel left at rho = 1 in the smooth-J equilibrium sits where the
free-boundary LCFS has short-wavelength wiggles (inboard, mid-period, where the
helical coils come closest). Here the LCFS Fourier modes with max(|m|, |n|) > KCUT are
set to zero and the equilibrium is re-solved (same smooth current and pressure). The
price is a worse match to the coil field, which is measured.

usage: python trim_boundary.py EQ.h5 COILS.h5 OUTDIR JAC_CHUNK_SIZE KCUT [KCUT ...]
"""

import json
import os
import sys

from desc import set_device

set_device("gpu")

import numpy as np  # noqa: E402

import desc.io  # noqa: E402
from desc.objectives import ForceBalance, ObjectiveFunction  # noqa: E402
from fb_tools import log, summary  # noqa: E402

eq0 = desc.io.load(sys.argv[1])
coils = desc.io.load(sys.argv[2])
outdir = sys.argv[3]
chunk = int(sys.argv[4])
kcuts = [int(k) for k in sys.argv[5:]]
os.makedirs(outdir, exist_ok=True)
diags = {}
for kcut in kcuts:
    eq = eq0.copy()
    for name, basis in [("Rb_lmn", eq.surface.R_basis), ("Zb_lmn", eq.surface.Z_basis)]:
        x = np.array(getattr(eq, name))
        x[np.max(np.abs(basis.modes[:, 1:]), axis=1) > kcut] = 0.0
        setattr(eq, name, x)
    log(f"kcut = {kcut}: boundary modes with max(|m|,|n|) > {kcut} removed; solving")
    eq.solve(
        objective=ObjectiveFunction(ForceBalance(eq=eq), jac_chunk_size=chunk),
        ftol=1e-10, xtol=1e-12, gtol=1e-12, maxiter=300, verbose=1,
    )
    diags[f"kcut_{kcut}"] = summary(eq, coils, f"boundary trimmed to |m|,|n| <= {kcut}")
    out = os.path.join(outdir, f"precise_QA_smooth_J_bdry_k{kcut}.h5")
    eq.save(out)
    with open(os.path.join(outdir, "trim_diagnostics.json"), "w") as f:
        json.dump(diags, f, indent=2)
log("done")
