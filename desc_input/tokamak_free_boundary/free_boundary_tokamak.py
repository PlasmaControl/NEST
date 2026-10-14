"""Free-boundary check of the tokamak in its FourierCurrentPotentialField.

``make_tokamak.py`` solved a fixed-boundary equilibrium and fitted the current
potential (REGCOIL) to cancel B.n on that LCFS. Here the LCFS is released in that
field, held fixed (Phi_mn, G), and DESC solves the free-boundary problem:
``BoundaryError`` (B.n = 0 and [[B^2 + 2 mu0 p]] = 0 on the LCFS, plasma field by
virtual casing) minimized under force balance with p, I and Psi fixed, by the
proximal projection, releasing the LCFS modes |m| <= k stage by stage.

How far the boundary moves measures how free-boundary consistent the start was.
Saves tokamak_L{L}M{M}_fb.h5, _fb_diagnostics.json (BoundaryError, force error
and the displacement after each stage) and _fb.png (both LCFS, the displacement
against theta, B.n / |B| and the pressure jump before and after).

usage: python free_boundary_tokamak.py [--LM 16 16] [--k 4 8 12] [--device gpu]
"""

import argparse
import json
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, "..", "precise_QA_free_boundary"))

ap = argparse.ArgumentParser()
ap.add_argument("--LM", type=int, nargs=2, default=[16, 16])
ap.add_argument("--k", type=int, nargs="+", default=[4, 8, 12],
                help="release LCFS modes |m| <= k, stage by stage")
ap.add_argument("--maxiter", type=int, default=50, help="per stage")
ap.add_argument("--device", default="cpu")
a = ap.parse_args()

from desc import set_device  # noqa: E402

set_device(a.device)

import matplotlib  # noqa: E402

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

import desc.io  # noqa: E402
from desc.grid import LinearGrid  # noqa: E402
from desc.objectives import (BoundaryError, FixBoundaryR, FixBoundaryZ,  # noqa: E402
                             FixCurrent, FixPressure, FixPsi, ForceBalance,
                             ObjectiveFunction)
from fb_tools import log, summary  # noqa: E402

L, M = a.LM
tag = os.path.join(HERE, f"tokamak_L{L}M{M}")
eq0 = desc.io.load(tag + ".h5")
field = desc.io.load(tag + "_field.h5")
eq = eq0.copy()
# the surface current's quadrature: fine in zeta (N = 0 winding surface)
field_grid = LinearGrid(M=max(64, 4 * field.M_Phi), N=128, NFP=1)
th = np.linspace(0, 2 * np.pi, 513)


def lcfs(e):
    d = e.compute(["R", "Z"], grid=LinearGrid(rho=1.0, theta=th, zeta=0.0))
    return np.asarray(d["R"]), np.asarray(d["Z"])


def displacement(e):
    """Distance of each point of e's LCFS from eq0's (dense polyline), signed by
    whether it lies outside (+) or inside (-) eq0's LCFS."""
    from matplotlib.path import Path
    R0, Z0 = lcfs(eq0)
    R1, Z1 = lcfs(e)
    t = np.linspace(0, 1, 50)[:, None]
    seg = [(R0[:-1] + t * np.diff(R0), Z0[:-1] + t * np.diff(Z0))]
    P = np.stack([seg[0][0].ravel(), seg[0][1].ravel()], 1)
    dist = np.min(np.hypot(R1[:, None] - P[:, 0], Z1[:, None] - P[:, 1]), 1)
    inside = Path(np.stack([R0, Z0], 1)).contains_points(np.stack([R1, Z1], 1))
    return np.where(inside, -dist, dist)


def bdry_residuals(e):
    obj = BoundaryError(eq=e, field=field, field_fixed=True, field_grid=field_grid)
    obj.build(verbose=0)
    f = np.asarray(obj.compute(*obj.xs(e)))
    grid = obj.constants["eval_transforms"]["grid"]
    d = e.compute(["|e_theta x e_zeta|", "|B|"], grid=grid)
    n = grid.num_nodes
    t = np.asarray(grid.nodes[:, 1])
    o = np.argsort(t)
    return (t[o], (f[:n] / (d["|e_theta x e_zeta|"] * d["|B|"]))[o],
            (f[n : 2 * n] / (d["|e_theta x e_zeta|"] * d["|B|"] ** 2))[o])


diags = {"args": vars(a), "start": summary(eq, field, "start (fixed boundary)")}
res0 = bdry_residuals(eq)
objective = ObjectiveFunction(BoundaryError(eq=eq, field=field, field_fixed=True,
                                            field_grid=field_grid))
constraints = (ForceBalance(eq=eq), FixPressure(eq=eq), FixCurrent(eq=eq), FixPsi(eq=eq))
for k in a.k:
    modes = eq.surface.R_basis.modes
    fixR = FixBoundaryR(eq=eq, modes=modes[np.max(np.abs(modes), 1) > k])
    modes = eq.surface.Z_basis.modes
    fixZ = FixBoundaryZ(eq=eq, modes=modes[np.max(np.abs(modes), 1) > k])
    log(f"releasing LCFS modes |m| <= {k}")
    eq, out = eq.optimize(objective, constraints + (fixR, fixZ), optimizer="proximal-lsq-exact",
                          maxiter=a.maxiter, ftol=1e-10, xtol=1e-12, gtol=1e-12, verbose=2,
                          options={"solve_options": {"ftol": 1e-10, "xtol": 1e-12,
                                                     "gtol": 1e-12, "maxiter": 100}},
                          copy=False)
    dn = displacement(eq)
    s = summary(eq, field, f"free boundary |m| <= {k}")
    s.update({"message": str(out["message"]), "nit": int(out["nit"]),
              "max |displacement| [m]": float(np.abs(dn).max()),
              "rms displacement [m]": float(np.sqrt(np.mean(dn**2))),
              "max |displacement| / a": float(np.abs(dn).max() / s["a"])})
    log(f"|m| <= {k}: LCFS moved max {np.abs(dn).max():.2e} m, rms {np.sqrt(np.mean(dn**2)):.2e} m")
    diags[f"k{k}"] = s
    eq.save(tag + "_fb.h5")
    with open(tag + "_fb_diagnostics.json", "w") as f:
        json.dump(diags, f, indent=1)

# figure
res1 = bdry_residuals(eq)
fig, ax = plt.subplots(1, 4, figsize=(18, 4.5))
R0, Z0 = lcfs(eq0)
R1, Z1 = lcfs(eq)
ax[0].plot(R0, Z0, "k-", label="fixed-boundary start")
ax[0].plot(R1, Z1, "C3--", label="free-boundary solution")
ax[0].set_aspect("equal")
ax[0].legend(fontsize=8)
ax[0].set_title("LCFS")
ax[1].plot(th, displacement(eq) * 1e3)
ax[1].set_xlabel(r"$\theta$")
ax[1].set_ylabel("displacement [mm] (+ outward)")
ax[1].set_title(f"LCFS displacement (a = {diags['start']['a']:.2f} m)")
for i, (name, j) in enumerate((("B.n / |B|", 1), (r"$[[B^2+2\mu_0 p]]/B^2$", 2))):
    ax[2 + i].plot(res0[0], res0[j], ".-", ms=3, label="start")
    ax[2 + i].plot(res1[0], res1[j], ".-", ms=3, label="free boundary")
    ax[2 + i].set_xlabel(r"$\theta$")
    ax[2 + i].set_title(name + " on the LCFS")
    ax[2 + i].legend(fontsize=8)
fig.tight_layout()
fig.savefig(tag + "_fb.png", dpi=130)
log(f"saved {tag}_fb.h5, _fb_diagnostics.json, _fb.png")
