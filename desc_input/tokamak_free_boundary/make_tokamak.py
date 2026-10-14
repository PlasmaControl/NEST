"""Axisymmetric free-boundary Grad-Shafranov equilibrium with J -> 0 at the LCFS.

The D-shaped tokamak of DESC's DSHAPE_CURRENT example (R0 = 3.5 m, a = 1 m,
kappa = 1.47, Psi = 1 Wb) with smooth profiles:

- p = p0 (1 - rho^2)^3, so p' and p'' vanish at the edge;
- the toroidal current I(s) = I0 (1 - (1 - s)^3), s = rho^2, so dI/ds = 3 I0 (1 - s)^2
  and the toroidal current density and its radial slope vanish at the edge.

In axisymmetry J_parallel has no 3D part, so J -> 0 at the LCFS with zero slope.

External field. With J -> 0 (no surface current) the fixed-boundary solution is a
free-boundary one: B_ext = B_DESC - B_plasma(J_DESC) is curl- and divergence-free in
the plasma, (B_ext + B_plasma).n = 0 on the LCFS, and B is continuous across it.
That B_ext is realized by a FourierCurrentPotentialField on a winding surface
offset from the LCFS: G is set by the toroidal field, and the single-valued
Phi(theta) by REGCOIL, minimizing B.n on the LCFS. Its B.n/|B| and pressure jump
(virtual casing) are in the diagnostics.

Saves tokamak_L{L}M{M}.h5 (the equilibrium), tokamak_L{L}M{M}_field.h5 (the
current potential) and tokamak_L{L}M{M}_diagnostics.json.

usage: python make_tokamak.py [--LM 12 12] [--offset 0.8] [--M-phi 16]
    [--lam 1e-20] [--device gpu]
"""

import argparse
import json
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, "..", "precise_QA_free_boundary"))

ap = argparse.ArgumentParser()
ap.add_argument("--LM", type=int, nargs=2, default=[12, 12])
ap.add_argument("--p0", type=float, default=1.6e3, help="Pa")
ap.add_argument("--I0", type=float, default=-1.2e5, help="A, the net toroidal current")
ap.add_argument("--offset", type=float, default=0.8, help="winding surface offset [m]")
ap.add_argument("--M-phi", type=int, default=16, help="poloidal modes of Phi")
ap.add_argument("--lam", type=float, default=1e-20, help="REGCOIL regularization")
ap.add_argument("--device", default="gpu")
a = ap.parse_args()

from desc import set_device  # noqa: E402

set_device(a.device)

import numpy as np  # noqa: E402

from desc.continuation import solve_continuation_automatic  # noqa: E402
from desc.equilibrium import Equilibrium  # noqa: E402
from desc.geometry import FourierRZToroidalSurface  # noqa: E402
from desc.grid import LinearGrid  # noqa: E402
from desc.magnetic_fields import (FourierCurrentPotentialField,  # noqa: E402
                                  solve_regularized_surface_current)
from desc.profiles import PowerSeriesProfile  # noqa: E402
from fb_tools import force_error, log, summary  # noqa: E402

L, M = a.LM
tag = os.path.join(HERE, f"tokamak_L{L}M{M}")

# DSHAPE_CURRENT boundary: R = R_mn, Z = Z_mn (m < 0: sin)
surface = FourierRZToroidalSurface(
    R_lmn=[3.51, 1.0, 0.106], modes_R=[[0, 0], [1, 0], [2, 0]],
    Z_lmn=[-1.47, 0.16], modes_Z=[[-1, 0], [-2, 0]], NFP=1, sym=True)
# (1 - rho^2)^3 = 1 - 3 rho^2 + 3 rho^4 - rho^6
w = np.array([1, 0, -3, 0, 3, 0, -1], float)
pressure = PowerSeriesProfile(a.p0 * w, modes=np.arange(7))
current = PowerSeriesProfile(a.I0 * (np.eye(7)[0] - w), modes=np.arange(7))
eq = Equilibrium(surface=surface, pressure=pressure, current=current, Psi=1.0, NFP=1,
                 L=L, M=M, N=0, L_grid=2 * L, M_grid=2 * M, N_grid=0, sym=True)
eqf = solve_continuation_automatic(eq, objective="force", ftol=1e-10, xtol=1e-12,
                                   gtol=1e-12, maxiter=200, verbose=2)
eq = eqf[-1]
eq.solve(objective="force", ftol=1e-12, xtol=1e-14, gtol=1e-14, maxiter=200, verbose=2)
log(f"force error {force_error(eq):.3e}")
eq.save(tag + ".h5")

# winding surface and REGCOIL current potential (axisymmetric: N_Phi = 0)
ws = eq.surface.constant_offset_surface(a.offset, M=max(8, M), N=0)
field = FourierCurrentPotentialField.from_surface(ws, M_Phi=a.M_phi, N_Phi=0)
eval_grid = LinearGrid(M=4 * M, N=0, NFP=1, sym=False)
vc_grid = LinearGrid(rho=np.array([1.0]), M=8 * M, N=64, NFP=1, sym=False, endpoint=False)
fields, data = solve_regularized_surface_current(
    field, eq, lambda_regularization=a.lam, current_helicity=(1, 0),
    eval_grid=eval_grid, vc_source_grid=vc_grid,
    source_grid=LinearGrid(M=4 * a.M_phi, N=64, NFP=1), verbose=2)
field = fields[0]
log(f"G = {float(data['G']):.6e} A, I = {float(data['I']):.3e} A, "
    f"chi^2_B = {float(np.ravel(data['chi^2_B'])[0]):.3e}")
field.save(tag + "_field.h5")

diags = {"args": vars(a), "eq": summary(eq, field, f"tokamak L{L} M{M}"),
         "G": float(data["G"]), "chi^2_B": float(np.ravel(data["chi^2_B"])[0])}
rho = np.linspace(0, 1, 11)
g = LinearGrid(rho=rho, M=2 * M, N=0, NFP=1)
d = eq.compute(["q"], grid=g)
diags["q(rho=0..1)"] = g.compress(d["q"]).tolist()
with open(tag + "_diagnostics.json", "w") as f:
    json.dump(diags, f, indent=1)
log(f"saved {tag}.h5, {tag}_field.h5")
