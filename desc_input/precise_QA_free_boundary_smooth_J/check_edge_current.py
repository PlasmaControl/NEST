"""Edge behaviour of the current density of DESC equilibria.

For each equilibrium prints, on surfaces approaching the LCFS (relative to
J_REF = 3.9e5 A/m^2, the peak <|J|> of the smooth-J case):
- <|J|>: surface-averaged current density magnitude;
- |<J.B>| / <B^2>^(1/2): the net (surface-averaged) parallel current;
- <|J_par|>: surface-averaged |J.B| / |B| (includes zero-average oscillations);
- <|F|> / <|grad B^2/2mu0|>_vol: local force error.

usage: python check_edge_current.py EQ1.h5 [EQ2.h5 ...]
"""

import sys

import numpy as np

import desc.io
from desc.grid import LinearGrid
from desc.integrals.surface_integral import surface_averages

J_REF = 3.9e5
rho = np.array([0.5, 0.8, 0.9, 0.94, 0.96, 0.98, 0.99, 1.0])
for f in sys.argv[1:]:
    eq = desc.io.load(f)
    g = LinearGrid(rho=rho, M=eq.M_grid, N=eq.N_grid, NFP=eq.NFP, sym=False)
    d = eq.compute(
        ["|J|", "J*B", "|B|", "sqrt(g)", "|F|", "<|grad(|B|^2)|/2mu0>_vol"], grid=g
    )

    def avg(q):
        return np.asarray(surface_averages(g, q, d["sqrt(g)"], expand_out=False))

    J = avg(d["|J|"]) / J_REF
    J_net = np.abs(avg(d["J*B"])) / np.sqrt(avg(d["|B|"] ** 2)) / J_REF
    J_par = avg(np.abs(d["J*B"] / d["|B|"])) / J_REF
    F = avg(d["|F|"]) / float(d["<|grad(|B|^2)|/2mu0>_vol"])
    print(f"{f}  (L, M, N = {eq.L}, {eq.M}, {eq.N})")
    print("  rho   <|J|>      |<J.B>|/<B^2>^.5   <|J_par|>   force error")
    for i, r in enumerate(rho):
        print(f"  {r:.2f}  {J[i]:.3e}  {J_net[i]:.3e}          {J_par[i]:.3e}   {F[i]:.2e}")
