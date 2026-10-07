"""Test 1: does enforcing force balance at the edge remove the residual edge current?

DESC's ForceBalance collocates on a QuadratureGrid whose radial (Gauss-Jacobi) nodes
are all strictly inside rho = 1 and sparse near it, so the last few percent of the
radius are weakly constrained. Here the smooth-J equilibrium is re-solved, with the
same LCFS and profiles, with an extra ForceBalance term on surfaces
rho = 0.95 ... 1.0 (including the LCFS), for each weight given.

With p' = 0 and dI/drho = 0 at the edge, an exact equilibrium has J = 0 there point
by point, so the edge current should drop if the diagnosis is right.

usage: python edge_force_test.py EQ.h5 COILS.h5 OUTDIR JAC_CHUNK_SIZE W [W ...]
"""

import json
import os
import sys

from desc import set_device

set_device("gpu")

import numpy as np  # noqa: E402

import desc.io  # noqa: E402
from desc.grid import LinearGrid  # noqa: E402
from desc.integrals.surface_integral import surface_averages  # noqa: E402
from desc.objectives import ForceBalance, ObjectiveFunction  # noqa: E402
from fb_tools import force_error, log, summary  # noqa: E402

eq0 = desc.io.load(sys.argv[1])
coils = desc.io.load(sys.argv[2])
outdir = sys.argv[3]
chunk = int(sys.argv[4])
weights = [float(w) for w in sys.argv[5:]]
os.makedirs(outdir, exist_ok=True)
J_REF = 3.9e5  # peak <|J|> of the smooth-J equilibrium [A/m^2]
rho_edge = np.array([0.95, 0.96, 0.97, 0.98, 0.99, 1.0])
rho_tab = np.array([0.5, 0.9, 0.96, 0.98, 0.99, 1.0])


def edge_table(eq):
    g = LinearGrid(rho=rho_tab, M=eq.M_grid, N=eq.N_grid, NFP=eq.NFP, sym=False)
    d = eq.compute(["|J|", "J*B", "|B|", "sqrt(g)", "|F|",
                    "<|grad(|B|^2)|/2mu0>_vol"], grid=g)

    def avg(q):
        return np.asarray(surface_averages(g, q, d["sqrt(g)"], expand_out=False))

    jpar = np.abs(d["J*B"] / d["|B|"])
    jpar_max = g.meshgrid_reshape(jpar, "rtz").max(axis=(1, 2))
    return {
        "rho": rho_tab.tolist(),
        "<|J|>/J_ref": (avg(d["|J|"]) / J_REF).tolist(),
        "net |<J.B>|/<B^2>^.5/J_ref": (np.abs(avg(d["J*B"]))
                                       / np.sqrt(avg(d["|B|"] ** 2)) / J_REF).tolist(),
        "max |J_par|/J_ref": (jpar_max / J_REF).tolist(),
        "local force error": (avg(d["|F|"])
                              / float(d["<|grad(|B|^2)|/2mu0>_vol"])).tolist(),
    }


def show(label, t):
    log(label)
    print("  rho   <|J|>     max|J_par|  net <J.B>   force error", flush=True)
    for i, r in enumerate(t["rho"]):
        print(f"  {r:.2f}  {t['<|J|>/J_ref'][i]:.3e}  {t['max |J_par|/J_ref'][i]:.3e}  "
              f"{t['net |<J.B>|/<B^2>^.5/J_ref'][i]:.3e}  {t['local force error'][i]:.2e}",
              flush=True)


results = {"reference": edge_table(eq0)}
results["reference"]["volume force error"] = force_error(eq0)
show("reference (volume ForceBalance only)", results["reference"])
for w in weights:
    eq = eq0.copy()
    edge_grid = LinearGrid(rho=rho_edge, M=2 * eq.M, N=2 * eq.N, NFP=eq.NFP,
                           sym=eq.sym)
    objective = ObjectiveFunction(
        (ForceBalance(eq=eq),
         ForceBalance(eq=eq, grid=edge_grid, weight=w, name="edge force")),
        jac_chunk_size=chunk,
    )
    log(f"edge ForceBalance weight {w}: solving")
    eq.solve(objective=objective, ftol=1e-10, xtol=1e-12, gtol=1e-12, maxiter=300,
             verbose=1)
    t = edge_table(eq)
    t["volume force error"] = force_error(eq)
    t.update({k: v for k, v in summary(eq, coils, f"edge weight {w}").items()
              if k in ("Bn/|B| rms", "Bn/|B| max", "[[B^2+2mu0p]]/B^2 rms",
                       "rational_surfaces")})
    results[f"edge_weight_{w:g}"] = t
    show(f"edge weight {w}: volume force error {t['volume force error']:.2e}", t)
    eq.save(os.path.join(outdir, f"smooth_J_edge_force_w{w:g}.h5"))
    with open(os.path.join(outdir, "edge_force_test.json"), "w") as f:
        json.dump(results, f, indent=2)
log("done")
