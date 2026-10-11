"""Diagnostics of the smooth-edge DSHAPE equilibria in their coils -> diagnostics.json.

    python check_dshape_smooth.py [--nest]

For DSHAPE_free_boundary_smooth.h5 (L, M, N = 26, 13, 0) and ..._M20.h5 (40, 20, 0):
  - force error <|F|>_vol / <|grad(|B|^2)|/2mu0>_vol on QuadratureGrid(L_grid, M_grid, N_grid)
  - <beta>_vol, minor radius, R0, iota on the axis
  - toroidal current I, iota, rms |J| and rms |F| on the surfaces rho = 0.5, 0.9, 0.98, 0.995, 1
  - B.n / |B| rms on the LCFS in the coils, B = B_coils + B_plasma (virtual casing)
With --nest also NEST's own checks (``DESCInput.load(..., field=coils)``, ``check``)
and the force residual at iteration 0 (``ForceResidual``, fitted J and p, h = a / 4).
"""

import json
import os
import sys

import numpy as np
from desc import set_device

set_device(os.environ.get("NEST_DEVICE", "cpu"))

import desc.io  # noqa: E402
from desc.grid import LinearGrid, QuadratureGrid  # noqa: E402
from desc.integrals.singularities import compute_B_plasma  # noqa: E402

HERE = os.path.dirname(os.path.abspath(__file__))
FILES = ["DSHAPE_free_boundary_smooth.h5", "DSHAPE_free_boundary_smooth_M20.h5"]
COILS = [os.path.join(HERE, "DSHAPE_smooth_TF.h5"), os.path.join(HERE, "DSHAPE_smooth_PF_coils.h5")]


def scalar(eq, key, grid):
    return float(np.ravel(eq.compute(key, grid=grid)[key])[0])


def diagnostics(eq, field):
    qg = QuadratureGrid(L=eq.L_grid, M=eq.M_grid, N=eq.N_grid, NFP=eq.NFP)
    out = {"L, M, N": [eq.L, eq.M, eq.N],
           "force_error": scalar(eq, "<|F|>_vol", qg) / scalar(eq, "<|grad(|B|^2)|/2mu0>_vol", qg),
           "beta_vol": scalar(eq, "<beta>_vol", qg), "a": scalar(eq, "a", qg),
           "R0": scalar(eq, "R0", qg)}
    g = LinearGrid(rho=np.array([0.0]), M=0, N=0, NFP=1)
    out["iota_axis"] = float(np.ravel(eq.compute("iota", grid=g)["iota"])[0])
    for r in (0.5, 0.9, 0.98, 0.995, 1.0):
        g = LinearGrid(rho=np.array([r]), M=2 * eq.M + 2, N=0, NFP=1)
        d = eq.compute(["current", "iota", "|J|", "|F|"], grid=g)
        out[f"rho={r:g}"] = {"I_A": float(np.ravel(d["current"])[0]),
                             "iota": float(np.ravel(d["iota"])[0]),
                             "J_rms_A_m2": float(np.sqrt(np.mean(np.asarray(d["|J|"]) ** 2))),
                             "F_rms_N_m3": float(np.sqrt(np.mean(np.asarray(d["|F|"]) ** 2)))}
    g = LinearGrid(rho=np.array([1.0]), M=96, N=0, NFP=64, sym=False)
    d = eq.compute(["R", "phi", "Z", "n_rho", "|B|"], grid=g)
    X = np.stack([d["R"], d["phi"], d["Z"]], -1)
    B_coils_n = np.sum(np.asarray(field.compute_magnetic_field(X, basis="rpz"))
                       * np.asarray(d["n_rho"]), -1)
    B_plasma_n = np.asarray(compute_B_plasma(eq, g, normal_only=True))
    out["Bn_over_B_rms"] = float(np.sqrt(np.mean(((B_coils_n + B_plasma_n)
                                                  / np.asarray(d["|B|"])) ** 2)))
    return out


def nest_check(path):
    sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(HERE)), "nest"))
    from boundary_potential import BoundaryPotential
    from desc_input import DESCInput
    from force import ForceResidual
    from grid import CartesianGrid
    from pressure import Pressure

    inp = DESCInput.load(path, field=COILS)  # runs DESCInput.check
    eq = inp.eq
    current = BoundaryPotential.fit(eq, eq.L, eq.M, eq.N)
    pressure = Pressure.fit(eq, eq.L, eq.M, eq.N, edge_order=2)
    box = CartesianGrid.periodic_box(inp, inp.a / 4).locate_plasma(inp)
    fr = ForceResidual(inp, box)
    m = fr.metrics(fr.evaluate(current, pressure)["F"])
    return {"summary": {k: (v if isinstance(v, (int, float, str)) else str(v))
                        for k, v in inp.summary().items()},
            "box": list(box.shape), "inside_points": int(len(box.inside)),
            "iteration0_F_over_norm": {"mean": m["mean"], "rms": m["rms"]},
            "DESC_F_over_norm": fr.F_desc_vol / fr.norm}


if __name__ == "__main__":
    from desc.magnetic_fields import SumMagneticField
    field = SumMagneticField([desc.io.load(f) for f in COILS])
    out = {}
    for f in FILES:
        eq = desc.io.load(os.path.join(HERE, f))
        eq = eq[-1] if hasattr(eq, "__len__") else eq
        out[f] = diagnostics(eq, field)
        if "--nest" in sys.argv:
            out[f]["nest"] = nest_check(os.path.join(HERE, f))
        print(f, json.dumps(out[f]), flush=True)
    with open(os.path.join(HERE, "diagnostics.json"), "w") as fh:
        json.dump(out, fh, indent=1)
