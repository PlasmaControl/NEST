"""Knot spacing x truncation scan of the spline current (and pressure counts) on the ripple equilibrium: unknowns
against Fourier-Zernike 32/32/6 and 40/40/6, and rms |J_fit - J_DESC| / rms J_DESC by rho band, so a configuration
can be chosen that is at least as accurate as FZ everywhere (and finer in the island band).

    python scripts/ripple_spline/knot_scan.py [--quick]   (--quick: counts only, no fits)
"""
import argparse
import json
import os
import sys
import time

import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.join(ROOT, "nest"))
from desc import set_device  # noqa: E402

set_device(os.environ.get("NEST_DEVICE", "cpu"))
import jax  # noqa: E402
import jax.numpy as jnp  # noqa: E402

jax.config.update("jax_enable_x64", True)
from boundary_potential import BoundaryPotential  # noqa: E402
from desc_input import DESCInput  # noqa: E402
from spline_basis import SplineBasis  # noqa: E402
from spline_pressure import SplinePressure  # noqa: E402

EQ = os.path.join(ROOT, "desc_input", "tokamak_boundary_perturbed", "bp2", "L32N4", "tokamak_ripple_C_L32M32N4_x0.001.h5")
BANDS = {"b01": ((0.45, 0.76, 0.01), (0.86, 0.95, 0.015)), "b02": ((0.45, 0.76, 0.02), (0.86, 0.95, 0.03)),
         "b015": ((0.45, 0.76, 0.015), (0.86, 0.95, 0.02))}
EDGES = ((0, 0.4), (0.4, 0.55), (0.55, 0.72), (0.72, 0.85), (0.85, 0.95), (0.95, 1.0))

ap = argparse.ArgumentParser()
ap.add_argument("--quick", action="store_true")
ap.add_argument("--out", default="out/knot_scan.json")
a = ap.parse_args()
t0 = time.time()
inp = DESCInput.load(EQ)
eq = inp.eq
CONFIGS = []
for c in (1.0, 0.6, 0.45):
    for bn in ("b01", "b015", "b02"):
        for tr in ({"m_fz": 32}, {"m_tol": 1e-3}, {"m_tol": 1e-3, "m_max": 20}, {"m_tol": 1e-2, "m_max": 16}):
            CONFIGS.append((c, bn, tr))
x = None
if not a.quick:
    g_r = np.linspace(0.01, 0.99, 99)
    R_, T_, Z_ = np.meshgrid(g_r, 2 * np.pi * np.arange(33) / 33, 2 * np.pi * np.arange(9) / 9, indexing="ij")
    x = np.column_stack([R_.ravel(), T_.ravel(), Z_.ravel()])
    Jd = np.asarray(inp.compute_points(["J"], x)["J"])
    rmsD = np.sqrt(np.mean(np.sum(Jd**2, 1)))


def jerr(pot):
    Jp = np.asarray(jax.jit(pot.point_evaluator())(*(jnp.asarray(x[:, i]) for i in range(3))))
    e = np.sum((Jp - Jd) ** 2, 1)
    return float(np.sqrt(e.mean()) / rmsD), [float(np.sqrt(e[(x[:, 0] >= lo) & (x[:, 0] < hi)].mean()) / rmsD)
                                             for lo, hi in EDGES]


out = {"edges": EDGES, "rows": []}
if not a.quick:
    for L in (32, 40):
        fz = BoundaryPotential.fit(eq, L=L, M=L, N=6)
        tot, by = jerr(fz)
        out["rows"].append({"name": f"FZ {L}/{L}/6", "n_u": int(fz.n_free), "J err": tot, "by band": by})
        print(f"FZ {L}/{L}/6: n_u {fz.n_free}; J err {tot:.2e}; by band " + " ".join(f"{v:.1e}" for v in by), flush=True)
for c, bn, tr in CONFIGS:
    kw = dict(bands=BANDS[bn], fz_knots=(32, c), reg="rhom", q_cap=6, **tr)
    sb = SplineBasis.from_knots(eq, 32, 6, helicities=("2/1",), **kw)
    sp = SplinePressure(sb.bp_rho, 32, 6, eq.NFP, bool(eq.sym), None, eq, 2, tr.get("m_tol", 0.0), tr.get("m_max"),
                        tr.get("m_fz"), "rhom", 6)
    row = {"name": f"c{c} {bn} {tr}", "c": c, "bands": bn, "trunc": tr, "n_s": sb.n_s, "a+b": 2 * sb.num_modes,
           "n_p": sp.n, "min spacing": float(np.diff(sb.bp_rho).min()), "max spacing": float(np.diff(sb.bp_rho).max())}
    msg = f"{row['name']:46s} n_s {sb.n_s:3d} a+b {2 * sb.num_modes:6d} n_p {sp.n:6d} h {row['max spacing']:.3f}"
    if not a.quick and (2 * sb.num_modes) < 40000:
        t1 = time.time()
        pot = BoundaryPotential.fit(eq, M=32, N=6, radial="spline", helicities=("2/1",), **kw)
        tot, by = jerr(pot)
        row.update({"n_u": int(pot.n_free), "J err": tot, "by band": by})
        msg += f" n_u {pot.n_free:6d}; J err {tot:.2e}; by band " + " ".join(f"{v:.1e}" for v in by) + f" ({time.time() - t1:.0f} s)"
    print(msg, flush=True)
    out["rows"].append(row)
    os.makedirs(os.path.dirname(a.out) or ".", exist_ok=True)
    with open(a.out, "w") as fh:
        json.dump(out, fh, indent=1)
print(f"done {time.time() - t0:.0f} s")
