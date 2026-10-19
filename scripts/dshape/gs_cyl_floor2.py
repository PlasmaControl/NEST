"""The rest of the annulus B_plasma floor (gs_cyl_floor.py): with J evaluated at the nodes and the
exact field at (512, 96) as the reference, ~1.2e-9 of rms B is left, uniform in rho.  Here, for
each h = a / frac, B_plasma's error against that reference with the solver's boundary part
changed one at a time:
  base        n_gl 16, ratio 0.15 (the single-layer rule of James' correction), margin 2 h;
  n_gl 32     twice the Gauss points per boundary panel;
  ratio 0.05  the near-panel grading finer;
  margin 4    the walls twice as far from the plasma.
Writes results/gs/cyl_floor2.json.

run (GPU): sbatch run.sbatch scripts/dshape/gs_cyl_floor2.py --fracs 160 320 640
"""

import argparse
import json
import os
import sys
import time

import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.join(ROOT, "nest"))
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))  # sibling scripts

from desc import set_device  # noqa: E402

set_device(os.environ.get("NEST_DEVICE", "cpu"))

import jax  # noqa: E402
import jax.numpy as jnp  # noqa: E402

import fit_cache  # noqa: E402
from annulus_freespace import FreeSpaceAnnulus  # noqa: E402
from axisym_field import plasma_field_cached  # noqa: E402
from boundary_potential import BoundaryPotential  # noqa: E402
from boxes import make_box  # noqa: E402
from cartesian_grid import cylindrical_to_cartesian  # noqa: E402
from cylindrical_bs import CylindricalSolver  # noqa: E402
from dense import force_matched_u  # noqa: E402
from desc_input import DESCInput  # noqa: E402
from force import ForceResidual  # noqa: E402
from gs_cyl_floor import EDGES, TOK, node_J  # noqa: E402
from objective import Fields, _cartesian  # noqa: E402
from pressure import Pressure  # noqa: E402

jax.config.update("jax_enable_x64", True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--eq", default=os.path.join(TOK, "tokamak_L32M32_fb.h5"))
    ap.add_argument("--coils", default=os.path.join(TOK, "tokamak_L32M32_field.h5"))
    ap.add_argument("--fracs", type=int, nargs="+", default=[160, 320, 640])
    a = ap.parse_args()
    t0 = time.time()
    inp = DESCInput.load(a.eq, field=a.coils)
    eq = inp.eq
    cur = fit_cache.fit(BoundaryPotential, inp, factors=(2, 2, 1), edge_power=3)
    pre = fit_cache.fit(Pressure, inp, factors=(2, 2, 1), edge_order=2)
    b = cur.basis
    res = tuple(int(round(2 * (p + q))) for p, q in ((b.L, eq.L), (b.M, eq.M), (b.N, eq.N)))
    fr0 = ForceResidual(inp, None, resolution=res)
    desc = {k: np.asarray(v) for k, v in fr0.desc_fields().items()}
    u0, _ = force_matched_u(fr0, cur, eps=np.inf, desc=desc)
    rtz, phi = np.asarray(fr0.rtz), np.asarray(fr0.phi)
    w = fr0.dV / fr0.dV.sum()
    rho = rtz[:, 0]
    rmsB = float(np.sqrt(np.sum(w * np.sum(desc["B"] ** 2, 1))))
    B2 = np.asarray(cylindrical_to_cartesian(plasma_field_cached(cur, rtz, u=u0, n_gamma=512, n_s=96), phi))
    rows = []
    print(f"{'h':>6s} {'case':14s} {'rms err':>9s}   " + " ".join(
        f"{lo:.2f}-{hi:.2f}"[:8].rjust(8) for lo, hi in zip(EDGES[:-1], EDGES[1:])) + "   build", flush=True)
    for frac in a.fracs:
        for name, margin, kw in (("base", 2.0, {}), ("n_gl 32", 2.0, {"n_gl": 32}),
                                 ("ratio 0.05", 2.0, {"ratio": 0.05}), ("margin 4", 4.0, {})):
            t1 = time.time()
            box = make_box(inp, frac, "cylindrical", bool(eq.sym), margin)
            fr = ForceResidual(inp, box, resolution=res)
            fields = Fields(fr, cur, pre, kernel="cylindrical", dtype=jnp.float64, cyl_half=False)
            if kw:  # the solver with another single-layer rule (built here, not from the cache)
                fs = FreeSpaceAnnulus(box.r_axis, box.z_axis, box.NFP, box.N_phi, shifts=(0, 1), **kw)
                fields.solver = CylindricalSolver(box.r_axis, box.z_axis, box.NFP, box.N_phi, fs=fs,
                                                  dtype=jnp.float64)
            t_build = time.time() - t1
            Jn = node_J(cur, u0, np.asarray(box.rtz))
            Jgrid = jnp.pad(jnp.asarray(Jn), ((0, 1), (0, 0)))[fields.src].reshape(fields.shape + (3,))
            Bp, Bz = fields.solver.field_modes(Jgrid)
            e = np.asarray(_cartesian(fields.to_points(Bp, Bz), fields.phi)) - B2
            m = float(np.sqrt(np.sum(w * np.sum(e**2, 1)))) / rmsB
            bins = [float(np.sqrt(np.sum((w * np.sum(e**2, 1))[(rho > lo) & (rho <= hi)])
                                  / np.sum(w[(rho > lo) & (rho <= hi)]))) / rmsB
                    for lo, hi in zip(EDGES[:-1], EDGES[1:])]
            rows.append({"frac": frac, "case": name, "err": m, "by rho": bins, "build s": t_build})
            print(f"a/{frac:<4d} {name:14s} {m:9.2e}   " + " ".join(f"{v:8.1e}" for v in bins)
                  + f"   {t_build:.0f} s", flush=True)
            json.dump({"args": vars(a), "rows": rows}, open(os.path.join(ROOT, "results", "gs", "cyl_floor2.json"), "w"),
                      indent=1)
            del fields, fr, box
            jax.clear_caches()
    print(f"total {time.time() - t0:.0f} s")


if __name__ == "__main__":
    main()
