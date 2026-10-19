"""Checks for the B_plasma floor fixes (gs_cyl_floor.py, gs_cyl_floor2.py).

1. (--quad) The exact field's quadrature (axisym_field.plasma_field) at the collocation points
   near the edge (rho > 0.8), against (1024, 192): (256, 48) is off by ~7e-9 of rms B and
   n_gamma does not help; (256, 96) is at round-off.  Hence the default n_s = 96.
2. The annulus B_plasma at each h against the exact field (256, 96), rms over rms B_DESC by rho
   bin, for: S (single-layer matrices) in float32 or float64, J from the flux grid or at the
   nodes (NodeFields); and the time of a dense G column (batch 8).

run (GPU): sbatch run.sbatch tests/floor_fixes.py --fracs 160 320 [--quad]
"""

import argparse
import os
import sys
import time

import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, "nest"))

from desc import set_device  # noqa: E402

set_device(os.environ.get("NEST_DEVICE", "cpu"))

import jax  # noqa: E402
import jax.numpy as jnp  # noqa: E402

import fit_cache  # noqa: E402
from axisym_field import plasma_field  # noqa: E402
from boundary_potential import BoundaryPotential  # noqa: E402
from boxes import make_box  # noqa: E402
from cartesian_grid import cylindrical_to_cartesian  # noqa: E402
from dense import force_matched_u  # noqa: E402
from desc_input import DESCInput  # noqa: E402
from force import ForceResidual  # noqa: E402
from node_current import NodeFields  # noqa: E402
from objective import Fields  # noqa: E402
from pressure import Pressure  # noqa: E402

jax.config.update("jax_enable_x64", True)
TOK = os.path.join(ROOT, "desc_input", "tokamak_free_boundary")
EDGES = (0.0, 0.5, 0.8, 0.9, 0.95, 1.0)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--fracs", type=int, nargs="+", default=[160, 320])
    ap.add_argument("--quad", action="store_true", help="part 1: the exact field's quadrature near the edge")
    ap.add_argument("--n-edge", type=int, default=3000)
    a = ap.parse_args()
    inp = DESCInput.load(os.path.join(TOK, "tokamak_L32M32_fb.h5"), field=os.path.join(TOK, "tokamak_L32M32_field.h5"))
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
    if a.quad:
        sub = np.random.default_rng(0).choice(np.flatnonzero(rho > 0.8), a.n_edge, replace=False)
        ref = plasma_field(cur, rtz[sub], u=u0, n_gamma=1024, n_s=192)
        print(f"quadrature at {a.n_edge} collocation points with rho > 0.8, against (1024, 192):", flush=True)
        for ng, ns in ((256, 48), (512, 48), (256, 96), (512, 96), (1024, 96), (512, 192)):
            e = plasma_field(cur, rtz[sub], u=u0, n_gamma=ng, n_s=ns) - ref
            print(f"  ({ng:4d}, {ns:3d})  {np.sqrt(np.mean(np.sum(e**2, 1))) / rmsB:.2e}", flush=True)
    t1 = time.time()
    Bx = np.asarray(cylindrical_to_cartesian(plasma_field(cur, rtz, u=u0, n_gamma=256, n_s=96), phi))
    print(f"exact field (256, 96) on all {len(rtz)} points: {time.time() - t1:.0f} s", flush=True)

    def err(B):
        e = np.asarray(B) - Bx
        m = float(np.sqrt(np.sum(w * np.sum(e**2, 1)))) / rmsB
        bins = [float(np.sqrt(np.sum((w * np.sum(e**2, 1))[(rho > lo) & (rho <= hi)])
                              / np.sum(w[(rho > lo) & (rho <= hi)]))) / rmsB for lo, hi in zip(EDGES[:-1], EDGES[1:])]
        return f"{m:.2e}   by rho " + " ".join(f"{v:.1e}" for v in bins)

    for frac in a.fracs:
        box = make_box(inp, frac, "cylindrical", bool(eq.sym), 2.0)
        fr = ForceResidual(inp, box, resolution=res)
        for sdt in ("float32", "float64"):
            t1 = time.time()
            fl = Fields(fr, cur, pre, kernel="cylindrical", dtype=jnp.float64, S_dtype=getattr(jnp, sdt))
            t_build = time.time() - t1
            print(f"a/{frac} S {sdt}, J from the flux grid: {err(fl.B_plasma(jnp.asarray(u0)))}   build {t_build:.0f} s",
                  flush=True)
            nf = NodeFields.from_fields(fl, cur, box)
            line = f"a/{frac} S {sdt}, J at the nodes:      {err(nf.B_plasma(jnp.asarray(u0)))}"
            if sdt == "float64":
                solve = jax.jit(jax.vmap(lambda v, f: f.B_plasma(v, 0.0), in_axes=(0, None)))
                E = jnp.eye(nf.n_u)[:8]
                solve(E, nf).block_until_ready()
                t2 = time.time()
                solve(E, nf).block_until_ready()
                t_col = (time.time() - t2) / 8
                line += f"; {t_col:.3f} s per G column, {t_col * nf.n_u / 60:.1f} min for G"
            print(line, flush=True)
            del fl, nf
            jax.clear_caches()
        del fr, box


if __name__ == "__main__":
    main()
