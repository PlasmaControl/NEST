"""Independent check of a free-boundary state on another cylindrical annulus: <|F|>/norm
(overall and by rho) on a finer collocation grid, and B.n/|B| on the boundary by direct
Biot-Savart (``optimize_fb.force_check_cyl``, ``boundary_check``).

run (GPU): sbatch run_della_cyl.sbatch scripts/check_fb_cyl.py --state results/ncsx/cyl240_2 \
    --frac 160 --N-phi 361 --colloc 3
"""

import argparse
import json
import os
import sys
import time

import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, "nest"))
sys.path.insert(0, os.path.join(ROOT, "scripts"))

from desc import set_device  # noqa: E402

set_device(os.environ.get("NEST_DEVICE", "cpu"))

import jax  # noqa: E402

jax.config.update("jax_enable_x64", True)
import desc.io  # noqa: E402

import fit_cache  # noqa: E402
from boundary_potential import BoundaryPotential  # noqa: E402
from desc_input import DESCInput  # noqa: E402
from free_boundary import reference  # noqa: E402
from optimize_fb import annulus, boundary_check, force_check_cyl, rebase  # noqa: E402
from pressure import Pressure  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--eq", default=os.path.join(ROOT, "desc_input", "vmec_free_boundary", "NCSX",
                                                 "NCSX_fb_L16M16N12.h5"))
    ap.add_argument("--coils", default=os.path.join(ROOT, "desc_input", "vmec_free_boundary",
                                                    "NCSX", "NCSX_coils.h5"))
    ap.add_argument("--state", required=True, help="prefix of <state>.npz and <state>_eq.h5")
    ap.add_argument("--frac", type=float, default=160)
    ap.add_argument("--N-phi", type=int, default=361)
    ap.add_argument("--colloc", type=float, default=3)
    ap.add_argument("--pad", type=float, default=0.02)
    ap.add_argument("--factors", type=float, nargs=3, default=(3, 3, 2.5))
    a = ap.parse_args()
    t0 = time.time()
    inp0 = DESCInput.load(a.eq, field=a.coils)
    cur0 = fit_cache.fit(BoundaryPotential, inp0, factors=tuple(a.factors))
    pre0 = fit_cache.fit(Pressure, inp0, factors=tuple(a.factors), edge_order=2)
    b, eq0 = cur0.basis, inp0.eq
    res = tuple(int(round(a.colloc * (u + v))) for u, v in
                ((b.L, eq0.L), (b.M, eq0.M), (b.N, eq0.N)))
    ref = reference(inp0, tuple(int(round(2 * (u + v))) for u, v in
                                ((b.L, eq0.L), (b.M, eq0.M), (b.N, eq0.N))))
    eq = desc.io.load(a.state + "_eq.h5")
    x = np.asarray(np.load(a.state + ".npz")["x"])
    inp = DESCInput(eq, inp0.field, name="state", check=False)
    cur, pre = rebase(cur0, pre0, eq, x)
    g = annulus(inp, inp0.a / a.frac, a.pad, a.N_phi).locate_plasma_planes(inp, sym=True)
    out, shape = force_check_cyl(inp, g, cur, pre, res, x, mode_chunk=16, cyl_fused=True,
                                 cyl_rblock=32, cyl_order=8)
    norm_ratio = ref["norm"]
    bn = boundary_check(inp, g, cur, pre, x, "truncated")
    print(f"{a.state}: check on {g} ({len(g.inside):,} inside), collocation {shape}: "
          f"<|F|>/norm {out['mean']:.3e} (DESC {ref['F_desc_vol'] / norm_ratio:.3e}); by rho "
          + ", ".join(f"{k} {v:.2e}" for k, v in out["bins"].items())
          + f"; B.n/|B| rms {bn['Bn/|B| rms']:.3e}, max {bn['Bn/|B| max']:.3e} (direct); "
          f"{time.time() - t0:.0f} s", flush=True)
    with open(a.state + f"_check_{a.frac:g}_{a.N_phi}.json", "w") as f:
        json.dump({"force": out, "boundary": bn, "grid": [int(v) for v in g.shape],
                   "collocation": [int(v) for v in shape], "frac": a.frac, "N_phi": a.N_phi},
                  f, indent=1)


if __name__ == "__main__":
    main()
