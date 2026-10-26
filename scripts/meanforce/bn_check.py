"""B.n / |B| on the boundary: the annulus' against direct Biot-Savart, on the same points
(the free-boundary fields' boundary grid) and with the same plain mean (as DESC's
diagnostic), for the start fit or a saved state (``mf_run.py`` <out>_a<alpha>.npz and
its _eq.h5, xi = 0 there after applying it).

    python scripts/meanforce/bn_check.py [--state run_a0.7.npz --state-eq run_a0.7_eq.h5]
"""

import argparse
import json
import os
import sys
import time

import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.join(ROOT, "nest"))
sys.path.insert(0, os.path.join(ROOT, "scripts"))

from desc import set_device  # noqa: E402

set_device(os.environ.get("NEST_DEVICE", "cpu"))
import jax  # noqa: E402
import jax.numpy as jnp  # noqa: E402

jax.config.update("jax_enable_x64", True)

import desc.io  # noqa: E402
import fit_cache  # noqa: E402
from boundary_potential import BoundaryPotential  # noqa: E402
from cylindrical_grid import CylindricalGrid  # noqa: E402
from desc_input import DESCInput  # noqa: E402
from force import ForceResidual  # noqa: E402
from free_boundary import Displacement, FreeBoundaryFields, reference  # noqa: E402
from optimize_fb import annulus, rebase  # noqa: E402
from pressure import Pressure  # noqa: E402

D = os.path.join(ROOT, "desc_input", "vmec_free_boundary", "NCSX")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--eq", default=os.path.join(D, "NCSX_fb_L16M16N12.h5"))
    ap.add_argument("--coils", default=os.path.join(D, "NCSX_coils.h5"))
    ap.add_argument("--state", default=None)
    ap.add_argument("--state-eq", default=None)
    ap.add_argument("--frac", type=float, default=40)
    ap.add_argument("--colloc", type=float, default=1.5)
    ap.add_argument("--direct", type=int, nargs=2, default=(32, 4))
    ap.add_argument("--out", default=None)
    a = ap.parse_args()
    t0 = time.time()
    inp0 = DESCInput.load(a.eq, field=a.coils)
    eq0 = inp0.eq
    cur0 = fit_cache.fit(BoundaryPotential, inp0, factors=(1, 1, 1))
    pre0 = fit_cache.fit(Pressure, inp0, factors=(1, 1, 1), edge_order=2)
    b = cur0.basis
    res = tuple(int(round(a.colloc * (u + v))) for u, v in
                ((b.L, eq0.L), (b.M, eq0.M), (b.N, eq0.N)))
    ref = reference(inp0, res, current=cur0)
    eq = eq0 if a.state_eq is None else desc.io.load(a.state_eq)
    u = cur0.u if a.state is None else np.load(a.state)["u"]
    inp = DESCInput(eq, inp0.field, name="check", check=False)
    grid0 = annulus(inp0, inp0.a / a.frac, 0.02)
    box = CylindricalGrid(grid0.r_axis, grid0.z_axis, grid0.NFP, grid0.N_phi)
    box.locate_plasma_planes(inp, sym=bool(eq0.sym))
    fr = ForceResidual(inp, box, resolution=res)
    fr.norm = ref["norm"]
    cur, pre = rebase(cur0, pre0, eq, np.concatenate([np.asarray(u), pre0.c]))
    out = {}
    for name, direct in (("annulus", None), ("direct", tuple(a.direct))):
        fb = FreeBoundaryFields(fr, cur, pre, Displacement(eq, 16, 12), ref, dtype=jnp.float64,
                                cyl_stencil=True, cyl_fused=True, cyl_rblock=32, cyl_order=8,
                                direct=direct, verbose=False)
        f = fb.compute(jnp.concatenate([jnp.asarray(u), jnp.asarray(pre0.c), jnp.zeros(fb.n_xi)]))
        Bb = np.asarray(f["B_bnd"])
        rel = np.asarray(f["Bn"]) / np.linalg.norm(Bb, axis=1)
        w = np.asarray(fb.w_bnd) ** 2
        out[name] = {"rel": rel, "B": Bb, "plain rms": float(np.sqrt(np.mean(rel**2))),
                     "area rms": float(np.sqrt(np.sum(w * rel**2))), "max": float(np.abs(rel).max())}
        print(f"{name}: B.n/|B| rms {out[name]['plain rms']:.4e} (plain mean), "
              f"{out[name]['area rms']:.4e} (area), max {out[name]['max']:.3e}; "
              f"{time.time() - t0:.0f} s", flush=True)
        del fb
    d = out["annulus"]["rel"] - out["direct"]["rel"]
    dB = np.linalg.norm(out["annulus"]["B"] - out["direct"]["B"], axis=1) / np.linalg.norm(
        out["direct"]["B"], axis=1)
    res_ = {k: {kk: vv for kk, vv in v.items() if kk not in ("rel", "B")} for k, v in out.items()}
    res_["annulus - direct: rms of B.n/|B|"] = float(np.sqrt(np.mean(d**2)))
    res_["annulus - direct: max of B.n/|B|"] = float(np.abs(d).max())
    res_["|B_annulus - B_direct| / |B|: rms, max"] = [float(np.sqrt(np.mean(dB**2))),
                                                       float(dB.max())]
    print(json.dumps({k: v for k, v in res_.items() if "-" in k}, indent=1))
    if a.out:
        with open(a.out, "w") as fh:
            json.dump(res_, fh, indent=1)


if __name__ == "__main__":
    main()
