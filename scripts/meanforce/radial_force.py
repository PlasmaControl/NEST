"""Where the force error is: |F| / norm averaged over each coordinate surface rho = const
(dV-weighted over theta, zeta at the collocation radii) and over rho ranges, for DESC's
solution, NEST's start (the fits, Poisson p) and saved NEST states (``mf_run.py``
<out>_a<alpha>.npz with its _eq.h5; xi = 0 there). Also each range's share of the
volume mean M = <|F|>_vol / norm, so the reduction of M can be attributed to radii.

The surfaces are those of each state's own flux coordinates (the moved geometry for a
free-boundary state); NEST's B need not lie on them, so this is an average over the
coordinate surface, which is a flux-surface average only where the field is nested.

    python scripts/meanforce/radial_force.py --run gltr=results/meanforce/gltr_a40_a0.7 \
        --run qr=results/meanforce/qr_a40_a0.7 --out results/meanforce/radial_a40
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
import meanforce as mf  # noqa: E402
from boundary_potential import BoundaryPotential  # noqa: E402
from cylindrical_grid import CylindricalGrid  # noqa: E402
from desc_input import DESCInput  # noqa: E402
from force import ForceResidual  # noqa: E402
from free_boundary import Displacement, FreeBoundaryFields, reference, release  # noqa: E402
from optimize_fb import annulus, rebase  # noqa: E402
from pressure import Pressure  # noqa: E402

D = os.path.join(ROOT, "desc_input", "vmec_free_boundary", "NCSX")
BINS = (0.0, 0.2, 0.4, 0.6, 0.7, 0.8, 0.85, 0.9, 0.95, 0.98, 1.0)


def profile(Fn, dV, rho_pts, rho_surf):
    """Per surface: dV-weighted mean and max of Fn; per bin: mean, and share of M."""
    shape = (len(rho_surf), -1)
    F2, w2 = Fn.reshape(shape), dV.reshape(shape)
    V = dV.sum()
    surf = {"mean": (np.sum(F2 * w2, 1) / np.sum(w2, 1)).tolist(), "max": F2.max(1).tolist(),
            "p99": np.percentile(F2, 99, axis=1).tolist()}
    bins = {}
    for a, b in zip(BINS[:-1], BINS[1:]):
        m = (rho_pts > a) & (rho_pts <= b)
        bins[f"{a:g}-{b:g}"] = {"mean": float(np.sum(Fn[m] * dV[m]) / np.sum(dV[m])),
                                "part of M": float(np.sum(Fn[m] * dV[m]) / V),
                                "volume fraction": float(np.sum(dV[m]) / V)}
    return {"M": float(np.sum(Fn * dV) / V), "surface": surf, "bins": bins}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--eq", default=os.path.join(D, "NCSX_fb_L16M16N12.h5"))
    ap.add_argument("--coils", default=os.path.join(D, "NCSX_coils.h5"))
    ap.add_argument("--run", action="append", default=[], help="label=<out>_a<alpha> (no ext)")
    ap.add_argument("--frac", type=float, default=40)
    ap.add_argument("--colloc", type=float, default=1.5)
    ap.add_argument("--out", required=True)
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
    norm = ref["norm"]
    out = {"norm": norm, "M_desc (DESC quadrature)": ref["F_desc_vol"] / norm, "bins": BINS,
           "states": {}}

    # DESC's own |F| on NEST's collocation points (the start geometry)
    fr = ForceResidual(inp0, None, resolution=res)
    tg = fr.tgrid
    rho_surf = np.asarray(tg.rho)
    out["rho"] = rho_surf.tolist()
    dF = inp0.compute_tensor(["|F|"], tg.rho, tg.theta, tg.zeta)
    Fd = np.asarray(dF["|F|"]).ravel() / norm
    out["states"]["DESC"] = profile(Fd, fr.dV, fr.rtz[:, 0], rho_surf)
    print(f"DESC: M {out['states']['DESC']['M']:.4e}; {time.time() - t0:.0f} s", flush=True)
    del fr, dF

    grid0 = annulus(inp0, inp0.a / a.frac, 0.02)
    solver = [None]

    def nest_state(eq, u):
        inp = DESCInput(eq, inp0.field, name="state", check=False)
        box = CylindricalGrid(grid0.r_axis, grid0.z_axis, grid0.NFP, grid0.N_phi)
        box.locate_plasma_planes(inp, sym=bool(eq0.sym))
        fr = ForceResidual(inp, box, resolution=res)
        fr.norm = norm
        cur, pre = rebase(cur0, pre0, eq, np.concatenate([np.asarray(u), pre0.c]))
        fb = FreeBoundaryFields(fr, cur, pre, Displacement(eq, 16, 12), ref, dtype=jnp.float64,
                                cyl_stencil=True, cyl_solver=solver[0], cyl_fused=True,
                                cyl_rblock=32, cyl_order=8, verbose=False)
        solver[0] = fb.solver
        A = mf.Anchor(fb, cur0.u, 1.0, norm, chunk=32)
        s = A.state(jnp.concatenate([jnp.asarray(u), jnp.zeros(A.n_xi)]))
        Fn = np.asarray(s["absF"]) / norm
        w = np.asarray(s["sw"]) ** 2
        prof = profile(Fn, w, fr.rtz[:, 0], rho_surf)
        prof["M (state)"] = float(s["M"])
        release(fb, fr, box)
        del A, s
        jax.clear_caches()
        return prof

    out["states"]["NEST start"] = nest_state(eq0, cur0.u)
    print(f"NEST start: M {out['states']['NEST start']['M']:.4e}; {time.time() - t0:.0f} s",
          flush=True)
    for spec in a.run:
        lab, stem = spec.split("=", 1)
        st = np.load(stem + ".npz")
        eq = desc.io.load(stem + "_eq.h5")
        out["states"][lab] = nest_state(eq, st["u"])
        print(f"{lab}: M {out['states'][lab]['M']:.4e}; {time.time() - t0:.0f} s", flush=True)
    with open(a.out + ".json", "w") as fh:
        json.dump(out, fh, indent=1)
    print("bin            " + "  ".join(f"{k:>12}" for k in out["states"]))
    for k in out["states"]["DESC"]["bins"]:
        print(f"{k:>10} mean " + "  ".join(f"{v['bins'][k]['mean']:12.3e}"
                                           for v in out["states"].values()))
        print(f"{'':>10} of M " + "  ".join(f"{v['bins'][k]['part of M']:12.3e}"
                                            for v in out["states"].values()))
    print(f"saved {a.out}.json; {time.time() - t0:.0f} s")


if __name__ == "__main__":
    main()
