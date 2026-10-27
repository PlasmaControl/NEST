"""Saved states of ``mf_run.py`` evaluated by one apparatus, to compare them across runs.

Every state (its current coefficients u and its geometry <stem>_eq.h5, xi = 0) is evaluated
on the same collocation grid, the same annulus and the same pressure space:

  M       <|F|>_vol / norm with p the Poisson pressure of the EVALUATION basis (``--factors``)
  dJ      |J - J_DESC| / rms J_DESC, a fixed target (DESC's current carried with the
          coordinates), instead of each run's own fit J_fit
  Bn, Psi as in the runs

so the bases of study 2 (each with its own J_fit and grid) and the spacings of study 1 (each on
its own annulus) become comparable. DESC's own solution is added (M by its quadrature).

    python scripts/meanforce/eval_common.py --factors 1.75 1.75 1 --frac 40 \
        --state L16=results/meanforce/sweep_gltr_a40_a0.1:1,1,1 \
        --state L28=results/meanforce/s2_gltr_L28N12_a0.1:1.75,1.75,1 --out results/meanforce/common_L28
"""

import argparse
import gc
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
from optimize_fb import annulus  # noqa: E402
from pressure import Pressure  # noqa: E402

D = os.path.join(ROOT, "desc_input", "vmec_free_boundary", "NCSX")


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--eq", default=os.path.join(D, "NCSX_fb_L16M16N12.h5"))
    ap.add_argument("--coils", default=os.path.join(D, "NCSX_coils.h5"))
    ap.add_argument("--factors", type=float, nargs=3, default=(1.75, 1.75, 1),
                    help="evaluation basis (pressure space and collocation grid)")
    ap.add_argument("--colloc", type=float, default=1.5)
    ap.add_argument("--frac", type=float, default=40, help="annulus spacing h = a / frac")
    ap.add_argument("--tan-chunk", type=int, default=None)
    ap.add_argument("--state", action="append", required=True,
                    help="label=<out>_a<alpha>:fL,fM,fN (the state's own basis factors)")
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    t0 = time.time()
    inp0 = DESCInput.load(a.eq, field=a.coils)
    eq0 = inp0.eq
    pre_e = fit_cache.fit(Pressure, inp0, factors=tuple(a.factors), edge_order=2)
    cur_e = fit_cache.fit(BoundaryPotential, inp0, factors=tuple(a.factors))
    b = cur_e.basis
    res = tuple(int(round(a.colloc * (u + v))) for u, v in ((b.L, eq0.L), (b.M, eq0.M), (b.N, eq0.N)))
    ref = reference(inp0, res)  # DESC's own J as the target
    norm = ref["norm"]
    M_desc = ref["F_desc_vol"] / norm
    rmsJ = ref["rms"]["J"]
    # B.n on the evaluation basis' boundary grid for every state
    bgrid = (2 * (b.M + eq0.M) + 1, (b.N + eq0.N) + 1 if eq0.sym else 2 * (b.N + eq0.N) + 1)
    grid0 = annulus(inp0, inp0.a / a.frac, 0.02)
    tan_chunk = a.tan_chunk or max(1, min(32, int(32 * (40 / a.frac) ** 3)))
    out = {"args": vars(a), "M_desc": M_desc, "resolution": res, "states": {}}
    print(f"evaluation: pressure space {b.L}/{b.M}/{b.N}, collocation {res}, annulus a/{a.frac}; "
          f"M_DESC {M_desc:.4e}; {time.time() - t0:.0f} s", flush=True)
    solver = None
    for spec in a.state:
        lab, rest = spec.split("=", 1)
        stem, fs = rest.rsplit(":", 1)
        fs = tuple(float(x) for x in fs.split(","))
        t1 = time.time()
        st = np.load(stem + ".npz")
        eq = desc.io.load(stem + "_eq.h5")
        u = np.asarray(st["u"])
        cur0 = fit_cache.fit(BoundaryPotential, inp0, factors=fs)
        assert cur0.n_free == len(u), f"{lab}: {len(u)} coefficients, basis factors {fs} give {cur0.n_free}"
        inp = DESCInput(eq, inp0.field, name=lab, check=False)
        box = CylindricalGrid(grid0.r_axis, grid0.z_axis, grid0.NFP, grid0.N_phi)
        box.locate_plasma_planes(inp, sym=bool(eq0.sym))
        fr = ForceResidual(inp, box, resolution=res)
        fr.norm, fr.F_desc_vol = norm, ref["F_desc_vol"]
        cur = BoundaryPotential(cur0.basis, u, cur0.I, eq, cur0.edge_power, cur0.S)
        pre = pre_e.with_c(pre_e.c, eq)
        fb = FreeBoundaryFields(fr, cur, pre, Displacement(eq, 16, 12), ref, dtype=jnp.float64,
                                boundary=bgrid, cyl_stencil=True, cyl_solver=solver,
                                cyl_fused=True, cyl_rblock=32, cyl_order=8, verbose=False,
                                mode_chunk=16)
        solver = fb.solver
        A = mf.Anchor(fb, u, rmsJ, norm, chunk=32, tan_chunk=tan_chunk)
        z = jnp.concatenate([jnp.asarray(u), jnp.zeros(A.n_xi)])
        s = A.state(z)
        xf = jnp.concatenate([jnp.asarray(u), jnp.zeros(A.n_p + A.n_xi)])
        loc = fb._local(xf)
        w = np.asarray(s["sw"]) ** 2
        dJ = np.asarray(loc["J"]) - np.asarray(loc["J_desc"])
        r = {"M": float(s["M"]), "M/M_desc": float(s["M"]) / M_desc, "rms|F|": float(s["rmsF"]),
             "Bn/|B| rms": float(s["Bn_rms"]), "Psi": float(s["Psi"]),
             "|J-J_DESC|/rms": float(np.sqrt(np.sum(w[:, None] * dJ**2))) / rmsJ,
             "own |J-J_fit|/rms": None, "galerkin": float(s["galerkin"]), "n_u": len(u),
             "time": time.time() - t1}
        out["states"][lab] = r
        print(f"{lab}: M/M_DESC {r['M/M_desc']:.4f}, |J - J_DESC|/rms {r['|J-J_DESC|/rms']:.4e}, "
              f"B.n {r['Bn/|B| rms']:.3e}, Psi {r['Psi']:.6f}; {r['time']:.0f} s", flush=True)
        release(fb, fr, box)
        del A, s, loc, fb
        jax.clear_caches()
        gc.collect()
        with open(a.out + ".json", "w") as fh:
            json.dump(out, fh, indent=1)
    print(f"saved {a.out}.json; {time.time() - t0:.0f} s")


if __name__ == "__main__":
    main()
