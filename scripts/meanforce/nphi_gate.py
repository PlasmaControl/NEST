"""N_phi gate for a nearly axisymmetric case: annulus planes per period vs B_plasma of the start current.

For each N_phi: the run's annulus (a/frac, pad), locate, ForceResidual, Fields as mf_fixed.py builds them
(cyl_fused, cyl_stencil, interpolation order), and B_ann[J_fit] at the collocation points (the same points for
every N_phi: they depend on the basis and the equilibrium only). Reported against the largest N_phi: relative rms
and max of the difference, and of the resonant-relevant n != 0 part (B minus its toroidal average is not available
at scattered points, so the full field is compared). Also DESC's J at the inside nodes is checked for its toroidal
content: the fraction of J^2 in |n| > (N_phi - 1) / 2 (exact Fourier over the planes at fixed (r, z) node).

usage: python scripts/meanforce/nphi_gate.py --eq <h5> --LMN 32 32 6 --frac 40 --N-phi 13 25 49 97 --out gate.json
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

import fit_cache  # noqa: E402
from boundary_potential import BoundaryPotential  # noqa: E402
from desc_input import DESCInput  # noqa: E402
from force import ForceResidual  # noqa: E402
from free_boundary import reference  # noqa: E402
from objective import VECTOR_PARITY, Fields  # noqa: E402
from optimize_fb import annulus  # noqa: E402
from pressure import Pressure  # noqa: E402

T0 = time.time()


def log(*s):
    print(f"[{time.time() - T0:8.1f} s]", *s, flush=True)


ap = argparse.ArgumentParser()
ap.add_argument("--eq", required=True)
ap.add_argument("--LMN", type=int, nargs=3, required=True)
ap.add_argument("--colloc", type=float, default=1.5)
ap.add_argument("--frac", type=float, default=40)
ap.add_argument("--pad", type=float, default=0.02)
ap.add_argument("--N-phi", type=int, nargs="+", default=[13, 25, 49, 97])
ap.add_argument("--rblock", type=int, default=32)
ap.add_argument("--interp-order", type=int, default=8)
ap.add_argument("--out", required=True)
a = ap.parse_args()

inp = DESCInput.load(a.eq, field=None)
eq = inp.eq
kw = {"L": a.LMN[0], "M": a.LMN[1], "N": a.LMN[2]}
cur = fit_cache.fit(BoundaryPotential, inp, **kw)
pre = fit_cache.fit(Pressure, inp, edge_order=2, **kw)
b = cur.basis
res = tuple(int(round(a.colloc * (u + v))) for u, v in ((b.L, eq.L), (b.M, eq.M), (b.N, eq.N)))
ref = reference(inp, res, current=cur)
u0 = jnp.asarray(cur.u)
out = {"args": vars(a), "n_u": int(cur.n_free), "collocation": list(res), "cases": {}}
B = {}
for Np in sorted(a.N_phi):
    t1 = time.time()
    box = annulus(inp, inp.a / a.frac, a.pad, Np)
    box.locate_plasma_planes(inp, sym=bool(eq.sym), verbose=False)
    fr = ForceResidual(inp, box, resolution=res)
    fr.norm, fr.F_desc_vol = ref["norm"], ref["F_desc_vol"]
    fr.B_ext = np.zeros((len(fr.x), 3))
    f = Fields(fr, cur, pre, dtype=jnp.float64, cyl_stencil=True, cyl_fused=True, cyl_rblock=a.rblock,
               cyl_order=a.interp_order)
    B[Np] = np.asarray(f.B_plasma(u0))
    # toroidal content of DESC's J and of J_fit on the annulus (planes at fixed (r, z) nodes)
    J_D = np.asarray(inp.compute_points(["J"], np.asarray(f.rows_rtz))["J"])
    J_f = np.asarray(f.to_box(f.J_flux(u0, f.I).astype(f.cyl_dtype), VECTOR_PARITY))
    hi = {}
    for nm, Jr in (("J_DESC", J_D), ("J_fit", J_f)):
        Jb = np.asarray(f._cyl_rows(jnp.asarray(Jr)))[np.asarray(f.src)].reshape(f.shape + (3,))
        Fh = np.fft.fft(Jb, axis=0)
        pw = np.sum(np.abs(Fh) ** 2, axis=(1, 2, 3))
        n = np.abs(np.fft.fftfreq(Np, 1.0 / Np))
        hi[nm] = {f"fraction of J^2 in |n| > {k}": float(pw[n > k].sum() / pw.sum()) for k in (1, 2, 4, 6, 8)
                  if k < Np // 2}
    out["cases"][Np] = {"annulus": str(box), "rows": int(len(f.rows_rtz)), "seconds": time.time() - t1,
                        "toroidal content": hi}
    log(f"N_phi {Np}: {box}; {time.time() - t1:.0f} s; J content {hi}")
    json.dump(out, open(a.out, "w"), indent=1, default=float)
Nmax = max(B)
w = fr.dV / fr.dV.sum()
rms = float(np.sqrt(np.sum(w * np.sum(B[Nmax] ** 2, 1))))
for Np in sorted(B):
    d = B[Np] - B[Nmax]
    out["cases"][Np]["vs largest: rms |dB| / rms |B_ann|"] = float(np.sqrt(np.sum(w * np.sum(d**2, 1))) / rms)
    out["cases"][Np]["vs largest: max |dB| / rms |B_ann|"] = float(np.max(np.linalg.norm(d, axis=1)) / rms)
    log(f"N_phi {Np} vs {Nmax}: rms {out['cases'][Np]['vs largest: rms |dB| / rms |B_ann|']:.3e}, "
        f"max {out['cases'][Np]['vs largest: max |dB| / rms |B_ann|']:.3e}")
json.dump(out, open(a.out, "w"), indent=1, default=float)
log("done")
