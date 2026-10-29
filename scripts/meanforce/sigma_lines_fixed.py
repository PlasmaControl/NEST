"""sigma = J.B/B^2 of a fixed-boundary state at its own field lines' crossings of phi = pi/NFP
(poincare_fixed.py sections, keys <key>_Rh, <key>_Zh). Coordinate-free input for the MDE along lines:
on each line, sigma(theta) sampled at its crossings; its poloidal harmonics vs the line's iota show a
Pfirsch-Schlueter tail as |sigma_m| ~ 1/|iota - n/m|. J from the basis at the crossings' DESC
coordinates (eq.map_coordinates), B = B_ext + the annulus field at the points. Also p at the points.
usage: python sigma_lines_fixed.py --LMN 48 32 18 --frac 40 --state name=state.npz:sections.npz:key --out x.npz
"""
import argparse
import os
import sys
import time
from types import SimpleNamespace

import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.join(ROOT, "nest"))
sys.path.insert(0, os.path.join(ROOT, "scripts"))
VFB = os.path.join(ROOT, "desc_input", "vmec_free_boundary")
sys.path.insert(0, VFB)

from desc import set_device  # noqa: E402

set_device(os.environ.get("NEST_DEVICE", "cpu"))

import jax  # noqa: E402
import jax.numpy as jnp  # noqa: E402

jax.config.update("jax_enable_x64", True)

import fit_cache  # noqa: E402
from boundary_potential import BoundaryPotential  # noqa: E402
from cylindrical_grid import CylindricalInterpolator  # noqa: E402
from desc_input import DESCInput  # noqa: E402
from flux_interp import TensorGeometry, TensorGrid  # noqa: E402
from force import ForceResidual  # noqa: E402
from free_boundary import _chunked, reference  # noqa: E402
from objective import Fields  # noqa: E402
from optimize_fb import annulus  # noqa: E402
from pressure import Pressure  # noqa: E402
from separable import Separable  # noqa: E402

D = os.path.join(VFB, "NCSX")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--eq", default=os.path.join(D, "NCSX_fb_L16M16N12.h5"))
    ap.add_argument("--coils", default=os.path.join(D, "NCSX_coils.h5"))
    ap.add_argument("--factors", type=float, nargs=3, default=(1, 1, 1))
    ap.add_argument("--LMN", type=int, nargs=3, default=None)
    ap.add_argument("--colloc", type=float, default=1.5)
    ap.add_argument("--frac", type=float, default=40)
    ap.add_argument("--pad", type=float, default=0.02)
    ap.add_argument("--state", action="append", default=[], help="name=state.npz:sections.npz:key; repeatable")
    ap.add_argument("--K", type=int, default=400, help="crossings per line")
    ap.add_argument("--rho", type=float, nargs=3, default=(0.05, 0.97, 185), help="lo hi n")
    ap.add_argument("--nt", type=int, default=96)
    ap.add_argument("--nz", type=int, default=32, help="zeta nodes per field period")
    ap.add_argument("--no-desc", action="store_true")
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    t0 = time.time()
    inp = DESCInput.load(a.eq, field=a.coils)
    eq = inp.eq
    kw = {"L": a.LMN[0], "M": a.LMN[1], "N": a.LMN[2]} if a.LMN else {"factors": tuple(a.factors)}
    cur = fit_cache.fit(BoundaryPotential, inp, **kw)
    pre = fit_cache.fit(Pressure, inp, edge_order=2, **kw)
    b = cur.basis
    res = tuple(int(round(a.colloc * (u + v))) for u, v in ((b.L, eq.L), (b.M, eq.M), (b.N, eq.N)))
    ref = reference(inp, res, current=cur)
    box = annulus(inp, inp.a / a.frac, a.pad, None)
    box.locate_plasma_planes(inp, sym=bool(eq.sym), verbose=False)
    fr = ForceResidual(inp, box, resolution=res)
    fr.norm, fr.F_desc_vol = ref["norm"], ref["F_desc_vol"]
    if "B_ext" not in vars(fr):
        fr.B_ext = _chunked(inp.B_ext, fr.x)
    fields = Fields(fr, cur, pre, dtype=jnp.float64, cyl_stencil=True, mode_chunk=16, cyl_fused=True,
                    cyl_rblock=32, cyl_order=8)
    print(f"{inp}\nbasis {b.L}/{b.M}/{b.N}, annulus {box}; setup {time.time() - t0:.0f} s", flush=True)


    out = {}
    phi0 = np.pi / eq.NFP
    for spec in a.state:
        name, spath, secpath, key = (spec.split("=", 1)[0],) + tuple(spec.split("=", 1)[1].split(":"))
        st, sec = np.load(spath), np.load(secpath)
        Rh, Zh = sec[f"{key}_Rh"][: a.K], sec[f"{key}_Zh"][: a.K]
        shp = Rh.shape
        pts = np.column_stack([Rh.ravel(), np.full(Rh.size, phi0), Zh.ravel()])
        ok = np.isfinite(pts).all(1)
        t1 = time.time()
        rtz = np.full_like(pts, np.nan)
        rtz[ok] = np.asarray(eq.map_coordinates(pts[ok], inbasis=("R", "phi", "Z"), outbasis=("rho", "theta", "zeta"),
                                                maxiter=60, tol=1e-10))
        good = np.isfinite(rtz).all(1) & (rtz[:, 0] < 0.999) & (rtz[:, 0] > 1e-3)
        print(f"{name}: {good.sum()} of {len(pts)} crossings inside, mapped in {time.time() - t1:.0f} s", flush=True)
        J = np.full((len(pts), 3), np.nan)
        J[good] = np.asarray(cur.with_u(jnp.asarray(st["u"])).evaluate(rtz[good]))
        pg = np.full((len(pts), 4), np.nan)
        pg[good] = np.asarray(pre.with_c(jnp.asarray(st["c_p"])).evaluate(SimpleNamespace(nodes=rtz[good])))
        P = pts[good]
        X = np.column_stack([P[:, 0] * np.cos(P[:, 1]), P[:, 0] * np.sin(P[:, 1]), P[:, 2]])
        Bx = np.asarray(inp.B_ext(X))
        c, s_ = np.cos(P[:, 1]), np.sin(P[:, 1])
        Bext = np.column_stack([Bx[:, 0] * c + Bx[:, 1] * s_, -Bx[:, 0] * s_ + Bx[:, 1] * c, Bx[:, 2]])
        itp = CylindricalInterpolator(fields.cyl_grid, P, dtype=jnp.float64, order=8)
        Bp, Bz = fields.cyl_field_modes(jnp.asarray(st["u"]))
        B = np.full((len(pts), 3), np.nan)
        B[good] = np.asarray(itp(Bp, Bz)) + Bext
        B2 = np.sum(B * B, 1)
        sig = np.sum(J * B, 1) / B2
        out[f"{name}_sigma"] = sig.reshape(shp)
        out[f"{name}_p"] = pg[:, 0].reshape(shp)
        out[f"{name}_rho"] = rtz[:, 0].reshape(shp)
        out[f"{name}_absB"] = np.sqrt(B2).reshape(shp)
        print(f"{name}: sigma rms {np.sqrt(np.nanmean(sig**2)):.3e}; {time.time() - t1:.0f} s", flush=True)
    os.makedirs(os.path.dirname(os.path.abspath(a.out)), exist_ok=True)
    np.savez(a.out, **out)
    print(f"saved {a.out}; total {time.time() - t0:.0f} s")


if __name__ == "__main__":
    main()
