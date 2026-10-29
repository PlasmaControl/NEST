"""Poincare sections of a fixed-boundary mean-force state (mf_fixed.py) on the cylindrical
annulus field.

B = B_ext (coils) + B_plasma(u), with B_plasma from the run's own annulus solve: the modal
slabs of ``Fields.cyl_field_modes(u)`` evaluated on a cylindrical (R, phi, Z) table over one
field period by ``CylindricalInterpolator`` (the operator the solver uses to bring the field
to the collocation points). The annulus field is exactly NFP-periodic, so the box path's
periodizing step is not needed. The table is restricted to the annulus (lines that reach
its walls stop). Tracing (RK4 on cubic splines in phi), seeding and the output format are
those of desc_input/vmec_free_boundary/poincare.py, so poincare_rz.py draws the result.

Gate (--check, default on): the spline-interpolated table at the collocation points against
Fields.B_plasma(u) + B_ext there (relative rms and max).

usage: python scripts/meanforce/poincare_fixed.py --LMN 32 32 24 --frac 40 \\
           --state results/meanforce_fixed/states/fxD05b_a0.05.npz --out out/fxD05b_poincare.npz
"""
import argparse
import os
import sys
import time

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
from force import ForceResidual  # noqa: E402
from free_boundary import _chunked, reference  # noqa: E402
from objective import Fields, _cartesian  # noqa: E402
from optimize_fb import annulus  # noqa: E402
from pressure import Pressure  # noqa: E402
from separable import Separable  # noqa: E402
import poincare as pc  # noqa: E402  (tracer, resonances, seeds)

D = os.path.join(VFB, "NCSX")


def table(fields, inp, u, Rs, phis, Zs, chunk_planes=8, order=8):
    """B (R, phi, Z components) on the (Rs, phis, Zs) table: B_ext + B_plasma(u)."""
    Bp, Bz = fields.cyl_field_modes(jnp.asarray(u))
    nR, nP, nZ = len(Rs), len(phis), len(Zs)
    out = np.empty((nR, nP, nZ, 3))
    RR, ZZ = np.meshgrid(Rs, Zs, indexing="ij")
    for p0 in range(0, nP, chunk_planes):
        ps = phis[p0 : p0 + chunk_planes]
        pts = np.column_stack([np.tile(RR.ravel(), len(ps)), np.repeat(ps, RR.size), np.tile(ZZ.ravel(), len(ps))])
        itp = CylindricalInterpolator(fields.cyl_grid, pts, dtype=jnp.float64, order=order)
        Bpl = np.asarray(itp(Bp, Bz))  # cylindrical components
        X = np.column_stack([pts[:, 0] * np.cos(pts[:, 1]), pts[:, 0] * np.sin(pts[:, 1]), pts[:, 2]])
        Bx = np.concatenate([np.asarray(inp.B_ext(X[i : i + 500_000])) for i in range(0, len(X), 500_000)])
        c, s = np.cos(pts[:, 1]), np.sin(pts[:, 1])
        Bext = np.column_stack([Bx[:, 0] * c + Bx[:, 1] * s, -Bx[:, 0] * s + Bx[:, 1] * c, Bx[:, 2]])
        B = (Bpl + Bext).reshape(len(ps), nR, nZ, 3).transpose(1, 0, 2, 3)
        out[:, p0 : p0 + len(ps)] = B
    return out


def check(fields, Rs, phis, Zs, B, u, NFP):
    """Spline-interpolated table at the collocation points vs the solver's own B there."""
    from scipy.ndimage import map_coordinates, spline_filter
    x = np.asarray(fields.force.x) if hasattr(fields.force, "x") else None
    phi = np.asarray(fields.phi)
    Bsolver = np.asarray(fields.B_ext + fields.B_plasma(jnp.asarray(u)))  # Cartesian
    R = np.hypot(x[:, 0], x[:, 1]) if x is not None else None
    Z = x[:, 2]
    c, s = np.cos(phi), np.sin(phi)
    Bs = np.column_stack([Bsolver[:, 0] * c + Bsolver[:, 1] * s, -Bsolver[:, 0] * s + Bsolver[:, 1] * c, Bsolver[:, 2]])
    period = 2 * np.pi / NFP
    Bp = np.concatenate([B[:, -3:], B, B[:, :3]], axis=1)
    dR, dZ, dphi = Rs[1] - Rs[0], Zs[1] - Zs[0], phis[1] - phis[0]
    q = np.stack([(R - Rs[0]) / dR, np.mod(phi, period) / dphi + 3, (Z - Zs[0]) / dZ])
    Bt = np.column_stack([map_coordinates(spline_filter(Bp[..., k], order=3, mode="nearest"), q, order=3,
                                          prefilter=False, mode="nearest") for k in range(3)])
    err = np.linalg.norm(Bt - Bs, axis=1) / np.linalg.norm(Bs, axis=1)
    return float(np.sqrt(np.mean(err**2))), float(err.max())


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--eq", default=os.path.join(D, "NCSX_fb_L16M16N12.h5"))
    ap.add_argument("--coils", default=os.path.join(D, "NCSX_coils.h5"))
    ap.add_argument("--factors", type=float, nargs=3, default=(1, 1, 1))
    ap.add_argument("--LMN", type=int, nargs=3, default=None)
    ap.add_argument("--colloc", type=float, default=1.5)
    ap.add_argument("--frac", type=float, default=40)
    ap.add_argument("--pad", type=float, default=0.02)
    ap.add_argument("--state", action="append", default=[],
                    help="[name=]path.npz with u (mf_fixed state); repeatable; no name: 'optimized'")
    ap.add_argument("--res-seeds", default=None,
                    help="'rho:m,rho:m,...': seed only across these resonances (rho +- 0.05, 11 radii) "
                         "at theta = 0, pi/m, pi, pi + pi/m (replaces the default seeds)")
    ap.add_argument("--pi-m-seeds", type=int, default=0,
                    help="extra seeds per chain at theta = pi / m across rho_res +- 0.03")
    ap.add_argument("--skip-start", action="store_true")
    ap.add_argument("--grid", type=int, nargs=3, default=(161, 64, 161), help="table nR nphi nZ (nphi per period)")
    ap.add_argument("--periods", type=int, default=400)
    ap.add_argument("--steps", type=int, default=64)
    ap.add_argument("--lines", type=int, default=32)
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
    us = {} if a.skip_start else {"start": np.asarray(cur.u)}
    for spec in a.state:
        name, path = spec.split("=", 1) if "=" in spec else ("optimized", spec)
        us[name] = np.load(path)["u"]
    ra, za = fields.cyl_grid.r_axis, fields.cyl_grid.z_axis
    eps_w = 1e-4
    Rs = np.linspace(ra.a + eps_w, ra.b - eps_w, a.grid[0])
    Zs = np.linspace(za.a + eps_w, za.b - eps_w, a.grid[2])
    phis = 2 * np.pi / eq.NFP * np.arange(a.grid[1]) / a.grid[1]
    res_ = pc.resonances(inp)
    RZ = Separable([eq.R_basis, eq.Z_basis], [eq.R_lmn, eq.Z_lmn], eq.NFP)
    uniform = np.column_stack([np.linspace(0.04, 0.995, a.lines), np.zeros(a.lines)])
    extra = np.array([(min(r + d, 0.995), th) for _, _, r in res_ for d in (-0.02, -0.01, 0.0, 0.01, 0.02)
                      for th in (0.0, np.pi)]).reshape(-1, 2)
    more = np.array([(min(r + d, 0.995), np.pi / m) for _, m, r in res_
                     for d in np.linspace(-0.03, 0.03, a.pi_m_seeds)]).reshape(-1, 2)
    seeds = np.concatenate([uniform, extra, more])
    if a.res_seeds:
        pts = []
        for item in a.res_seeds.split(","):
            r0, m = item.split(":"); r0, m = float(r0), int(m)
            for d in np.linspace(-0.05, 0.05, 11):
                for th in (0.0, np.pi / m, np.pi, np.pi + np.pi / m):
                    pts.append((min(max(r0 + d, 0.02), 0.995), th))
        seeds = np.array(pts)
    st = np.asarray(RZ.evaluate(jnp.asarray(seeds[:, 0]), jnp.asarray(seeds[:, 1]), jnp.zeros(len(seeds))))
    sections, checks = {}, {}
    for name, u in us.items():
        t1 = time.time()
        B = table(fields, inp, u, Rs, phis, Zs)
        checks[name] = check(fields, Rs, phis, Zs, B, u, eq.NFP)
        print(f"{name}: table {B.shape[:3]} in {time.time() - t1:.0f} s; gate (table vs solver B at the "
              f"collocation points): rel rms {checks[name][0]:.2e}, max {checks[name][1]:.2e}", flush=True)
        t1 = time.time()
        sections[name] = pc.trace(Rs, phis, Zs, B, st[:, 0], st[:, 1], eq.NFP, a.periods, a.steps)
        lost = int(np.isnan(sections[name][0][-1]).sum())
        print(f"{name}: {len(seeds)} lines x {a.periods} periods, {lost} reached the annulus wall; "
              f"{time.time() - t1:.0f} s", flush=True)
    os.makedirs(os.path.dirname(os.path.abspath(a.out)), exist_ok=True)
    np.savez(a.out, resonances=np.array(res_), seeds=seeds,
             **{f"{k}_check": np.array(v) for k, v in checks.items()},
             **{f"{k}_R": v[0] for k, v in sections.items()}, **{f"{k}_Z": v[1] for k, v in sections.items()},
             **{f"{k}_Rh": v[2] for k, v in sections.items()}, **{f"{k}_Zh": v[3] for k, v in sections.items()})
    print(f"saved {a.out}; total {time.time() - t0:.0f} s")


if __name__ == "__main__":
    main()
