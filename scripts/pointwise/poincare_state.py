"""Poincare sections of pointwise-bound states in the annulus field, and p on the field lines.

B = B_ext + B_plasma on a cylindrical tracing grid over one field period, B_plasma from the
annulus solve of the run's resolution (its modal slabs synthesized at the grid: exactly
NFP-periodic, no averaging), traced by RK4 on cubic splines (``poincare.trace``). The
crossings of phi = 0 are mapped to DESC's (rho, theta) (``grid.map_to_flux``); per line:
iota = NFP <dtheta> / 2 pi and the state's p at each crossing.

    python scripts/pointwise/poincare_state.py --tables T.npz --state start \
        --state a0.1=results/pointwise/X_a0.1.npz --out results/pointwise/X_poincare.npz
"""

import argparse
import faulthandler
import json
import os
import sys
import time

import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.join(ROOT, "nest"))
sys.path.insert(0, os.path.join(ROOT, "scripts"))
sys.path.insert(0, os.path.join(ROOT, "scripts", "pointwise"))
sys.path.insert(0, os.path.join(ROOT, "desc_input", "vmec_free_boundary"))
from desc import set_device  # noqa: E402

set_device(os.environ.get("NEST_DEVICE", "cpu"))
import jax  # noqa: E402
import jax.numpy as jnp  # noqa: E402

from build_tables import setup  # noqa: E402
from cylindrical_grid import CylindricalInterpolator  # noqa: E402
from grid import map_to_flux, plasma_extent  # noqa: E402
from pressure import Pressure  # noqa: E402

jax.config.update("jax_enable_x64", True)
faulthandler.enable()

_BEXT = {}


def field_grid(fields, inp, box, u, nR, nphi, nZ, pad):
    """(Rs, phis, Zs, B (nR, nphi, nZ, 3) cylindrical) over one period; B_ext computed once
    (in chunks of points) and reused."""
    R_min, R_max, Z_min, Z_max = plasma_extent(inp)
    ra, za = box.r_axis, box.z_axis
    R_lo, R_hi = max(R_min - pad, ra.a), min(R_max + pad, ra.b)
    Z_lo, Z_hi = max(Z_min - pad, za.a), min(Z_max + pad, za.b)
    Rs, Zs = np.linspace(R_lo, R_hi, nR), np.linspace(Z_lo, Z_hi, nZ)
    phis = 2 * np.pi / inp.NFP * np.arange(nphi) / nphi
    RR, PP, ZZ = np.meshgrid(Rs, phis, Zs, indexing="ij")
    pts = np.column_stack([RR.ravel(), PP.ravel(), ZZ.ravel()])
    Bp, Bz = fields.solver.field_modes(fields.J_box(jnp.asarray(u)))
    Bpl = np.concatenate([np.asarray(CylindricalInterpolator(box, pts[i : i + 400000])(Bp, Bz))
                          for i in range(0, len(pts), 400000)])  # cylindrical (B_r, B_phi, B_z)
    key = (nR, nphi, nZ, pad)
    if key not in _BEXT:
        x = np.column_stack([pts[:, 0] * np.cos(pts[:, 1]), pts[:, 0] * np.sin(pts[:, 1]), pts[:, 2]])
        Be = np.concatenate([inp.B_ext(x[i : i + 200000]) for i in range(0, len(x), 200000)])
        c, s = np.cos(pts[:, 1]), np.sin(pts[:, 1])
        _BEXT[key] = np.column_stack([Be[:, 0] * c + Be[:, 1] * s, -Be[:, 0] * s + Be[:, 1] * c,
                                      Be[:, 2]])
    Bext = _BEXT[key]
    return Rs, phis, Zs, (Bpl + Bext).reshape(nR, nphi, nZ, 3)


def trace(Rs, phis, Zs, B, R0, Z0, NFP, n_periods=400, steps=64):
    """``poincare.trace`` with a guard: a line whose position becomes non-finite or leaves
    the grid is stopped (scipy's map_coordinates crashes on NaN coordinates). Returns the
    crossings of phi = 0 and of phi = pi / NFP."""
    from scipy.ndimage import map_coordinates, spline_filter
    Bp = np.concatenate([B[:, -3:], B, B[:, :3]], axis=1)
    coef = [spline_filter(Bp[..., k], order=3, mode="nearest") for k in range(3)]
    dR, dZ, dphi = Rs[1] - Rs[0], Zs[1] - Zs[0], phis[1] - phis[0]
    period = 2 * np.pi / NFP
    R, Z = np.array(R0, float), np.array(Z0, float)
    alive = np.isfinite(R) & np.isfinite(Z)

    def rhs(R, Z, phi):
        Rq = np.where(alive & np.isfinite(R), R, Rs[len(Rs) // 2])
        Zq = np.where(alive & np.isfinite(Z), Z, Zs[len(Zs) // 2])
        q = np.stack([np.clip((Rq - Rs[0]) / dR, 0, len(Rs) - 1),
                      np.full_like(Rq, np.mod(phi, period) / dphi + 3),
                      np.clip((Zq - Zs[0]) / dZ, 0, len(Zs) - 1)])
        BR, Bphi, BZ = (map_coordinates(c, q, order=3, prefilter=False, mode="nearest")
                        for c in coef)
        return Rq * BR / Bphi, Rq * BZ / Bphi

    h = period / steps
    out_R, out_Z, half_R, half_Z = [R.copy()], [Z.copy()], [], []
    for k in range(n_periods):
        for i in range(steps):
            phi = (k * steps + i) * h
            k1 = rhs(R, Z, phi)
            k2 = rhs(R + h / 2 * k1[0], Z + h / 2 * k1[1], phi + h / 2)
            k3 = rhs(R + h / 2 * k2[0], Z + h / 2 * k2[1], phi + h / 2)
            k4 = rhs(R + h * k3[0], Z + h * k3[1], phi + h)
            R = R + h / 6 * (k1[0] + 2 * k2[0] + 2 * k3[0] + k4[0])
            Z = Z + h / 6 * (k1[1] + 2 * k2[1] + 2 * k3[1] + k4[1])
            alive &= np.isfinite(R) & np.isfinite(Z) & (R > Rs[1]) & (R < Rs[-2]) & \
                (Z > Zs[1]) & (Z < Zs[-2])
            if i == steps // 2 - 1:
                half_R.append(np.where(alive, R, np.nan))
                half_Z.append(np.where(alive, Z, np.nan))
        out_R.append(np.where(alive, R, np.nan))
        out_Z.append(np.where(alive, Z, np.nan))
    return np.array(out_R), np.array(out_Z), np.array(half_R), np.array(half_Z)


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--tables", required=True)
    ap.add_argument("--state", action="append", required=True, help="start, or name=npz")
    ap.add_argument("--grid", type=int, nargs=3, default=(241, 96, 361))
    ap.add_argument("--pad", type=float, default=0.02)
    ap.add_argument("--periods", type=int, default=400)
    ap.add_argument("--steps", type=int, default=96)
    ap.add_argument("--lines", type=int, nargs=2, default=(64, 40),
                    help="seeds uniform in rho on theta = 0 and on theta = pi")
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    t0 = time.time()
    with np.load(a.tables, mmap_mode="r") as f:
        meta = json.loads(str(f["meta"]))
        u0, c0 = np.asarray(f["u0"]), np.asarray(f["c0"])
    inp, cur, pre, box, fr, fields = setup(meta["eq"], meta["coils"], meta["factors"],
                                           meta["p_factors"], meta["frac"], meta["colloc"])
    eq = inp.eq
    from separable import Separable
    RZ = Separable([eq.R_basis, eq.Z_basis], [eq.R_lmn, eq.Z_lmn], eq.NFP)
    seeds = np.concatenate([np.column_stack([np.linspace(0.03, 0.99, a.lines[0]),
                                             np.zeros(a.lines[0])]),
                            np.column_stack([np.linspace(0.3, 0.99, a.lines[1]),
                                             np.full(a.lines[1], np.pi)])])
    st = np.asarray(RZ.evaluate(jnp.asarray(seeds[:, 0]), jnp.asarray(seeds[:, 1]),
                                jnp.zeros(len(seeds))))
    out = {"seeds": seeds, "NFP": eq.NFP}
    n_u = cur.n_free
    for spec in a.state:
        name, path = (spec, None) if "=" not in spec else spec.split("=", 1)
        x = np.concatenate([u0, c0]) if path is None else np.load(path)["x"]
        t1 = time.time()
        Rs, phis, Zs, B = field_grid(fields, inp, box, x[:n_u], *a.grid, a.pad)
        print(f"{name}: field grid {B.shape}, non-finite {int((~np.isfinite(B)).sum())}, "
              f"min |B_phi| {np.abs(B[..., 1]).min():.3e}", flush=True)
        R, Z, Rh, Zh = trace(Rs, phis, Zs, B, st[:, 0], st[:, 1], eq.NFP, a.periods, a.steps)
        X = np.column_stack([R.ravel(), np.zeros(R.size), Z.ravel()])
        rtz = np.full_like(X, np.nan)
        ok = np.isfinite(X).all(1)
        rtz[ok] = map_to_flux(inp, X[ok])
        rho, th = rtz[:, 0].reshape(R.shape), rtz[:, 1].reshape(R.shape)
        good = np.isfinite(rtz).all(1)
        p = np.full(R.size, np.nan)
        P = Pressure(pre.basis, x[n_u:], eq, pre.edge_order).point_evaluator()
        p[good] = np.asarray(P(*(jnp.asarray(rtz[good, j]) for j in range(3))))[:, 0]
        thu = np.unwrap(np.where(np.isfinite(th), th, 0.0), axis=0)
        iota = eq.NFP * np.nanmean(np.diff(thu, axis=0), axis=0) / (2 * np.pi)
        lost = ~np.isfinite(rho).all(0)
        iota[lost] = np.nan
        out.update({f"{name}_R": R, f"{name}_Z": Z, f"{name}_Rh": Rh, f"{name}_Zh": Zh,
                    f"{name}_rho": rho, f"{name}_theta": th, f"{name}_p": p.reshape(R.shape),
                    f"{name}_iota": iota})
        print(f"{name}: {len(seeds)} lines x {a.periods} periods, {int(lost.sum())} lost or "
              f"outside; {time.time() - t1:.0f} s", flush=True)
    os.makedirs(os.path.dirname(os.path.abspath(a.out)), exist_ok=True)
    np.savez(a.out, **out)
    print(f"saved {a.out}; {time.time() - t0:.0f} s")


if __name__ == "__main__":
    main()
