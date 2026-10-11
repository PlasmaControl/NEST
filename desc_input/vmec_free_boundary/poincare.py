"""Field lines of a NEST solution: Poincare crossings of phi = 0 for any equilibrium and coils.

B_ext + B_plasma is tabulated once on a cylindrical grid over a field period and
the field lines are traced in phi with RK4 on cubic splines (the tracer of
``scripts/validate_opt.py``, copied here so that only ``nest/`` is needed). This is
done for the start (the fits of DESC's J and p) and for the optimized x.

On top of the uniform seeds along theta = 0, lines are seeded around each resonant
surface iota = n/m (n a multiple of NFP, m <= 12) on theta = 0 and theta = pi:
with stellarator symmetry the island O- and X-points of the phi = 0 section lie on
that line. With ``--island-lines`` k, k more seeds cross each low-order chain
(m <= 6) on theta = pi / m, for chains with X-points at both theta = 0 and pi.

The crossings go to figures/<x name>_poincare.npz (prefixed with ncsx_ for the NCSX
runs). ``poincare_flux.py`` maps them to DESC (rho, theta), measures iota per line
and draws the figure.

run (GPU): sbatch --mem=128G run.sbatch desc_input/vmec_free_boundary/poincare.py \
    --eq desc_input/vmec_free_boundary/NCSX/NCSX_L16M16N12.h5 \
    --coils desc_input/vmec_free_boundary/NCSX/NCSX_coils.h5 \
    --x results/ncsx/opt_L16M16N12.npz --frac 20
``--x start`` traces the start only (a smoke test).
"""

import argparse
import os
import sys
import time
from fractions import Fraction

import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.join(ROOT, "nest"))

from desc import set_device  # noqa: E402

set_device(os.environ.get("NEST_DEVICE", "cpu"))

import jax  # noqa: E402
import jax.numpy as jnp  # noqa: E402

import fit_cache  # noqa: E402
from boundary_potential import BoundaryPotential  # noqa: E402
from desc_input import DESCInput  # noqa: E402
from force import ForceResidual  # noqa: E402
from fourier import fft_J  # noqa: E402
from grid import CartesianGrid  # noqa: E402
from magnetic_field import interpolate  # noqa: E402
from objective import Fields  # noqa: E402
from pressure import Pressure  # noqa: E402
from separable import Separable  # noqa: E402

jax.config.update("jax_enable_x64", True)


def field_on_cylinder(fields, u, inp, nR=121, nphi=64, nZ=121, margin=0.25):
    """B_ext + B_plasma (R, phi, Z components) on a cylindrical grid over one field period."""
    eq = inp.eq
    RZ = Separable([eq.R_basis, eq.Z_basis], [eq.R_lmn, eq.Z_lmn], eq.NFP)
    t = np.linspace(0, 2 * np.pi, 200)
    z = np.linspace(0, 2 * np.pi / eq.NFP, 40)
    T, Zt = np.meshgrid(t, z)
    b = np.asarray(RZ.evaluate(jnp.ones(T.size), jnp.asarray(T.ravel()), jnp.asarray(Zt.ravel())))
    a = inp.a
    Rs = np.linspace(b[:, 0].min() - margin * a, b[:, 0].max() + margin * a, nR)
    Zs = np.linspace(b[:, 1].min() - margin * a, b[:, 1].max() + margin * a, nZ)
    phis = 2 * np.pi / eq.NFP * np.arange(nphi) / nphi
    R, P, Z = np.meshgrid(Rs, phis, Zs, indexing="ij")
    X = np.column_stack([(R * np.cos(P)).ravel(), (R * np.sin(P)).ravel(), Z.ravel()])
    B_box = fields._solve(fft_J(fields.J_box(jnp.asarray(u))), fields.axes, fields.K_hat)
    Bc = inp.B_ext(X) + np.asarray(interpolate(np.asarray(B_box), fields.axes, X))
    c, s = np.cos(P.ravel()), np.sin(P.ravel())
    Bcyl = np.column_stack([Bc[:, 0] * c + Bc[:, 1] * s, -Bc[:, 0] * s + Bc[:, 1] * c, Bc[:, 2]])
    return Rs, phis, Zs, Bcyl.reshape(nR, nphi, nZ, 3)


def trace(Rs, phis, Zs, B, R0, Z0, NFP, n_periods=400, steps=64):
    """Crossings of phi = 0 (mod one field period) of the lines from (R0, Z0)."""
    from scipy.ndimage import map_coordinates, spline_filter
    # periodic in phi: pad 3 planes on each side for the cubic spline
    Bp = np.concatenate([B[:, -3:], B, B[:, :3]], axis=1)
    coef = [spline_filter(Bp[..., k], order=3, mode="nearest") for k in range(3)]
    dR, dZ, dphi = Rs[1] - Rs[0], Zs[1] - Zs[0], phis[1] - phis[0]
    period = 2 * np.pi / NFP

    def rhs(R, Z, phi):
        q = np.stack([(R - Rs[0]) / dR, np.full_like(R, np.mod(phi, period) / dphi + 3),
                      (Z - Zs[0]) / dZ])
        BR, Bphi, BZ = (map_coordinates(c, q, order=3, prefilter=False, mode="nearest")
                        for c in coef)
        return R * BR / Bphi, R * BZ / Bphi

    R, Z = np.array(R0, float), np.array(Z0, float)
    h = period / steps
    out_R, out_Z = [R.copy()], [Z.copy()]
    alive = np.ones(len(R), bool)
    for k in range(n_periods):
        for i in range(steps):
            phi = (k * steps + i) * h
            k1 = rhs(R, Z, phi)
            k2 = rhs(R + h / 2 * k1[0], Z + h / 2 * k1[1], phi + h / 2)
            k3 = rhs(R + h / 2 * k2[0], Z + h / 2 * k2[1], phi + h / 2)
            k4 = rhs(R + h * k3[0], Z + h * k3[1], phi + h)
            R = R + h / 6 * (k1[0] + 2 * k2[0] + 2 * k3[0] + k4[0])
            Z = Z + h / 6 * (k1[1] + 2 * k2[1] + 2 * k3[1] + k4[1])
        alive &= (R > Rs[1]) & (R < Rs[-2]) & (Z > Zs[1]) & (Z < Zs[-2])
        out_R.append(np.where(alive, R, np.nan))
        out_Z.append(np.where(alive, Z, np.nan))
    return np.array(out_R), np.array(out_Z)


def resonances(inp, max_m=12):
    """(n, m, rho) of iota = n/m crossings, n a multiple of NFP, lowest m first."""
    rho = np.linspace(1e-3, 1, 2001)
    iota = np.abs(inp.iota(rho)[0])
    out, seen = [], set()
    for m in range(1, max_m + 1):
        for n in range(inp.NFP, 4 * inp.NFP + 1, inp.NFP):
            t = Fraction(n, m)
            if t in seen:
                continue
            idx = np.nonzero(np.diff(np.sign(iota - float(t))))[0]
            if len(idx):
                seen.add(t)
            for i in idx:
                r = rho[i] + (float(t) - iota[i]) * (rho[i + 1] - rho[i]) / (iota[i + 1] - iota[i])
                out.append((n, m, float(r)))
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--eq", required=True)
    ap.add_argument("--coils", required=True)
    ap.add_argument("--x", required=True, help="optimize_qa.py npz, or start")
    ap.add_argument("--frac", type=int, default=20)
    ap.add_argument("--kernel", default="truncated")
    ap.add_argument("--factors", type=float, nargs=3, default=(3, 3, 2.5))
    ap.add_argument("--periods", type=int, default=400)
    ap.add_argument("--lines", type=int, default=32, help="uniform seeds on theta = 0")
    ap.add_argument("--island-lines", type=int, default=0,
                    help="extra seeds across each low-order chain (m <= 6) on theta = pi / m")
    ap.add_argument("--island-width", type=float, default=0.12,
                    help="those seeds span the resonant rho +- this")
    ap.add_argument("--out", default=None, help="npz (default figures/<x name>_poincare.npz)")
    a = ap.parse_args()
    t0 = time.time()
    inp = DESCInput.load(a.eq, field=a.coils)
    eq = inp.eq
    cur = fit_cache.fit(BoundaryPotential, inp, factors=tuple(a.factors))
    pre = fit_cache.fit(Pressure, inp, factors=tuple(a.factors), edge_order=2)
    box = CartesianGrid.periodic_box(inp, inp.a / a.frac, odd=bool(eq.sym)).locate_plasma(inp)
    us = {"start": cur.u}
    if a.x != "start":
        us["optimized"] = np.load(a.x)["x"][: cur.n_free]
    fields = Fields(ForceResidual(inp, box), cur, pre, kernel=a.kernel)
    res = resonances(inp)
    print(f"{inp}\nx: {a.x}; box h = a/{a.frac} {box.shape}\nresonances (n, m, rho): "
          + ", ".join(f"{n}/{m} at {r:.3f}" for n, m, r in res), flush=True)

    RZ = Separable([eq.R_basis, eq.Z_basis], [eq.R_lmn, eq.Z_lmn], eq.NFP)
    seeds = [(r, 0.0) for r in np.linspace(0.04, 1.0, a.lines)]
    for _, _, r in res:
        seeds += [(min(r + d, 0.995), th) for d in (-0.02, -0.01, 0.0, 0.01, 0.02)
                  for th in (0.0, np.pi)]
    # theta = 0, pi hold the O- or X-points (stellarator symmetry); theta = pi / m is
    # half way to the next one, so it crosses the islands when 0 and pi are X-points
    for _, m, r in res:
        if m <= 6:
            seeds += [(min(r + d, 0.995), np.pi / m)
                      for d in np.linspace(-a.island_width, a.island_width, a.island_lines)]
    seeds = np.array([s for s in seeds if s[0] > 0.02])
    st = np.asarray(RZ.evaluate(jnp.asarray(seeds[:, 0]), jnp.asarray(seeds[:, 1]),
                                jnp.zeros(len(seeds))))
    sections = {}
    for name, u in us.items():
        t1 = time.time()
        Rs, phis, Zs, Bcyl = field_on_cylinder(fields, u, inp)
        sections[name] = trace(Rs, phis, Zs, Bcyl, st[:, 0], st[:, 1], eq.NFP, a.periods)
        lost = np.isnan(sections[name][0][-1]).sum()
        print(f"{name}: {len(seeds)} lines x {a.periods} field periods, {lost} left the grid; "
              f"{time.time() - t1:.0f} s", flush=True)

    out = a.out
    if out is None:
        name = os.path.splitext(os.path.basename(a.x))[0]
        if "ncsx" in a.x.lower() and not name.startswith("ncsx"):
            name = "ncsx_" + name
        out = os.path.join(ROOT, "figures", f"{name}_poincare.npz")
    os.makedirs(os.path.dirname(os.path.abspath(out)), exist_ok=True)
    np.savez(out, seeds=seeds, resonances=np.array(res),
             **{f"{k}_R": v[0] for k, v in sections.items()},
             **{f"{k}_Z": v[1] for k, v in sections.items()})
    print(f"saved {out}; total {time.time() - t0:.0f} s")


if __name__ == "__main__":
    main()
