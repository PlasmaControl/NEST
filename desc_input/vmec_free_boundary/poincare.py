"""Field lines of a NEST solution: Poincare crossings of phi = 0 for any equilibrium and coils.

B_ext + B_plasma is tabulated once on a cylindrical grid over a field period and
the field lines are traced in phi with RK4 on cubic splines (the tracer of
``scripts/validate_opt.py``, copied here so that only ``nest/`` is needed). This is
done for the start (the fits of DESC's J and p) and for the optimized x. B_plasma is
averaged over the NFP rotations first: the box's field differs between periods by
~1e-4 |B| rms, and the jump at the seam of the repeated period is a source of B that
pushes every line outward (``--no-periodize`` traces it as it is).

On top of the uniform seeds along theta = 0, lines are seeded around each resonant
surface iota = n/m (n a multiple of NFP, m <= 12). By default these are DESC's
resonances, on theta = 0 and pi; with stellarator symmetry the island O- and X-points
of the phi = 0 section lie on that line. ``--island-lines`` k adds k seeds across each
low-order chain (m <= 6) on theta = pi / m, for chains with X-points at both theta = 0
and pi. With ``--islands`` (the _poincare_flux.npz of a previous trace) each field is
instead seeded at its own measured resonances, across the chain (rho_res +- 0.03) and
at the straight-field-line angles theta* = 0, pi/m, pi, pi + pi/m. The O-points of the
phi = 0 section sit at one of these, so the field lines inside the islands are traced
too.

The crossings of phi = 0 (keys <run>_R, <run>_Z) and of phi = pi / NFP (<run>_Rh,
<run>_Zh) go to figures/<x name>_poincare.npz (prefixed with ncsx_ for the NCSX
runs). ``poincare_flux.py`` maps them to DESC (rho, theta), measures iota per line
and draws that figure; ``poincare_rz.py`` draws them in (R, Z) over DESC's surfaces.

run (GPU): sbatch --mem=128G run.sbatch desc_input/vmec_free_boundary/poincare.py \
    --eq desc_input/vmec_free_boundary/NCSX/NCSX_L16M16N12.h5 \
    --coils desc_input/vmec_free_boundary/NCSX/NCSX_coils.h5 \
    --x results/ncsx/opt_L16M16N12.npz --frac 20
``--x start`` traces the start only (a smoke test). For a free-boundary run
(``scripts/optimize_fb.py``) pass its moved geometry, ``--eq <out>_eq.h5 --x <out>.npz
--skip-start``.
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


def field_on_cylinder(fields, u, inp, nR=121, nphi=64, nZ=121, margin=0.25, periodize=True):
    """B_ext + B_plasma (R, phi, Z components) on a cylindrical grid over one field period.

    The box's B_plasma is not exactly NFP-periodic (a Cartesian box is not
    NFP-symmetric): tabulated over one period and repeated, it jumps at the period
    seam. With ``periodize``, B_plasma is averaged over the NFP rotations (the
    true field is periodic), and the size of the removed part is printed."""
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
    per = 1_000_000  # B_ext in chunks (a fine grid has millions of points)
    Bc = np.concatenate([inp.B_ext(X[i : i + per]) for i in range(0, len(X), per)])
    B_box = np.asarray(B_box)
    if not periodize:
        Bc += np.asarray(interpolate(B_box, fields.axes, X))
    else:
        Bk = []
        for k in range(eq.NFP):  # B_plasma at the rotated points, rotated back
            a_k = 2 * np.pi * k / eq.NFP
            ck, sk = np.cos(a_k), np.sin(a_k)
            Xk = np.column_stack([ck * X[:, 0] - sk * X[:, 1], sk * X[:, 0] + ck * X[:, 1], X[:, 2]])
            B = np.asarray(interpolate(B_box, fields.axes, Xk))
            Bk.append(np.column_stack([ck * B[:, 0] + sk * B[:, 1], -sk * B[:, 0] + ck * B[:, 1],
                                       B[:, 2]]))
        Bm = np.mean(Bk, axis=0)
        dev = np.sqrt(np.mean([np.sum((b - Bm) ** 2, 1) for b in Bk], axis=0))
        mag = np.linalg.norm(Bc + Bm, axis=1)
        print(f"periodized B_plasma: its period-to-period part is {np.sqrt(np.mean((dev / mag) ** 2)):.1e} "
              f"of |B| rms ({np.max(dev / mag):.1e} max) on the tracing grid", flush=True)
        Bc += Bm
    c, s = np.cos(P.ravel()), np.sin(P.ravel())
    Bcyl = np.column_stack([Bc[:, 0] * c + Bc[:, 1] * s, -Bc[:, 0] * s + Bc[:, 1] * c, Bc[:, 2]])
    return Rs, phis, Zs, Bcyl.reshape(nR, nphi, nZ, 3)


def trace(Rs, phis, Zs, B, R0, Z0, NFP, n_periods=400, steps=64):
    """Crossings of phi = 0 and of phi = pi / NFP (mod one field period) of the lines
    from (R0, Z0): (R, Z) at phi = 0 and (R, Z) at half a period."""
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
    out_R, out_Z, half_R, half_Z = [R.copy()], [Z.copy()], [], []
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
            if i == steps // 2 - 1:  # phi = pi / NFP reached
                ok = alive & (R > Rs[1]) & (R < Rs[-2]) & (Z > Zs[1]) & (Z < Zs[-2])
                half_R.append(np.where(ok, R, np.nan))
                half_Z.append(np.where(ok, Z, np.nan))
        alive &= (R > Rs[1]) & (R < Rs[-2]) & (Z > Zs[1]) & (Z < Zs[-2])
        out_R.append(np.where(alive, R, np.nan))
        out_Z.append(np.where(alive, Z, np.nan))
    return np.array(out_R), np.array(out_Z), np.array(half_R), np.array(half_Z)


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


def resonances_of_lines(rho, iota, NFP, max_m=12, merge=0.03):
    """(n, m, rho) where the measured iota of the traced lines (mean rho, iota per
    line) crosses n/m; crossings of the same n/m closer than ``merge`` are one."""
    o = np.argsort(rho)
    rho, iota = rho[o], iota[o]
    keep = np.r_[True, np.diff(rho) > 5e-3]
    rho, iota = rho[keep], iota[keep]
    out, seen = [], set()
    for m in range(1, max_m + 1):
        for n in range(NFP, 4 * NFP + 1, NFP):
            t = Fraction(n, m)
            if t in seen:
                continue
            idx = np.nonzero(np.diff(np.sign(iota - float(t))))[0]
            if len(idx):
                seen.add(t)
            last = -1.0
            for i in idx:
                r = rho[i] + (float(t) - iota[i]) * (rho[i + 1] - rho[i]) / (iota[i + 1] - iota[i])
                if r - last > merge:
                    out.append((n, m, float(r)))
                    last = r
    return out


def island_seeds(eq, res, width=0.03, n_rho=9):
    """(rho, theta) seeds across each chain n/m at rho_res: rho_res +- width and the
    straight-field-line angles theta* = 0, pi/m, pi, pi + pi/m at phi = 0. With
    stellarator symmetry the O-points of the phi = 0 section sit at one of these."""
    pts = []
    for _, m, r in res:
        for d in np.linspace(-width, width, n_rho):
            for ts in (0.0, np.pi / m, np.pi, np.pi + np.pi / m):
                pts.append((min(max(r + d, 0.02), 0.995), ts, 0.0))
    if not pts:
        return np.zeros((0, 2))
    out = np.asarray(eq.map_coordinates(np.array(pts), inbasis=("rho", "theta_PEST", "zeta"),
                                        outbasis=("rho", "theta", "zeta"), maxiter=60,
                                        tol=1e-10))
    ok = np.isfinite(out).all(1)
    return out[ok, :2]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--eq", required=True)
    ap.add_argument("--coils", required=True)
    ap.add_argument("--x", required=True, help="optimize_qa.py npz, or start")
    ap.add_argument("--frac", type=int, default=20)
    ap.add_argument("--kernel", default="truncated")
    ap.add_argument("--factors", type=float, nargs=3, default=(3, 3, 2.5))
    ap.add_argument("--periods", type=int, default=400)
    ap.add_argument("--grid", type=int, nargs=3, default=(121, 64, 121),
                    help="cylindrical grid nR nphi nZ of the traced field (nphi per period)")
    ap.add_argument("--steps", type=int, default=64, help="RK4 steps per field period")
    ap.add_argument("--periodize", action=argparse.BooleanOptionalAction, default=True,
                    help="average B_plasma over the NFP rotations before tracing (default). "
                         "The box's field is not exactly periodic, and its jump at the period "
                         "seam moves every line outward (NCSX: rho +0.07 in 800 periods)")
    ap.add_argument("--lines", type=int, default=32, help="uniform seeds on theta = 0")
    ap.add_argument("--island-lines", type=int, default=0,
                    help="extra seeds across each low-order chain (m <= 6) on theta = pi / m")
    ap.add_argument("--island-width", type=float, default=0.12,
                    help="those seeds span the resonant rho +- this")
    ap.add_argument("--out", default=None, help="npz (default figures/<x name>_poincare.npz)")
    ap.add_argument("--skip-start", action="store_true",
                    help="trace x only (e.g. a free-boundary NEST run: --eq its _eq.h5, whose "
                         "fits are not a start)")
    ap.add_argument("--islands", default=None,
                    help="_poincare_flux.npz of a previous trace (poincare_flux.py): seed the "
                         "chains of each field at its own measured resonances, across the "
                         "chain and at the O-point angles (else around DESC's resonances)")
    a = ap.parse_args()
    t0 = time.time()
    inp = DESCInput.load(a.eq, field=a.coils)
    eq = inp.eq
    cur = fit_cache.fit(BoundaryPotential, inp, factors=tuple(a.factors))
    pre = fit_cache.fit(Pressure, inp, factors=tuple(a.factors), edge_order=2)
    box = CartesianGrid.periodic_box(inp, inp.a / a.frac, odd=bool(eq.sym)).locate_plasma(inp)
    us = {} if a.skip_start and a.x != "start" else {"start": cur.u}
    if a.x != "start":
        us["optimized"] = np.load(a.x)["x"][: cur.n_free]
    fields = Fields(ForceResidual(inp, box), cur, pre, kernel=a.kernel)
    res = resonances(inp)
    print(f"{inp}\nx: {a.x}; box h = a/{a.frac} {box.shape}\nresonances (n, m, rho): "
          + ", ".join(f"{n}/{m} at {r:.3f}" for n, m, r in res), flush=True)

    RZ = Separable([eq.R_basis, eq.Z_basis], [eq.R_lmn, eq.Z_lmn], eq.NFP)
    uniform = np.column_stack([np.linspace(0.04, 1.0, a.lines), np.zeros(a.lines)])
    prev = np.load(a.islands) if a.islands else None
    sections, seeds, run_res = {}, {}, {}
    for name, u in us.items():
        if prev is not None and f"{name}_line_rho" in prev:
            run_res[name] = resonances_of_lines(prev[f"{name}_line_rho"],
                                                prev[f"{name}_line_iota"], eq.NFP)
            extra = island_seeds(eq, run_res[name])
        else:
            run_res[name] = res
            extra = np.array([(min(r + d, 0.995), th) for _, _, r in res
                              for d in (-0.02, -0.01, 0.0, 0.01, 0.02) for th in (0.0, np.pi)])
            # theta = 0, pi hold the O- or X-points (stellarator symmetry); theta = pi / m
            # is half way to the next one, so it crosses the islands when 0 and pi are
            # X-points
            more = [(min(r + d, 0.995), np.pi / m) for _, m, r in res if m <= 6
                    for d in np.linspace(-a.island_width, a.island_width, a.island_lines)]
            if more:
                extra = np.concatenate([extra, np.array(more)])
        s = np.concatenate([uniform, extra.reshape(-1, 2)])
        seeds[name] = s[s[:, 0] > 0.02]
        st = np.asarray(RZ.evaluate(jnp.asarray(seeds[name][:, 0]), jnp.asarray(seeds[name][:, 1]),
                                    jnp.zeros(len(seeds[name]))))
        t1 = time.time()
        Rs, phis, Zs, Bcyl = field_on_cylinder(fields, u, inp, *a.grid, periodize=a.periodize)
        sections[name] = trace(Rs, phis, Zs, Bcyl, st[:, 0], st[:, 1], eq.NFP, a.periods,
                               a.steps)
        lost = np.isnan(sections[name][0][-1]).sum()
        print(f"{name}: chains seeded at " + ", ".join(f"{n}/{m} at {r:.3f}"
                                                      for n, m, r in run_res[name])
              + f"; {len(seeds[name])} lines x {a.periods} field periods, {lost} left the "
              f"grid; {time.time() - t1:.0f} s", flush=True)

    out = a.out
    if out is None:
        name = os.path.splitext(os.path.basename(a.x))[0]
        if "ncsx" in a.x.lower() and not name.startswith("ncsx"):
            name = "ncsx_" + name
        out = os.path.join(ROOT, "figures", f"{name}_poincare.npz")
    os.makedirs(os.path.dirname(os.path.abspath(out)), exist_ok=True)
    np.savez(out, resonances=np.array(res),
             **{f"{k}_seeds": v for k, v in seeds.items()},
             **{f"{k}_resonances": np.array(v) for k, v in run_res.items()},
             **{f"{k}_R": v[0] for k, v in sections.items()},
             **{f"{k}_Z": v[1] for k, v in sections.items()},
             **{f"{k}_Rh": v[2] for k, v in sections.items()},
             **{f"{k}_Zh": v[3] for k, v in sections.items()})
    print(f"saved {out}; total {time.time() - t0:.0f} s")


if __name__ == "__main__":
    main()
