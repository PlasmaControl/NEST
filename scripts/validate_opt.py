"""Physical checks of an optimized x (``optimize_qa.py`` npz) and Poincare plots.

  1. pressure: min p / max p over the whole plasma (rho up to 1), and where
     p < 0 if anywhere; the change from the fitted (DESC) pressure
  2. current: the net toroidal current through two cross-sections (zeta = 0 and
     half a field period) against I; max |J| on the LCFS / max |J|
  3. F with J exactly on the box (pointwise series at the box points and at the
     collocation points, ``ForceResidual.evaluate``) against the interpolated J
     of the objective, same grid, same (Hockney-Eastwood) kernel
  4. F split along b = B / |B| (F_par = -b . grad p) and across, by radius
  5. Poincare sections at phi = 0 of the field B_ext + B_plasma for the start
     (the fits) and for x: the total field is tabulated once on a cylindrical
     grid around the plasma, field lines are traced in phi (RK4, cubic splines);
     DESC's flux surfaces are drawn on top; a third panel shows p(x) - p(start)
Figure: figures/<name>_poincare.png.

run (GPU): sbatch --mem=128G run.sbatch scripts/validate_opt.py --x results/freeP20.npz --frac 20
"""

import argparse
import gc
import os
import sys
import time

import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, "nest"))

from desc import set_device  # noqa: E402

set_device(os.environ.get("NEST_DEVICE", "cpu"))

import jax  # noqa: E402
import jax.numpy as jnp  # noqa: E402

import fit_cache  # noqa: E402
from boundary_potential import BoundaryPotential  # noqa: E402
from desc_input import DESCInput  # noqa: E402
from flux_interp import TensorCurrent, TensorGeometry, TensorGrid  # noqa: E402
from force import ForceResidual  # noqa: E402
from fourier import fft_J  # noqa: E402
from grid import CartesianGrid  # noqa: E402
from magnetic_field import interpolate  # noqa: E402
from objective import Fields  # noqa: E402
from pressure import Pressure  # noqa: E402
from separable import Separable  # noqa: E402

jax.config.update("jax_enable_x64", True)
EQ = os.path.join(ROOT, "desc_input", "precise_QA_free_boundary_smooth_J",
                  "precise_QA_free_boundary_smooth_J_edgeJ_r0.3.h5")
COILS = os.path.join(ROOT, "desc_input", "precise_QA_free_boundary", "precise_QA_helical_coils.h5")


class _Nodes:
    def __init__(self, nodes):
        self.nodes = nodes


def free():
    jax.clear_caches()
    gc.collect()


def pressure_check(pre0, pre1):
    rho = np.linspace(0, 1, 81)
    th = np.linspace(0, 2 * np.pi, 128, endpoint=False)
    ze = np.linspace(0, 2 * np.pi / pre0.eq.NFP, 64, endpoint=False)
    nodes = np.stack(np.meshgrid(rho, th, ze, indexing="ij"), -1).reshape(-1, 3)
    p0, p1 = pre0.evaluate(_Nodes(nodes))[:, 0], pre1.evaluate(_Nodes(nodes))[:, 0]
    neg = p1 < 0
    print(f"1. pressure on {len(nodes):,} points (rho in [0, 1]): min p / max p = "
          f"{p1.min() / p1.max():.2e} (start {p0.min() / p0.max():.2e}); p < 0 at {neg.sum()} "
          f"points" + (f", rho >= {nodes[neg, 0].min():.3f}, min {p1[neg].min() / p1.max():.1e} "
                       f"of max p" if neg.any() else "")
          + f"; |p - p0| max {np.abs(p1 - p0).max() / p0.max():.2e} of max p0")


def current_check(cur, u):
    eq = cur.eq
    x, w = np.polynomial.legendre.leggauss(48)
    rho, w = (x + 1) / 2, w / 2
    th = 2 * np.pi * np.arange(96) / 96
    for zeta in (0.0, np.pi / eq.NFP / 2):
        g = TensorGrid(rho, th, np.array([zeta]), eq.NFP)
        geo = TensorGeometry(cur._RZ, g)
        J = np.asarray(TensorCurrent(cur.S, cur._RZ, g, geo)(jnp.asarray(u), cur.I))
        R = np.asarray(geo.R)
        Iz = np.sum(J[..., 1] / R * np.abs(np.asarray(geo.sqrt_g)) * w[:, None, None]
                    * (2 * np.pi / 96))
        print(f"2. net toroidal current through zeta = {zeta:.3f}: {Iz:.6e} A, I = {cur.I:.6e} A "
              f"(rel. diff {abs(Iz - cur.I) / abs(cur.I):.1e})")
    g = TensorGrid(np.array([0.5, 1.0]), th, np.linspace(0, np.pi / eq.NFP, 33), eq.NFP)
    J = np.linalg.norm(np.asarray(TensorCurrent(cur.S, cur._RZ, g)(jnp.asarray(u), cur.I)), axis=-1)
    print(f"   max |J| on the LCFS / max |J| at rho = 0.5: {J[1].max() / J[0].max():.1e}")


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
    Bp = np.asarray(interpolate(np.asarray(B_box), fields.axes, X))
    Bc = inp.B_ext(X) + Bp
    c, s = np.cos(P.ravel()), np.sin(P.ravel())
    Bcyl = np.column_stack([Bc[:, 0] * c + Bc[:, 1] * s, -Bc[:, 0] * s + Bc[:, 1] * c, Bc[:, 2]])
    return Rs, phis, Zs, Bcyl.reshape(nR, nphi, nZ, 3)


def poincare(Rs, phis, Zs, B, R0, Z0, NFP, n_periods=300, steps=64):
    """Crossings of phi = 0 (mod one field period) of the lines from (R0, Z0)."""
    from scipy.ndimage import map_coordinates, spline_filter
    nR, nphi, nZ = B.shape[:3]
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


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--x", required=True)
    ap.add_argument("--frac", type=int, default=20)
    ap.add_argument("--kernel", default="truncated")
    ap.add_argument("--factors", type=float, nargs=3, default=(3, 3, 2.5))
    ap.add_argument("--periods", type=int, default=300)
    ap.add_argument("--lines", type=int, default=24)
    a = ap.parse_args()
    t0 = time.time()
    inp = DESCInput.load(EQ, field=COILS)
    eq = inp.eq
    cur = fit_cache.fit(BoundaryPotential, inp, factors=tuple(a.factors))
    pre = fit_cache.fit(Pressure, inp, factors=tuple(a.factors), edge_order=2)
    box = CartesianGrid.periodic_box(inp, inp.a / a.frac, odd=bool(eq.sym)).locate_plasma(inp)
    x1 = (np.concatenate([cur.u, pre.c]) if a.x == "start"  # (a smoke test)
          else np.load(a.x)["x"])
    n_u = cur.n_free
    u0, u1, c1 = cur.u, x1[:n_u], x1[n_u:]
    cur1 = cur.with_u(u1)
    pre1 = Pressure(pre.basis, c1, eq, pre.edge_order)
    print(f"{inp}\nx: {a.x}; box h = a/{a.frac} {box.shape}")

    pressure_check(pre, pre1)
    current_check(cur, u1)

    # 3. exact J on the box against the interpolated J (same grid, Hockney-Eastwood)
    fr = ForceResidual(inp, box)
    F_exact = fr.evaluate(cur1, pre1)["F"]
    fh = Fields(fr, cur, pre, kernel="hockney")
    F_interp = np.asarray(fh.compute(jnp.asarray(x1))["F"])
    print(f"3. <|F|>/norm on {fr.tgrid.shape} (Hockney-Eastwood): J exact on the box "
          f"{fr.metrics(F_exact)['mean']:.3e}, interpolated (objective) "
          f"{fr.metrics(F_interp)['mean']:.3e}")
    del fh
    free()

    # 4. split along b with the optimization's kernel
    fields = Fields(fr, cur, pre, kernel=a.kernel)
    out = {}
    for name, x in (("start", fields.x0), ("optimized", jnp.asarray(x1))):
        f = {k: np.asarray(v) for k, v in fields.compute(x).items()}
        bh = f["B"] / np.linalg.norm(f["B"], axis=1)[:, None]
        par = np.sum(f["F"] * bh, 1)
        out[name] = (fr.metrics(f["F"])["mean"], fr.metrics(par[:, None])["mean"],
                     fr.metrics(f["F"] - par[:, None] * bh)["mean"])
    for name, (F, p, q) in out.items():
        print(f"4. {name:9s}: <|F|> {F:.2e}, <|F_par|> = <|b.grad p|> {p:.2e}, <|F_perp|> {q:.2e} "
              f"(/ norm; DESC <|F|> {fr.F_desc_vol / fr.norm:.2e})")

    # 5. Poincare sections at phi = 0
    RZ = Separable([eq.R_basis, eq.Z_basis], [eq.R_lmn, eq.Z_lmn], eq.NFP)
    rho0 = np.linspace(0.04, 1.0, a.lines)
    st = np.asarray(RZ.evaluate(jnp.asarray(rho0), jnp.zeros(a.lines), jnp.zeros(a.lines)))
    sections = {}
    for name, u in (("start (fits)", u0), ("optimized", u1)):
        t1 = time.time()
        Rs, phis, Zs, Bcyl = field_on_cylinder(fields, u, inp)
        sections[name] = poincare(Rs, phis, Zs, Bcyl, st[:, 0], st[:, 1], eq.NFP, a.periods)
        lost = np.isnan(sections[name][0][-1]).sum()
        print(f"5. Poincare, {name}: {a.lines} lines x {a.periods} field periods, {lost} left the "
              f"grid; {time.time() - t1:.0f} s", flush=True)

    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    th = np.linspace(0, 2 * np.pi, 400)
    surf_rho = np.linspace(0.1, 1.0, 10)
    fig, axs = plt.subplots(1, 3, figsize=(16, 6.2), constrained_layout=True)
    for ax, (name, (PR, PZ)) in zip(axs[:2], sections.items()):
        for r in surf_rho:
            s = np.asarray(RZ.evaluate(jnp.full(400, r), jnp.asarray(th), jnp.zeros(400)))
            ax.plot(s[:, 0], s[:, 1], color="0.6", lw=0.6)
        ax.scatter(PR[1:].ravel(), PZ[1:].ravel(), s=0.6, c="C0", lw=0)
        ax.set_title(f"field lines, phi = 0: {name}\n(grey: DESC flux surfaces)")
        ax.set_aspect("equal")
        ax.set_xlabel("R [m]")
    axs[0].set_ylabel("Z [m]")
    rr, tt = np.meshgrid(np.linspace(0, 1, 60), th[::4], indexing="ij")
    nodes = np.column_stack([rr.ravel(), tt.ravel(), np.zeros(rr.size)])
    s = np.asarray(RZ.evaluate(*(jnp.asarray(nodes[:, i]) for i in range(3))))
    p0 = pre.evaluate(_Nodes(nodes))[:, 0]
    dp = (pre1.evaluate(_Nodes(nodes))[:, 0] - p0) / p0.max()
    lim = np.abs(dp).max()
    tc = axs[2].tricontourf(s[:, 0], s[:, 1], dp, levels=31, cmap="RdBu_r", vmin=-lim, vmax=lim)
    PR, PZ = sections["optimized"]
    axs[2].scatter(PR[1:].ravel(), PZ[1:].ravel(), s=0.3, c="k", lw=0, alpha=0.3)
    axs[2].set_aspect("equal")
    axs[2].set_title("(p - p_DESC) / max p_DESC at phi = 0\n(dots: field lines, optimized)")
    axs[2].set_xlabel("R [m]")
    fig.colorbar(tc, ax=axs[2], shrink=0.8)
    os.makedirs(os.path.join(ROOT, "figures"), exist_ok=True)
    name = os.path.splitext(os.path.basename(a.x))[0] if a.x != "start" else "start"
    path = os.path.join(ROOT, "figures", f"{name}_poincare.png")
    fig.savefig(path, dpi=150)
    print(f"saved {path}; total {time.time() - t0:.0f} s")


if __name__ == "__main__":
    main()
