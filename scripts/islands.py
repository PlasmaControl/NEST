"""Magnetic islands at the rational surfaces of a NEST solution.

  1. the resonant radial field: on the surface iota = n/m of DESC's flux
     coordinates, b^rho = B . grad rho / B . grad zeta (the radial excursion of a
     field line per radian of zeta) of B = B_ext + B_plasma, Fourier analysed in
     the straight-field-line angle theta* = theta + lambda; its (m, n) amplitude
     gives the full island width w = 4 sqrt(|b_mn| / (m |d iota / d rho|))
     (DESC's own field, tangent to its surfaces, as a check of the machinery)
  2. Poincare section at phi = 0 of field lines started in a band of rho around
     the resonance, mapped back to DESC's flux coordinates and plotted as theta*
     (x) against rho (y): nested surfaces are horizontal lines, an m-island chain
     m O-shaped islands
for the start (the fits) and an optimized x (``optimize_qa.py`` npz).
Figure: figures/<name>_islands.png.

run (GPU): sbatch --mem=128G run.sbatch scripts/islands.py --x results/freeP20.npz --frac 20
"""

import argparse
import os
import sys
import time
from fractions import Fraction

import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, "nest"))
sys.path.insert(0, os.path.join(ROOT, "scripts"))

from desc import set_device  # noqa: E402

set_device(os.environ.get("NEST_DEVICE", "cpu"))

import jax  # noqa: E402
import jax.numpy as jnp  # noqa: E402

import fit_cache  # noqa: E402
from boundary_potential import BoundaryPotential  # noqa: E402
from desc_input import DESCInput  # noqa: E402
from flux_interp import TensorGeometry, TensorGrid  # noqa: E402
from force import ForceResidual  # noqa: E402
from fourier import fft_J  # noqa: E402
from grid import CartesianGrid  # noqa: E402
from magnetic_field import interpolate  # noqa: E402
from objective import Fields  # noqa: E402
from pressure import Pressure  # noqa: E402
from separable import Separable  # noqa: E402
from validate_opt import field_on_cylinder, poincare  # noqa: E402

jax.config.update("jax_enable_x64", True)
EQ = os.path.join(ROOT, "desc_input", "precise_QA_free_boundary_smooth_J",
                  "precise_QA_free_boundary_smooth_J_edgeJ_r0.3.h5")
COILS = os.path.join(ROOT, "desc_input", "precise_QA_free_boundary", "precise_QA_helical_coils.h5")


def rational_surfaces(inp, m_max=30):
    """{iota value: (lowest resonant m, n, rho)} for n a multiple of NFP."""
    NFP = inp.eq.NFP
    rho = np.linspace(0, 1, 4001)
    iota, iota_r = inp.iota(rho)
    out = {}
    for m in range(1, m_max + 1):
        for n in range(NFP, m + 1, NFP):
            q = Fraction(n, m)
            if iota.min() <= float(q) <= iota.max() and q not in out:
                i = np.argmin(np.abs(iota - float(q)))
                out[q] = (m, n, rho[i], iota_r[i])
    return out


def B_total(fields, u, inp, X):
    """B_ext + B_plasma (Cartesian) at the Cartesian points X."""
    B_box = fields._solve(fft_J(fields.J_box(jnp.asarray(u))), fields.axes, fields.K_hat)
    return inp.B_ext(X) + np.asarray(interpolate(np.asarray(B_box), fields.axes, X))


def resonant_spectrum(inp, rho0, B_of_X, n_t=128, n_z=64, harmonics=()):
    """|b^rho_mn| (cos/sin amplitude) in (theta*, zeta) on the surface rho0."""
    eq = inp.eq
    th = 2 * np.pi * np.arange(n_t) / n_t
    ze = 2 * np.pi / eq.NFP * np.arange(n_z) / n_z
    g = TensorGrid(np.array([rho0]), th, ze, eq.NFP)
    RZ = Separable([eq.R_basis, eq.Z_basis], [eq.R_lmn, eq.Z_lmn], eq.NFP)
    geo = TensorGeometry(RZ, g)
    R, Z = np.asarray(geo.R).ravel(), np.asarray(geo.Z).ravel()
    phi = g.nodes()[:, 2]
    X = np.column_stack([R * np.cos(phi), R * np.sin(phi), Z])
    B = B_of_X(X)
    c, s = np.cos(phi), np.sin(phi)
    Bcyl = np.column_stack([B[:, 0] * c + B[:, 1] * s, -B[:, 0] * s + B[:, 1] * c, B[:, 2]])
    one, zero = jnp.ones(g.shape), jnp.zeros(g.shape)
    grad_rho = np.asarray(geo.gradient(one, zero, zero)).reshape(-1, 3)
    b_rho = np.sum(Bcyl * grad_rho, 1) / (Bcyl[:, 1] / R)  # B . grad rho / B . grad zeta
    lam = inp.compute_tensor(["lambda", "lambda_t"], [rho0], th, ze)
    ths = g.nodes()[:, 1] + lam["lambda"]  # theta*
    jac = 1 + lam["lambda_t"]  # d theta* / d theta
    zeta = g.nodes()[:, 2]
    out = {}
    for m, n in harmonics:
        # c_mn = < b e^{-i (m theta* - n zeta)} >, averaging over theta* and zeta
        cmn = np.mean(b_rho * np.exp(-1j * (m * ths - n * zeta)) * jac)
        out[(m, n)] = 2 * abs(cmn)  # amplitude of the real cos/sin component
    return out, float(np.sqrt(np.mean(b_rho**2)))


def to_flux(inp, PR, PZ):
    """(rho, theta*) of the Poincare points (phi = 0), for points inside the LCFS."""
    pts = np.column_stack([PR.ravel(), np.zeros(PR.size), PZ.ravel()])
    ok = np.isfinite(pts).all(1)
    rtz = np.full_like(pts, np.nan)
    rtz[ok] = np.asarray(inp.eq.map_coordinates(pts[ok], inbasis=("R", "phi", "Z"),
                                                outbasis=("rho", "theta", "zeta")))
    inside = ok & (rtz[:, 0] <= 1.0) & np.isfinite(rtz[:, 0])
    lam = np.full(len(pts), np.nan)
    if inside.any():
        lam[inside] = inp.compute_points(["lambda"], rtz[inside])["lambda"]
    ths = np.mod(rtz[:, 1] + lam, 2 * np.pi)
    return rtz[:, 0].reshape(PR.shape), ths.reshape(PR.shape)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--x", required=True)
    ap.add_argument("--frac", type=int, default=20)
    ap.add_argument("--kernel", default="truncated")
    ap.add_argument("--factors", type=float, nargs=3, default=(3, 3, 2.5))
    ap.add_argument("--band", type=float, nargs=2, default=(0.42, 0.68))
    ap.add_argument("--lines", type=int, default=60)
    ap.add_argument("--periods", type=int, default=1000)
    ap.add_argument("--grid", type=int, nargs=3, default=(201, 96, 201),
                    help="cylindrical grid (R, phi, Z) of the field for the tracing")
    a = ap.parse_args()
    t0 = time.time()
    inp = DESCInput.load(EQ, field=COILS)
    eq = inp.eq
    cur = fit_cache.fit(BoundaryPotential, inp, factors=tuple(a.factors))
    pre = fit_cache.fit(Pressure, inp, factors=tuple(a.factors), edge_order=2)
    box = CartesianGrid.periodic_box(inp, inp.a / a.frac, odd=bool(eq.sym)).locate_plasma(inp)
    fields = Fields(ForceResidual(inp, box), cur, pre, kernel=a.kernel)
    x1 = (np.concatenate([cur.u, pre.c]) if a.x == "start"  # (a smoke test)
          else np.load(a.x)["x"])
    us = {"start (fits)": cur.u, "optimized": x1[: cur.n_free]}
    print(f"{inp}\nx: {a.x}; box h = a/{a.frac} {box.shape}")

    # 1. resonant radial field and island widths
    surfaces = rational_surfaces(inp)
    print("rational surfaces (lowest resonant harmonic):")
    for q, (m, n, r, ir) in sorted(surfaces.items(), key=lambda t: t[1][0])[:6]:
        print(f"   iota = {q}: (m, n) = ({m}, {n}), rho {r:.3f}, d iota/d rho {ir:.4f}")
    low = sorted(surfaces.items(), key=lambda t: t[1][0])[:4]

    def desc_B(X):  # DESC's own field (check: tangent to its surfaces, b^rho ~ 0)
        rtz = np.asarray(eq.map_coordinates(
            np.column_stack([np.hypot(X[:, 0], X[:, 1]), np.arctan2(X[:, 1], X[:, 0]), X[:, 2]]),
            inbasis=("R", "phi", "Z"), outbasis=("rho", "theta", "zeta")))
        Bc = inp.compute_points(["B"], rtz)["B"]
        ph = np.arctan2(X[:, 1], X[:, 0])
        return np.column_stack([Bc[:, 0] * np.cos(ph) - Bc[:, 1] * np.sin(ph),
                                Bc[:, 0] * np.sin(ph) + Bc[:, 1] * np.cos(ph), Bc[:, 2]])

    sources = {"DESC (check)": desc_B}
    for name, u in us.items():
        sources[name] = (lambda X, u=u: B_total(fields, u, inp, X))
    print("1. resonant radial field b^rho_mn (rho per radian of zeta) and island full width "
          "w = 4 sqrt(|b_mn| / (m |iota'|)) [in rho]:")
    widths = {}
    for q, (m, n, r, ir) in low:
        for name, B_of_X in sources.items():
            amp, rms = resonant_spectrum(inp, r, B_of_X, harmonics=[(m, n)])
            w = 4 * np.sqrt(amp[(m, n)] / (m * abs(ir)))
            widths[(name, q)] = w
            print(f"   iota = {str(q):5s} rho {r:.3f} {name:13s}: |b_{m},{n}| {amp[(m, n)]:.2e} "
                  f"(rms b^rho {rms:.2e}), width {w:.4f}", flush=True)

    # 2. Poincare sections in (theta*, rho)
    RZ = Separable([eq.R_basis, eq.Z_basis], [eq.R_lmn, eq.Z_lmn], eq.NFP)
    rho0 = np.linspace(*a.band, a.lines)
    th0 = np.where(np.arange(a.lines) % 2, np.pi / 4, 0.0)  # two start angles: the O and X points
    st = np.asarray(RZ.evaluate(jnp.asarray(rho0), jnp.asarray(th0), jnp.zeros(a.lines)))
    sections, iotas = {}, {}
    for name, u in us.items():
        t1 = time.time()
        nR, nphi, nZ = a.grid
        Rs, phis, Zs, Bcyl = field_on_cylinder(fields, u, inp, nR=nR, nphi=nphi, nZ=nZ)
        PR, PZ = poincare(Rs, phis, Zs, Bcyl, st[:, 0], st[:, 1], eq.NFP, a.periods)
        rho, ths = to_flux(inp, PR[1:], PZ[1:])
        sections[name] = (rho, ths)
        # the field's own rotational transform, line by line: theta* advances by
        # iota 2 pi / NFP per field period (less than pi here, so unwrapping is safe)
        d = np.diff(ths, axis=0)
        d = np.mod(d + np.pi, 2 * np.pi) - np.pi
        iotas[name] = (np.nanmean(rho, 0), np.nanmean(d, 0) / (2 * np.pi / eq.NFP))
        print(f"2. Poincare, {name}: {a.lines} lines x {a.periods} field periods; "
              f"{time.time() - t1:.0f} s", flush=True)

    rho_d = np.linspace(*a.band, 200)
    iota_d = inp.iota(rho_d)[0]
    for name, (r, io) in iotas.items():
        ok = np.isfinite(r) & np.isfinite(io)
        print(f"3. measured iota, {name}: " + ", ".join(
            f"rho {rr:.3f}: {ii:.4f} (DESC {np.interp(rr, rho_d, iota_d):.4f})"
            for rr, ii in list(zip(r[ok], io[ok]))[:: max(1, ok.sum() // 8)]))
        # where the field's own iota crosses each low-order rational
        order = np.argsort(r[ok])
        rs, ios = r[ok][order], io[ok][order]
        for q, (m, n, _, _) in low:
            cross = np.flatnonzero(np.diff(np.sign(ios - float(q))))
            for c in cross:
                rq = rs[c] + (float(q) - ios[c]) * (rs[c + 1] - rs[c]) / (ios[c + 1] - ios[c])
                slope = (ios[c + 1] - ios[c]) / (rs[c + 1] - rs[c])
                amp, _ = resonant_spectrum(inp, rq, sources[name], harmonics=[(m, n)])
                w = 4 * np.sqrt(amp[(m, n)] / (m * abs(slope)))
                print(f"   {name}: iota = {q} at rho {rq:.3f} (d iota/d rho {slope:.3f}); "
                      f"|b_{m},{n}| there {amp[(m, n)]:.2e}, width {w:.4f}", flush=True)

    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fig, axs = plt.subplots(1, 3, figsize=(21, 6.5), constrained_layout=True,
                            gridspec_kw={"width_ratios": [1, 1, 0.55]})
    q, (m, n, r, ir) = low[0]
    for ax, (name, (rho, ths)) in zip(axs[:2], sections.items()):
        for j in range(rho.shape[1]):
            ax.plot(ths[:, j], rho[:, j], ".", ms=0.8, color=plt.cm.viridis(j / rho.shape[1]))
        for qq, (mm, nn, rr, _) in low:
            ax.axhline(rr, color="0.5", lw=0.7, ls="--")
            ax.text(2 * np.pi * 1.005, rr, f"iota = {qq}", va="center", fontsize=8)
        w = widths.get((name, q))
        ax.set_title(f"{name}: field lines at phi = 0 in DESC's coordinates\n"
                     f"(predicted width at iota = {q}: {w:.4f} in rho)")
        ax.set_xlabel("theta* = theta + lambda (straight-field-line angle)")
        ax.set_xlim(0, 2 * np.pi)
        ax.set_ylim(*a.band)
        ax.set_ylabel("rho (DESC)")
    axs[0].set_ylabel("rho (DESC)")
    ax = axs[2]
    ax.plot(iota_d, rho_d, "k-", lw=1.5, label="DESC")
    for (name, (r, io)), c in zip(iotas.items(), ("C1", "C0")):
        ax.plot(io, r, "o", ms=3, color=c, label=f"field lines: {name}")
    for qq, (mm, nn, rr, _) in low:
        ax.axvline(float(qq), color="0.6", lw=0.7, ls=":")
        ax.text(float(qq), a.band[1], f" {qq}", rotation=90, va="top", fontsize=8)
    ax.set_xlabel("rotational transform iota")
    ax.set_ylim(*a.band)
    ax.legend(fontsize=8)
    ax.set_title("iota: DESC's profile and the field's own\n(from the theta* advance per period)")
    os.makedirs(os.path.join(ROOT, "figures"), exist_ok=True)
    path = os.path.join(ROOT, "figures",
                        f"{os.path.splitext(os.path.basename(a.x))[0] if a.x != 'start' else 'start'}"
                        "_islands.png")
    fig.savefig(path, dpi=160)
    print(f"saved {path}; total {time.time() - t0:.0f} s")


if __name__ == "__main__":
    main()
