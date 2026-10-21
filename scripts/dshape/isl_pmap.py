"""Does p follow the 2/1 island?  Maps over DESC's (rho, theta*) at phi = 0 of the eps scan
(isl_scan.py) <name>, with the lines trapped in the island (isl_islands.py's sections) on top:

  delta p = p(x_eps) - p(x_0)          the pressure's change from the axisymmetric state: a
                                       flattened island holds p at its middle value, so a dipole
                                       inside the separatrix (< 0 on its inner half, > 0 outer);
                                       a displacement of the surfaces gives a broad cos(2 theta* + zeta)
  |B . grad p| / (|B| |grad p|)        how far p is from constant along the field lines (0 in an
                                       exact equilibrium: B . F = -B . grad p)

with B = B_coils + eps B_pert + B_plasma(x_eps) (the annulus solve on N_phi = 2N + 1 planes).
Figure figures/dshape/islands/isl_<name>_pmap.png; the maps in results/isl/<name>_pmap.npz.

run (GPU, after isl_islands.py): sbatch run_sai.sbatch scripts/dshape/isl_pmap.py --name a120T6wq
"""

import argparse
import json
import os
import sys
import time

import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
FIG = os.path.join(ROOT, "figures", "dshape", "islands")
os.makedirs(FIG, exist_ok=True)
sys.path.insert(0, os.path.join(ROOT, "nest"))

from desc import set_device  # noqa: E402

set_device(os.environ.get("NEST_DEVICE", "cpu"))

import jax  # noqa: E402
import jax.numpy as jnp  # noqa: E402
import matplotlib  # noqa: E402

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

from boxes import make_box  # noqa: E402
from cylindrical_grid import CylindricalInterpolator  # noqa: E402
from flux_interp import TensorGeometry, TensorGrid  # noqa: E402
from force import ForceResidual  # noqa: E402
from harmonic_dense import to_cyl  # noqa: E402
from isl_islands import cart  # noqa: E402
from isl_scan import setup  # noqa: E402
from objective import Fields  # noqa: E402
from perturbation import Perturbation  # noqa: E402
from pressure import Pressure  # noqa: E402

jax.config.update("jax_enable_x64", True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--name", required=True)
    ap.add_argument("--eps", type=float, nargs="*", default=None, help="a subset of the scan's")
    ap.add_argument("--band", type=float, default=0.09, help="half width in rho around q = 2")
    ap.add_argument("--n", type=int, nargs=2, default=(121, 256), help="(rho, theta*) points")
    ap.add_argument("--dp", type=float, default=10.0, help="isobar spacing (Pa) of the Poincare overlay")
    a = ap.parse_args()
    t0 = time.time()
    summ = json.load(open(os.path.join(ROOT, "results", "isl", a.name + ".json")))
    sa = summ["args"]
    eps_list = a.eps if a.eps else [r["eps"] for r in summ["runs"] if r["eps"] > 0]

    class A:  # isl_scan's arguments
        pass
    for k, v in sa.items():
        setattr(A, k, v)
    inp, cur, pre, x_start, fr0, run = setup(A)
    eq = inp.eq
    pert = Perturbation(inp.field, sa["m"], sa["n"])
    N = sa["N"]
    xs = {e: np.asarray(np.load(os.path.join(ROOT, "results", "isl", f"{a.name}_eps{e:g}.npz"))["x"]) for e in eps_list}
    # the annulus and B_plasma as isl_islands.py
    b = cur.basis
    N_phi = 2 * N + 1
    box = make_box(inp, run["frac"], "cylindrical", True, run.get("margin", 2.0), verbose=True, N_phi=N_phi)
    frs = ForceResidual(inp, box, resolution=(2 * b.L, 2 * b.M, 2 * max(b.N, 1)), rho_slice=slice(0, 1))
    fg = TensorGrid.uniform(4 * b.L, 2 * (3 * b.M), 4 * N_phi, eq.NFP, sym=True)
    fields = Fields(frs, cur, pre, kernel="cylindrical", dtype=jnp.float64, flux_grid=fg, factored=True,
                    S_dtype=jnp.float64)
    if run.get("j_nodes") == "direct":
        from node_current import NodeFields
        fields = NodeFields.from_fields(fields, cur, box)
    modes = jax.jit(lambda u, fl: fl.solver.field_modes(fl.J_box(u)))
    n_u = cur.n_free

    def B_plasma(x, rpz):
        Bp, Bz = modes(jnp.asarray(x[:n_u]), fields)
        return np.concatenate([np.asarray(CylindricalInterpolator(box, rpz[i : i + 200000])(Bp, Bz))
                               for i in range(0, len(rpz), 200000)])

    # the (rho, theta*) mesh at phi = 0: theta from theta* with DESC's lambda
    rho = np.argmin(np.abs(inp.iota(np.linspace(0, 1, 4001))[0] + 0.5)) / 4000
    rr = np.linspace(rho - a.band, rho + a.band, a.n[0])
    tt = 2 * np.pi * np.arange(a.n[1]) / a.n[1]
    RR, TT = np.meshgrid(rr, tt, indexing="ij")
    q = np.column_stack([RR.ravel(), TT.ravel(), np.zeros(RR.size)])
    for _ in range(5):
        q[:, 1] = TT.ravel() - np.asarray(inp.compute_points(["lambda"], q)["lambda"]).ravel()
    d = inp.compute_points(["R", "Z"], q)
    rpz = np.column_stack([np.asarray(d["R"]).ravel(), np.zeros(len(q)), np.asarray(d["Z"]).ravel()])
    X = cart(rpz)
    Bc = to_cyl(np.concatenate([np.asarray(inp.B_ext(X[i : i + 100000])) for i in range(0, len(X), 100000)]), rpz[:, 1])
    Bpt = to_cyl(np.concatenate([np.asarray(pert(X[i : i + 100000])) for i in range(0, len(X), 100000)]), rpz[:, 1])

    def p_and_grad(x):
        pev = jax.jit(Pressure(pre.basis, x[n_u:], eq, pre.edge_order).point_evaluator(eq))
        return np.asarray(pev(*(jnp.asarray(q[:, j]) for j in range(3))))  # (n, 4): p, grad p (R, phi, Z)

    pg0 = p_and_grad(x_start)
    B0 = Bc + B_plasma(x_start, rpz)
    meas0 = np.abs(np.sum(B0 * pg0[:, 1:], 1)) / (np.linalg.norm(B0, axis=1) * np.linalg.norm(pg0[:, 1:], axis=1))
    print(f"{a.name}: q = 2 at rho {rho:.4f}; mesh {RR.shape}; axisymmetric state |B.grad p|/(|B||grad p|) "
          f"median {np.median(meas0):.1e}; {time.time() - t0:.0f} s", flush=True)
    sec = np.load(os.path.join(ROOT, "results", "isl", a.name + "_sections.npz"))
    out = {"rho": rr, "theta_star": tt, "meas0": meas0.reshape(RR.shape)}

    # the complex resonant b_21 = B . grad rho / B . grad zeta ~ Re(b exp(i chi)), chi = 2 theta* + zeta, on
    # q = 2, and the island coordinate of the single-harmonic model (Boozer p. 1078; Landreman's note):
    # along a line d rho / d zeta = Re(b e^{i chi}), d chi / d zeta = m iota' x (x = rho - rho_s), so
    # x^2 = 2 Re(-i b e^{i chi}) / (m iota') + const, i.e.
    #   s = x^2 m |iota'| / (4 |b|) - sin^2 alpha,  alpha = (chi + phi') / 2 + pi / 4,
    # phi' = arg b (+ pi when m iota' < 0); separatrix s = 0, O-point s = -1 at chi + phi' = pi / 2
    m_ = 2
    iota_s, iota_r = (float(v[0]) for v in inp.iota([rho]))
    n_t, n_z = 128, 16
    th_ = 2 * np.pi * np.arange(n_t) / n_t
    ze_ = 2 * np.pi * np.arange(n_z) / n_z
    g = TensorGrid(np.array([rho]), th_, ze_, eq.NFP)
    geo = TensorGeometry(cur._RZ, g)
    nodes = g.nodes()
    rpz_s = np.column_stack([np.asarray(geo.R).ravel(), nodes[:, 2], np.asarray(geo.Z).ravel()])
    grad_rho = np.asarray(geo.gradient(jnp.ones(g.shape), jnp.zeros(g.shape), jnp.zeros(g.shape))).reshape(-1, 3)
    d_ = inp.compute_tensor(["lambda", "lambda_t"], [rho], th_, ze_)
    chi_s, jac_s = m_ * (nodes[:, 1] + d_["lambda"]) + nodes[:, 2], 1 + d_["lambda_t"]
    Xs = cart(rpz_s)
    Bc_s = to_cyl(np.asarray(inp.B_ext(Xs)), rpz_s[:, 1])
    Bpt_s = to_cyl(np.asarray(pert(Xs)), rpz_s[:, 1])

    def b21(B):
        br = np.sum(B * grad_rho, 1) / (B[:, 1] / rpz_s[:, 0])
        return 2 * np.mean(br * np.exp(-1j * chi_s) * jac_s)

    Bp0_s = B_plasma(x_start, rpz_s)
    chi = (m_ * TT).ravel()  # at phi = 0

    def s_coord(b):
        ph = np.angle(b) + (np.pi if m_ * iota_r < 0 else 0.0)
        al = (chi + ph) / 2 + np.pi / 4
        s = (RR.ravel() - rho) ** 2 * m_ * abs(iota_r) / (4 * abs(b)) - np.sin(al) ** 2
        th_O = ((np.pi / 2 - ph) / m_) % (2 * np.pi / m_)
        return s.reshape(RR.shape), th_O
    ne = len(eps_list)
    fig, ax = plt.subplots(2, ne, figsize=(6 * ne, 9), squeeze=False)
    for i, e in enumerate(eps_list):
        pg = p_and_grad(xs[e])
        B = Bc + e * Bpt + B_plasma(xs[e], rpz)
        dp = (pg[:, 0] - pg0[:, 0]).reshape(RR.shape)
        meas = (np.abs(np.sum(B * pg[:, 1:], 1)) / (np.linalg.norm(B, axis=1) * np.linalg.norm(pg[:, 1:], axis=1))
                ).reshape(RR.shape)
        # the measure on the lines trapped in the island, against the rest of the band
        key = f"NEST_{e:g}_"
        tr = sec[key + "trapped"] if key + "trapped" in sec else np.zeros(0, bool)
        rs_, ts_ = (sec[key + "rho"][:, tr], sec[key + "ths"][:, tr]) if tr.any() else (None, None)
        if tr.any():
            inside = np.zeros(RR.shape, bool)
            lo, hi = np.nanmin(rs_), np.nanmax(rs_)
            inside[(rr >= lo) & (rr <= hi), :] = True
            txt = (f"band of the island rho {lo:.3f}-{hi:.3f}: median {np.median(meas[inside]):.1e}; "
                   f"outside {np.median(meas[~inside]):.1e}")
        else:
            txt = "no trapped lines"
        print(f"eps {e:g} A: delta p {dp.min():+.1f} .. {dp.max():+.1f} Pa; |B.grad p|/(|B||grad p|) median "
              f"{np.median(meas):.1e}, max {meas.max():.1e}; {txt}", flush=True)
        out[f"dp_{e:g}"], out[f"meas_{e:g}"] = dp, meas
        out[f"p_{e:g}"] = pg[:, 0].reshape(RR.shape)
        # the resonant field's phase, the O-point it predicts, against the traced one
        bv, bn = b21(Bc_s + e * Bpt_s + Bp0_s), b21(Bc_s + e * Bpt_s + B_plasma(xs[e], rpz_s))
        s_n, thO_n = s_coord(bn)
        _, thO_v = s_coord(bv)
        thO_tr = (float(np.angle(np.nanmean(np.exp(1j * m_ * ts_))) / m_) % np.pi) if rs_ is not None else np.nan
        out[f"s_{e:g}"], out[f"b21_{e:g}"] = s_n, np.array([bv, bn])
        # p against s: inside the separatrix (s < 0) p = p(s) in an equilibrium; the spread of p at fixed s
        # (rms over bins of s, relative to the drop of p0 across the island)
        pin = out[f"p_{e:g}"]
        inside_s = s_n < 0
        bins = np.linspace(-1, 0, 11)
        k = np.digitize(s_n[inside_s], bins)
        spread = np.sqrt(np.mean([np.var(pin[inside_s][k == j]) for j in np.unique(k) if (k == j).sum() > 3]))
        w_s = 4 * np.sqrt(abs(bn) / (m_ * abs(iota_r)))
        drop = abs(np.polyfit(rr, pg0[:, 0].reshape(RR.shape).mean(1), 1)[0]) * w_s
        out[f"spread_{e:g}"] = spread / drop
        print(f"eps {e:g} A: b_21 vacuum {abs(bv):.2e} at {np.degrees(np.angle(bv)):+.0f} deg, NEST {abs(bn):.2e} at "
              f"{np.degrees(np.angle(bn)):+.0f} deg (iota' {iota_r:+.3f}); O-point theta* predicted: vacuum "
              f"{thO_v:.2f}, NEST {thO_n:.2f}; traced (NEST) {thO_tr:.2f} (mod pi); inside s < 0: "
              f"|B.grad p|/(|B||grad p|) median {np.median(meas[inside_s]) if inside_s.any() else np.nan:.1e}; "
              f"rms spread of p at fixed s {spread:.1f} Pa = {spread / drop:.2f} of the drop of p0 across the "
              f"island ({drop:.0f} Pa)", flush=True)
        v = np.abs(dp).max()
        im = ax[0, i].pcolormesh(tt, rr, dp, cmap="RdBu_r", vmin=-v, vmax=v, shading="auto")
        fig.colorbar(im, ax=ax[0, i], label=r"$\delta p$ (Pa)")
        im = ax[1, i].pcolormesh(tt, rr, np.log10(meas), cmap="magma", vmin=-8, vmax=-2, shading="auto")
        fig.colorbar(im, ax=ax[1, i], label=r"log$_{10}$ |B$\cdot\nabla$p| / (|B||$\nabla$p|)")
        for row in (0, 1):
            if rs_ is not None:
                ax[row, i].plot(ts_.ravel(), rs_.ravel(), ".", ms=0.4, color="k" if row == 0 else "c", alpha=0.6)
            ax[row, i].axhline(rho, color="gray", lw=0.5)
            ax[row, i].set_xlim(0, 2 * np.pi)
            ax[row, i].set_ylim(rr[0], rr[-1])
            ax[row, i].set_xlabel(r"$\theta^*$")
            ax[row, i].set_ylabel(r"$\rho$")
        ax[0, i].set_title(f"eps {e:g} A: " + r"$\delta p = p - p_{axisym}$" + "; dots: lines in the island", fontsize=9)
        ax[1, i].set_title(f"eps {e:g} A: " + txt, fontsize=8)
    fig.suptitle(f"{a.name} (N = {N}): does the pressure follow the 2/1 island? (phi = 0, DESC's rho and theta*; "
                 f"axisymmetric state: median {np.median(meas0):.0e})")
    fig.tight_layout()
    fig.savefig(os.path.join(FIG, f"isl_{a.name}_pmap.png"), dpi=120)
    np.savez(os.path.join(ROOT, "results", "isl", a.name + "_pmap.npz"), **out)

    # NEST's Poincare section with the isobars of p on top (levels every a.dp Pa)
    fig, ax = plt.subplots(1, ne, figsize=(7 * ne, 6.5), squeeze=False)
    for i, e in enumerate(eps_list):
        x_ = ax[0, i]
        key = f"NEST_{e:g}_"
        r, ts, tr = sec[key + "rho"], sec[key + "ths"], sec[key + "trapped"]
        for j in range(r.shape[1]):
            x_.plot(ts[:, j], r[:, j], ".", ms=0.7, color="C3" if tr[j] else "0.55", zorder=1)
        p = out[f"p_{e:g}"]
        lev = np.arange(np.floor(p.min() / a.dp) * a.dp, p.max() + a.dp, a.dp)
        cs = x_.contour(tt, rr, p, levels=lev, cmap="viridis", linewidths=1.0, zorder=2)
        x_.clabel(cs, cs.levels[::4], fontsize=6, fmt="%.0f")
        sn = out[f"s_{e:g}"]
        x_.contour(tt, rr, sn, levels=[-0.75, -0.5, -0.25], colors="k", linestyles=":", linewidths=0.8, zorder=3)
        x_.contour(tt, rr, sn, levels=[0.0], colors="k", linewidths=1.6, zorder=3)
        x_.axhline(rho, color="gray", lw=0.5)
        x_.set_xlim(0, 2 * np.pi)
        x_.set_ylim(rr[0], rr[-1])
        x_.set_xlabel(r"$\theta^*$")
        x_.set_ylabel(r"$\rho$")
        x_.set_title(f"eps {e:g} A: NEST field lines (red: in the 2/1 island), isobars every {a.dp:g} Pa; "
                     "black: model separatrix (s = 0), dotted s < 0", fontsize=8)
    fig.suptitle(f"{a.name} (N = {N}): Poincare section and pressure contours at phi = 0 (DESC's rho, theta*); "
                 "p constant on the field lines would put the isobars along the island's curves")
    fig.tight_layout()
    fig.savefig(os.path.join(FIG, f"isl_{a.name}_poincare_p.png"), dpi=130)

    # p against the island coordinate s: one curve inside the separatrix in an equilibrium
    fig, ax = plt.subplots(1, ne, figsize=(6 * ne, 5), squeeze=False)
    for i, e in enumerate(eps_list):
        x_ = ax[0, i]
        sn, p = out[f"s_{e:g}"].ravel(), out[f"p_{e:g}"].ravel()
        xx = (RR.ravel() - rho)
        sel = sn < 1.5
        for side, c, lab in ((xx < 0, "C0", r"$\rho < \rho_{q=2}$"), (xx >= 0, "C3", r"$\rho > \rho_{q=2}$")):
            x_.plot(sn[sel & side], p[sel & side], ".", ms=1.5, color=c, label=lab)
        x_.axvline(0, color="k", lw=1)
        x_.set_xlabel("s (island coordinate: -1 O-point, 0 separatrix)")
        x_.set_ylabel("p (Pa)")
        x_.set_title(f"eps {e:g} A: spread of p at fixed s inside: {out[f'spread_{e:g}']:.2f} of the drop across "
                     "the island", fontsize=9)
        x_.legend(fontsize=8, markerscale=6)
    fig.suptitle(f"{a.name} (N = {N}): p against the single-harmonic island coordinate s (phi = 0). With B.grad p = 0 "
                 "the two branches merge into one curve for s < 0 (p constant on each island surface)")
    fig.tight_layout()
    fig.savefig(os.path.join(FIG, f"isl_{a.name}_p_vs_s.png"), dpi=120)
    np.savez(os.path.join(ROOT, "results", "isl", a.name + "_pmap.npz"), **out)
    print(f"saved figures/dshape/islands/isl_{a.name}_pmap.png, isl_{a.name}_poincare_p.png, isl_{a.name}_p_vs_s.png; "
          f"total {time.time() - t0:.0f} s")


if __name__ == "__main__":
    main()
