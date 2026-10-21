"""Islands of the eps scan (isl_scan.py): resonant fields, Poincare sections and
island widths, NEST (with the plasma's response) against the vacuum superposition.

For each amplitude eps of the scan <name>:
  NEST    B = B_coils + eps B_pert + B_plasma(x_eps)   (the 3-D equilibrium)
  vacuum  B = B_coils + eps B_pert + B_plasma(x_0)     (the axisymmetric NEST state, no response)
with B_plasma from the annulus solve on N_phi = 2N + 1 planes (exact for |n| <= N).

  1. b^rho_mn = B . grad rho / B . grad zeta on DESC's rational surfaces (q = 2, 3, 4;
     the resonant (m, -1) harmonic in DESC's straight-field-line angle) and the island
     width w = 4 sqrt(|b_mn| / (m |iota'|)) of each;
  2. Poincare sections at phi = 0: field lines traced with phi as time (RK4) through
     B(R, phi, Z) = sum_k C_k(R, Z) cos k phi + S_k(R, Z) sin k phi, the harmonic tables
     (bicubic splines on an (R, Z) mesh) from 2K + 1 planes; crossings mapped to DESC's
     (rho, theta*). The 2/1 island's width: the rho extent of the lines trapped in it
     (their 2 theta* at the crossings leaves a gap > 1 rad);
  3. p through the O- and X-points of the 2/1 island (flattening), NEST.

Figures: figures/dshape/islands/isl_<name>_poincare.png, isl_<name>_width.png; results/isl/<name>_islands.json.

run (GPU): sbatch --mem=128G run.sbatch scripts/isl_islands.py --name n1
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
from scipy.interpolate import RectBivariateSpline  # noqa: E402

from boxes import make_box  # noqa: E402
from cylindrical_grid import CylindricalInterpolator  # noqa: E402
from flux_interp import TensorGeometry, TensorGrid  # noqa: E402
from force import ForceResidual  # noqa: E402
from grid import map_to_flux  # noqa: E402
from harmonic_dense import to_cyl  # noqa: E402
from isl_scan import setup  # noqa: E402
from objective import Fields  # noqa: E402
from perturbation import Perturbation  # noqa: E402
from pressure import Pressure  # noqa: E402

jax.config.update("jax_enable_x64", True)


def cart(rpz):
    return np.column_stack([rpz[:, 0] * np.cos(rpz[:, 1]), rpz[:, 0] * np.sin(rpz[:, 1]), rpz[:, 2]])


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--name", required=True)
    ap.add_argument("--lines", type=int, default=48, help="per start angle, in the band around q = 2")
    ap.add_argument("--band", type=float, default=0.16, help="half width in rho around q = 2")
    ap.add_argument("--turns", type=int, default=400)
    ap.add_argument("--steps", type=int, default=128)
    ap.add_argument("--table", type=int, nargs=2, default=(360, 520))
    ap.add_argument("--eps", type=float, nargs="*", default=None, help="a subset of the scan's")
    a = ap.parse_args()
    t0 = time.time()
    summ = json.load(open(os.path.join(ROOT, "results", "isl", a.name + ".json")))
    sa = summ["args"]
    eps_list = a.eps if a.eps else [r["eps"] for r in summ["runs"]]

    class A:  # isl_scan's arguments
        pass
    for k, v in sa.items():
        setattr(A, k, v)
    inp, cur, pre, x_start, fr0, run = setup(A)
    eq = inp.eq
    pert = Perturbation(inp.field, sa["m"], sa["n"])
    N = sa["N"]
    K = max(N, 1)
    xs = {e: np.asarray(np.load(os.path.join(ROOT, "results", "isl", f"{a.name}_eps{e:g}.npz"))["x"]) for e in eps_list}
    x0 = xs[0.0] if 0.0 in xs else x_start
    # the annulus with the run's spacing and N_phi = 2N + 1, its field at any point
    b = cur.basis
    N_phi = 2 * N + 1
    box = make_box(inp, run["frac"], "cylindrical", True, run.get("margin", 2.0), verbose=True, N_phi=N_phi)
    frs = ForceResidual(inp, box, resolution=(2 * b.L, 2 * b.M, 2 * max(b.N, 1)), rho_slice=slice(0, 1))
    fg = TensorGrid.uniform(4 * b.L, 2 * (3 * b.M), 4 * N_phi, eq.NFP, sym=True)
    fields = Fields(frs, cur, pre, kernel="cylindrical", dtype=jnp.float64, flux_grid=fg, factored=True,
                    S_dtype=jnp.float64)
    if run.get("j_nodes") == "direct":  # J at the nodes from the basis, as the run
        from node_current import NodeFields
        fields = NodeFields.from_fields(fields, cur, box)
    modes = jax.jit(lambda u, fl: fl.solver.field_modes(fl.J_box(u)))
    n_u = cur.n_free

    def B_plasma(x, rpz):  # cylindrical components at (R, phi, Z)
        Bp, Bz = modes(jnp.asarray(x[:n_u]), fields)
        return np.concatenate([np.asarray(CylindricalInterpolator(box, rpz[i : i + 200000])(Bp, Bz))
                               for i in range(0, len(rpz), 200000)])

    def B_ext(rpz):  # coils and the unit perturbation, cylindrical
        X = cart(rpz)
        bc = np.concatenate([np.asarray(inp.B_ext(X[i : i + 100000])) for i in range(0, len(X), 100000)])
        bp = np.concatenate([np.asarray(pert(X[i : i + 100000])) for i in range(0, len(X), 100000)])
        return to_cyl(bc, rpz[:, 1]), to_cyl(bp, rpz[:, 1])

    print(f"{inp}\n{pert}; scan {a.name}: N = {N}, eps {eps_list}; annulus {box.shape}; {time.time() - t0:.0f} s",
          flush=True)
    # 1. resonant fields on DESC's rational surfaces
    rho = np.linspace(0, 1, 4001)
    iota, iota_r = inp.iota(rho)
    surf = {}
    for m in (2, 3, 4):
        i = np.argmin(np.abs(iota - (-1.0 / m)))
        surf[m] = (rho[i], iota_r[i])
    n_t, n_z = 128, 16
    th = 2 * np.pi * np.arange(n_t) / n_t
    ze = 2 * np.pi * np.arange(n_z) / n_z
    res = {"vacuum": {}, "NEST": {}}
    for m, (rs, irs) in surf.items():
        g = TensorGrid(np.array([rs]), th, ze, eq.NFP)
        geo = TensorGeometry(cur._RZ, g)
        nodes = g.nodes()
        rpz = np.column_stack([np.asarray(geo.R).ravel(), nodes[:, 2], np.asarray(geo.Z).ravel()])
        grad_rho = np.asarray(geo.gradient(jnp.ones(g.shape), jnp.zeros(g.shape), jnp.zeros(g.shape))).reshape(-1, 3)
        d = inp.compute_tensor(["lambda", "lambda_t"], [rs], th, ze)
        ths, jac = nodes[:, 1] + d["lambda"], 1 + d["lambda_t"]
        Bc, Bpt = B_ext(rpz)
        Bp0 = B_plasma(x0, rpz)

        def bmn(B):
            br = np.sum(B * grad_rho, 1) / (B[:, 1] / rpz[:, 0])
            return 2 * abs(np.mean(br * np.exp(-1j * (m * ths + nodes[:, 2])) * jac))

        for e in eps_list:
            res["vacuum"].setdefault(m, []).append(bmn(Bc + e * Bpt + Bp0))
            res["NEST"].setdefault(m, []).append(bmn(Bc + e * Bpt + B_plasma(xs[e], rpz)))
        print(f"q = {m}: rho {rs:.3f}, iota' {irs:+.3f}; |b_{m},-1| vacuum "
              + " ".join(f"{v:.2e}" for v in res["vacuum"][m]) + "; NEST "
              + " ".join(f"{v:.2e}" for v in res["NEST"][m]), flush=True)
    w_formula = {k: {m: [4 * np.sqrt(v / (m * abs(surf[m][1]))) for v in res[k][m]] for m in surf} for k in res}

    # 2. Poincare sections: harmonic tables on an (R, Z) mesh
    tt = np.linspace(0, 2 * np.pi, 721)
    lc = np.asarray(cur._RZ.evaluate(jnp.ones(len(tt)), jnp.asarray(tt), jnp.zeros(len(tt))))
    pad = 0.6 * box.margin * box.h
    Rg = np.linspace(lc[:, 0].min() - pad, lc[:, 0].max() + pad, a.table[0])
    Zg = np.linspace(lc[:, 1].min() - pad, lc[:, 1].max() + pad, a.table[1])
    RR, ZZ = np.meshgrid(Rg, Zg, indexing="ij")
    n_s = 2 * K + 1
    phis = 2 * np.pi * np.arange(n_s) / n_s
    rpz_all = np.concatenate([np.column_stack([RR.ravel(), np.full(RR.size, p), ZZ.ravel()]) for p in phis])
    t1 = time.time()
    Bc_all, Bpt_all = B_ext(rpz_all)
    print(f"tables: coils and perturbation on {n_s} planes x {RR.shape}, {time.time() - t1:.0f} s", flush=True)
    E = np.concatenate([np.ones((n_s, 1))] + [np.stack([np.cos(k * phis), np.sin(k * phis)], 1) for k in range(1, K + 1)],
                       1)  # (planes, 2K + 1): synthesis
    Einv = np.linalg.inv(E)
    Bp0_all = B_plasma(x0, rpz_all)

    def splines(Bplanes):
        """(planes x mesh, 3) -> splines[harmonic][component]."""
        V = Bplanes.reshape(n_s, RR.size, 3)
        H = np.einsum("hp,pic->hic", Einv, V)
        return [[RectBivariateSpline(Rg, Zg, H[h, :, c].reshape(RR.shape), kx=3, ky=3) for c in range(3)]
                for h in range(n_s)]

    rs2, _ = surf[2]
    rho0 = np.linspace(rs2 - a.band, rs2 + a.band, a.lines)
    st = []
    for th0 in (0.0, np.pi / 2):
        X0 = np.asarray(cur._RZ.evaluate(jnp.asarray(rho0), jnp.full(a.lines, th0), jnp.zeros(a.lines)))
        st.append(X0)
    st = np.concatenate(st)
    n_lines = len(st)

    def trace(sp):
        R, Z = st[:, 0].copy(), st[:, 1].copy()
        dphi = 2 * np.pi / a.steps
        cross = np.empty((a.turns + 1, n_lines, 2))
        cross[0] = st

        def rhs(R, Z, phi):
            w = [1.0] + sum(([np.cos(k * phi), np.sin(k * phi)] for k in range(1, K + 1)), [])
            BR = sum(wh * s[0].ev(R, Z) for wh, s in zip(w, sp))
            Bf = sum(wh * s[1].ev(R, Z) for wh, s in zip(w, sp))
            BZ = sum(wh * s[2].ev(R, Z) for wh, s in zip(w, sp))
            return R * BR / Bf, R * BZ / Bf

        phi = 0.0
        for k in range(a.turns):
            for _ in range(a.steps):
                k1 = rhs(R, Z, phi)
                k2 = rhs(R + 0.5 * dphi * k1[0], Z + 0.5 * dphi * k1[1], phi + 0.5 * dphi)
                k3 = rhs(R + 0.5 * dphi * k2[0], Z + 0.5 * dphi * k2[1], phi + 0.5 * dphi)
                k4 = rhs(R + dphi * k3[0], Z + dphi * k3[1], phi + dphi)
                R = R + dphi / 6 * (k1[0] + 2 * k2[0] + 2 * k3[0] + k4[0])
                Z = Z + dphi / 6 * (k1[1] + 2 * k2[1] + 2 * k3[1] + k4[1])
                phi += dphi
            phi = 0.0
            cross[k + 1] = np.column_stack([R, Z])
        pts = cross.reshape(-1, 2)
        ok = np.all(np.isfinite(pts), 1) & (pts[:, 0] > Rg[0]) & (pts[:, 0] < Rg[-1]) & (pts[:, 1] > Zg[0]) & (
            pts[:, 1] < Zg[-1])
        flux = np.full((len(pts), 3), np.nan)
        flux[ok] = np.asarray(map_to_flux(inp, np.column_stack([pts[ok, 0], np.zeros(ok.sum()), pts[ok, 1]])))
        inside = np.isfinite(flux[:, 0])
        lam = np.full(len(pts), np.nan)
        lam[inside] = np.asarray(inp.compute_points(["lambda"], flux[inside])["lambda"]).ravel()
        r = flux[:, 0].reshape(a.turns + 1, n_lines)
        ts = np.mod(flux[:, 1] + lam, 2 * np.pi).reshape(a.turns + 1, n_lines)
        return r, ts

    def width(r, ts, m=2):
        """rho extent of the lines trapped in the m/1 island, and which lines: along a line
        the helical angle m theta* (mod 2 pi) at the crossings stays put on the resonance;
        unwrapped turn by turn it drifts without bound on a circulating line and stays
        within less than 2 pi on a trapped one."""
        trapped = np.zeros(r.shape[1], bool)
        for j in range(r.shape[1]):
            al = m * ts[:, j][np.isfinite(ts[:, j])]
            if len(al) < 20:
                continue
            un = np.concatenate([[0.0], np.cumsum(np.mod(np.diff(al) + np.pi, 2 * np.pi) - np.pi)])
            trapped[j] = un.max() - un.min() < 2 * np.pi
        if not trapped.any():
            return 0.0, trapped
        rt = r[:, trapped]
        return float(np.nanmax(rt) - np.nanmin(rt)), trapped

    sections, w_poinc = {"vacuum": {}, "NEST": {}}, {"vacuum": [], "NEST": []}
    for e in eps_list:
        for kind, xx in (("vacuum", x0), ("NEST", xs[e])):
            if kind == "vacuum" and e == 0:
                sections[kind][e] = sections.get("NEST", {}).get(e)
            t1 = time.time()
            Bpl = Bp0_all if kind == "vacuum" else B_plasma(xx, rpz_all)
            r, ts = trace(splines(Bc_all + e * Bpt_all + Bpl))
            w, tr = width(r, ts)
            sections[kind][e] = (r, ts, tr)
            w_poinc[kind].append(w)
            print(f"eps {e:g} {kind}: 2/1 island width (Poincare) {w:.4f}, formula {w_formula[kind][2][eps_list.index(e)]:.4f}; "
                  f"{tr.sum()} trapped lines; {time.time() - t1:.0f} s", flush=True)

    # 3. p through the O- and X-points (NEST): the theta* of the trapped lines' mean
    prof = {}
    rr = np.linspace(rs2 - a.band, rs2 + a.band, 161)
    for e in eps_list:
        r, ts, tr = sections["NEST"][e]
        if not tr.any():
            continue
        al = 2 * ts[:, tr]
        thO = float(np.angle(np.nanmean(np.exp(1j * al))) / 2) % np.pi  # the O-point's theta* (mod pi)
        pr = Pressure(pre.basis, xs[e][n_u:], eq, pre.edge_order)
        pev = jax.jit(pr.point_evaluator(eq))
        out = {}
        for lab, ths_ in (("O", thO), ("X", thO + np.pi / 2)):
            # theta from theta* along the line rho (DESC's lambda, at phi = 0)
            q = np.column_stack([rr, np.full(len(rr), ths_), np.zeros(len(rr))])
            for _ in range(4):
                lam = np.asarray(inp.compute_points(["lambda"], q)["lambda"]).ravel()
                q[:, 1] = ths_ - lam
            out[lab] = np.asarray(pev(jnp.asarray(q[:, 0]), jnp.asarray(q[:, 1]), jnp.asarray(q[:, 2])))[:, 0].tolist()
        prof[e] = out

    js = {"surfaces": {m: list(v) for m, v in surf.items()}, "eps": eps_list,
          "b_mn": {k: {m: list(map(float, v)) for m, v in d.items()} for k, d in res.items()},
          "w_formula": {k: {m: list(map(float, v)) for m, v in d.items()} for k, d in w_formula.items()},
          "w_poincare": w_poinc, "p_profiles": {str(e): v for e, v in prof.items()}, "rho_profiles": rr.tolist()}
    with open(os.path.join(ROOT, "results", "isl", a.name + "_islands.json"), "w") as fo:
        json.dump(js, fo, indent=1)
    np.savez(os.path.join(ROOT, "results", "isl", a.name + "_sections.npz"),
             **{f"{k}_{e:g}_{q}": v for k, d in sections.items() for e, s in d.items() if s is not None
                for q, v in zip(("rho", "ths", "trapped"), s)})

    # figures
    ne = len(eps_list)
    fig, ax = plt.subplots(2, ne, figsize=(4.2 * ne, 8.4), squeeze=False)
    for i, e in enumerate(eps_list):
        for row, kind in enumerate(("vacuum", "NEST")):
            s = sections[kind][e]
            if s is None:
                continue
            r, ts, tr = s
            for j in range(r.shape[1]):
                ax[row, i].plot(ts[:, j], r[:, j], ".", ms=0.6, color="C3" if tr[j] else "k")
            ax[row, i].axhline(rs2, color="C0", lw=0.5)
            ax[row, i].set_ylim(rs2 - a.band, rs2 + a.band)
            ax[row, i].set_xlim(0, 2 * np.pi)
            ax[row, i].set_title(f"{kind}, eps {e:g} A: w = {w_poinc[kind][i]:.3f}", fontsize=9)
            ax[row, i].set_xlabel(r"$\theta^*$")
        ax[0, i].set_ylabel(r"$\rho$")
        ax[1, i].set_ylabel(r"$\rho$")
    fig.suptitle(f"Poincare sections at phi = 0 in DESC's (theta*, rho): top, the perturbation added to the axisymmetric "
                 f"NEST state (no response); bottom, NEST's 3-D equilibrium (N = {N}); red: lines trapped in the 2/1 island")
    fig.tight_layout()
    fig.savefig(os.path.join(FIG, f"isl_{a.name}_poincare.png"), dpi=130)

    fig, ax = plt.subplots(1, 3, figsize=(19, 5.5))
    ep = np.array(eps_list, float)
    pos = ep > 0
    for kind, c in (("vacuum", "k"), ("NEST", "C3")):
        ax[0].loglog(ep[pos], np.array(w_poinc[kind])[pos], "o-", color=c, label=f"{kind}: Poincare")
        ax[0].loglog(ep[pos], np.array(w_formula[kind][2])[pos], "s--", color=c, mfc="none",
                     label=f"{kind}: 4 sqrt(|b_21| / (2 iota'))")
        for m, ls in zip((2, 3, 4), ("-", "--", ":")):
            ax[1].loglog(ep[pos], np.array(res[kind][m])[pos], ls, marker="o", color=c, ms=3,
                         label=f"{kind}: |b_{m},-1| at q = {m}")
    if pos.any():
        e1 = ep[pos]
        ax[0].loglog(e1, w_formula["vacuum"][2][list(ep).index(e1[0])] * np.sqrt(e1 / e1[0]), "k:", lw=0.7,
                     label=r"$\propto \sqrt{\epsilon}$")
    ax[0].set_xlabel(r"$\epsilon$ [A] (amplitude of $\Phi = \epsilon \sin(2\theta + \zeta)$)")
    ax[0].set_ylabel(r"2/1 island width in $\rho$")
    ax[0].legend(fontsize=8)
    ax[0].set_title("2/1 island width")
    ax[1].set_xlabel(r"$\epsilon$ [A]")
    ax[1].set_title(r"resonant $|b^\rho_{m,-1}|$ on DESC's q = m surfaces")
    ax[1].legend(fontsize=7)
    for e in prof:
        ax[2].plot(rr, prof[e]["O"], "-", label=f"eps {e:g}: O-point")
        ax[2].plot(rr, prof[e]["X"], "--", color=ax[2].lines[-1].get_color(), label="X-point")
    if not prof:
        ax[2].text(0.5, 0.5, "no 2/1 island in NEST", transform=ax[2].transAxes, ha="center")
    ax[2].axvline(rs2, color="gray", lw=0.5)
    ax[2].set_xlabel(r"$\rho$ (DESC)")
    ax[2].set_ylabel("p [Pa]")
    ax[2].set_title("NEST pressure through the 2/1 O- and X-points (phi = 0)")
    if prof:
        ax[2].legend(fontsize=7)
    fig.suptitle(f"DSHAPE tokamak, 2/1 error field on the coils' winding surface: NEST (a/{run['frac']}, x"
                 f"{run['factors'][0]:g}, |n| <= {N}) against the vacuum superposition")
    fig.tight_layout()
    fig.savefig(os.path.join(FIG, f"isl_{a.name}_width.png"), dpi=130)
    print(f"saved figures/dshape/islands/isl_{a.name}_poincare.png, isl_{a.name}_width.png; total {time.time() - t0:.0f} s")


if __name__ == "__main__":
    main()
