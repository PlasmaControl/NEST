"""Poincare section of a cylindrical-grid NEST run of the tokamak, against DESC's surfaces.

The run's field is B = B_coil + B_plasma(x): the coils' field and the annulus solve of
the run's final current (N_phi = 1, so B is exactly axisymmetric).  Field lines are then
traced in the (R, Z) plane with phi as time,

    dR/dphi = R B_R / B_phi,   dZ/dphi = R B_Z / B_phi,

with B from bicubic tables on a fine (R, Z) mesh: B_plasma by the solver's own
barycentric Chebyshev interpolation (CylindricalInterpolator) of the modal field, the
coils' field by its Biot-Savart sum.  RK4, ``--steps`` per toroidal turn; the crossings
of phi = 0 are mapped to DESC's flux coordinates (Newton, grid.map_to_flux).
Each line starts on DESC's surface rho_0 at theta = 0 (outboard midplane).

figures/<name>.png:
  a. the section at phi = 0: NEST's crossings over DESC's surfaces rho_0 (lines);
  b. rho - rho_0 against theta for each line: NEST's surfaces seen in DESC's coordinates;
  c. max |rho - rho_0| of each line against rho_0;
  d. iota of each line (the slope of DESC's straight-field-line angle theta* = theta +
     lambda over the turns), NEST against DESC's iota(rho_0).

run (GPU): sbatch run.sbatch scripts/gs_cyl_poincare.py --run results/gs/cylD_a160
"""

import argparse
import json
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
import matplotlib  # noqa: E402

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
from scipy.interpolate import RectBivariateSpline  # noqa: E402

import fit_cache  # noqa: E402
from boundary_potential import BoundaryPotential  # noqa: E402
from boxes import make_box  # noqa: E402
from cylindrical_grid import CylindricalInterpolator  # noqa: E402
from desc_input import DESCInput  # noqa: E402
from force import ForceResidual  # noqa: E402
from grid import map_to_flux  # noqa: E402
from objective import Fields  # noqa: E402
from pressure import Pressure  # noqa: E402

jax.config.update("jax_enable_x64", True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", required=True)
    ap.add_argument("--lines", type=int, default=20)
    ap.add_argument("--rho-max", type=float, default=0.995)
    ap.add_argument("--turns", type=int, default=300)
    ap.add_argument("--steps", type=int, default=256, help="RK4 steps per toroidal turn")
    ap.add_argument("--table", type=int, nargs=2, default=(640, 960), help="(R, Z) mesh of the B tables")
    ap.add_argument("--name", default=None)
    ap.add_argument("--field", default="nest", choices=["nest", "desc"],
                    help="trace NEST's B (coils + B_plasma), or DESC's own B inside the LCFS with the same "
                         "tables, tracing and mapping: the method's floor (DESC's lines lie on rho_0)")
    a = ap.parse_args()
    t0 = time.time()
    run = json.load(open(a.run + "_iter.json"))["args"]
    x = np.asarray(np.load(a.run + "_iter.npz")["x"])
    inp = DESCInput.load(run["eq"], field=run["coils"])
    eq = inp.eq
    ep = run.get("edge_power", 2)
    cur = fit_cache.fit(BoundaryPotential, inp, factors=tuple(run["factors"]),
                        **({"edge_power": ep} if ep != 2 else {}))
    pre = fit_cache.fit(Pressure, inp, factors=tuple(run["factors"]), edge_order=2)
    b = cur.basis
    res = tuple(int(round(run["colloc"] * (u + v))) for u, v in ((b.L, eq.L), (b.M, eq.M), (b.N, eq.N)))
    box = make_box(inp, run["frac"], run.get("grid", "cartesian"), bool(eq.sym), run.get("margin", 2.0))
    if run.get("grid") != "cylindrical" or box.N_phi != 1:
        raise SystemExit("needs a run on the axisymmetric cylindrical grid (N_phi = 1)")
    fr = ForceResidual(inp, box, resolution=res)
    fields = Fields(fr, cur, pre, kernel="cylindrical", dtype=jnp.float64,
                    S_dtype=getattr(jnp, run.get("S_dtype", "float32")))
    if run.get("j_nodes") == "direct":  # J at the annulus nodes from the basis, as the run
        from node_current import NodeFields
        fields = NodeFields.from_fields(fields, cur, box)
    u = jnp.asarray(x[: fields.n_u])
    Bp, Bz = fields.solver.field_modes(fields.J_box(u))
    print(f"{inp}\nrun {a.run}: annulus {box.shape}, field modes, {time.time() - t0:.0f} s", flush=True)
    # the tables on a mesh just covering the plasma
    th = np.linspace(0, 2 * np.pi, 721)
    lc = np.asarray(cur._RZ.evaluate(jnp.ones(len(th)), jnp.asarray(th), jnp.zeros(len(th))))
    pad = 0.6 * box.margin * box.h  # inside the annulus' walls (margin h beyond the plasma)
    Rg = np.linspace(lc[:, 0].min() - pad, lc[:, 0].max() + pad, a.table[0])
    Zg = np.linspace(lc[:, 1].min() - pad, lc[:, 1].max() + pad, a.table[1])
    RR, ZZ = np.meshgrid(Rg, Zg, indexing="ij")
    rpz = np.column_stack([RR.ravel(), np.zeros(RR.size), ZZ.ravel()])
    t1 = time.time()
    Bpl = np.concatenate([np.asarray(CylindricalInterpolator(box, rpz[i : i + 200000])(Bp, Bz))
                          for i in range(0, len(rpz), 200000)])  # (B_r, B_phi, B_z)
    xc = np.column_stack([RR.ravel(), np.zeros(RR.size), ZZ.ravel()])  # phi = 0: (x, y, z) = (R, 0, Z)
    Bc = np.concatenate([np.asarray(inp.B_ext(xc[i : i + 100000])) for i in range(0, len(xc), 100000)])
    B = Bpl + Bc  # at phi = 0 Cartesian (B_x, B_y, B_z) = (B_R, B_phi, B_Z)
    if a.field == "desc":  # DESC's own B inside the LCFS (the method's own floor); NEST's outside
        fl = np.concatenate([np.asarray(map_to_flux(inp, rpz[i : i + 50000])) for i in range(0, len(rpz), 50000)])
        ins = np.isfinite(fl[:, 0]) & (fl[:, 0] <= 1.0)
        BD_mesh = np.concatenate([np.asarray(inp.compute_points(["B"], fl[ins][i : i + 100000])["B"])
                                  for i in range(0, int(ins.sum()), 100000)])
        B[ins] = BD_mesh
        print(f"field: DESC's B at {int(ins.sum()):,} mesh points inside the LCFS", flush=True)
    print(f"tables {a.table[0]} x {a.table[1]}: {time.time() - t1:.0f} s", flush=True)
    sp = [RectBivariateSpline(Rg, Zg, B[:, k].reshape(RR.shape), kx=3, ky=3) for k in range(3)]
    # the tables against B_DESC inside the plasma (NEST's B differs from it by ~1e-6)
    rq = np.array([0.5, 0.9, 0.945, 0.98, 0.995, 0.999])
    tq = np.linspace(0, 2 * np.pi, 64, endpoint=False)
    Q = np.array([(r_, t_, 0.0) for r_ in rq for t_ in tq])
    XQ = np.asarray(cur._RZ.evaluate(jnp.asarray(Q[:, 0]), jnp.asarray(Q[:, 1]), jnp.asarray(Q[:, 2])))
    BD = np.asarray(inp.compute_points(["B"], Q)["B"])  # (B_R, B_phi, B_Z)
    BT = np.column_stack([s_.ev(XQ[:, 0], XQ[:, 1]) for s_ in sp])
    rel = np.linalg.norm(BT - BD, axis=1) / np.linalg.norm(BD, axis=1)
    for r_ in rq:
        m = Q[:, 0] == r_
        print(f"  table B vs B_DESC at rho {r_}: max rel {rel[m].max():.2e}, mean {rel[m].mean():.2e}", flush=True)

    def rhs(R, Z):
        BR, Bphi, BZ = (s.ev(R, Z) for s in sp)
        return R * BR / Bphi, R * BZ / Bphi

    # the lines: DESC's surfaces rho_0 at theta = 0
    rho0 = np.linspace(0.05, a.rho_max, a.lines)
    X0 = np.asarray(cur._RZ.evaluate(jnp.asarray(rho0), jnp.zeros(a.lines), jnp.zeros(a.lines)))
    R, Z = X0[:, 0].copy(), X0[:, 1].copy()
    dphi = 2 * np.pi / a.steps
    cross = np.empty((a.turns + 1, a.lines, 2))
    cross[0] = np.column_stack([R, Z])

    t1 = time.time()
    for k in range(a.turns):
        for _ in range(a.steps):
            k1 = rhs(R, Z)
            k2 = rhs(R + 0.5 * dphi * k1[0], Z + 0.5 * dphi * k1[1])
            k3 = rhs(R + 0.5 * dphi * k2[0], Z + 0.5 * dphi * k2[1])
            k4 = rhs(R + dphi * k3[0], Z + dphi * k3[1])
            R = R + dphi / 6 * (k1[0] + 2 * k2[0] + 2 * k3[0] + k4[0])
            Z = Z + dphi / 6 * (k1[1] + 2 * k2[1] + 2 * k3[1] + k4[1])
        cross[k + 1] = np.column_stack([R, Z])
    print(f"traced {a.lines} lines x {a.turns} turns: {time.time() - t1:.0f} s", flush=True)
    pts = cross.reshape(-1, 2)
    flux = np.asarray(map_to_flux(inp, np.column_stack([pts[:, 0], np.zeros(len(pts)), pts[:, 1]])))
    rho = flux[:, 0].reshape(a.turns + 1, a.lines)
    theta = flux[:, 1].reshape(a.turns + 1, a.lines)
    # iota: DESC's straight-field-line angle theta* = theta + lambda at the crossings, which
    # advances linearly along a line on DESC's surfaces; unwrapped with DESC's own iota as
    # the predictor (|iota| ~ 0.85 > 1/2 turn per turn) and fitted linearly over the turns
    from desc.grid import LinearGrid
    g = LinearGrid(rho=rho0, M=eq.M_grid, N=eq.N_grid, NFP=eq.NFP)
    iota_desc = np.asarray(g.compress(eq.compute("iota", grid=g)["iota"]))
    fl = flux.copy()
    ok = np.isfinite(fl[:, 0])
    lam = np.full(len(fl), np.nan)
    lam[ok] = np.asarray(inp.compute_points(["lambda"], fl[ok])["lambda"]).ravel()
    ths = (fl[:, 1] + lam).reshape(a.turns + 1, a.lines)
    kk = np.arange(a.turns + 1)
    iota_nest = np.full(a.lines, np.nan)
    for j in range(a.lines):
        unw = ths[:, j] + 2 * np.pi * np.round((ths[0, j] + 2 * np.pi * iota_desc[j] * kk - ths[:, j]) / (2 * np.pi))
        m = np.isfinite(unw)
        if m.sum() > 2:
            iota_nest[j] = np.polyfit(kk[m], unw[m], 1)[0] / (2 * np.pi)
    drho = rho - rho0[None]
    out = {"rho0": rho0.tolist(), "max |rho - rho0|": np.nanmax(np.abs(drho), 0).tolist(),
           "iota NEST": iota_nest.tolist(), "iota DESC": iota_desc.tolist()}
    name = a.name or f"gs_{os.path.basename(a.run)}_poincare"
    with open(os.path.join(ROOT, "results", "gs", name + ".json"), "w") as fo:
        json.dump(out, fo, indent=1)
    np.savez(os.path.join(ROOT, "results", "gs", name + ".npz"), cross=cross, rho=rho, theta=theta, rho0=rho0)
    for r0, m, i1, i0 in zip(rho0, out["max |rho - rho0|"], iota_nest, iota_desc):
        print(f"  rho0 {r0:.3f}: max |rho - rho0| {m:.2e}; iota NEST {i1:+.8f} DESC {i0:+.8f} diff {i1 - i0:+.1e}")
    # figure
    fig, ax = plt.subplots(1, 4, figsize=(24, 6))
    tt = np.linspace(0, 2 * np.pi, 361)
    for r0 in rho0:
        X = np.asarray(cur._RZ.evaluate(jnp.full(len(tt), r0), jnp.asarray(tt), jnp.zeros(len(tt))))
        ax[0].plot(X[:, 0], X[:, 1], "k-", lw=0.6)
    ax[0].plot(lc[:, 0], lc[:, 1], "k-", lw=1.4)
    cmap = plt.get_cmap("viridis")
    for j in range(a.lines):
        c = cmap(j / max(a.lines - 1, 1))
        ax[0].plot(cross[:, j, 0], cross[:, j, 1], ".", ms=1.2, color=c)
        o = np.argsort(np.mod(theta[:, j], 2 * np.pi))
        ax[1].plot(np.mod(theta[o, j], 2 * np.pi), drho[o, j], "-", lw=0.8, color=c, label=f"{rho0[j]:.2f}"
                   if j % 4 == 0 else None)
    ax[0].set_aspect("equal")
    ax[0].set_xlabel("R [m]")
    ax[0].set_ylabel("Z [m]")
    ax[0].set_title("Poincare section, phi = 0: NEST lines (dots), DESC surfaces (black)")
    ax[1].set_xlabel(r"$\theta$ (DESC)")
    ax[1].set_ylabel(r"$\rho - \rho_0$")
    ax[1].set_title(r"NEST's surfaces in DESC's coordinates: $\rho-\rho_0$ against $\theta$")
    ax[1].legend(fontsize=7, title=r"$\rho_0$")
    ax[2].semilogy(rho0, out["max |rho - rho0|"], "o-")
    ax[2].set_xlabel(r"$\rho_0$")
    ax[2].set_title(r"max $|\rho-\rho_0|$ per line")
    ax[3].plot(rho0, iota_desc, "k-", label="DESC")
    ax[3].plot(rho0, iota_nest, "C3o", ms=3, label="NEST (lines)")
    ax3 = ax[3].twinx()
    ax3.semilogy(rho0, np.abs(iota_nest - iota_desc), "C0s--", ms=3, label="|iota NEST - DESC|")
    ax[3].set_xlabel(r"$\rho_0$")
    ax[3].set_title(r"$\iota$ of each line")
    ax[3].legend(fontsize=8, loc="upper left")
    ax3.legend(fontsize=8, loc="lower right")
    fig.suptitle(f"{os.path.basename(a.run)}: coils + cylindrical B_plasma (annulus {box.shape}), "
                 f"{a.turns} turns, RK4 {a.steps}/turn, against DESC L32 free boundary")
    fig.tight_layout()
    path = os.path.join(ROOT, "figures", name + ".png")
    fig.savefig(path, dpi=140)
    print(f"saved {path}; total {time.time() - t0:.0f} s")


if __name__ == "__main__":
    main()
