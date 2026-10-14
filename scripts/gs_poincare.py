"""Poincare section of a NEST run with --bext desc, against DESC (axisymmetric tokamak).

The run's field at the collocation points is B_ext + B_plasma(x) with B_ext =
B_DESC - B_plasma(x0), i.e. B = B_DESC + B_plasma(u - u0); B_plasma is linear in u.
That field is traced everywhere, in DESC's flux coordinates (zeta = phi as time):

    d rho / d zeta = B^rho / B^zeta,   d theta / d zeta = B^theta / B^zeta,

with B^i = B_DESC^i + dB . grad(x^i) (B_DESC^rho = 0) and dB the box field of
u - u0 (net current part excluded) at x(rho, theta, zeta). The components are
tabulated on a (rho, theta, zeta) grid and interpolated (cubic); RK4 with a fixed
number of steps per toroidal turn, crossings of phi = 0 recorded. The box field is
not exactly axisymmetric (Cartesian box), so the trace is 3D. DESC's own field
(dB = 0) is traced the same way, for iota and as a check of the tracer.

figures/gs_<run>_poincare.png:
  a. the section at phi = 0 in (R, Z): NEST's crossings over DESC's surfaces;
  b. the radial excursion of each NEST line, max |rho - rho_0| over its crossings;
  c. rho - rho_0 against theta for a few lines (the shape of the displaced surfaces);
  d. iota of each line, NEST and DESC, and their difference.

usage: python scripts/gs_poincare.py --run results/gs/fdrift64_L24_f2 [more copies] [--turns 300]
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
from scipy.interpolate import RegularGridInterpolator  # noqa: E402

import fit_cache  # noqa: E402
from boundary_potential import BoundaryPotential  # noqa: E402
from desc_input import DESCInput  # noqa: E402
from force import ForceResidual  # noqa: E402
from fourier import fft_J  # noqa: E402
from grid import CartesianGrid  # noqa: E402
from magnetic_field import interpolate  # noqa: E402
from objective import Fields  # noqa: E402
from pressure import Pressure  # noqa: E402

jax.config.update("jax_enable_x64", True)


def periodic(a, axis, pad=3):
    """a extended by pad points on each side along a periodic axis."""
    return np.concatenate([np.take(a, range(-pad, 0), axis), a, np.take(a, range(pad), axis)], axis)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", required=True, nargs="+",
                    help="optimize_qa.py --out of a --bext desc run (the first of several "
                         "that has finished: copies of one run)")
    ap.add_argument("--turns", type=int, default=300)
    ap.add_argument("--steps", type=int, default=128, help="RK4 steps per toroidal turn")
    ap.add_argument("--lines", type=int, default=24)
    ap.add_argument("--grid", type=int, nargs=3, default=(72, 128, 64), help="rho, theta, zeta")
    a = ap.parse_args()
    t0 = time.time()
    done = [r for r in a.run if os.path.exists(r + ".json")]
    a.run = done[0] if done else a.run[0]
    info = json.load(open(a.run + ".json")) if os.path.exists(a.run + ".json") else \
        json.load(open(a.run + "_iter.json"))
    args = info["args"]
    st = np.load(a.run + ".npz") if os.path.exists(a.run + ".npz") else np.load(a.run + "_iter.npz")
    x0, x = np.asarray(st["x0"]), np.asarray(st["x"])
    inp = DESCInput.load(args["eq"], field=args["coils"])
    eq = inp.eq
    cur = fit_cache.fit(BoundaryPotential, inp, factors=tuple(args["factors"]))
    pre = fit_cache.fit(Pressure, inp, factors=tuple(args["factors"]), edge_order=2)
    b = cur.basis
    res = tuple(int(round(args["colloc"] * (u + v))) for u, v in ((b.L, eq.L), (b.M, eq.M), (b.N, eq.N)))
    box = CartesianGrid.periodic_box(inp, inp.a / args["frac"], odd=bool(eq.sym)).locate_plasma(inp)
    fields = Fields(ForceResidual(inp, box, resolution=res), cur, pre, kernel=args["kernel"],
                    dtype=getattr(jnp, args.get("dtype", "float32")))
    n_u = fields.n_u
    du = jnp.asarray(x[:n_u] - x0[:n_u])
    B_box = np.asarray(fields._solve(fft_J(fields.J_box(du, 0.0)), fields.axes, fields.K_hat))
    print(f"{inp}\nrun {a.run}: box a/{args['frac']} {box.shape}, |u - u0| / |u0| "
          f"{np.linalg.norm(du) / np.linalg.norm(x0[:n_u]):.2e}; setup {time.time() - t0:.0f} s",
          flush=True)

    # components on a (rho, theta, zeta) grid
    nr, nt, nz = a.grid
    rho = np.unique(np.concatenate([np.linspace(0.02, 0.9, nr - 24),
                                    1 - np.logspace(-1, np.log10(2e-3), 24)]))
    theta = np.linspace(0, 2 * np.pi, nt, endpoint=False)
    zeta = np.linspace(0, 2 * np.pi, nz, endpoint=False)
    d = inp.compute_tensor(["R", "Z", "B^theta", "B^zeta", "e^rho", "e^theta", "e^zeta"],
                           rho, theta, zeta)
    R, Z = d["R"], d["Z"]
    ph = np.broadcast_to(zeta[None, None, :], (len(rho), nt, nz)).ravel()
    X = np.column_stack([R * np.cos(ph), R * np.sin(ph), Z])
    dBc = np.asarray(interpolate(B_box, fields.axes, X))  # Cartesian
    c, s = np.cos(ph), np.sin(ph)
    dB = np.column_stack([dBc[:, 0] * c + dBc[:, 1] * s, -dBc[:, 0] * s + dBc[:, 1] * c, dBc[:, 2]])
    dBr, dBt, dBz = (np.sum(dB * d[k], 1) for k in ("e^rho", "e^theta", "e^zeta"))
    shape = (len(rho), nt, nz)
    Bz_D, Bt_D = d["B^zeta"], d["B^theta"]
    fields_tab = {"DESC": (np.zeros(len(R)), Bt_D / Bz_D),
                  "NEST": (dBr / (Bz_D + dBz), (Bt_D + dBt) / (Bz_D + dBz))}
    print(f"max |dB| / |B_DESC| on the grid: "
          f"{np.max(np.linalg.norm(dB, axis=1)) / np.median(np.abs(Bz_D * R)):.1e}; "
          f"max |B^rho / B^zeta| {np.abs(fields_tab['NEST'][0]).max():.1e}", flush=True)
    pad = 3
    th_e = np.concatenate([theta[-pad:] - 2 * np.pi, theta, theta[:pad] + 2 * np.pi])
    ze_e = np.concatenate([zeta[-pad:] - 2 * np.pi, zeta, zeta[:pad] + 2 * np.pi])

    def interpolators(fr, ft):
        out = []
        for f in (fr, ft):
            F = periodic(periodic(f.reshape(shape), 1, pad), 2, pad)
            out.append(RegularGridInterpolator((rho, th_e, ze_e), F, method="cubic",
                                               bounds_error=False, fill_value=None))
        return out

    rho0 = np.concatenate([np.linspace(0.06, 0.95, a.lines - 3), [0.97, 0.985, 0.995]])
    traces = {}
    for name, (fr, ft) in fields_tab.items():
        Ir, It = interpolators(fr, ft)

        def rhs(r, t, z):
            P = np.column_stack([np.clip(r, rho[0], rho[-1]), np.mod(t, 2 * np.pi),
                                 np.full_like(r, np.mod(z, 2 * np.pi))])
            return Ir(P), It(P)

        r, t = rho0.copy(), np.zeros_like(rho0)
        h = 2 * np.pi / a.steps
        Rr, Tt = [r.copy()], [t.copy()]
        t1 = time.time()
        for k in range(a.turns):
            for j in range(a.steps):
                z = j * h
                k1 = rhs(r, t, z)
                k2 = rhs(r + h / 2 * k1[0], t + h / 2 * k1[1], z + h / 2)
                k3 = rhs(r + h / 2 * k2[0], t + h / 2 * k2[1], z + h / 2)
                k4 = rhs(r + h * k3[0], t + h * k3[1], z + h)
                r = r + h / 6 * (k1[0] + 2 * k2[0] + 2 * k3[0] + k4[0])
                t = t + h / 6 * (k1[1] + 2 * k2[1] + 2 * k3[1] + k4[1])
            Rr.append(r.copy())
            Tt.append(t.copy())
        traces[name] = (np.array(Rr), np.array(Tt))  # (turns + 1, lines); theta unwrapped
        print(f"traced {name}: {a.lines} lines x {a.turns} turns, {time.time() - t1:.0f} s", flush=True)

    # (R, Z) of the crossings at phi = 0, from DESC's map
    def to_RZ(r, t):
        from desc.grid import Grid
        nodes = np.column_stack([r.ravel(), np.mod(t.ravel(), 2 * np.pi), np.zeros(r.size)])
        dd = inp.compute_points(["R", "Z"], nodes)
        return dd["R"].reshape(r.shape), dd["Z"].reshape(r.shape)

    rN, tN = traces["NEST"]
    rD, tD = traces["DESC"]
    RN, ZN = to_RZ(rN, tN)
    iota = {k: (v[1][-1] - v[1][0]) / (2 * np.pi * a.turns) for k, v in traces.items()}
    exc = np.max(np.abs(rN - rN[0]), axis=0)
    fig, ax = plt.subplots(2, 2, figsize=(13, 11))
    ax = ax.ravel()
    ths = np.linspace(0, 2 * np.pi, 300)
    for r0 in rho0:
        RD_, ZD_ = to_RZ(np.full(300, r0), ths)
        ax[0].plot(RD_, ZD_, "C0-", lw=0.6)
    ax[0].plot(RN.ravel(), ZN.ravel(), "k.", ms=0.6)
    ax[0].plot([], [], "C0-", label="DESC surfaces (seeds)")
    ax[0].plot([], [], "k.", label=f"NEST field lines ({a.turns} turns)")
    ax[0].set_aspect("equal")
    ax[0].legend(fontsize=8, loc="upper right")
    ax[0].set_xlabel("R [m]")
    ax[0].set_ylabel("Z [m]")
    ax[0].set_title(r"a. Poincare section, $\phi$ = 0")
    ax[1].semilogy(rho0, np.maximum(exc, 1e-18), "ko-", label="NEST")
    ax[1].semilogy(rho0, np.maximum(np.max(np.abs(rD - rD[0]), axis=0), 1e-18), "C0s--",
                   label="DESC (traced, = 0 up to the tracer)")
    ax[1].set_xlabel(r"$\rho_0$ (seed on $\theta$ = 0)")
    ax[1].set_title(r"b. radial excursion max $|\rho - \rho_0|$ of each line")
    ax[1].legend(fontsize=8)
    for i in np.linspace(0, len(rho0) - 1, 6).astype(int):
        o = np.argsort(np.mod(tN[:, i], 2 * np.pi))
        ax[2].plot(np.mod(tN[o, i], 2 * np.pi), (rN[o, i] - rho0[i]), ".", ms=2,
                   label=rf"$\rho_0$ = {rho0[i]:.3f}")
    ax[2].set_xlabel(r"$\theta$")
    ax[2].set_ylabel(r"$\rho - \rho_0$")
    ax[2].set_title("c. NEST surfaces against DESC's (crossings of each line)")
    ax[2].legend(fontsize=7, ncol=2)
    rr = np.linspace(0.02, 1, 200)
    from desc.grid import LinearGrid
    g = LinearGrid(rho=rr, M=eq.M_grid, N=0, NFP=1)
    iota_D = g.compress(eq.compute("iota", grid=g)["iota"])
    ax[3].plot(rr, iota_D, "C0-", label=r"DESC $\iota(\rho)$")
    ax[3].plot(rho0, iota["DESC"], "C0s", ms=4, mfc="none", label="DESC field, traced")
    ax[3].plot(rho0, iota["NEST"], "k.", label="NEST field, traced")
    ax[3].set_xlabel(r"$\rho_0$")
    ax[3].set_ylabel(r"$\iota$")
    a3 = ax[3].twinx()
    a3.semilogy(rho0, np.abs(iota["NEST"] - iota["DESC"]), "C3^-", ms=4, label=r"$|\iota_{NEST}-\iota_{DESC}|$")
    a3.set_ylabel(r"$|\Delta\iota|$", color="C3")
    ax[3].legend(fontsize=8, loc="upper right")
    ax[3].set_title(r"d. rotational transform per line")
    fig.suptitle(f"{os.path.basename(a.run)}: field lines of B_DESC + B_plasma(u - u0) "
                 f"(box a/{args['frac']}) against DESC")
    fig.tight_layout()
    out = os.path.join(ROOT, "figures", f"gs_{os.path.basename(a.run)}_poincare.png")
    fig.savefig(out, dpi=140)
    np.savez(out[:-4] + ".npz", rho0=rho0, rN=rN, tN=tN, rD=rD, tD=tD, iota_N=iota["NEST"],
             iota_D=iota["DESC"])
    print(f"radial excursion: max {exc.max():.2e}, median {np.median(exc):.2e}; |iota_NEST - "
          f"iota_DESC| max {np.abs(iota['NEST'] - iota['DESC']).max():.2e}; traced DESC iota vs "
          f"DESC's profile: max {np.abs(iota['DESC'] - np.interp(rho0, rr, iota_D)).max():.1e}")
    print(f"saved {out}; total {time.time() - t0:.0f} s")


if __name__ == "__main__":
    main()
