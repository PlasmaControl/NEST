"""B.n on the boundary and the boundary shape of NEST runs, fixed and free boundary.

For each run (a geometry, its x), B = B_ext + B_plasma on the boundary rho = 1 of
that geometry, and B.n / |B| there. B_plasma is by direct Biot-Savart (``--direct``,
default; the box's error at the boundary is ~1e-3 |B| at a/10), or from J on a box
h = a / --frac with ``--direct 0 0``. The
figure has one map of B.n / |B| over (zeta, theta) per run and the boundaries at
phi = 0 and pi / NFP; the numbers go to the npz.

    --run LABEL=EQ.h5:X.npz     X: an optimize_qa.py or optimize_fb.py npz, or "start"
                                (the fits of EQ)

run (GPU): sbatch run.sbatch scripts/stellarator/fb_compare.py --coils NCSX_coils.h5 \
    --run start=NCSX_fb_L16M16N12.h5:start --run fixed=NCSX_fb_L16M16N12.h5:results/ncsx/fixed.npz \
    --run free=results/ncsx/fb_eq.h5:results/ncsx/fb.npz --frac 20 --out figures/ncsx_fb_Bn
"""

import argparse
import os
import sys

import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
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
from free_boundary import boundary_normal  # noqa: E402
from grid import CartesianGrid  # noqa: E402
from magnetic_field import interpolate  # noqa: E402
from objective import Fields, _cartesian  # noqa: E402
from pressure import Pressure  # noqa: E402
from separable import Separable  # noqa: E402

jax.config.update("jax_enable_x64", True)


def boundary_field(eq_file, coils, x_file, frac, factors, n_theta, n_zeta, kernel, base=None,
                   direct=(32, 2)):
    """B.n / |B| on the boundary (n_theta, n_zeta over one period) and the boundary
    (R, Z) at phi = 0 and pi / NFP. ``base``: the fits come from this equilibrium
    (the one the run started from), the geometry from ``eq_file``."""
    inp = DESCInput.load(eq_file, field=coils, check=False)
    eq = inp.eq
    fit_inp = inp if base is None else DESCInput.load(base, check=False)
    cur = fit_cache.fit(BoundaryPotential, fit_inp, factors=tuple(factors))
    pre = fit_cache.fit(Pressure, fit_inp, factors=tuple(factors), edge_order=2)
    cur = BoundaryPotential(cur.basis, cur.u, cur.I, eq, cur.edge_power, cur.S)
    pre = Pressure(pre.basis, pre.c, eq, pre.edge_order)
    u = cur.u if x_file == "start" else np.load(x_file)["x"][: cur.n_free]
    a = (fit_inp.a if base is not None else inp.a)
    box = CartesianGrid.periodic_box(inp, a / frac, odd=bool(eq.sym)).locate_plasma(inp)
    fields = Fields(ForceResidual(inp, box, rho_slice=slice(0, 1)), cur, pre, kernel=kernel)
    B_box = np.asarray(fields._solve(fft_J(fields.J_box(jnp.asarray(u))), fields.axes,
                                     fields.K_hat))
    theta = 2 * np.pi * np.arange(n_theta) / n_theta
    zeta = 2 * np.pi * np.arange(n_zeta) / (n_zeta * eq.NFP)
    g = TensorGrid([1.0], theta, zeta, eq.NFP)
    RZ = Separable([eq.R_basis, eq.Z_basis], [eq.R_lmn, eq.Z_lmn], eq.NFP)
    geo = TensorGeometry(RZ, g)
    phi = g.nodes()[:, 2]
    R, Z = np.asarray(geo.R).ravel(), np.asarray(geo.Z).ravel()
    xb = np.column_stack([R * np.cos(phi), R * np.sin(phi), Z])
    if direct is None:
        B = inp.B_ext(xb) + interpolate(B_box, fields.axes, xb)
    else:  # B_plasma by direct Biot-Savart (independent of the box)
        from free_boundary import Displacement, DirectField
        df = DirectField(cur, Displacement(eq, 1, 0), (n_theta, n_zeta), direct[0], direct[1],
                         sym=False)
        B = inp.B_ext(xb) + np.asarray(df(jnp.asarray(u), cur.I, jnp.zeros(df.mg.A.shape[0]),
                                          jnp.asarray(xb)))
    n = np.asarray(_cartesian(boundary_normal(geo).reshape(-1, 3), phi))
    dA = np.linalg.norm(n, axis=1)
    rel = np.sum(B * n, 1) / dA / np.linalg.norm(B, axis=1)
    t = np.linspace(0, 2 * np.pi, 361)
    planes = []
    for z0 in (0.0, np.pi / eq.NFP):
        X = np.asarray(RZ.evaluate(jnp.ones_like(t), jnp.asarray(t), jnp.full_like(t, z0)))
        planes.append(X)
    # displacement from the boundary of ``base`` along its in-plane normal [m]
    disp = np.zeros(len(R))
    if base is not None:
        eb = fit_inp.eq
        RZb = Separable([eb.R_basis, eb.Z_basis], [eb.R_lmn, eb.Z_lmn], eb.NFP)
        gb = TensorGeometry(RZb, g)
        Rb, Zb = np.asarray(gb.R).ravel(), np.asarray(gb.Z).ravel()
        Xt = np.asarray(gb.Xt).reshape(-1, 2)
        nR, nZ = Xt[:, 1], -Xt[:, 0]
        X0 = np.asarray(RZb.evaluate(jnp.zeros(len(R)), jnp.asarray(g.nodes()[:, 1]),
                                     jnp.asarray(phi)))
        s = np.sign(np.mean(nR * (Rb - X0[:, 0]) + nZ * (Zb - X0[:, 1])))
        disp = s * ((R - Rb) * nR + (Z - Zb) * nZ) / np.hypot(nR, nZ)
    del fields, B_box
    jax.clear_caches()
    return {"Bn": rel.reshape(n_theta, n_zeta), "dA": dA.reshape(n_theta, n_zeta),
            "disp": disp.reshape(n_theta, n_zeta),
            "rms": float(np.sqrt(np.sum(dA * rel**2) / dA.sum())),
            "max": float(np.abs(rel).max()), "planes": np.array(planes), "theta": theta,
            "zeta": zeta}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--coils", required=True)
    ap.add_argument("--run", action="append", required=True, help="LABEL=EQ.h5:X.npz")
    ap.add_argument("--base", default=None,
                    help="the equilibrium every run started from (its fits; default each EQ)")
    ap.add_argument("--frac", type=int, default=20)
    ap.add_argument("--factors", type=float, nargs=3, default=(3, 3, 2.5))
    ap.add_argument("--kernel", default="truncated")
    ap.add_argument("--direct", type=int, nargs=2, default=(32, 2), metavar=("N_RHO", "Q"),
                    help="B_plasma on the boundary by direct Biot-Savart (0 0: from the box)")
    ap.add_argument("--n-theta", type=int, default=128)
    ap.add_argument("--n-zeta", type=int, default=64)
    ap.add_argument("--out", default=os.path.join(ROOT, "figures", "fb_Bn"))
    a = ap.parse_args()
    res = {}
    for r in a.run:
        label, rest = r.split("=", 1)
        eq_file, x_file = rest.rsplit(":", 1)
        res[label] = boundary_field(eq_file, a.coils, x_file, a.frac, a.factors, a.n_theta,
                                    a.n_zeta, a.kernel, a.base,
                                    tuple(a.direct) if a.direct[0] > 0 else None)
        print(f"{label}: B.n/|B| rms {res[label]['rms']:.3e}, max {res[label]['max']:.3e}",
              flush=True)
    np.savez(a.out + ".npz", **{f"{k}_{q}": v[q] for k, v in res.items()
                                for q in ("Bn", "dA", "disp", "planes", "theta", "zeta", "rms",
                                          "max")})

    import matplotlib.pyplot as plt
    k = len(res)
    moved = [lb for lb, v in res.items() if np.abs(v["disp"]).max() > 0]
    fig, axs = plt.subplots(2 if moved else 1, k + 1, figsize=(4.2 * k + 4.5, 7.4 if moved else 3.8),
                            gridspec_kw={"width_ratios": [1] * k + [1.1]}, squeeze=False)
    ax = axs[0]
    vmax = max(np.abs(v["Bn"]).max() for v in res.values())
    for i, (label, v) in enumerate(res.items()):
        im = ax[i].pcolormesh(v["zeta"], v["theta"], v["Bn"], cmap="RdBu_r", vmin=-vmax,
                              vmax=vmax, shading="auto")
        ax[i].set_title(f"{label}: B·n/|B| rms {v['rms']:.1e}", fontsize=10)
        ax[i].set_xlabel("ζ (one period)")
        if i == 0:
            ax[i].set_ylabel("θ")
    fig.colorbar(im, ax=ax[k - 1], fraction=0.05)
    styles = ["k-", "C0--", "C3-", "C2-.", "C1:"]
    for i, (label, v) in enumerate(res.items()):
        for j, X in enumerate(v["planes"]):
            ax[k].plot(X[:, 0], X[:, 1], styles[i % len(styles)], lw=1,
                       label=label if j == 0 else None)
    ax[k].set_aspect("equal")
    ax[k].set_xlabel("R [m]")
    ax[k].set_ylabel("Z [m]")
    ax[k].set_title("boundary at φ = 0 and π/NFP", fontsize=10)
    ax[k].legend(fontsize=8)
    if moved:
        dmax = max(np.abs(res[lb]["disp"]).max() for lb in moved) * 1e3
        for i, (label, v) in enumerate(res.items()):
            if label not in moved:
                axs[1, i].axis("off")
                continue
            im = axs[1, i].pcolormesh(v["zeta"], v["theta"], v["disp"] * 1e3, cmap="PuOr_r",
                                      vmin=-dmax, vmax=dmax, shading="auto")
            axs[1, i].set_title(f"{label}: boundary moved [mm] (max {np.abs(v['disp']).max() * 1e3:.1f})",
                                fontsize=10)
            axs[1, i].set_xlabel("ζ (one period)")
            axs[1, i].set_ylabel("θ")
        fig.colorbar(im, ax=axs[1, k - 1], fraction=0.05)
        axs[1, k].axis("off")
    fig.tight_layout()
    fig.savefig(a.out + ".png", dpi=150)
    print(f"saved {a.out}.png, {a.out}.npz")


if __name__ == "__main__":
    main()
