"""Map J_parallel = J.B/|B| over (theta, zeta) on flux surfaces near the LCFS.

Panels: J_parallel at rho = 1 for the source and the smooth-J equilibrium (same
symmetric color scale), J_parallel at rho = 0.9 of the smooth-J equilibrium, and the
mean curvature of the LCFS, to see whether the residual edge current follows the
boundary shaping. One field period in zeta.

usage: python plot_jpar_boundary.py SOURCE.h5 SMOOTH.h5 OUT.png [LABEL_1 LABEL_2]
"""

import sys

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

import desc.io  # noqa: E402
from desc.grid import LinearGrid  # noqa: E402

eq_src = desc.io.load(sys.argv[1])
eq = desc.io.load(sys.argv[2])
out = sys.argv[3]
lab_src, lab = (sys.argv[4:6] if len(sys.argv) > 5
                else [r"source ($\iota$ prescribed)", "smooth J"])
nt, nz = 128, 96


def surface_map(eq, name, rho):
    zeta = np.linspace(0, 2 * np.pi / eq.NFP, nz, endpoint=False)
    theta = np.linspace(0, 2 * np.pi, nt, endpoint=False)
    g = LinearGrid(rho=np.array([rho]), theta=theta, zeta=zeta, NFP=eq.NFP)
    d = eq.compute([name, "|e_theta x e_zeta|"], grid=g)
    q = g.meshgrid_reshape(d[name], "rtz")[0]  # (theta, zeta)
    w = g.meshgrid_reshape(d["|e_theta x e_zeta|"], "rtz")[0]
    return theta, zeta, q, float(np.sum(q * w) / np.sum(w))  # area-weighted mean


panels = [
    (eq_src, "J_parallel", 1.0, lab_src + r": $J_\parallel$ at $\rho$ = 1"),
    (eq, "J_parallel", 1.0, lab + r": $J_\parallel$ at $\rho$ = 1"),
    (eq, "J_parallel", 0.9, lab + r": $J_\parallel$ at $\rho$ = 0.9"),
    (eq, "curvature_H_rho", 1.0, lab + r": mean curvature $H$ of the LCFS"),
]
maps = [surface_map(*p[:3]) for p in panels]
vmax_edge = max(np.abs(m[2]).max() for m in maps[:2])

fig, axes = plt.subplots(2, 2, figsize=(12, 9), constrained_layout=True)
for ax, (e, name, rho, title), (theta, zeta, q, mean) in zip(axes.flat, panels, maps):
    if name == "J_parallel":
        vmax = vmax_edge if rho == 1.0 else np.abs(q).max()
        cs = ax.contourf(zeta, theta, q / 1e3, levels=np.linspace(-vmax, vmax, 41) / 1e3,
                         cmap="RdBu_r")
        label = r"$J_\parallel$ [kA/m$^2$]"
        title += f"\nsurface mean {mean / 1e3:+.1f}, rms {np.sqrt(np.mean(q**2)) / 1e3:.1f} kA/m$^2$"
    else:
        cs = ax.contourf(zeta, theta, q, levels=41, cmap="viridis")
        label = r"$H$ [1/m]"
    fig.colorbar(cs, ax=ax, label=label)
    ax.set_title(title, fontsize=10)
    ax.set_xlabel(r"$\zeta$ (one field period)")
    ax.set_ylabel(r"$\theta$")
fig.savefig(out, dpi=130)
print("saved", out)
