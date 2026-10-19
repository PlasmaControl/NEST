"""DESC J vs the BoundaryPotential reconstruction (direct T fit), smooth-current precise_QA.

Writes figures/precise_QA/J_vs_desc.png: |J| on each flux surface (max and rms over a 96 x 24
(theta, zeta) grid of one field period), a zoom on the edge, the difference of
the surface max and rms from DESC, and |J| and |J - J_DESC| in the cross-section
zeta = 0, all in units of max |J_DESC|.

run (GPU): sbatch run.sbatch scripts/stellarator/plot_J_vs_desc.py
"""

import os
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.join(ROOT, "nest"))

from desc import set_device  # noqa: E402

# DESC uses the CPU unless set_device('gpu') runs before any other DESC import
set_device(os.environ.get("NEST_DEVICE", "cpu"))

import matplotlib  # noqa: E402

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
from desc.grid import Grid  # noqa: E402

from boundary_potential import BoundaryPotential  # noqa: E402
from desc_input import DESCInput  # noqa: E402

path = os.path.join(ROOT, "desc_input", "precise_QA_free_boundary_smooth_J",
                    "precise_QA_free_boundary_smooth_J_edgeJ_r0.3.h5")
inp = DESCInput.load(path)
eq = inp.eq
pot = BoundaryPotential.fit(eq)
os.makedirs(os.path.join(ROOT, "figures", "precise_QA"), exist_ok=True)


def desc_J(nodes):
    """DESC's J at scattered flux coordinates (iota passed in, no surface integrals)."""
    return inp.compute_points(["J"], nodes)["J"]


# |J| on each surface: max and rms over a 96 x 24 (theta, zeta) grid of one field period
x = np.linspace(0, 1, 121)
rho = np.unique(np.concatenate([0.5 * (1 - np.cos(np.pi * x)), 1 - np.geomspace(1e-3, 0.2, 40), [1.0]]))
rho = rho[rho > 1e-3]
t, z = np.meshgrid(np.linspace(0, 2 * np.pi, 96, endpoint=False),
                   np.linspace(0, 2 * np.pi / eq.NFP, 24, endpoint=False), indexing="ij")
prof = {}
for name, J_of in [("DESC", desc_J), ("fit", pot.evaluate)]:
    mx, rms = [], []
    for r in rho:
        a = np.linalg.norm(J_of(np.column_stack([np.full(t.size, r), t.ravel(), z.ravel()])), axis=1)
        mx.append(a.max()); rms.append(np.sqrt(np.mean(a**2)))
    prof[name] = (np.array(mx), np.array(rms))
J_ref = prof["DESC"][0].max()

# cross-section zeta = 0
rr = np.linspace(1e-3, 1, 160); tt = np.linspace(0, 2 * np.pi, 257)
Rg, Tg = np.meshgrid(rr, tt, indexing="ij")
nodes = np.column_stack([Rg.ravel(), Tg.ravel(), np.zeros(Rg.size)])
geo = eq.compute(["R", "Z"], grid=Grid(nodes, sort=False))
R, Z = np.asarray(geo["R"]).reshape(Rg.shape), np.asarray(geo["Z"]).reshape(Rg.shape)
JD, JF = desc_J(nodes), pot.evaluate(nodes)
maps = {"DESC": np.linalg.norm(JD, axis=1).reshape(Rg.shape) / J_ref,
        "fit": np.linalg.norm(JF, axis=1).reshape(Rg.shape) / J_ref}
err = np.linalg.norm(JF - JD, axis=1).reshape(Rg.shape) / J_ref

ink, ink2, grid_c = "#0b0b0b", "#52514e", "#e4e3df"
col = {"DESC": "#2a78d6", "fit": "#eb6834"}
plt.rcParams.update({"font.size": 10, "axes.edgecolor": ink2, "axes.labelcolor": ink,
                     "xtick.color": ink2, "ytick.color": ink2, "axes.spines.top": False,
                     "axes.spines.right": False})
fig = plt.figure(figsize=(14, 9.5), facecolor="#fcfcfb")
gs = fig.add_gridspec(2, 3, height_ratios=[1, 1.25], hspace=0.35, wspace=0.32)
for j, (lo, title) in enumerate([(0.0, "|J| on each surface: max (solid), rms (dashed)"),
                                 (0.8, "zoom on the edge, 0.8 ≤ ρ ≤ 1")]):
    ax = fig.add_subplot(gs[0, j])
    for k, (mx, rms) in prof.items():
        ax.plot(rho, mx / J_ref, color=col[k], lw=2, label=k)
        ax.plot(rho, rms / J_ref, color=col[k], lw=1.2, ls="--")
    ax.set_xlim(lo, 1); ax.set_ylim(bottom=0); ax.grid(color=grid_c, lw=0.8)
    ax.set_xlabel("ρ"); ax.set_ylabel("|J| / max |J_DESC|")
    ax.set_title(title, fontsize=10, color=ink)
    if j == 1:
        ax.legend(frameon=False, loc="upper right")
ax = fig.add_subplot(gs[0, 2])
ax.semilogy(rho, np.abs(prof["fit"][0] - prof["DESC"][0]) / J_ref, color=col["fit"], lw=2, label="max")
ax.semilogy(rho, np.abs(prof["fit"][1] - prof["DESC"][1]) / J_ref, color=col["fit"], lw=1.2, ls="--",
            label="rms")
ax.set_xlim(0, 1); ax.grid(color=grid_c, lw=0.8); ax.set_xlabel("ρ")
ax.set_ylabel("|fit − DESC| / max |J_DESC|"); ax.legend(frameon=False, loc="upper left")
ax.set_title("surface max and rms of |J|: fit minus DESC", fontsize=10, color=ink)

vmax = max(m.max() for m in maps.values())
for j, (k, m) in enumerate(maps.items()):
    ax = fig.add_subplot(gs[1, j])
    pc = ax.pcolormesh(R, Z, m, cmap="Blues", vmin=0, vmax=vmax, shading="gouraud")
    ax.plot(R[-1], Z[-1], color=ink2, lw=0.8); ax.set_aspect("equal")
    ax.set_title(f"|J| / max |J_DESC|, ζ = 0: {k}", fontsize=10, color=ink)
    ax.set_xlabel("R [m]"); ax.set_ylabel("Z [m]")
fig.colorbar(pc, ax=fig.axes[3:5], shrink=0.8, pad=0.02)
ax = fig.add_subplot(gs[1, 2])
pe = ax.pcolormesh(R, Z, err, cmap="Oranges", vmin=0, vmax=err.max(), shading="gouraud")
ax.plot(R[-1], Z[-1], color=ink2, lw=0.8); ax.set_aspect("equal")
ax.set_title("|J − J_DESC| / max |J_DESC|, ζ = 0", fontsize=10, color=ink)
ax.set_xlabel("R [m]"); ax.set_ylabel("Z [m]")
fig.colorbar(pe, ax=ax, shrink=0.8, pad=0.02)

ex = {r0: err[rr <= r0].max() for r0 in (0.5, 0.8, 0.9, 0.99, 1.0)}
edge = {r0: (np.interp(r0, rho, prof["DESC"][0]) / J_ref, np.interp(r0, rho, prof["fit"][0]) / J_ref)
        for r0 in (0.9, 0.95, 0.99, 0.999, 1.0)}
note = ("max |J − J_DESC| / max |J_DESC| at ζ = 0 for ρ ≤ " +
        ", ".join(f"{r0}: {v:.1e}" for r0, v in ex.items()) + "\nmax |J| on the surface ρ = " +
        ", ".join(f"{r0}: DESC {a:.3f}, fit {b:.3f}" for r0, (a, b) in edge.items()))
fig.text(0.5, 0.015, note, ha="center", fontsize=9, color=ink2)
fig.suptitle("Smooth-current precise_QA: DESC J vs BoundaryPotential (direct T fit)", fontsize=12,
             color=ink)
fig.savefig(os.path.join(ROOT, "figures", "precise_QA", "J_vs_desc.png"), dpi=130, bbox_inches="tight")
print(note)
