"""DESC J vs the BoundaryPotential fit without and with the edge taper (figures/precise_QA/edge_layer_J.png).

Needs the ``edge_width`` option of ``BoundaryPotential.fit``, which was removed
afterwards: run it at commit a5cbd41 (git checkout a5cbd41 -- nest/).

run (GPU): sbatch run.sbatch scripts/stellarator/plot_edge_layer.py
"""

import os
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.join(ROOT, "nest")); sys.path.insert(0, os.path.join(ROOT, "tests"))
from desc import set_device
set_device(os.environ.get("NEST_DEVICE", "cpu"))
import numpy as np, desc.io
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
from desc.grid import Grid
from boundary_potential import BoundaryPotential
from test_magnetic_field import iota_at

path = os.path.join(ROOT, "desc_input/precise_QA_free_boundary_smooth_J/precise_QA_free_boundary_smooth_J_edgeJ_r0.3.h5")
eq = desc.io.load(path)
fits = {"fit, no taper": BoundaryPotential.fit(eq, edge_width=None),
        "fit, taper 0.1": BoundaryPotential.fit(eq, edge_width=0.1)}

def desc_J(nodes):
    """DESC's J at scattered nodes, iota passed in (no surface integrals)."""
    i0, i1 = iota_at(eq, nodes[:, 0])
    return np.asarray(eq.compute("J", grid=Grid(nodes, sort=False),
                                 data={"iota": i0, "iota_r": i1})["J"])

# radial profiles: max |J| and rms |J| on each surface
x = np.linspace(0, 1, 121)
rho = np.unique(np.concatenate([0.5 * (1 - np.cos(np.pi * x)), 1 - np.geomspace(1e-3, 0.2, 40), [1.0]]))
rho = rho[rho > 1e-3]
nt, nz = 96, 24
t, z = np.meshgrid(np.linspace(0, 2 * np.pi, nt, endpoint=False),
                   np.linspace(0, 2 * np.pi / eq.NFP, nz, endpoint=False), indexing="ij")
prof = {}
for name, J_of in [("DESC", desc_J)] + [(k, p.evaluate) for k, p in fits.items()]:
    mx, rms = [], []
    for r in rho:
        nodes = np.column_stack([np.full(t.size, r), t.ravel(), z.ravel()])
        a = np.linalg.norm(J_of(nodes), axis=1)
        mx.append(a.max()); rms.append(np.sqrt(np.mean(a**2)))
    prof[name] = (np.array(mx), np.array(rms))
J_ref = prof["DESC"][0].max()

# cross-section zeta = 0
rr = np.linspace(1e-3, 1, 160); tt = np.linspace(0, 2 * np.pi, 257)
Rg, Tg = np.meshgrid(rr, tt, indexing="ij")
nodes = np.column_stack([Rg.ravel(), Tg.ravel(), np.zeros(Rg.size)])
geo = eq.compute(["R", "Z"], grid=Grid(nodes, sort=False))
R, Z = np.asarray(geo["R"]).reshape(Rg.shape), np.asarray(geo["Z"]).reshape(Rg.shape)
JD = desc_J(nodes)
maps = {"DESC": np.linalg.norm(JD, axis=1).reshape(Rg.shape) / J_ref}
errs = {}
for k, p in fits.items():
    Jf = p.evaluate(nodes)
    maps[k] = np.linalg.norm(Jf, axis=1).reshape(Rg.shape) / J_ref
    errs[k] = np.linalg.norm(Jf - JD, axis=1).reshape(Rg.shape) / J_ref
np.savez(os.path.join(ROOT, "figures", "precise_QA", "edge_layer_J.npz"), rho=rho, R=R, Z=Z,
         **{f"prof_{k}": v for k, v in prof.items()}, **{f"map_{k}": v for k, v in maps.items()},
         **{f"err_{k}": v for k, v in errs.items()})

# figure
ink, ink2, grid_c = "#0b0b0b", "#52514e", "#e4e3df"
col = {"DESC": "#2a78d6", "fit, no taper": "#eb6834", "fit, taper 0.1": "#1baf7a"}
plt.rcParams.update({"font.size": 10, "axes.edgecolor": ink2, "axes.labelcolor": ink,
                     "xtick.color": ink2, "ytick.color": ink2, "axes.spines.top": False,
                     "axes.spines.right": False})
fig = plt.figure(figsize=(14, 12), facecolor="#fcfcfb")
gs = fig.add_gridspec(3, 3, height_ratios=[1, 1.1, 1.1], hspace=0.38, wspace=0.32)
for j, (lo, title) in enumerate([(0.0, "|J| on each surface: max (solid), rms (dashed)"),
                                 (0.8, "zoom on the edge, 0.8 ≤ ρ ≤ 1")]):
    ax = fig.add_subplot(gs[0, j])
    for k, (mx, rms) in prof.items():
        ax.plot(rho, mx / J_ref, color=col[k], lw=2, label=k)
        ax.plot(rho, rms / J_ref, color=col[k], lw=1.2, ls="--")
    ax.set_xlim(lo, 1); ax.set_ylim(bottom=0)
    ax.grid(color=grid_c, lw=0.8); ax.set_xlabel("ρ")
    ax.set_ylabel("|J| / max |J_DESC|")
    ax.set_title(title, fontsize=10, color=ink)
    if j == 1:
        ax.legend(frameon=False, loc="upper right")
ax = fig.add_subplot(gs[0, 2])
for k in fits:
    ax.semilogy(rho, np.abs(prof[k][1] - prof["DESC"][1]) / J_ref, color=col[k], lw=2, label=k)
ax.set_xlim(0, 1); ax.grid(color=grid_c, lw=0.8); ax.set_xlabel("ρ")
ax.set_ylabel("|Δ rms|J|| / max |J_DESC|")
ax.set_title("surface rms of |J|: fit minus DESC", fontsize=10, color=ink)
ax.legend(frameon=False, loc="upper left")

vmax = max(m.max() for m in maps.values())
for j, (k, m) in enumerate(maps.items()):
    ax = fig.add_subplot(gs[1, j])
    pc = ax.pcolormesh(R, Z, m, cmap="Blues", vmin=0, vmax=vmax, shading="gouraud")
    ax.plot(R[-1], Z[-1], color=ink2, lw=0.8)
    ax.set_aspect("equal"); ax.set_title(f"|J|, ζ = 0: {k}", fontsize=10, color=ink)
    ax.set_xlabel("R [m]"); ax.set_ylabel("Z [m]")
fig.colorbar(pc, ax=fig.axes[3:6], shrink=0.85, pad=0.02)
emax = max(e.max() for e in errs.values())
for j, (k, e) in enumerate(errs.items()):
    ax = fig.add_subplot(gs[2, j])
    pe = ax.pcolormesh(R, Z, e, cmap="Oranges", vmin=0, vmax=emax, shading="gouraud")
    ax.plot(R[-1], Z[-1], color=ink2, lw=0.8)
    ax.set_aspect("equal"); ax.set_title(f"|J − J_DESC|, ζ = 0: {k}", fontsize=10, color=ink)
    ax.set_xlabel("R [m]"); ax.set_ylabel("Z [m]")
ax = fig.add_subplot(gs[2, 2]); ax.axis("off")
fig.colorbar(pe, ax=fig.axes[-3:-1], shrink=0.85, pad=0.02)
lines = ["All |J| in units of max |J_DESC|.", "", "max |J - J_DESC| at zeta = 0"]
for k, e in errs.items():
    lines.append(f"  {k:15s} rho <= 0.9: {e[rr <= 0.9].max():.1e}  all: {e.max():.1e}")
lines += ["", "max |J| on the surface rho ="]
lines.append(" " * 17 + "".join(f"{r:>7g}" for r in (0.9, 0.95, 0.99, 0.999, 1.0)))
for k in prof:
    lines.append(f"  {k:15s}" + "".join(f"{np.interp(r, rho, prof[k][0]) / J_ref:7.3f}"
                                        for r in (0.9, 0.95, 0.99, 0.999, 1.0)))
ax.text(-0.05, 0.95, "\n".join(lines), va="top", fontsize=8.5, color=ink, family="monospace")
fig.suptitle("Smooth-current precise_QA: DESC J vs BoundaryPotential (T fit), without and with the edge taper",
             fontsize=12, color=ink)
fig.savefig(os.path.join(ROOT, "figures", "precise_QA", "edge_layer_J.png"), dpi=130, bbox_inches="tight")
print("\n".join(lines))
