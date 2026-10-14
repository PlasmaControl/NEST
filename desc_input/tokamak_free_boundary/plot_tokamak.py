"""Figures of the axisymmetric free-boundary tokamak and its external field.

tokamak_L{L}M{M}_overview.png, 3 x 3 panels:
  a. flux surfaces (rho = 0.1 ... 1), the theta lines and the winding surface;
  b. p, I and q against rho;
  c. <|J|> (flux-surface average) against rho, with log axes near the edge:
     J -> 0 with zero slope;
  d. |J| in the cross-section;
  e. the local force error |F| / <|grad |B|^2 / 2 mu0|>_vol in the cross-section;
  f. the force error by rho (flux-surface averages), L = 12 and 16;
  g. B.n / |B| on the LCFS (coils + plasma, virtual casing) against theta;
  h. the pressure jump [[B^2 + 2 mu0 p]] / B^2 on the LCFS against theta;
  i. the current potential: Phi_SV(theta) and |K| on the winding surface.

usage: python plot_tokamak.py [L M]   (default 16 16)
"""

import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, "..", "precise_QA_free_boundary"))

from desc import set_device  # noqa: E402

set_device("cpu")

import matplotlib  # noqa: E402

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

import desc.io  # noqa: E402
from desc.grid import LinearGrid, QuadratureGrid  # noqa: E402
from desc.integrals.surface_integral import surface_averages  # noqa: E402
from desc.objectives import BoundaryError  # noqa: E402

L, M = (int(v) for v in sys.argv[1:3]) if len(sys.argv) > 2 else (16, 16)
tag = os.path.join(HERE, f"tokamak_L{L}M{M}")
eq = desc.io.load(tag + ".h5")
field = desc.io.load(tag + "_field.h5")

fig, axs = plt.subplots(3, 3, figsize=(15, 14))
ax = axs.flat

# a. flux surfaces and the winding surface
th = np.linspace(0, 2 * np.pi, 257)
for r in np.linspace(0.1, 1, 10):
    d = eq.compute(["R", "Z"], grid=LinearGrid(rho=r, theta=th, zeta=0.0))
    ax[0].plot(d["R"], d["Z"], "k-" if r == 1 else "C0-", lw=1.5 if r == 1 else 0.7)
for t in np.linspace(0, 2 * np.pi, 16, endpoint=False):
    d = eq.compute(["R", "Z"], grid=LinearGrid(rho=np.linspace(0, 1, 30), theta=t, zeta=0.0))
    ax[0].plot(d["R"], d["Z"], "C0:", lw=0.5)
dw = field.compute(["R", "Z"], grid=LinearGrid(theta=th, zeta=0.0))
ax[0].plot(dw["R"], dw["Z"], "C3-", lw=1.5, label="winding surface (offset 0.8 m)")
ax[0].set_aspect("equal")
ax[0].set_xlabel("R [m]")
ax[0].set_ylabel("Z [m]")
ax[0].legend(fontsize=8, loc="upper right")
ax[0].set_title("a. flux surfaces and winding surface")

# b, c. profiles and <|J|>
rho = np.linspace(1e-3, 1, 200)
g = LinearGrid(rho=rho, M=2 * eq.M, N=0, NFP=1)
d = eq.compute(["p", "current", "q", "|J|", "sqrt(g)", "|F|", "<|grad(|B|^2)|/2mu0>_vol"],
               grid=g)
avg = lambda q: np.asarray(surface_averages(g, q, d["sqrt(g)"], expand_out=False))  # noqa: E731
p, I, q = g.compress(d["p"]), g.compress(d["current"]), g.compress(d["q"])
J = avg(d["|J|"])
b = ax[1]
b.plot(rho, p / p.max(), label=f"p / {p.max():.0f} Pa")
b.plot(rho, I / I[-1], label=f"I / ({I[-1] / 1e3:.0f} kA)")
b.set_xlabel(r"$\rho$")
b.legend(loc="center left", fontsize=8)
b2 = b.twinx()
b2.plot(rho, np.abs(q), "C2", label="|q|")
b2.set_ylabel("|q|", color="C2")
b.set_title(r"b. profiles: p $\propto (1-\rho^2)^3$, $I \propto 1-(1-\rho^2)^3$")
for LL, c in ((12, "C1"), (16, "C0")):
    path = os.path.join(HERE, f"tokamak_L{LL}M{LL}.h5")
    if os.path.exists(path):
        e = desc.io.load(path)
        ge = LinearGrid(rho=rho, M=2 * e.M, N=0, NFP=1)
        de = e.compute(["|J|", "sqrt(g)"], grid=ge)
        Je = np.asarray(surface_averages(ge, de["|J|"], de["sqrt(g)"], expand_out=False))
        ax[2].semilogy(1 - rho, Je / Je.max(), c, label=rf"$\langle|J|\rangle$ / max, L = M = {LL}")
ax[2].semilogy(1 - rho, (1 - rho) ** 2 * 4, "k--", lw=0.8, label=r"$\propto (1-\rho)^2$")
ax[2].set_xscale("log")
ax[2].set_xlabel(r"$1 - \rho$")
ax[2].legend(fontsize=8)
ax[2].set_title(r"c. $\langle|J|\rangle$ near the LCFS (exact: $\propto(1-\rho)^2$)")

# d, e. |J| and the local force error in the cross-section
rs, ts = np.linspace(0.01, 1, 60), np.linspace(0, 2 * np.pi, 129)
gs = LinearGrid(rho=rs, theta=ts, zeta=0.0)
ds = eq.compute(["R", "Z", "|J|", "|F|"], grid=gs)
# (rho, theta) arrays from the grid's own node order
ir = np.argmin(np.abs(np.asarray(gs.nodes[:, 0])[:, None] - rs), 1)
it = np.argmin(np.abs(np.asarray(gs.nodes[:, 1])[:, None] - ts), 1)


def tab(v):
    out = np.empty((len(rs), len(ts)))
    out[ir, it] = np.asarray(v)
    return out

norm = float(np.ravel(eq.compute("<|grad(|B|^2)|/2mu0>_vol",
                                 grid=QuadratureGrid(2 * L, 2 * M, 0, 1))[
    "<|grad(|B|^2)|/2mu0>_vol"])[0])
R, Z = tab(ds["R"]), tab(ds["Z"])
im = ax[3].contourf(R, Z, tab(ds["|J|"]) / 1e3, 30, cmap="viridis")
plt.colorbar(im, ax=ax[3], label="|J| [kA/m²]")
ax[3].set_aspect("equal")
ax[3].set_title("d. |J| (current density)")
Fl = np.log10(np.maximum(tab(ds["|F|"]), 1e-30) / norm)
im = ax[4].contourf(R, Z, Fl, np.linspace(-9, -3, 25), cmap="magma", extend="both")
plt.colorbar(im, ax=ax[4], label=r"log$_{10}$ |F| / $\langle|\nabla B^2/2\mu_0|\rangle$")
ax[4].set_aspect("equal")
ax[4].set_title("e. local force error (DESC)")

# f. force error by rho, both resolutions
for LL in (12, 16):
    path = os.path.join(HERE, f"tokamak_L{LL}M{LL}.h5")
    if not os.path.exists(path):
        continue
    e = desc.io.load(path)
    gg = LinearGrid(rho=rho, M=2 * e.M, N=0, NFP=1)
    de = e.compute(["|F|", "sqrt(g)"], grid=gg)
    nn = float(np.ravel(e.compute("<|grad(|B|^2)|/2mu0>_vol",
                                  grid=QuadratureGrid(2 * e.L, 2 * e.M, 0, 1))[
        "<|grad(|B|^2)|/2mu0>_vol"])[0])
    Fr = np.asarray(surface_averages(gg, de["|F|"], de["sqrt(g)"], expand_out=False)) / nn
    ax[5].semilogy(rho, Fr, label=f"L = M = {LL}")
ax[5].set_xlabel(r"$\rho$")
ax[5].set_ylabel(r"$\langle|F|\rangle_\rho$ / norm")
ax[5].legend(fontsize=8)
ax[5].set_title("f. force error by flux surface")

# g, h. free-boundary residuals on the LCFS (virtual casing)
obj = BoundaryError(eq=eq, field=field, field_fixed=True)
obj.build(verbose=0)
f = np.asarray(obj.compute(*obj.xs(eq)))
grid = obj.constants["eval_transforms"]["grid"]
dg = eq.compute(["|e_theta x e_zeta|", "|B|"], grid=grid)
n = grid.num_nodes
t = np.asarray(grid.nodes[:, 1])
o = np.argsort(t)
Bn = f[:n] / (dg["|e_theta x e_zeta|"] * dg["|B|"])
dP = f[n : 2 * n] / (dg["|e_theta x e_zeta|"] * dg["|B|"] ** 2)
ax[6].plot(t[o], Bn[o], ".-", ms=3)
ax[6].set_title(f"g. B.n / |B| on the LCFS (rms {np.sqrt(np.mean(Bn**2)):.1e})")
ax[7].plot(t[o], dP[o], ".-", ms=3, color="C1")
ax[7].set_title(rf"h. $[[B^2 + 2\mu_0 p]] / B^2$ on the LCFS (rms {np.sqrt(np.mean(dP**2)):.1e})")
for a in (ax[6], ax[7]):
    a.set_xlabel(r"$\theta$")
    a.axhline(0, color="k", lw=0.5)

# i. the current potential on the winding surface
gw = LinearGrid(theta=th, zeta=0.0)
dw = field.compute(["Phi", "K"], grid=gw)
K = np.linalg.norm(np.asarray(dw["K"]), axis=1)
Phi_sv = np.asarray(dw["Phi"])  # at zeta = 0 the secular part G zeta / 2 pi vanishes (I = 0)
ax[8].plot(th, Phi_sv / 1e3, label=r"$\Phi_{SV}(\theta)$ [kA]")
ax[8].set_xlabel(r"$\theta$")
ax[8].legend(loc="upper left", fontsize=8)
a2 = ax[8].twinx()
a2.plot(th, K / 1e3, "C3", label="|K|")
a2.set_ylabel("|K| [kA/m]", color="C3")
ax[8].set_title(f"i. current potential (G = {field.G / 1e6:.2f} MA)")

fig.suptitle(f"Axisymmetric free-boundary tokamak L = M = {L}: <beta> = "
             f"{float(np.ravel(eq.compute('<beta>_vol')['<beta>_vol'])[0]) * 100:.2f} %", y=0.995)
fig.tight_layout()
out = tag + "_overview.png"
fig.savefig(out, dpi=130)
print("saved", out)
