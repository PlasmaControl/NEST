"""Poincare sections at phi = 0 of a NEST solution for any equilibrium and coils.

Uses the field and the tracer of ``scripts/validate_opt.py``: B_ext + B_plasma is
tabulated once on a cylindrical grid over a field period, and field lines are
traced in phi with RK4. On top of the uniform seeds along theta = 0, lines are
seeded around each resonant surface iota = n/m (n a multiple of NFP, m <= 12) on
theta = 0 and theta = pi: with stellarator symmetry the island O- and X-points of
the phi = 0 section lie on that line. The resonant DESC surfaces are drawn and
labeled.

Panels: the start (the fits) and the optimized x, and a zoom of the optimized
section on the outboard side. The sections are saved to the npz next to the figure.

run (GPU): sbatch --mem=128G run.sbatch desc_input/vmec_free_boundary/poincare.py \
    --eq desc_input/vmec_free_boundary/NCSX/NCSX_L16M16N12.h5 \
    --coils desc_input/vmec_free_boundary/NCSX/NCSX_coils.h5 \
    --x results/ncsx/opt_L16M16N12.npz --frac 20
"""

import argparse
import os
import sys
import time
from fractions import Fraction

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
from force import ForceResidual  # noqa: E402
from grid import CartesianGrid  # noqa: E402
from objective import Fields  # noqa: E402
from pressure import Pressure  # noqa: E402
from separable import Separable  # noqa: E402
from validate_opt import field_on_cylinder, poincare  # noqa: E402

jax.config.update("jax_enable_x64", True)


def resonances(inp, max_m=12):
    """(n, m, rho) of iota = n/m crossings, n a multiple of NFP, lowest m first."""
    rho = np.linspace(1e-3, 1, 2001)
    iota = np.abs(inp.iota(rho)[0])
    out, seen = [], set()
    for m in range(1, max_m + 1):
        for n in range(inp.NFP, 4 * inp.NFP + 1, inp.NFP):
            t = Fraction(n, m)
            if t in seen:
                continue
            idx = np.nonzero(np.diff(np.sign(iota - float(t))))[0]
            if len(idx):
                seen.add(t)
            for i in idx:
                r = rho[i] + (float(t) - iota[i]) * (rho[i + 1] - rho[i]) / (iota[i + 1] - iota[i])
                out.append((n, m, float(r)))
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--eq", required=True)
    ap.add_argument("--coils", required=True)
    ap.add_argument("--x", required=True, help="optimize_qa.py npz")
    ap.add_argument("--frac", type=int, default=20)
    ap.add_argument("--kernel", default="truncated")
    ap.add_argument("--factors", type=float, nargs=3, default=(3, 3, 2.5))
    ap.add_argument("--periods", type=int, default=400)
    ap.add_argument("--lines", type=int, default=32, help="uniform seeds on theta = 0")
    a = ap.parse_args()
    t0 = time.time()
    inp = DESCInput.load(a.eq, field=a.coils)
    eq = inp.eq
    cur = fit_cache.fit(BoundaryPotential, inp, factors=tuple(a.factors))
    pre = fit_cache.fit(Pressure, inp, factors=tuple(a.factors), edge_order=2)
    box = CartesianGrid.periodic_box(inp, inp.a / a.frac, odd=bool(eq.sym)).locate_plasma(inp)
    x1 = np.load(a.x)["x"]
    u0, u1 = cur.u, x1[: cur.n_free]
    fields = Fields(ForceResidual(inp, box), cur, pre, kernel=a.kernel)
    res = resonances(inp)
    print(f"{inp}\nx: {a.x}; box h = a/{a.frac} {box.shape}\nresonances (n, m, rho): "
          + ", ".join(f"{n}/{m} at {r:.3f}" for n, m, r in res), flush=True)

    RZ = Separable([eq.R_basis, eq.Z_basis], [eq.R_lmn, eq.Z_lmn], eq.NFP)
    seeds = [(r, 0.0) for r in np.linspace(0.04, 1.0, a.lines)]
    for _, _, r in res:
        seeds += [(min(r + d, 0.995), th) for d in (-0.02, -0.01, 0.0, 0.01, 0.02)
                  for th in (0.0, np.pi)]
    seeds = np.array([s for s in seeds if s[0] > 0.02])
    st = np.asarray(RZ.evaluate(jnp.asarray(seeds[:, 0]), jnp.asarray(seeds[:, 1]),
                                jnp.zeros(len(seeds))))
    sections = {}
    for name, u in (("start (fits)", u0), ("optimized", u1)):
        t1 = time.time()
        Rs, phis, Zs, Bcyl = field_on_cylinder(fields, u, inp)
        sections[name] = poincare(Rs, phis, Zs, Bcyl, st[:, 0], st[:, 1], eq.NFP, a.periods)
        lost = np.isnan(sections[name][0][-1]).sum()
        print(f"Poincare, {name}: {len(seeds)} lines x {a.periods} field periods, {lost} left "
              f"the grid; {time.time() - t1:.0f} s", flush=True)

    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    th = np.linspace(0, 2 * np.pi, 400)

    def surface(r):
        return np.asarray(RZ.evaluate(jnp.full(400, r), jnp.asarray(th), jnp.zeros(400)))

    colors = plt.cm.tab10(np.arange(len(res)) % 10)
    fig, axs = plt.subplots(1, 3, figsize=(19, 6.5), constrained_layout=True)
    panels = [("start (fits)", axs[0]), ("optimized", axs[1]), ("optimized", axs[2])]
    for name, ax in panels:
        PR, PZ = sections[name]
        for r in np.linspace(0.1, 1.0, 10):
            s = surface(r)
            ax.plot(s[:, 0], s[:, 1], color="0.75", lw=0.5)
        for (n, m, r), c in zip(res, colors):
            s = surface(r)
            ax.plot(s[:, 0], s[:, 1], color=c, lw=0.9, ls="--",
                    label=f"iota = {n}/{m} ({m} islands), rho = {r:.2f}")
        ax.scatter(PR[1:].ravel(), PZ[1:].ravel(), s=0.5, c="k", lw=0)
        ax.set_aspect("equal")
        ax.set_xlabel("R [m]")
    axs[0].set_ylabel("Z [m]")
    axs[0].set_title("start (fits of DESC's J and p), phi = 0")
    axs[1].set_title(f"optimized ({os.path.basename(a.x)}), phi = 0")
    # zoom: outboard half, rho >= 0.45
    edge, mid = surface(1.0), surface(0.45)
    R0 = mid[:, 0].max() - 0.05 * inp.a
    axs[2].set_xlim(R0, edge[:, 0].max() + 0.1 * inp.a)
    zmax = edge[edge[:, 0] > R0, 1]
    axs[2].set_ylim(zmax.min() - 0.05 * inp.a, zmax.max() + 0.05 * inp.a)
    axs[2].set_title("optimized, outboard zoom")
    axs[1].legend(loc="upper center", bbox_to_anchor=(0.5, -0.1), ncol=2, fontsize=8)
    fig.suptitle(f"{inp.name}: field lines of B_ext + B_plasma "
                 "(grey: DESC flux surfaces; dashed: resonant surfaces)")
    os.makedirs(os.path.join(ROOT, "figures"), exist_ok=True)
    base = os.path.join(ROOT, "figures", os.path.splitext(os.path.basename(a.x))[0])
    if not os.path.basename(base).startswith("ncsx") and "ncsx" in a.x.lower():
        base = os.path.join(os.path.dirname(base), "ncsx_" + os.path.basename(base))
    fig.savefig(base + "_poincare.png", dpi=150)
    np.savez(base + "_poincare.npz", seeds=seeds, resonances=np.array(res),
             **{k.split()[0] + "_R": v[0] for k, v in sections.items()},
             **{k.split()[0] + "_Z": v[1] for k, v in sections.items()})
    print(f"saved {base}_poincare.png/.npz; total {time.time() - t0:.0f} s")


if __name__ == "__main__":
    main()
