"""Summary of the force-along-B runs at 300 A (isl_scan.py --par), against the earlier formulations:
net resonant field b_21 at q = 2 relative to the vacuum field, the traced island width, and B.grad p.

run: python scripts/dshape/isl_par_summary.py   -> figures/dshape/islands/isl_par_summary.png
"""

import json
import os

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
EPS = 300.0
# (name, label, group, median B.grad p/(|B||grad p|) in the island band from isl_pmap.py)
RUNS = [("a120n1", "J on all n\n(w_J 1e-3)", "earlier", None),
        ("a120T4q", "transport\nkappa 1e4", "earlier", None),
        ("a120T6wq", "transport\nkappa 1e6", "earlier", None),
        ("a120T8e300", "transport\nkappa 1e8", "earlier", 4.7e-6),
        ("a120P2r1", "w=1e2\nrough 1e-1", "par", 4.3e-7),
        ("a120P2r3", "w=1e2\nrough 1e-3", "par", 5.2e-7),
        ("a120P2r5", "w=1e2\nrough 1e-5", "par", 4.2e-7),
        ("a120P4r3", "w=1e4\nrough 1e-3", "par", 5.0e-9)]
COL = {"earlier": "#8c8c8c", "par": "#1f6fb4"}


def load(name):
    d = json.load(open(os.path.join(ROOT, "results", "isl", name + "_islands.json")))
    i = d["eps"].index(EPS)
    b = d["b_mn"]["NEST"]["2"][i] / d["b_mn"]["vacuum"]["2"][i]
    return b, d["w_poincare"]["NEST"][i], d["w_poincare"]["vacuum"][i], d["w_formula"]["NEST"]["2"][i]


def main():
    rows = [(lab, g, load(n), bp) for n, lab, g, bp in RUNS]
    x = np.arange(len(rows))
    fig, ax = plt.subplots(1, 3, figsize=(19, 5))
    c = [COL[g] for _, g, _, _ in rows]
    labs = [r[0] for r in rows]
    ax[0].bar(x, [r[2][0] for r in rows], color=c)
    ax[0].set_yscale("log")
    ax[0].set_ylabel("|b_21 (NEST)| / |b_21 (vacuum)|")
    ax[0].set_title("net resonant field left at q = 2 (300 A)")
    for xi, r in zip(x, rows):
        ax[0].text(xi, r[2][0] * 1.25, f"{r[2][0]:.1e}", ha="center", fontsize=7)
    wv = rows[0][2][2]
    ax[1].bar(x, [max(r[2][1], 1e-4) for r in rows], color=c, label="traced (Poincare)")
    ax[1].plot(x, [r[2][3] for r in rows], "k_", ms=14, mew=2, label="4 sqrt(|b|/(m |iota'|))")
    ax[1].axhline(wv, color="C3", ls="--", lw=1, label=f"vacuum {wv:.3f}")
    ax[1].set_yscale("log")
    ax[1].set_ylabel("island width (a)")
    ax[1].set_title("2/1 island width (bar at 1e-4: no island traced)")
    ax[1].legend(fontsize=7, loc="lower left")
    bp = [r[3] for r in rows]
    ax[2].bar([i for i, v in enumerate(bp) if v], [v for v in bp if v], color=[c[i] for i, v in enumerate(bp) if v])
    ax[2].set_yscale("log")
    ax[2].set_ylabel("median |B.grad p| / (|B||grad p|) in the island band")
    ax[2].set_title("pressure along the field")
    for a in ax:
        a.set_xticks(x)
        a.set_xticklabels(labs, fontsize=7.5)
    fig.suptitle("DSHAPE, 2/1 helical perturbation at 300 A, N = 1: grey earlier formulations, blue force-along-B weight")
    fig.tight_layout()
    path = os.path.join(ROOT, "figures", "dshape", "islands", "isl_par_summary.png")
    fig.savefig(path, dpi=120)
    for r in rows:
        print(r[0].replace("\n", " "), f"b ratio {r[2][0]:.2e}, width {r[2][1]:.4f} (formula {r[2][3]:.4f})")
    print("saved", path)


if __name__ == "__main__":
    main()
