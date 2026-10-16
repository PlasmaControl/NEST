"""Poincare sections of several runs side by side, in each run's own flux coordinates.

Reads the ``_poincare_flux.npz`` of ``poincare_flux.py`` for each run and draws, per
run, the section at phi = 0 for rho > 0.5 (where NCSX's chains and the edge are)
and at phi = pi / NFP, then the iota of the traced lines of all runs on one plot.
For a fixed-boundary run the coordinates are the DESC equilibrium's; for a
free-boundary run (``optimize_fb.py``) those of its moved geometry.

usage: python poincare_compare.py OUT.png LABEL=RUN_poincare_flux.npz[:FIELD] ...
       (FIELD: "optimized" by default, or "start")
"""

import sys

import numpy as np


def main(out, specs):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    runs = []
    for s in specs:
        label, rest = s.rsplit("=", 1)
        path, field = rest.rsplit(":", 1) if ":" in rest else (rest, "optimized")
        runs.append((label, np.load(path), field))
    k = len(runs)
    fig = plt.figure(figsize=(6.5 * k, 16), constrained_layout=True)
    gs = fig.add_gridspec(3, k, height_ratios=[1, 1, 0.8])
    cmap = plt.cm.turbo
    for j, (label, d, f) in enumerate(runs):
        for i, (suffix, plane) in enumerate((("", "phi = 0"), ("_h", "phi = pi / NFP"))):
            key = f"{f}_rho{suffix}"
            if key not in d:
                continue
            rho, th = d[key], d[f"{f}_theta{suffix}"]
            idx = np.arange(rho.shape[1])
            col = cmap((np.broadcast_to(idx, rho.shape).ravel() * 0.618) % 1)
            m = np.nan_to_num(rho.ravel(), nan=-1) >= 0.5
            ax = fig.add_subplot(gs[i, j])
            ax.scatter(th.ravel()[m] / np.pi, rho.ravel()[m], s=0.8, c=col[m], lw=0)
            ax.set_xlim(0, 2)
            ax.set_ylim(0.5, 1.0)
            ax.set_xlabel("theta / pi")
            ax.set_ylabel("rho")
            ax.set_title(f"{label}, {plane}", fontsize=11)
    ax = fig.add_subplot(gs[2, :])
    d0 = runs[0][1]
    ax.plot(d0["r_desc"], d0["iota_desc"], "k-", lw=1.2, label="DESC")
    for (label, d, f), mk in zip(runs, "os^dv"):
        ax.plot(d[f"{f}_line_rho"], d[f"{f}_line_iota"], mk, ms=3.5, mfc="none",
                label=f"field lines, {label}")
    for n, m in ((3, 5), (3, 6), (3, 7), (3, 8), (6, 11), (9, 14), (9, 16)):
        ax.axhline(n / m, color="0.6", lw=0.5, ls=":")
        ax.text(1.005, n / m, f"{n}/{m}", fontsize=7, va="center")
    ax.set_xlim(0, 1)
    ax.set_ylim(0.3, 0.7)
    ax.set_xlabel("rho (mean over each line)")
    ax.set_ylabel("|iota|")
    ax.legend(loc="lower right", fontsize=9)
    fig.savefig(out, dpi=130)
    print(f"saved {out}")


if __name__ == "__main__":
    main(sys.argv[1], sys.argv[2:])
