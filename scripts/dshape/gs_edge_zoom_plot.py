"""Redraw gs_edge_zoom.py's figure from its saved curves (results/gs/<name>.json), cut at 1 - rho
>= --min: below ~1e-7 the force is at the double-precision floor (~3e-17) of J x B - grad p and the
exact field's quadrature meets round-off in the node positions (NaN), so nothing is measured there.

usage: python scripts/dshape/gs_edge_zoom_plot.py --name gs_edge_zoom_h [--min 1e-7]
"""

import argparse
import json
import os

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--name", default="gs_edge_zoom_h")
    ap.add_argument("--min", type=float, default=1e-7)
    ap.add_argument("--out", default=None)
    a = ap.parse_args()
    d = json.load(open(os.path.join(ROOT, "results", "gs", f"{a.name}.json")))
    om = np.asarray(d["one_minus_rho"])
    keep = om >= a.min * (1 - 1e-9)
    fig, ax = plt.subplots(1, 2, figsize=(14, 5.2))
    cols = iter(["C0", "C3", "C2", "C1", "C4", "C5"])
    first = None
    for k, (Jc, Fc) in ((k, v) for k, v in d.items() if k != "one_minus_rho"):
        st = dict(color="k", lw=2.2) if k == "DESC" else dict(color=next(cols), lw=1.4)
        Jc, Fc = np.asarray(Jc, float)[keep], np.asarray(Fc, float)[keep]
        ax[0].loglog(om[keep], Jc, label=k, **st)
        ax[1].loglog(om[keep], Fc, label=k, **st)
        if first is None and k != "DESC":
            first = Jc
    ax[0].loglog(om[keep], first[0] * (om[keep] / om[keep][0]) ** 2, "k--", lw=0.7, label=r"$\propto (1-\rho)^2$")
    ax[0].set_title(r"$\langle|J|\rangle_{surface}$ / max$\langle|J_{DESC}|\rangle$")
    ax[1].set_title(r"$\langle|J\times B-\nabla p|\rangle_{surface}$ / norm (NEST: coils + exact $B_{plasma}$)")
    for x_ in ax:
        x_.set_xlabel(r"$1-\rho$")
        x_.legend(fontsize=8)
    fig.suptitle("DSHAPE L = M = 32 free boundary: DESC against NEST at the LCFS (direct evaluation, no grid), "
                 f"down to 1 - rho = {a.min:g}")
    fig.tight_layout()
    out = os.path.join(ROOT, "figures", "dshape", f"{a.out or a.name + '_cut'}.png")
    os.makedirs(os.path.dirname(out), exist_ok=True)
    fig.savefig(out, dpi=140)
    print("saved", out)


if __name__ == "__main__":
    main()
