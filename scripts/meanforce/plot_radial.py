"""Plot ``radial_force.py`` output: |F|/norm averaged over each coordinate surface, the
mean over rho ranges, and each range's share of M (DESC, NEST start, NEST end states).

    python scripts/meanforce/plot_radial.py results/meanforce/radial_a40.json \
        --out figures/meanforce/radial_a40.png
"""

import argparse
import json

import matplotlib
import numpy as np

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

STYLE = {"DESC": ("k", "-"), "NEST start": ("0.55", "--")}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("json")
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    d = json.load(open(a.json))
    rho = np.asarray(d["rho"])
    bins = d["bins"]
    labels = list(d["states"])
    colors = {k: STYLE.get(k, (f"C{i}", "-")) for i, k in enumerate(labels)}
    fig, ax = plt.subplots(1, 3, figsize=(17, 5))
    a0 = ax[0]
    for k, v in d["states"].items():
        c, ls = colors[k]
        a0.semilogy(rho, v["surface"]["mean"], ls, color=c, marker=".", label=f"{k} (M {v['M']:.2e})")
        a0.semilogy(rho, v["surface"]["max"], ls, color=c, lw=0.6, alpha=0.6)
    a0.set_xlabel("rho")
    a0.set_ylabel("|F| / norm")
    a0.set_title("over each surface rho = const: mean (thick), max (thin)")
    a0.legend(fontsize=8)
    keys = list(d["states"][labels[0]]["bins"])
    x = np.arange(len(keys))
    wbar = 0.8 / len(labels)
    for j, (name, q, axx, ttl) in enumerate(
            (("mean", "mean", ax[1], "mean |F| / norm over each rho range"),
             ("part", "part of M", ax[2], "each range's part of M = <|F|>_vol / norm"))):
        for i, k in enumerate(labels):
            vals = [d["states"][k]["bins"][b][q] for b in keys]
            axx.bar(x + (i - (len(labels) - 1) / 2) * wbar, vals, wbar, color=colors[k][0],
                    label=k)
        axx.set_xticks(x)
        axx.set_xticklabels(keys, rotation=45)
        axx.set_xlabel("rho range")
        axx.set_title(ttl)
        if name == "mean":
            axx.set_yscale("log")
        axx.legend(fontsize=8)
    vf = [d["states"][labels[0]]["bins"][b]["volume fraction"] for b in keys]
    for xi, f in zip(x, vf):
        ax[2].text(xi, ax[2].get_ylim()[1] * 0.97, f"{100 * f:.0f}%", ha="center", va="top",
                   fontsize=7, color="0.4")
    ax[2].text(0.01, 0.99, "grey: volume share", transform=ax[2].transAxes, fontsize=7,
               color="0.4", va="top")
    fig.tight_layout()
    fig.savefig(a.out, dpi=120)
    print(f"saved {a.out}")


if __name__ == "__main__":
    main()
