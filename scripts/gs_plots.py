"""Figures for the axisymmetric tokamak runs (results/gs).

1. figures/gs_null_spectrum.png: singular values of the column-scaled force
   Jacobian at x0 for each potential resolution (gs_nullspace.py), and the number
   below relative thresholds against the number of unknowns.
2. figures/gs_drift.png: force-only runs (optimize_qa.py --track): <|F|>, the
   change of u and c_p, <p>, rms J and |B.grad rho| / |B| |grad rho| per iteration.
3. figures/gs_weights.png: the end of each weighted run (results/gs/w_*.json).

usage: python scripts/gs_plots.py [results/gs]
"""

import glob
import json
import os
import re
import sys

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
RES = sys.argv[1] if len(sys.argv) > 1 else os.path.join(ROOT, "results", "gs")
FIG = os.path.join(ROOT, "figures")


def spectra():
    files = sorted(glob.glob(os.path.join(RES, "null_L*.npz")),
                   key=lambda f: (int(re.search(r"L(\d+)", f).group(1)), f))
    if not files:
        return
    fig, ax = plt.subplots(1, 2, figsize=(11, 4.2))
    tols = (1e-2, 1e-3, 1e-4)
    counts = {t: [] for t in tols}
    ns = []
    for f in files:
        d = np.load(f)
        sv = d["sv"] / d["sv"][0]
        lab = os.path.basename(f)[5:-4]
        ax[0].semilogy(np.arange(len(sv))[::-1] / len(sv), sv, label=f"{lab} (n = {len(sv)})")
        ns.append(len(sv))
        for t in tols:
            counts[t].append(np.sum(sv < t))
    ax[0].set_xlabel("index from the smallest / n")
    ax[0].set_ylabel(r"$\sigma / \sigma_{max}$ (column-scaled)")
    ax[0].legend(fontsize=8)
    ax[0].set_title("force Jacobian at x0")
    for t in tols:
        ax[1].plot(ns, counts[t], "o-", label=rf"$\sigma/\sigma_{{max}} < {t:g}$")
    ax[1].set_xlabel("unknowns n")
    ax[1].set_ylabel("number of small singular values")
    ax[1].legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(os.path.join(FIG, "gs_null_spectrum.png"), dpi=150)
    print("saved figures/gs_null_spectrum.png")


def history(path):
    h = json.load(open(path))["history"]
    return [e for e in h if "drift" in e], json.load(open(path))["args"]


KEYS = (("<|F|>", "<|F|> / norm"), ("du", "|u - u0| / |u0|"), ("dc_p", "|c_p - c_p0| / |c_p0|"),
        ("<p>", "<p> / <p0>"), ("rms J", "rms J / rms J0"),
        ("B.grad rho", "<|B.grad rho|> / <|B||grad rho|>"))


def drift_runs():
    files = sorted(glob.glob(os.path.join(RES, "drift_f*_iter.json")))
    if not files:
        return
    fig, axs = plt.subplots(2, 3, figsize=(13, 7))
    for f in files:
        h, args = history(f)
        it = [e["it"] for e in h]
        lab = f"potential x{args['factors'][0]:g}, a/{args['frac']}"
        for ax, (k, name) in zip(axs.flat, KEYS):
            y = [e["drift"][k] for e in h]
            (ax.semilogy if k in ("<|F|>", "du", "dc_p", "B.grad rho") else ax.plot)(it, y, label=lab)
            ax.set_title(name, fontsize=10)
            ax.set_xlabel("iteration")
    axs[0, 0].legend(fontsize=8)
    fig.suptitle("force balance only, current and pressure free (tokamak L16M16)")
    fig.tight_layout()
    fig.savefig(os.path.join(FIG, "gs_drift.png"), dpi=150)
    print("saved figures/gs_drift.png")


def weight_runs():
    files = sorted(glob.glob(os.path.join(RES, "w_*_iter.json")))
    if not files:
        return
    rows = {}
    for f in files:
        m = re.match(r"w_([A-Za-z0-9]+)_([0-9.e+-]+)_iter\.json", os.path.basename(f))
        if not m:
            continue
        h, _ = history(f)
        if h:
            rows.setdefault(m.group(1), []).append((float(m.group(2)), h[-1]["drift"]))
    fig, axs = plt.subplots(2, 3, figsize=(13, 7))
    for k, v in rows.items():
        v.sort()
        w = [a for a, _ in v]
        for ax, (q, name) in zip(axs.flat, KEYS):
            ax.loglog(w, [abs(d[q]) for _, d in v], "o-", label=k)
            ax.set_title(name, fontsize=10)
            ax.set_xlabel("weight")
    axs[0, 0].legend(fontsize=8)
    fig.suptitle("force balance + one more objective: end of each run")
    fig.tight_layout()
    fig.savefig(os.path.join(FIG, "gs_weights.png"), dpi=150)
    print("saved figures/gs_weights.png")


if __name__ == "__main__":
    spectra()
    drift_runs()
    weight_runs()
