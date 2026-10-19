"""Convergence with the iterations: optimize_qa.py's LM (truncated CG) against the
exact-step LM of gs_dense_solve.py, on the same model (coils + box B_plasma).

Reads figures/gs_<run>_convergence.npz (gs_long_run.py --ref <converged run>) for the
LM run and the dense runs started from its end and from x0, and draws
figures/<name>.png:
  (a) <|F|>/norm on the run grid against the iteration (the dense run from the LM
      end continues at the LM's last iteration);
  (b) f - f_min (f_min: the converged dense run), log;
  (c) the change of the fields per iteration (B, J along and across B, p);
  (d) the distance of each iterate to the converged minimum.
"""

import argparse
import json
import os

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
GS = os.path.join(ROOT, "results", "gs")
QS = ("B", "J along", "J across", "p")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--lm", default="fbJe3_wj1e-3_f2_a20")
    ap.add_argument("--name", default="gs_dense_convergence")
    a = ap.parse_args()
    runs = [(a.lm, "LM, truncated CG (optimize_qa.py)", "k", 0),
            (a.lm + "_dense", "exact steps, from the LM end", "C3", None),
            (a.lm + "_dense0", "exact steps, from x0", "C0", 0)]
    D = {}
    for run, lab, c, off in runs:
        d = dict(np.load(os.path.join(ROOT, "figures", f"gs_{run}_convergence.npz")))
        hist = json.load(open(os.path.join(GS, f"{run}_iter.json")))["history"]
        d["f"] = np.array([e["f"] for e in hist])
        d["t"] = np.array([e.get("time", np.nan) for e in hist])
        if len(d[f"toref_{QS[0]}"]) == len(d["F"]) + 1:  # the iterates begin with the start (gs_dense_solve.py)
            for q in QS:
                d[f"toref_{q}"], d[f"step_{q}"] = d[f"toref_{q}"][1:], d[f"step_{q}"][1:]
        D[run] = d
    n_lm = len(D[a.lm]["F"]) - 1
    f_min = min(D[a.lm + "_dense0"]["f"].min(), D[a.lm + "_dense"]["f"].min())
    fig, ax = plt.subplots(1, 4, figsize=(24, 5))
    for run, lab, c, off in runs:
        d = D[run]
        it = np.arange(len(d["F"])) + (n_lm if off is None else off)
        tpi = np.nanmedian(d["t"][1:])
        L = f"{lab} ({tpi:.1f} s/it)"
        ax[0].semilogy(it, d["F"], color=c, marker="." if run != a.lm else None, label=L)
        ax[1].semilogy(it, np.maximum(d["f"] - f_min, 1e-30) / f_min, color=c,
                       marker="." if run != a.lm else None, label=L)
        for q, ls in zip(QS, ("-", "--", "-.", ":")):
            ax[2].semilogy(it[1:], d[f"step_{q}"], color=c, ls=ls, label=f"{q}: {lab}" if run != a.lm + "_dense" else None)
            ax[3].semilogy(it, np.maximum(d[f"toref_{q}"], 1e-16), color=c, ls=ls,
                           label=f"{q}: {lab}" if run != a.lm + "_dense" else None)
    ax[0].set_title(r"$\langle|F|\rangle$/norm, run grid (coils + box $B_{plasma}$, a/20)")
    ax[1].set_title(r"$(f - f_{min}) / f_{min}$, $f = |r|^2/2$")
    ax[2].set_title("change of the fields per iteration (rms / start's rms)")
    ax[3].set_title("distance to the converged minimum (rms / start's rms)")
    for x in ax:
        x.set_xlabel("iteration")
        x.legend(fontsize=7)
    fig.suptitle("L = M = 32 free boundary, coils' field with NEST's box B_plasma (no correction), x2, a/20, "
                 "edge power 3, J-fitted start, w_J 1e-3 at rho < 0.9")
    fig.tight_layout()
    out = os.path.join(ROOT, "figures", f"{a.name}.png")
    fig.savefig(out, dpi=130)
    print("saved", out)


if __name__ == "__main__":
    main()
