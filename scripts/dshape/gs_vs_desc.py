"""How far the converged NEST end states are from the DESC equilibrium they started from.

Reads results/gs/fields_<run>.npz (gs_check_end.py --save: the end state's J, B, p and DESC's
on the common check grid) and, by radius, the rms of NEST - DESC for B, J along and across
B_DESC, and p, over DESC's volume rms (as gs_ladder.py), plus the largest pointwise
|B_NEST - B_DESC| / |B_DESC|.  Prints a table and draws figures/<name>.png (difference
against rho, one line per run).

usage: python scripts/dshape/gs_vs_desc.py --runs results/gs/hN_a40D ... --labels a/40 ...
"""

import argparse
import os

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
QS = ("B", "J along B", "J across B", "p")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--runs", nargs="+", required=True)
    ap.add_argument("--labels", nargs="+", required=True)
    ap.add_argument("--bins", type=int, default=24)
    ap.add_argument("--name", default="gs_vs_desc")
    a = ap.parse_args()
    fig, ax = plt.subplots(1, 4, figsize=(22, 4.8))
    edges = 1 - np.geomspace(1, 1e-3, a.bins + 1)  # radial bins packed towards the LCFS
    print(f"{'run':10s} " + " ".join(f"{q:>12s}" for q in QS) + f" {'max|dB|/|B|':>12s}")
    for run, lab in zip(a.runs, a.labels):
        d = np.load(os.path.join(ROOT, "results", "gs", f"fields_{os.path.basename(run)}.npz"))
        w = d["dV"] * d["ok"]
        rho = d["rtz"][:, 0]
        B0, J0, p0 = d["desc_B"], d["desc_J"], d["desc_p"]
        bh = B0 / np.linalg.norm(B0, axis=1)[:, None]
        dJ = d["end_J"] - J0
        par = np.sum(dJ * bh, 1)
        diffs = {"B": np.sum((d["end_B"] - B0) ** 2, 1), "J along B": par**2,
                 "J across B": np.sum((dJ - par[:, None] * bh) ** 2, 1), "p": (d["end_p"] - p0) ** 2}
        ref = {"B": np.sum(B0**2, 1), "J along B": np.sum(J0**2, 1), "J across B": np.sum(J0**2, 1), "p": p0**2}
        tot = {q: np.sqrt(np.sum(w * diffs[q]) / np.sum(w * ref[q])) for q in QS}
        mx = np.max(np.linalg.norm(d["end_B"] - B0, axis=1) / np.linalg.norm(B0, axis=1))
        print(f"{lab:10s} " + " ".join(f"{tot[q]:12.2e}" for q in QS) + f" {mx:12.2e}")
        mid, curves = [], {q: [] for q in QS}
        for lo, hi in zip(edges[:-1], edges[1:]):
            m = (rho > lo) & (rho <= hi)
            if not m.any():
                continue
            mid.append(1 - 0.5 * (lo + hi))
            for q in QS:
                curves[q].append(np.sqrt(np.sum((w * diffs[q])[m]) / np.sum(w * ref[q])))
        for x_, q in zip(ax, QS):
            x_.loglog(mid, curves[q], "o-", ms=3, label=lab)
    for x_, q in zip(ax, QS):
        x_.set_xlabel(r"$1-\rho$")
        x_.set_title(f"NEST - DESC L32: {q}\n(rms in the bin / DESC's volume rms)", fontsize=10)
        x_.legend(fontsize=8)
    fig.suptitle("Converged NEST end states against the DESC L32 free-boundary equilibrium they started from")
    fig.tight_layout()
    out = os.path.join(ROOT, "figures", "dshape", f"{a.name}.png")
    os.makedirs(os.path.dirname(out), exist_ok=True)
    fig.savefig(out, dpi=130)
    print("saved", out)


if __name__ == "__main__":
    main()
