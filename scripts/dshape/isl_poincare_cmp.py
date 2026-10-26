"""Poincare sections at phi = 0 of several runs side by side, theta* on the x axis, rho on the y axis
(from results/isl/<name>_sections.npz of isl_islands.py): the vacuum superposition of the first
run, then the NEST equilibrium of each.

run: python scripts/dshape/isl_poincare_cmp.py --names a120P2r1 a120P2r3 a120P2r5 a120P4r3 --eps 300 --out isl_par_poincare
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
    ap.add_argument("--names", nargs="+", required=True)
    ap.add_argument("--labels", nargs="*", default=None)
    ap.add_argument("--eps", type=float, default=300.0)
    ap.add_argument("--band", type=float, default=0.09, help="half width in rho around q = 2")
    ap.add_argument("--out", default="isl_par_poincare")
    a = ap.parse_args()
    labels = a.labels or a.names
    panels = []
    for i, n in enumerate(a.names):
        sec = np.load(os.path.join(ROOT, "results", "isl", n + "_sections.npz"))
        d = json.load(open(os.path.join(ROOT, "results", "isl", n + "_islands.json")))
        k = d["eps"].index(a.eps)
        rs = d["surfaces"]["2"][0]
        if i == 0:
            panels.append(("vacuum (no response)", sec, "vacuum", d["w_poincare"]["vacuum"][k], rs))
        panels.append((labels[i], sec, "NEST", d["w_poincare"]["NEST"][k], rs))
    fig, ax = plt.subplots(1, len(panels), figsize=(4.2 * len(panels), 5.2), squeeze=False, sharey=True)
    for x_, (lab, sec, kind, w, rs) in zip(ax[0], panels):
        key = f"{kind}_{a.eps:g}_"
        r, ts = sec[key + "rho"], sec[key + "ths"]
        tr = sec[key + "trapped"] if key + "trapped" in sec else np.zeros(r.shape[1], bool)
        for j in range(r.shape[1]):
            x_.plot(ts[:, j], r[:, j], ".", ms=0.6, color="C3" if tr[j] else "k")
        x_.axhline(rs, color="C0", lw=0.5)
        x_.set_xlim(0, 2 * np.pi)
        x_.set_ylim(rs - a.band, rs + a.band)
        x_.set_xlabel(r"$\theta^*$")
        x_.set_title(f"{lab}\nw = {w:.3f}", fontsize=9)
    ax[0, 0].set_ylabel(r"$\rho$")
    fig.suptitle(f"Poincare sections at phi = 0, eps {a.eps:g} A (red: lines trapped in the 2/1 island, blue: q = 2)")
    fig.tight_layout()
    path = os.path.join(ROOT, "figures", "dshape", "islands", a.out + ".png")
    fig.savefig(path, dpi=130)
    print("saved", path)


if __name__ == "__main__":
    main()
