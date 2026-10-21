"""Zoom on the pressure through the 2/1 island's O- and X-points (from isl_islands.py's
results/isl/<name>_islands.json): p, p minus a straight line (the X-point profile's linear fit
over the window, so that a flat spot shows up as a bump), and dp/drho, against rho; the shaded
band is q = 2 +- w/2 (w the Poincare width).

run: python scripts/dshape/isl_pzoom.py --names a120T4q a120T6wq --out isl_pzoom
"""

import argparse
import ast
import json
import os

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
FIG = os.path.join(ROOT, "figures", "dshape", "islands")
os.makedirs(FIG, exist_ok=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--names", nargs="+", required=True)
    ap.add_argument("--labels", nargs="*", default=None)
    ap.add_argument("--half", type=float, default=0.07, help="half width of the window in rho")
    ap.add_argument("--out", default="isl_pzoom")
    a = ap.parse_args()
    labels = a.labels or a.names
    runs = [json.load(open(os.path.join(ROOT, "results", "isl", n + "_islands.json"))) for n in a.names]
    eps = [e for e in runs[0]["eps"] if str(e) in runs[0]["p_profiles"]]
    fig, ax = plt.subplots(3 * len(runs), len(eps), figsize=(5 * len(eps), 3.4 * 3 * len(runs)), squeeze=False)
    for i, (d, lab) in enumerate(zip(runs, labels)):
        r = np.asarray(d["rho_profiles"])
        rs = d["surfaces"]["2"][0]
        m = np.abs(r - rs) <= a.half
        for j, e in enumerate(eps):
            p = d["p_profiles"][str(e)]
            p = ast.literal_eval(p) if isinstance(p, str) else p
            O, X = np.asarray(p["O"]), np.asarray(p["X"])
            w = d["w_poincare"]["NEST"][d["eps"].index(e)]
            line = np.polyval(np.polyfit(r[m], X[m], 1), r)
            gO, gX = np.gradient(O, r), np.gradient(X, r)
            for k, (yO, yX, ylab) in enumerate(((O, X, "p (Pa)"), (O - line, X - line, "p - straight line (Pa)"),
                                                 (gO, gX, r"dp/d$\rho$ (Pa)"))):
                x_ = ax[3 * i + k, j]
                x_.plot(r[m], yO[m], "-", color="C3", lw=2, label="through the O-point")
                x_.plot(r[m], yX[m], "--", color="k", lw=1.5, label="through the X-point")
                x_.axvspan(rs - w / 2, rs + w / 2, color="C3", alpha=0.12, label=f"q = 2 $\\pm$ w/2 (w = {w:.3f})")
                x_.axvline(rs, color="gray", lw=0.5)
                x_.set_ylabel(ylab)
                x_.set_xlabel(r"$\rho$")
                if k == 0:
                    x_.set_title(f"{lab}, eps {e:g} A")
                if k == 2:
                    sel = np.abs(r - rs) < max(w / 2, 0.005)
                    x_.set_title(f"in the island: dp/drho O / X = {gO[sel].mean() / gX[sel].mean():.2f}", fontsize=9)
                if j == 0:
                    x_.legend(fontsize=7)
    fig.suptitle("Pressure through the 2/1 island (phi = 0): does it flatten?")
    fig.tight_layout()
    path = os.path.join(FIG, a.out + ".png")
    fig.savefig(path, dpi=110)
    print(f"saved {path}")


if __name__ == "__main__":
    main()
