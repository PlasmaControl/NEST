"""Trust-region history of mean-force runs (``mf_fixed.py`` / ``mf_fixed_qr.py`` JSON records).

Every trial step (accepted or rejected) is one point, numbered across outer iterations and
bounds alpha. Panels, sharing the trial index:

  1. the trust radius Delta before the trial (line) and the trial step |y| (dots: filled =
     accepted, open = rejected); both in the run's metric units, log scale
  2. ratio = actual / predicted decrease (0.25 and 0.75: the radius rules)
  3. Gauss-Newton share (fraction of the merit the full GN step would remove)
  4. M / eps (the bound is met at 1)
  5. Lanczos iterations (GLTR) or QR time [s]

Dashed verticals: outer iterations; solid: a new alpha. Several runs overlay, one color each
(e.g. GLTR against QR on the same ladder). Also writes <png>.csv: one row per trial.

  python scripts/meanforce/tr_history.py out/gltr.json out/qr.json --labels GLTR QR -o fig.png
"""

import argparse
import csv
import json

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

SERIES = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4", "#008300", "#6250d6"]
INK, MUTED, GRID = "#0b0b0b", "#52514e", "#e4e3df"


def trials(run):
    """Flat list of trial dicts with alpha / outer / index tags (includes the outer in progress)."""
    out = []
    for ia, e in enumerate(run.get("alphas", [])):
        outers = [o["inner"] for o in e.get("outer", [])]
        if e.get("inner (current outer)"):
            outers.append(e["inner (current outer)"])
        for io, hist in enumerate(outers):
            for h in hist:
                out.append(dict(h, alpha=e["alpha"], i_alpha=ia, i_outer=io))
    for k, t in enumerate(out):
        t["k"] = k
    return out


def plot(runs, labels, path):
    fig, ax = plt.subplots(5, 1, figsize=(11, 12), sharex=True, gridspec_kw={"hspace": 0.08})
    for i, (tr, lab) in enumerate(zip(runs, labels)):
        col = SERIES[i % len(SERIES)]
        if not tr:
            continue
        k = np.array([t["k"] for t in tr])
        acc = np.array([bool(t["accepted"]) for t in tr])
        D = np.array([t["Delta"] for t in tr])
        y = np.array([t["|y|"] for t in tr])
        ax[0].step(k, D, where="mid", color=col, lw=1.5, label=f"{lab}: Delta")
        ax[0].plot(k[acc], y[acc], "o", ms=4, color=col, label=f"{lab}: |y| accepted")
        ax[0].plot(k[~acc], y[~acc], "o", ms=5, mfc="none", mec=col, mew=1.2, label=f"{lab}: |y| rejected")
        ax[1].plot(k, np.clip([t["ratio"] for t in tr], -1, 2), ".-", color=col, lw=1, ms=4, label=lab)
        ax[2].semilogy(k, np.maximum([t["GN share"] for t in tr], 1e-12), ".-", color=col, lw=1, ms=4, label=lab)
        ax[3].plot(k, [1 + t["c"][0] for t in tr], ".-", color=col, lw=1, ms=4, label=lab)
        if "lanczos" in tr[0]:
            ax[4].plot(k, [t.get("lanczos", np.nan) for t in tr], ".-", color=col, lw=1, ms=4, label=f"{lab} (Lanczos)")
        else:
            ax[4].plot(k, [t.get("t_qr", t.get("time", np.nan)) for t in tr], ".-", color=col, lw=1, ms=4,
                       label=f"{lab} (QR s)")
        for j in range(1, len(tr)):
            if tr[j]["i_alpha"] != tr[j - 1]["i_alpha"]:
                for a_ in ax:
                    a_.axvline(k[j] - 0.5, color=MUTED, lw=1.0)
                if i == 0:
                    ax[0].annotate(f"alpha {tr[j]['alpha']:g}", (k[j], 1), xycoords=("data", "axes fraction"),
                                   fontsize=8, color=MUTED, va="top", ha="left")
            elif tr[j]["i_outer"] != tr[j - 1]["i_outer"]:
                for a_ in ax:
                    a_.axvline(k[j] - 0.5, color=GRID, lw=1.0, ls="--")
    ax[0].set_yscale("log")
    ax[0].set_ylabel("Delta, |y| (metric)")
    for v in (0.25, 0.75):
        ax[1].axhline(v, color=MUTED, lw=0.8, ls=":")
    ax[1].set_ylabel("ratio act/pred")
    ax[2].set_ylabel("GN share")
    ax[3].axhline(1.0, color=MUTED, lw=0.8, ls=":")
    ax[3].set_ylabel("M / eps")
    ax[4].set_ylabel("Lanczos / QR s")
    ax[4].set_xlabel("trial step (accepted and rejected, all outers and alphas)")
    for a_ in ax:
        a_.grid(True, color=GRID, lw=0.6)
        for s_ in ("top", "right"):
            a_.spines[s_].set_visible(False)
        a_.tick_params(colors=MUTED, labelsize=8)
        a_.yaxis.label.set_color(INK)
    ax[0].legend(fontsize=7, ncol=max(1, len(runs)), frameon=False, loc="lower left")
    if len(runs) > 1:
        ax[1].legend(fontsize=7, frameon=False)
    fig.savefig(path, dpi=140, bbox_inches="tight")
    print(f"wrote {path}")


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("runs", nargs="+", help="run JSON records")
    ap.add_argument("--labels", nargs="+", default=None)
    ap.add_argument("-o", "--out", default="tr_history.png")
    a = ap.parse_args()
    labels = a.labels or [r.rsplit("/", 1)[-1].replace(".json", "") for r in a.runs]
    runs = [trials(json.load(open(r))) for r in a.runs]
    plot(runs, labels, a.out)
    keys = ["run", "k", "alpha", "i_outer", "accepted", "Delta", "Delta_next", "|y|", "hit", "ratio", "pred",
            "f", "f_new", "GN share", "|grad|", "M", "lanczos", "mu", "t_qr", "time"]
    with open(a.out.rsplit(".", 1)[0] + ".csv", "w", newline="") as fh:
        w = csv.writer(fh)
        w.writerow(keys)
        for lab, tr in zip(labels, runs):
            for t in tr:
                w.writerow([lab] + [t.get(k_, "") for k_ in keys[1:]])
    print(f"wrote {a.out.rsplit('.', 1)[0]}.csv ({sum(map(len, runs))} trials)")


if __name__ == "__main__":
    main()
