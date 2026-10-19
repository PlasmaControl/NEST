"""Convergence with the annulus spacing h (and the basis) of the coils-only DSHAPE runs.

Reads, for each basis factor and h = a / frac:
  results/gs/<prefix>D_a<frac>_iter.json        the converged run (gs_dense_solve.py);
  results/gs/check_<prefix>D_a<frac>.json       its check (exact field, common grid);
  results/gs/fields_<prefix>D_a<frac>.npz       J, B, p of its end on that grid;
  results/gs/cyl_ladder*.json                   B_plasma(x0) against the exact field.
Draws figures/<name>.png:
  (a) |B_plasma - exact| at x0 against h;
  (b) <|F|>/norm of the converged end, run grid (dashed) and check (solid), DESC;
  (c) the change of the end state between successive h (B, J along/across B, p);
  (d) <|F|>/norm on the check by radial bin at the finest h of each basis, DESC.
The exact-G runs (B_plasma from the exact axisymmetric field, no grid: the h -> 0 limit;
results/gs/check_<run>.json, ``--exact``) are drawn as horizontal lines in (b), their
run-grid force dashed, and as curves in (d).
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


def load_fields(run):
    p = os.path.join(GS, f"fields_{run}.npz")
    return dict(np.load(p)) if os.path.exists(p) else None


def diff(X, Y, ref):
    w = ref["dV"] * ref["ok"]
    w = w / w.sum()
    B = np.nan_to_num(ref["desc_B"])
    bh = B / np.linalg.norm(B, axis=1)[:, None]
    rms = lambda v: np.sqrt(np.sum(w * (np.sum(v**2, 1) if v.ndim == 2 else v**2)))  # noqa: E731
    dJ = X["end_J"] - Y["end_J"]
    par = np.sum(dJ * bh, 1)
    return {"B": rms(X["end_B"] - Y["end_B"]) / rms(ref["desc_B"]),
            "J along": np.sqrt(np.sum(w * par**2)) / rms(ref["desc_J"]),
            "J across": rms(dJ - par[:, None] * bh) / rms(ref["desc_J"]),
            "p": rms(X["end_p"] - Y["end_p"]) / rms(ref["desc_p"])}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--fracs", type=int, nargs="+", default=[20, 40, 80, 160, 320, 640])
    ap.add_argument("--bases", nargs="+", default=["cyl:x2", "cyl3:x3"], help="prefix:label")
    ap.add_argument("--exact", nargs="+", default=["exD:x2", "ex3D2,ex3D:x3"],
                    help="exact-G runs as runs:label, the first existing of the comma-separated runs")
    ap.add_argument("--name", default="gs_cyl_convergence")
    a = ap.parse_args()
    fig, ax = plt.subplots(1, 4, figsize=(24, 5.2))
    # (a) B_plasma at x0
    rows = {}
    for fn in ("cyl_ladder_h.json", "cyl_ladder_fast.json"):
        p = os.path.join(GS, fn)
        if os.path.exists(p):
            for r in json.load(open(p))["rows"]:
                if r.get("flux") in (None,) and r.get("order", 4) == 4:
                    rows[r["frac"]] = r
    fr = sorted(rows)
    ax[0].loglog([1 / f for f in fr], [rows[f]["B_plasma err"] for f in fr], "ko-", label="annulus (x2 start)")
    for f in fr:
        ax[0].annotate(f"{rows[f]['build s']:.0f} s", (1 / f, rows[f]["B_plasma err"]), fontsize=7,
                       xytext=(3, 3), textcoords="offset points")
    ax[0].set_title("|B_plasma - exact| / rms B at x0 (labels: build time)", fontsize=10)
    desc_F, desc_bins = None, None
    print(f"{'basis':>6s} {'h':>6s} {'F run':>9s} {'F check':>9s} " + " ".join(f"{'step ' + q:>14s}" for q in QS))
    for (pref, lab), col in zip([b.split(":") for b in a.bases], ("C0", "C3", "C2")):
        hs, Fr, Fc, steps, prev, finest = [], [], [], [], None, None
        for f in a.fracs:
            run = f"{pref}D_a{f}"
            pj = os.path.join(GS, f"{run}_iter.json")
            pc = os.path.join(GS, f"check_{run}.json")
            if not (os.path.exists(pj) and os.path.exists(pc)):
                continue
            hist = json.load(open(pj))["history"]
            ch = json.load(open(pc))["force"]
            dk = [k for k in ch if k.startswith("DESC")][0]
            desc_F, desc_bins = ch[dk]["mean"], ch[dk]["bins"]
            hs.append(1 / f)
            Fr.append(hist[-1]["<|F|>/norm"])
            Fc.append(ch["NEST end x"]["mean"])
            finest = (f, ch["NEST end x"]["bins"])
            fl = load_fields(run)
            st = diff(fl, prev, fl) if (fl is not None and prev is not None) else None
            steps.append(st)
            print(f"{lab:>6s} a/{f:<4d} {Fr[-1]:9.2e} {Fc[-1]:9.2e} "
                  + (" ".join(f"{st[q]:14.2e}" for q in QS) if st else ""))
            prev = fl
        if not hs:
            continue
        ax[1].loglog(hs, Fc, "o-", color=col, label=f"{lab}: check (exact field)")
        ax[1].loglog(hs, Fr, "s--", color=col, lw=0.8, label=f"{lab}: run grid")
        for q, ls in zip(QS, ("-", "--", "-.", ":")):
            pts = [(h, s[q]) for h, s in zip(hs, steps) if s]
            if pts:
                ax[2].loglog(*zip(*pts), ls, color=col, marker="o", ms=3, label=f"{lab}: {q}")
        if finest:
            ax[3].semilogy(range(len(finest[1])), list(finest[1].values()), "o-", color=col,
                           label=f"{lab}, a/{finest[0]}")
    for spec, col in zip(a.exact, ("C0", "C3", "C2")):
        runs, lab = spec.split(":")
        run = next((r for r in runs.split(",") if os.path.exists(os.path.join(GS, f"check_{r}.json"))), None)
        if run is None:
            continue
        ch = json.load(open(os.path.join(GS, f"check_{run}.json")))["force"]
        Fr = json.load(open(os.path.join(GS, f"{run}_iter.json")))["history"][-1]["<|F|>/norm"]
        Fc = ch["NEST end x"]["mean"]
        print(f"{lab:>6s} exact G ({run}): F run {Fr:9.2e}, F check {Fc:9.2e}")
        ax[1].axhline(Fc, color=col, lw=2.2, alpha=0.6, label=f"{lab}: exact G, check ({Fc:.2e})")
        ax[1].axhline(Fr, color=col, lw=1.0, ls="--", alpha=0.6, label=f"{lab}: exact G, run grid")
        b = ch["NEST end x"]["bins"]
        ax[3].semilogy(range(len(b)), list(b.values()), "s--", color=col, lw=2, alpha=0.7,
                       label=f"{lab}, exact G")
    if desc_F:
        ax[1].axhline(desc_F, color="k", ls=":", label="DESC L32 free boundary")
        ax[3].semilogy(range(len(desc_bins)), list(desc_bins.values()), "ko-", label="DESC")
        ax[3].set_xticks(range(len(desc_bins)))
        ax[3].set_xticklabels(list(desc_bins), fontsize=8)
    ax[1].set_title(r"$\langle|F|\rangle$/norm of the converged end", fontsize=10)
    ax[2].set_title("change of the end state from the previous h (rms / rms)", fontsize=10)
    ax[3].set_title(r"$\langle|F|\rangle$/norm by $\rho$ bin, finest h (check)", fontsize=10)
    for x in ax[:3]:
        x.set_xlabel("h / a")
    for x in ax:
        x.legend(fontsize=7)
    fig.suptitle("DSHAPE L = M = 32 free boundary, coils' field: cylindrical-annulus B_plasma (N_phi = 1, "
                 "no correction) against h, and the exact-G limit (lines); exact LM, edge power 3, "
                 "w_J 1e-3 at rho < 0.9")
    fig.tight_layout()
    out = os.path.join(ROOT, "figures", f"{a.name}.png")
    fig.savefig(out, dpi=130)
    print("saved", out)


if __name__ == "__main__":
    main()
