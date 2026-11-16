"""Collect the ripple spline study into tables and figures (run on sai from the repository root, after p3_common):

    python scripts/ripple_spline/collect.py [--runs report_fz fzc h1] [--out out/rs/summary]

Sources (missing ones are skipped):
  out/rs/common/<run>.json          states on the common grid (a/120, the hybrid's radial collocation): band p16,
                                    volume mean, b21, force split (parallel |b.grad p| / perpendicular, by rho bin)
  out/rs/<run>/diag_m2.json         Poincare and pressure diagnostics (start_check_bext): pendulum width, traced spread,
                                    p' near q = 2
  out/rs/<run>/bs<beta>.json        the pinned-beta runs (trials, time, the run's own M, |J - J_fit|)
  out/rs/<run>/rel<beta>.json       the release tests (per-outer M and the run's own record)
  out/rs/<run>/cont.json            the shielded-start continuation
Writes <out>.md (tables), <out>.json and <out>_*.png.
"""
import argparse
import glob
import json
import os

import numpy as np

BETAS = ["0.02", "0.05", "0.1", "0.2", "0.35", "0.5", "0.75", "1.0"]
LABEL = {"report_fz": "FZ 32/32/6 (report states, a/60 runs)", "fzc": "FZ 32/32/6 control (a/100 runs)",
         "h1": "FZ 32/32/6 + band splines (a/100 runs)", "s1": "pure splines s1", "s2": "pure splines s2"}


def load(p):
    try:
        with open(p) as fh:
            return json.load(fh)
    except (OSError, ValueError):
        return None


def common(run):
    d = load(f"out/rs/common/{run}.json")
    if d is None:
        return {}
    out = {}
    for name, reps in (d.get("eval") or {}).items():
        rep = reps[-1] if isinstance(reps, list) and reps else {}
        out.setdefault(name, {})["M"] = rep.get("mean/M_desc")
        out[name]["band p16"] = rep.get("M_p/DESC")
    for name, v in (d.get("b_mk report") or {}).items():
        b = complex(*v)
        out.setdefault(name, {})["b21%"] = 100 * abs(b) / 3.778e-4
        out[name]["b21 phase"] = float(np.degrees(np.angle(b)))
    fs = d.get("force split") or {}
    for name, v in fs.items():
        if isinstance(v, dict) and "total" in v:
            out.setdefault(name, {}).update({"par": v["total"]["par"], "perp": v["total"]["perp"],
                                             "par by bin": v["par by bin"], "perp by bin": v["perp by bin"]})
    out["_edges"] = fs.get("edges")
    return out


def diag(run):
    d = load(f"out/rs/{run}/diag_m2.json")
    if d is None and run == "report_fz":
        d = load("results/ripple_landscape/diagnostics/bd_m2.json")
    if d is None:
        return {}
    out = {}
    for k, v in (d.get("perturbations") or {}).items():
        nm = k.split(":")[0]
        out.setdefault(nm, {}).update({"pendulum width": v.get("pendulum width"), "traced spread": v.get("max radial spread")})
    for k, v in (d.get("pressure") or {}).items():
        out.setdefault(k, {})["min |p'/p'_DESC| near q=2"] = v.get("min |p'/p'_DESC| within 0.06")
    return out


def runrec(run, stem):
    d = load(f"out/rs/{run}/{stem}.json")
    if d is None:
        return None
    al = (d.get("alphas") or [{}])[-1]
    trials = 0
    for a_ in d.get("alphas") or []:
        for o in a_.get("outer") or []:
            trials += int(o.get("inner its", o.get("its", 0)) or 0)
    live = f"out/rs/{run}/{stem}_live.tsv"
    n_live = sum(1 for ln in open(live) if not ln.startswith("#")) - 1 if os.path.exists(live) else None
    return {"M (run grid)": al.get("M/M_desc"), "|J-J_fit|": al.get("|J-J_fit|/rms"), "time": d.get("total [s]"),
            "trials": n_live}


def beta_rec(run, b):
    """The pinned-beta run plus its extension (bsx<beta>, p2_beta_ext.sh) when there is one: trials summed, the
    extension's final M."""
    r1, r2 = runrec(run, f"bs{b}"), runrec(run, f"bsx{b}")
    if r1 is None or r2 is None:
        return r1 or r2
    out = dict(r2)
    out["trials"] = (r1.get("trials") or 0) + (r2.get("trials") or 0)
    out["time"] = (r1.get("time") or 0) + (r2.get("time") or 0)
    out["M first run"] = r1.get("M (run grid)")
    return out


def fmt(v, f="{:.4f}"):
    return "-" if v is None else (f.format(v) if isinstance(v, (int, float)) else str(v))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--runs", nargs="+", default=["report_fz", "fzc", "h1"])
    ap.add_argument("--out", default="out/rs/summary")
    a = ap.parse_args()
    S = {r: {"common": common(r), "diag": diag(r), "beta runs": {b: beta_rec(r, b) for b in BETAS},
             "release": {b: runrec(r, f"rel{b}") for b in ("0.05", "0.5")}, "cont": runrec(r, "cont")} for r in a.runs}
    L = ["# Ripple landscape: FZ vs FZ + band splines\n",
         "All M in M_DESC on the common grid (field a/120, the hybrid's radial collocation) unless marked 'run grid'.\n"]
    # table 1: beta scan
    L.append("## Beta scan (relaxed M on the common grid; parallel / perpendicular force)\n")
    hdr = "| beta | " + " | ".join(f"{r}: M | {r}: par / perp | {r}: b21%" for r in a.runs) + " |"
    L.append(hdr)
    L.append("|" + "---|" * (1 + 3 * len(a.runs)))
    for name in ["start"] + [f"b{b}" for b in BETAS] + ["rel0.05", "rel0.5"]:
        row = [name]
        for r in a.runs:
            c = S[r]["common"].get(name, {})
            row += [fmt(c.get("M")), f"{fmt(c.get('par'))} / {fmt(c.get('perp'))}", fmt(c.get("b21%"), "{:.2f}")]
        L.append("| " + " | ".join(row) + " |")
    # table 2: islands and pressure
    L.append("\n## Islands and pressure flattening (run grid diagnostics)\n")
    L.append("| state | " + " | ".join(f"{r}: pendulum w | {r}: traced spread | {r}: min p'/p'_DESC" for r in a.runs) + " |")
    L.append("|" + "---|" * (1 + 3 * len(a.runs)))
    for name in ["vacuum", "start", "b0.05", "b0.2", "b0.5", "b1.0"]:
        row = [name]
        for r in a.runs:
            dg = S[r]["diag"].get(name, {})
            row += [fmt(dg.get("pendulum width")), fmt(dg.get("traced spread")), fmt(dg.get("min |p'/p'_DESC| near q=2"), "{:.3f}")]
        L.append("| " + " | ".join(row) + " |")
    # table 3: effort
    L.append("\n## Effort (pinned-beta runs, run grid)\n")
    L.append("| beta | " + " | ".join(f"{r}: trials | {r}: M run grid | {r}: abs(J-J_fit)" for r in a.runs if r != "report_fz") + " |")
    L.append("|" + "---|" * (1 + 3 * len([r for r in a.runs if r != "report_fz"])))
    for b in BETAS:
        row = [b]
        for r in a.runs:
            if r == "report_fz":
                continue
            br = S[r]["beta runs"].get(b) or {}
            row += [fmt(br.get("trials"), "{}"), fmt(br.get("M (run grid)")), fmt(br.get("|J-J_fit|"), "{:.4f}")]
        L.append("| " + " | ".join(row) + " |")
    with open(a.out + ".md", "w") as fh:
        fh.write("\n".join(L) + "\n")
    with open(a.out + ".json", "w") as fh:
        json.dump(S, fh, indent=1, default=float)
    # figures
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        fig, ax = plt.subplots(1, 3, figsize=(15, 4.2))
        for r in a.runs:
            c = S[r]["common"]
            xs = [0.0] + [float(b) for b in BETAS]
            nm = ["start"] + [f"b{b}" for b in BETAS]
            M = [c.get(n, {}).get("M") for n in nm]
            P = [c.get(n, {}).get("par") for n in nm]
            Q = [c.get(n, {}).get("perp") for n in nm]
            ok = [i for i, v in enumerate(M) if v is not None]
            if not ok:
                continue
            ax[0].plot([xs[i] for i in ok], [M[i] for i in ok], "o-", label=LABEL.get(r, r))
            ax[1].plot([xs[i] for i in ok], [P[i] for i in ok], "o-", label=LABEL.get(r, r))
            ax[2].plot([xs[i] for i in ok], [Q[i] for i in ok], "o-", label=LABEL.get(r, r))
        for a_, t in zip(ax, ("volume-mean |F| / M_DESC", "parallel |b.grad p| / M_DESC", "perpendicular / M_DESC")):
            a_.set_xlabel("pinned b21 / vacuum (beta; 0 = shielded start)")
            a_.set_title(t)
            a_.grid(alpha=0.3)
        ax[0].legend(fontsize=8)
        fig.tight_layout()
        fig.savefig(a.out + "_beta.png", dpi=130)
        edges = None
        fig, ax = plt.subplots(1, 2, figsize=(11, 4.2))
        for r in a.runs:
            c = S[r]["common"]
            edges = c.get("_edges") or edges
            for j, nm in enumerate(("start", "b1.0")):
                v = c.get(nm, {})
                if "par by bin" in v and edges:
                    mid = 0.5 * (np.array(edges[1:]) + np.array(edges[:-1]))
                    ax[j].plot(mid, v["par by bin"], "o-", label=f"{r} parallel")
                    ax[j].plot(mid, v["perp by bin"], "s--", label=f"{r} perpendicular")
        for j, nm in enumerate(("shielded start", "beta = 1")):
            ax[j].set_title(f"force density by rho bin, {nm}")
            ax[j].set_xlabel("rho")
            ax[j].set_yscale("log")
            ax[j].grid(alpha=0.3)
            ax[j].axvline(0.6375, color="k", lw=0.6, ls=":")
            ax[j].axvline(0.9131, color="k", lw=0.6, ls=":")
        ax[0].legend(fontsize=7)
        fig.tight_layout()
        fig.savefig(a.out + "_split.png", dpi=130)
    except Exception as e:  # figures are optional
        print("figures:", e)
    print("\n".join(L))


if __name__ == "__main__":
    main()
