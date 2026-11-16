"""Follow a running mf_fixed job from its <out>_live.tsv (one line per trust-region trial).

Prints the last trials as a table and a summary of the recent window:
  eta = actual / predicted decrease (the trust-region agreement ratio; ~1: the model is right; < 0.25: the radius
  shrinks), |s|/Delta (1: the step sits on the trust-region boundary), |g|/|g0| (stationarity within the outer
  iteration), Lanczos count / cap and the GLTR residual res against its tolerance, the Ritz values of the
  metric-scaled model Hessian (theta_min, theta_max, cond, negatives), M/eps.

    python scripts/meanforce/tr_live.py out/cmp/A_live.tsv            # once
    python scripts/meanforce/tr_live.py out/cmp/A_live.tsv -f -n 25   # refresh every 30 s
    python scripts/meanforce/tr_live.py out/rs/                       # every *_live.tsv there, one line each

The warnings of <out>_status.json (written by the job) are printed under the table.
"""

import argparse
import glob
import json
import math
import os
import time

SHOW = [("alpha", "{:>5}"), ("outer", "{:>3}"), ("it", "{:>3}"), ("acc", "{:>3}"), ("kind", "{:>8}"),
        ("eta", "{:>6.2f}"), ("pred", "{:>8.1e}"), ("actual", "{:>8.1e}"), ("sigma_f", "{:>8.1e}"),
        ("|s|/Delta", "{:>5.2f}"), ("Delta", "{:>8.1e}"), ("Delta_next", "{:>8.1e}"),
        ("|g|", "{:>8.1e}"), ("|g|/|g0|", "{:>7.1e}"), ("lanczos", "{:>4}"), ("cap", "{:>3}"), ("res", "{:>7.1e}"),
        ("theta_min", "{:>8.1e}"), ("theta_max", "{:>8.1e}"), ("n_neg", "{:>3}"),
        ("M/eps", "{:>7.4f}"), ("M/M_ref", "{:>7.4f}"), ("b21%", "{:>6.2f}"), ("t_trial", "{:>6.0f}")]
HEAD = {"|s|/Delta": "s/D", "Delta_next": "D_next", "sigma_f": "sig_f", "|g|/|g0|": "g/g0", "lanczos": "lz", "theta_min": "th_min", "theta_max": "th_max",
        "n_neg": "neg", "t_trial": "t[s]"}


def num(v):
    try:
        return float(v)
    except (TypeError, ValueError):
        return math.nan


def width(f):
    return len(f.format(1.0)) if ("e}" in f or "f}" in f) else len(f.format(""))


def fmt(k, f, v):
    if "e}" in f or "f}" in f:
        x = num(v)
        return f.format(x) if math.isfinite(x) else "-".rjust(width(f))
    return f.format(v)


def read(path):
    rows, outers, hdr = [], [], None
    with open(path) as fh:
        for ln in fh:
            if ln.startswith("# outer"):
                outers.append(ln.strip()[2:])
            elif hdr is None:
                hdr = ln.rstrip("\n").split("\t")
            else:
                rows.append(dict(zip(hdr, ln.rstrip("\n").split("\t"))))
    return rows, outers


def status(path):
    sp = path.replace("_live.tsv", "") + "_status.json"
    try:
        with open(sp) as fh:
            return json.load(fh)
    except (OSError, ValueError):
        return None


def overview(paths, root="."):
    """One line per run: last stage, M/eps, M/M_ref, Delta, accept rate, warnings."""
    print(f"{'run':28s} {'age':>5s} {'trials':>6s} {'alpha':>5s} {'out':>3s} {'M/eps':>7s} {'M/Mref':>7s} "
          f"{'b21%':>6s} {'Delta':>8s} {'acc%':>4s} {'eta':>5s} warnings")
    for p in sorted(paths):
        rows, _ = read(p)
        if not rows:
            continue
        r, st = rows[-1], status(p) or {}
        acc = st.get("accept rate")
        b = next((num(x["b21%"]) for x in reversed(rows) if math.isfinite(num(x.get("b21%")))), math.nan)
        me = st.get("median eta")
        nm = os.path.relpath(p, root).replace("_live.tsv", "")
        print(f"{nm[-28:]:28s} {time.time() - os.path.getmtime(p):5.0f} "
              f"{len(rows):6d} {r['alpha']:>5s} {r['outer']:>3s} {num(r['M/eps']):7.4f} {num(r.get('M/M_ref')):7.4f} "
              f"{b:6.2f} {num(r['Delta']):8.1e} {100 * acc if acc is not None else math.nan:4.0f} "
              f"{me if me is not None else math.nan:5.2f} {'; '.join(st.get('warnings', []))}")


def show(path, n):
    rows, outers = [], []
    with open(path) as fh:
        lines = fh.readlines()
    hdr = None
    for ln in lines:
        if ln.startswith("# outer"):
            outers.append(ln.strip()[2:])
            continue
        if hdr is None:
            hdr = ln.rstrip("\n").split("\t")
            continue
        rows.append(dict(zip(hdr, ln.rstrip("\n").split("\t"))))
    print(f"{path}: {len(rows)} trials, {len(outers)} outers done; "
          f"last update {time.time() - os.path.getmtime(path):.0f} s ago")
    for o in outers[-3:]:
        print("  " + o)
    print(" ".join(HEAD.get(k, k).rjust(width(f)) for k, f in SHOW))
    for r in rows[-n:]:
        print(" ".join(fmt(k, f, r.get(k, "")) for k, f in SHOW)
              + (f"  [rebuild: {r['rebuild']}]" if r.get("rebuild") else ""))
    w = rows[-n:]
    if w:
        eta = [num(r["eta"]) for r in w]
        cap = sum(num(r["cap"]) > 0 for r in w)
        on = sum(num(r["|s|/Delta"]) > 0.99 for r in w)
        rej = sum(num(r["acc"]) == 0 for r in w)
        conv = sum(num(r["conv"]) > 0 for r in w)
        print(f"window of {len(w)}: eta median {sorted(eta)[len(eta) // 2]:.2f} (rejected {rej}); on the boundary "
              f"{on}; Lanczos capped {cap}, converged {conv}; last res {num(w[-1]['res']):.1e} vs tol "
              f"{num(w[-1]['tol']):.0e}; mean trial {sum(num(r['t_trial']) for r in w) / len(w):.0f} s")
    st = status(path)
    if st:
        print(f"status: {st.get('s since last accept', 0):.0f} s since the last accepted step; consecutive "
              f"rejections {st.get('consecutive rejections')}; noise-accept share {st.get('noise-accept share', 0):.2f}; "
              f"Delta decades over window {st.get('Delta decades over window')}")
        for w_ in st.get("warnings", []):
            print("  WARNING: " + w_)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("path")
    ap.add_argument("-n", type=int, default=20)
    ap.add_argument("-f", "--follow", action="store_true")
    ap.add_argument("--every", type=float, default=30.0)
    a = ap.parse_args()
    while True:
        if a.follow:
            print("\033[2J\033[H", end="")
        if os.path.isdir(a.path):
            overview(glob.glob(os.path.join(a.path, "**", "*_live.tsv"), recursive=True), a.path)
        else:
            show(a.path, a.n)
        if not a.follow:
            break
        time.sleep(a.every)


if __name__ == "__main__":
    main()
