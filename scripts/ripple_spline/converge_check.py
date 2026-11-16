"""Convergence and stationarity of runs from their live logs (item 1 of the follow-up):

    python scripts/ripple_spline/converge_check.py out/rs/h1/conv_*_live.tsv out/rs/fzc/conv_*_live.tsv [--md out.md]

Per run: trials, final M/M_DESC and M/eps, the relative change of M over the last 20% of the trials (a converged run
has ~0), |g|/|g0| at the end of the last outer (stationarity of the inner solve) and its minimum, the share of
noise-level steps and of boundary steps in the last 50 trials (noise-dominated = at the merit's noise floor), the outer
stop reasons, b21 at the end, and a verdict:
  converged   : |dM/M| over the last 20% < 0.5% and the last outers stopped as stationary or flat
  slow        : M still falling by 0.5-5% over the last 20%
  noise floor : >= 80% of the last 50 steps noise-level and M flat (stuck: e.g. a huge penalty swamping the merit)
  not converged otherwise.
"""
import argparse
import math
import os
import re


def num(v):
    try:
        return float(v)
    except (TypeError, ValueError):
        return math.nan


def read(path):
    rows, outers, hdr = [], [], None
    with open(path) as fh:
        for ln in fh:
            if ln.startswith("# outer"):
                outers.append(ln.strip())
            elif hdr is None:
                hdr = ln.rstrip("\n").split("\t")
            else:
                rows.append(dict(zip(hdr, ln.rstrip("\n").split("\t"))))
    return rows, outers


def check(path):
    rows, outers = read(path)
    if not rows:
        return None
    acc = [r for r in rows if r.get("acc") == "1"]
    M = [num(r.get("M/M_ref")) for r in acc]
    n = len(M)
    k = max(int(0.8 * n), 0)
    dM = (M[-1] - M[k]) / M[k] if n > 5 and M[k] else math.nan
    last = rows[-50:]
    stops = [re.search(r"stop (\S+)", o).group(1) for o in outers if re.search(r"stop (\S+)", o)]
    b21 = next((num(r["b21%"]) for r in reversed(rows) if math.isfinite(num(r.get("b21%")))), math.nan)
    g_last = num(rows[-1].get("|g|/|g0|"))
    g_min = min((num(r.get("|g|/|g0|")) for r in rows[-100:] if math.isfinite(num(r.get("|g|/|g0|")))), default=math.nan)
    noise = sum(r.get("kind") == "noise" for r in last) / len(last)
    bnd = sum(r.get("kind") == "boundary" for r in last) / len(last)
    tail_stops = stops[-3:]
    if noise >= 0.8 and math.isfinite(dM) and abs(dM) < 5e-3:
        verdict = "noise floor (steps below the merit's noise: stuck, not converged)"
    elif math.isfinite(dM) and abs(dM) < 5e-3 and any(s in ("stationary", "flat") for s in tail_stops):
        verdict = "converged"
    elif math.isfinite(dM) and -5e-2 < dM < 0:
        verdict = "slow"
    else:
        verdict = "not converged"
    return {"run": path, "trials": len(rows), "outers": len(outers), "M": M[-1] if M else math.nan,
            "M/eps": num(rows[-1].get("M/eps")), "dM/M last 20%": dM, "|g|/|g0| last": g_last,
            "|g|/|g0| min (last 100)": g_min, "noise share (last 50)": noise, "boundary share (last 50)": bnd,
            "stops": ",".join(stops[-5:]), "b21%": b21, "t_run [h]": num(rows[-1].get("t_run")) / 3600, "verdict": verdict}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("paths", nargs="+")
    ap.add_argument("--md", default=None)
    a = ap.parse_args()
    out = [c for c in (check(p) for p in a.paths) if c]
    cols = ["trials", "outers", "M", "M/eps", "dM/M last 20%", "|g|/|g0| last", "|g|/|g0| min (last 100)",
            "noise share (last 50)", "boundary share (last 50)", "b21%", "t_run [h]", "stops", "verdict"]
    L = ["| run | " + " | ".join(cols) + " |", "|" + "---|" * (len(cols) + 1)]
    for c in out:
        nm = os.path.relpath(c["run"]).replace("_live.tsv", "")
        cells = []
        for k in cols:
            v = c[k]
            cells.append(f"{v:.4g}" if isinstance(v, float) else str(v))
        L.append(f"| {nm} | " + " | ".join(cells) + " |")
    txt = "\n".join(L)
    print(txt)
    if a.md:
        with open(a.md, "w") as fh:
            fh.write(txt + "\n")


if __name__ == "__main__":
    main()
