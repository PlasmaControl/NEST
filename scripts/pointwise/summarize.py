"""Markdown tables of pointwise-bound runs (``al_run.py`` JSONs, continuations merged).

    python scripts/pointwise/summarize.py "label=run.json+fine.json+deep.json" ...

Per run: the smallest alpha met and the first not met after it, and at chosen alphas
the distance |J - J_fit|, its parts along and across B, the change of p, the force
(largest, rms, mean) and the work (trust-region steps, time).
"""

import json
import os
import sys

ALPHAS = (2.0, 1.0, 0.5, 0.3, 0.2)


def merged(spec):
    lab, ps = spec.split("=", 1)
    by = {}
    for p in ps.split("+"):
        if not os.path.exists(p):
            continue
        for e in json.load(open(p))["alphas"]:
            m = e["metrics"]
            al = m["alpha"]
            if al not in by or (m["feasible"] and not by[al]["feasible"]):
                by[al] = m
    return lab, [by[k] for k in sorted(by, reverse=True)]


def main():
    runs = [merged(s) for s in sys.argv[1:]]
    print("| run | smallest alpha met | next tried, not met | \\|J − J_fit\\| there | max / rms / mean \\|F\\| there | "
          "TR steps (all alphas) | time [s] |")
    print("|---|---|---|---|---|---|---|")
    for lab, ms in runs:
        ok = [m for m in ms if m["feasible"]]
        best = min(ok, key=lambda m: m["alpha"]) if ok else None
        bad = [m["alpha"] for m in ms if not m["feasible"] and best and m["alpha"] < best["alpha"]]
        nb = f"{max(bad):.4g}" if bad else "-"
        steps = sum(m["tr steps"] for m in ms)
        t = sum(m["time"] for m in ms)
        if best:
            print(f"| {lab} | {best['alpha']:.4g} | {nb} | {best['|J-J0|']:.3e} | {best['max|F|']:.2e} / "
                  f"{best['rms|F|']:.2e} / {best['<|F|>']:.2e} | {steps} | {t:.0f} |")
    print()
    print("| run | alpha | \\|J − J_fit\\| | along B | across B | p − p0 | rms \\|F\\| | ⟨\\|F\\|⟩ | B·∇p rel. | TR steps |")
    print("|---|---|---|---|---|---|---|---|---|---|")
    for lab, ms in runs:
        for m in ms:
            if m["feasible"] and any(abs(m["alpha"] - a) < 1e-9 for a in ALPHAS):
                print(f"| {lab} | {m['alpha']:g} | {m['|J-J0|']:.3e} | {m['J-J0 along B']:.2e} | "
                      f"{m['J-J0 across B']:.2e} | {m['p-p0']:.2e} | {m['rms|F|']:.2e} | {m['<|F|>']:.2e} | "
                      f"{m['<|B.grad p|>/<|B||grad p|>']:.2e} | {m['tr steps']} |")


if __name__ == "__main__":
    main()
