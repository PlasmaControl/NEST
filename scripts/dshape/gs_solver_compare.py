"""Plain LM (optimize_qa.py: truncated PCG) against exact steps (gs_dense_solve.py: QR of
[A | r], SVD of R, damping scanned) on the same problem and from the same x0.

Reads both runs' histories (<|F|>/norm on the run grid and seconds per iteration), the
checks (gs_check_end.py --accurate --save on one common grid: exact B_plasma there) and
their fields there, and prints:
  - iterations, solve time (the dense run's one-time Jacobian tables added, --tables),
    time to come within 10 %, 1 % and 0.1 % of the lower final <|F|>;
  - final <|F|>/norm on the run grid and on the check;
  - the difference of the two end states (B, J along and across B, p; rms over rms).
Draws figures/<name>.png: <|F|> against iteration and against solve time.

usage: python scripts/dshape/gs_solver_compare.py --lm results/gs/cylL_a160 --dense results/gs/cylD_a160
    [--tables 31] [--name gs_solver_compare_a160]
"""

import argparse
import json
import os

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
GS = os.path.join(ROOT, "results", "gs")


def load(run):
    h = json.load(open(run + "_iter.json"))["history"]
    F = np.array([e["<|F|>/norm"] for e in h])
    t = np.array([e.get("time", 0.0) for e in h])
    t[0] = 0.0
    c = json.load(open(os.path.join(GS, f"check_{os.path.basename(run)}.json")))["force"]
    fl = dict(np.load(os.path.join(GS, f"fields_{os.path.basename(run)}.npz")))
    return F, np.cumsum(t), c, fl


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--lm", required=True)
    ap.add_argument("--dense", required=True)
    ap.add_argument("--tables", type=float, default=31.0, help="the dense run's one-time Jacobian tables [s]")
    ap.add_argument("--name", default="gs_solver_compare_a160")
    a = ap.parse_args()
    FL, tL, cL, fL = load(a.lm)
    FD, tD, cD, fD = load(a.dense)
    tD = tD + a.tables
    tD[0] = 0.0
    F_min = min(FL.min(), FD.min())
    print(f"{'':28s} {'plain LM (trunc. CG)':>22s} {'exact steps (QR/SVD)':>22s}")
    print(f"{'iterations':28s} {len(FL) - 1:22d} {len(FD) - 1:22d}")
    print(f"{'solve time [s]':28s} {tL[-1]:22.0f} {tD[-1]:22.0f}")
    print(f"{'s per iteration (median)':28s} {np.median(np.diff(tL)):22.2f} {np.median(np.diff(tD[1:])):22.2f}")
    for tol in (0.1, 0.01, 0.001):
        def first(F, t):
            k = np.flatnonzero(F <= F_min * (1 + tol))
            return (f"it {k[0]:5d}, {t[k[0]]:7.0f} s" if len(k) else "never")
        print(f"{'within ' + f'{100 * tol:g} % of min F':28s} {first(FL, tL):>22s} {first(FD, tD):>22s}")
    print(f"{'<|F|>/norm, run grid':28s} {FL[-1]:22.4e} {FD[-1]:22.4e}")
    print(f"{'<|F|>/norm, check (exact B)':28s} {cL['NEST end x']['mean']:22.4e} {cD['NEST end x']['mean']:22.4e}")
    print(f"{'  rho 0.95-1':28s} {cL['NEST end x']['bins']['0.95-1']:22.4e} {cD['NEST end x']['bins']['0.95-1']:22.4e}")
    dk = [k for k in cD if k.startswith("DESC")][0]
    print(f"{'DESC, check':28s} {cD[dk]['mean']:22.4e}")
    # the two end states on the common grid
    assert np.allclose(fL["x"], fD["x"]), "not the same check grid"
    w = fD["dV"] * fD["ok"]
    w = w / w.sum()
    Bref = np.nan_to_num(fD["desc_B"])
    bh = Bref / np.linalg.norm(Bref, axis=1)[:, None]
    rv = lambda v: np.sqrt(np.sum(w * np.sum(v**2, 1)))  # noqa: E731
    rs = lambda v: np.sqrt(np.sum(w * v**2))  # noqa: E731
    dJ = fL["end_J"] - fD["end_J"]
    par = np.sum(dJ * bh, 1)
    diff = {"B": rv(fL["end_B"] - fD["end_B"]) / rv(fD["desc_B"]),
            "J along": rs(par) / rv(fD["desc_J"]),
            "J across": rv(dJ - par[:, None] * bh) / rv(fD["desc_J"]),
            "p": rs(fL["end_p"] - fD["end_p"]) / rs(fD["desc_p"])}
    print("end LM - end exact steps: " + "  ".join(f"{k} {v:.2e}" for k, v in diff.items()))
    fig, ax = plt.subplots(1, 2, figsize=(14, 5))
    for F, t, lab, c in ((FL, tL, "plain LM, truncated CG", "k"), (FD, tD, "exact steps (QR / SVD)", "C3")):
        ax[0].semilogy(np.arange(len(F)), F, color=c, marker="." if c != "k" else None, label=lab)
        ax[1].semilogy(t, F, color=c, marker="." if c != "k" else None, label=lab)
    for x_ in ax:
        x_.axhline(cD[dk]["mean"], color="gray", ls="--", lw=0.8, label="DESC (check)")
        x_.legend(fontsize=8)
        x_.set_ylabel(r"$\langle|F|\rangle$/norm, run grid")
    ax[0].set_xscale("symlog", linthresh=10)
    ax[0].set_xlabel("iteration")
    ax[1].set_xscale("symlog", linthresh=10)
    ax[1].set_xlabel("solve time [s] (exact steps: Jacobian tables included)")
    fig.suptitle("DSHAPE L32 free boundary, coils + cylindrical B_plasma a/160 (no correction), x2: "
                 "the same problem and start, two solvers")
    fig.tight_layout()
    out = os.path.join(ROOT, "figures", "dshape", f"{a.name}.png")
    os.makedirs(os.path.dirname(out), exist_ok=True)
    fig.savefig(out, dpi=140)
    print("saved", out)


if __name__ == "__main__":
    main()
