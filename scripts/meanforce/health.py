"""Optimization health of mean-force runs (``mf_run.py`` JSONs): one figure per run and a
Markdown table comparing them (plan section 12, the check-in after step 2).

    python scripts/meanforce/health.py label=run.json ... --out figures/meanforce/health

Per run: trust radius and step, actual/predicted ratio, Gauss-Newton share, mu, the
three constraints and their multipliers and the penalty, the Galerkin identity, the time
per step (Lanczos iterations), the re-anchorings (displacement, jump of M), iota per anchor.
"""

import argparse
import json
import os

import matplotlib
import numpy as np

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402


def steps_of(run):
    """All trust-region steps of a run in order, with the outer index and alpha."""
    out = []
    for e in run["alphas"]:
        for o in e.get("outer", []):
            for h in o["inner"]:
                out.append(dict(h, outer=o["outer"], alpha=e["alpha"]))
        for h in e.get("inner (current outer)", []):
            out.append(dict(h, outer=len(e.get("outer", [])), alpha=e["alpha"]))
    return out


def summary(label, run):
    st = steps_of(run)
    if not st:
        return None
    rat = np.array([h["ratio"] for h in st])
    acc = np.array([h["accepted"] for h in st])
    hit = np.array([h["hit"] for h in st])
    mu0 = np.array([h.get("mu", 0.0) == 0.0 for h in st])
    gn = np.array([h["GN share"] for h in st])
    t = np.array([h["time"] for h in st])
    e = run["alphas"][-1]
    last = st[-1]
    lan = [h["lanczos"] for h in st if "lanczos" in h]
    return {
        "run": label, "steps": len(st), "accepted": int(acc.sum()), "rejected": int((~acc).sum()),
        "ratio > 0.75": float(np.mean(rat > 0.75)), "ratio < 0.25": float(np.mean(rat < 0.25)),
        "on boundary": float(np.mean(hit)), "mu = 0 (GN fits)": float(np.mean(mu0)),
        "GN share median / last": (float(np.median(gn)), float(gn[-1])),
        "Delta min / last": (float(min(h["Delta"] for h in st)), float(last["Delta_next"])),
        "rho last": float(last["rho"]), "nu last": last["nu"],
        "M/eps last": 1 + last["c"][0], "Bn/beta last": float(np.sqrt(max(1 + last["c"][1], 0))),
        "Psi/Psi_t - 1 last": last["c"][2], "galerkin max": float(max(h["galerkin"] for h in st)),
        "anchors": len(run.get("anchors", [])), "stalls": len(e.get("stalls", [])),
        "time/step [s]": float(np.median(t)), "Lanczos median": float(np.median(lan)) if lan else None,
        "|J-J_fit|/rms": e.get("|J-J_fit|/rms"), "M/M_desc": e.get("M/M_desc"),
        "Bn/|B| rms": e.get("Bn/|B| rms"),
        "Bn direct": (e.get("B.n direct") or {}).get("Bn/|B| rms"),
    }


def figure(label, run, out):
    st = steps_of(run)
    if not st:
        return
    k = np.arange(len(st))
    acc = np.array([h["accepted"] for h in st])
    fig, ax = plt.subplots(3, 3, figsize=(15, 11))
    a = ax[0, 0]
    a.semilogy(k, [h["Delta"] for h in st], "-", label="Delta")
    a.semilogy(k, [h["|y|"] for h in st], ".", label="|y|")
    a.set_title("trust radius and step (metric)")
    a.legend()
    a = ax[0, 1]
    rat = np.clip([h["ratio"] for h in st], -1, 2)
    a.scatter(k[acc], rat[acc], s=10, c="C0", label="accepted")
    a.scatter(k[~acc], rat[~acc], s=10, c="C3", label="rejected")
    for y in (0.25, 0.75):
        a.axhline(y, color="0.6", lw=0.8)
    a.set_title("actual / predicted (clipped to [-1, 2])")
    a.legend()
    a = ax[0, 2]
    a.plot(k, [h["GN share"] for h in st], ".-")
    a.set_ylim(0, 1.02)
    a.set_title("Gauss-Newton share")
    a = ax[1, 0]
    a.semilogy(k, [max(h.get("mu", 0.0), 1e-30) for h in st], ".")
    a.set_title("mu (1e-30: Gauss-Newton step fits)")
    a = ax[1, 1]
    a.plot(k, [1 + h["c"][0] for h in st], label="M / eps")
    a.plot(k, [np.sqrt(max(1 + h["c"][1], 0)) for h in st], label="B.n rms / beta")
    a.plot(k, [1 + 100 * h["c"][2] for h in st], label="1 + 100 (Psi / Psi_t - 1)")
    a.axhline(1, color="0.6", lw=0.8)
    a.set_yscale("log")
    a.set_title("constraints per step")
    a.legend()
    a = ax[1, 2]
    for j, n in enumerate(("nu_F", "nu_B", "nu_Psi")):
        a.semilogy(k, [max(abs(h["nu"][j]), 1e-30) for h in st], label=n)
    a.semilogy(k, [h["rho"] for h in st], "k--", label="rho")
    a.set_title("multipliers and penalty")
    a.legend()
    a = ax[2, 0]
    a.semilogy(k, [h["phi"] for h in st], label="|J - J_fit|^2 / 2")
    a.semilogy(k, [h["f"] for h in st], label="L_A")
    a.set_title("objective")
    a.legend()
    a = ax[2, 1]
    a.plot(k, [h["time"] for h in st], ".", label="time [s]")
    if any("lanczos" in h for h in st):
        a.plot(k, [h.get("lanczos", np.nan) for h in st], ".", label="Lanczos iterations")
    a.semilogy(k, [h["galerkin"] for h in st], "x", label="Galerkin")
    a.set_yscale("log")
    a.set_title("cost and Galerkin identity")
    a.legend()
    a = ax[2, 2]
    anc = run.get("anchors", [])
    rho = np.asarray(run["start"]["rho"])
    a.plot(rho, run["start"]["iota"], "k-", label="start")
    for i, r in enumerate(anc):
        a.plot(rho, r["iota"], "-", color=plt.cm.viridis(i / max(len(anc), 1)), lw=0.8)
    for e in run["alphas"]:
        if "iota" in e:
            a.plot(rho, e["iota"], "r--", label=f"final alpha {e['alpha']}")
    a.set_title(f"iota (surface average), {len(anc)} re-anchorings")
    a.set_xlabel("rho")
    a.legend()
    for i, h in enumerate(st):
        if h.get("reanchor"):
            for b in ax.ravel()[:8]:
                b.axvline(i, color="0.85", lw=0.6, zorder=0)
    for b in ax.ravel()[:8]:
        b.set_xlabel("trust-region step")
    fig.suptitle(label)
    fig.tight_layout()
    fig.savefig(f"{out}_{label}.png", dpi=110)
    plt.close(fig)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("runs", nargs="+", help="label=run.json")
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    os.makedirs(os.path.dirname(a.out) or ".", exist_ok=True)
    rows = []
    for spec in a.runs:
        label, path = spec.split("=", 1)
        run = json.load(open(path))
        figure(label, run, a.out)
        s = summary(label, run)
        if s:
            rows.append(s)
    if rows:
        keys = list(rows[0])
        print("| " + " | ".join(keys) + " |")
        print("|" + "---|" * len(keys))
        for r in rows:
            fmt = []
            for k in keys:
                v = r[k]
                if isinstance(v, float):
                    fmt.append(f"{v:.3g}")
                elif isinstance(v, (tuple, list)):
                    fmt.append(" / ".join(f"{x:.3g}" for x in v))
                else:
                    fmt.append(str(v))
            print("| " + " | ".join(fmt) + " |")


if __name__ == "__main__":
    main()
