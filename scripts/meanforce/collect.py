"""Tables and convergence plots of the mean-force runs (``mf_run.py`` JSONs).

    python scripts/meanforce/collect.py --dir results/meanforce --out figures/meanforce/study

Writes <out>_table.md (every run, every bound: |J - J_fit|/rms, M/M_DESC, B.n/|B| rms, Psi,
whether feasible and stationary, time) and the figures:

    <out>_front.png    |J - J_fit| against M / M_DESC, one curve per run (the Pareto front)
    <out>_spacing.png  study 1: |J - J_fit| against the annulus spacing h/a at fixed alpha,
                       and the difference to the finest spacing
    <out>_basis.png    study 2: |J - J_fit| against L = M at fixed alpha (N = 12), and the
                       16/16/16 check

A run's label is its file's name without ``.json``; continued runs (``--start``) are read
as their own runs. ``RUNS`` maps a file to (study, frac, basis L = M, N, method).
"""

import argparse
import json
import os

import matplotlib
import numpy as np

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

# file stem -> (annulus a/frac, basis L = M, N, method)
RUNS = {
    "sweep_qr_a40": (40, 16, 12, "QR"), "sweep_gltr_a40": (40, 16, 12, "GLTR"),
    "sweep_qr_a40_r05": (40, 16, 12, "QR"),
    "s1_qr_a20": (20, 16, 12, "QR"), "s1_gltr_a20": (20, 16, 12, "GLTR"),
    "s1_qr_a20_r05": (20, 16, 12, "QR"),
    "s1_qr_a60": (60, 16, 12, "QR"), "s1_gltr_a60": (60, 16, 12, "GLTR"),
    "s1_qr_a80": (80, 16, 12, "QR"), "s1_gltr_a80": (80, 16, 12, "GLTR"),
    "s1c_gltr_a100_a0.5": (100, 16, 12, "GLTR"), "s1c_gltr_a120_a0.5": (120, 16, 12, "GLTR"),
    "s1c_gltr_a100_a0.1": (100, 16, 12, "GLTR"), "s1c_gltr_a120_a0.1": (120, 16, 12, "GLTR"),
    "s2_gltr_L20N12": (40, 20, 12, "GLTR"), "s2_gltr_L24N12": (40, 24, 12, "GLTR"),
    "s2_gltr_L28N12": (40, 28, 12, "GLTR"), "s2_gltr_L16N16": (40, 16, 16, "GLTR"),
}
ALPHAS = (0.7, 0.5, 0.3, 0.2, 0.1, 0.05)


def load(d):
    out = {}
    for stem, meta in RUNS.items():
        p = os.path.join(d, stem + ".json")
        if not os.path.exists(p):
            continue
        run = json.load(open(p))
        rows = []
        for e in run["alphas"]:
            if "|J-J_fit|/rms" not in e:
                continue
            rows.append({"alpha": e["alpha"], "g": e["|J-J_fit|/rms"], "M": e["M/M_desc"],
                         "Bn": e["Bn/|B| rms"], "Psi": e["Psi"], "time": e["time"],
                         "feasible": e.get("feasible", True),
                         "stationary": e.get("stationary", None),
                         "anchors": e.get("anchors")})
        if rows:
            out[stem] = {"meta": meta, "rows": rows, "Psi_t": run.get("start", {}).get("Psi")}
    return out


def table(runs):
    L = ["| run | spacing | basis (L = M / N) | method | alpha | \\|J − J_fit\\| / rms | M / M_DESC | "
         "B·n/\\|B\\| rms | Ψ/Ψ_start − 1 | stationary | time [min] |",
         "|---|---|---|---|---|---|---|---|---|---|---|"]
    for stem, r in runs.items():
        frac, lm, n, meth = r["meta"]
        for x in r["rows"]:
            dpsi = (x["Psi"] / r["Psi_t"] - 1) if r["Psi_t"] else float("nan")
            L.append(f"| {stem} | a/{frac} | {lm} / {n} | {meth} | {x['alpha']:g} | {x['g']:.4e} | "
                     f"{x['M']:.4f} | {x['Bn']:.3e} | {dpsi:+.1e} | "
                     f"{'yes' if x['stationary'] else ('no' if x['stationary'] is not None else '-')} | "
                     f"{x['time'] / 60:.0f} |")
    return "\n".join(L)


def value(runs, stems, alpha):
    """|J - J_fit|/rms of the first run in ``stems`` that has the bound (feasible)."""
    for s in stems:
        if s in runs:
            for x in runs[s]["rows"]:
                if abs(x["alpha"] - alpha) < 1e-9 and x["feasible"]:
                    return x["g"]
    return None


def fronts(runs, path):
    fig, ax = plt.subplots(figsize=(8, 6))
    for i, (stem, r) in enumerate(runs.items()):
        frac, lm, n, meth = r["meta"]
        xs = [x["M"] for x in r["rows"] if x["feasible"]]
        ys = [x["g"] for x in r["rows"] if x["feasible"]]
        ax.loglog(xs, ys, "o-" if meth == "GLTR" else "s--", ms=4, lw=1,
                  color=plt.cm.tab20(i % 20), label=f"{stem}")
    ax.set_xlabel("M / M_DESC (mean force bound)")
    ax.set_ylabel("|J − J_fit| / rms J_fit")
    ax.set_title("front: distance to the fitted current against the mean force")
    ax.legend(fontsize=6, ncol=2)
    ax.grid(alpha=0.3, which="both")
    fig.tight_layout()
    fig.savefig(path, dpi=120)
    plt.close(fig)


def spacing(runs, path):
    methods = {"GLTR": ["sweep_gltr_a40", "s1_gltr_a20", "s1_gltr_a60", "s1_gltr_a80"],
               "QR": ["sweep_qr_a40", "s1_qr_a20", "s1_qr_a60", "s1_qr_a80"]}
    cont = {0.5: ["s1c_gltr_a100_a0.5", "s1c_gltr_a120_a0.5"],
            0.1: ["s1c_gltr_a100_a0.1", "s1c_gltr_a120_a0.1"]}
    fig, ax = plt.subplots(1, 2, figsize=(13, 5))
    for k, alpha in enumerate((0.5, 0.3, 0.2, 0.1, 0.05)):
        for meth, stems in methods.items():
            pts = [(runs[s]["meta"][0], value(runs, [s], alpha)) for s in stems if s in runs]
            pts += [(runs[s]["meta"][0], value(runs, [s], alpha))
                    for s in cont.get(alpha, []) if s in runs]
            pts = sorted((f, v) for f, v in pts if v is not None)
            if len(pts) < 2:
                continue
            f, v = np.array(pts, float).T
            ax[0].plot(1 / f * 1, v, "o-" if meth == "GLTR" else "s--", color=f"C{k}", ms=4, lw=1,
                       label=f"α = {alpha:g}, {meth}")
            ref = v[-1]
            ax[1].semilogy(1 / f[:-1], np.abs(v[:-1] - ref) / ref + 1e-9,  # (not the reference)
                           "o-" if meth == "GLTR" else "s--", color=f"C{k}", ms=4, lw=1,
                           label=f"α = {alpha:g}, {meth}")
    ax[0].set_xlabel("h / a")
    ax[0].set_ylabel("|J − J_fit| / rms J_fit")
    ax[0].set_yscale("log")
    ax[0].set_title("study 1: annulus spacing")
    ax[1].set_xlabel("h / a")
    ax[1].set_ylabel("relative difference to the finest spacing")
    ax[1].set_title("")
    ax[0].legend(fontsize=6, ncol=2)
    for a_ in ax:
        a_.grid(alpha=0.3, which="both")
    fig.tight_layout()
    fig.savefig(path, dpi=120)
    plt.close(fig)


def basis(runs, path):
    stems = {16: "sweep_gltr_a40", 20: "s2_gltr_L20N12", 24: "s2_gltr_L24N12", 28: "s2_gltr_L28N12"}
    fig, ax = plt.subplots(1, 2, figsize=(13, 5))
    for k, alpha in enumerate(ALPHAS):
        pts = [(lm, value(runs, [s], alpha)) for lm, s in stems.items() if s in runs]
        pts = [(lm, v) for lm, v in pts if v is not None]
        if len(pts) < 2:
            continue
        lm, v = np.array(pts, float).T
        ax[0].semilogy(lm, v, "o-", color=f"C{k}", label=f"α = {alpha:g}")
        n16 = value(runs, ["s2_gltr_L16N16"], alpha)
        if n16 is not None:
            ax[0].semilogy([16], [n16], "x", color=f"C{k}", ms=9)
        ax[1].semilogy(lm[:-1], np.abs(v[:-1] - v[-1]) / v[-1] + 1e-9, "o-", color=f"C{k}",
                       label=f"α = {alpha:g} (vs L = M = {int(lm[-1])})")
    ax[0].set_xlabel("L = M (N = 12; x: 16/16/16)")
    ax[0].set_ylabel("|J − J_fit| / rms J_fit (each basis' own fit)")
    ax[0].set_title("study 2: NEST basis")
    ax[1].set_xlabel("L = M")
    ax[1].set_ylabel("relative difference to the largest basis of that α")
    ax[1].legend(fontsize=7)
    ax[0].legend(fontsize=7)
    for a_ in ax:
        a_.grid(alpha=0.3, which="both")
    fig.tight_layout()
    fig.savefig(path, dpi=120)
    plt.close(fig)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dir", default="results/meanforce")
    ap.add_argument("--out", default="figures/meanforce/study")
    a = ap.parse_args()
    os.makedirs(os.path.dirname(a.out) or ".", exist_ok=True)
    runs = load(a.dir)
    with open(a.out + "_table.md", "w") as fh:
        fh.write(table(runs) + "\n")
    fronts(runs, a.out + "_front.png")
    spacing(runs, a.out + "_spacing.png")
    basis(runs, a.out + "_basis.png")
    print(f"{len(runs)} runs; wrote {a.out}_table.md and the three figures")


if __name__ == "__main__":
    main()
