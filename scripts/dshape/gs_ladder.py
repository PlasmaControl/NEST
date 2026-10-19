"""Resolution ladder of force-only runs: does each step up change the end state less?

Reads results/gs/fields_<run>.npz (gs_check_end.py --grid-res ... --save): the end
state's J, B, p (and F) of every run on one common grid and box, with DESC L24 and the
reference (DESC L32) at the same points. For each ladder (the runs in order of
resolution):
  - the change between successive steps, |X_k+1 - X_k| for B, J along and across B,
    and p (volume rms over the reference's rms);
  - the distance of each end state to DESC L32 (and DESC L24's own, for scale);
  - <|F|> / norm of each end state on the common grid and box.
Prints the tables and draws figures/gs_ladder.png.

usage: python scripts/dshape/gs_ladder.py --basis RUN ... --basis-labels 1 1.5 2 2.5 3
    --box RUN ... --box-labels 10 15 20 25 30 [--name gs_ladder --title "force balance only"]
"""

import argparse
import os

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
QS = ("B", "J along", "J across", "p")


def load(run):
    return dict(np.load(os.path.join(ROOT, "results", "gs", f"fields_{os.path.basename(run)}.npz")))


def measures(d):
    """Volume weights, unit b of the reference, the reference rms of B, J, p."""
    ok = d["ok"]
    w = d["dV"] * ok
    w = w / w.sum()
    B = np.nan_to_num(d["ref_B"])
    bh = B / np.maximum(np.linalg.norm(B, axis=1), 1e-300)[:, None]
    rms = {q: np.sqrt(np.sum(w * np.sum(np.nan_to_num(d[f"ref_{q}"]) ** 2, axis=-1)
                             if d[f"ref_{q}"].ndim > 1 else w * np.nan_to_num(d[f"ref_{q}"]) ** 2))
           for q in ("B", "J", "p")}
    return w, bh, rms


def diff(X, Y, w, bh, rms):
    """Volume rms of X - Y (dicts with B, J, p) over the reference's rms."""
    dJ = np.nan_to_num(X["J"] - Y["J"])
    par = np.sum(dJ * bh, 1)
    perp = dJ - par[:, None] * bh
    return {"B": np.sqrt(np.sum(w * np.sum(np.nan_to_num(X["B"] - Y["B"]) ** 2, 1))) / rms["B"],
            "J along": np.sqrt(np.sum(w * par**2)) / rms["J"],
            "J across": np.sqrt(np.sum(w * np.sum(perp**2, 1))) / rms["J"],
            "p": np.sqrt(np.sum(w * np.nan_to_num(X["p"] - Y["p"]) ** 2)) / rms["p"]}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--basis", nargs="*", default=[])
    ap.add_argument("--basis-labels", nargs="*", default=[])
    ap.add_argument("--box", nargs="*", default=[])
    ap.add_argument("--box-labels", nargs="*", default=[])
    ap.add_argument("--ladder", action="append", default=[],
                    help="another ladder: 'title;RUN,RUN,...;label,label,...' (a label starting "
                         "with 'exact' is drawn as the limit, a star)")
    ap.add_argument("--name", default="gs_ladder", help="figures/<name>.png")
    ap.add_argument("--title", default="force balance only", help="the runs' objectives (suptitle)")
    ap.add_argument("--no-ref-panel", action="store_true", help="leave out the distance-to-reference panel")
    ap.add_argument("--ref-F", type=float, default=None, help="the reference's own <|F|>/norm (a line)")
    ap.add_argument("--ref-name", default="DESC L32", help="the reference (gs_check_end.py --ref)")
    ap.add_argument("--desc-name", default="DESC L24", help="the runs' own DESC equilibrium; '' when it is "
                    "the reference itself")
    a = ap.parse_args()
    ladders = [(f"basis (x factor of DESC's L = M), box a/20", a.basis, a.basis_labels, "x"),
               (f"box a/N, basis x2", a.box, a.box_labels, "a/")]
    for spec in a.ladder:
        t, r, l = spec.rsplit(";", 2)  # the title may contain ";"
        ladders.append((t, r.split(","), l.split(","), ""))
    ladders = [l for l in ladders if l[1]]
    nc = 2 if a.no_ref_panel else 3
    fig, axs = plt.subplots(len(ladders), nc, figsize=(6 * nc, 4.8 * len(ladders)), squeeze=False)
    if a.no_ref_panel:  # the distance panel drawn on a figure that is never saved
        spare = plt.figure().subplots(len(ladders), 1, squeeze=False)
        axs = [[r[0], spare[i][0], r[1]] for i, r in enumerate(axs)]
    for row, (title, runs, labels, pre) in enumerate(ladders):
        D = [load(r) for r in runs]
        x0 = D[0]["x"]
        for d, r in zip(D, runs):
            assert d["x"].shape == x0.shape and np.allclose(d["x"], x0), f"{r}: not the common grid"
        w, bh, rms = measures(D[0])
        ends = [{q: d[f"end_{q}"] for q in ("J", "B", "p")} for d in D]
        ref = {q: D[0][f"ref_{q}"] for q in ("J", "B", "p")}
        desc = {q: D[0][f"desc_{q}"] for q in ("J", "B", "p")}
        Fm = [float(np.sum(w * np.linalg.norm(d["end_F"], axis=1)) / d["norm"]) for d in D]
        to_ref = [diff(e, ref, w, bh, rms) for e in ends]
        desc_ref = diff(desc, ref, w, bh, rms)
        steps = [diff(ends[k + 1], ends[k], w, bh, rms) for k in range(len(ends) - 1)]
        lab = [f"{pre}{l}" for l in labels] if labels else [os.path.basename(r) for r in runs]
        print(f"\n== ladder: {title}")
        print(f"   {'':16s} {'F on grid':>10s} " + " ".join(f"{q + ' to ref':>14s}" for q in QS))
        if a.desc_name:
            print(f"   {a.desc_name:16s} {'':>10s} " + " ".join(f"{desc_ref[q]:14.2e}" for q in QS))
        for l, F, t in zip(lab, Fm, to_ref):
            print(f"   {l:16s} {F:10.2e} " + " ".join(f"{t[q]:14.2e}" for q in QS))
        print(f"   change between successive steps:")
        for k, s in enumerate(steps):
            print(f"   {lab[k] + ' -> ' + lab[k + 1]:16s} {'':10s} " + " ".join(f"{s[q]:14.2e}" for q in QS))
        xs = np.arange(len(lab))
        ax = axs[row]
        for q, mk in zip(QS, "osd^"):
            ax[0].semilogy(xs[:-1] + 0.5, [s[q] for s in steps], mk + "-", label=q)
            ax[1].semilogy(xs, [t[q] for t in to_ref], mk + "-", label=q)
            if a.desc_name:
                ax[1].axhline(desc_ref[q], color=ax[1].lines[-1].get_color(), ls=":", lw=0.8)
        ax[2].semilogy(xs, Fm, "ko-")
        lim = [i for i, l in enumerate(lab) if l.startswith("exact")]
        if lim:  # the limit (no grid): stars, and the force value
            ax[2].semilogy(lim, [Fm[i] for i in lim], "k*", ms=16)
            for i in lim:
                ax[2].annotate(f"{Fm[i]:.2e}", (i, Fm[i]), xytext=(-60, 8), textcoords="offset points",
                               fontsize=9)
            for q, mk in zip(QS, "osd^"):
                ax[1].semilogy(lim, [to_ref[i][q] for i in lim], "*", ms=14,
                               color=[ln for ln in ax[1].lines if ln.get_label() == q][0].get_color())
        if a.ref_F is not None:
            ax[2].axhline(a.ref_F, color="k", ls=":", lw=1, label=f"{a.ref_name} ({a.ref_F:.2e})")
            ax[2].legend(fontsize=8)
        for i, t in enumerate(("change between successive steps",
                               f"distance to {a.ref_name}" + (f" (dotted: {a.desc_name}'s)" if a.desc_name else ""),
                               r"$\langle|F|\rangle$/norm of the end state (common grid, box)")):
            ax[i].set_title(f"{t}\n{title}", fontsize=9)
            ax[i].set_xticks(xs)
            ax[i].set_xticklabels(lab)
        # a step compares two resolutions: its point sits between them, labelled with both
        ax[0].set_xticks(xs[:-1] + 0.5)
        ax[0].set_xticklabels([f"{lab[k]}\u2192{lab[k + 1]}" for k in range(len(lab) - 1)],
                              rotation=30, ha="right", fontsize=8)
        ax[0].legend(fontsize=8)
        ax[1].legend(fontsize=8)
    import textwrap
    fig.suptitle("\n".join(textwrap.wrap(f"resolution ladder: {a.title}", 55 * nc)))
    fig.tight_layout()
    out = os.path.join(ROOT, "figures", "dshape", f"{a.name}.png")
    os.makedirs(os.path.dirname(out), exist_ok=True)
    fig.savefig(out, dpi=140)
    print(f"\nsaved {out}")


if __name__ == "__main__":
    main()
