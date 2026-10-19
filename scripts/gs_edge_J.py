"""Current density and force error toward the edge: DESC against NEST runs.

From gs_check_end.py --save (results/gs/fields_<run>.npz: J, B, p and F of DESC, of
the NEST start and of the NEST end on the independent check grid and box), for each
radius of that grid:
  - <|J|>_surface / max (as edge_J_convergence.png: the DESC profile I(rho) makes J
    vanish like (1 - rho)^2 at the LCFS);
  - <|J x B - grad p|>_surface / norm;
  - <|F|> / norm in radial bins, and over the volume.
Draws figures/<name>.png.

usage: python scripts/gs_edge_J.py --runs results/gs/RUN ... [--labels ...] [--name gs_edge_J]
"""

import argparse
import os

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def surface(d, v):
    """<v>_surface on each radius of the grid (dV weights), and the radii."""
    r = d["rtz"][:, 0]
    rho = np.unique(r)
    return rho, np.array([np.sum(d["dV"][r == q] * v[r == q]) / np.sum(d["dV"][r == q]) for q in rho])


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--runs", nargs="+", required=True)
    ap.add_argument("--labels", nargs="*", default=None)
    ap.add_argument("--name", default="gs_edge_J")
    ap.add_argument("--title", default="")
    a = ap.parse_args()
    labels = a.labels or [os.path.basename(r) for r in a.runs]
    D = [dict(np.load(os.path.join(ROOT, "results", "gs", f"fields_{os.path.basename(r)}.npz"))) for r in a.runs]
    d0 = D[0]
    norm = float(d0["norm"])
    fig, ax = plt.subplots(1, 3, figsize=(18, 5))
    rho, JD = surface(d0, np.linalg.norm(d0["desc_J"], axis=1))
    Jmax = JD.max()
    ax[0].loglog(1 - rho, JD / Jmax, "k", lw=2, label="DESC")
    has_FD = "desc_F" in d0  # (saved by gs_check_end.py since the coil-field runs)
    if has_FD:
        _, FD = surface(d0, np.linalg.norm(d0["desc_F"], axis=1) / norm)
        ax[1].loglog(1 - rho, FD, "k", lw=2, label="DESC")
    edges = (0.0, 0.5, 0.8, 0.9, 0.95, 0.99, 1.0)
    r = d0["rtz"][:, 0]
    w = d0["dV"]

    def bins(F):
        f = np.linalg.norm(F, axis=1) / norm
        return [float(np.sum(w * f) / np.sum(w))] + [
            float(np.sum((w * f)[(r > x) & (r <= y)]) / np.sum(w[(r > x) & (r <= y)]))
            for x, y in zip(edges[:-1], edges[1:])]

    names = ["all"] + [f"{x:g}-{y:g}" for x, y in zip(edges[:-1], edges[1:])]
    rows = [("DESC", bins(d0["desc_F"]))] if has_FD else []
    for d, lab, c in zip(D, labels, ("C0", "C3", "C2", "C1", "C4")):
        assert np.allclose(d["x"], d0["x"]), "not the same check grid"
        _, J1 = surface(d, np.linalg.norm(d["end_J"], axis=1))
        _, F1 = surface(d, np.linalg.norm(d["end_F"], axis=1) / norm)
        ax[0].loglog(1 - rho, J1 / Jmax, c, label=f"NEST {lab}")
        ax[1].loglog(1 - rho, F1, c, label=f"NEST {lab}")
        if d is d0:
            _, J0 = surface(d, np.linalg.norm(d["start_J"], axis=1))
            _, F0 = surface(d, np.linalg.norm(d["start_F"], axis=1) / norm)
            ax[0].loglog(1 - rho, J0 / Jmax, "C7:", label="NEST start")
            ax[1].loglog(1 - rho, F0, "C7:", label="NEST start")
            rows.append(("NEST start", bins(d["start_F"])))
        rows.append((f"NEST {lab}", bins(d["end_F"])))
    x = np.logspace(-4, 0, 50)
    ax[0].loglog(x, x**2 * JD[np.argmin(np.abs(1 - rho - 0.1))] / Jmax / 0.01, "k--", lw=0.8,
                 label=r"$\propto (1-\rho)^2$")
    ax[0].set_title(r"$\langle|J|\rangle_{surface}$ / max")
    ax[1].set_title(r"$\langle|J\times B-\nabla p|\rangle_{surface}$ / norm")
    for x_ in ax[:2]:
        x_.set_xlabel(r"$1-\rho$")
        x_.legend(fontsize=8)
    xs = np.arange(len(names))
    for k, (lab, v) in enumerate(rows):
        ax[2].semilogy(xs, v, "o-", color="k" if lab == "DESC" else None, label=lab)
    ax[2].set_xticks(xs)
    ax[2].set_xticklabels(names, fontsize=8)
    ax[2].set_xlabel(r"$\rho$ bin")
    ax[2].set_title(r"$\langle|F|\rangle$/norm: volume and radial bins (check grid and box)")
    ax[2].legend(fontsize=8)
    print(f"{'':28s}" + " ".join(f"{n:>9s}" for n in names))
    for lab, v in rows:
        print(f"{lab:28s}" + " ".join(f"{x_:9.2e}" for x_ in v))
    fig.suptitle(a.title)
    fig.tight_layout()
    out = os.path.join(ROOT, "figures", f"{a.name}.png")
    fig.savefig(out, dpi=140)
    print("saved", out)


if __name__ == "__main__":
    main()
