"""Poincare sections of ``poincare.py`` in DESC flux coordinates, and iota of each line.

The crossings (R, Z) at phi = 0 are mapped to the DESC (rho, theta) of the starting
equilibrium (``Equilibrium.map_coordinates``). Island chains then show up as rows of
m islands. The rotational transform of each traced line is its mean poloidal
advance per field period, NFP <dtheta> / 2 pi, plotted at its mean rho against
DESC's iota. Resonances iota = n/m (n a multiple of NFP, m <= 16) of each field are
labeled from that field's own iota.

Runs on a CPU in about two minutes.

usage: python poincare_flux.py EQ.h5 SECTIONS_poincare.npz
Writes SECTIONS_flux.png and SECTIONS_flux.npz next to the sections (one column per
traced field: the start and, unless poincare.py ran with --x start, the optimized x).
"""

import sys
from fractions import Fraction

import numpy as np

from desc import set_device

set_device("cpu")

import desc.io  # noqa: E402
from desc.grid import LinearGrid  # noqa: E402


def resonances(rho, iota, NFP, max_m=16):
    """(n, m, rho) where the piecewise-linear iota(rho) crosses n/m."""
    out, seen = [], set()
    for m in range(1, max_m + 1):
        for n in range(NFP, 4 * NFP + 1, NFP):
            t = Fraction(n, m)
            if t in seen:
                continue
            idx = np.nonzero(np.diff(np.sign(iota - float(t))))[0]
            if len(idx):
                seen.add(t)
            for i in idx:
                out.append((n, m, rho[i] + (float(t) - iota[i]) * (rho[i + 1] - rho[i])
                            / (iota[i + 1] - iota[i])))
    return out


def main(eq_path, sec_path):
    eq = desc.io.load(eq_path)
    d = np.load(sec_path)
    n_lines = d["start_R"].shape[1]
    flux, lines = {}, {}
    runs = [k for k in ("start", "optimized") if f"{k}_R" in d]
    for k in runs:
        R, Z = d[f"{k}_R"][1:], d[f"{k}_Z"][1:]
        ok = np.isfinite(R).all(0)  # lines that stayed on the grid
        c = np.column_stack([R[:, ok].ravel(), np.zeros(R[:, ok].size), Z[:, ok].ravel()])
        rtz = np.asarray(eq.map_coordinates(c, inbasis=("R", "phi", "Z"),
                                            outbasis=("rho", "theta", "zeta"),
                                            maxiter=60, tol=1e-10))
        rho = rtz[:, 0].reshape(-1, ok.sum())
        th = np.mod(rtz[:, 1], 2 * np.pi).reshape(-1, ok.sum())
        dth = np.diff(np.unwrap(th, axis=0), axis=0)
        flux[k] = (rho, th, np.flatnonzero(ok))
        lines[k] = (rho.mean(0), np.abs(dth.mean(0)) * eq.NFP / (2 * np.pi))
        print(f"{k}: {ok.sum()} of {n_lines} lines mapped", flush=True)

    r = np.linspace(0.01, 1, 200)
    g = LinearGrid(rho=r, M=eq.M_grid, N=eq.N_grid, NFP=eq.NFP)
    iota_desc = np.abs(g.compress(eq.compute("iota", grid=g)["iota"]))
    res = {"DESC": resonances(r, iota_desc, eq.NFP)}
    for k, (rm, io) in lines.items():
        # the uniform seeds (theta = 0, the first ones) give a clean iota(rho)
        o = np.argsort(rm)
        rm, io = rm[o], io[o]
        keep = np.r_[True, np.diff(rm) > 5e-3]
        res[k] = resonances(rm[keep], io[keep], eq.NFP)
        print(f"{k} resonances: " + ", ".join(f"{n}/{m} at {x:.2f}" for n, m, x in res[k]))

    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fig = plt.figure(figsize=(7.5 * len(runs), 15), constrained_layout=True)
    gs = fig.add_gridspec(3, len(runs), height_ratios=[1, 1.1, 0.8])
    cmap = plt.cm.turbo
    titles = {"start": "start (fits of DESC's J and p)", "optimized": "NEST optimized"}
    for j, k in enumerate(runs):
        rho, th, idx = flux[k]
        col = cmap((np.broadcast_to(idx, rho.shape).ravel() * 0.618) % 1)
        for i, lo in enumerate((0.0, 0.5)):
            ax = fig.add_subplot(gs[i, j])
            m = rho.ravel() >= lo
            ax.scatter(th.ravel()[m] / np.pi, rho.ravel()[m], s=0.4 if i == 0 else 0.8,
                       c=col[m], lw=0)
            for n, mm, x in res[k]:
                if lo <= x <= 1:
                    ax.axhline(x, color="0.4", lw=0.5, ls=":")
                    ax.text(2.01, x, f"{n}/{mm}", fontsize=7, va="center")
            ax.set_xlim(0, 2)
            ax.set_ylim(lo, 1.0)
            ax.set_xlabel("DESC theta / pi")
            ax.set_ylabel("DESC rho")
            ax.set_title(titles[k] + (", rho > 0.5" if lo else "")
                         + "\n(dotted: iota = n/m of this field)")
    ax = fig.add_subplot(gs[2, :])
    ax.plot(r, iota_desc, "k-", lw=1.2, label="DESC equilibrium")
    for k, mk in zip(runs, "os"):
        ax.plot(*lines[k], mk, ms=3.5, mfc="none", label=f"field lines, {titles[k]}")
    for n, mm in sorted({(n, m) for v in res.values() for n, m, _ in v}):
        ax.axhline(n / mm, color="0.6", lw=0.5, ls=":")
        ax.text(1.005, n / mm, f"{n}/{mm}", fontsize=7, va="center")
    ax.set_xlim(0, 1)
    ax.set_xlabel("rho (DESC; mean over each line)")
    ax.set_ylabel("|iota|")
    ax.legend(loc="lower right", fontsize=9)
    ax.set_title("rotational transform: DESC, and NFP <dtheta> / 2 pi of each traced line")
    fig.suptitle(f"{eq_path.split('/')[-1]}: Poincare sections at phi = 0 in DESC flux coordinates")
    base = sec_path[: -len(".npz")]
    fig.savefig(base + "_flux.png", dpi=130)
    np.savez(base + "_flux.npz", r_desc=r, iota_desc=iota_desc,
             **{f"{k}_rho": v[0] for k, v in flux.items()},
             **{f"{k}_theta": v[1] for k, v in flux.items()},
             **{f"{k}_line_rho": v[0] for k, v in lines.items()},
             **{f"{k}_line_iota": v[1] for k, v in lines.items()})
    print(f"saved {base}_flux.png/.npz")


if __name__ == "__main__":
    main(*sys.argv[1:3])
