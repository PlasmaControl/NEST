"""Overlap of the resonances in a band of rho, and which traced lines are chaotic.

For each field (``label=RESONANT.npz:run`` from ``resonant_field.py``, and the
``_poincare_flux.npz`` of a long trace):

  - every resonance iota = n/m in the band (n a multiple of NFP, m <= --m-max, lowest
    m per rational) at DESC's rho_D, with the resonant b_mn = 2 |c_mn| there and the
    pendulum width w = 4 sqrt(b_mn / (m |iota'|)) (as ``resonant_widths.py``);
  - Chirikov's parameter of neighbouring chains, sigma = (w_1 + w_2) / (2 |rho_1 - rho_2|)
    (> 1: the separatrices overlap; ~2/3 already destroys the KAM surfaces between them,
    once the secondary chains are counted), and the fraction of the band the islands
    cover;
  - per traced line, its thickness at phi = 0: the rms of rho about a Fourier fit in
    theta (|k| <= 24). A flux surface is a smooth curve rho(theta), so the rms is the
    tracer's error (~1e-4 in the core, < 6e-3 next to the small chains); a chaotic line
    fills a layer (rms ~ its width, 1.5-2e-2 at NCSX's 3/5 separatrix). Lines whose
    theta still has a gap > 0.2 rad (inside an island chain) are counted apart. The
    weighted Birkhoff rotation number (Das et al. 2017) does not separate the two here:
    with poincare.py's tracer it stops converging at 1e-4-1e-5 on surfaces too;
  - per traced line, its secular drift in rho over the trace (a linear fit of rho
    against the crossing number). A divergence-free field cannot move all lines one
    way: a drift common to the regular lines is an error of the traced field.

usage: python resonant_overlap.py OUT.png --band 0.62 0.78 \
    --field label=RESONANT.npz:run=FLUX.npz[:prefix] ...
"""

import argparse
from fractions import Fraction

import numpy as np

CHAOTIC = 1e-2  # thickness above this: a chaotic line
GAP = 0.2  # a line with a larger gap in theta is inside an island chain


def thickness(rho, theta, K=24):
    """rms of rho about a Fourier fit rho(theta), |k| <= K, per line (rho, theta:
    (n_crossings, n_lines)); NaN for a line with a gap in theta > GAP (an island line)."""
    out = np.full(rho.shape[1], np.nan)
    k = np.arange(1, K + 1)
    for j in range(rho.shape[1]):
        ok = np.isfinite(rho[:, j]) & np.isfinite(theta[:, j])
        r, t = rho[ok, j], np.mod(theta[ok, j], 2 * np.pi)
        if len(r) < 8 * K or np.diff(np.sort(np.r_[t, t.min() + 2 * np.pi])).max() > GAP:
            continue
        A = np.column_stack([np.ones_like(t), np.cos(np.outer(t, k)), np.sin(np.outer(t, k))])
        c, *_ = np.linalg.lstsq(A, r, rcond=None)
        out[j] = np.sqrt(np.mean((r - A @ c) ** 2))
    return out


def drift(rho):
    """Change of rho over the trace per line, from a linear fit against the crossing
    number (rho: (n_crossings, n_lines))."""
    k = np.arange(rho.shape[0], dtype=float)
    k -= k.mean()
    r = np.where(np.isfinite(rho), rho, np.nanmean(rho, 0))
    return (k @ (r - r.mean(0))) / (k @ k) * rho.shape[0]


def resonances(rho, iota, NFP, lo, hi, m_max, n_max):
    """(n, m, rho_D, iota') of iota = n/m in [lo, hi] (the lowest m for each rational with
    n a multiple of NFP)."""
    out, seen = [], set()
    for m in range(1, m_max + 1):
        for n in range(NFP, n_max + 1, NFP):
            q = Fraction(n, m)
            if q in seen:
                continue
            idx = np.nonzero(np.diff(np.sign(iota - float(q))))[0]
            for i in idx:
                x = rho[i] + (float(q) - iota[i]) * (rho[i + 1] - rho[i]) / (iota[i + 1] - iota[i])
                if lo - 0.05 <= x <= hi + 0.05:
                    seen.add(q)
                    out.append((n, m, x, (iota[i + 1] - iota[i]) / (rho[i + 1] - rho[i])))
    return sorted(out, key=lambda r: r[2])


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("out")
    ap.add_argument("--field", nargs="+", required=True,
                    help="label=RESONANT.npz:run=FLUX.npz[:prefix]")
    ap.add_argument("--band", type=float, nargs=2, default=(0.62, 0.78))
    ap.add_argument("--m-max", type=int, default=None, help="default: the spectra's")
    ap.add_argument("--w-min", type=float, default=2e-3,
                    help="chains narrower than this are left out of the overlap table")
    a = ap.parse_args()
    lo, hi = a.band
    runs = []
    for s in a.field:
        label, rest = s.split("=", 1)
        res, flux = rest.split("=", 1)
        rpath, run = res.rsplit(":", 1)
        fpath, _, prefix = flux.partition(":")
        runs.append((label, np.load(rpath), run, np.load(fpath), prefix or "optimized"))

    rows, lines = {}, {}
    for label, d, run, f, pre in runs:
        rho, NFP = d["rho"], int(d["NFP"])
        iota = np.abs(d["iota"])
        sgn = np.sign(np.mean(d["iota"]))
        n_list = list(d["n"])
        m_max = min(a.m_max or len(d["m"]) - 1, len(d["m"]) - 1)
        c = d[f"c_{run}"]
        rr = []
        for n, m, x, slope in resonances(rho, iota, NFP, lo, hi, m_max, max(n_list)):
            b = float(np.interp(x, rho, 2 * np.abs(c[:, m, n_list.index(int(sgn * n))])))
            rr.append(dict(n=n, m=m, rho=x, slope=slope, b=b,
                           w=4 * np.sqrt(b / (m * abs(slope)))))
        rows[label] = rr
        lines[label] = (f[f"{pre}_line_rho"], thickness(f[f"{pre}_rho"], f[f"{pre}_theta"]),
                        f[f"{pre}_rho"], f[f"{pre}_theta"], drift(f[f"{pre}_rho"]))

    for label in rows:
        rr = rows[label]
        print(f"\n{label}: resonances in rho {lo - 0.05:.2f}-{hi + 0.05:.2f} (DESC's iota)")
        print(f"  {'chain':>6s} {'rho':>6s} {'iota_p':>7s} {'b_mn':>9s} {'w':>7s}")
        for r in rr:
            print(f"  {r['n']:>3d}/{r['m']:<3d} {r['rho']:6.3f} {r['slope']:7.3f} {r['b']:9.2e} "
                  f"{r['w']:7.4f}")
        big = [r for r in rr if r["w"] >= a.w_min]
        print(f"  Chirikov sigma of neighbours (w >= {a.w_min}):")
        for r1, r2 in zip(big, big[1:]):
            s = (r1["w"] + r2["w"]) / (2 * (r2["rho"] - r1["rho"]))
            print(f"    {r1['n']}/{r1['m']} - {r2['n']}/{r2['m']}: {s:.2f}")
        inb = [r for r in rr if lo <= r["rho"] <= hi]
        cover = sum(r["w"] for r in inb) / (hi - lo)
        lr, th = lines[label][:2]
        sel = (lr >= lo) & (lr <= hi)
        circ = sel & np.isfinite(th)
        print(f"  islands cover {cover:.2f} of the band; traced lines in the band: {sel.sum()}, "
              f"{(sel & ~circ).sum()} in islands; of the {circ.sum()} others, chaotic "
              f"(thickness > {CHAOTIC:g}): {(th[circ] > CHAOTIC).sum()}, median thickness "
              f"{np.median(th[circ]) if circ.any() else np.nan:.1e}")
        dr = lines[label][4]
        core = (lr > 0.1) & (lr < 0.5)
        print(f"  drift of rho over the trace: median {np.median(dr[core]):+.4f} for rho "
              f"0.1-0.5, {np.median(dr[sel]):+.4f} in the band")

    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.colors import LogNorm
    k = len(runs)
    fig = plt.figure(figsize=(6.2 * k, 21), constrained_layout=True)
    gs = fig.add_gridspec(4, k, height_ratios=[1.25, 0.55, 0.55, 0.8])
    norm = LogNorm(1e-4, 3e-2)
    cmap = plt.cm.RdYlBu_r
    for j, (label, *_rest) in enumerate(runs):
        lr, thick, rho, th, dr = lines[label]
        tc = np.nan_to_num(thick, nan=1e-4)  # island lines drawn as surfaces
        ax = fig.add_subplot(gs[0, j])
        col = cmap(norm(np.broadcast_to(tc, rho.shape).ravel()))
        m = np.nan_to_num(rho.ravel(), nan=-1) >= 0.5
        ax.scatter(th.ravel()[m] / np.pi, rho.ravel()[m], s=0.5, c=col[m], lw=0)
        for r in rows[label]:
            if r["w"] >= a.w_min:
                ax.axhspan(r["rho"] - r["w"] / 2, r["rho"] + r["w"] / 2, xmin=0.985, xmax=1,
                           color="k", alpha=0.6)
                ax.text(2.01, r["rho"], f"{r['n']}/{r['m']}", fontsize=7, va="center")
        ax.axhline(lo, color="k", lw=0.6, ls="--")
        ax.axhline(hi, color="k", lw=0.6, ls="--")
        ax.set_xlim(0, 2)
        ax.set_ylim(0.5, 0.9)
        ax.set_xlabel("theta / pi")
        ax.set_ylabel("rho")
        ax.set_title(f"{label}, phi = 0: colour = thickness of the line\n"
                     "(red: chaotic; blue: surfaces or islands); bars: predicted widths")
        ax = fig.add_subplot(gs[1, j])
        ax.semilogy(lr, thick, "o", ms=3, mfc="none")
        ax.axvspan(lo, hi, color="0.85", zorder=-1)
        ax.axhline(CHAOTIC, color="k", lw=0.6, ls=":")
        ax.set_xlim(0.0, 1.0)
        ax.set_ylim(1e-5, 1e-1)
        ax.set_xlabel("rho (mean over each line)")
        ax.set_ylabel("rms of rho about a fit rho(theta)")
        ax.set_title(f"{label}: thickness per line (islands left out)")
        ax = fig.add_subplot(gs[2, j])
        ax.scatter(lr, dr, s=8, c=cmap(norm(tc)), edgecolor="k", lw=0.3)
        ax.axvspan(lo, hi, color="0.85", zorder=-1)
        ax.axhline(0, color="k", lw=0.6)
        ax.set_xlim(0.0, 1.0)
        ax.set_ylim(-0.08, 0.08)
        ax.set_xlabel("rho (mean over each line)")
        ax.set_ylabel(f"drift of rho over the {rho.shape[0]} crossings")
        ax.set_title(f"{label}: secular drift (a divergence-free field has none)")
    ax = fig.add_subplot(gs[3, :])
    labels = {}
    for rr in rows.values():
        for r in rr:
            labels.setdefault((r["n"], r["m"]), r["rho"])
    keys = sorted(labels, key=labels.get)
    for (label, *_rest), mk, dx in zip(runs, "os^dv", np.linspace(-0.2, 0.2, k)):
        rr = rows[label]
        x = np.array([keys.index((r["n"], r["m"])) for r in rr]) + dx
        ax.semilogy(x, [r["b"] for r in rr], mk, mfc="none", label=f"{label}: b_mn")
    ax.set_xticks(range(len(keys)))
    ax.set_xticklabels([f"{n}/{m}\n{labels[(n, m)]:.3f}" for n, m in keys], fontsize=7)
    ax.set_xlabel("chain n/m and DESC's resonant rho")
    ax.set_ylabel("resonant 2 |c_mn| of b^rho")
    ax.legend(fontsize=9)
    ax.set_title(f"resonant forcing of every chain in rho {lo - 0.05:.2f}-{hi + 0.05:.2f}")
    fig.savefig(a.out, dpi=120)
    print(f"saved {a.out}")


if __name__ == "__main__":
    main()
