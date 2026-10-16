"""Resonant forcing against rotational transform: predicted and measured island widths.

Reads the b^rho spectra of ``resonant_field.py`` and, for each field, the Poincare
sections in DESC flux coordinates of ``poincare_flux.py``. For each chain iota = n/m
(n a multiple of NFP; iota < 0 means n < 0 in e^{i (m theta* - n zeta)}):

  - where it sits: DESC's resonance rho_D, and the field's own rho_F, where the
    measured iota of the traced lines crosses n/m, with the slope iota'_F there;
  - the forcing: b_mn = 2 |c_mn| of the resonant harmonic at rho_F (and at rho_D);
  - the predicted full width w = 4 sqrt(b_mn / (m |iota'_F|)) (pendulum, in rho); not
    valid where the shear vanishes (a chain at an iota maximum);
  - the measured width: lines locked to the chain (the phase m theta - 2 pi n k / NFP
    at the k-th crossing of phi = 0 does not wind) give the extent in DESC rho of the
    widest one, after subtracting the wobble of the field's surfaces in DESC's rho
    (the shape of the nearest circulating lines on either side of the chain, without
    its harmonics that are multiples of m). The
    raw extent is shown alongside.

This separates a change of the island from a change of iota (where the chain sits,
and the shear iota' that sets the width) and from a change of b_mn (the forcing).

usage: python resonant_widths.py RESONANT.npz [MORE.npz ...] label=FLUX.npz[:prefix] ...
(prefix: optimized (default) or start). Writes RESONANT_widths.png and a table. The
fields of MORE.npz (e.g. a free-boundary run, on its own geometry) are added, without
their DESC, coils and start; all on the same rho. ``--chains 3/5 9/16 ...`` picks the
chains (m up to the spectra's m_max).
"""

import argparse
import os
import numpy as np

CHAINS = [(7, 3), (6, 3), (11, 6), (5, 3), (9, 6)]
SHEAR_MIN = 0.15  # |d iota / d rho| below this (NCSX 6/9 at the iota maximum): shearless


def crossing(rho, iota, q, near=None):
    """rho and d iota / d rho where the line iota(rho) crosses q (the crossing closest
    to ``near``); a linear fit over the lines within 0.04 gives the slope."""
    o = np.argsort(rho)
    r, i = rho[o], iota[o]
    idx = np.nonzero(np.diff(np.sign(i - q)))[0]
    if not len(idx):
        return np.nan, np.nan
    xs = r[idx] + (q - i[idx]) * (r[idx + 1] - r[idx]) / (i[idx + 1] - i[idx])
    x = xs[np.argmin(np.abs(xs - near))] if near is not None else xs[0]
    sel = np.abs(r - x) < 0.04
    slope = np.polyfit(r[sel], i[sel], 1)[0] if sel.sum() >= 3 else np.nan
    return float(x), float(slope)


def shape(rho, th, m, bins=64):
    """rho - <rho> of one circulating line as a periodic function of theta (binned),
    without the harmonics that are multiples of m (near the chain these are the
    island's own deformation of the neighbouring surfaces, not the wobble)."""
    k = np.floor(np.mod(th, 2 * np.pi) / (2 * np.pi) * bins).astype(int)
    s = np.array([np.mean(rho[k == i]) if np.any(k == i) else np.nan for i in range(bins)])
    c = 2 * np.pi * (np.arange(bins) + 0.5) / bins
    ok = np.isfinite(s)
    s = np.interp(c, c[ok], s[ok], period=2 * np.pi)
    S = np.fft.rfft(s)
    S[::m] = 0
    s = np.fft.irfft(S, bins)
    return lambda t: np.interp(np.mod(t, 2 * np.pi), c, s, period=2 * np.pi)


def measured(flux, prefix, m, n, NFP, rho_F):
    """Width of the chain in DESC rho, after removing the wobble of the field's surfaces:
    (widest corrected extent of a locked line, number of locked lines, widest
    uncorrected extent). The wobble delta(theta) is the mean shape of the nearest
    circulating lines inside and outside the chain."""
    rho, th = flux[f"{prefix}_rho"], flux[f"{prefix}_theta"]
    lr = flux[f"{prefix}_line_rho"]
    dpsi = np.diff(m * th, axis=0) - 2 * np.pi * n / NFP
    dpsi = np.mod(dpsi + np.pi, 2 * np.pi) - np.pi
    psi = np.cumsum(dpsi, 0)
    locked = (np.ptp(psi, 0) < 2 * np.pi) & (np.abs(lr - rho_F) < 0.1)
    if not locked.any():
        return 0.0, 0, 0.0
    lo, hi = lr[locked].min(), lr[locked].max()
    circ = ~locked & (np.ptp(psi, 0) > 4 * np.pi)
    inner = np.flatnonzero(circ & (lr < lo))
    outer = np.flatnonzero(circ & (lr > hi))
    sides = [inner[np.argmax(lr[inner])]] if len(inner) else []
    sides += [outer[np.argmin(lr[outer])]] if len(outer) else []
    fs = [shape(rho[:, j], th[:, j], m) for j in sides]
    delta = (lambda t: sum(f(t) for f in fs) / len(fs)) if fs else (lambda t: 0 * t)
    corr = [np.ptp(rho[:, j] - delta(th[:, j])) for j in np.flatnonzero(locked)]
    return float(max(corr)), int(locked.sum()), float(np.ptp(rho[:, locked], 0).max())


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("files", nargs="+",
                    help="RESONANT.npz [MORE.npz ...], then label=FLUX.npz[:prefix] ...")
    ap.add_argument("--chains", nargs="*", default=None, help="n/m ... (default: CHAINS)")
    a = ap.parse_args()
    res_files = [f for f in a.files if "=" not in f]
    d0 = np.load(res_files[0])
    d = dict(d0)
    for f in res_files[1:]:
        e = np.load(f)
        assert np.allclose(e["rho"], d["rho"]), f"{f}: another rho grid"
        new = [str(r) for r in e["runs"] if str(r) not in ("DESC", "coils", "start")]
        d.update({f"c_{r}": e[f"c_{r}"] for r in new})
        d["runs"] = np.array([str(r) for r in d["runs"]] + new)
    chains = CHAINS if a.chains is None else [tuple(int(v) for v in c.split("/"))[::-1]
                                              for c in a.chains]
    rho, iota_D, NFP = d["rho"], d["iota"], int(d["NFP"])
    sgn = np.sign(np.mean(iota_D))
    n_list = list(d["n"])
    runs = [str(r) for r in d["runs"] if str(r) != "DESC"]
    fluxes = {}
    for item in [f for f in a.files if "=" in f]:
        label, rest = item.split("=", 1)
        path, _, prefix = rest.partition(":")
        fluxes[label] = (np.load(path), prefix or "optimized")

    def b_res(run, m, n, x):
        c = d[f"c_{run}"][:, m, n_list.index(int(sgn * n))]
        return float(np.interp(x, rho, 2 * np.abs(c)))

    rows = []
    print(f"{'run':12s} {'chain':6s} {'rho_D':>6s} {'rho_F':>6s} {'slope':>8s} "
          f"{'b(rho_D)':>9s} {'b(rho_F)':>9s} {'w_pred':>7s} {'w_meas':>7s} {'lock':>4s} "
          f"{'raw':>6s}")
    for run in runs:
        for m, n in chains:
            q = n / m
            rho_D, slope_D = crossing(rho, np.abs(iota_D), q)
            rho_F, slope_F, w_meas, nl, bg = np.nan, np.nan, np.nan, 0, np.nan
            if run in fluxes:
                f, pre = fluxes[run]
                rho_F, slope_F = crossing(f[f"{pre}_line_rho"], f[f"{pre}_line_iota"], q,
                                          near=rho_D if np.isfinite(rho_D) else None)
                if np.isfinite(rho_F):
                    w_meas, nl, bg = measured(f, pre, m, n, NFP, rho_F)
            x, s = (rho_F, slope_F) if np.isfinite(rho_F) else (rho_D, slope_D)
            if not np.isfinite(x):
                continue
            bD = b_res(run, m, n, rho_D) if np.isfinite(rho_D) else np.nan
            bF = b_res(run, m, n, x)
            w = 4 * np.sqrt(bF / (m * abs(s)))
            rows.append(dict(run=run, m=m, n=n, rho_D=rho_D, rho_F=rho_F, slope=s, bD=bD,
                             bF=bF, w=w, w_meas=w_meas, locked=nl, bg=bg))
            print(f"{run:12s} {f'{n}/{m}':6s} {rho_D:6.3f} "
                  f"{rho_F:6.3f} {s:8.3f} {bD:9.2e} {bF:9.2e} {w:7.4f} {w_meas:7.4f} "
                  f"{nl:4d} {bg:6.4f}" + ("  shearless: no pendulum width" if abs(s) < SHEAR_MIN
                                          else ""))

    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fig = plt.figure(figsize=(4.2 * len(chains), 9.5), constrained_layout=True)
    gs = fig.add_gridspec(2, len(chains))
    colors = {r: f"C{i}" for i, r in enumerate(runs)}
    for j, (m, n) in enumerate(chains):
        ax = fig.add_subplot(gs[0, j])
        for run in runs:
            c = d[f"c_{run}"][:, m, n_list.index(int(sgn * n))]
            ax.semilogy(rho, 2 * np.abs(c), color=colors[run], lw=1.2, label=run)
            cnr = d[f"c_{run}"][:, m, n_list.index(int(-sgn * n))]
            ax.semilogy(rho, 2 * np.abs(cnr), color=colors[run], lw=0.6, ls=":")
        for r in rows:
            if (r["m"], r["n"]) == (m, n):
                ax.plot(r["rho_F"] if np.isfinite(r["rho_F"]) else r["rho_D"], r["bF"], "o",
                        color=colors[r["run"]], ms=6, mfc="none")
        rD, _ = crossing(rho, np.abs(iota_D), n / m)
        if np.isfinite(rD):
            ax.axvline(rD, color="k", lw=0.8, ls="--")
        ax.set_title(f"iota = {n}/{m}: |b_{m},{int(sgn * n)}| (solid, resonant)\n"
                     f"|b_{m},{int(-sgn * n)}| (dotted); o: each field's iota = {n}/{m}")
        ax.set_xlabel("DESC rho (dashed: DESC's resonance)")
        if j == 0:
            ax.set_ylabel("2 |c_mn| of b^rho = B.grad rho / B.grad zeta")
            ax.legend(fontsize=7)
    ax = fig.add_subplot(gs[1, :2])
    for r in rows:
        if np.isfinite(r["w_meas"]) and abs(r["slope"]) >= SHEAR_MIN:
            ax.plot(r["w_meas"], r["w"], "o", color=colors[r["run"]])
            ax.annotate(f"{r['n']}/{r['m']}", (r["w_meas"], r["w"]), fontsize=7,
                        xytext=(3, 3), textcoords="offset points")
    ok = [r for r in rows if np.isfinite(r["w_meas"]) and abs(r["slope"]) >= SHEAR_MIN]
    lim = max([r["w"] for r in ok] + [r["w_meas"] for r in ok]) * 1.1
    ax.plot([0, lim], [0, lim], "k:", lw=0.8)
    ax.set_xlim(0, lim)
    ax.set_ylim(0, lim)
    ax.set_xlabel("measured width (widest locked line, DESC rho)")
    ax.set_ylabel("predicted 4 sqrt(b_mn / (m |iota'|))")
    ax.set_title("island widths: forcing x shear against the Poincare sections\n"
                 f"(chains with |iota'| < {SHEAR_MIN} left out: no pendulum width)")
    ax = fig.add_subplot(gs[1, 2:])
    for run in runs:
        if run in fluxes:
            f, pre = fluxes[run]
            lr, li = f[f"{pre}_line_rho"], f[f"{pre}_line_iota"]
            ax.plot(lr, li - np.interp(lr, rho, np.abs(iota_D)), "o", ms=2.5,
                    color=colors[run], label=run)
    ax.axhline(0, color="k", lw=0.8)
    ax.set_xlabel("rho (mean over each traced line)")
    ax.set_ylabel("|iota| (field lines) - |iota| (DESC)")
    ax.set_xlim(0.0, 1.0)
    ax.legend(fontsize=8)
    ax.set_title("rotational transform of each field relative to DESC's\n"
                 "(lines locked in chains sit at n/m exactly)")
    out = os.path.splitext(res_files[0])[0] + "_widths.png"
    fig.savefig(out, dpi=130)
    print(f"saved {out}")


if __name__ == "__main__":
    main()
