"""Collect sheet_diag.py outputs into a table and figures.

python sheet_plot.py DIAGDIR [--figs FIGDIR]
"""
import argparse
import glob
import json
import os
import re

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

ap = argparse.ArgumentParser()
ap.add_argument("diag")
ap.add_argument("--figs", default=None)
a = ap.parse_args()
figs = a.figs or a.diag

rows = []
for j in sorted(glob.glob(os.path.join(a.diag, "*_sheet.json"))):
    d = json.load(open(j))
    d["tag"] = os.path.basename(j)[:-11]
    d["npz"] = j[:-5] + ".npz"
    rows.append(d)
rows.sort(key=lambda d: (d["L"], d["N"], d["tag"]))

hdr = ("tag", "L", "N", "force vol", "peak |s21|", "peak off", "FWHM", "FWHM*L", "int .005", "int .01", "int .02",
       "int .04", "ring .02-.05", "max outside", "F at rho_s")
print(" | ".join(hdr))
for d in rows:
    I = d["integral signed"]
    print(" | ".join([d["tag"], str(d["L"]), str(d["N"]), f"{d['force vol']:.2e}", f"{d['peak |sigma_21|']:.4g}",
                      f"{d['peak offset']:+.1e}", f"{d['FWHM']:.2e}", f"{d['FWHM'] * d['L']:.2f}",
                      f"{I['0.005']:.4g}", f"{I['0.01']:.4g}", f"{I['0.02']:.4g}", f"{I['0.04']:.4g}",
                      f"{d['max |sigma_21| at 0.02<|rho-rho_s|<=0.05']:.3g}", f"{d['max |sigma_21| outside window']:.3g}",
                      f"{d['force at rho_s']:.2e}"]))

cm = plt.get_cmap("viridis")
fig, ax = plt.subplots(2, 2, figsize=(12, 8.5))
sel = [d for d in rows if d["tag"].startswith("ripple")]
for i, d in enumerate(sel):
    z = np.load(d["npz"])
    rho, s21, rs = z["rho"], z["s21"], float(z["rho_s"])
    sg = np.real(s21 * np.exp(-1j * d["peak phase"]))
    c = cm(i / max(len(sel) - 1, 1))
    lab = f"L=M={d['L']}, N={d['N']}"
    ax[0, 0].plot(rho, np.abs(s21), color=c, lw=1, label=lab)
    ax[0, 1].plot(rho - rs, sg, color=c, lw=1, label=lab)
    x = (rho - rs)
    s = np.abs(x) <= 0.08
    cum = np.concatenate([[0], np.cumsum(0.5 * (sg[s][1:] + sg[s][:-1]) * np.diff(x[s]))])
    ax[1, 0].plot(x[s], cum - np.interp(0, x[s], cum), color=c, lw=1, label=lab)
    ax[1, 1].semilogy(rho, z["F"] / z["Fn"], color=c, lw=1, label=lab)
ax[0, 0].axvline(sel[0]["rho_s"] if sel else 0.6375, color="k", ls=":", lw=0.8)
ax[0, 0].set(xlabel=r"$\rho$", ylabel=r"$|\sigma_{21}|$", title=r"(2,1) harmonic of $\sigma=J\cdot B/B^2$ (PEST)")
ax[0, 1].set(xlim=(-0.06, 0.06), xlabel=r"$\rho-\rho_s$", ylabel=r"Re$[\sigma_{21} e^{-i\phi_{peak}}]$",
             title="zoom on q = 2, signed")
ax[1, 0].set(xlabel=r"$\rho-\rho_s$", ylabel=r"$\int_{\rho_s}^{\rho}\sigma_{21}\,d\rho'$",
             title="running integral (a sheet = a step)")
ax[1, 1].set(xlabel=r"$\rho$", ylabel=r"$\langle|F|\rangle_s/\langle|\nabla B^2/2\mu_0|\rangle_s$",
             title="force error by surface")
for x_ in ax.flat:
    x_.legend(fontsize=7)
fig.tight_layout()
fig.savefig(os.path.join(figs, "sheet_profiles.png"), dpi=150)

# scaling with L at the lowest-N-converged entries: peak, FWHM, integral
fig, ax = plt.subplots(1, 3, figsize=(13, 4))
for N in sorted({d["N"] for d in sel}):
    ss = sorted([d for d in sel if d["N"] == N], key=lambda d: d["L"])
    Ls = np.array([d["L"] for d in ss])
    ax[0].loglog(Ls, [d["peak |sigma_21|"] for d in ss], "o-", label=f"N={N}")
    ax[1].loglog(Ls, [d["FWHM"] for d in ss], "o-", label=f"N={N}")
    for k, m in (("0.01", "o-"), ("0.04", "s--")):
        ax[2].semilogx(Ls, [d["integral signed"][k] for d in ss], m, label=f"N={N}, |dr|<={k}")
if sel:
    Ls = np.array(sorted({d["L"] for d in sel}))
    p0 = sel[0]
    ax[0].loglog(Ls, p0["peak |sigma_21|"] * Ls / p0["L"], "k:", label=r"$\propto L$")
    ax[1].loglog(Ls, p0["FWHM"] * p0["L"] / Ls, "k:", label=r"$\propto 1/L$")
ax[0].set(xlabel="L = M", ylabel=r"peak $|\sigma_{21}|$")
ax[1].set(xlabel="L = M", ylabel="FWHM in rho")
ax[2].set(xlabel="L = M", ylabel=r"$\int\sigma_{21}d\rho$ across q=2")
for x_ in ax:
    x_.legend(fontsize=7)
fig.tight_layout()
fig.savefig(os.path.join(figs, "sheet_scaling.png"), dpi=150)
print("figures in", figs)
