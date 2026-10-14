"""Singular-value spectra of the force Jacobian for the float64 tokamak runs.

Reads results/gs/null64_<a10|a20>_<f2|f3>_<start|end>.npz (gs_nullspace.py --run)
and draws figures/gs_null64_spectra.png:
  a. sigma / sigma_max against the index counted from the smallest, every case;
  b. the smallest 60 on a log axis (the near-null tail);
  c. the number of singular values below 1e-4, 1e-6, 1e-8 of the largest.
Prints the same counts.
"""

import glob
import os
import re

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
files = sorted(glob.glob(os.path.join(ROOT, "results", "gs", "null64_*.npz")))
fig, ax = plt.subplots(1, 3, figsize=(17, 5))
style = {"a10": "-", "a20": "--"}
color = {"f2": "C0", "f3": "C3"}
counts = {}
print(f"{'case':16s} {'n':>5s} {'min':>9s}  below 1e-4  1e-6  1e-8  1e-10")
for f in files:
    m = re.search(r"null64_(a\d+)_(f\d)_(start|end)", f)
    if not m:
        continue
    box, fac, at = m.groups()
    sv = np.load(f)["sv"]
    s = sv / sv[0]
    lab = f"{fac[1]}x, {box[0]}/{box[1:]}, {at}"
    kw = dict(ls=style[box], color=color[fac], alpha=1.0 if at == "end" else 0.5,
              lw=1.5 if at == "end" else 1.0)
    ax[0].semilogy(np.arange(len(s))[::-1] / len(s), s, label=lab, **kw)
    ax[1].semilogy(np.arange(60), s[::-1][:60], label=lab, **kw)
    c = [int(np.sum(s < t)) for t in (1e-4, 1e-6, 1e-8, 1e-10)]
    counts[lab] = (len(s), c)
    print(f"{lab:16s} {len(s):5d} {s[-1]:9.1e}  {c[0]:9d} {c[1]:5d} {c[2]:5d} {c[3]:6d}")
ax[0].set_xlabel("index from the smallest / n")
ax[0].set_ylabel(r"$\sigma/\sigma_{max}$ (column-scaled)")
ax[0].set_title("a. whole spectrum")
ax[0].legend(fontsize=7)
ax[1].set_xlabel("index from the smallest")
ax[1].set_title("b. the near-null tail")
labs = list(counts)
x = np.arange(len(labs))
for k, (t, mk) in enumerate(zip((1e-4, 1e-6, 1e-8), ("o", "s", "^"))):
    ax[2].semilogy(x, [max(counts[l][1][k], 0.8) for l in labs], mk + "-", label=f"below {t:g}")
ax[2].set_xticks(x)
ax[2].set_xticklabels(labs, rotation=45, ha="right", fontsize=7)
ax[2].set_title("c. number of small singular values")
ax[2].legend(fontsize=8)
fig.suptitle("force-only Jacobian, tokamak L = M = 24, float64, DESC-matched start and end of the run")
fig.tight_layout()
out = os.path.join(ROOT, "figures", "gs_null64_spectra.png")
fig.savefig(out, dpi=140)
print("saved", out)
