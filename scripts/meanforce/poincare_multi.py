"""Poincare sections of several traced fields (poincare_fixed.py npz) over DESC's surfaces.

usage: python poincare_multi.py EQ.h5 SECTIONS.npz OUT.png name[:label] name[:label] ...
One column per field, rows phi = 0 and phi = pi / NFP. DESC's rho = 0.1..0.9 grey, LCFS black.
"""
import sys

import numpy as np

from desc import set_device

set_device("cpu")

import desc.io  # noqa: E402
from desc.grid import LinearGrid  # noqa: E402


def main(eq_path, sec_path, out, specs):
    eq = desc.io.load(eq_path)
    d = np.load(sec_path)
    runs = [(s.split(":", 1) + [s])[:2] for s in specs]
    runs = [(k, lab) for k, lab in runs if f"{k}_R" in d]
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    planes = [("", 0.0, "φ = 0"), ("h", np.pi / eq.NFP, "φ = π/NFP")]
    fig, axs = plt.subplots(2, len(runs), figsize=(4.2 * len(runs), 9.5), constrained_layout=True,
                            squeeze=False, gridspec_kw={"height_ratios": [1.45, 1]})
    th = np.linspace(0, 2 * np.pi, 361)
    for i, (suffix, phi, label) in enumerate(planes):
        surf = {}
        for r in np.r_[np.arange(0.1, 1.0, 0.1), 1.0]:
            g = LinearGrid(rho=np.array([r]), theta=th, zeta=np.array([phi]), NFP=eq.NFP)
            s = eq.compute(["R", "Z"], grid=g)
            surf[r] = (s["R"], s["Z"])
        for j, (k, lab) in enumerate(runs):
            ax = axs[i, j]
            R, Z = d[f"{k}_R{suffix}"], d[f"{k}_Z{suffix}"]
            if suffix == "":
                R, Z = R[1:], Z[1:]
            line = np.broadcast_to(np.arange(R.shape[1]), R.shape)
            ok = np.isfinite(R)
            for r, (sR, sZ) in surf.items():
                ax.plot(sR, sZ, color="k" if r == 1.0 else "0.65", lw=1.0 if r == 1.0 else 0.5, zorder=1)
            ax.scatter(R[ok], Z[ok], s=0.35, c=plt.cm.turbo((line[ok] * 0.618) % 1), lw=0, zorder=2,
                       rasterized=True)
            sR, sZ = surf[1.0]
            pad = 0.04
            ax.set_xlim(sR.min() - pad, sR.max() + pad)
            ax.set_ylim(sZ.min() - pad, sZ.max() + pad)
            ax.set_aspect("equal")
            ax.set_xlabel("R [m]")
            if j == 0:
                ax.set_ylabel(f"Z [m]   ({label})")
            if i == 0:
                ax.set_title(lab, loc="left")
    fig.savefig(out, dpi=170)
    print(f"saved {out}")


if __name__ == "__main__":
    main(sys.argv[1], sys.argv[2], sys.argv[3], sys.argv[4:])
