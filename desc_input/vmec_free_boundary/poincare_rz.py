"""Poincare sections of ``poincare.py`` in (R, Z) over DESC's flux surfaces.

One row per plane (phi = 0 and phi = pi / NFP), one column per traced field (the
start and the optimized x). The crossings are colored by field line. DESC's
surfaces rho = 0.1, ..., 0.9 are drawn as thin grey lines and the LCFS in black.
Where the field lines follow the grey lines, the solution keeps DESC's surfaces;
islands and chaotic layers show where it does not.

usage: python poincare_rz.py EQ.h5 SECTIONS_poincare.npz
Writes SECTIONS_rz.png next to the sections. Runs on a CPU in seconds.
"""

import sys

import numpy as np

from desc import set_device

set_device("cpu")

import desc.io  # noqa: E402
from desc.grid import LinearGrid  # noqa: E402


def main(eq_path, sec_path):
    eq = desc.io.load(eq_path)
    d = np.load(sec_path)
    runs = [k for k in ("start", "optimized") if f"{k}_R" in d]
    planes = [("", 0.0, "phi = 0")]
    if f"{runs[0]}_Rh" in d:
        planes.append(("h", np.pi / eq.NFP, "phi = pi / NFP"))

    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    titles = {"start": "start (fits of DESC's J and p)", "optimized": "NEST optimized"}
    fig, axs = plt.subplots(len(planes), len(runs), figsize=(7 * len(runs), 7.5 * len(planes)),
                            constrained_layout=True, squeeze=False)
    th = np.linspace(0, 2 * np.pi, 361)
    for i, (suffix, phi, label) in enumerate(planes):
        surf = {}
        for r in np.r_[np.arange(0.1, 1.0, 0.1), 1.0]:
            g = LinearGrid(rho=np.array([r]), theta=th, zeta=np.array([phi]), NFP=eq.NFP)
            s = eq.compute(["R", "Z"], grid=g)
            surf[r] = (s["R"], s["Z"])
        for j, k in enumerate(runs):
            ax = axs[i, j]
            R, Z = d[f"{k}_R{suffix}"], d[f"{k}_Z{suffix}"]
            if suffix == "":
                R, Z = R[1:], Z[1:]  # the first row is the seeds
            line = np.broadcast_to(np.arange(R.shape[1]), R.shape)
            ok = np.isfinite(R)
            for r, (sR, sZ) in surf.items():
                ax.plot(sR, sZ, color="k" if r == 1.0 else "0.6", lw=1.0 if r == 1.0 else 0.5,
                        zorder=1)
            ax.scatter(R[ok], Z[ok], s=0.4, c=plt.cm.turbo((line[ok] * 0.618) % 1), lw=0,
                       zorder=2)
            ax.set_aspect("equal")
            ax.set_xlabel("R [m]")
            ax.set_ylabel("Z [m]")
            ax.set_title(f"{titles[k]}, {label}")
    fig.suptitle(f"{eq_path.split('/')[-1]}: field lines of B_ext + B_plasma (colored by line)\n"
                 "over DESC's flux surfaces rho = 0.1, ..., 0.9 (grey) and the LCFS (black)")
    out = sec_path[: -len(".npz")] + "_rz.png"
    fig.savefig(out, dpi=150)
    print(f"saved {out}")


if __name__ == "__main__":
    main(*sys.argv[1:3])
