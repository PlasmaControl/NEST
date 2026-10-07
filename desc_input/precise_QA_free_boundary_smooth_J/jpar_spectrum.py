"""Fourier spectrum of J_parallel on surfaces near the edge, in straight-field-line
(PEST) angles, to identify resonant structures (m, n) with iota ~ n/m.

J_par is evaluated on a uniform (theta_PEST, zeta) grid (theta mapped back to DESC's
theta with map_coordinates), FFT'd, and the largest harmonics printed, with the
radius where iota = n/m for each.

usage: python jpar_spectrum.py EQ.h5 [RHO ...]
"""

import sys

import numpy as np

import desc.io
from desc.grid import Grid, LinearGrid

eq = desc.io.load(sys.argv[1])
rhos = [float(r) for r in sys.argv[2:]] or [0.9, 0.95, 0.98, 1.0]
nt, nz = 96, 96  # zeta over the full torus so that n is the true toroidal mode number
r1 = np.linspace(0, 1, 2001)
g1 = LinearGrid(rho=r1, M=eq.M_grid, N=eq.N_grid, NFP=eq.NFP)
iota = g1.compress(eq.compute("iota", grid=g1)["iota"])
print(f"iota(0) = {iota[0]:.4f}, iota(1) = {iota[-1]:.4f}")
for rho in rhos:
    tp, z = np.meshgrid(np.linspace(0, 2 * np.pi, nt, endpoint=False),
                        np.linspace(0, 2 * np.pi, nz, endpoint=False), indexing="ij")
    pest = np.c_[np.full(tp.size, rho), tp.ravel(), z.ravel()]
    desc_nodes = eq.map_coordinates(pest, inbasis=("rho", "theta_PEST", "zeta"),
                                    outbasis=("rho", "theta", "zeta"))
    g = Grid(np.asarray(desc_nodes), sort=False)
    d = eq.compute(["J*B", "|B|"], grid=g)
    jpar = np.asarray(d["J*B"] / d["|B|"]).reshape(nt, nz)
    c = np.fft.fft2(jpar) / jpar.size
    m = np.fft.fftfreq(nt, 1 / nt).astype(int)
    n = np.fft.fftfreq(nz, 1 / nz).astype(int)
    P = np.abs(c)
    P[0, 0] = 0  # surface mean
    print(f"\nrho = {rho}: rms J_par = {np.sqrt(np.mean(jpar**2)) / 1e3:.1f} kA/m^2; "
          "largest harmonics exp(i(m theta* + n zeta)):")
    seen = set()
    for i in np.argsort(P.ravel())[::-1]:
        a, b = np.unravel_index(i, P.shape)
        mm, nn = int(m[a]), int(n[b])
        if mm < 0 or (mm == 0 and nn < 0):
            mm, nn = -mm, -nn
        if (mm, nn) in seen:
            continue
        seen.add((mm, nn))
        res = ""
        if mm > 0 and nn != 0:
            t = abs(nn) / mm
            idx = np.nonzero(np.diff(np.sign(iota - t)))[0]
            res = (f"  iota = {abs(nn)}/{mm} = {t:.4f} at rho = "
                   + (", ".join(f"{r1[k]:.3f}" for k in idx) if len(idx) else "none"))
        print(f"  (m, n) = ({mm:3d}, {nn:4d})  {2 * P[a, b] / 1e3:7.1f} kA/m^2{res}")
        if len(seen) >= 10:
            break
