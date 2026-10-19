"""Gate A6: scalar free-space solve on the annulus against Gaussian charges.

f = sum over the NFP rotated copies of two off-axis, z-asymmetric Gaussian charges
(width s, charge q; no z or stellarator symmetry). The exact free-space solution of
Delta u = f is u = sum -q erf(rho / (sqrt 2 s)) / (4 pi rho). Checked:

  (a) g = S sigma against u_exact on the boundary (isolates S)
  (b) u against u_exact at the nodes and at random off-node points (Fourier in phi,
      barycentric Chebyshev in r, z)
  (c) convergence in (n_r = n_z = n, N_phi)
  (d) source-to-wall distance: the e^{-1/2 (d / s)^2} tail on the wall from 1e-8 to 1e-18

for shift 0 (real weights, modes j NFP) and shift 1 (weights e^{2 pi i k / NFP} on
copy k, modes j NFP + 1: the A_+ path).

run: python tests/checks/cyl_a6.py [--quick] [--out results/cyl/a6.json]
"""

import argparse
import json
import os
import sys
import time

import numpy as np
from scipy.special import erf

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.join(ROOT, "nest"))

from annulus_freespace import FreeSpaceAnnulus, mode_indices  # noqa: E402
from chebyshev import ChebAxis  # noqa: E402

BOX = dict(R_in=0.6, R_out=1.4, z0=0.45, NFP=2)
# base charges (r, phi, z, s, q)
CHARGES = [(0.95, 0.3, 0.12, 0.035, 1.0), (1.12, 1.2, -0.18, 0.03, -0.6)]


def copies(charges, NFP, shift):
    out = []
    for (r, p, z, s, q) in charges:
        for k in range(NFP):
            w = np.exp(2j * np.pi * k * shift / NFP)
            out.append((r, p + 2 * np.pi * k / NFP, z, s, q * w))
    return out


def gauss_f(X, Y, Z, src):
    f = 0j
    for (r, p, z, s, q) in src:
        d2 = (X - r * np.cos(p)) ** 2 + (Y - r * np.sin(p)) ** 2 + (Z - z) ** 2
        f = f + q * np.exp(-d2 / (2 * s**2)) / (2 * np.pi * s**2) ** 1.5
    return f


def gauss_u(X, Y, Z, src):
    u = 0j
    for (r, p, z, s, q) in src:
        rho = np.sqrt((X - r * np.cos(p)) ** 2 + (Y - r * np.sin(p)) ** 2 + (Z - z) ** 2)
        small = rho < 1e-6 * s
        rs = np.where(small, 1.0, rho)
        val = np.where(small, np.sqrt(2 / np.pi) / s, erf(rs / (np.sqrt(2) * s)) / rs)
        u = u - q * val / (4 * np.pi)
    return u


def grid_xyz(fs):
    P, R, Zg = np.meshgrid(fs.phi, fs.ap.r_axis.s, fs.ap.z_axis.s, indexing="ij")
    return R * np.cos(P), R * np.sin(P), Zg, P


def wall_tail(charges, box):
    """max over charges of exp(-d^2 / 2 s^2), d the distance to the nearest wall."""
    t = 0.0
    for (r, p, z, s, q) in charges:
        d = min(r - box["R_in"], box["R_out"] - r, box["z0"] - z, z + box["z0"])
        t = max(t, np.exp(-0.5 * (d / s) ** 2))
    return t


def evaluate_off_node(fs, U_modes, j, shift, pts):
    """u at (r, phi, z) points from the modal slabs (Fourier in phi, Chebyshev in r, z)."""
    r, p, z = pts.T
    Ir = fs.ap.r_axis.interp_matrix(r)
    Iz = fs.ap.z_axis.interp_matrix(z)
    # per point: sum_j e^{i j NFP phi} (Ir U_j Iz^T)
    vals = np.einsum("pa,jab,pb->pj", Ir, U_modes, Iz)
    e = np.exp(1j * np.outer(p, j * fs.NFP))
    if shift == 0:
        w = np.where(j == 0, 1.0, 2.0)
        return np.real(np.sum(w * vals * e, axis=1)) / fs.N_phi
    return np.sum(vals * e, axis=1) / fs.N_phi


def run_case(n, N_phi, charges=CHARGES, box=BOX, shifts=(0, 1), n_off=300, seed=0, fs=None):
    ra = ChebAxis(box["R_in"], box["R_out"], n)
    za = ChebAxis(-box["z0"], box["z0"], n)
    t0 = time.time()
    fs = fs or FreeSpaceAnnulus(ra, za, box["NFP"], N_phi, shifts=shifts)
    t_pre = time.time() - t0
    X, Y, Zg, P = grid_xyz(fs)
    rng = np.random.default_rng(seed)
    off = np.column_stack([rng.uniform(box["R_in"], box["R_out"], n_off),
                           rng.uniform(0, 2 * np.pi / box["NFP"], n_off),
                           rng.uniform(-box["z0"], box["z0"], n_off)])
    res = {"n": n, "N_phi": N_phi, "precompute_s": t_pre, "S_MB": fs.S_nbytes / 1e6,
           "wall_tail": wall_tail(charges, box)}
    for s in shifts:
        src = copies(charges, box["NFP"], s)
        f = gauss_f(X, Y, Zg, src)
        uex = gauss_u(X, Y, Zg, src)
        if s == 0:
            assert np.abs(f.imag).max() == 0 and np.abs(uex.imag).max() == 0
            f, uex = f.real, uex.real
        else:  # the NFP-periodic factor e^{-i s phi} f
            f, uex = f * np.exp(-1j * s * P), uex * np.exp(-1j * s * P)
        t1 = time.time()
        u, parts = fs.solve(f, shift=s, return_parts=True)
        t_solve = time.time() - t1
        scale = np.abs(uex).max()
        # (a) g against u_exact on the boundary, mode by mode -> back to the planes
        j = mode_indices(N_phi, real=(s == 0))
        g = fs.ap.unpack(parts["g"])
        gb = {k: (np.fft.irfft(v, n=N_phi, axis=0) if s == 0 else np.fft.ifft(v, axis=0)) for k, v in g.items()}
        ub = fs.ap.faces(uex)
        e_a = max(np.abs(gb[k] - ub[k]).max() for k in gb) / scale
        # (b) nodes and off-node points
        e_b = np.abs(u - uex).max() / scale
        U = (np.fft.rfft(u, axis=0) if s == 0 else np.fft.fft(u, axis=0))
        Xo, Yo = off[:, 0] * np.cos(off[:, 1]), off[:, 0] * np.sin(off[:, 1])
        uo = gauss_u(Xo, Yo, off[:, 2], src)
        if s == 0:
            uo = uo.real
        else:
            uo = uo * np.exp(-1j * s * off[:, 1])
        e_off = np.abs(evaluate_off_node(fs, U, j, s, off) - uo).max() / scale
        res[f"shift{s}"] = {"boundary_g": e_a, "nodes": e_b, "off_node": e_off, "solve_s": t_solve}
    return res, fs


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--quick", action="store_true")
    ap.add_argument("--out", default=os.path.join(ROOT, "results", "cyl", "a6.json"))
    a = ap.parse_args()
    out = {"box": BOX, "charges": CHARGES, "convergence": [], "wall": []}
    sizes = [(32, 61), (48, 101), (64, 151), (80, 201), (96, 251)]
    if a.quick:
        sizes = sizes[:3]
    print(f"A6 (c) convergence; wall tail {wall_tail(CHARGES, BOX):.1e}", flush=True)
    for n, N_phi in sizes:
        r, _ = run_case(n, N_phi)
        out["convergence"].append(r)
        print(f"  n={n:3d} N_phi={N_phi:3d}: precompute {r['precompute_s']:6.1f} s (S {r['S_MB']:.0f} MB); "
              + "; ".join(f"shift {s}: g {r[f'shift{s}']['boundary_g']:.1e} nodes {r[f'shift{s}']['nodes']:.1e} "
                          f"off {r[f'shift{s}']['off_node']:.1e} ({r[f'shift{s}']['solve_s']:.1f} s)"
                          for s in (0, 1)), flush=True)
    # (d) source-to-wall distance at the finest size: move charge 1 toward the top cap
    n, N_phi = sizes[-1]
    print("A6 (d) wall distance", flush=True)
    fs = None
    for tail in (1e-8, 1e-12, 1e-16, 1e-18):
        s1 = CHARGES[0][3]
        d = s1 * np.sqrt(-2 * np.log(tail))
        ch = [(CHARGES[0][0], CHARGES[0][1], BOX["z0"] - d, s1, CHARGES[0][4]), CHARGES[1]]
        r, fs = run_case(n, N_phi, charges=ch, shifts=(0,), fs=fs)
        r["tail_target"] = tail
        out["wall"].append(r)
        print(f"  tail {tail:.0e} (d = {d / s1:.1f} s): g {r['shift0']['boundary_g']:.1e} "
              f"nodes {r['shift0']['nodes']:.1e} off {r['shift0']['off_node']:.1e}", flush=True)
    os.makedirs(os.path.dirname(a.out), exist_ok=True)
    with open(a.out, "w") as fh:
        json.dump(out, fh, indent=1)
    print(f"saved {a.out}")


if __name__ == "__main__":
    main()
