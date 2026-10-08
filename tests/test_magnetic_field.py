"""Step 5: B_plasma from J on the box G3, periodic FFT vs free-space convolution.

J = curl T (``CurrentPotential``) on G3 with h = a / frac, then B_plasma
by both methods of ``magnetic_field``, compared with three references:

  1. discrete Biot-Savart: the direct sum over the same grid J at sample points
     inside and outside the plasma (within its bounding box, where the free
     space convolution has no images). The free-space convolution must match it
     to float32 round-off; for the periodic FFT the difference is the field of
     the periodic images.
  2. continuous Biot-Savart outside the plasma: the volume integral of the same
     J = curl T in flux coordinates (Gauss quadrature, no staircase) at grid
     points in a shell 0.3 a to 0.6 a from the plasma (fixed in space, so the
     error converges with h), where the free-space
     convolution has no images. Measures the discretization of J on the grid
     (staircase edge) near the plasma, plus the images for the periodic FFT.
  3. inside the plasma (free-boundary precise_QA, which has its coils): the
     total field B_coil + B_plasma vs DESC's B, as |B_desc - B| / |B_desc| at
     each point (max and rms). Printed only: it is the field error the force
     residual will see, limited by DESC's free-boundary residual (B.n / |B|
     rms 2.45e-3 for the smooth-current precise_QA) rather than by B_plasma.

Errors of checks 1 and 2 are max |B - B_ref| / max |B_ref| over the points.

run: pytest tests/test_magnetic_field.py -s     or     python tests/test_magnetic_field.py [frac ...]
"""

import os
import sys
import time

import numpy as np

try:  # pytest is optional: the file also runs as a plain script
    import pytest
except ImportError:
    pass

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, "nest"))

from desc import set_device  # noqa: E402

# DESC uses the CPU unless set_device('gpu') runs before any other DESC import
set_device(os.environ.get("NEST_DEVICE", "cpu"))

import desc.io  # noqa: E402
from desc.grid import Grid, LinearGrid, QuadratureGrid  # noqa: E402
from desc.magnetic_fields._core import biot_savart_general  # noqa: E402
from scipy.spatial import cKDTree  # noqa: E402

from cartesian_grid import cartesian_grid, current_on_cartesian  # noqa: E402
from current_potential import CurrentPotential  # noqa: E402
from fourier import fft_J  # noqa: E402
from magnetic_field import (  # noqa: E402
    b_plasma_free,
    b_plasma_periodic,
    biot_savart_kernel_hat,
)

COILS = os.path.join(ROOT, "desc_input", "precise_QA_free_boundary",
                     "precise_QA_helical_coils.h5")
EQ_FILES = {  # equilibrium: its coils (None: no coils, no interior check)
    os.path.join(ROOT, "desc_input", "precise_QA_free_boundary_smooth_J",
                 "precise_QA_free_boundary_smooth_J_edgeJ_r0.3.h5"): COILS,
    os.path.join(ROOT, "desc_input", "W7-X_output.h5"): None,
}
TOL_DISCRETE = 1e-4  # free space vs direct sum (float32)
TOL_EXTERIOR = 2e-2  # free space vs continuous Biot-Savart at h = a/10


def rel(B, B_ref):
    return np.abs(B - B_ref).max() / np.abs(B_ref).max()


def timed(f, *args):
    t0 = time.time()
    out = f(*args)
    out.block_until_ready()
    return np.asarray(out), time.time() - t0


def rpz_to_xyz(V, phi):
    c, s = np.cos(phi), np.sin(phi)
    return np.column_stack([V[:, 0] * c - V[:, 1] * s, V[:, 0] * s + V[:, 1] * c, V[:, 2]])


def continuous_biot_savart(eq, current, x, chunk=20):
    """B of J = curl T at Cartesian points x (outside the plasma), by Gauss
    quadrature in flux coordinates over the whole torus."""
    b = current.basis
    g = QuadratureGrid(L=b.L, M=b.M, N=b.N, NFP=eq.NFP)
    d = eq.compute(["R", "phi", "Z", "sqrt(g)"], grid=g)
    J = current.evaluate(g)
    dV = np.abs(np.asarray(d["sqrt(g)"])) * np.asarray(g.weights) / eq.NFP
    rs, Js = [], []
    for k in range(eq.NFP):  # copy the period around the torus
        phi = np.asarray(d["phi"]) + 2 * np.pi * k / eq.NFP
        R = np.asarray(d["R"])
        rs.append(np.column_stack([R * np.cos(phi), R * np.sin(phi), np.asarray(d["Z"])]))
        Js.append(rpz_to_xyz(J, phi))
    return np.asarray(biot_savart_general(x, np.concatenate(rs), np.concatenate(Js),
                                          np.tile(dV, eq.NFP), chunk_size=chunk))


def iota_at(eq, rho, deg=24):
    """(iota, d iota / d rho) at the radii ``rho``: from the profile when iota is
    prescribed, else a Chebyshev fit of iota computed on 64 flux surfaces."""
    if eq.iota is not None:
        g = Grid(np.column_stack([rho, 0 * rho, 0 * rho]), sort=False)
        return [np.asarray(eq.iota.compute(g, dr=i)) for i in range(2)]
    r = (1 - np.cos(np.pi * (np.arange(64) + 0.5) / 64)) / 2
    gr = LinearGrid(rho=r, M=eq.M_grid, N=eq.N_grid, NFP=eq.NFP)
    f = np.polynomial.Chebyshev.fit(r, gr.compress(eq.compute("iota", grid=gr)["iota"]),
                                    deg, domain=[0, 1])
    return [f(rho), f.deriv()(rho)]


def check_magnetic_field(path, coils=None, frac=10, n_sample=100, seed=0):
    eq = desc.io.load(path)
    eq = eq[-1] if hasattr(eq, "__len__") else eq
    a = float(eq.compute("a")["a"])
    h = a / frac
    current = CurrentPotential.fit(eq)
    axes = cartesian_grid(eq, h)
    d = current_on_cartesian(eq, current, axes)
    shape = d["J"].shape[:3]
    J_hat = fft_J(d["J"])
    B_per, t_per = timed(b_plasma_periodic, J_hat, axes)
    K_hat, t_K = timed(biot_savart_kernel_hat, axes)
    B_free, t_free = timed(b_plasma_free, J_hat, axes, K_hat)
    print(f"\n{os.path.basename(path)}: box {shape}, h = a/{frac} = {h * 1e3:.2f} mm; "
          f"time: periodic {t_per:.2f} s, kernel {t_K:.2f} s, free space {t_free:.2f} s")

    X = np.stack(np.meshgrid(*axes, indexing="ij"), axis=-1).reshape(-1, 3)
    B_per, B_free = B_per.reshape(-1, 3), B_free.reshape(-1, 3)
    inside = d["inside"]
    rng = np.random.default_rng(seed)
    print(f"inside the plasma: max |B_periodic - B_free| / max |B_free| = "
          f"{rel(B_per[inside], B_free[inside]):.1e}, max |B_free| = "
          f"{np.abs(B_free[inside]).max() * 1e3:.2f} mT")

    # points outside the plasma in a shell 0.3 a to 0.6 a from it, where every
    # source is within half a period in each direction (no images)
    L = np.array([len(ax) * h for ax in axes])
    lo, hi = X[inside].max(axis=0) - L / 2, X[inside].min(axis=0) + L / 2
    box = np.flatnonzero(np.all((X > lo) & (X < hi), axis=1))
    dist, _ = cKDTree(X[inside]).query(X[box], distance_upper_bound=0.65 * a)
    shell = box[(dist >= 0.3 * a) & (dist <= 0.6 * a)]
    out = rng.choice(shell, min(n_sample, len(shell)), replace=False)

    # 1. direct sum over the grid J
    pts = np.concatenate([rng.choice(inside, n_sample, replace=False), out])
    Jg = d["J"].reshape(-1, 3)[inside].astype(np.float64)
    B_dir = np.asarray(biot_savart_general(X[pts], X[inside], Jg, np.full(len(inside), h**3),
                                           chunk_size=10))
    e_free, e_per = rel(B_free[pts], B_dir), rel(B_per[pts], B_dir)
    print(f"1. vs direct grid sum ({len(pts)} pts): free space {e_free:.1e}, "
          f"periodic {e_per:.1e}")

    # 2. continuous Biot-Savart outside the plasma
    B_ext = continuous_biot_savart(eq, current, X[out])
    e2_free, e2_per = rel(B_free[out], B_ext), rel(B_per[out], B_ext)
    print(f"2. vs continuous Biot-Savart outside ({len(out)} of {len(shell)} pts, 0.3a to 0.6a away): "
          f"free space {e2_free:.1e}, periodic {e2_per:.1e}")

    # 3. inside: DESC B vs B_coil + B_plasma, relative to |B|
    if coils is not None:
        coilset = desc.io.load(coils)
        rtz, x_in = d["rtz"], X[inside]
        sub = rng.choice(len(inside), min(20000, len(inside)), replace=False)
        phi = np.arctan2(x_in[sub, 1], x_in[sub, 0])
        g = Grid(rtz[sub], sort=False)
        # iota passed in: computing it on scattered points needs surface integrals
        iota = dict(zip(["iota", "iota_r"], iota_at(eq, rtz[sub, 0])))
        B_desc = np.asarray(eq.compute("B", grid=g, data=iota)["B"])
        R = np.hypot(x_in[sub, 0], x_in[sub, 1])
        B_coil = np.asarray(coilset.compute_magnetic_field(
            np.column_stack([R, phi, x_in[sub, 2]]), basis="rpz"))
        B_desc, B_coil = rpz_to_xyz(B_desc, phi), rpz_to_xyz(B_coil, phi)
        idx = inside[sub]
        core = rtz[sub, 0] <= 0.8
        mod = np.linalg.norm(B_desc, axis=1)
        print(f"3. B_coil + B_plasma vs DESC B inside ({len(sub)} pts, |B| "
              f"{mod.min():.2f} to {mod.max():.2f} T), |dB| / |B| max (rms):")
        for name, B in [("free space", B_free), ("periodic", B_per)]:
            e = np.linalg.norm(B_coil + B[idx] - B_desc, axis=1) / mod
            print(f"   {name}: all {e.max():.1e} ({np.sqrt(np.mean(e**2)):.1e}), "
                  f"rho <= 0.8 {e[core].max():.1e} ({np.sqrt(np.mean(e[core]**2)):.1e})")
        e = np.linalg.norm(B_coil - B_desc, axis=1) / mod
        print(f"   coils only (B_plasma = 0): all {e.max():.1e} ({np.sqrt(np.mean(e**2)):.1e})")

    assert e_free < TOL_DISCRETE
    assert frac < 10 or e2_free < TOL_EXTERIOR


try:
    @pytest.mark.parametrize("path", list(EQ_FILES),
                             ids=[os.path.basename(f) for f in EQ_FILES])
    def test_magnetic_field(path):
        check_magnetic_field(path, EQ_FILES[path])
except NameError:
    pass


if __name__ == "__main__":
    for frac in [int(f) for f in sys.argv[1:]] or [10]:
        for path, coils in EQ_FILES.items():
            check_magnetic_field(path, coils, frac)
