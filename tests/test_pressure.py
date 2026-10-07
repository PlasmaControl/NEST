"""Pressure p as a Fourier-Zernike series (``Pressure``) vs DESC's p.

For the free-boundary precise_QA and W7-X:
  1. fit: p and grad p on the flux grid G1 vs DESC's p and grad(p)
  2. p = 0 on rho = 1: holds for the fit, and ``impose_zero_edge`` enforces it
     on random coefficients
  3. on the Cartesian box (h = a / 10): grad p at the inside nodes vs DESC at
     the same flux coordinates, and int p dV (staircase edge) vs flux coordinates

run: pytest tests/test_pressure.py -s     or     python tests/test_pressure.py
"""

import os
import sys

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

from cartesian_grid import (  # noqa: E402
    cartesian_grid,
    current_on_cartesian,
    pressure_on_cartesian,
)
from current_potential import CurrentPotential  # noqa: E402
from desc_grid import flux_grid  # noqa: E402
from pressure import Pressure  # noqa: E402

EQ_FILES = [
    os.path.join(ROOT, "desc_input", "precise_QA_free_boundary_L16N14.h5"),
    os.path.join(ROOT, "desc_input", "W7-X_output.h5"),
]
TOL_FIT = 1e-8  # max |p - p_desc| / max p, same for grad p
TOL_EDGE = 1e-12  # max |p(rho = 1)| / max |p|
TOL_BOX = 1e-5  # grad p on the box (float32) vs DESC
TOL_VOLUME = 2e-2  # int p dV on the box (staircase) vs flux coordinates


def rel(a, b):
    return np.abs(a - b).max() / np.abs(b).max()


def check_pressure(path, frac=10, seed=0):
    eq = desc.io.load(path)
    eq = eq[-1] if hasattr(eq, "__len__") else eq
    pres = Pressure.fit(eq)
    b = pres.basis
    print(f"\n{os.path.basename(path)}: p series (L,M,N)=({b.L},{b.M},{b.N}), "
          f"{b.num_modes} modes")

    # 1. fit vs DESC on G1
    g = flux_grid(eq, 2 * eq.L, 2 * eq.M, 2 * eq.N)
    out = pres.evaluate(g)
    d = eq.compute(["p", "grad(p)"], grid=g)
    e_p, e_g = rel(out[:, 0], np.asarray(d["p"])), rel(out[:, 1:], np.asarray(d["grad(p)"]))
    print(f"1. vs DESC on G1: p {e_p:.1e}, grad p {e_g:.1e} (tol {TOL_FIT:.0e})")

    # 2. p = 0 on rho = 1, for the fit and for random coefficients
    gb = LinearGrid(rho=1.0, theta=2 * b.M + 1, zeta=2 * b.N + 1, NFP=eq.NFP, sym=False)
    p_max = np.abs(out[:, 0]).max()
    e_fit = np.abs(pres.evaluate(gb)[:, 0]).max() / p_max
    rand = Pressure(b, np.random.default_rng(seed).normal(size=b.num_modes), eq)
    p_rand = rand.impose_zero_edge()
    e_rand = (np.abs(p_rand.evaluate(gb)[:, 0]).max()
              / np.abs(p_rand.evaluate(g)[:, 0]).max())
    print(f"2. max |p(rho=1)| / max |p|: fit {e_fit:.1e}, random coefficients "
          f"{np.abs(rand.evaluate(gb)[:, 0]).max() / np.abs(rand.evaluate(g)[:, 0]).max():.1e}"
          f" -> {e_rand:.1e} after impose_zero_edge (tol {TOL_EDGE:.0e})")

    # 3. on the box
    h = float(eq.compute("a")["a"]) / frac
    axes = cartesian_grid(eq, h)
    dc = current_on_cartesian(eq, CurrentPotential.fit(eq), axes)
    p, grad_p = pressure_on_cartesian(eq, pres, dc)
    sub = np.random.default_rng(seed).choice(len(dc["inside"]), 20000, replace=False)
    rtz = dc["rtz"][sub]
    gp_desc = np.asarray(eq.compute("grad(p)", grid=Grid(rtz, sort=False))["grad(p)"])
    X = [a[i] for a, i in zip(axes, np.unravel_index(dc["inside"][sub], p.shape))]
    phi = np.arctan2(X[1], X[0])
    c, s = np.cos(phi), np.sin(phi)
    gp_desc = np.column_stack([gp_desc[:, 0] * c - gp_desc[:, 1] * s,
                               gp_desc[:, 0] * s + gp_desc[:, 1] * c, gp_desc[:, 2]])
    e_box = rel(grad_p.reshape(-1, 3)[dc["inside"][sub]], gp_desc)
    gq = QuadratureGrid(L=2 * eq.L, M=2 * eq.M, N=2 * eq.N, NFP=eq.NFP)
    dq = eq.compute(["p", "sqrt(g)"], grid=gq)
    int_flux = np.sum(np.asarray(dq["p"]) * np.abs(np.asarray(dq["sqrt(g)"])) * gq.weights)
    int_box = p.sum(dtype=np.float64) * h**3
    e_vol = abs(int_box - int_flux) / int_flux
    assert not p[~dc["mask"]].any() and not grad_p[~dc["mask"]].any()
    print(f"3. box {p.shape}, h = a/{frac}: grad p vs DESC at 20000 inside nodes {e_box:.1e} "
          f"(tol {TOL_BOX:.0e}); int p dV {int_box:.4e} vs {int_flux:.4e} Pa m^3, "
          f"rel. diff {e_vol:.1e} (tol {TOL_VOLUME:.0e})")

    assert e_p < TOL_FIT and e_g < TOL_FIT
    assert e_fit < TOL_EDGE and e_rand < TOL_EDGE
    assert e_box < TOL_BOX and e_vol < TOL_VOLUME


try:
    @pytest.mark.parametrize("path", EQ_FILES, ids=[os.path.basename(f) for f in EQ_FILES])
    def test_pressure(path):
        check_pressure(path)
except NameError:
    pass


if __name__ == "__main__":
    for path in EQ_FILES:
        check_pressure(path)
