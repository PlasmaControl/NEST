"""J on a uniform Cartesian grid (step 2) vs DESC, for the precise_QA example.

  1. toroidal current: sum of J_y dx dz over the half-plane phi = 0 (y = 0, x > 0)
     vs DESC's J^zeta sqrt(g) integrated over the zeta = 0 cross-section
  2. int |J| dV: sum of |J| dx dy dz over the grid vs DESC's quadrature

Both sums treat the plasma edge as a staircase, an O(dx/a) error, which sets
the tolerances.

run: pytest tests/test_cartesian_grid.py -s     or     python tests/test_cartesian_grid.py
"""

import os
import sys

import numpy as np

try:  # pytest is optional: the file also runs as a plain script
    import pytest

    module_fixture = pytest.fixture(scope="module")
except ImportError:
    module_fixture = lambda f: f  # noqa: E731

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, "nest"))

from desc import set_device  # noqa: E402

# DESC uses the CPU unless set_device('gpu') runs before any other DESC import
set_device(os.environ.get("NEST_DEVICE", "cpu"))

import desc.io  # noqa: E402
from desc.grid import LinearGrid, QuadratureGrid  # noqa: E402

from cartesian_grid import (  # noqa: E402
    cartesian_grid,
    current_on_cartesian,
    plasma_extent,
)
from current_fz import CurrentFZ  # noqa: E402

EQ_FILE = os.path.join(ROOT, "desc_input", "precise_QA_output.h5")


@module_fixture
def eq_current():
    eq = desc.io.load(EQ_FILE)
    eq = eq[-1] if hasattr(eq, "__len__") else eq
    return eq, CurrentFZ.fit(eq, oversample=2)


def test_current_through_phi0(eq_current, h=3e-3, tol=1e-2):
    """Current through the half-plane phi = 0 from J_y on a Cartesian (x, z) grid."""
    eq, current = eq_current
    R_min, R_max, Z_min, Z_max = plasma_extent(eq)
    xs = np.arange(R_min - h, R_max + h, h)
    zs = np.arange(Z_min - h, Z_max + h, h)
    d = current_on_cartesian(eq, current, (xs, np.array([0.0]), zs))
    Jy = d["J"][:, 0, :, 1]
    I, I_abs = Jy.sum() * h * h, np.abs(Jy).sum() * h * h

    # DESC: not I(rho=1), since where J.grad(rho) != 0 at the boundary, as in
    # this low-resolution equilibrium, the current through a cross-section
    # varies with zeta
    I_desc = cross_section_current(eq, zeta=0.0)

    rel = abs(I - I_desc) / I_abs
    print(f"\ncurrent through phi=0 (h = {h * 1e3:g} mm): {I:.3f} A vs DESC "
          f"{I_desc:.3f} A, |diff| / int|J_y| dA = {rel:.1e} (tol {tol:.0e})")
    assert rel < tol


def test_volume_integral_abs_J(eq_current, N=(81, 81, 41), tol=2e-2):
    """int |J| dV from the Cartesian grid G2 vs DESC."""
    eq, current = eq_current
    d = current_on_cartesian(eq, current, cartesian_grid(eq, N))
    integral = np.linalg.norm(d["J"], axis=-1).sum() * np.prod(d["dx"])

    integral_desc = volume_integral(eq, "|J|")

    rel = abs(integral - integral_desc) / integral_desc
    print(f"\nint |J| dV on G2 {N}, dx = {np.round(d['dx'], 4)} m: {integral:.2f} A m "
          f"vs DESC {integral_desc:.2f} A m, rel. diff {rel:.1e} (tol {tol:.0e})")
    assert rel < tol



def cross_section_current(eq, zeta=0.0, L=64, M=256):
    """Current through the zeta = const cross-section, int J^zeta sqrt(g) drho dtheta.

    Gauss-Legendre in rho. Equals I(rho=1) only if J.grad(rho) = 0 on the
    boundary; otherwise it varies with zeta.
    """
    x, w = np.polynomial.legendre.leggauss(L)
    g = LinearGrid(rho=(x + 1) / 2, theta=M, zeta=np.array([zeta]), NFP=1, sym=False)
    d = eq.compute(["J^zeta", "sqrt(g)"], grid=g)
    f = np.asarray(g.meshgrid_reshape(np.asarray(d["J^zeta"] * d["sqrt(g)"]), "rtz"))
    return np.sum(f[:, :, 0].mean(axis=1) * 2 * np.pi * w / 2)


def volume_integral(eq, name="|J|", factor=2):
    """int f dV over the plasma for the DESC quantity ``name`` (default |J|)."""
    g = QuadratureGrid(L=factor * eq.L, M=factor * eq.M, N=factor * eq.N, NFP=eq.NFP)
    d = eq.compute([name, "sqrt(g)"], grid=g)
    return np.sum(
        np.asarray(d[name]) * np.abs(np.asarray(d["sqrt(g)"])) * np.asarray(g.weights)
    )

if __name__ == "__main__":
    eq = desc.io.load(EQ_FILE)
    eq = eq[-1] if hasattr(eq, "__len__") else eq
    ec = (eq, CurrentFZ.fit(eq, oversample=2))
    test_current_through_phi0(ec)
    test_volume_integral_abs_J(ec)
