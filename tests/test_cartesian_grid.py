"""J = curl T on a uniform Cartesian grid (steps 2-3), for finite-beta equilibria.

J comes from the current potential T (``CurrentPotential``, with J.n = 0 on
the boundary), so the current through every cross-section is the net current.

  1. toroidal current: sum of J_y dx dz over the half-plane phi = 0 (y = 0, x > 0)
     vs DESC's net toroidal current I(rho = 1)
  2. int |J| dV: sum of |J| dx dy dz over the periodic box G3 vs the same J
     integrated in flux coordinates, plus checks of the box (spacing, size,
     zero padding)

Both sums treat the plasma edge as a staircase, an O(dx/a) error, which sets
the tolerances.

run: pytest tests/test_cartesian_grid.py -s     or     python tests/test_cartesian_grid.py
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
from desc.grid import LinearGrid, QuadratureGrid  # noqa: E402

from cartesian_grid import (  # noqa: E402
    cartesian_grid,
    current_on_cartesian,
    plasma_extent,
)
from current_potential import CurrentPotential  # noqa: E402

EQ_FILES = [
    os.path.join(ROOT, "desc_input", "precise_QA_free_boundary_L16N14.h5"),
    os.path.join(ROOT, "desc_input", "W7-X_output.h5"),
]


def load(path):
    eq = desc.io.load(path)
    eq = eq[-1] if hasattr(eq, "__len__") else eq
    print(f"\n{os.path.basename(path)}: NFP={eq.NFP}, (L,M,N)=({eq.L},{eq.M},{eq.N})")
    return eq, CurrentPotential.fit(eq)


try:
    eq_current = pytest.fixture(scope="module", params=EQ_FILES,
                                ids=lambda f: os.path.basename(f))(
        lambda request: load(request.param))
except NameError:
    pass


def test_current_through_phi0(eq_current, frac=60, tol=1e-2):
    """Current through the half-plane phi = 0 from J_y on a Cartesian (x, z) grid
    with spacing h = a / frac."""
    eq, current = eq_current
    h = float(eq.compute("a")["a"]) / frac
    R_min, R_max, Z_min, Z_max = plasma_extent(eq)
    xs = np.arange(R_min - h, R_max + h, h)
    zs = np.arange(Z_min - h, Z_max + h, h)
    d = current_on_cartesian(eq, current, (xs, np.array([0.0]), zs))
    Jy = d["J"][:, 0, :, 1]
    I = Jy.sum(dtype=np.float64) * h * h
    I_abs = np.abs(Jy).sum(dtype=np.float64) * h * h

    # div J = 0 and J.n = 0: the current through any cross-section is I(rho=1)
    g1 = LinearGrid(rho=1.0, M=eq.M_grid, N=eq.N_grid, NFP=eq.NFP)
    I_desc = float(g1.compress(eq.compute("current", grid=g1)["current"])[0])

    rel = abs(I - I_desc) / I_abs
    print(f"current through phi=0 (h = a/{frac} = {h * 1e3:.2f} mm): {I:.3f} A vs DESC "
          f"{I_desc:.3f} A, |diff| / int|J_y| dA = {rel:.1e} (tol {tol:.0e})")
    assert rel < tol


def test_volume_integral_abs_J(eq_current, frac=10, factor=2.0, tol=2e-2):
    """int |J| dV on the periodic box G3 (h = a / frac) vs DESC."""
    eq, current = eq_current
    h = float(eq.compute("a")["a"]) / frac
    axes = cartesian_grid(eq, h, factor=factor)
    d = current_on_cartesian(eq, current, axes)
    integral = np.linalg.norm(d["J"], axis=-1).sum(dtype=np.float64) * h**3
    integral_desc = volume_integral_abs_J(eq, current)

    # the box: spacing h, period N h >= factor x the plasma extent, J = 0 off the plasma
    _, R_max, Z_min, Z_max = plasma_extent(eq)
    n = np.array([len(ax) for ax in axes])
    assert np.allclose(d["dx"], h)
    assert d["J"].dtype == np.float32
    assert np.all(n * h >= factor * np.array([2 * R_max, 2 * R_max, Z_max - Z_min]))
    assert not d["J"][~d["mask"]].any()

    rel = abs(integral - integral_desc) / integral_desc
    print(f"int |J| dV on G3 {tuple(n.tolist())}, h = a/{frac} = {h * 1e3:.2f} mm: "
          f"{integral:.2f} A m vs flux coordinates {integral_desc:.2f} A m, rel. diff {rel:.1e} "
          f"(tol {tol:.0e})")
    assert rel < tol

def volume_integral_abs_J(eq, current, factor=4):
    """int |J| dV of the potential's J in flux coordinates (quadrature grid)."""
    g = QuadratureGrid(L=factor * eq.L, M=factor * eq.M, N=factor * eq.N, NFP=eq.NFP)
    sqrt_g = np.abs(np.asarray(eq.compute("sqrt(g)", grid=g)["sqrt(g)"]))
    J = current.evaluate(g)
    return np.sum(np.linalg.norm(J, axis=1) * sqrt_g * np.asarray(g.weights))


if __name__ == "__main__":
    for path in EQ_FILES:
        ec = load(path)
        test_current_through_phi0(ec)
        test_volume_integral_abs_J(ec)
