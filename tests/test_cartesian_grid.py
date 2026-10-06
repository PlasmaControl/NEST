"""J on a uniform Cartesian grid (steps 2-3) vs DESC, for finite-beta equilibria.

  1. toroidal current: sum of J_y dx dz over the half-plane phi = 0 (y = 0, x > 0)
     vs DESC's J^zeta sqrt(g) integrated over the zeta = 0 cross-section
  2. int |J| dV: sum of |J| dx dy dz over the periodic box G3 vs DESC's
     quadrature, plus checks of the box (spacing, size, zero padding)

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
from current_fz import CurrentFZ  # noqa: E402

EQ_FILES = [
    os.path.join(ROOT, "desc_input", "precise_QA_free_boundary_L16N14.h5"),
    os.path.join(ROOT, "desc_input", "W7-X_output.h5"),
]


def load(path):
    eq = desc.io.load(path)
    eq = eq[-1] if hasattr(eq, "__len__") else eq
    print(f"\n{os.path.basename(path)}: NFP={eq.NFP}, (L,M,N)=({eq.L},{eq.M},{eq.N})")
    return eq, CurrentFZ.fit(eq, oversample=2)


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

    # DESC: not I(rho=1), since where J.grad(rho) != 0 at the boundary, as in
    # this low-resolution equilibrium, the current through a cross-section
    # varies with zeta
    I_desc = cross_section_current(eq, zeta=0.0)

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
    integral_desc = volume_integral(eq, "|J|")

    # the box: spacing h, period N h >= factor x the plasma extent, J = 0 off the plasma
    _, R_max, Z_min, Z_max = plasma_extent(eq)
    n = np.array([len(ax) for ax in axes])
    assert np.allclose(d["dx"], h)
    assert d["J"].dtype == np.float32
    assert np.all(n * h >= factor * np.array([2 * R_max, 2 * R_max, Z_max - Z_min]))
    assert not d["J"][~d["mask"]].any()

    rel = abs(integral - integral_desc) / integral_desc
    print(f"int |J| dV on G3 {tuple(n.tolist())}, h = a/{frac} = {h * 1e3:.2f} mm: "
          f"{integral:.2f} A m vs DESC {integral_desc:.2f} A m, rel. diff {rel:.1e} "
          f"(tol {tol:.0e})")
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
    for path in EQ_FILES:
        ec = load(path)
        test_current_through_phi0(ec)
        test_volume_integral_abs_J(ec)
