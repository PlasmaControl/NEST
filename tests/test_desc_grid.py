"""The flux grid G1 and ``evaluate_on_flux_grid`` (desc_grid.py).

Builds G1 for finite-beta DESC equilibria and evaluates a current on it with
``evaluate_on_flux_grid`` (full torus, Cartesian components), the current being
DESC's own J: the result must equal DESC's J on the same grid, tiled over the
field periods and rotated to Cartesian independently here (round-off).

run: pytest tests/test_desc_grid.py -s     or     python tests/test_desc_grid.py
"""

import os
import sys

import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, "nest"))

from desc import set_device  # noqa: E402

# DESC uses the CPU unless set_device('gpu') runs before any other DESC import
set_device(os.environ.get("NEST_DEVICE", "cpu"))

import desc.io  # noqa: E402

from desc_grid import evaluate_on_flux_grid, flux_grid  # noqa: E402

EQ_FILES = [
    os.path.join(ROOT, "desc_input", "precise_QA_free_boundary_smooth_J",
                 "precise_QA_free_boundary_smooth_J_edgeJ_r0.3.h5"),
    os.path.join(ROOT, "desc_input", "W7-X_output.h5"),
]
TOL = 1e-12  # max |J - J_desc| / max |J_desc|: the same J, so round-off

try:  # pytest is optional: the file also runs as a plain script
    import pytest

    over_equilibria = pytest.mark.parametrize(
        "eq_file", EQ_FILES, ids=[os.path.basename(f) for f in EQ_FILES])
except ImportError:
    over_equilibria = lambda f: f  # noqa: E731


@over_equilibria
def test_current_on_flux_grid(eq_file):
    eq = desc.io.load(eq_file)
    eq = eq[-1] if hasattr(eq, "__len__") else eq  # EquilibriaFamily

    class DescJ:  # DESC's J as the current to evaluate (the interface of BoundaryPotential)
        def evaluate(self, g):
            return np.asarray(eq.compute("J", grid=g)["J"])

    grid = flux_grid(eq, 2 * eq.L, 2 * eq.M, 2 * eq.N)
    d = evaluate_on_flux_grid(eq, grid, DescJ())

    # DESC J on the same grid, in (R, phi, Z) components, copied to the full
    # torus and rotated to Cartesian the same way as in evaluate_on_flux_grid
    J_cyl = np.tile(np.asarray(eq.compute("J", grid=grid)["J"]), (eq.NFP, 1))
    c, s = np.cos(d["phi"]), np.sin(d["phi"])
    J_desc = np.stack(
        [J_cyl[:, 0] * c - J_cyl[:, 1] * s,
         J_cyl[:, 0] * s + J_cyl[:, 1] * c,
         J_cyl[:, 2]],
        axis=-1,
    )

    diff = d["J"] - J_desc
    err_max = np.max(np.abs(diff)) / np.max(np.abs(J_desc))
    err_rms = np.sqrt(np.mean(diff**2) / np.mean(J_desc**2))
    print(f"\n{os.path.basename(eq_file)}: NFP={eq.NFP}, eq (L,M,N)=({eq.L},{eq.M},{eq.N})")
    print(f"G1: {grid.num_nodes} nodes/period, {len(d['rho'])} on the torus, "
          f"rho in [{d['rho'].min():.4f}, {d['rho'].max():.4f}]")
    print(f"max err / max|J| = {err_max:.2e}   rms err / rms|J| = {err_rms:.2e}"
          f"   (tol {TOL:.0e})")

    assert np.all(np.isfinite(J_desc)), "DESC J not finite on G1"
    assert err_max < TOL


if __name__ == "__main__":
    for eq_file in EQ_FILES:
        test_current_on_flux_grid(eq_file)
