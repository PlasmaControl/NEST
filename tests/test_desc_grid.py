"""J on the ConcentricGrid G1: Fourier-Zernike fit vs DESC.

Fits J of a DESC stellarator example with CurrentFZ (oversample=2), evaluates
it on G1 with ``evaluate_on_flux_grid`` (full torus, Cartesian components) and
compares with DESC's J computed on the same grid.

run: pytest tests/test_desc_grid.py -s     or     python tests/test_desc_grid.py
"""

import os
import sys

import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, "nest"))

import desc.io  # noqa: E402

from current_fz import CurrentFZ  # noqa: E402
from desc_grid import evaluate_on_flux_grid, flux_grid  # noqa: E402

EQ_FILE = os.path.join(ROOT, "desc_input", "precise_QA_output.h5")
TOL = 1e-3  # max |J_fit - J_desc| / max |J_desc|


def test_current_on_flux_grid():
    eq = desc.io.load(EQ_FILE)
    eq = eq[-1] if hasattr(eq, "__len__") else eq  # EquilibriaFamily

    current = CurrentFZ.fit(eq, oversample=2)
    b = current.basis
    grid = flux_grid(eq, b.L, b.M, b.N)
    d = evaluate_on_flux_grid(eq, grid, current)

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
    print(f"\n{os.path.basename(EQ_FILE)}: NFP={eq.NFP}, eq (L,M,N)=({eq.L},{eq.M},{eq.N})")
    print(f"J series (L,M,N)=({b.L},{b.M},{b.N}), {b.num_modes} modes, oversample=2")
    print(f"G1: {grid.num_nodes} nodes/period, {len(d['rho'])} on the torus, "
          f"rho in [{d['rho'].min():.4f}, {d['rho'].max():.4f}]")
    print(f"max err / max|J| = {err_max:.2e}   rms err / rms|J| = {err_rms:.2e}"
          f"   (tol {TOL:.0e})")

    assert np.all(np.isfinite(J_desc)), "DESC J not finite on G1"
    assert err_max < TOL


if __name__ == "__main__":
    test_current_on_flux_grid()
