"""J = curl T from the fitted potential T vs DESC's J.

For the free-boundary precise_QA and W7-X:
  fit: T_theta, T_zeta (gauge T_rho = 0) fitted to DESC's J; J = curl T on
  the flux grid G1 (cheb1 pattern, rho up to 1) vs DESC's J at the same nodes

Also printed: sqrt(g) J^rho on rho = 1 (J.n, not imposed) and the net
toroidal current 2pi <T_theta(rho = 1)>.

run: pytest tests/test_current_potential.py -s     or     python tests/test_current_potential.py
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
from desc.grid import LinearGrid  # noqa: E402
from desc.transform import Transform  # noqa: E402

from current_potential import CurrentPotential  # noqa: E402
from desc_grid import flux_grid  # noqa: E402

EQ_FILES = [
    os.path.join(ROOT, "desc_input", "precise_QA_free_boundary_smooth_J",
                 "precise_QA_free_boundary_smooth_J_edgeJ_r0.3.h5"),
    os.path.join(ROOT, "desc_input", "W7-X_output.h5"),
]
TOL = 1e-3  # max |curl T - J_desc| / max |J_desc|


def check_curl_T(path):
    eq = desc.io.load(path)
    eq = eq[-1] if hasattr(eq, "__len__") else eq
    pot = CurrentPotential.fit(eq)
    b = pot.basis
    grid = flux_grid(eq, b.L, b.M, b.N)
    J = pot.evaluate(grid)
    J_desc = np.asarray(eq.compute("J", grid=grid)["J"])
    err_max = np.abs(J - J_desc).max() / np.abs(J_desc).max()
    err_rms = np.sqrt(np.mean((J - J_desc) ** 2) / np.mean(J_desc**2))

    # J.n: sqrt(g) J^rho = d_theta T_zeta - d_zeta T_theta on rho = 1
    gb = LinearGrid(rho=1.0, theta=4 * b.M + 3, zeta=4 * b.N + 3, NFP=eq.NFP, sym=False)
    tr = Transform(gb, b, derivs=1, build=True)
    jn = np.abs(np.asarray(tr.transform(pot.c[:, 1], dt=1))
                - np.asarray(tr.transform(pot.c[:, 0], dz=1))).max()
    I = 2 * np.pi * np.asarray(tr.transform(pot.c[:, 0])).mean()
    scale = np.abs(np.asarray(tr.transform(pot.c[:, 0], dr=1))).max()  # max sqrt(g) J^zeta
    r = np.asarray(grid.nodes[:, 0])

    print(f"\n{os.path.basename(path)}: T series (L,M,N)=({b.L},{b.M},{b.N}), "
          f"{b.num_modes} modes x 2; G1 rho in [{r.min():.4f}, {r.max():.4f}]")
    print(f"1. curl T vs DESC J on G1: max err / max|J| = {err_max:.2e}, "
          f"rms {err_rms:.2e} (tol {TOL:.0e})")
    print(f"   J.n (not imposed): max |sqrt(g) J^rho| on rho = 1 / max|sqrt(g) J^zeta| = "
          f"{jn / scale:.1e}; net toroidal current {I:.3f} A")
    assert err_max < TOL


try:
    @pytest.mark.parametrize("path", EQ_FILES, ids=[os.path.basename(f) for f in EQ_FILES])
    def test_curl_T(path):
        check_curl_T(path)
except NameError:
    pass


if __name__ == "__main__":
    for path in EQ_FILES:
        check_curl_T(path)
