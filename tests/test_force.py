"""Step 6: B = B_ext + B_plasma and the force residual F = J x B - grad p at iteration 0.

For the smooth-current free-boundary precise_QA in its coils: J from the fitted
``BoundaryPotential``, p from the fitted ``Pressure``, B_ext from the coils, and
B_plasma by FFT of J sampled on the box (h = a / frac), interpolated spectrally
(non-uniform FFT) to the collocation points, DESC's QuadratureGrid(2 L, 2 M, 2 N).
Printed:
  - the force error <|F|>_vol / <|grad |B|^2 / 2 mu0|>_vol over the plasma and by
    radius, against DESC's own on the same points
  - the split F = F_desc + (J - J_desc) x B_desc + J x (B - B_desc)
    - (grad p - grad p_desc), each term in the same normalization
  - the time of the one-off setup (B_ext, kernel) and of one evaluation of F
  - |B_ext + B_plasma - B_desc| / |B| at the points (DESC's free-boundary residual)
Asserted: the split adds up to F, and DESC's force balance on the collocation
points gives DESC's own force error.

run: pytest tests/test_force.py -s     or     python tests/test_force.py [frac ...]
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

from boundary_potential import BoundaryPotential  # noqa: E402
from desc_input import DESCInput  # noqa: E402
from force import ForceResidual  # noqa: E402
from grid import CartesianGrid  # noqa: E402
from pressure import Pressure  # noqa: E402

EQ = os.path.join(ROOT, "desc_input", "precise_QA_free_boundary_smooth_J",
                  "precise_QA_free_boundary_smooth_J_edgeJ_r0.3.h5")
COILS = os.path.join(ROOT, "desc_input", "precise_QA_free_boundary",
                     "precise_QA_helical_coils.h5")
TOL_SPLIT = 1e-10  # max |F - sum of the parts| / max |J| |B|
TOL_DESC = 1e-8  # <|F_desc|> on the points vs DESC's <|F|>_vol, relative


def check_force(frac=10):
    t0 = time.time()
    inp = DESCInput.load(EQ, field=COILS)
    h = inp.a / frac
    grid = CartesianGrid.periodic_box(inp, h).locate_plasma(inp)
    current, pressure = BoundaryPotential.fit(inp.eq), Pressure.fit(inp.eq)
    t1 = time.time()
    fr = ForceResidual(inp, grid)
    t2 = time.time()
    out = fr.evaluate(current, pressure)
    t3 = time.time()
    m = fr.metrics(out["F"])
    g = fr.grid
    print(f"\n{inp}\nbox {grid.shape}, h = a/{frac} = {h * 1e3:.2f} mm, {len(grid.inside):,} "
          f"points inside (J sampled for B_plasma); collocation grid "
          f"{type(g).__name__}(L={g.L}, M={g.M}, N={g.N}), {g.num_nodes:,} points per period")
    print(f"time: fits + box {t1 - t0:.0f} s, setup (positions, B_ext, kernel) "
          f"{t2 - t1:.0f} s, one evaluation of F {t3 - t2:.1f} s")
    print(f"normalization <|grad |B|^2 / 2 mu0|>_vol = {fr.norm:.4e} N/m^3; "
          f"DESC force error {fr.F_desc_vol / fr.norm:.3e}")
    bins = "  ".join(f"{k}: {v:.1e}" for k, v in m["bins"].items())
    print(f"NEST iteration 0, <|F|> / norm: {m['mean']:.2e} (rms {m['rms']:.2e}, max "
          f"{m['max']:.2e})\n   by rho: {bins}")

    parts, residual, desc = fr.decompose(out)
    dB = np.linalg.norm(out["B"] - desc["B"], axis=1) / np.linalg.norm(desc["B"], axis=1)
    print(f"field: |B_ext + B_plasma - B_desc| / |B_desc|: volume mean {fr.average(dB):.2e}, "
          f"rms {np.sqrt(fr.average(dB**2)):.2e}, max {dB.max():.2e}")
    scale = np.max(np.linalg.norm(out["J"], axis=1) * np.linalg.norm(out["B"], axis=1))
    print("split of F, <|part|> / norm (by rho):")
    for name, F in parts.items():
        mp = fr.metrics(F)
        print(f"   {name:40s} {mp['mean']:.2e}   "
              + "  ".join(f"{v:.1e}" for v in mp["bins"].values()))
    e_split = np.abs(residual).max() / scale
    e_desc = abs(fr.metrics(parts["DESC force balance"])["mean"] * fr.norm / fr.F_desc_vol - 1)
    print(f"   sum of the parts vs F: max |difference| / max |J||B| = {e_split:.1e}; DESC "
          f"force balance on the points vs DESC's <|F|>_vol: rel. diff {e_desc:.1e}")
    assert e_split < TOL_SPLIT
    assert e_desc < TOL_DESC


try:
    def test_force():
        check_force()
except NameError:
    pass


if __name__ == "__main__":
    for frac in [int(f) for f in sys.argv[1:]] or [10]:
        check_force(frac)
