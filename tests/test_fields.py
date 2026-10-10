"""Step 7: ``fields.Fields`` (J, B, p, grad p as affine maps) and ``DESCExternalField``.

Smooth-current precise_QA in its coils, at a low resolution so that it runs on a
laptop: current and pressure bases 4, 4, 3, box h = a / frac (default 4).
  1. J_0 + D_J u, D_p x_p, D_gp x_p against the series' own point evaluators, and
     B_0 + D_B u against B_direct (J(u) evaluated on the box, then one Biot-Savart
     application): equal to round-off, since the maps are exact.
  2. The D_B cache: a second build restores every column, identical.
  3. The force-optimal pressure (``fit_pressure``) lowers the force residual.
  4. ``DESCExternalField`` with the fitted current: B = B_ext + BS_h(J) equals DESC's
     B at the collocation points for that current (to round-off), and its B_DESC at
     scattered points (``DESCInput.compute_points``) agrees with DESC's B on the grid.

run: pytest tests/test_fields.py -s     or     python tests/test_fields.py [frac]
"""

import os
import sys
import tempfile
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

import jax.numpy as jnp  # noqa: E402

from boundary_potential import BoundaryPotential  # noqa: E402
from cartesian_grid import cylindrical_to_cartesian, evaluate_chunked  # noqa: E402
from desc_external_field import DESCExternalField  # noqa: E402
from desc_input import DESCInput  # noqa: E402
from fields import Fields  # noqa: E402
from grid import CartesianGrid  # noqa: E402
from pressure import Pressure  # noqa: E402

EQ = os.path.join(ROOT, "desc_input", "precise_QA_free_boundary_smooth_J",
                  "precise_QA_free_boundary_smooth_J_edgeJ_r0.3.h5")
COILS = os.path.join(ROOT, "desc_input", "precise_QA_free_boundary",
                     "precise_QA_helical_coils.h5")
RES = (4, 4, 3)
TOL = 1e-12


def rel(a, b):
    return float(np.abs(np.asarray(a) - b).max() / np.abs(b).max())


def setup(frac):
    inp = DESCInput.load(EQ, field=COILS)
    current = BoundaryPotential.fit(inp.eq, *RES)
    pressure = Pressure.fit(inp.eq, *RES, edge_order=2)
    box = CartesianGrid.periodic_box(inp, inp.a / frac).locate_plasma(inp)
    return inp, current, pressure, box


def force(F, x):
    d = F.state(jnp.asarray(x), F.data())
    f = np.cross(d["J"], d["B"]) - np.asarray(d["grad_p"])
    return float(np.sqrt(np.sum(F.dV[:, None] * f**2) / np.sum(F.dV[:, None] * d["grad_p"]**2)))


def check_fields(frac=4):
    inp, current, pressure, box = setup(frac)
    with tempfile.TemporaryDirectory() as cache:
        t0 = time.time()
        F = Fields(inp, current, pressure, box, col_chunk=8, cache_dir=cache)
        t_build = time.time() - t0
        t0 = time.time()
        F2 = Fields(inp, current, pressure, box, col_chunk=8, cache_dir=cache)
        print(f"build {t_build:.0f} s, from the cache {time.time() - t0:.0f} s; D_B cache "
              f"identical: {np.array_equal(F.D_B, F2.D_B)}")
        assert np.array_equal(F.D_B, F2.D_B)
    x = F.initial_state()
    d = F.state(jnp.asarray(x), F.data())
    J_ref = cylindrical_to_cartesian(evaluate_chunked(current.point_evaluator(), F.rtz), F.phi)
    pg = evaluate_chunked(pressure.point_evaluator(), F.rtz)
    errs = {"J": rel(d["J"], J_ref), "p": rel(d["p"], pg[:, 0]),
            "grad p": rel(d["grad_p"], cylindrical_to_cartesian(pg[:, 1:], F.phi)),
            "B": rel(d["B"], F.B_direct(np.asarray(x[: F.n_u])))}
    print("affine maps vs direct evaluation: " + ", ".join(f"{k} {v:.1e}" for k, v in errs.items()))
    assert max(errs.values()) < TOL
    f0, f1 = force(F, x), force(F, F.fit_pressure(x))
    print(f"rms |J x B - grad p| / rms |grad p|: DESC's p(rho) {f0:.3e}, force-optimal p {f1:.3e}")
    assert f1 <= f0


def check_external_field(frac=4):
    inp, current, pressure, box = setup(frac)
    field = DESCExternalField(inp, box, current=current)
    inp_v = DESCInput(inp.eq, field=field, name="precise_QA (DESC external field)")
    F = Fields(inp_v, current, pressure, box, build_D_B=False)
    B = F.B_direct(current.u)
    B_desc = inp.compute(["B"], F.grid)["B"]
    B_desc = cylindrical_to_cartesian(B_desc, F.phi)
    B_points = cylindrical_to_cartesian(inp.compute_points(["B"], F.rtz)["B"], F.phi)
    e1, e2 = rel(B, B_desc), rel(B_points, B_desc)
    print(f"B_ext + BS_h(J) vs DESC B: {e1:.1e}; DESC B at scattered points vs on the grid: {e2:.1e}")
    assert e1 < 1e-9 and e2 < 1e-9


try:
    def test_fields():
        check_fields()

    def test_external_field():
        check_external_field()
except NameError:
    pass


if __name__ == "__main__":
    frac = float(sys.argv[1]) if len(sys.argv) > 1 else 4
    check_fields(frac)
    check_external_field(frac)
