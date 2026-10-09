"""Steps 8-9: objectives and the Gauss-Newton trust region.

Smooth-current precise_QA in its coils at a low resolution (bases 4, 4, 3, box
h = a / frac, default 4), objective FieldDistance + lambda^-1 ForceBalance (+ a
PressureDistance at weight 0, printed only).
  1. The Jacobian of the objective against a central difference (the residual is
     quadratic in x, so the difference is exact up to round-off).
  2. Row blocks: J^T J and J^T r from blocks of points equal the dense ones.
  3. lambda continuation (3 values over 4 decades, from field-distance dominated
     down): every solve lowers f and stops at the rounding level; the force term
     falls as lambda decreases.
  4. The QR step with row blocks reaches the same end state as the dense Cholesky
     step (to sqrt(eps) relative, the flat directions at the rounding stop).

run: pytest tests/test_optimize.py -s     or     python tests/test_optimize.py [frac]
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
sys.path.insert(0, os.path.join(ROOT, "tests"))

from desc import set_device  # noqa: E402

# DESC uses the CPU unless set_device('gpu') runs before any other DESC import
set_device(os.environ.get("NEST_DEVICE", "cpu"))

import jax.numpy as jnp  # noqa: E402

import optimize  # noqa: E402
from fields import Fields  # noqa: E402
from objectives import FieldDistance, ForceBalance, ObjectiveFunction, PressureDistance  # noqa: E402
from optimize import continuation, gauss_newton  # noqa: E402
from test_fields import setup  # noqa: E402


def check_optimize(frac=4):
    inp, current, pressure, box = setup(frac)
    F = Fields(inp, current, pressure, box, col_chunk=8)
    x0 = F.fit_pressure(F.initial_state())
    obj = ObjectiveFunction(F, [FieldDistance(F), ForceBalance(F), PressureDistance(F, weight=0)])
    s0 = obj.summary(x0)
    print(f"start: {s0}")

    J = np.asarray(obj.jacobian(x0))
    v = np.random.default_rng(0).standard_normal(F.n) * 1e-3 * np.abs(x0).max()
    fd = (np.asarray(obj.residual(x0 + v)) - np.asarray(obj.residual(x0 - v))) / 2
    e_jac = np.abs(J @ v - fd).max() / np.abs(fd).max()
    _, (A1, b1), _ = optimize._reduce(obj, jnp.asarray(x0), F.n, "cho", None)
    _, (A2, b2), _ = optimize._reduce(obj, jnp.asarray(x0), F.n, "cho", 300)
    e_blk = max(float(jnp.abs(A1 - A2).max() / jnp.abs(A1).max()),
                float(jnp.abs(b1 - b2).max() / jnp.abs(b1).max()))
    print(f"Jacobian vs central difference {e_jac:.1e}; row blocks vs dense {e_blk:.1e}")
    assert e_jac < 1e-10 and e_blk < 1e-12

    lam_ref = s0["force"] ** 2 / 1e-3**2  # force term = a field distance 1e-3 at the start
    res = continuation(obj, x0, [(1, 1 / (lam_ref * 10.0**e), 0) for e in (2, 0, -2)],
                       maxiter=50, verbose=False)
    forces = []
    for e, r in zip((2, 0, -2), res):
        h = r["history"]
        print(f"lambda = lambda_ref 1e{e:+d}: {len(h)} trial steps, {r['stop_reason']}, "
              f"f {h[0]['f']:.3e} -> {h[-1]['f_new'] if h[-1]['accepted'] else h[-1]['f']:.3e}, "
              f"{r['summary']}")
        assert r["stop_reason"] == "reduction_at_rounding"
        forces.append(r["summary"]["force"])
    assert forces[0] > forces[1] > forces[2]

    rq = gauss_newton(obj, x0, maxiter=50, tr_method="qr", points_per_block=300, verbose=False)
    dx = float(jnp.linalg.norm(rq["x"] - res[-1]["x"]) / jnp.linalg.norm(res[-1]["x"]))
    print(f"qr with row blocks vs cho: |dx| / |x| = {dx:.1e}, {rq['stop_reason']}")
    assert dx < 1e-6


try:
    def test_optimize():
        check_optimize()
except NameError:
    pass


if __name__ == "__main__":
    check_optimize(float(sys.argv[1]) if len(sys.argv) > 1 else 4)
