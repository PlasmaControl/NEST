"""The objective (``objective.ObjectiveFunction``) and its matrix-free Jacobian.

Smooth-current free-boundary precise_QA in its coils, box h = a / frac, x = (u,
c_p) from the fitted ``BoundaryPotential`` and the smooth-edge ``Pressure``,
objectives ForceBalance + FieldError + CurrentError + PressureError:
  1. stellarator symmetry: the symmetric setup (half collocation grid, half flux
     grid, half box, cos/sin bases) against the full one at x0
  2. J v against central differences (exact up to round-off: f is quadratic in
     x) along u, c_p and both; the adjoint identity <w, J v> = <J^T w, v>
  3. the truncated (Vico-Greengard) kernel against Hockney-Eastwood at x0
  4. times of f, J v and J^T w after compilation
Box fields in float64 for tight checks.

run: python tests/test_objective.py [frac]
"""

import os
import sys
import time

import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, "nest"))

from desc import set_device  # noqa: E402

# DESC uses the CPU unless set_device('gpu') runs before any other DESC import
set_device(os.environ.get("NEST_DEVICE", "cpu"))

import jax  # noqa: E402
import jax.numpy as jnp  # noqa: E402

import fit_cache  # noqa: E402
from boundary_potential import BoundaryPotential  # noqa: E402
from desc_input import DESCInput  # noqa: E402
from force import ForceResidual  # noqa: E402
from grid import CartesianGrid  # noqa: E402
from objective import (CurrentError, FieldError, Fields, ForceBalance,  # noqa: E402
                       ObjectiveFunction, PressureError)
from pressure import Pressure  # noqa: E402

jax.config.update("jax_enable_x64", True)

EQ = os.path.join(ROOT, "desc_input", "precise_QA_free_boundary_smooth_J",
                  "precise_QA_free_boundary_smooth_J_edgeJ_r0.3.h5")
COILS = os.path.join(ROOT, "desc_input", "precise_QA_free_boundary",
                     "precise_QA_helical_coils.h5")
TOL_FD, TOL_ADJ, TOL_SYM = 1e-8, 1e-10, 5e-2


def timed(f, *a):
    jax.block_until_ready(f(*a))
    t0 = time.time()
    out = jax.block_until_ready(f(*a))
    return out, time.time() - t0


def build(inp, frac, sym, kernel="hockney", fit=None):
    fit = fit or {}
    box = CartesianGrid.periodic_box(inp, inp.a / frac, odd=sym).locate_plasma(inp)
    cur = fit_cache.fit(BoundaryPotential, inp, sym=sym, **fit)
    pre = fit_cache.fit(Pressure, inp, edge_order=2, sym=sym, **fit)
    fields = Fields(ForceResidual(inp, box, sym=sym), cur, pre, kernel=kernel, dtype=jnp.float64)
    objs = [ForceBalance(), FieldError(0.3), CurrentError(0.1), PressureError(0.1)]
    return ObjectiveFunction(objs, fields).build()


def check_objective(frac=10, fit=None, seed=0):
    inp = DESCInput.load(EQ, field=COILS)
    vals = {}
    for sym in (False, True):
        obj = build(inp, frac, sym, fit=fit)
        print(f"\nsym {sym}: dim_x {obj.dim_x:,}, dim_f {obj.dim_f:,}, box {obj.fields.shape}, "
              f"half box {obj.fields.half is not None}")
        vals[sym] = obj.print_value(obj.x0)
    e_sym = max(abs(vals[True][k] / vals[False][k] - 1) for k in ("force", "B", "J"))
    print(f"1. symmetric vs full at x0: max rel. diff of the terms {e_sym:.1e}")

    x0, rng = obj.x0, np.random.default_rng(seed)
    n_u = obj.fields.n_u
    scale = np.concatenate([np.full(n_u, np.sqrt(np.mean(np.asarray(x0[:n_u]) ** 2))),
                            np.full(obj.dim_x - n_u, np.sqrt(np.mean(np.asarray(x0[n_u:]) ** 2)))])
    e_fd = {}
    for name, mask in [("u", np.arange(obj.dim_x) < n_u), ("c_p", np.arange(obj.dim_x) >= n_u),
                       ("both", np.ones(obj.dim_x, bool))]:
        v = jnp.asarray(1e-3 * scale * rng.normal(size=obj.dim_x) * mask)
        Jv = obj.jvp_scaled_error(v, x0)
        fd = (obj.compute_scaled_error(x0 + v) - obj.compute_scaled_error(x0 - v)) / 2
        e_fd[name] = float(jnp.abs(Jv - fd).max() / jnp.abs(fd).max())
    w = jnp.asarray(rng.normal(size=obj.dim_f))
    lhs, rhs = float(jnp.vdot(w, Jv)), float(jnp.vdot(obj.vjp_scaled_error(w, x0), v))
    e_adj = abs(lhs - rhs) / abs(lhs)
    print("2. J v vs central differences: " + ", ".join(f"{k} {e:.1e}" for k, e in e_fd.items())
          + f"; adjoint {e_adj:.1e}")

    tr = build(inp, frac, True, kernel="truncated", fit=fit)
    vt = tr.print_value(tr.x0)
    print("3. truncated vs Hockney-Eastwood at x0: "
          + ", ".join(f"{k} {vt[k]:.3e} vs {vals[True][k]:.3e}" for k in ("<|F|>/norm", "force", "B")))

    _, t_f = timed(obj.compute_scaled_error, x0)
    _, t_jvp = timed(obj.jvp_scaled_error, v, x0)
    _, t_vjp = timed(obj.vjp_scaled_error, w, x0)
    print(f"4. time (symmetric): f {t_f:.2f} s, J v {t_jvp:.2f} s, J^T w {t_vjp:.2f} s")
    assert max(e_fd.values()) < TOL_FD and e_adj < TOL_ADJ and e_sym < TOL_SYM


if __name__ == "__main__":
    check_objective(int(sys.argv[1]) if len(sys.argv) > 1 else 10)
