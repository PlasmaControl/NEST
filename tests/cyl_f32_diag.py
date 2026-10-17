"""Diagnostic: is the cylindrical Jacobian consistent with its residual in float32?

D2 (production optimize_qa --grid cylindrical, float32) took line-search steps ~0.6 with
rising f, while main's box at the same settings took ~1.0 and decreased. For the small
C-series case (a/6, factors 1) and three Fields (cylindrical float32 / float64, VG box
float32): the tangent vs jax.jvp, the optimizer's J^T w vs vjp, and the Taylor remainder
q(t) = F(x + t v) - F(x) - t J v, which for the bilinear force is exactly t^2 q(1):
q(t) / t^2 must stay constant as t -> 0.

run: python tests/cyl_f32_diag.py
"""
import os
import sys
import time

import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, "nest"))
sys.path.insert(0, os.path.join(ROOT, "tests"))
import jax  # noqa: E402

jax.config.update("jax_enable_x64", True)
import jax.numpy as jnp  # noqa: E402

import test_cylindrical_integration as T  # noqa: E402


def check(name, f):
    from objective import ForceBalance, ObjectiveFunction
    t0 = time.time()
    rng = np.random.default_rng(2)
    x0 = jnp.asarray(f.x0)
    v = jnp.asarray(rng.standard_normal(f.n)) * 1e-2 * jnp.linalg.norm(x0) / np.sqrt(f.n)
    comp = jax.jit(lambda fl, x: fl._compute(x)["F"])
    base = f._compute(x0)
    F0 = comp(f, x0)
    tan = jax.jit(lambda fl, v, J0, B0: fl._tangent(v, J0, B0)["F"])(f, v, base["J"], base["B"])
    _, jv = jax.jvp(lambda x: f._compute(x)["F"], (x0,), (v,))
    e_t = float(jnp.abs(tan - jv).max() / jnp.abs(jv).max())
    obj = ObjectiveFunction([ForceBalance()], f).build()
    wf = jnp.asarray(rng.standard_normal(obj.dim_f))
    jtw = obj.vjp_scaled_error(wf, x0)
    _, vjp_o = jax.vjp(lambda xx: obj.jvp_scaled_error(xx, x0), v)
    e_a = float(jnp.abs(jtw - vjp_o(wf)[0]).max() / jnp.abs(jtw).max())
    Jv_o = obj.jvp_scaled_error(v, x0)
    lhs, rhs = float(jnp.vdot(Jv_o, wf)), float(jnp.vdot(v, jtw))
    e_i = abs(lhs - rhs) / abs(lhs)
    print(f"{name}: tangent vs jvp {e_t:.1e}; linear_transpose J^T w vs vjp {e_a:.1e}; "
          f"<Jv, w> vs <v, J^T w> {e_i:.1e}", flush=True)
    for t in (1.0, 0.1, 0.01, 1e-3):
        q = np.asarray(comp(f, x0 + t * v) - F0 - t * tan, np.float64)
        print(f"   t {t:7.0e}: |q(t)|/t^2 {np.linalg.norm(q) / t**2:.4e}   |t Jv| {t * float(jnp.linalg.norm(tan)):.3e}", flush=True)
    print(f"   [{time.time() - t0:.0f} s]", flush=True)
    return dict(tangent=e_t, transpose=e_a, adjoint=e_i)


if __name__ == "__main__":
    os.environ["NEST_CACHE"] = os.environ.get("NEST_CACHE", "on")
    out = {}
    out["cyl_f32"] = check("cylindrical float32", T._fields(dtype=jnp.float32))
    out["cyl_f64"] = check("cylindrical float64", T._fields(dtype=jnp.float64))
    out["box_f32"] = check("VG box a/6 float32", T._fields(kernel="truncated", dtype=jnp.float32))
