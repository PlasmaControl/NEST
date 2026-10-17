"""Gates C1-C3 of docs/cylindrical_grid_plan.md: the cylindrical solver inside NEST.

Small configuration (smooth-current precise_QA, potential and pressure at factors
(1, 1, 1), annulus at h = a / 6, ForceResidual at its default resolution with
stellarator symmetry), so that it runs locally in a few minutes.

  C1  CylindricalGrid: inside-node volume against DESC's; toroidal current through
      the phi = 0 plane against I
  C2  Fields(kernel="cylindrical"): B_plasma equals the standalone path (J on the
      grid -> solver -> NumPy evaluation at the points); F(x0) against the VG box
  C3  tangent (J v against jax.jvp of the fields) and the adjoint identity

run: python tests/test_cylindrical_integration.py [name ...]
"""

import os
import sys
import time

import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, "nest"))

import jax  # noqa: E402

jax.config.update("jax_enable_x64", True)
import jax.numpy as jnp  # noqa: E402

EQ = os.path.join(ROOT, "desc_input", "precise_QA_free_boundary_smooth_J",
                  "precise_QA_free_boundary_smooth_J_edgeJ_r0.3.h5")
COILS = os.path.join(ROOT, "desc_input", "precise_QA_free_boundary", "precise_QA_helical_coils.h5")
FRAC = 6
_CACHE = {}


def setup():
    if _CACHE:
        return _CACHE
    import fit_cache
    from boundary_potential import BoundaryPotential
    from cylindrical_grid import CylindricalGrid
    from desc_input import DESCInput
    from pressure import Pressure
    inp = DESCInput.load(EQ, field=COILS)
    cur = fit_cache.fit(BoundaryPotential, inp, factors=(1, 1, 1))
    pre = fit_cache.fit(Pressure, inp, factors=(1, 1, 1), edge_order=2)
    t0 = time.time()
    g = CylindricalGrid.around(inp, inp.a / FRAC).locate_plasma(inp)
    print(f"  setup: {g}, {len(g.inside):,} inside ({time.time() - t0:.0f} s); potential "
          f"{cur.n_free} free", flush=True)
    _CACHE.update(inp=inp, cur=cur, pre=pre, grid=g)
    return _CACHE


def test_C1_grid():
    """Inside-node volume (quadrature weights) and the section current."""
    from cartesian_grid import evaluate_chunked
    c = setup()
    inp, cur, g = c["inp"], c["cur"], c["grid"]
    wr, wz = g.r_axis.w_int, g.z_axis.w_int
    k, i, l = np.unravel_index(g.inside, g.shape)
    dphi = 2 * np.pi / (g.NFP * g.N_phi)
    V = g.NFP * np.sum(g.r_axis.s[i] * wr[i] * wz[l]) * dphi
    eV = abs(V - inp.volume) / inp.volume
    # toroidal current through phi = 0: int J_phi dr dz over the plane's inside nodes
    J = evaluate_chunked(cur.point_evaluator(), g.rtz, 20000)
    on = k == 0
    I_plane = np.sum(J[on, 1] * wr[i[on]] * wz[l[on]])
    eI = abs(I_plane - cur.I) / abs(cur.I)
    print(f"  C1 volume {V:.5f} vs DESC {inp.volume:.5f} m^3 (rel {eV:.1e}); section current "
          f"{I_plane:.2f} vs I {cur.I:.2f} A (rel {eI:.1e})")
    assert eV < 2e-2 and eI < 2e-2  # staircase / kinked-integrand quadrature at h = a/6


def _fields(kernel=None):
    from force import ForceResidual
    from grid import CartesianGrid
    from objective import Fields
    c = setup()
    inp = c["inp"]
    if kernel is None:
        box = c["grid"]
    else:
        box = CartesianGrid.periodic_box(inp, inp.a / FRAC, odd=True).locate_plasma(inp)
    key = ("fr", kernel)
    if key not in c:
        c[key] = ForceResidual(inp, box)
    return Fields(c[key], c["cur"], c["pre"], kernel=kernel or "cylindrical",
                  dtype=jnp.float64 if kernel else jnp.float32)


def test_C2_fields():
    """Fields' cylindrical B_plasma against the standalone path; F(x0) against the box."""
    from cylindrical_grid import evaluate
    c = setup()
    t0 = time.time()
    f = _fields()
    print(f"  C2 Fields (cylindrical) built in {time.time() - t0:.0f} s; collocation "
          f"{f.force.tgrid.shape}, {f.to_points.n_planes} planes", flush=True)
    u = jnp.asarray(f.x0[: f.n_u])
    B = np.asarray(jax.jit(lambda fl, u: fl.B_plasma(u))(f, u))
    J = f.J_box(u)
    Bp, Bz = jax.jit(lambda s, x: s.field_modes(x))(f.solver, J)
    x = np.asarray(f.force.x)
    phi = np.asarray(f.force.phi)
    rpz = np.column_stack([np.hypot(x[:, 0], x[:, 1]), phi, x[:, 2]])
    from cartesian_grid import cylindrical_to_cartesian
    Bref = cylindrical_to_cartesian(evaluate(c["grid"], Bp, Bz, rpz), phi)
    e = np.abs(B - Bref).max() / np.abs(Bref).max()
    out = f.compute(f.x0)
    fb = _fields("truncated")
    ob = fb.compute(fb.x0)
    w = f.force.dV / f.force.dV.sum()
    norm = f.force.norm
    Fc = np.sum(w * np.linalg.norm(np.asarray(out["F"]), axis=1)) / norm
    Fb = np.sum(w * np.linalg.norm(np.asarray(ob["F"]), axis=1)) / norm
    dB = np.sqrt(np.sum(w * np.sum((np.asarray(out["B_plasma"]) - np.asarray(ob["B_plasma"])) ** 2, 1)))
    print(f"  C2 B_plasma: Fields vs standalone {e:.1e}; <|F|>/norm at x0: cylindrical {Fc:.4e}, "
          f"VG box a/{FRAC} {Fb:.4e}; rms |B_cyl - B_box| / max|B| "
          f"{dB / np.abs(np.asarray(ob['B_plasma'])).max():.1e}")
    assert e < 1e-12
    assert abs(Fc - Fb) / Fb < 0.5  # same physics; they differ by the a/6 discretizations


def test_C3_tangent_adjoint():
    """J v of the fields against jax.jvp of _compute; <J v, w> = <v, J^T w>."""
    f = _fields()
    rng = np.random.default_rng(2)
    x0 = jnp.asarray(f.x0)
    v = jnp.asarray(rng.standard_normal(f.n))
    _, jv = jax.jvp(lambda x: f._compute(x)["F"], (x0,), (v,))
    base = f._compute(x0)
    tan = jax.jit(lambda fl, v, J0, B0: fl._tangent(v, J0, B0)["F"])(f, v, base["J"], base["B"])
    e = float(jnp.abs(tan - jv).max() / jnp.abs(jv).max())
    w = jnp.asarray(rng.standard_normal(jv.shape))
    lin = lambda vv: f._tangent(vv, base["J"], base["B"])["F"]  # noqa: E731
    _, vjp = jax.vjp(lin, v)
    lhs, rhs = float(jnp.sum(lin(v) * w)), float(jnp.sum(v * vjp(w)[0]))
    ea = abs(lhs - rhs) / abs(lhs)
    # the optimizer's J^T w is jax.linear_transpose of the tangent (objective._lin_T),
    # a different mechanism from jax.vjp: it needs the transposable scan form
    from objective import ForceBalance, ObjectiveFunction
    obj = ObjectiveFunction([ForceBalance()], f).build()
    wf = jnp.asarray(rng.standard_normal(obj.dim_f))
    jtw = obj.vjp_scaled_error(wf, x0)
    _, vjp_o = jax.vjp(lambda xx: obj.jvp_scaled_error(xx, x0), v)
    elt = float(jnp.abs(jtw - vjp_o(wf)[0]).max() / jnp.abs(jtw).max())
    print(f"  C3 tangent vs jax.jvp {e:.1e}; adjoint <Jv, w> vs <v, J^T w> {ea:.1e}; "
          f"ObjectiveFunction J^T w (linear_transpose) vs vjp {elt:.1e}")
    assert e < 1e-12 and ea < 1e-12 and elt < 1e-12


if __name__ == "__main__":
    names = sys.argv[1:]
    for k, fn in sorted((k, v) for k, v in globals().items() if k.startswith("test_") and callable(v)):
        if names and not any(n in k for n in names):
            continue
        t0 = time.time()
        print(f"{k} ...", flush=True)
        fn()
        print(f"{k} passed ({time.time() - t0:.1f} s)", flush=True)
