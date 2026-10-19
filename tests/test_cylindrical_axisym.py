"""Gates D3 for the axisymmetric cylindrical path (N_phi = 1) on the free-boundary tokamak
(``desc_input/tokamak_free_boundary``, L = M = 24, N = 0, NFP = 1).

With one plane the annulus solve is two 2-D problems (|m| = 0 for A_z, |m| = 1 for
A_r + i A_phi). For an axisymmetric J it must agree with the same (r, z) axes on N_phi > 1
planes, whose other modes then vanish.

  D3a  ``CylindricalGrid.around`` gives N_phi = 1 for N = 0 (and N_phi > 1 with axisym=False).
  D3b  B_plasma at the collocation points: N_phi = 1 against N_phi = 5, to rounding.
  D3c  ``Fields`` (J sampling, solve, interpolation): F(x0) and a Jacobian-vector product on
       N_phi = 1 against N_phi = 5, to rounding.
  D3d  ``DenseFields`` through the N_phi = 1 solver: G v against the solve.
  D3e  B_plasma at the collocation points converges with h: a/10, a/20 against a/40.

run: PYTHONPATH=nest:$DESC python -m pytest -q tests/test_cylindrical_axisym.py -s
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

D = os.path.join(ROOT, "desc_input", "tokamak_free_boundary")
EQ, FIELD = os.path.join(D, "tokamak_L24M24.h5"), os.path.join(D, "tokamak_L24M24_field.h5")
FRAC = 10
_C = {}


def setup():
    if _C:
        return _C
    import fit_cache
    from boundary_potential import BoundaryPotential
    from desc_input import DESCInput
    from pressure import Pressure
    inp = DESCInput.load(EQ, field=FIELD)
    cur = fit_cache.fit(BoundaryPotential, inp, factors=(1, 1, 1))
    pre = fit_cache.fit(Pressure, inp, factors=(1, 1, 1), edge_order=2)
    _C.update(inp=inp, cur=cur, pre=pre)
    return _C


def _grid(frac, N_phi=None):
    from cylindrical_grid import CylindricalGrid
    inp = setup()["inp"]
    g = CylindricalGrid.around(inp, inp.a / frac)
    if N_phi is not None and N_phi != g.N_phi:
        g = CylindricalGrid(g.r_axis, g.z_axis, g.NFP, N_phi)
    return g.locate_plasma(inp, sym=True)


def _B_plasma(fr, grid):
    from force import ForceResidual
    f = ForceResidual.__new__(ForceResidual)
    f.__dict__.update(fr.__dict__)
    f.box = grid
    return f.B_plasma(setup()["cur"])


def _rel(a, b):
    return float(np.sqrt(np.sum((a - b) ** 2) / np.sum(b ** 2)))


def test_D3a_grid():
    from cylindrical_grid import CylindricalGrid
    inp = setup()["inp"]
    g = CylindricalGrid.around(inp, inp.a / FRAC)
    g3 = CylindricalGrid.around(inp, inp.a / FRAC, axisym=False)
    print(f"  D3a around: {g.shape} (axisym), {g3.shape} (axisym=False)")
    assert g.N_phi == 1 and g3.N_phi > 1 and g3.shape[1:] == g.shape[1:]


def test_D3b_modes():
    from force import ForceResidual
    c = setup()
    fr = ForceResidual(c["inp"], None)
    B1, B5 = _B_plasma(fr, _grid(FRAC)), _B_plasma(fr, _grid(FRAC, 5))
    e = _rel(B1, B5)
    print(f"  D3b B_plasma N_phi = 1 vs 5 at a/{FRAC}: {e:.1e} ({len(fr.x)} points)")
    assert e < 1e-12


def test_D3c_fields():
    from force import ForceResidual
    from objective import Fields
    c = setup()
    out = {}
    v = None
    for n in (1, 5):
        g = _grid(FRAC, n)
        f = Fields(ForceResidual(c["inp"], g), c["cur"], c["pre"], dtype=jnp.float64)
        if v is None:
            v = jnp.asarray(np.random.default_rng(3).standard_normal(f.n))
        x0 = jnp.asarray(f.x0)
        o = f.compute(x0)
        _, dF = jax.jvp(lambda x: f.compute(x)["F"], (x0,), (v,))
        out[n] = (np.asarray(o["B_plasma"]), np.asarray(o["F"]), np.asarray(o["grad_p"]), np.asarray(dF))
    # F(x0) is a small residual (~3e-6 of |grad p|): its difference is measured against |grad p|
    eB = _rel(out[1][0], out[5][0])
    eF = float(np.sqrt(np.sum((out[1][1] - out[5][1]) ** 2) / np.sum(out[5][2] ** 2)))
    eJ = _rel(out[1][3], out[5][3])
    print(f"  D3c Fields N_phi = 1 vs 5: B_plasma(x0) {eB:.1e}, F(x0) / |grad p| {eF:.1e}, J v {eJ:.1e}")
    assert eB < 1e-12 and eF < 1e-12 and eJ < 1e-12


def test_D3d_dense():
    from boundary_potential import BoundaryPotential
    from dense import DenseFields
    from force import ForceResidual
    from objective import Fields
    c = setup()
    cur = BoundaryPotential.fit(c["inp"].eq, L=6, M=6, N=0)
    os.environ["NEST_CACHE"] = "off"
    f = Fields(ForceResidual(c["inp"], _grid(FRAC), resolution=(12, 12, 0)), cur, c["pre"], dtype=jnp.float64)
    v = jnp.asarray(np.random.default_rng(5).standard_normal(f.n_u))
    t0 = time.time()
    d = DenseFields.from_fields(f, cur, batch=8, cache=False)
    e = _rel(np.asarray(d.B_plasma(v)), np.asarray(f.B_plasma(v)))
    print(f"  D3d dense G ({f.n_u} columns, {time.time() - t0:.0f} s) vs solve {e:.1e}")
    assert e < 1e-12


def test_D3e_convergence():
    from force import ForceResidual
    c = setup()
    fr = ForceResidual(c["inp"], None)
    B = {k: _B_plasma(fr, _grid(k)) for k in (10, 20, 40)}
    e10, e20 = _rel(B[10], B[40]), _rel(B[20], B[40])
    print(f"  D3e B_plasma vs a/40: a/10 {e10:.1e}, a/20 {e20:.1e} (order {np.log2(e10 / e20):.1f})")
    assert e20 < e10 / 4 and e20 < 1e-3


if __name__ == "__main__":
    names = sys.argv[1:]
    for k, fn in sorted((k, v) for k, v in globals().items() if k.startswith("test_") and callable(v)):
        if names and not any(n in k for n in names):
            continue
        t0 = time.time()
        print(f"{k} ...", flush=True)
        fn()
        print(f"  ({time.time() - t0:.0f} s)", flush=True)
