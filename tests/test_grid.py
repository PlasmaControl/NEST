"""DESCInput, FluxGrid and CartesianGrid on the finite-beta test equilibria.

  1. DESCInput: loading (Equilibrium or EquilibriaFamily), the scalars and the
     boundary extent against DESC, the input checks, and B_ext against the coil
     set's own field (when there is one)
  2. FluxGrid: torus copies of the one-period nodes, their positions against DESC
  3. CartesianGrid: the periodic box (spacing, FFT-friendly sizes, period, a
     point on y = 0), the plasma it contains (volume against DESC, inverse map
     residual), scatter / gather, and that the function interface
     (``cartesian_grid``, ``flux_grid``) gives the same grids as the classes

run: pytest tests/test_grid.py -s     or     python tests/test_grid.py
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
from desc.grid import Grid, LinearGrid  # noqa: E402
from scipy.fft import next_fast_len  # noqa: E402

from cartesian_grid import cartesian_grid  # noqa: E402
from desc_grid import flux_grid  # noqa: E402
from desc_input import DESCInput, equilibrium  # noqa: E402
from fourier import wavevectors  # noqa: E402
from grid import CartesianGrid, FluxGrid  # noqa: E402

COILS = os.path.join(ROOT, "desc_input", "precise_QA_free_boundary",
                     "precise_QA_helical_coils.h5")
CASES = {  # equilibrium file: external field file (or None)
    os.path.join(ROOT, "desc_input", "precise_QA_free_boundary_smooth_J",
                 "precise_QA_free_boundary_smooth_J_edgeJ_r0.3.h5"): COILS,
    os.path.join(ROOT, "desc_input", "W7-X_output.h5"): None,
}


def load(path):
    inp = DESCInput.load(path, field=CASES[path])
    print(f"\n{inp}")
    return inp


try:  # pytest is optional: the file also runs as a plain script
    import pytest

    inp_fixture = pytest.fixture(scope="module", params=list(CASES), name="inp_fixture",
                                 ids=lambda f: os.path.basename(f))(
        lambda request: load(request.param))
except ImportError:
    pass


def test_desc_input(inp_fixture):
    inp = inp_fixture
    eq = inp.eq
    assert equilibrium(inp) is eq and equilibrium(eq) is eq
    assert (inp.NFP, inp.resolution) == (eq.NFP, (eq.L, eq.M, eq.N))
    assert inp.prescribed_profile == ("iota" if eq.iota is not None else "current")

    for name, value in [("a", inp.a), ("R0", inp.R0), ("V", inp.volume)]:
        ref = float(np.ravel(eq.compute(name)[name])[0])
        assert abs(value - ref) <= 5e-3 * abs(ref), (name, value, ref)

    # extent: bounds of the boundary on a much finer sampling
    g = LinearGrid(rho=1.0, theta=16 * eq.M + 1, zeta=16 * eq.N + 1, NFP=eq.NFP, sym=False)
    d = eq.compute(["R", "Z"], grid=g)
    fine = np.array([np.min(d["R"]), np.max(d["R"]), np.min(d["Z"]), np.max(d["Z"])])
    err = np.max(np.abs(np.array(inp.extent()) - fine)) / inp.a
    print(f"extent {np.round(inp.extent(), 4)}: max diff vs 16x sampling {err:.1e} a")
    assert err < 1e-2

    # the geometry series reproduces DESC's R, Z
    nodes = np.asarray(g.nodes)
    RZ = np.asarray(inp.geometry.evaluate(*(nodes[:, i] for i in range(3))))
    assert np.allclose(RZ, np.column_stack([d["R"], d["Z"]]), atol=1e-12)
    assert inp.check() is inp


def test_external_field(inp_fixture):
    inp = inp_fixture
    if inp.field is None:
        try:
            inp.B_ext(np.zeros((1, 3)))
        except ValueError:
            return
        raise AssertionError("B_ext without a field should raise")
    g = LinearGrid(rho=1.0, M=6, N=6, NFP=inp.NFP, sym=False)
    pos = FluxGrid(g).positions(inp)
    B = inp.B_ext(pos["x"], source_grid=LinearGrid(N=100))
    rpz = np.column_stack([pos["R"], pos["phi"], pos["Z"]])
    B_rpz = np.asarray(inp.field.compute_magnetic_field(rpz, basis="rpz",
                                                        source_grid=LinearGrid(N=100)))
    c, s = np.cos(pos["phi"]), np.sin(pos["phi"])
    ref = np.column_stack([B_rpz[:, 0] * c - B_rpz[:, 1] * s,
                           B_rpz[:, 0] * s + B_rpz[:, 1] * c, B_rpz[:, 2]])
    err = np.max(np.abs(B - ref)) / np.max(np.abs(ref))
    print(f"B_ext at {len(B)} boundary points vs the coils in (R, phi, Z): {err:.1e}")
    assert err < 1e-10


def test_flux_grid(inp_fixture):
    inp = inp_fixture
    L, M, N = inp.resolution
    fg = FluxGrid.concentric(inp, L, M, N)
    assert fg.NFP == inp.NFP
    nodes = fg.torus_nodes()
    assert nodes.shape == (inp.NFP * fg.num_nodes, 3)
    assert np.isclose(fg.torus_weights().sum(), np.asarray(fg.desc.weights).sum())
    # positions of the copies = DESC at the shifted nodes
    pos = fg.positions(inp)
    d = inp.compute(["R", "Z", "phi"], Grid(nodes, sort=False))
    assert np.allclose(pos["R"], d["R"]) and np.allclose(pos["Z"], d["Z"])
    assert np.allclose(np.mod(pos["phi"], 2 * np.pi), np.mod(d["phi"], 2 * np.pi))
    # function interface: the same DESC grid
    assert np.array_equal(np.asarray(flux_grid(inp, L, M, N).nodes), np.asarray(fg.desc.nodes))


def test_cartesian_grid(inp_fixture, frac=10, factor=2.0, tol=3e-2):
    inp = inp_fixture
    h = inp.a / frac
    box = CartesianGrid.periodic_box(inp, h, factor)
    _, R_max, Z_min, Z_max = inp.extent()
    n = np.array(box.shape)
    assert box.periodic and np.allclose(box.dx, h) and np.isclose(box.dV, h**3)
    assert all(next_fast_len(int(k)) == k for k in n)
    assert np.all(box.period >= factor * np.array([2 * R_max, 2 * R_max, Z_max - Z_min]))
    assert np.min(np.abs(box.ys)) < 1e-12  # a plane y = 0
    for k, k_ref in zip(box.wavevectors(), wavevectors(box.axes)):
        assert np.array_equal(k, k_ref)
    for a, b in zip(cartesian_grid(inp, h, factor), box.axes):  # function interface
        assert np.array_equal(a, b)

    box.locate_plasma(inp)
    rel = abs(box.plasma_volume - inp.volume) / inp.volume
    print(f"box {tuple(n.tolist())}, h = a/{frac}: {len(box.inside):,} points inside, "
          f"volume {box.plasma_volume:.5f} vs DESC {inp.volume:.5f} m^3, rel. diff "
          f"{rel:.1e} (tol {tol:.0e})")
    assert rel < tol

    # the flux coordinates of the inside points map back to the points
    x = box.points(box.inside)
    rtz = box.rtz
    RZ = np.asarray(inp.geometry.evaluate(rtz[:, 0], rtz[:, 1], rtz[:, 2]))
    assert np.all(rtz[:, 0] <= 1)
    assert np.max(np.abs(RZ - np.column_stack([np.hypot(x[:, 0], x[:, 1]), x[:, 2]]))) < 1e-8

    # scatter / gather
    v = np.random.default_rng(0).normal(size=(len(box.inside), 3))
    f = box.scatter(v, dtype=np.float64)
    assert f.shape == box.shape + (3,) and not f[~box.mask].any()
    assert np.array_equal(box.gather(f), v) and box.mask.sum() == len(box.inside)


if __name__ == "__main__":
    for path in CASES:
        inp = load(path)
        test_desc_input(inp)
        test_external_field(inp)
        test_flux_grid(inp)
        test_cartesian_grid(inp)
