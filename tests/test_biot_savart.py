"""``biot_savart.BiotSavart``: B_plasma as a linear JAX map of J, both kernels.

1. Gaussian current blob J = exp(-r^2 / 2 s^2) z_hat on boxes of even and odd
   sizes, against the exact B (curl of A_z = mu0 Q erf(r / sqrt2 s) / (4 pi r),
   Q = (2 pi s^2)^(3/2)) at random points around it:
     vico_greengard  spectrally accurate: error at round-off level already at
                     s = 2 h;
     hockney         the discrete sum, second order: the error falls about 4x
                     when h halves;
   and hockney against the existing path (b_plasma_free, then interpolate), the
   same discrete operator: equal to round-off.
2. The map is linear and pure JAX: linearity, jvp, the adjoint identity
   <A J, V> = <J, A^T V> (vjp and linear_transpose), vmap over columns against a
   loop, and float32 against float64.
3. On the smooth-current precise_QA box (h = a / frac): hockney equals the
   existing path at the force-residual points; vico_greengard against hockney,
   which measures the second-order error of the discrete sum.

run: pytest tests/test_biot_savart.py -s     or     python tests/test_biot_savart.py [frac]
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

import jax  # noqa: E402
import jax.numpy as jnp  # noqa: E402
from desc.grid import QuadratureGrid  # noqa: E402
from scipy.constants import mu_0  # noqa: E402
from scipy.special import erf  # noqa: E402

from biot_savart import BiotSavart  # noqa: E402
from boundary_potential import BoundaryPotential  # noqa: E402
from cartesian_grid import cylindrical_to_cartesian, evaluate_chunked  # noqa: E402
from desc_input import DESCInput  # noqa: E402
from fourier import fft_J  # noqa: E402
from grid import CartesianGrid  # noqa: E402
from magnetic_field import b_plasma_free, interpolate  # noqa: E402

EQ = os.path.join(ROOT, "desc_input", "precise_QA_free_boundary_smooth_J",
                  "precise_QA_free_boundary_smooth_J_edgeJ_r0.3.h5")
TOL_VG = 1e-8  # vico_greengard vs the exact blob field
TOL_SAME = 1e-12  # hockney vs the existing path (same operator, float64)
TOL_AD = 1e-11  # linearity, jvp, adjoint, vmap


def rel(B, B_ref):
    return float(np.abs(np.asarray(B) - B_ref).max() / np.abs(B_ref).max())


def blob_B(x, s):
    """Exact B of J = exp(-r^2 / 2 s^2) z_hat (blob at the origin) [T per A/m^2]."""
    r = np.linalg.norm(x, axis=1)
    Q = (2 * np.pi * s**2) ** 1.5
    dA = mu_0 / (4 * np.pi) * Q * (np.sqrt(2 / np.pi) / s * np.exp(-r**2 / (2 * s**2)) / r
                                   - erf(r / (np.sqrt(2) * s)) / r**2)
    return np.column_stack([dA / r * x[:, 1], -dA / r * x[:, 0], 0 * r])


def blob_box(shape, h, s, cut=8.5):
    """Box of ``shape`` and spacing h centred on the blob; the 'plasma' is r < cut s."""
    axes = tuple(h * (np.arange(n) - n // 2) for n in shape)
    box = CartesianGrid(axes, periodic=True)
    r = np.linalg.norm(box.points(), axis=1)
    box.inside = np.flatnonzero(r < cut * s)
    box.rtz = np.zeros((len(box.inside), 3))
    J = np.zeros((len(box.inside), 3))
    J[:, 2] = np.exp(-r[box.inside] ** 2 / (2 * s**2))
    return box, J


def check_blob(s=0.1, n_targets=2000, seed=0):
    rng = np.random.default_rng(seed)
    x = rng.uniform(-0.25, 0.25, (n_targets, 3))
    B_exact = blob_B(x, s)
    err = {}
    for shape, h in [((48, 40, 36), 0.05), ((49, 41, 35), 0.05), ((96, 80, 72), 0.025)]:
        box, J = blob_box(shape, h, s)
        row = {}
        for kernel in ("hockney", "vico_greengard"):
            t0 = time.time()
            row[kernel] = np.asarray(BiotSavart(box, x, kernel=kernel).apply(J))
            print(f"box {shape}, h = {h}: {kernel:15s} error vs exact "
                  f"{rel(row[kernel], B_exact):.1e} ({time.time() - t0:.1f} s)")
        B_box = np.asarray(b_plasma_free(fft_J(box.scatter(J, np.float64)), box.axes))
        e_same = rel(row["hockney"], interpolate(B_box, box.axes, x))
        print(f"   hockney vs b_plasma_free + interpolate: {e_same:.1e}")
        assert rel(row["vico_greengard"], B_exact) < TOL_VG
        assert e_same < TOL_SAME
        err[(shape, h)] = rel(row["hockney"], B_exact)
    order = np.log2(err[((48, 40, 36), 0.05)] / err[((96, 80, 72), 0.025)])
    print(f"hockney order in h: {order:.2f}")
    assert 1.7 < order < 2.3


def check_linear(seed=1):
    rng = np.random.default_rng(seed)
    box, _ = blob_box((40, 36, 30), 0.05, 0.05, cut=9)
    x = rng.uniform(-0.3, 0.3, (3000, 3))
    for kernel in ("hockney", "vico_greengard"):
        bs = BiotSavart(box, x, kernel=kernel, chunk=1000)
        C = bs.consts()
        f = jax.jit(bs.apply)
        J1, J2 = (jnp.asarray(v) for v in rng.standard_normal((2, bs.n_inside, 3)))
        B1, B2 = f(J1, C), f(J2, C)
        e_lin = rel(f(J1 + 2 * J2, C), B1 + 2 * B2)
        e_jvp = rel(jax.jvp(lambda J: f(J, C), (J1,), (J2,))[1], B2)
        V = jnp.asarray(rng.standard_normal((len(x), 3)))
        ATV = jax.vjp(lambda J: f(J, C), J1)[1](V)[0]
        e_adj = abs(float(jnp.sum(B1 * V) - jnp.sum(J1 * ATV))) / abs(float(jnp.sum(B1 * V)))
        e_lt = rel(jax.linear_transpose(lambda J: f(J, C), J1)(V)[0], ATV)
        Js = jnp.asarray(rng.standard_normal((4, bs.n_inside, 3)))
        e_vmap = rel(jax.vmap(lambda J: f(J, C))(Js), np.stack([f(J, C) for J in Js]))
        e32 = rel(BiotSavart(box, x, kernel=kernel, dtype=np.float32, chunk=1000).apply(J1), B1)
        print(f"{kernel:15s} linearity {e_lin:.1e}, jvp {e_jvp:.1e}, adjoint {e_adj:.1e}, "
              f"linear_transpose vs vjp {e_lt:.1e}, vmap {e_vmap:.1e}, float32 {e32:.1e}")
        assert max(e_lin, e_jvp, e_adj, e_lt, e_vmap) < TOL_AD
        assert e32 < 1e-4


def check_equilibrium(frac=10):
    inp = DESCInput.load(EQ)
    eq = inp.eq
    current = BoundaryPotential.fit(eq, eq.L, eq.M, eq.N)
    box = CartesianGrid.periodic_box(inp, inp.a / frac).locate_plasma(inp)
    J = cylindrical_to_cartesian(evaluate_chunked(current.point_evaluator(), box.rtz),
                                 box.phi_inside)
    g = QuadratureGrid(L=2 * eq.L, M=2 * eq.M, N=2 * eq.N, NFP=eq.NFP)
    d = inp.compute(["R", "phi", "Z"], g)
    x = np.column_stack([d["R"] * np.cos(d["phi"]), d["R"] * np.sin(d["phi"]), d["Z"]])
    t0 = time.time()
    B_h = np.asarray(BiotSavart(box, x, kernel="hockney").apply(J))
    t_h = time.time() - t0
    t0 = time.time()
    B_vg = np.asarray(BiotSavart(box, x, kernel="vico_greengard").apply(J))
    t_vg = time.time() - t0
    B_box = np.asarray(b_plasma_free(fft_J(box.scatter(J, np.float64)), box.axes))
    e_same = rel(B_h, interpolate(B_box, box.axes, x))
    print(f"\n{inp.name}: box {box.shape}, h = a/{frac}, {len(box.inside):,} inside points, "
          f"{len(x):,} targets; hockney {t_h:.1f} s, vico_greengard {t_vg:.1f} s (with setup)")
    print(f"hockney vs b_plasma_free + interpolate: {e_same:.1e}; "
          f"max |B_hockney - B_vg| / max |B_vg| = {rel(B_h, B_vg):.1e}")
    assert e_same < TOL_SAME


try:
    def test_blob():
        check_blob()

    def test_linear():
        check_linear()

    def test_equilibrium():
        check_equilibrium()
except NameError:
    pass


if __name__ == "__main__":
    check_blob()
    check_linear()
    for frac in [int(f) for f in sys.argv[1:]] or [10]:
        check_equilibrium(frac)
