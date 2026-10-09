"""Reconstruction of J and B through the box: FFT, Biot-Savart and spectral interpolation.

  1. Manufactured solution: A = psi(x) e_z with psi = (1 - s)^6 a smooth toroidal
     bump, s = ((R - R0)^2 + z^2) / w^2 < 1. B = curl A and J = curl B / mu0 are
     exact (JAX derivatives of psi) and vanish outside the bump, so the
     free-space field of J is B everywhere. J is sampled on a periodic box
     (spacing h = w / frac), B obtained by FFT (``b_plasma_free``) and
     interpolated (``interpolate``, non-uniform FFT) to random points inside
     and around the bump, then compared with the exact B.
  2. J reconstruction: the box J of the fitted ``BoundaryPotential`` (smooth-current
     precise_QA) interpolated to the collocation grid QuadratureGrid(2 L, 2 M, 2 N)
     vs J from the series at the same points, by radius.
  3. Convergence: B_plasma at the collocation points for h = a/10, a/20, a/30;
     differences between successive resolutions, by radius, and the rate.

Errors are max |error| / max |reference| over the points (Cartesian components).

run: pytest tests/test_reconstruction.py -s     or     python tests/test_reconstruction.py [manufactured] [qa]
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
from scipy.fft import next_fast_len  # noqa: E402
from scipy.optimize import brentq  # noqa: E402

from boundary_potential import BoundaryPotential  # noqa: E402
from cartesian_grid import cylindrical_to_cartesian, evaluate_chunked  # noqa: E402
from desc_input import DESCInput  # noqa: E402
from fourier import fft_J  # noqa: E402
from grid import CartesianGrid  # noqa: E402
from magnetic_field import b_plasma_free, interpolate  # noqa: E402

jax.config.update("jax_enable_x64", True)

EQ = os.path.join(ROOT, "desc_input", "precise_QA_free_boundary_smooth_J",
                  "precise_QA_free_boundary_smooth_J_edgeJ_r0.3.h5")
BINS = (0.0, 0.5, 0.8, 0.9, 0.95, 1.0)
TOL_MANUFACTURED = 2e-2  # at h = w / 20

# ------------------------------------------------------------ 1. manufactured
R0, W = 1.0, 0.3  # bump: major radius, half-width [m]


def psi(x):
    s = ((jnp.sqrt(x[0] ** 2 + x[1] ** 2) - R0) ** 2 + x[2] ** 2) / W**2
    return jnp.where(s < 1, (1 - s) ** 6, 0.0)


def B_exact(x):  # curl (psi e_z) = (d_y psi, -d_x psi, 0)
    g = jax.grad(psi)(x)
    return jnp.stack([g[1], -g[0], 0.0])


def J_exact(x):  # curl B / mu0
    D = jax.jacfwd(B_exact)(x)  # D[i, j] = d_j B_i
    return jnp.stack([D[2, 1] - D[1, 2], D[0, 2] - D[2, 0], D[1, 0] - D[0, 1]]) / mu_0


_B = jax.jit(jax.vmap(B_exact))
_J = jax.jit(jax.vmap(J_exact))


def _chunked(f, x, chunk=200000):
    return np.concatenate([np.asarray(f(jnp.asarray(x[i : i + chunk])))
                           for i in range(0, len(x), chunk)])


def check_manufactured(fracs=(5, 10, 20), n_points=2000, seed=0):
    """FFT + interpolation of the bump's J against its exact B."""
    rng = np.random.default_rng(seed)
    # points in the bump (s < 1) and around it (1 < s < 2.25), in its bounding box
    pts = []
    while sum(len(p) for p in pts) < n_points:
        x = rng.uniform([-R0 - W, -R0 - W, -W], [R0 + W, R0 + W, W], (20000, 3))
        s = ((np.hypot(x[:, 0], x[:, 1]) - R0) ** 2 + x[:, 2] ** 2) / W**2
        pts.append(x[s < 2.25])
    x = np.concatenate(pts)[:n_points]
    s = ((np.hypot(x[:, 0], x[:, 1]) - R0) ** 2 + x[:, 2] ** 2) / W**2
    B_ref = _chunked(_B, x)
    print(f"\n1. manufactured bump (R0 = {R0} m, w = {W} m), {np.sum(s < 1)} points inside, "
          f"{np.sum(s >= 1)} around; max |B| = {np.abs(B_ref).max():.3e}")
    errs = []
    for frac in fracs:
        h = W / frac
        extent = np.array([2 * (R0 + W), 2 * (R0 + W), 2 * W])
        n = [next_fast_len(int(np.ceil(2 * e / h))) for e in extent]
        axes = tuple(h * (np.arange(k) - k // 2) for k in n)
        X = np.stack(np.meshgrid(*axes, indexing="ij"), axis=-1).reshape(-1, 3)
        R = np.hypot(X[:, 0], X[:, 1])
        on = np.flatnonzero(((R - R0) ** 2 + X[:, 2] ** 2) < W**2)
        J = np.zeros((X.shape[0], 3))
        J[on] = _chunked(_J, X[on])
        t0 = time.time()
        B_box = b_plasma_free(fft_J(J.reshape(tuple(n) + (3,))), axes)
        B = np.asarray(interpolate(B_box, axes, x))
        e_in = np.abs(B - B_ref)[s < 1].max() / np.abs(B_ref).max()
        e_out = np.abs(B - B_ref)[s >= 1].max() / np.abs(B_ref).max()
        errs.append(max(e_in, e_out))
        print(f"   h = w/{frac}: box {tuple(n)}, max |B - B_exact| / max |B|: inside {e_in:.2e}, "
              f"around {e_out:.2e} ({time.time() - t0:.1f} s)")
    rates = [np.log(errs[i] / errs[i + 1]) / np.log(fracs[i + 1] / fracs[i])
             for i in range(len(fracs) - 1)]
    print("   convergence rate: " + ", ".join(f"{r:.2f}" for r in rates))
    assert errs[-1] < TOL_MANUFACTURED


# ------------------------------------------------- 2-3. J and B for the QA
def check_qa(fracs=(10, 20, 30), edge_power=2):
    inp = DESCInput.load(EQ)
    eq = inp.eq
    pot = BoundaryPotential.fit(eq, edge_power=edge_power)
    g = QuadratureGrid(L=2 * eq.L, M=2 * eq.M, N=2 * eq.N, NFP=eq.NFP)
    d = inp.compute(["R", "phi", "Z"], g)
    xc = np.column_stack([d["R"] * np.cos(d["phi"]), d["R"] * np.sin(d["phi"]), d["Z"]])
    rho = np.asarray(g.nodes[:, 0])
    J_ser = cylindrical_to_cartesian(pot.evaluate(g), d["phi"])
    J_max = np.abs(J_ser).max()
    t, z = np.meshgrid(np.linspace(0, 2 * np.pi, 96, endpoint=False),
                       np.linspace(0, 2 * np.pi / eq.NFP, 24, endpoint=False), indexing="ij")
    layer = ", ".join(
        f"{r:g}: {np.abs(pot.evaluate(np.column_stack([np.full(t.size, r), t.ravel(), z.ravel()]))).max() / J_max:.3f}"
        for r in (0.9, 0.95, 0.98, 0.99, 0.995, 0.999, 1.0))
    print(f"\nfitted J (edge_power s = {edge_power}), max |J| / max |J| on rho = {layer}")
    print(f"\n2. J reconstruction: box J interpolated to {g.num_nodes:,} collocation points "
          f"vs the series; max |dJ| / max |J| by rho")
    B = {}
    for frac in fracs:
        t0 = time.time()
        box = CartesianGrid.periodic_box(inp, inp.a / frac).locate_plasma(inp)
        J_box = box.scatter(cylindrical_to_cartesian(
            evaluate_chunked(pot.point_evaluator(eq), box.rtz), box.phi_inside), np.float32)
        J_int = np.asarray(interpolate(J_box, box.axes, xc))
        dJ = np.abs(J_int - J_ser).max(axis=1) / J_max
        bins = "  ".join(f"{a:g}-{b:g}: {dJ[(rho > a) & (rho <= b)].max():.1e}"
                         for a, b in zip(BINS[:-1], BINS[1:]))
        B_box = np.asarray(b_plasma_free(fft_J(J_box), box.axes))  # to the host
        del J_box
        B[frac] = interpolate(B_box, box.axes, xc)
        del B_box
        print(f"   h = a/{frac}: box {box.shape}, {len(box.inside):,} points inside: {bins} "
              f"({time.time() - t0:.0f} s)")

    print("\n3. convergence of B_plasma at the collocation points: max |B_h - B_h'| / max |B| "
          "by rho")
    B_max = np.abs(B[fracs[-1]]).max()
    diffs = []
    for f1, f2 in zip(fracs[:-1], fracs[1:]):
        dB = np.abs(B[f2] - B[f1]).max(axis=1) / B_max
        diffs.append(dB.max())
        bins = "  ".join(f"{a:g}-{b:g}: {dB[(rho > a) & (rho <= b)].max():.1e}"
                         for a, b in zip(BINS[:-1], BINS[1:]))
        print(f"   a/{f1} -> a/{f2}: all {dB.max():.1e}; {bins}")
    if len(diffs) == 2:
        # successive differences ~ C (h1^p - h2^p): solve for p, then C h^p at the finest h
        h = 1 / np.asarray(fracs, dtype=float)
        ratio = lambda p: (h[0] ** p - h[1] ** p) / (h[1] ** p - h[2] ** p) - diffs[0] / diffs[1]
        p = brentq(ratio, 0.05, 8)
        rest = diffs[1] * h[2] ** p / (h[1] ** p - h[2] ** p)
        print(f"   estimated order p = {p:.2f}; remaining error at a/{fracs[-1]} ~ {rest:.1e} "
              f"of max |B|")


try:
    def test_manufactured():
        check_manufactured()

    def test_qa():
        check_qa()
except NameError:
    pass


if __name__ == "__main__":
    # parts in separate processes on a GPU; "qa 30" runs the QA part at a/30 only,
    # "qa s=3" with edge_power 3. Large boxes: XLA_PYTHON_CLIENT_ALLOCATOR=platform
    parts = sys.argv[1:] or ["manufactured", "qa"]
    if "manufactured" in parts:
        check_manufactured()
    if "qa" in parts:  # "qa s=3": edge_power 3
        fracs = tuple(int(p) for p in parts if p.isdigit()) or (10, 20, 30)
        s = [int(p[2:]) for p in parts if p.startswith("s=")]
        check_qa(fracs, s[0] if s else 2)
