"""Smooth edge (k = 2): J and p vanish smoothly at the LCFS.

For each equilibrium, with ``edge_order=2``:
  1. the tangential current sqrt(g) (J^theta, J^zeta) = (-d_rho T_zeta,
     d_rho T_theta) is zero on rho = 1 and falls like (1 - rho)^2 (on rho =
     1 - eps for eps, eps/2 well inside the edge layer: ratio ~ 4); also
     printed, max |J| at rho = 0.8 .. 0.99, the width of the layer where J
     drops, and max |J| on rho = 1, where only J^rho (J.n, not imposed) is left
  2. the net toroidal current 2pi <T_theta(rho = 1)> is DESC's I(rho = 1)
  3. random coefficients after ``impose_smooth_edge``: tangential current
     zero on rho = 1
  4. p = (1 - rho^2)^3 Q: p and grad p zero on rho = 1
Also printed: the fit of J vs DESC's J by radius (DESC's J is not zero at the
edge, so the fit must differ there) and of p.

run: pytest tests/test_smooth_edge.py -s     or     python tests/test_smooth_edge.py
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

from desc import set_device  # noqa: E402

# DESC uses the CPU unless set_device('gpu') runs before any other DESC import
set_device(os.environ.get("NEST_DEVICE", "cpu"))

import desc.io  # noqa: E402
import jax.numpy as jnp  # noqa: E402
from desc.grid import LinearGrid  # noqa: E402

from current_potential import CurrentPotential  # noqa: E402
from desc_grid import flux_grid  # noqa: E402
from separable import Separable  # noqa: E402
from pressure import Pressure  # noqa: E402

EQ_FILES = [
    os.path.join(ROOT, "desc_input", "precise_QA_free_boundary_smooth_J",
                 "precise_QA_free_boundary_smooth_J_edgeJ_r0.3.h5"),
    os.path.join(ROOT, "desc_input", "W7-X_output.h5"),
]
K = 2
TOL_EDGE = 1e-10  # max |J(rho = 1)| / max |J|, same for p and grad p
# current through zeta = 0 vs DESC I(1), relative; DESC's I(1) (from B) and the
# integral of its J differ by 2e-6 for W7-X
TOL_CURRENT = 1e-5


def edge_values(f, eq, eps, n_theta=64, n_zeta=32):
    """max |f| over a (theta, zeta) grid on rho = 1 - eps."""
    t, z = np.meshgrid(np.linspace(0, 2 * np.pi, n_theta, endpoint=False),
                       np.linspace(0, 2 * np.pi / eq.NFP, n_zeta, endpoint=False), indexing="ij")
    r = np.full(t.size, 1.0 - eps)
    return np.abs(np.asarray(f(*(jnp.asarray(a) for a in (r, t.ravel(), z.ravel()))))).max()


def tangential(pot):
    """f(rho, theta, zeta) -> sqrt(g) (J^zeta, -J^theta) = d_rho (T_theta, T_zeta)."""
    T = Separable([pot.basis] * 2, [pot.c[:, k] for k in range(2)], pot.basis.NFP)
    return lambda *a: T.evaluate(*a, dr=1)


def check_smooth_edge(path, seed=0):
    eq = desc.io.load(path)
    eq = eq[-1] if hasattr(eq, "__len__") else eq
    pot = CurrentPotential.fit(eq, edge_order=K)
    b = pot.basis
    print(f"\n{os.path.basename(path)}: T series (L,M,N)=({b.L},{b.M},{b.N}), k = {K}")

    # fit vs DESC, by radius
    g = flux_grid(eq, 2 * eq.L, 2 * eq.M, 2 * eq.N)
    J, J_desc = pot.evaluate(g), np.asarray(eq.compute("J", grid=g)["J"])
    rho, J_max = np.asarray(g.nodes[:, 0]), np.abs(J_desc).max()
    print("   fit: max |J - J_desc| / max |J_desc|: " + ", ".join(
        f"rho <= {r}: {np.abs(J - J_desc)[rho <= r].max() / J_max:.1e}"
        for r in (0.5, 0.8, 0.9, 1.0)))

    # 1. tangential current zero on the edge, (1 - rho)^2 near it
    nodes = [jnp.asarray(g.nodes[:, j]) for j in range(3)]
    ft = tangential(pot)
    scale = np.abs(np.asarray(ft(*nodes))).max()
    e0, e1, e2 = (edge_values(ft, eq, eps) / scale for eps in (0.0, 1e-5, 5e-6))
    f = pot.point_evaluator()
    layer = ", ".join(f"{1 - eps}: {edge_values(f, eq, eps) / J_max:.3f}"
                      for eps in (0.2, 0.1, 0.05, 0.02, 0.01, 0.0))
    print(f"1. max sqrt(g) |J^theta, J^zeta| / max at rho = 1: {e0:.1e}; at 1 - 1e-5: {e1:.2e}, "
          f"1 - 5e-6: {e2:.2e} (ratio {e1 / e2:.2f}, (1 - rho)^2 -> 4)\n"
          f"   max |J| / max |J_desc| at rho = {layer} (at 1: J.n only)")

    # 2. net toroidal current: the mean over (theta, zeta) of the current
    # through the cross-sections, 2pi <T_theta(rho = 1)>
    gb = LinearGrid(rho=1.0, theta=2 * b.M + 1, zeta=2 * b.N + 1, NFP=eq.NFP, sym=False)
    I = 2 * np.pi * pot.T(gb)[:, 0].mean()
    g1 = LinearGrid(rho=1.0, M=eq.M_grid, N=eq.N_grid, NFP=eq.NFP)
    I_desc = float(g1.compress(eq.compute("current", grid=g1)["current"])[0])
    e_I = abs(I - I_desc) / abs(I_desc)
    print(f"2. net toroidal current: {I:.6f} A vs DESC {I_desc:.6f} A, rel. diff {e_I:.1e}")

    # 3. random coefficients
    rnd = CurrentPotential(b, np.random.default_rng(seed).normal(size=pot.c.shape), eq, K)
    rnd = rnd.impose_smooth_edge()
    fr = tangential(rnd)
    r0 = edge_values(fr, eq, 0.0) / np.abs(np.asarray(fr(*nodes))).max()
    print(f"3. random coefficients after impose_smooth_edge: tangential current on rho = 1 "
          f"/ max = {r0:.1e}")

    # 4. pressure
    pres = Pressure.fit(eq, edge_order=K)
    out, dp = pres.evaluate(g), eq.compute(["p", "grad(p)"], grid=g)
    p_max, gp_max = np.abs(np.asarray(dp["p"])).max(), np.abs(np.asarray(dp["grad(p)"])).max()
    fp = pres.point_evaluator()
    p0 = edge_values(lambda *a: fp(*a)[:, 0], eq, 0.0) / p_max
    g0 = edge_values(lambda *a: fp(*a)[:, 1:], eq, 0.0) / gp_max
    print(f"4. p fit: max |p - p_desc| / max p = {np.abs(out[:, 0] - dp['p']).max() / p_max:.1e}, "
          f"grad p {np.abs(out[:, 1:] - dp['grad(p)']).max() / gp_max:.1e}; on rho = 1: "
          f"p {p0:.1e}, grad p {g0:.1e}")

    assert e0 < TOL_EDGE and 3.0 < e1 / e2 < 4.5
    assert e_I < TOL_CURRENT
    assert r0 < TOL_EDGE
    assert p0 < TOL_EDGE and g0 < TOL_EDGE


try:
    @pytest.mark.parametrize("path", EQ_FILES, ids=[os.path.basename(f) for f in EQ_FILES])
    def test_smooth_edge(path):
        check_smooth_edge(path)
except NameError:
    pass


if __name__ == "__main__":
    for path in EQ_FILES:
        check_smooth_edge(path)
