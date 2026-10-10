"""Stellarator symmetry (sym="cos" bases, ``ForceResidual(sym=True)``).

Smooth-current precise_QA at the ``test_fields`` resolution (bases 4, 4, 3, box
h = a / 4), in the stellarator-symmetric external field B_DESC - BS_h(J_DESC)
(``DESCExternalField``; the coil set itself is symmetric only to ~1e-4 of |B|).
  1. ``grid.symmetric_half``: the half theta in [0, pi] of a QuadratureGrid with
     doubled weights off theta = 0, pi integrates symmetric scalars exactly; a node
     set without its mirror is rejected.
  2. The symmetric fits (``BoundaryPotential.fit(sym=True)``, ``Pressure.fit``) give
     the same J and p as the full fits (whose odd modes are zero to round-off).
  3. ``Fields`` with symmetric bases has about half the points and half the columns;
     the field and force terms of a symmetric state on the half grid equal those of
     the same state (embedded in the full bases) on the whole grid, and Gauss-Newton
     ends at the same state.

run: pytest tests/test_symmetry.py -s     or     python tests/test_symmetry.py
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
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from desc import set_device  # noqa: E402

# DESC uses the CPU unless set_device('gpu') runs before any other DESC import
set_device(os.environ.get("NEST_DEVICE", "cpu"))

import jax.numpy as jnp  # noqa: E402
from desc.grid import QuadratureGrid  # noqa: E402

from boundary_potential import BoundaryPotential  # noqa: E402
from cartesian_grid import evaluate_chunked  # noqa: E402
from desc_external_field import DESCExternalField  # noqa: E402
from desc_input import DESCInput  # noqa: E402
from fields import Fields  # noqa: E402
from grid import symmetric_half  # noqa: E402
from objectives import FieldDistance, ForceBalance, ObjectiveFunction  # noqa: E402
from optimize import gauss_newton  # noqa: E402
from pressure import Pressure  # noqa: E402
from test_fields import RES, setup  # noqa: E402


def rel(a, b):
    return float(np.abs(np.asarray(a) - b).max() / np.abs(b).max())


def check_half_grid():
    g = QuadratureGrid(L=8, M=8, N=6, NFP=2)
    nodes, w = np.asarray(g.nodes), np.asarray(g.weights)
    half, mult = symmetric_half(nodes, 2)
    f = lambda n: (1 + n[:, 0] ** 2 * np.cos(n[:, 1] - 2 * n[:, 2])) ** 2  # symmetric
    e = abs(np.sum(w[half] * mult * f(nodes[half])) / np.sum(w * f(nodes)) - 1)
    print(f"half grid: {len(half)} of {len(nodes)} nodes; symmetric integral rel. error {e:.1e}")
    assert e < 1e-13 and len(half) < 0.6 * len(nodes)
    try:
        symmetric_half(nodes[nodes[:, 1] < 3], 2)
        raise AssertionError("an unsymmetric node set was accepted")
    except ValueError:
        pass


def _embed(xS, FS, F):
    """A symmetric state as coefficients of the full bases (odd modes zero)."""
    cS, c = FS.current, F.current
    u = np.zeros(F.n_u)
    chi = {tuple(m): i for i, m in enumerate(c.chi_modes)}
    mode = {tuple(m): i for i, m in enumerate(c.basis.modes)}
    uS = np.asarray(xS[:FS.n_u])
    for i, m in enumerate(cS.chi_modes):
        u[chi[tuple(m)]] = uS[i]
    for i, m in enumerate(cS.basis.modes):
        u[c.S.n_chi + mode[tuple(m)]] = uS[cS.S.n_chi + i]
        u[c.S.n_chi + c.S.K + mode[tuple(m)]] = uS[cS.S.n_chi + cS.S.K + i]
    pS, p = FS.pressure, F.pressure
    coef = np.zeros(p.basis.num_modes)
    pmode = {tuple(m): i for i, m in enumerate(p.basis.modes)}
    for i, v in enumerate(np.asarray(pS.to_coefficients(xS[FS.n_u:]))):
        coef[pmode[tuple(pS.basis.modes[i])]] = v
    return np.concatenate([u, coef[p.free_coefficients()[0]]])


def check_symmetric_solve(frac=4):
    inp, current, pressure, box = setup(frac)
    inp = DESCInput(inp.eq, field=DESCExternalField(inp, box), name="precise_QA (DESC external field)")
    currentS = BoundaryPotential.fit(inp.eq, *RES, sym=True)
    pressureS = Pressure.fit(inp.eq, *RES, edge_order=2, sym=True)
    rtz = np.random.default_rng(1).uniform([0.01, 0, 0], [1, 2 * np.pi, 2 * np.pi], (3000, 3))
    eJ = rel(evaluate_chunked(currentS.point_evaluator(), rtz, 3000),
             evaluate_chunked(current.point_evaluator(), rtz, 3000))
    ep = rel(evaluate_chunked(pressureS.point_evaluator(), rtz, 3000),
             evaluate_chunked(pressure.point_evaluator(), rtz, 3000))
    print(f"symmetric vs full fit: J {eJ:.1e}, p {ep:.1e}")
    assert eJ < 1e-10 and ep < 1e-10

    F = Fields(inp, current, pressure, box, col_chunk=8, verbose=False)
    FS = Fields(inp, currentS, pressureS, box, col_chunk=8, verbose=False)
    print(f"points {len(FS.x)} of {len(F.x)}, columns {FS.n} of {F.n}")
    assert len(FS.x) < 0.6 * len(F.x) and FS.n < 0.6 * F.n
    xS = FS.fit_pressure(FS.initial_state())
    x = _embed(xS, FS, F)
    for term in (FieldDistance, ForceBalance):
        a = float(jnp.linalg.norm(ObjectiveFunction(FS, [term(FS)]).residual(xS)))
        b = float(jnp.linalg.norm(ObjectiveFunction(F, [term(F)]).residual(x)))
        print(f"{term.__name__}: half grid {a:.10e}, whole grid {b:.10e}")
        assert abs(a / b - 1) < 1e-8

    terms = lambda f: [FieldDistance(f), ForceBalance(f, weight=0.1)]
    rS = gauss_newton(ObjectiveFunction(FS, terms(FS)), xS, maxiter=30, verbose=False)
    r = gauss_newton(ObjectiveFunction(F, terms(F)), x, maxiter=30, verbose=False)
    JS = F.state(jnp.asarray(_embed(np.asarray(rS["x"]), FS, F)), F.data())["J"]
    J = F.state(r["x"], F.data())["J"]
    eJ = rel(JS, J)
    print(f"Gauss-Newton: {rS['stop_reason']} after {len(rS['history'])} / {r['stop_reason']} after "
          f"{len(r['history'])} iterations; end-state J symmetric vs full {eJ:.1e}")
    assert eJ < 1e-7


try:
    def test_half_grid():
        check_half_grid()

    def test_symmetric_solve():
        check_symmetric_solve()
except NameError:
    pass


if __name__ == "__main__":
    check_half_grid()
    check_symmetric_solve(float(sys.argv[1]) if len(sys.argv) > 1 else 4)
