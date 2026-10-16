"""Gates A1-B3 of docs/cylindrical_grid_plan.md: the cylindrical-annulus J -> B solver.

run: python tests/test_cylindrical_bs.py [name ...]   (or pytest tests/test_cylindrical_bs.py)

Each ``test_*`` prints its measured numbers and asserts the gate's pass criterion.
"""

import os
import sys
import time

import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, "nest"))

from chebyshev import ChebAxis  # noqa: E402


# ------------------------------------------------------------------ A1
def test_A1_polynomials_exact():
    """D, D2 and the quadrature are exact on polynomials of degree <= n (unmapped)."""
    worst = 0.0
    for n in (8, 16, 32, 64, 128):
        ax = ChebAxis(0.6, 1.4, n)
        t = (ax.s - 1.0) / 0.4
        c = np.random.default_rng(n).standard_normal(n + 1)
        P = np.polynomial.Polynomial(c)
        f = P(t)
        scale = np.abs(f).max()
        e1 = np.abs(ax.D @ f - P.deriv(1)(t) / 0.4).max() / (np.abs(P.deriv(1)(t)).max() / 0.4)
        e2 = np.abs(ax.D2 @ f - P.deriv(2)(t) / 0.16).max() / (np.abs(P.deriv(2)(t)).max() / 0.16)
        I = P.integ()
        eI = abs(ax.w_int @ f - 0.4 * (I(1.0) - I(-1.0))) / (0.8 * scale)
        worst = max(worst, e1 / n**2, e2 / n**4, eI)
        print(f"  A1 poly n={n:4d}: D {e1:.1e}  D2 {e2:.1e}  quad {eI:.1e}")
    assert worst < 1e-12, worst


def test_A1_smooth_functions():
    """Spectral convergence on sin(5x), e^x, plain and mapped; off-node interpolation."""
    rows = []
    for alpha in (0.0, 0.5, 0.9, 0.99):
        for n in (16, 32, 64):
            ax = ChebAxis(-1.0, 1.0, n, alpha)
            s = ax.s
            f = np.sin(5 * s) * np.exp(s)
            df = (5 * np.cos(5 * s) + np.sin(5 * s)) * np.exp(s)
            d2f = (10 * np.cos(5 * s) - 24 * np.sin(5 * s)) * np.exp(s)
            st = np.linspace(-1, 1, 1001)
            ft = np.sin(5 * st) * np.exp(st)
            e1 = np.abs(ax.D @ f - df).max() / np.abs(df).max()
            e2 = np.abs(ax.D2 @ f - d2f).max() / np.abs(d2f).max()
            ei = np.abs(ax.interp_matrix(st) @ f - ft).max() / np.abs(ft).max()
            rows.append((alpha, n, e1, e2, ei))
            print(f"  A1 smooth alpha={alpha:4.2f} n={n:3d}: D {e1:.1e}  D2 {e2:.1e}  interp {ei:.1e}")
    for alpha, n, e1, e2, ei in rows:
        if n == 64 and alpha <= 0.9:
            assert max(e1, ei) < 1e-12 and e2 < 1e-10, (alpha, e1, e2, ei)
    # interpolation reproduces the nodes
    ax = ChebAxis(0.3, 2.0, 20, 0.9)
    assert np.allclose(ax.interp_matrix(ax.s), np.eye(21), atol=1e-14)


def test_A1_spacing_rule():
    """for_spacing gives the smallest n with max spacing <= h over the plasma interval."""
    ax = ChebAxis.for_spacing(0.6, 1.4, 0.65, 1.35, 0.0173)
    assert ax.max_spacing_in(0.65, 1.35) <= 0.0173
    prev = ChebAxis(0.6, 1.4, ax.n - 1)
    assert prev.max_spacing_in(0.65, 1.35) > 0.0173
    print(f"  A1 spacing: n = {ax.n}, unmapped; "
          f"alpha 0.9 n = {ChebAxis.for_spacing(0.6, 1.4, 0.65, 1.35, 0.0173, 0.9).n}")


if __name__ == "__main__":
    names = sys.argv[1:]
    tests = [(k, v) for k, v in sorted(globals().items()) if k.startswith("test_") and callable(v)]
    for k, f in tests:
        if names and not any(n in k for n in names):
            continue
        t0 = time.time()
        print(f"{k} ...", flush=True)
        f()
        print(f"{k} passed ({time.time() - t0:.1f} s)", flush=True)
