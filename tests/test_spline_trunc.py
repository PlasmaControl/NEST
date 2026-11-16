"""Gates of the m-dependent truncation of the spline bases (spline_pressure.spline_keep: drop m != 0 splines whose
support ends where rho^|m| < m_tol, and every |m| > m_max) on the perturbed DSHAPE state, M/N 24/8.

  T1  unknowns: spline a + b and spline p, full against truncated
  T2  the fit of DESC's potential: rms |J - J_DESC| / rms J_DESC, full against truncated (should agree to ~1e-3
      relative), and the truncated J_fit against the full one
  T3  spline p: fit of DESC's p(rho) unchanged (m = 0 keeps every spline)
  T4  save / load round trip of both truncated objects (same modes, same coefficients, m_tol / m_max kept)

    python tests/test_spline_trunc.py [--eq PATH] [--m-tol 1e-3] [--m-max 12]
"""

import argparse
import os
import sys
import tempfile
import time

import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, "nest"))
from desc import set_device  # noqa: E402

set_device(os.environ.get("NEST_DEVICE", "cpu"))
import jax  # noqa: E402
import jax.numpy as jnp  # noqa: E402

jax.config.update("jax_enable_x64", True)

from boundary_potential import BoundaryPotential  # noqa: E402
from desc_input import DESCInput  # noqa: E402
from spline_pressure import SplinePressure  # noqa: E402

TOK = os.path.join(ROOT, "desc_input", "tokamak_free_boundary")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--eq", default=os.path.join(TOK, "perturbed_L40N5", "eps1000_final.h5"))
    ap.add_argument("--MN", type=int, nargs=2, default=(24, 8))
    ap.add_argument("--m-tol", type=float, default=1e-3)
    ap.add_argument("--m-max", type=int, default=12)
    a = ap.parse_args()
    t0 = time.time()
    inp = DESCInput.load(a.eq)
    eq = inp.eq
    M, N = a.MN
    ok = True
    full = BoundaryPotential.fit(eq, M=M, N=N, radial="spline")
    tr = BoundaryPotential.fit(eq, M=M, N=N, radial="spline", m_tol=a.m_tol, m_max=a.m_max)
    pf = SplinePressure.fit(eq, M=M, N=N)
    pt = SplinePressure.fit(eq, M=M, N=N, m_tol=a.m_tol, m_max=a.m_max)
    # T1
    ab_f, ab_t = 2 * full.basis.num_modes, 2 * tr.basis.num_modes
    print(f"T1 a + b: full {ab_f}, truncated {ab_t} ({ab_t / ab_f:.2f}); n_u {full.n_free} -> {tr.n_free} "
          f"(chi {full.S.n_chi} -> {tr.S.n_chi}); p: {pf.n} -> {pt.n} ({pt.n / pf.n:.2f}); fit {time.time() - t0:.0f} s",
          flush=True)
    ok &= full.S.n_chi == tr.S.n_chi and ab_t < ab_f and pt.n < pf.n
    # T2
    g_r = np.linspace(0.02, 0.98, 25)
    th = 2 * np.pi * np.arange(49) / 49
    ze = 2 * np.pi * np.arange(17) / 17
    R_, T_, Z_ = np.meshgrid(g_r, th, ze, indexing="ij")
    rtz = np.column_stack([R_.ravel(), T_.ravel(), Z_.ravel()])
    Jd = np.asarray(inp.compute_points(["J"], rtz)["J"])
    nd = np.sqrt(np.mean(np.sum(Jd**2, 1)))
    J = {k: np.asarray(jax.jit(p.point_evaluator())(*(jnp.asarray(rtz[:, i]) for i in range(3))))
         for k, p in (("full", full), ("trunc", tr))}
    e = {k: float(np.sqrt(np.mean(np.sum((v - Jd) ** 2, 1))) / nd) for k, v in J.items()}
    dft = float(np.sqrt(np.mean(np.sum((J["full"] - J["trunc"]) ** 2, 1))) / nd)
    print(f"T2 rms |J - J_DESC| / rms J_DESC: full {e['full']:.3e}, truncated {e['trunc']:.3e}; "
          f"rms |J_full - J_trunc| / rms J_DESC {dft:.1e}")
    ok &= e["trunc"] < 1.05 * e["full"] + 1e-4
    # T3
    rho = np.linspace(0, 0.99, 50)
    nodes = np.column_stack([rho, np.zeros_like(rho), np.zeros_like(rho)])
    from desc.grid import Grid
    g = Grid(nodes)
    e3 = float(np.abs(pt.evaluate(g)[:, 0] - pf.evaluate(g)[:, 0]).max() / np.abs(pf.evaluate(g)[:, 0]).max())
    print(f"T3 spline p fit, truncated vs full (max rel): {e3:.1e}")
    ok &= e3 < 1e-10
    # T4
    with tempfile.TemporaryDirectory() as d:
        tr.save(os.path.join(d, "t.npz"))
        pt.save(os.path.join(d, "p.npz"))
        tr2 = BoundaryPotential.load(os.path.join(d, "t.npz"), eq)
        pt2 = SplinePressure.load(os.path.join(d, "p.npz"), eq)
    e4 = (np.array_equal(tr2.basis.modes, tr.basis.modes) and np.array_equal(tr2.u, tr.u)
          and np.array_equal(pt2.j_idx, pt.j_idx) and np.array_equal(pt2.c, pt.c)
          and tr2.basis.m_max == a.m_max and pt2.m_tol == a.m_tol)
    print(f"T4 save/load: {e4}")
    ok &= e4
    print(f"ALL PASSED ({time.time() - t0:.0f} s)" if ok else "FAILED")
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
