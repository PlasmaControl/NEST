"""Gates of the spline current potential (nest/spline_basis.py, BoundaryPotential(radial="spline")) on the
perturbed DSHAPE state.

  C1  spline_radial (Cox-de Boor + derivative matrices, JAX) against SciPy: B_j(rho^2) and d/drho, d2/drho2
  C2  the fit of DESC's potential: rms |J - J_DESC| / rms J_DESC on an independent grid, spline (M/N 16/6)
      against Fourier-Zernike 24/24/6 (same M, N scale), and J on the LCFS (zero by construction)
  C3  tensor evaluator (flux_interp.TensorCurrent, as Fields uses it) against the point evaluator, random u
  C4  save / load round trip (same u, same J)
  C5  unknowns: spline a, b (+ chi) against Fourier-Zernike at the same M, N

    python tests/test_spline_current.py [--eq PATH]
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
from scipy.interpolate import BSpline  # noqa: E402

from boundary_potential import BoundaryPotential  # noqa: E402
from desc_input import DESCInput  # noqa: E402
from flux_interp import TensorCurrent, TensorGeometry, TensorGrid  # noqa: E402
from spline_basis import spline_radial  # noqa: E402

TOK = os.path.join(ROOT, "desc_input", "tokamak_free_boundary")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--eq", default=os.path.join(TOK, "perturbed_L40N5", "eps1000_final.h5"))
    ap.add_argument("--MN", type=int, nargs=2, default=(16, 6))
    a = ap.parse_args()
    t0 = time.time()
    inp = DESCInput.load(a.eq)
    eq = inp.eq
    ok = True
    M, N = a.MN
    sp = BoundaryPotential.fit(eq, M=M, N=N, radial="spline")
    b = sp.basis
    print(f"spline current: {b.summary()}, n_u {sp.n_free}, I {sp.I:.4e} A; fit {time.time() - t0:.0f} s", flush=True)
    # C1
    rho = np.random.default_rng(0).uniform(0, 1, 400)
    errs = []
    for dr in (0, 1, 2):
        ours = np.asarray(spline_radial(b.t, jnp.asarray(rho), dr))
        ref = np.zeros_like(ours)
        for j in range(b.n_s):
            f = BSpline(np.asarray(b.t), np.eye(b.n_s)[j], 3, extrapolate=False)
            s = rho**2
            if dr == 0:
                ref[:, j] = f(s)
            elif dr == 1:
                ref[:, j] = 2 * rho * f(s, 1)
            else:
                ref[:, j] = 2 * f(s, 1) + 4 * rho**2 * f(s, 2)
        ref = np.nan_to_num(ref)
        errs.append(float(np.abs(ours - ref).max() / max(np.abs(ref).max(), 1e-300)))
    print(f"C1 spline_radial vs SciPy (max rel): value {errs[0]:.1e}, d/drho {errs[1]:.1e}, d2/drho2 {errs[2]:.1e}")
    ok &= max(errs) < 1e-10
    # C2
    fz = BoundaryPotential.fit(eq, L=24, M=24, N=N)
    g_r = np.linspace(0.02, 0.98, 25)
    th = 2 * np.pi * np.arange(17) / 17
    ze = 2 * np.pi * np.arange(9) / 9
    R_, T_, Z_ = np.meshgrid(g_r, th, ze, indexing="ij")
    rtz = np.column_stack([R_.ravel(), T_.ravel(), Z_.ravel()])
    Jd = np.asarray(inp.compute_points(["J"], rtz)["J"])
    out = {}
    for name, pot in (("spline", sp), ("FZ 24/24", fz)):
        Jp = np.asarray(jax.jit(pot.point_evaluator())(*(jnp.asarray(rtz[:, i]) for i in range(3))))
        out[name] = float(np.sqrt(np.mean(np.sum((Jp - Jd) ** 2, 1)) / np.mean(np.sum(Jd**2, 1))))
    edge = np.column_stack([np.ones(51), 2 * np.pi * np.arange(51) / 51, np.zeros(51)])
    Je = np.asarray(jax.jit(sp.point_evaluator())(*(jnp.asarray(edge[:, i]) for i in range(3))))
    e_edge = float(np.abs(Je).max() / np.sqrt(np.mean(np.sum(Jd**2, 1))))
    print(f"C2 rms |J - J_DESC| / rms J_DESC: spline {out['spline']:.2e} (n_u {sp.n_free}), FZ 24/24/{N} "
          f"{out['FZ 24/24']:.2e} (n_u {fz.n_free}); max |J| on the LCFS / rms J {e_edge:.1e}")
    ok &= out["spline"] < 0.05 and e_edge < 1e-10
    # C3
    u = np.random.default_rng(1).normal(size=sp.n_free) * np.abs(sp.u).max()
    sp2 = sp.with_u(u)
    g = TensorGrid(np.linspace(0.05, 0.97, 11), 2 * np.pi * np.arange(13) / 13, 2 * np.pi * np.arange(7) / 7, eq.NFP)
    tc = TensorCurrent(sp2.S, sp2._RZ, g, TensorGeometry(sp2._RZ, g))
    Jt = np.asarray(tc(jnp.asarray(u), sp2.I)).reshape(-1, 3)
    nodes = g.nodes()
    Jp = np.asarray(jax.jit(sp2.point_evaluator())(*(jnp.asarray(nodes[:, i]) for i in range(3))))
    e3 = float(np.abs(Jt - Jp).max() / np.abs(Jp).max())
    print(f"C3 TensorCurrent vs point evaluator (random u): {e3:.1e}")
    ok &= e3 < 1e-9
    # C4
    with tempfile.TemporaryDirectory() as d:
        path = os.path.join(d, "sp.npz")
        sp.save(path)
        sp3 = BoundaryPotential.load(path, eq)
    e4 = float(np.abs(sp3.u - sp.u).max())
    print(f"C4 save/load: max |du| {e4:.1e}, modes equal {np.array_equal(sp3.basis.modes, b.modes)}")
    ok &= e4 == 0 and np.array_equal(sp3.basis.modes, b.modes)
    # C5
    print("C5 unknowns of a + b (chi adds ~2MN more in both):")
    from desc.basis import FourierZernikeBasis
    from spline_basis import SplineBasis
    for MN in ((16, 6), (24, 8), (32, 8)):
        sb = SplineBasis.from_knots(eq, *MN)
        fzb = FourierZernikeBasis(MN[0], MN[0], MN[1], NFP=eq.NFP, sym="cos" if eq.sym else False,
                                  spectral_indexing=eq.spectral_indexing)
        print(f"   M/N {MN[0]}/{MN[1]}: spline (n_s {sb.n_s}) {2 * sb.num_modes}; FZ {MN[0]}/{MN[0]}/{MN[1]} "
              f"{2 * fzb.num_modes}")
    print(f"ALL PASSED ({time.time() - t0:.0f} s)" if ok else "FAILED")
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
