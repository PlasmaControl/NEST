"""Gates of the spline pressure (nest/spline_pressure.py) on the perturbed DSHAPE state.

  S1  the fit of DESC's p(rho) (m = n = 0 splines x (1 - rho^2)^3)
  S2  tensor evaluator (SciPy B-splines + TensorGeometry.gradient) against the point evaluator
      (Cox-de Boor in JAX + explicit metric), random coefficients: p and grad p
  S3  regularity: p near the axis independent of theta (random coefficients); p = 0 on the edge
  S4  grad p of the tensor evaluator against central differences of the point evaluator in
      (rho, theta, zeta), contracted with the metric (the chain rule checked independently)
  S5  number of unknowns against Fourier-Zernike for a few knot settings

    python tests/test_spline_pressure.py [--eq PATH]
"""

import argparse
import os
import sys

import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, "nest"))
from desc import set_device  # noqa: E402

set_device(os.environ.get("NEST_DEVICE", "cpu"))
import jax  # noqa: E402
import jax.numpy as jnp  # noqa: E402

jax.config.update("jax_enable_x64", True)
from desc.grid import LinearGrid  # noqa: E402

from desc_input import DESCInput  # noqa: E402
from flux_interp import TensorGeometry, TensorGrid  # noqa: E402
from separable import Separable  # noqa: E402
from spline_pressure import SplinePressure, count, count_fz  # noqa: E402

TOK = os.path.join(ROOT, "desc_input", "tokamak_free_boundary")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--eq", default=os.path.join(TOK, "perturbed_L40N5", "eps1000_final.h5"))
    a = ap.parse_args()
    import time
    t0 = time.time()
    inp = DESCInput.load(a.eq)
    eq = inp.eq
    print(f"loaded {inp} in {time.time() - t0:.0f} s", flush=True)
    ok = True
    sp = SplinePressure.fit(eq, M=8, N=4, coarse=0.05, dense=0.01, half=0.08, helicities=("2/1",))
    print(f"spline pressure: {sp.summary()} ({time.time() - t0:.0f} s)", flush=True)
    # S1
    rho = np.linspace(0, 1, 401)
    p_desc = np.asarray(eq.compute("p", grid=LinearGrid(rho=rho, NFP=eq.NFP))["p"])
    pe = jax.jit(sp.point_evaluator())
    p_fit = np.asarray(pe(jnp.asarray(rho), jnp.zeros_like(jnp.asarray(rho)), jnp.zeros_like(jnp.asarray(rho))))[:, 0]
    e1 = float(np.abs(p_fit - p_desc).max() / np.abs(p_desc).max())
    print(f"S1 fit of DESC p(rho): max error / max p = {e1:.2e}")
    ok &= e1 < 1e-3
    # S2
    rng = np.random.default_rng(1)
    c = rng.normal(size=sp.n) * np.abs(sp.c).max()
    sp2 = sp.with_c(c)
    g = TensorGrid(np.linspace(0.05, 0.98, 9), 2 * np.pi * np.arange(12) / 12, 2 * np.pi * np.arange(7) / 7, eq.NFP)
    RZ = Separable([eq.R_basis, eq.Z_basis], [eq.R_lmn, eq.Z_lmn], eq.NFP)
    geo = TensorGeometry(RZ, g)
    tp = sp2.tensor(g, geo, RZ)
    p_t, gp_t = (np.asarray(x) for x in tp(jnp.asarray(c)))
    R_, T_, Z_ = np.meshgrid(np.asarray(g.rho), np.asarray(g.theta), np.asarray(g.zeta), indexing="ij")
    pe2 = jax.jit(sp2.point_evaluator())
    pt = np.asarray(pe2(jnp.asarray(R_.ravel()), jnp.asarray(T_.ravel()), jnp.asarray(Z_.ravel())))
    e2p = float(np.abs(pt[:, 0] - p_t.ravel()).max() / np.abs(pt[:, 0]).max())
    e2g = float(np.abs(pt[:, 1:] - gp_t.reshape(-1, 3)).max() / np.abs(pt[:, 1:]).max())
    print(f"S2 tensor vs point evaluator (random c): p {e2p:.2e}, grad p {e2g:.2e}")
    ok &= e2p < 1e-10 and e2g < 1e-8
    # S3
    th = np.linspace(0, 2 * np.pi, 33)
    pa = np.asarray(pe2(jnp.full(33, 1e-3), jnp.asarray(th), jnp.zeros(33)))[:, 0]
    pb = np.asarray(pe2(jnp.ones(33), jnp.asarray(th), jnp.asarray(th)))[:, 0]
    e3 = float(np.ptp(pa) / np.abs(pt[:, 0]).max())
    print(f"S3 regularity: spread of p over theta at rho = 1e-3 / max p = {e3:.2e}; max |p| on the edge {np.abs(pb).max():.2e}")
    ok &= e3 < 1e-6 and np.abs(pb).max() < 1e-10 * np.abs(pt[:, 0]).max()
    # S4: central differences of the point p in (rho, theta, zeta), through TensorGeometry.gradient
    h = 1e-6
    base = [jnp.asarray(R_.ravel()), jnp.asarray(T_.ravel()), jnp.asarray(Z_.ravel())]
    d = []
    for k in range(3):
        up = list(base)
        dn = list(base)
        up[k], dn[k] = up[k] + h, dn[k] - h
        d.append((np.asarray(pe2(*up))[:, 0] - np.asarray(pe2(*dn))[:, 0]) / (2 * h))
    gfd = np.asarray(geo.gradient(*(jnp.asarray(x.reshape(g.shape)) for x in d))).reshape(-1, 3)
    e4 = float(np.abs(gfd - gp_t.reshape(-1, 3)).max() / np.abs(gfd).max())
    print(f"S4 grad p vs central differences (h = {h:g}): {e4:.2e}")
    ok &= e4 < 1e-5
    # S5
    print("S5 unknowns (stellarator symmetric): spline n_s x (M, N) against FZ (L = M):")
    for M, N in ((16, 6), (24, 8), (32, 12)):
        fz = count_fz(M, M, N)
        row = [f"FZ {M}/{M}/{N}: {fz}"]
        for coarse, dense, half in ((0.05, 0.01, 0.08), (0.04, 0.005, 0.08), (0.025, 0.005, 0.1)):
            from spline_pressure import breakpoints
            from spline_pressure import rational_centres
            bp = breakpoints(0, 1, coarse, dense, rational_centres(eq, ["2/1"]), half)
            n_s = len(bp) + 2
            row.append(f"knots ({coarse}, {dense}, {half}) n_s {n_s}: {count(n_s, M, N)}")
        print("   " + "; ".join(row))
    print("ALL PASSED" if ok else "FAILED")
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
