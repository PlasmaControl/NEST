"""single_layer_fast against single_layer (the slow, reference path) on the DSHAPE annulus.

For h = a / frac: max |S_fast - S_ref| / max |S_ref| per mode, and the two build times.
run (GPU): sbatch run.sbatch tests/checks/cyl_fast_S.py --fracs 20 40 80
"""
import argparse
import os
import sys
import time

import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.join(ROOT, "nest"))
from desc import set_device  # noqa: E402

set_device(os.environ.get("NEST_DEVICE", "cpu"))
import jax  # noqa: E402

jax.config.update("jax_enable_x64", True)
from annulus_poisson import AnnulusPoisson  # noqa: E402
from cylindrical_grid import CylindricalGrid  # noqa: E402
from desc_input import DESCInput  # noqa: E402
from single_layer import single_layer_matrix  # noqa: E402
from single_layer_fast import q_half, single_layer_matrix_fast  # noqa: E402
from toroidal_kernel import toroidal_Q  # noqa: E402

ap_ = argparse.ArgumentParser()
ap_.add_argument("--fracs", type=int, nargs="+", default=[20, 40])
a = ap_.parse_args()
x = np.logspace(-14, 1.5, 4001)
Q = toroidal_Q(x, 1)
Q0, Q1 = q_half(jax.numpy.asarray(x))
print(f"q_half against toroidal_Q, chi - 1 in [1e-14, 30]: Q-1/2 {np.max(np.abs(np.asarray(Q0) / Q[0] - 1)):.1e}, "
      f"Q1/2 {np.max(np.abs(np.asarray(Q1) / Q[1] - 1)):.1e}", flush=True)
inp = DESCInput.load(os.path.join(ROOT, "desc_input", "tokamak_free_boundary", "tokamak_L32M32_fb.h5"))
for frac in a.fracs:
    g = CylindricalGrid.around(inp, inp.a / frac, 2.0)
    ap = AnnulusPoisson(g.r_axis, g.z_axis, [0, 1])
    pts, _ = ap.boundary_points()
    t0 = time.time()
    Sf = single_layer_matrix_fast(ap, [0, 1], pts, verbose=True)
    tf = time.time() - t0
    os.environ["NEST_S_FAST"] = "0"
    t0 = time.time()
    Sr = single_layer_matrix(ap, [0, 1], pts)
    tr = time.time() - t0
    os.environ["NEST_S_FAST"] = "1"
    for k, m in enumerate((0, 1)):
        e = np.max(np.abs(Sf[k] - Sr[k])) / np.max(np.abs(Sr[k]))
        ef = np.linalg.norm(Sf[k] - Sr[k]) / np.linalg.norm(Sr[k])
        print(f"a/{frac}: N_b {ap.n_boundary}, m {m}: max {e:.1e}, Frobenius {ef:.1e}", flush=True)
    print(f"a/{frac}: fast {tf:.1f} s, reference {tr:.1f} s ({tr / tf:.0f}x)", flush=True)
