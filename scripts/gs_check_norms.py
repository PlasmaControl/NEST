"""Check NEST's force metric against DESC's: weights, normalization, F itself.

On NEST's collocation grid (``ForceResidual``, as in gs_x0_force.py):
  1. volume: sum of the weights dV against DESC's V (x 1/2 with stellarator
     symmetry: zeta covers half a period);
  2. normalization: fr.norm against DESC's <|grad(|B|^2)|/2mu0>_vol, recomputed
     here on DESC's QuadratureGrid, and that average on NEST's grid and weights;
  3. F: |J x B - grad p| from DESC's J, B, grad p (what gs_x0_force.py uses)
     against DESC's own "|F|" at the same points;
  4. <|F|>_vol / norm on NEST's grid against DESC's <|F|>_vol / norm on its
     QuadratureGrid, for collocation 2x and 3x (quadrature convergence).

run: sbatch --time=00:30:00 run_cpu.sbatch scripts/gs_check_norms.py [--eq ...] [--factors 2 2 1]
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
from desc.grid import QuadratureGrid  # noqa: E402

from desc_input import DESCInput  # noqa: E402
from force import ForceResidual  # noqa: E402

jax.config.update("jax_enable_x64", True)
TOK = os.path.join(ROOT, "desc_input", "tokamak_free_boundary", "tokamak_L16M16")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--eq", default=TOK + ".h5")
    ap.add_argument("--factors", type=float, nargs=3, default=(2, 2, 1),
                    help="potential resolution / equilibrium's (sets the collocation grid)")
    a = ap.parse_args()
    inp = DESCInput.load(a.eq)
    eq = inp.eq
    q = QuadratureGrid(L=2 * eq.L, M=2 * eq.M, N=2 * eq.N, NFP=eq.NFP)
    dq = inp.compute(["V", "<|F|>_vol", "<|grad(|B|^2)|/2mu0>_vol"], q)
    V = float(np.ravel(dq["V"])[0])
    norm_desc = float(np.ravel(dq["<|grad(|B|^2)|/2mu0>_vol"])[0])
    F_desc = float(np.ravel(dq["<|F|>_vol"])[0]) / norm_desc
    print(f"{inp}\nDESC (QuadratureGrid {2 * eq.L}, {2 * eq.M}, {2 * eq.N}): V = {V:.6f} m^3, "
          f"norm = {norm_desc:.6e}, <|F|>_vol / norm = {F_desc:.4e}\n")
    LP = [int(round(f * e)) for f, e in zip(a.factors, (eq.L, eq.M, eq.N))]
    for colloc in (2, 3):
        res = tuple(int(round(colloc * (u + v))) for u, v in zip(LP, (eq.L, eq.M, eq.N)))
        fr = ForceResidual(inp, None, resolution=res)
        frac = 0.5 if fr.sym else 1.0
        d = inp.compute_tensor(["|F|", "|grad(|B|^2)|/2mu0"], fr.tgrid.rho, fr.tgrid.theta,
                               fr.tgrid.zeta)
        desc = fr.desc_fields()
        F_mine = np.linalg.norm(np.cross(desc["J"], desc["B"]) - desc["grad(p)"], axis=1)
        F_D, gB = np.ravel(d["|F|"]), np.ravel(d["|grad(|B|^2)|/2mu0"])
        print(f"collocation {colloc}x of F's resolution: grid {fr.tgrid.shape}, "
              f"{len(fr.rtz):,} points")
        print(f"  1. sum dV = {fr.dV.sum():.6f} m^3, DESC V x {frac:g} = {frac * V:.6f} "
              f"(ratio {fr.dV.sum() / (frac * V):.8f})")
        print(f"  2. fr.norm = {fr.norm:.6e} (DESC {norm_desc:.6e}); <|grad B^2/2mu0|> on "
              f"NEST's grid = {fr.average(gB):.6e} (ratio {fr.average(gB) / norm_desc:.6f})")
        print(f"  3. |J x B - grad p| from DESC's J, B, grad p vs DESC's |F|: max rel diff "
              f"{np.max(np.abs(F_mine - F_D)) / np.max(F_D):.1e}")
        print(f"  4. <|F|> / norm on NEST's grid: {fr.average(F_D) / fr.norm:.4e} "
              f"(DESC's quadrature {F_desc:.4e}, ratio {fr.average(F_D) / fr.norm / F_desc:.4f})\n",
              flush=True)


if __name__ == "__main__":
    main()
