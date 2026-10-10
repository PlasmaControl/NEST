"""Local-column block preconditioner (``precondition.BlockPreconditioner``) for
the setup of ``spectrum.py`` (same arguments), saved for ``precond_test.py``.

run (GPU): sbatch run.sbatch scripts/local_blocks.py [--LMN 16 16 14] --out results/blocks16
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

import fit_cache  # noqa: E402
from boundary_potential import BoundaryPotential  # noqa: E402
from desc_input import DESCInput  # noqa: E402
from force import ForceResidual  # noqa: E402
from grid import CartesianGrid  # noqa: E402
from objective import Fields, ForceBalance, ObjectiveFunction  # noqa: E402
from precondition import BlockPreconditioner, mode_groups, mode_labels  # noqa: E402
from pressure import Pressure  # noqa: E402

jax.config.update("jax_enable_x64", True)
EQ = os.path.join(ROOT, "desc_input", "precise_QA_free_boundary_smooth_J",
                  "precise_QA_free_boundary_smooth_J_edgeJ_r0.3.h5")
COILS = os.path.join(ROOT, "desc_input", "precise_QA_free_boundary", "precise_QA_helical_coils.h5")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--LMN", type=int, nargs=3, default=(16, 16, 14))
    ap.add_argument("--frac", type=int, default=10)
    ap.add_argument("--colloc", type=float, default=2)
    ap.add_argument("--out", default=os.path.join(ROOT, "results", "blocks16"))
    a = ap.parse_args()
    inp = DESCInput.load(EQ, field=COILS)
    eq = inp.eq
    L, M, N = a.LMN
    cur = fit_cache.fit(BoundaryPotential, inp, L=L, M=M, N=N)
    pre = fit_cache.fit(Pressure, inp, edge_order=2)
    box = CartesianGrid.periodic_box(inp, inp.a / a.frac, odd=True).locate_plasma(inp)
    b = cur.basis
    res = lambda f: tuple(int(round(f * (u + v))) for u, v in ((b.L, eq.L), (b.M, eq.M), (b.N, eq.N)))
    fields = Fields(ForceResidual(inp, box, resolution=res(a.colloc)), cur, pre, dtype=jnp.float64)
    obj = ObjectiveFunction([ForceBalance()], fields).build()
    free = np.arange(obj.dim_x) < fields.n_u
    groups = mode_groups(mode_labels(fields, cur, pre), free)
    P = BlockPreconditioner(obj, obj.x0, groups)
    np.savez(a.out + ".npz", idx=np.asarray(P.idx), V=np.asarray(P.V), w=np.asarray(P.w),
             n=fields.n_u)
    print(f"saved {a.out}.npz")


if __name__ == "__main__":
    main()
