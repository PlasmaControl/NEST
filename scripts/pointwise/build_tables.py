"""Column tables of the pointwise problem for one resolution (``nest/pointwise_tables.py``).

    python scripts/pointwise/build_tables.py --frac 40 --factors 1.5 1.5 1.5 \
        --out results/pointwise/tables/a40_f15.npz

NCSX_L12M12N8 in its coils by default; the annulus at h = a / frac (planes from the arc
spacing h); collocation colloc x (basis + equilibrium) resolution; pressure basis
--p-factors (default --factors), p = (1 - rho^2) Q (Dirichlet). Checks the dense model
against the matrix-free Fields at the start and prints the start's force statistics.
"""

import argparse
import json
import os
import sys
import time

import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.join(ROOT, "nest"))
sys.path.insert(0, os.path.join(ROOT, "scripts"))
from desc import set_device  # noqa: E402

set_device(os.environ.get("NEST_DEVICE", "cpu"))
import jax  # noqa: E402
import jax.numpy as jnp  # noqa: E402

import fit_cache  # noqa: E402
import pointwise as pw  # noqa: E402
import pointwise_tables  # noqa: E402
from boundary_potential import BoundaryPotential  # noqa: E402
from desc_input import DESCInput  # noqa: E402
from force import ForceResidual  # noqa: E402
from objective import Fields  # noqa: E402
from optimize_qa import make_box  # noqa: E402
from pressure import Pressure  # noqa: E402

jax.config.update("jax_enable_x64", True)
NC = os.path.join(ROOT, "desc_input", "vmec_free_boundary", "NCSX")


def setup(eq, coils, factors, p_factors, frac, colloc, verbose=True):
    """The equilibrium input, the fits, the annulus, the collocation grid and Fields."""
    inp = DESCInput.load(eq, field=coils)
    cur = fit_cache.fit(BoundaryPotential, inp, factors=tuple(factors))
    pre = fit_cache.fit(Pressure, inp, factors=tuple(p_factors), edge_order=0)
    b, e = cur.basis, inp.eq
    pb = pre.basis
    res = tuple(int(round(colloc * (max(u, w) + v))) for u, w, v in
                ((b.L, pb.L, e.L), (b.M, pb.M, e.M), (b.N, pb.N, e.N)))
    box = make_box(inp, frac, "cylindrical", True)
    fr = ForceResidual(inp, box, resolution=res)
    fields = Fields(fr, cur, pre, dtype=jnp.float64, S_dtype=jnp.float64)
    if verbose:
        print(f"{inp}\nannulus a/{frac}: N_phi {box.N_phi}, r x z {box.r_axis.n + 1} x "
              f"{box.z_axis.n + 1}; potential {b.L}/{b.M}/{b.N} ({cur.n_free}), pressure "
              f"{pb.L}/{pb.M}/{pb.N} ({pb.num_modes}); collocation {fr.tgrid.shape} "
              f"({len(fr.rtz)} points)", flush=True)
    return inp, cur, pre, box, fr, fields


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--eq", default=os.path.join(NC, "NCSX_L12M12N8.h5"))
    ap.add_argument("--coils", default=os.path.join(NC, "NCSX_coils.h5"))
    ap.add_argument("--factors", type=float, nargs=3, default=(1.5, 1.5, 1.5))
    ap.add_argument("--p-factors", type=float, nargs=3, default=None)
    ap.add_argument("--frac", type=int, default=40)
    ap.add_argument("--colloc", type=float, default=1.5)
    ap.add_argument("--g-batch", type=int, default=16, help="annulus solves per batch for G "
                    "(a/80: 4; 16 fails there to create the cuFFT plan)")
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    t0 = time.time()
    pf = a.p_factors or a.factors
    inp, cur, pre, box, fr, fields = setup(a.eq, a.coils, a.factors, pf, a.frac, a.colloc)
    T = pointwise_tables.build(fields, cur, pre, g_batch=a.g_batch)
    T["meta"] = json.dumps({"eq": a.eq, "coils": a.coils, "factors": list(a.factors),
                            "p_factors": list(pf), "frac": a.frac, "colloc": a.colloc,
                            "N_phi": int(box.N_phi), "n_u": int(cur.n_free),
                            "n_p": int(pre.basis.num_modes), "points": int(len(fr.rtz)),
                            "basis": [int(cur.basis.L), int(cur.basis.M), int(cur.basis.N)],
                            "p_basis": [int(pre.basis.L), int(pre.basis.M), int(pre.basis.N)]})
    # the dense model against Fields at the start
    x0 = jnp.concatenate([jnp.asarray(T["u0"]), jnp.asarray(T["c0"])])
    bd = pw.DenseModel(T).base(x0)
    bf = pw.FieldsModel(fields).base(x0)
    err = float(jnp.max(jnp.abs(bd["F"] - bf["F"])) / jnp.max(jnp.abs(bf["F"])))
    Fn = np.linalg.norm(np.asarray(bd["F"]), axis=1) / T["norm"]
    FD = np.linalg.norm(np.cross(T["J_desc"], T["B_desc"]) - T["grad_p_desc"], axis=1) / T["norm"]
    w = T["w"]
    T["F_ref_p99"], T["F_ref_max"] = float(np.percentile(Fn, 99)), float(Fn.max())
    print(f"dense vs Fields at x0: max |dF| / max |F| {err:.1e}\nstart |F|/norm: max "
          f"{Fn.max():.3e}, p99 {T['F_ref_p99']:.3e}, rms {np.sqrt(np.sum(w * Fn**2)):.3e}, "
          f"mean {np.sum(w * Fn):.3e}; DESC's own: max {FD.max():.3e}, p99 "
          f"{np.percentile(FD, 99):.3e}, rms {np.sqrt(np.sum(w * FD**2)):.3e}, mean "
          f"{np.sum(w * FD):.3e}", flush=True)
    os.makedirs(os.path.dirname(os.path.abspath(a.out)), exist_ok=True)
    pointwise_tables.save(a.out, T)
    print(f"saved {a.out} ({os.path.getsize(a.out) / 1e9:.1f} GB); {time.time() - t0:.0f} s")


if __name__ == "__main__":
    main()
