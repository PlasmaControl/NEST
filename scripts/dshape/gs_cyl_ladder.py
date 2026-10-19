"""The cylindrical-annulus B_plasma on the DSHAPE tokamak against the exact axisymmetric
field, as h -> 0: how far can one GPU push it, and does it reach DESC's level?

At the J-fitted start x0 (J fitted to J_DESC on the run's collocation points) of the
L = M = 32 free-boundary tokamak in its coils' field, for each h = a / frac:
  - the annulus grid (N_phi = 1: axisymmetric), its node count, the build and the
    B_plasma times, the device's peak memory;
  - rms |B_plasma,cyl - B_plasma,exact| / rms |B_DESC| (axisym_field.py, converged to
    ~1e-9), by rho bin;
  - rms |B_coil + B_plasma,cyl - B_DESC| and <|F|>/norm at x0 (DESC: 5.9e-9).
Writes results/gs/cyl_ladder.json.

run (GPU): sbatch run.sbatch scripts/dshape/gs_cyl_ladder.py --fracs 20 40 80 160 [--factors 2 2 1]
"""

import argparse
import json
import os
import sys
import time

import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.join(ROOT, "nest"))

from desc import set_device  # noqa: E402

set_device(os.environ.get("NEST_DEVICE", "cpu"))

import jax  # noqa: E402
import jax.numpy as jnp  # noqa: E402

import fit_cache  # noqa: E402
from axisym_field import plasma_field_cached  # noqa: E402
from boundary_potential import BoundaryPotential  # noqa: E402
from boxes import make_box  # noqa: E402
from cartesian_grid import cylindrical_to_cartesian  # noqa: E402
from dense import force_matched_u  # noqa: E402
from desc_input import DESCInput  # noqa: E402
from flux_interp import TensorGrid  # noqa: E402
from force import ForceResidual  # noqa: E402
from objective import Fields  # noqa: E402
from pressure import Pressure  # noqa: E402

jax.config.update("jax_enable_x64", True)
TOK = os.path.join(ROOT, "desc_input", "tokamak_free_boundary")


def peak_gb():
    try:
        st = jax.devices()[0].memory_stats() or {}
        return st.get("peak_bytes_in_use", float("nan")) / 1e9
    except Exception:  # noqa: BLE001
        return float("nan")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--eq", default=os.path.join(TOK, "tokamak_L32M32_fb.h5"))
    ap.add_argument("--coils", default=os.path.join(TOK, "tokamak_L32M32_field.h5"))
    ap.add_argument("--factors", type=float, nargs=3, default=(2, 2, 1))
    ap.add_argument("--edge-power", type=int, default=3)
    ap.add_argument("--fracs", type=int, nargs="+", default=[20, 40, 80])
    ap.add_argument("--margin", type=float, default=2.0)
    ap.add_argument("--flux", type=int, nargs="*", default=[],
                    help="flux-grid sizes n_rho n_theta [n_zeta] carrying J to the nodes (default "
                         "4L x 6M x 8, Fields' own); several pairs: one row per pair at each h")
    ap.add_argument("--order", type=int, nargs="*", default=[4], help="Lagrange orders of that interpolation")
    ap.add_argument("--out", default=os.path.join(ROOT, "results", "gs", "cyl_ladder.json"))
    a = ap.parse_args()
    t0 = time.time()
    inp = DESCInput.load(a.eq, field=a.coils)
    eq = inp.eq
    ep = {"edge_power": a.edge_power} if a.edge_power != 2 else {}
    cur = fit_cache.fit(BoundaryPotential, inp, factors=tuple(a.factors), **ep)
    pre = fit_cache.fit(Pressure, inp, factors=tuple(a.factors), edge_order=2)
    b = cur.basis
    res = tuple(int(round(2 * (u + v))) for u, v in ((b.L, eq.L), (b.M, eq.M), (b.N, eq.N)))
    fr0 = ForceResidual(inp, None, resolution=res)
    desc = {k: np.asarray(v) for k, v in fr0.desc_fields().items()}
    u0, _ = force_matched_u(fr0, cur, eps=np.inf, desc=desc)
    rtz, phi = np.asarray(fr0.rtz), np.asarray(fr0.phi)
    w = fr0.dV / fr0.dV.sum()
    rho = rtz[:, 0]
    rmsB = float(np.sqrt(np.sum(w * np.sum(desc["B"] ** 2, 1))))
    B_coil = np.asarray(fr0.B_ext)
    t1 = time.time()
    B_acc = np.asarray(cylindrical_to_cartesian(plasma_field_cached(cur, rtz, u=u0), phi))
    print(f"{inp}\ncollocation {fr0.tgrid.shape} ({len(rho):,} points); exact field {time.time() - t1:.0f} s; "
          f"|B_coil + B_exact - B_DESC| {np.sqrt(np.sum(w * np.sum((B_coil + B_acc - desc['B']) ** 2, 1))) / rmsB:.2e}",
          flush=True)
    edges = (0.0, 0.5, 0.8, 0.9, 0.95, 1.0)

    def rms(v, m=None):
        m = np.ones(len(v), bool) if m is None else m
        return float(np.sqrt(np.sum((w * np.sum(v**2, 1))[m]) / np.sum(w[m]))) / rmsB

    out = []
    fluxes = [None] if not a.flux else [tuple(a.flux[i : i + 3]) for i in range(0, len(a.flux), 3)]
    for frac in a.fracs:
        t1 = time.time()
        box = make_box(inp, frac, "cylindrical", bool(eq.sym), a.margin)
        t_loc = time.time() - t1
        for fl in fluxes:
            for order in a.order:
                t1 = time.time()
                fr = ForceResidual(inp, box, resolution=res)
                fg = None if fl is None else TensorGrid.uniform(fl[0], fl[1], fl[2], eq.NFP, sym=bool(eq.sym))
                fields = Fields(fr, cur, pre, kernel="cylindrical", dtype=jnp.float64, flux_grid=fg, order=order)
                t_build = time.time() - t1
                x0 = jnp.concatenate([jnp.asarray(u0), fields.x0[fields.n_u:]])
                f = {k: np.asarray(v) for k, v in fields.compute(x0).items()}
                t2 = time.time()
                f = {k: np.asarray(v) for k, v in fields.compute(x0).items()}  # timed second call
                t_eval = time.time() - t2
                e = f["B_plasma"] - B_acc
                F = np.linalg.norm(np.cross(f["J"], B_coil + f["B_plasma"]) - f["grad_p"], axis=1) / fr.norm
                row = {"frac": frac, "flux": None if fl is None else list(fl), "order": order,
                       "grid": list(box.shape), "nodes": int(box.size), "inside": int(len(box.inside)),
                       "locate s": t_loc, "build s": t_build, "eval s": t_eval, "peak GB": peak_gb(),
                       "B_plasma err": rms(e),
                       "B_plasma err by rho": [rms(e, (rho > x) & (rho <= y)) for x, y in zip(edges[:-1], edges[1:])],
                       "B - B_DESC": rms(B_coil + f["B_plasma"] - desc["B"]),
                       "F(x0)": float(np.sum(w * F))}
                out.append(row)
                print(json.dumps(row), flush=True)
                with open(a.out, "w") as fo:
                    json.dump({"args": vars(a), "rows": out}, fo, indent=1)
                del fields, fr, f
                jax.clear_caches()
        del box
    print(f"saved {a.out}; total {time.time() - t0:.0f} s")


if __name__ == "__main__":
    main()
