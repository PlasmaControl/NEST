"""Island-seeded start states for the ripple study: J_fit with its lab-frame helical current removed in
a DESC-rho window around a resonant surface, refitted into the run's BoundaryPotential basis.

On the annulus box (cylindrical nodes (R, phi, Z)), J_fit's helical part is J minus its toroidal
mean at fixed (R, Z) (the current whose field shields the ripple; removing all of it gives the
'vacuum' perturbation of start_check_bext.py). The target is

    J_t = J_fit - W(rho) J_hel,   W = 1 for |rho - rho_c| <= hw, cos^2 taper to 0 over a further tw,

and u_seed = u_fit + du minimizes sum over the plasma box nodes of |J(u) - J_t|^2 (matrix-free CG on
the normal equations; J is affine in u at fixed net current I). Under option A the seed's field is
B_DESC - B_ann[J_fit - J(u_seed)], i.e. DESC's field with that part of the shielding current removed.
--window -1 removes the helical current everywhere (the vacuum-drive reference).

usage: python scripts/meanforce/island_seed.py --eq EQ --LMN 48 48 6 --frac 60 --N-phi 25 \\
           --window 0.6375 0.07 0.03 --window -1 0 0 --out out/seed48
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

jax.config.update("jax_enable_x64", True)

import fit_cache  # noqa: E402
from boundary_potential import BoundaryPotential  # noqa: E402
from desc_input import DESCInput  # noqa: E402
from force import ForceResidual  # noqa: E402
from free_boundary import reference  # noqa: E402
from objective import VECTOR_PARITY, Fields  # noqa: E402
from optimize_fb import annulus  # noqa: E402
from pressure import Pressure  # noqa: E402
import basis_args as BA  # noqa: E402

ap = argparse.ArgumentParser()
ap.add_argument("--eq", required=True)
ap.add_argument("--LMN", type=int, nargs=3, required=True)
ap.add_argument("--colloc", type=float, default=1.5)
ap.add_argument("--frac", type=float, default=60)
ap.add_argument("--pad", type=float, default=0.02)
ap.add_argument("--N-phi", type=int, default=25)
ap.add_argument("--window", type=float, nargs=3, action="append", required=True, metavar=("RHO_C", "HW", "TW"),
                help="remove the helical current for |rho - RHO_C| <= HW, tapered over TW; RHO_C < 0: everywhere")
ap.add_argument("--cg", type=int, default=600)
ap.add_argument("--cg-tol", type=float, default=1e-10)
ap.add_argument("--out", required=True)
BA.add_basis_args(ap)
a = ap.parse_args()
t0 = time.time()


def log(*s):
    print(f"[{time.time() - t0:8.1f} s]", *s, flush=True)


inp = DESCInput.load(a.eq, field=None)
eq = inp.eq
cur, pre, res, rho_nodes = BA.build(inp, a, a.LMN, a.colloc, log=log)
b = cur.basis
ref = reference(inp, res, current=cur, rho_nodes=rho_nodes)
box = annulus(inp, inp.a / a.frac, a.pad, a.N_phi)
box.locate_plasma_planes(inp, sym=bool(eq.sym), verbose=False)
fr = ForceResidual(inp, box, resolution=res, rho_nodes=rho_nodes)
fr.norm, fr.F_desc_vol = ref["norm"], ref["F_desc_vol"]
fr.B_ext = np.zeros((len(fr.x), 3))
fields = Fields(fr, cur, pre, dtype=jnp.float64, cyl_stencil=True)
log(f"basis {b.L}/{b.M}/{b.N}, n_u {cur.n_free}; annulus {box}")

u0 = jnp.asarray(cur.u)
src = np.asarray(fields.src)


def box_J(u):
    """J(u) on the annulus box in cylindrical components (zero outside the plasma)."""
    rows = fields.to_box(fields.J_flux(u, fields.I).astype(fields.cyl_dtype), VECTOR_PARITY)
    ext = fields._cyl_rows(rows)
    return ext[src].reshape(fields.shape + (3,))


rho_rows = np.asarray(fields.rows_rtz)[:, 0]
rho_ext = rho_rows if fields.half is None else np.concatenate([rho_rows, rho_rows[np.asarray(fields.half_rows)]])
rho_box = np.concatenate([rho_ext, [np.nan]])[src].reshape(fields.shape)
inside = np.isfinite(rho_box)
Jb0 = np.asarray(box_J(u0))
Jhel = Jb0 - Jb0.mean(axis=0, keepdims=True)
rmsJ = float(np.sqrt(np.mean(np.sum(Jb0[inside] ** 2, -1))))
log(f"J_fit on the box: rms {rmsJ:.4e} A/m^2 over {int(inside.sum())} plasma nodes; helical rms "
    f"{float(np.sqrt(np.mean(np.sum(Jhel[inside] ** 2, -1)))) / rmsJ:.4f} of it")

fJ, lin = jax.linearize(box_J, u0)
lin_T = jax.linear_transpose(lin, u0)
mask = jnp.asarray(inside[..., None], dtype=Jb0.dtype)


def normal(v):
    return lin_T(mask * lin(v))[0]


rep = {"args": vars(a), "rms J_fit [A/m^2]": rmsJ, "seeds": []}
for rc, hw, tw in a.window:
    if rc < 0:
        Wt = np.ones_like(rho_box)
        tag = "all"
    else:
        x = np.abs(rho_box - rc)
        Wt = np.where(x <= hw, 1.0, np.where(x <= hw + tw, np.cos(0.5 * np.pi * (x - hw) / max(tw, 1e-12)) ** 2, 0.0))
        tag = f"r{rc:.4f}_w{hw:g}"
    Wt = np.where(inside, Wt, 0.0)
    dJt = -Wt[..., None] * Jhel  # target change of the box current
    rhs = lin_T(mask * jnp.asarray(dJt))[0]
    t1 = time.time()
    du, info = jax.scipy.sparse.linalg.cg(normal, rhs, tol=a.cg_tol, maxiter=a.cg)
    dJ = np.asarray(lin(du))
    err = float(np.sqrt(np.sum((dJ - dJt)[inside] ** 2) / max(np.sum(dJt[inside] ** 2), 1e-300)))
    rel = float(np.sqrt(np.mean(np.sum(dJ[inside] ** 2, -1)))) / rmsJ
    tgt = float(np.sqrt(np.mean(np.sum(dJt[inside] ** 2, -1)))) / rmsJ
    us = np.asarray(u0 + du)
    np.savez(a.out + f"_{tag}.npz", u=us, c_p=np.asarray(pre.c), nu=np.zeros(3), rho=1.0, eps=0.0)
    e = {"tag": tag, "window": [rc, hw, tw], "target rms |dJ|/rms J": tgt, "fitted rms |dJ|/rms J": rel,
         "relative fit error of dJ": err, "cg info": None if info is None else int(np.asarray(info)),
         "time [s]": time.time() - t1}
    rep["seeds"].append(e)
    log(f"seed {tag}: target rms|dJ|/rms J {tgt:.4e}, fitted {rel:.4e}, relative fit error {err:.3f} "
        f"({time.time() - t1:.0f} s) -> {a.out}_{tag}.npz")
    json.dump(rep, open(a.out + ".json", "w"), indent=1)
log("done")
