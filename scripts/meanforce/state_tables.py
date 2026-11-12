"""Tables of field, current, pressure and force for NEST fixed-boundary states (option A) on a DESC flux tensor grid,
for offline field-line tracing and field-line-averaged profiles (scripts/meanforce/compare_states.py).

For each --state (and for DESC itself, and the vacuum field = DESC without its helical current) the script saves,
on the grid (rho, theta, zeta) of DESC's flux coordinates, float32:
  brho, bth  = B.grad(rho) / B.grad(zeta), B.grad(theta) / B.grad(zeta)   (field-line ODE in (rho, theta), zeta as time)
  Bz, Bmag   = B.grad(zeta), |B|
  p          = the state's least-squares pressure (its c_p); DESC's p for DESC
  lam        = J.B / B^2; Jpar = J.B / |B|; Jperp = |J x B| / |B|; Jmag = |J|
  F = J x B - grad p:  Fmag = |F|, Fpar = F.b (= -b.grad p), Frad = F.grad(rho)/|grad(rho)|
B = B_DESC + B_ann[J(u) - J_fit] (option A), J = J(u) (TensorCurrent), as start_check_bext.py builds them. Also saved:
the grid, R and Z, DESC's lambda (theta* = theta + lambda), d theta*/d theta, sqrt(g), DESC's iota(rho), the force
normalization (norm, F_desc_vol), and the (m, k) spectra b_mk(rho) of b^rho for m = 1..6, k = 1.

usage: python scripts/meanforce/state_tables.py --eq EQ --LMN 40 40 6 --frac 100 --N-phi 25 \\
           --state S=path.npz --state b1=path.npz --out out/tables
"""
import argparse
import json
import os
import sys
import time
from types import SimpleNamespace

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
from cylindrical_grid import CylindricalInterpolator  # noqa: E402
from desc_input import DESCInput  # noqa: E402
from flux_interp import TensorCurrent, TensorGrid  # noqa: E402
from force import ForceResidual  # noqa: E402
from free_boundary import reference  # noqa: E402
from objective import VECTOR_PARITY, Fields  # noqa: E402
from optimize_fb import annulus  # noqa: E402
from pressure import Pressure  # noqa: E402

T0 = time.time()


def log(*s):
    print(f"[{time.time() - T0:8.1f} s]", *s, flush=True)


ap = argparse.ArgumentParser()
ap.add_argument("--eq", required=True)
ap.add_argument("--LMN", type=int, nargs=3, required=True)
ap.add_argument("--colloc", type=float, default=1.5)
ap.add_argument("--frac", type=float, default=60)
ap.add_argument("--pad", type=float, default=0.02)
ap.add_argument("--N-phi", type=int, default=25)
ap.add_argument("--rho-grid", type=float, nargs=3, default=(0.02, 0.995, 391))
ap.add_argument("--ntheta", type=int, default=128)
ap.add_argument("--nzeta", type=int, default=32)
ap.add_argument("--mmax", type=int, default=6)
ap.add_argument("--state", action="append", default=[], help="NAME=path.npz (u, c_p)")
ap.add_argument("--chunk", type=int, default=8, help="rho slabs per chunk for the per-point algebra")
ap.add_argument("--out", required=True)
a = ap.parse_args()
os.makedirs(a.out, exist_ok=True)
f32 = np.float32

# ------------------------------------------------------------------ setup (as start_check_bext.py)
inp = DESCInput.load(a.eq, field=None)
eq = inp.eq
kw = {"L": a.LMN[0], "M": a.LMN[1], "N": a.LMN[2]}
cur = fit_cache.fit(BoundaryPotential, inp, **kw)
pre = fit_cache.fit(Pressure, inp, edge_order=2, **kw)
b = cur.basis
res = tuple(int(round(a.colloc * (u + v))) for u, v in ((b.L, eq.L), (b.M, eq.M), (b.N, eq.N)))
ref = reference(inp, res, current=cur)
box = annulus(inp, inp.a / a.frac, a.pad, a.N_phi)
box.locate_plasma_planes(inp, sym=bool(eq.sym), verbose=False)
fr = ForceResidual(inp, box, resolution=res)
fr.norm, fr.F_desc_vol = ref["norm"], ref["F_desc_vol"]
fr.B_ext = np.zeros((len(fr.x), 3))
fields = Fields(fr, cur, pre, dtype=jnp.float64, cyl_stencil=True)
solver = fields.solver
u0 = jnp.asarray(cur.u)
log(f"{inp}; basis {b.L}/{b.M}/{b.N}; annulus {box}; M_DESC {ref['F_desc_vol'] / ref['norm']:.4e}")

# ------------------------------------------------------------------ grid and DESC quantities
rho = np.linspace(*a.rho_grid[:2], int(a.rho_grid[2]))
th = 2 * np.pi * np.arange(a.ntheta) / a.ntheta
NFP = eq.NFP
ze = 2 * np.pi * np.arange(a.nzeta) / (a.nzeta * NFP)
shape = (len(rho), len(th), len(ze))
names = ["R", "Z", "e^rho", "e^theta", "e^zeta", "B^theta", "B^zeta", "lambda", "lambda_t", "iota", "J", "B",
         "sqrt(g)", "p", "grad(p)"]
d = inp.compute_tensor(names, rho, th, ze)
log("DESC tensor quantities done")
pts = np.column_stack([d["R"], np.broadcast_to(ze[None, None, :], shape).ravel(), d["Z"]])
itp = CylindricalInterpolator(fields.cyl_grid, pts, dtype=jnp.float64)
tp = np.broadcast_to(th[None, :, None], shape).ravel() + d["lambda"]
tpt = 1.0 + d["lambda_t"]
zz = np.broadcast_to(ze[None, None, :], shape).ravel()
nodes = np.stack(np.meshgrid(rho, th, ze, indexing="ij"), -1).reshape(-1, 3)
erho = np.asarray(d["e^rho"])
gr = np.linalg.norm(erho, axis=1)
np.savez(os.path.join(a.out, "grid.npz"), rho=rho, theta=th, zeta=ze, NFP=NFP,
         R=d["R"].reshape(shape).astype(f32), Z=d["Z"].reshape(shape).astype(f32),
         lam_desc=d["lambda"].reshape(shape).astype(f32), tpt=tpt.reshape(shape).astype(f32),
         sqrtg=np.asarray(d["sqrt(g)"]).reshape(shape).astype(f32), iota=np.asarray(d["iota"]).reshape(shape)[:, 0, 0],
         norm=ref["norm"], F_desc_vol=ref["F_desc_vol"])
spec = {}


def b_spectrum(brho):
    out = {}
    for m in range(1, a.mmax + 1):
        ph = np.exp(-1j * (m * tp + NFP * zz)) * tpt
        c = 2 * (brho * ph).reshape(shape).mean(axis=(1, 2))
        out[m] = [np.abs(c).tolist(), np.angle(c).tolist()]
    return out


def tables(tag, B, J, p, gp):
    """Per-point algebra on (n, 3) cylindrical components; saves the tag's npz."""
    Bz = d["B^zeta"] + np.sum((B - d["B"]) * d["e^zeta"], 1)
    brho = np.sum(B * erho, 1) / Bz if tag != "DESC" else np.zeros(len(B))
    bth = np.sum(B * d["e^theta"], 1) / Bz
    if tag == "DESC":
        bth = d["B^theta"] / d["B^zeta"]
    Bm = np.linalg.norm(B, axis=1)
    out = {"brho": brho, "bth": bth, "Bz": Bz, "Bmag": Bm}
    if J is not None:
        JB = np.sum(J * B, 1)
        JxB = np.cross(J, B)
        F = JxB - gp
        out.update(p=p, lam=JB / Bm**2, Jpar=JB / Bm, Jperp=np.linalg.norm(JxB, axis=1) / Bm,
                   Jmag=np.linalg.norm(J, axis=1), Fmag=np.linalg.norm(F, axis=1), Fpar=np.sum(F * B, 1) / Bm,
                   Frad=np.sum(F * erho, 1) / gr)
    np.savez(os.path.join(a.out, f"{tag}.npz"), **{k: np.asarray(v).reshape(shape).astype(f32) for k, v in out.items()})
    spec[tag] = b_spectrum(brho) if tag != "DESC" else None
    if J is not None:
        w = (np.abs(np.asarray(d["sqrt(g)"]))).ravel()
        log(f"{tag}: grid volume-mean |F| / M_DESC {np.sum(w * out['Fmag']) / np.sum(w) / ref['F_desc_vol']:.4f}")
    json.dump(spec, open(os.path.join(a.out, "spectra.json"), "w"))


# DESC itself
tables("DESC", np.asarray(d["B"]), np.asarray(d["J"]), np.asarray(d["p"]).ravel(), np.asarray(d["grad(p)"]).reshape(-1, 3))
# J_fit's field at the points (option A reference) and the vacuum field (DESC minus its helical current)
J_fit_rows = np.asarray(fields.to_box(fields.J_flux(u0, fields.I).astype(fields.cyl_dtype), VECTOR_PARITY))
J_D_rows = np.asarray(inp.compute_points(["J"], np.asarray(fields.rows_rtz))["J"])
m_fit = solver.field_modes_F(*fields._cyl_sources(fields._cyl_rows(jnp.asarray(J_fit_rows))))
Jbox_D = np.asarray(fields._cyl_rows(jnp.asarray(J_D_rows)))[np.asarray(fields.src)].reshape(fields.shape + (3,))
J_hel = Jbox_D - Jbox_D.mean(axis=0, keepdims=True)
m_vac = solver.field_modes_F(*solver.sources(jnp.asarray(J_hel)))
tables("vacuum", np.asarray(d["B"]) - np.asarray(itp(*m_vac)), None, None, None)
log("vacuum done")
tc = TensorCurrent(cur.S, cur._RZ, TensorGrid(rho, th, ze, NFP))
for spec_ in a.state:
    tag, path = spec_.split("=", 1)
    st = np.load(path)
    u = jnp.asarray(st["u"])
    m_u = fields.cyl_field_modes(u)
    dB = np.asarray(itp(*jax.tree.map(lambda x, y: x - y, m_u, m_fit)))
    J = np.asarray(tc(u, fields.I)).reshape(-1, 3)
    pg = np.asarray(pre.with_c(jnp.asarray(st["c_p"])).evaluate(SimpleNamespace(nodes=nodes)))
    tables(tag, np.asarray(d["B"]) + dB, J, pg[:, 0], pg[:, 1:4])
    log(f"{tag} done")
log("all done")
