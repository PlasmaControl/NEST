"""Magnetic energy, thermal energy, magnetic helicity and fluxes of NEST fixed-boundary states (option A) in the plasma
volume (DESC's fixed LCFS), in float64 on a fine DESC flux grid, processed in radial chunks.

Fields: B = B_DESC + B_ann[J(u) - J_fit] (option A, as start_check_bext.py / state_tables.py). For each state:
  W_B  = int B^2 / (2 mu0) dV;  W_th = int p / (gamma - 1) dV (gamma 5/3, the state's least-squares p from c_p)
  K    = int A.B dV with A built from B in DESC coordinates (rho, theta, zeta), gauge A_rho = 0, A = 0 on DESC's axis:
           A_theta = int_0^rho sqrt(g) B^zeta drho',   A_zeta = - int_0^rho sqrt(g) B^theta drho'
         (curl A = B for B^theta and B^zeta exactly; B^rho follows from div B = 0, reported as a check).
         K is gauge invariant only if B.n = 0 on the LCFS; NEST's fixed-boundary states do not enforce it, so the
         normal field rms(sqrt(g) B^rho) / rms(sqrt(g) B^zeta) at rho = 1 is reported with K.
  Phi_tor = oint A_theta(rho = 1) d theta (toroidal flux), Phi_pol = - oint A_zeta(rho = 1) d zeta (poloidal flux from
         DESC's axis to the LCFS), each averaged over the other angle.
DESC itself and J_fit (whose field equals DESC's exactly under option A) are the references.

usage: python scripts/meanforce/energy_helicity.py --eq EQ --LMN 40 40 6 --frac 100 --N-phi 25 \\
           --state S=path.npz --state b1=path.npz --out out/eh_R1.json
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
from force import ForceResidual  # noqa: E402
from free_boundary import reference  # noqa: E402
from objective import VECTOR_PARITY, Fields  # noqa: E402
from optimize_fb import annulus  # noqa: E402
from pressure import Pressure  # noqa: E402

T0 = time.time()
MU0 = 4e-7 * np.pi


def log(*s):
    print(f"[{time.time() - T0:8.1f} s]", *s, flush=True)


ap = argparse.ArgumentParser()
ap.add_argument("--eq", required=True)
ap.add_argument("--LMN", type=int, nargs=3, required=True)
ap.add_argument("--colloc", type=float, default=1.5)
ap.add_argument("--frac", type=float, default=60)
ap.add_argument("--pad", type=float, default=0.02)
ap.add_argument("--N-phi", type=int, default=25)
ap.add_argument("--nrho", type=int, default=801)
ap.add_argument("--rho0", type=float, default=1e-3)
ap.add_argument("--ntheta", type=int, default=256)
ap.add_argument("--nzeta", type=int, default=32)
ap.add_argument("--chunk", type=int, default=40, help="radii per chunk")
ap.add_argument("--state", action="append", default=[], help="NAME=path.npz (u, c_p)")
ap.add_argument("--out", required=True)
a = ap.parse_args()

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
log(f"{inp}; basis {b.L}/{b.M}/{b.N}; annulus {box}")
J_fit_rows = np.asarray(fields.to_box(fields.J_flux(u0, fields.I).astype(fields.cyl_dtype), VECTOR_PARITY))
m_fit = solver.field_modes_F(*fields._cyl_sources(fields._cyl_rows(jnp.asarray(J_fit_rows))))

cases = {"DESC": (None, None)}
for spec in a.state:
    tag, path = spec.split("=", 1)
    st = np.load(path)
    m_u = fields.cyl_field_modes(jnp.asarray(st["u"]))
    cases[tag] = (jax.tree.map(lambda x, y: x - y, m_u, m_fit), np.asarray(st["c_p"]))
log(f"field modes for {len(cases) - 1} states")

rho = np.linspace(a.rho0, 1.0, a.nrho)
th = 2 * np.pi * np.arange(a.ntheta) / a.ntheta
NFP = eq.NFP
ze = 2 * np.pi * np.arange(a.nzeta) / (a.nzeta * NFP)
store = {k: {q: np.zeros((a.nrho, a.ntheta, a.nzeta)) for q in ("Sr", "St", "Sz", "B2", "p")} for k in cases}
sg_all = np.zeros((a.nrho, a.ntheta, a.nzeta))
names = ["R", "Z", "e^rho", "e^theta", "e^zeta", "sqrt(g)", "B", "p"]
for i0 in range(0, a.nrho, a.chunk):
    rc = rho[i0:i0 + a.chunk]
    shp = (len(rc), a.ntheta, a.nzeta)
    d = inp.compute_tensor(names, rc, th, ze)
    pts = np.column_stack([d["R"], np.broadcast_to(ze[None, None, :], shp).ravel(), d["Z"]])
    itp = CylindricalInterpolator(fields.cyl_grid, pts, dtype=jnp.float64)
    sg = np.asarray(d["sqrt(g)"]).ravel()
    sg_all[i0:i0 + len(rc)] = sg.reshape(shp)
    nodes = np.stack(np.meshgrid(rc, th, ze, indexing="ij"), -1).reshape(-1, 3)
    BD = np.asarray(d["B"])
    for tag, (md, cp) in cases.items():
        B = BD if md is None else BD + np.asarray(itp(*md))
        S = store[tag]
        S["Sr"][i0:i0 + len(rc)] = (sg * np.sum(B * d["e^rho"], 1)).reshape(shp)
        S["St"][i0:i0 + len(rc)] = (sg * np.sum(B * d["e^theta"], 1)).reshape(shp)
        S["Sz"][i0:i0 + len(rc)] = (sg * np.sum(B * d["e^zeta"], 1)).reshape(shp)
        S["B2"][i0:i0 + len(rc)] = np.sum(B * B, 1).reshape(shp)
        pp = np.asarray(d["p"]).ravel() if cp is None else \
            np.asarray(pre.with_c(jnp.asarray(cp)).evaluate(SimpleNamespace(nodes=nodes)))[:, 0]
        S["p"][i0:i0 + len(rc)] = pp.reshape(shp)
    log(f"chunk rho {rc[0]:.4f}-{rc[-1]:.4f} done")

dth, dze = 2 * np.pi / a.ntheta, 2 * np.pi / (a.nzeta * NFP)
wr = np.zeros_like(rho)
wr[1:] += 0.5 * np.diff(rho)
wr[:-1] += 0.5 * np.diff(rho)
sgn = float(np.sign(np.mean(sg_all)))
dV = np.abs(sg_all) * wr[:, None, None] * dth * dze * NFP


def cumtrapz(f):
    out = np.zeros_like(f)
    out[1:] = np.cumsum(0.5 * (f[1:] + f[:-1]) * np.diff(rho)[:, None, None], axis=0)
    return out


def dfft(f, axis, n):
    k = np.fft.fftfreq(f.shape[axis], 1.0 / f.shape[axis]) * n
    sh = [1, 1, 1]
    sh[axis] = -1
    return np.real(np.fft.ifft(1j * k.reshape(sh) * np.fft.fft(f, axis=axis), axis=axis))


out = {"args": vars(a), "V [m^3]": float(dV.sum()), "states": {}}
try:  # DESC's own magnetic energy and volume (quadrature cross-check)
    dq = eq.compute(["W_B", "V"])
    out["DESC compute"] = {"W_B [J]": float(dq["W_B"]), "V [m^3]": float(dq["V"])}
    log(f"DESC compute: W_B {float(dq['W_B']):.9e} J, V {float(dq['V']):.6f} m^3; this grid V {float(dV.sum()):.6f}")
except Exception as e:  # noqa: BLE001
    log(f"DESC compute W_B failed: {e!r}")
for tag, S in store.items():
    At, Az = cumtrapz(S["Sz"]), -cumtrapz(S["St"])
    K = sgn * float(np.sum((At * S["St"] + Az * S["Sz"]) * wr[:, None, None]) * dth * dze * NFP)
    WB = float(np.sum(S["B2"] * dV) / (2 * MU0))
    Wth = float(np.sum(S["p"] * dV) * 1.5)
    phit = float(np.mean(At[-1].sum(0) * dth))  # oint A_theta d theta, averaged over zeta
    phip = float(np.mean(-Az[-1].sum(1) * dze * NFP))  # -oint A_zeta d zeta, averaged over theta
    rec = dfft(Az, 1, 1) - dfft(At, 2, NFP)  # d_theta A_zeta - d_zeta A_theta  ~  sqrt(g) B^rho - (sqrt(g) B^rho)|rho0
    tgt = S["Sr"] - S["Sr"][:1]
    divchk = float(np.sqrt(np.mean((rec - tgt)[1:-1] ** 2)) / max(np.sqrt(np.mean(S["Sz"] ** 2)), 1e-300))
    bn = float(np.sqrt(np.mean(S["Sr"][-1] ** 2)) / np.sqrt(np.mean(S["Sz"][-1] ** 2)))
    out["states"][tag] = {"W_B [J]": WB, "W_th [J]": Wth, "W_B + W_th [J]": WB + Wth, "K [Wb^2]": K,
                          "Phi_tor [Wb]": phit, "Phi_pol [Wb]": phip,
                          "B.n at LCFS: rms(sqrtg B^rho)/rms(sqrtg B^zeta)": bn,
                          "div check: rms(curl A - B)^rho / rms sqrtg B^zeta": divchk}
    log(f"{tag}: W_B {WB:.9e} J, W_th {Wth:.9e} J, K {K:.9e} Wb^2, Phi_tor {phit:.9e}, Phi_pol {phip:.9e}, "
        f"B.n {bn:.2e}, div {divchk:.2e}")
D = out["states"]["DESC"]
for tag, v in out["states"].items():
    v["vs DESC"] = {k: v[k] - D[k] for k in ("W_B [J]", "W_th [J]", "W_B + W_th [J]", "K [Wb^2]", "Phi_tor [Wb]",
                                             "Phi_pol [Wb]")}
json.dump(out, open(a.out, "w"), indent=1)
log("done")
