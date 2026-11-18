"""G1.2 / G1.3: does the LCFS B.n constraint make the field outside the plasma independent of J?

With B.n = 0 on the LCFS, I_p and Phi_tor fixed, the exterior field change dB_out = B_BS[dJ] outside Omega is
curl- and divergence-free, tangent to the LCFS, with zero circulation: it must vanish, and then W_Omega is the
free energy up to a constant. Here the annulus solver is not trusted: dB is computed by DIRECT Biot-Savart
quadrature of dJ (free_boundary.DirectField) at
  - the LCFS points (the collocation theta, zeta; true B.n, G1.2), and
  - offset shells x_bnd + d a n_hat (d = --shells, a = minor radius) and a few far points (G1.3),
for perturbations du of matched |J - J_fit| / rms:
  - random du, unconstrained and projected onto the constraints (LinearConstraints.restore, metric H_J);
  - saved states (--state NAME=path: du = u - u_fit) and pairs (--pair NAME=pathA,pathB: du = u_B - u_A).
Reported per du: rms(n.dB_direct)/rms|B_DESC| on the LCFS, the annulus' B.n modes, rms|dB_direct|/rms|B_DESC| on
each shell, and bounds on the exterior energy change, (1/mu0) int |B_LCFS| |dB| (linear term, with the LCFS field
magnitude as the field scale) and (1/2mu0) int |dB|^2, over the layer d in [0, max shell], relative to W_Omega.

usage: python scripts/energy/exterior_check.py --eq EQ --LMN 40 40 6 --frac 100 --N-phi 25 --colloc-rho 2.7 \\
           --bn-modes 40 6 --direct 32 4 --state S0=... --pair island=S0.npz,I1.npz --out out/g1
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
import meanforce as mf  # noqa: E402
import meanforce_fixed as mff  # noqa: E402
from boundary_potential import BoundaryPotential  # noqa: E402
from desc_input import DESCInput  # noqa: E402
from force import ForceResidual  # noqa: E402
from free_boundary import DirectField, Displacement, reference  # noqa: E402
from objective import Fields  # noqa: E402
from optimize_fb import annulus  # noqa: E402
from pressure import Pressure  # noqa: E402

T0 = time.time()
MU0 = 4e-7 * np.pi


def trap(y, x):
    y, x = np.asarray(y, float), np.asarray(x, float)
    return float(np.sum(0.5 * (y[1:] + y[:-1]) * np.diff(x)))


def log(*s):
    print(f"[{time.time() - T0:7.1f} s]", *s, flush=True)


ap = argparse.ArgumentParser()
ap.add_argument("--eq", required=True)
ap.add_argument("--bext", default="desc-fit", choices=("desc-fit",), help="option A only (as the solves)")
ap.add_argument("--LMN", type=int, nargs=3, required=True)
ap.add_argument("--frac", type=float, default=40)
ap.add_argument("--N-phi", type=int, default=25)
ap.add_argument("--pad", type=float, default=0.02)
ap.add_argument("--colloc", type=float, default=1.5)
ap.add_argument("--colloc-rho", type=float, default=None)
ap.add_argument("--bn-modes", type=int, nargs=2, required=True)
ap.add_argument("--direct", type=int, nargs=2, default=(32, 4), help="n_rho, q of the direct quadrature")
ap.add_argument("--bn-direct", type=int, nargs=2, default=None, metavar=("NRHO", "Q"),
                help="build the B.n constraint rows from direct Biot-Savart (LCFSDirect) instead of the annulus")
ap.add_argument("--shells", type=float, nargs="+", default=[0.02, 0.1, 0.5])
ap.add_argument("--n-random", type=int, default=3)
ap.add_argument("--state", action="append", default=[])
ap.add_argument("--pair", action="append", default=[])
ap.add_argument("--out", required=True)
a = ap.parse_args()
os.makedirs(os.path.dirname(a.out) or ".", exist_ok=True)
rng = np.random.default_rng(1)
res_all = {"args": vars(a)}

# ------------------------------------------------------------------ setup (as mf_fixed, --bext desc-fit)
inp0 = DESCInput.load(a.eq)
eq0 = inp0.eq
kw = {"L": a.LMN[0], "M": a.LMN[1], "N": a.LMN[2]}
cur0 = fit_cache.fit(BoundaryPotential, inp0, **kw)
pre0 = fit_cache.fit(Pressure, inp0, edge_order=2, **kw)
b = cur0.basis
res = tuple(int(round(a.colloc * (u + v))) for u, v in ((b.L, eq0.L), (b.M, eq0.M), (b.N, eq0.N)))
if a.colloc_rho:
    res = (int(round(a.colloc_rho * (b.L + eq0.L))),) + res[1:]
ref = reference(inp0, res, current=cur0)
box = annulus(inp0, inp0.a / a.frac, a.pad, a.N_phi)
box.locate_plasma_planes(inp0, sym=bool(eq0.sym), verbose=False, chunk=1 << 17)
fr = ForceResidual(inp0, box, resolution=res)
fr.norm, fr.F_desc_vol = ref["norm"], ref["F_desc_vol"]
fr.B_ext = np.zeros((len(fr.x), 3))
fields = Fields(fr, cur0, pre0, dtype=jnp.float64, cyl_stencil=True, mode_chunk=16, cyl_fused=True, cyl_rblock=32,
                cyl_order=8)  # exactly as mf_fixed.py (defaults --mode-chunk 16 --rblock 32 --interp-order 8)
u0 = jnp.asarray(cur0.u)
B_D = np.asarray(fields.desc["B"])
fields.B_ext = jnp.asarray(B_D - np.asarray(fields.B_plasma(u0)))
fr.B_ext = np.asarray(fields.B_ext)
Jfit = mff._J(fields, u0, fields.I)
sw = jnp.asarray(fields.sw)
rmsJ = float(jnp.sqrt(jnp.sum(sw[:, None] ** 2 * Jfit**2)))
G = mff.gram_P(fields, cur0, pre0, log=log)
A = mff.FixedAnchor(fields, cur0, pre0, u0, rmsJ, ref["norm"], G, log=log)
del G
per = 2 * eq0.NFP if fr.sym else eq0.NFP
A.enable_energy(0.0, mf.Weights(1.0, 1.0, 1.0), float(per * np.sum(fr.dV)), log=log)
n_u = A.n_u
log(f"basis {a.LMN}, n_u {n_u}, points {A.n_pts}, collocation {res}, annulus {box}")

bnf = mff.LCFSField(inp0, fields, fr, a.bn_modes[0], a.bn_modes[1], log=log)
bnd = None
if a.bn_direct:
    bnd = mff.LCFSDirect(bnf, cur0, eq0, a.bn_direct[0], a.bn_direct[1], log=log)
    Ct = bnd.rows(n_u, log=log)
    v_ = jnp.asarray(rng.standard_normal(n_u))
    m_f, m_r = np.asarray(bnd.modes_of(v_)), Ct.T @ np.asarray(v_)
    res_all["direct rows check"] = float(np.max(np.abs(m_f - m_r)) / np.max(np.abs(m_f)))
    log(f"[B.n direct] rows vs forward quadrature: max rel diff {res_all['direct rows check']:.2e}")
else:
    Ct = bnf.rows(fields, n_u, log=log)
LC = mff.LinearConstraints(cur0, fields, fr, u0, pin_current=False, pin_flux=True, log=log,
                           blocks=[("Bn", Ct, np.zeros(bnf.k))])
HJ = mff.gram_J(fields, cur0, pre0, rmsJ, log=log)
L = mff.metric_L(HJ)
del HJ

# ------------------------------------------------------------------ targets: LCFS, offset shells, far points
d = inp0.compute_points(["R", "Z", "e^rho", "sqrt(g)", "B"], bnf.rtz)
ze = bnf.rtz[:, 2]
R, Z = np.asarray(d["R"]), np.asarray(d["Z"])
er = np.asarray(d["e^rho"])  # cylindrical (R, phi, Z)
nh = er / np.linalg.norm(er, axis=1, keepdims=True)


def cart(vR, vp, vZ, phi):
    return np.column_stack([vR * np.cos(phi) - vp * np.sin(phi), vR * np.sin(phi) + vp * np.cos(phi), vZ])


x_b = cart(R, np.zeros_like(R), Z, ze)
n_c = cart(nh[:, 0], nh[:, 1], nh[:, 2], ze)
Bb = np.asarray(d["B"])
Bmag_b = np.linalg.norm(Bb, axis=1)
rmsB_b = float(np.sqrt(np.mean(Bmag_b**2)))
area_el = np.abs(np.asarray(d["sqrt(g)"])) * np.linalg.norm(er, axis=1)  # |e_theta x e_zeta|
dth, dze = 2 * np.pi / len(np.unique(bnf.rtz[:, 1])), (np.pi if fr.sym else 2 * np.pi) / eq0.NFP / len(np.unique(ze))
area = float(np.sum(area_el) * dth * dze * per)
a_min = float(inp0.a)
R0 = float(np.mean(R))
far = np.array([[R0, 0.0, 3 * a_min], [R0, 0.0, -3 * a_min], [0.0, 0.0, 0.0], [0.0, 0.0, 2 * a_min],
                [3 * R0, 0.0, 0.0], [0.0, 3 * R0, 0.0]])
targets = {"LCFS": x_b, **{f"shell {s:g}a": x_b + s * a_min * n_c for s in a.shells}, "far": far}
res_all["geometry"] = {"LCFS points": int(len(x_b)), "area [m^2]": area, "a [m]": a_min, "R0 [m]": R0,
                       "rms |B| on LCFS [T]": rmsB_b, "W_ref [J]": A.W_ref_J}
log(f"targets: LCFS {len(x_b)} points (area {area:.3f} m^2), shells {a.shells} a, {len(far)} far points")
disp = Displacement(eq0, 2, 2)
xi0 = jnp.zeros(disp.n)
df = DirectField(cur0, disp, (len(np.unique(bnf.rtz[:, 1])), len(np.unique(ze))), a.direct[0], a.direct[1],
                 bool(fr.sym), dtype=jnp.float64, chunk=64)
log(f"direct quadrature: {df.n_sources} sources (n_rho {a.direct[0]}, q {a.direct[1]})")
tj = {k: jnp.asarray(v) for k, v in targets.items()}


def measure(name, du):
    du = jnp.asarray(du)
    t1 = time.time()
    ss = A.state(u0 + du)
    out = {"|J-J_fit|/rms": float(jnp.linalg.norm(ss["gJ"])), "dW/W (Omega)": A.energy_parts(ss)["dW/W"]}
    bm = np.asarray(bnf.modes_of(fields, du))
    out["annulus B.n modes max|b|"] = float(np.max(np.abs(bm)))
    if bnd is not None:
        out["constraint (direct-row) modes max|b|"] = float(np.max(np.abs(np.asarray(Ct.T @ np.asarray(du)))))
    f_ann = np.asarray(bnf.f(fields, du))
    dB = {k: np.asarray(df(du, 0.0, xi0, v)) for k, v in tj.items()}
    Bn_dir = np.sum(dB["LCFS"] * n_c, 1)
    out["true B.n: rms(n.dB_direct) / rms|B|"] = float(np.sqrt(np.mean(Bn_dir**2)) / rmsB_b)
    # the annulus' own B.n on the same points (sqrt(g) dB.e^rho / (sqrt(g) |e^rho|) = n.dB)
    Bn_ann = f_ann / (np.abs(np.asarray(d["sqrt(g)"])) * np.linalg.norm(er, axis=1))
    out["annulus B.n: rms(n.dB) / rms|B|"] = float(np.sqrt(np.mean(Bn_ann**2)) / rmsB_b)
    out["annulus - direct B.n: rms / rms|B|"] = float(np.sqrt(np.mean((Bn_ann - Bn_dir) ** 2)) / rmsB_b)
    prof = {0.0: np.linalg.norm(dB["LCFS"], axis=1)}
    for s in a.shells:
        prof[s] = np.linalg.norm(dB[f"shell {s:g}a"], axis=1)
        out[f"rms|dB| shell {s:g}a / rms|B|"] = float(np.sqrt(np.mean(prof[s] ** 2)) / rmsB_b)
    out["rms|dB| LCFS (all components) / rms|B|"] = float(np.sqrt(np.mean(prof[0.0] ** 2)) / rmsB_b)
    out["max|dB| far / rms|B|"] = float(np.max(np.linalg.norm(dB["far"], axis=1)) / rmsB_b)
    # exterior energy bounds over the layer d in [0, max shell] (trapezoid in d, LCFS area element; outer
    # shells only grow the area on the convex side, so this is an estimate, not a strict bound)
    ds = np.array([0.0] + list(a.shells)) * a_min
    lin = [float(np.sum(area_el * Bmag_b * prof[s]) * dth * dze * per) / MU0 for s in [0.0] + list(a.shells)]
    quad = [float(np.sum(area_el * prof[s] ** 2) * dth * dze * per) / (2 * MU0) for s in [0.0] + list(a.shells)]
    out["exterior linear-term bound / W"] = trap(lin, ds) / A.W_ref_J
    out["exterior quadratic term / W"] = trap(quad, ds) / A.W_ref_J
    out["time [s]"] = time.time() - t1
    log(f"[{name}] " + ", ".join(f"{k} {v:.3e}" for k, v in out.items()))
    return out


res_all["cases"] = {}
for j in range(a.n_random):
    du_r = jnp.asarray(rng.standard_normal(n_u))
    g1 = float(jnp.linalg.norm(A.state(u0 + 1e-6 * du_r)["gJ"])) / 1e-6
    du_r = du_r * (1e-2 / g1)
    du_c = LC.restore(u0 + du_r, L) - u0
    gc = float(jnp.linalg.norm(A.state(u0 + du_c)["gJ"]))
    du_c = du_c * (1e-2 / gc)
    res_all["cases"][f"random {j} unconstrained"] = measure(f"random {j} unconstrained", du_r)
    res_all["cases"][f"random {j} constrained"] = measure(f"random {j} constrained", du_c)
    with open(a.out + "_g1.json", "w") as fh:
        json.dump(res_all, fh, indent=1, default=float)
for spec in a.state:
    nm, pth = spec.split("=", 1)
    u = jnp.asarray(np.load(pth)["u"])
    res_all["cases"][nm] = measure(nm, u - u0)
    res_all["cases"][nm]["violation"] = LC.violation(u)
for spec in a.pair:
    nm, pths = spec.split("=", 1)
    pa, pb = pths.split(",")
    du = jnp.asarray(np.load(pb)["u"]) - jnp.asarray(np.load(pa)["u"])
    res_all["cases"][nm] = measure(nm, du)
with open(a.out + "_g1.json", "w") as fh:
    json.dump(res_all, fh, indent=1, default=float)
log(f"saved {a.out}_g1.json")
