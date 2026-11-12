"""Islands in the NEST start of a fixed-boundary DESC equilibrium without coils (two external fields).

The external field is manufactured from DESC's field (no coils):
  A: B_ext = B_DESC - B_ann[J_fit]   (the hand-off's --bext desc). Start (u = 0, J = J_fit): B = B_DESC wherever
     B_ext is evaluated with the same formula, so the start has no islands by construction; checked at the points.
  B: B_ext = B_DESC - B_ann[J_DESC]  (vacuum up to the annulus error). Start: B = B_DESC - B_ann[J_DESC - J_fit],
     so the start islands measure what the NEST basis does not represent of DESC's current (its shielding layer).
B_ann[J] is NEST's own annulus Biot-Savart (the solver's operator): J at the annulus' inside nodes (their flux
coordinates from the locate step), cylindrical modes, ``CylindricalInterpolator`` at any point. DESC's J is
evaluated at those nodes directly.

Perturbations of DESC's nested field analysed (delta B = B - B_DESC), all in DESC's flux coordinates:
  start_B : -B_ann[J_DESC - J_fit]
  vacuum  : -B_ann[J_DESC - <J_DESC>_phi]  (DESC without its non-axisymmetric current: the (2,1) drive that
            DESC's shielding current cancels; scale for the widths)
  ann_fit : B_ann[J_fit] - B_DESC at the points only (the annulus + fit error that option A puts into B_ext)
For each: b^rho = B.grad(rho) / B.grad(zeta) on a (rho, theta, zeta) tensor grid, its (m, k) harmonic in PEST
angles theta* = theta + lambda (weight d theta*/d theta) by radius, the pendulum width w = 4 sqrt(|b_mk| / (m |iota'|))
at the resonance, and Poincare sections of B_DESC + delta B traced in (rho, theta) with zeta as time (RK4 on cubic
splines), recorded in (rho, theta*) at zeta = 0.

usage: python scripts/meanforce/start_check_bext.py --eq <ripple .h5> --LMN 16 16 4 --frac 20 --N-phi 17 \
           --out out/startchk_L16
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
from cylindrical_grid import CylindricalInterpolator  # noqa: E402
from desc_input import DESCInput  # noqa: E402
from flux_interp import TensorCurrent, TensorGrid  # noqa: E402

from force import ForceResidual  # noqa: E402
from free_boundary import reference  # noqa: E402
from objective import VECTOR_PARITY, Fields, _cartesian  # noqa: E402
from optimize_fb import annulus  # noqa: E402
from pressure import Pressure  # noqa: E402

T0 = time.time()


def log(*s):
    print(f"[{time.time() - T0:8.1f} s]", *s, flush=True)


ap = argparse.ArgumentParser()
ap.add_argument("--eq", required=True)
ap.add_argument("--LMN", type=int, nargs=3, required=True)
ap.add_argument("--colloc", type=float, default=1.5)
ap.add_argument("--frac", type=float, default=20)
ap.add_argument("--pad", type=float, default=0.02)
ap.add_argument("--N-phi", type=int, default=17, help="annulus planes per period (odd)")
ap.add_argument("--m", type=int, default=2)
ap.add_argument("--k", type=int, default=1, help="resonant helicity cos(m theta* + k zeta)")
ap.add_argument("--rho-grid", type=float, nargs=3, default=(0.30, 0.95, 131))
ap.add_argument("--ntheta", type=int, default=128)
ap.add_argument("--nzeta", type=int, default=32)
ap.add_argument("--lines", type=int, default=36)
ap.add_argument("--seed-band", type=float, nargs=2, default=(0.45, 0.85))
ap.add_argument("--seed-theta", type=float, default=0.0, help="DESC theta of the seeds")
ap.add_argument("--turns", type=int, default=300)
ap.add_argument("--steps", type=int, default=64, help="RK4 steps per toroidal turn")
ap.add_argument("--state", action="append", default=[],
                help="npz with u (mf_fixed state); analysed as delta B = B_ann[J(u) - J_fit] (bext A) and "
                     "B_ann[J(u) - J_DESC] (bext B). Same --LMN, --frac, --N-phi, --pad as the run.")
ap.add_argument("--perts", nargs="+", default=None,
                help="which perturbations to analyse (default: start_B, vacuum and every state's A and B)")
ap.add_argument("--band", type=float, nargs="+", default=[0.04, 0.08, 0.16, 0.3], help="half-widths in rho of the layers for the shielding ratio")
ap.add_argument("--pressure", action="store_true",
                help="pressure diagnostics: <p>(rho), p', |p_mk|, rms B.grad p / |B||grad p| (DESC, the fit, each "
                     "state's least-squares p from its c_p) and p along the Poincare lines (island flattening)")
ap.add_argument("--out", required=True)
a = ap.parse_args()
os.makedirs(os.path.dirname(a.out) or ".", exist_ok=True)
rep = {"args": vars(a)}


def save():
    with open(a.out + ".json", "w") as f:
        json.dump(rep, f, indent=1, default=float)


# ------------------------------------------------------------------ setup (as mf_fixed.py)
inp = DESCInput.load(a.eq, field=None)
eq = inp.eq
kw = {"L": a.LMN[0], "M": a.LMN[1], "N": a.LMN[2]}
cur = fit_cache.fit(BoundaryPotential, inp, **kw)
pre = fit_cache.fit(Pressure, inp, edge_order=2, **kw)
b = cur.basis
res = tuple(int(round(a.colloc * (u + v))) for u, v in ((b.L, eq.L), (b.M, eq.M), (b.N, eq.N)))
ref = reference(inp, res, current=cur)
log(f"{inp}\nbasis {b.L}/{b.M}/{b.N}, n_u {cur.n_free}, collocation {res}")
box = annulus(inp, inp.a / a.frac, a.pad, a.N_phi)
box.locate_plasma_planes(inp, sym=bool(eq.sym), verbose=False)
fr = ForceResidual(inp, box, resolution=res)
fr.norm, fr.F_desc_vol = ref["norm"], ref["F_desc_vol"]
fr.B_ext = np.zeros((len(fr.x), 3))  # replaced below (per option)
fields = Fields(fr, cur, pre, dtype=jnp.float64, cyl_stencil=True)
rep["setup"] = {"basis": [b.L, b.M, b.N], "n_u": int(cur.n_free), "collocation": list(res), "annulus": str(box),
                "a/h": a.frac, "N_phi": int(box.N_phi), "rows": int(len(fields.rows_rtz))}
log(f"annulus {box}; inside rows {len(fields.rows_rtz)}")

u0 = jnp.asarray(cur.u)
solver = fields.solver


def modes_rows(J_rows):
    """Modal slabs of B_ann for J (cylindrical components) at the interpolated inside nodes."""
    return solver.field_modes_F(*fields._cyl_sources(fields._cyl_rows(jnp.asarray(J_rows))))


def modes_box(J_box):
    return solver.field_modes_F(*solver.sources(jnp.asarray(J_box)))


def at_points(modes):
    return np.asarray(_cartesian(fields.to_points(*modes), fields.phi))


# J at the inside nodes: the fit (the solver's path) and DESC's own
J_fit_rows = np.asarray(fields.to_box(fields.J_flux(u0, fields.I).astype(fields.cyl_dtype), VECTOR_PARITY))
J_D_rows = np.asarray(inp.compute_points(["J"], np.asarray(fields.rows_rtz))["J"])
rr = np.asarray(fields.rows_rtz)[:, 0]
dJ = J_D_rows - J_fit_rows
bands = [(0, 0.3), (0.3, 0.55), (0.55, 0.72), (0.72, 0.9), (0.9, 1.0)]
rmsD = np.sqrt(np.mean(np.sum(J_D_rows**2, 1)))
rep["fit"] = {"rms |J_DESC - J_fit| / rms |J_DESC| (annulus nodes)": float(np.sqrt(np.mean(np.sum(dJ**2, 1))) / rmsD),
              "by rho band": {f"{lo}-{hi}": float(np.sqrt(np.mean(np.sum(dJ[(rr >= lo) & (rr < hi)] ** 2, 1))) / rmsD)
                              for lo, hi in bands}}
log("fit:", json.dumps(rep["fit"]))

m_fit = modes_rows(J_fit_rows)
m_D = modes_rows(J_D_rows)
B_D = np.asarray(fields.desc["B"])
B_fit = at_points(m_fit)
B_annD = at_points(m_D)
Bn = np.sqrt(np.mean(np.sum(B_D**2, 1)))
ext = {"A": B_D - B_fit, "B": B_D - B_annD}
w = fr.dV / fr.dV.sum()
rho_c = np.asarray(fr.rtz)[:, 0]
rep["start at the points"] = {}
for k_, Be in ext.items():
    dB = Be + B_fit - B_D
    rep["start at the points"][k_] = {
        "rms |B_start - B_DESC| / rms|B|": float(np.sqrt(np.sum(w * np.sum(dB**2, 1))) / Bn),
        "max": float(np.max(np.linalg.norm(dB, axis=1)) / Bn)}
log("start at the points:", json.dumps(rep["start at the points"]))
np.savez(a.out + "_bext.npz", B_ext_A=ext["A"], B_ext_B=ext["B"], B_DESC=B_D, x=np.asarray(fr.x),
         rtz=np.asarray(fr.rtz), u0=np.asarray(u0))
save()

# ------------------------------------------------------------------ perturbations on a flux tensor grid
rho = np.linspace(*a.rho_grid[:2], int(a.rho_grid[2]))
th = 2 * np.pi * np.arange(a.ntheta) / a.ntheta
NFP = eq.NFP
ze = 2 * np.pi * np.arange(a.nzeta) / (a.nzeta * NFP)
names = ["R", "Z", "e^rho", "e^theta", "e^zeta", "B^theta", "B^zeta", "lambda", "lambda_t", "iota", "J", "B"]
d = inp.compute_tensor(names, rho, th, ze)
shape = (len(rho), len(th), len(ze))
pts = np.column_stack([d["R"], np.broadcast_to(ze[None, None, :], shape).ravel(), d["Z"]])
itp = CylindricalInterpolator(fields.cyl_grid, pts, dtype=jnp.float64)
iota = d["iota"].reshape(shape)[:, 0, 0]
iota_r = np.gradient(iota, rho)
target = -a.k / a.m  # iota at the (m, k) resonance
i0 = np.flatnonzero(np.sign(iota[:-1] - target) != np.sign(iota[1:] - target))
rho_r = float(np.interp(0.0, [iota[i0[0]] - target, iota[i0[0] + 1] - target], rho[i0[0]:i0[0] + 2])) if len(i0) else None
iop_r = float(np.interp(rho_r, rho, iota_r)) if rho_r else None
rep["resonance"] = {"rho": rho_r, "iota'": iop_r}
log(f"resonance iota = {target}: rho {rho_r}, iota' {iop_r}")

# the helical part of J_DESC on the annulus (DESC with its non-axisymmetric current removed)
Jbox_D = np.asarray(fields._cyl_rows(jnp.asarray(J_D_rows)))[np.asarray(fields.src)].reshape(fields.shape + (3,))
J_hel = Jbox_D - Jbox_D.mean(axis=0, keepdims=True)
pert_modes = {"start_B": jax.tree.map(lambda x, y: -(x - y), m_D, m_fit),
              "vacuum": jax.tree.map(lambda x: -x, modes_box(J_hel))}
states_u, state_paths = {}, {}
for sp in a.state:
    tag = os.path.splitext(os.path.basename(sp))[0]
    u = jnp.asarray(np.load(sp)["u"])
    states_u[tag] = u
    state_paths[tag] = sp
    m_u = fields.cyl_field_modes(u)
    pert_modes[f"{tag}:A"] = jax.tree.map(lambda x, y: x - y, m_u, m_fit)
    pert_modes[f"{tag}:B"] = jax.tree.map(lambda x, y: x - y, m_u, m_D)
    # misfit of J(u) to J_fit at the annulus nodes (what the run's objective sees, roughly)
    J_u_rows = np.asarray(fields.to_box(fields.J_flux(u, fields.I).astype(fields.cyl_dtype), VECTOR_PARITY))
    rep.setdefault("states", {})[tag] = {
        "rms |J(u) - J_fit| / rms |J_DESC| (annulus nodes)": float(np.sqrt(np.mean(np.sum((J_u_rows - J_fit_rows) ** 2, 1))) / rmsD)}
if a.perts:
    pert_modes = {k: v for k, v in pert_modes.items() if k in a.perts}
tp = (np.broadcast_to(th[None, :, None], shape).ravel() + d["lambda"])  # theta*
tpt = 1.0 + d["lambda_t"]
zz = np.broadcast_to(ze[None, None, :], shape).ravel()
ph = np.exp(-1j * (a.m * tp + a.k * NFP * zz)) * tpt
rep["perturbations"] = {}
tables, dB_grid, b21_at = {}, {}, {}
for name, md in pert_modes.items():
    dBc = np.asarray(itp(*md))  # cylindrical (R, phi, Z) components, as DESC's vectors
    Bz = d["B^zeta"] + np.sum(dBc * d["e^zeta"], 1)
    brho = np.sum(dBc * d["e^rho"], 1) / Bz
    bth = (d["B^theta"] + np.sum(dBc * d["e^theta"], 1)) / Bz
    c = 2 * (brho * ph).reshape(shape).mean(axis=(1, 2))
    amp = np.abs(c)
    b_r = float(np.interp(rho_r, rho, amp)) if rho_r else None
    wid = float(4 * np.sqrt(b_r / (a.m * abs(iop_r)))) if rho_r else None
    rep["perturbations"][name] = {"|b_mk|(rho)": amp.tolist(), "arg b_mk(rho)": np.angle(c).tolist(),
                                  "|b_mk| at resonance": b_r, "pendulum width": wid,
                                  "max |b^rho|": float(np.abs(brho).max()),
                                  "rms |dB| / rms |B| on grid": float(np.sqrt(np.mean(np.sum(dBc**2, 1))) / Bn)}
    tables[name] = (brho.reshape(shape), bth.reshape(shape))
    dB_grid[name] = dBc
    b21_at[name] = complex(np.interp(rho_r, rho, c.real) + 1j * np.interp(rho_r, rho, c.imag))
    log(f"{name}: |b_{a.m}{a.k}| at q res {b_r:.3e}, pendulum width {wid:.4f}; max |b^rho| {np.abs(brho).max():.2e}")
rep["rho grid"] = rho.tolist()
save()

# ------------------------------------------------------------------ evidence for shielding
# (1) lambda = J.B / |B|^2: its (m, k) harmonic in PEST angles by radius, for DESC, J_fit and the states
tc = TensorCurrent(cur.S, cur._RZ, TensorGrid(rho, th, ze, NFP))
B_Dt = d["B"]
cur_cases = {"DESC": (d["J"], B_Dt), "J_fit": (np.asarray(tc(u0, fields.I)).reshape(-1, 3), B_Dt)}
for tag, u in states_u.items():
    Bt = B_Dt + dB_grid.get(f"{tag}:A", 0.0)  # option A: B = B_DESC + B_ann[J(u) - J_fit]
    cur_cases[tag] = (np.asarray(tc(u, fields.I)).reshape(-1, 3), Bt)
rep["J_parallel"] = {}
for name, (Jt, Bt) in cur_cases.items():
    lam_ = np.sum(Jt * Bt, 1) / np.sum(Bt * Bt, 1)
    c = 2 * (lam_ * ph).reshape(shape).mean(axis=(1, 2))
    l00 = (lam_ * tpt).reshape(shape).mean(axis=(1, 2))
    amp = np.abs(c)
    lm = np.flatnonzero((amp[1:-1] >= amp[:-2]) & (amp[1:-1] >= amp[2:])) + 1  # local maxima
    ip = lm[np.argmin(np.abs(rho[lm] - rho_r))] if lm.size else int(np.argmax(amp))
    hm = amp >= amp[ip] / 2
    lo = hi = ip
    while lo > 0 and hm[lo - 1]:
        lo -= 1
    while hi < len(rho) - 1 and hm[hi + 1]:
        hi += 1
    rep["J_parallel"][name] = {"|lambda_mk|(rho)": amp.tolist(), "arg lambda_mk(rho)": np.angle(c).tolist(),
                               "lambda_00(rho)": l00.tolist(), "|lambda_mk| at resonance": float(np.interp(rho_r, rho, amp)),
                               "local max nearest resonance": {"rho": float(rho[ip]), "|lambda_mk|": float(amp[ip]),
                                                       "half-max band": [float(rho[lo]), float(rho[hi])]}}
    log(f"J_par {name}: |lambda_21| at q res {np.interp(rho_r, rho, amp):.4e}; peak {amp[ip]:.4e} at rho {rho[ip]:.3f}, "
        f"half-max band [{rho[lo]:.3f}, {rho[hi]:.3f}]")

# (1b) pressure: DESC's p, the fit of it (the start's p) and each state's least-squares p (its c_p) on the flux grid.
# A plateau on nested surfaces: <p>' -> 0 near rho_res with |p_mk| and B.grad p staying small; an island with
# flattening: |p_mk| peaks at rho_res (p follows the island's surfaces) and p is constant along the lines inside it.
p_tables = {}
if a.pressure:
    from types import SimpleNamespace
    dP = inp.compute_tensor(["p", "grad(p)"], rho, th, ze)
    nodes_ = np.stack(np.meshgrid(rho, th, ze, indexing="ij"), -1).reshape(-1, 3)
    pf = np.asarray(pre.evaluate(SimpleNamespace(nodes=nodes_)))
    p_cases = {"DESC": (np.asarray(dP["p"]).ravel(), np.asarray(dP["grad(p)"]).reshape(-1, 3), B_Dt),
               "p_fit": (pf[:, 0], pf[:, 1:4], B_Dt)}
    for tag, sp in state_paths.items():
        pg = np.asarray(pre.with_c(jnp.asarray(np.load(sp)["c_p"])).evaluate(SimpleNamespace(nodes=nodes_)))
        p_cases[tag] = (pg[:, 0], pg[:, 1:4], B_Dt + dB_grid.get(f"{tag}:A", 0.0))
    rep["pressure"] = {}
    win = np.abs(rho - rho_r) <= 0.06
    for name, (p_, gp, Bt) in p_cases.items():
        pav = (p_ * tpt).reshape(shape).mean(axis=(1, 2))
        dpav = np.gradient(pav, rho)
        c = 2 * (p_ * ph).reshape(shape).mean(axis=(1, 2))
        bgp = np.sum(Bt * gp, 1) / (np.linalg.norm(Bt, axis=1) * np.linalg.norm(gp, axis=1) + 1e-300)
        bgp_s = np.sqrt(np.mean(bgp.reshape(shape) ** 2, axis=(1, 2)))
        if name in states_u or name == "DESC":
            p_tables[name] = p_.reshape(shape)
        rep["pressure"][name] = {"<p>(rho)": pav.tolist(), "d<p>/drho": dpav.tolist(), "|p_mk|(rho)": np.abs(c).tolist(),
                                 "rms B.grad p/|B||grad p|(rho)": bgp_s.tolist(),
                                 "at resonance": {"<p>": float(np.interp(rho_r, rho, pav)),
                                                  "d<p>/drho": float(np.interp(rho_r, rho, dpav)),
                                                  "|p_mk|": float(np.interp(rho_r, rho, np.abs(c))),
                                                  "rms B.grad p": float(np.interp(rho_r, rho, bgp_s))}}
    d0 = np.asarray(rep["pressure"]["DESC"]["d<p>/drho"])
    pk = np.round(np.arange(rho_r - 0.05, rho_r + 0.0501, 0.01), 3)
    log(f"[p] d<p>/drho / DESC's at rho_res +- 0.05 (rho {pk[0]:.3f} .. {pk[-1]:.3f}, step 0.01); DESC p' at rho_res "
        f"{np.interp(rho_r, rho, d0):.4e}")
    for name, v in rep["pressure"].items():
        dd = np.asarray(v["d<p>/drho"])
        rat = np.interp(pk, rho, dd) / np.interp(pk, rho, d0)
        v["p'/p'_DESC at rho_res +- 0.05"] = rat.tolist()
        ar = v["at resonance"]
        ir = win & (np.abs(d0) > 0.1 * np.abs(d0[win]).max())
        v["min |p'/p'_DESC| within 0.06"] = float(np.min(np.abs(dd[ir] / d0[ir]))) if ir.any() else None
        log(f"[p] {name:24s} p'/p'_D " + " ".join(f"{x:+.3f}" for x in rat)
            + f" | at res: <p> {ar['<p>']:.2f} Pa, p' {ar['d<p>/drho']:.3e}, |p_mk| {ar['|p_mk|']:.3e} Pa, "
              f"rms B.grad p {ar['rms B.grad p']:.2e}")
    save()

# (2) shielding ratio: resonant field at q = 2 of the helical current (J minus its toroidal average at fixed (r, z)),
# whole and restricted to |rho - rho_res| < band, over the vacuum drive (-1: the current cancels the drive)
rho_rows = np.asarray(fields.rows_rtz)[:, 0]
rho_ext = rho_rows if fields.half is None else np.concatenate([rho_rows, rho_rows[np.asarray(fields.half_rows)]])
rho_box = np.concatenate([rho_ext, [np.nan]])[np.asarray(fields.src)].reshape(fields.shape)
dist_box = np.abs(rho_box - rho_r)  # nan outside the plasma
rows_cases = {"DESC": J_D_rows, "J_fit": J_fit_rows}
for tag, u in states_u.items():
    rows_cases[tag] = np.asarray(fields.to_box(fields.J_flux(u, fields.I).astype(fields.cyl_dtype), VECTOR_PARITY))
b_vac = b21_at["vacuum"] if "vacuum" in b21_at else None


def b21_of(md):
    dBc = np.asarray(itp(*md))
    Bz = d["B^zeta"] + np.sum(dBc * d["e^zeta"], 1)
    br = np.sum(dBc * d["e^rho"], 1) / Bz
    c = 2 * (br * ph).reshape(shape).mean(axis=(1, 2))
    return complex(np.interp(rho_r, rho, c.real) + 1j * np.interp(rho_r, rho, c.imag))


rep["shielding"] = {"band": a.band, "b21 vacuum": [b_vac.real, b_vac.imag] if b_vac is not None else None}
for name, Jr in rows_cases.items():
    Jb = np.asarray(fields._cyl_rows(jnp.asarray(Jr)))[np.asarray(fields.src)].reshape(fields.shape + (3,))
    Jh = Jb - Jb.mean(axis=0, keepdims=True)
    out_ = {}
    parts = [("all helical", Jh)] + [(f"helical, |rho - rho_res| < {w}", np.where((dist_box < w)[..., None], Jh, 0.0))
                                     for w in a.band]
    for part, Jx in parts:
        bb = b21_of(modes_box(Jx))
        out_[part] = {"b21": [bb.real, bb.imag], "ratio to vacuum": ([(bb / b_vac).real, (bb / b_vac).imag]
                                                                      if b_vac else None)}
    rep["shielding"][name] = out_
    log(f"shielding {name}: " + "; ".join(f"{k}: b21/b21_vac = {complex(*v['ratio to vacuum']).real:+.4f}"
                                          for k, v in out_.items() if v["ratio to vacuum"]))
save()

# ------------------------------------------------------------------ Poincare sections in (rho, theta*)
from scipy.ndimage import map_coordinates, spline_filter  # noqa: E402

lam = d["lambda"].reshape(shape)
P = 3  # periodic padding (theta, zeta)


def prep(f):
    g = np.concatenate([f[:, -P:], f, f[:, :P]], axis=1)
    g = np.concatenate([g[:, :, -P:], g, g[:, :, :P]], axis=2)
    return spline_filter(g, order=3, mode="nearest")


dr = rho[1] - rho[0]
dth = th[1] - th[0]
dze = ze[1] - ze[0] if len(ze) > 1 else 2 * np.pi / NFP


def sample(F, r, t, z):
    z = np.broadcast_to(z, np.shape(r))
    q = np.stack([(r - rho[0]) / dr, np.mod(t, 2 * np.pi) / dth + P, np.mod(z, 2 * np.pi / NFP) / dze + P])
    return map_coordinates(F, q, order=3, prefilter=False, mode="nearest")


Flam = prep(lam)
seeds = np.linspace(*a.seed_band, a.lines)
for name, (brho, bth) in tables.items():
    Fr, Ft = prep(brho), prep(bth)
    ptab = p_tables.get(name.split(":")[0], p_tables.get("DESC"))  # the state's own p; DESC's for vacuum/start_B
    Fp = prep(ptab) if ptab is not None else None
    out_p = []
    r, t = seeds.copy(), np.full_like(seeds, a.seed_theta)
    h = 2 * np.pi / (NFP * a.steps)
    out_r, out_t = [], []
    alive = np.ones_like(r, bool)

    def rhs(r, t, z):
        return sample(Fr, r, t, z), sample(Ft, r, t, z)

    z = 0.0
    for turn in range(a.turns):
        for _ in range(a.steps):
            k1 = rhs(r, t, z)
            k2 = rhs(r + h / 2 * k1[0], t + h / 2 * k1[1], z + h / 2)
            k3 = rhs(r + h / 2 * k2[0], t + h / 2 * k2[1], z + h / 2)
            k4 = rhs(r + h * k3[0], t + h * k3[1], z + h)
            r = r + h / 6 * (k1[0] + 2 * k2[0] + 2 * k3[0] + k4[0])
            t = t + h / 6 * (k1[1] + 2 * k2[1] + 2 * k3[1] + k4[1])
            z += h
        alive &= (r > rho[0]) & (r < rho[-1])
        tstar = np.mod(t + sample(Flam, r, t, np.zeros_like(r)), 2 * np.pi)
        out_r.append(np.where(alive, r, np.nan))
        out_t.append(np.where(alive, tstar, np.nan))
        if Fp is not None:
            out_p.append(np.where(alive, sample(Fp, r, t, np.zeros_like(r)), np.nan))
        if turn % 50 == 0:
            log(f"{name}: turn {turn}, alive {alive.sum()}/{len(alive)}")
    np.savez(a.out + f"_poincare_{name.replace(':', '_')}.npz", rho=np.array(out_r), theta_star=np.array(out_t), seeds=seeds,
             rho_res=rho_r)
    R_ = np.array(out_r)
    spread = np.nanmax(R_, 0) - np.nanmin(R_, 0)
    rep["perturbations"][name]["poincare radial spread per line"] = spread.tolist()
    rep["perturbations"][name]["max radial spread"] = float(np.nanmax(spread))
    log(f"{name}: Poincare done; max radial spread of a line {np.nanmax(spread):.4f}")
    if Fp is not None:  # flattening: p spread along a line vs DESC's <p> drop over the line's radial range
        P_ = np.array(out_p)
        pD = np.asarray(rep["pressure"]["DESC"]["<p>(rho)"])
        sp_p = np.nanmax(P_, 0) - np.nanmin(P_, 0)
        drop = np.abs(np.interp(np.nanmax(R_, 0), rho, pD) - np.interp(np.nanmin(R_, 0), rho, pD))
        ratio = sp_p / np.maximum(drop, 1e-300)
        sel = spread > 0.002
        rep["perturbations"][name]["p spread along line [Pa]"] = sp_p.tolist()
        rep["perturbations"][name]["DESC p drop over line's rho range [Pa]"] = drop.tolist()
        rep["perturbations"][name]["along-line / DESC p ratio"] = ratio.tolist()
        np.savez(a.out + f"_poincare_p_{name.replace(':', '_')}.npz", p=P_)
        log(f"{name}: along-line p spread / DESC <p> drop over the line's rho range (lines with rho spread > 0.002: "
            f"{int(sel.sum())}): " + (" ".join(f"{seeds[i]:.3f}:{spread[i]:.4f}:{ratio[i]:.2f}" for i in np.flatnonzero(sel))
                                    if sel.any() else "none") + " [seed:rho spread:ratio]")
    save()
log("done")
