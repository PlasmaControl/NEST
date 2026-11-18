"""G0 correctness checks for the energy objective and the LCFS B.n constraint (CPU, tiny basis).

Builds the same objects as mf_fixed.py (--bext desc-fit), then:
  G0.1  adjoints: <J v, w> = <v, J^T w> for the rows with the energy block; <C v, y> = <v, C^T y> for the
        B.n modes; and the LCFS interpolator's component order (rebuilt on the collocation points it must
        reproduce fields.B_plasma).
  G0.2  gradient: J^T r (the solver's gradient) vs central differences of the full merit, energy on.
  G0.3  E(u + t d) is exactly quadratic: three points predict a fourth.
  G0.4  the merit's cross-term energy equals (1/2)|g_E|^2 - (1/2)|g_ref|^2, and stays accurate at tiny du.
  G0.5  projected steps keep C du = 0; restore() lands on C du = 0.
  G0.7  (first half) the closed-form minimizer of (1/2)|g_J|^2 + E subject to the B.n and flux rows (no force
        bound), saved for comparison with mf_fixed runs (``--compare run_a*.npz``).

usage: python scripts/energy/g0_checks.py --eq EQ --LMN 8 8 2 --frac 20 --N-phi 13 --lam 30 --bn-modes 8 2 \\
           --out out/g0 [--compare out/run1_a1000000.0.npz ...]
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
from cylindrical_grid import CylindricalInterpolator  # noqa: E402
from desc_input import DESCInput  # noqa: E402
from force import ForceResidual  # noqa: E402
from free_boundary import reference  # noqa: E402
from objective import Fields, _cartesian  # noqa: E402
from optimize_fb import annulus  # noqa: E402
from pressure import Pressure  # noqa: E402

T0 = time.time()


def log(*s):
    print(f"[{time.time() - T0:7.1f} s]", *s, flush=True)


ap = argparse.ArgumentParser()
ap.add_argument("--eq", required=True)
ap.add_argument("--LMN", type=int, nargs=3, required=True)
ap.add_argument("--frac", type=float, default=20)
ap.add_argument("--N-phi", type=int, default=13)
ap.add_argument("--pad", type=float, default=0.02)
ap.add_argument("--colloc", type=float, default=1.5)
ap.add_argument("--lam", type=float, default=30.0)
ap.add_argument("--bn-modes", type=int, nargs=2, required=True)
ap.add_argument("--out", required=True)
ap.add_argument("--compare", nargs="*", default=[])
ap.add_argument("--skip-closed", action="store_true")
a = ap.parse_args()
os.makedirs(os.path.dirname(a.out) or ".", exist_ok=True)
res_all = {}
rng = np.random.default_rng(0)

# ------------------------------------------------------------------ setup (as mf_fixed, --bext desc-fit)
inp0 = DESCInput.load(a.eq)
eq0 = inp0.eq
cur0 = fit_cache.fit(BoundaryPotential, inp0, L=a.LMN[0], M=a.LMN[1], N=a.LMN[2])
pre0 = fit_cache.fit(Pressure, inp0, edge_order=2, L=a.LMN[0], M=a.LMN[1], N=a.LMN[2])
b = cur0.basis
res = tuple(int(round(a.colloc * (u + v))) for u, v in ((b.L, eq0.L), (b.M, eq0.M), (b.N, eq0.N)))
ref = reference(inp0, res, current=cur0)
norm = ref["norm"]
M_desc = ref["F_desc_vol"] / norm
box = annulus(inp0, inp0.a / a.frac, a.pad, a.N_phi)
box.locate_plasma_planes(inp0, sym=bool(eq0.sym), verbose=False)
fr = ForceResidual(inp0, box, resolution=res)
fr.norm, fr.F_desc_vol = norm, ref["F_desc_vol"]
fr.B_ext = np.zeros((len(fr.x), 3))
fields = Fields(fr, cur0, pre0, dtype=jnp.float64, cyl_stencil=True, mode_chunk=16, cyl_fused=True, cyl_rblock=32,
                cyl_order=8)  # exactly as mf_fixed.py (defaults --mode-chunk 16 --rblock 32 --interp-order 8)
G = mff.gram_P(fields, cur0, pre0, log=log)
u0 = jnp.asarray(cur0.u)
B_D = np.asarray(fields.desc["B"])
B_ext = B_D - np.asarray(fields.B_plasma(u0))
fields.B_ext = jnp.asarray(B_ext)
fr.B_ext = B_ext
Jfit = mff._J(fields, u0, fields.I)
sw = jnp.asarray(fields.sw)
rmsJ = float(jnp.sqrt(jnp.sum(sw[:, None] ** 2 * Jfit**2)))
A = mff.FixedAnchor(fields, cur0, pre0, u0, rmsJ, norm, G, log=log)
n_u = A.n_u
log(f"basis {a.LMN}, n_u {n_u}, points {A.n_pts}, collocation {res}, annulus {box}")
W = mf.Weights(0.05 * M_desc, 1.0, 1.0, rho=10.0)
per = 2 * eq0.NFP if fr.sym else eq0.NFP
vol = float(per * np.sum(fr.dV))
A.enable_energy(a.lam, W, vol, log=log)
V_desc = float(np.asarray(eq0.compute("V")["V"]))
W_desc = float(np.asarray(eq0.compute("W_B")["W_B"]))
res_all["setup"] = {"n_u": int(n_u), "points": int(A.n_pts), "volume": vol, "DESC volume": V_desc,
                    "W_ref [J]": A.W_ref_J, "DESC W_B [J]": W_desc, "W_ref/W_DESC - 1": A.W_ref_J / W_desc - 1.0}
log(f"[G1.1 preview] W_Omega(u0) {A.W_ref_J:.6e} J vs DESC W_B {W_desc:.6e} J (rel {A.W_ref_J / W_desc - 1:+.2e}); "
    f"volume {vol:.6f} vs DESC {V_desc:.6f}")

# a state away from u0 (random, scaled so |J - J_fit|/rms ~ 1e-2), with an active force term
d_rand = jnp.asarray(rng.standard_normal(n_u))
s_tmp = A.state(u0 + 1e-6 * d_rand)
scl = 1e-2 / (float(jnp.linalg.norm(s_tmp["gJ"])) / 1e-6)
z1 = u0 + scl * d_rand
s1 = A.state(z1)
W.eps = 0.5 * float(s1["M"])  # bound violated: the force rows are active
log(f"test state: |J-J_fit|/rms {float(jnp.linalg.norm(s1['gJ'])):.3e}, M/M_desc {float(s1['M']) / M_desc:.4f}, "
    f"M/eps {float(s1['M']) / W.eps:.3f}")

# ------------------------------------------------------------------ G0.1 adjoints and component order
out = {}
for cls in (mff.FixedRows, mff.FixedRowsB):
    rows = cls(A, s1, W, norm)
    v = jnp.asarray(rng.standard_normal(n_u))
    w = jnp.asarray(rng.standard_normal(rows.r.size))
    lhs, rhs = float(rows.J(v) @ w), float(v @ rows.JT(w))
    out[cls.__name__] = abs(lhs - rhs) / max(abs(lhs), abs(rhs))
    assert rows.ng == 6 * A.n_pts, rows.ng
bnf = mff.LCFSField(inp0, fields, fr, a.bn_modes[0], a.bn_modes[1], log=log)
Ct = bnf.rows(fields, n_u, log=log, every=0)
v = jnp.asarray(rng.standard_normal(n_u))
y = rng.standard_normal(bnf.k)
lhs, rhs = float(np.asarray(bnf.modes_of(fields, v)) @ y), float(np.asarray(v) @ (Ct @ y))
out["Bn modes"] = abs(lhs - rhs) / max(abs(lhs), abs(rhs))
# component order: an interpolator rebuilt on the collocation points must reproduce fields.B_plasma
x = np.asarray(fr.x)
rpz = np.column_stack([np.hypot(x[:, 0], x[:, 1]), np.asarray(fr.phi), x[:, 2]])
itp0 = getattr(fields.to_points, "interp", fields.to_points)
itp_c = CylindricalInterpolator(fields.cyl_grid, rpz, dtype=itp0.yr.dtype, order=getattr(fields, "cyl_order", None))
Bc = itp_c.from_planes(*fields.cyl_planes(v, 0.0))[: len(rpz)]
Bx = _cartesian(Bc, jnp.asarray(fr.phi))
Bref = fields.B_plasma(v, 0.0)
out["interp component order"] = float(jnp.max(jnp.abs(Bx - Bref)) / jnp.max(jnp.abs(Bref)))
# a DESC-field check on the LCFS points: sqrt(g) B_DESC . e^rho = 0 there
dD = inp0.compute_points(["B", "e^rho", "sqrt(g)", "B^zeta"], bnf.rtz)
bnD = np.asarray(dD["sqrt(g)"]) * np.sum(np.asarray(dD["B"]) * np.asarray(dD["e^rho"]), 1)
out["DESC B.n / scale (rms)"] = float(np.sqrt(np.mean(bnD**2)) / bnf.scale)
res_all["G0.1"] = out
log(f"[G0.1] {out}")

# ------------------------------------------------------------------ G0.2 gradient vs central differences
rows = mff.FixedRows(A, s1, W, norm)
gr = rows.grad()
out = []
for j in range(5):
    dd = jnp.asarray(rng.standard_normal(n_u))
    dd = dd * (scl * np.sqrt(n_u) / float(jnp.linalg.norm(dd)))  # same scale as the test displacement
    gd = float(gr @ dd)
    row = {}
    for h in (1e-2, 1e-3, 1e-4, 1e-5):
        fp, fm = W.merit(A.state(z1 + h * dd))[0], W.merit(A.state(z1 - h * dd))[0]
        row[f"{h:g}"] = abs((fp - fm) / (2 * h) - gd) / abs(gd)
    out.append(row)
res_all["G0.2"] = out
log(f"[G0.2] rel. error of J^T r . d vs central differences (h -> rel err): {out}")
# G0.2b: force bound inactive, so the merit is phi_J + E, exactly quadratic: central differences are exact
eps_keep = W.eps
W.eps = 1e6 * float(s1["M"])
rows = mff.FixedRows(A, s1, W, norm)
gr = rows.grad()
out = []
for j in range(5):
    dd = jnp.asarray(rng.standard_normal(n_u))
    dd = dd * (scl * np.sqrt(n_u) / float(jnp.linalg.norm(dd)))
    gd = float(gr @ dd)
    row = {}
    for h in (1.0, 1e-1, 1e-2):
        fp, fm = W.merit(A.state(z1 + h * dd))[0], W.merit(A.state(z1 - h * dd))[0]
        row[f"{h:g}"] = abs((fp - fm) / (2 * h) - gd) / abs(gd)
    out.append(row)
W.eps = eps_keep
res_all["G0.2b"] = out
log(f"[G0.2b] force inactive (phi_J + E, exactly quadratic): rel. error vs central differences: {out}")

# ------------------------------------------------------------------ G0.3 exact quadratic of E
dd = scl * jnp.asarray(rng.standard_normal(n_u))


def E_of(zz):
    return W.objective_parts(A.state(zz)["g"])[1]


ts = [0.0, 1.0, 2.0, 3.0]
Es = [E_of(z1 + t * dd) for t in ts]
c = np.polyfit(ts[:3], Es[:3], 2)
pred = np.polyval(c, ts[3])
res_all["G0.3"] = {"E values": Es, "4th-point error / E change": abs(pred - Es[3]) / abs(Es[3] - Es[0])}
log(f"[G0.3] {res_all['G0.3']}")

# ------------------------------------------------------------------ G0.4 cross-term form
out = {}
for amp in (1.0, 1e-4, 1e-8):
    ss = A.state(u0 + amp * scl * d_rand)
    gE = ss["g"][A.n_pts * 3 :]
    E_cross = W.objective_parts(ss["g"])[1]
    E_diff = float(0.5 * jnp.sum(gE**2) - 0.5 * jnp.sum(W.e_ref**2))
    dW = A.energy_parts(ss)["dW/W"]
    out[f"{amp:g}"] = {"E cross": E_cross, "E difference of totals": E_diff, "lambda dW/W": a.lam * dW,
                       "|cross - lambda dW/W| / |E|": abs(E_cross - a.lam * dW) / max(abs(E_cross), 1e-300)}
res_all["G0.4"] = out
log(f"[G0.4] {out}")

# ------------------------------------------------------------------ G0.5 constraint keeping
LC = mff.LinearConstraints(cur0, fields, fr, u0, pin_current=False, pin_flux=True, log=log,
                           blocks=[("Bn", Ct, np.zeros(bnf.k))])
HJ = mff.gram_J(fields, cur0, pre0, rmsJ, log=log)
L = mff.metric_L(HJ)
met = LC.metric(L)
Ctj = jnp.asarray(LC.Ct)
worst = 0.0
for j in range(50):
    yv = jnp.asarray(rng.standard_normal(n_u))
    dv = met.back(yv)
    worst = max(worst, float(jnp.max(jnp.abs(Ctj.T @ dv)) / (jnp.max(jnp.abs(Ctj)) * jnp.linalg.norm(dv))))
zr = LC.restore(z1, L)
vr = LC.violation(zr)
bm_field = np.asarray(bnf.modes_of(fields, zr - u0))
res_all["G0.5"] = {"max |C d| / (|C| |d|) over 50 projected steps": worst, "rank of [L^-1 C^T]": int(met.rank),
                   "rows": int(Ctj.shape[1]), "violation after restore": vr,
                   "B.n modes after restore, field evaluation, max": float(np.max(np.abs(bm_field))),
                   "B.n modes at the test state before restore, max": float(np.max(np.abs(
                       np.asarray(bnf.modes_of(fields, z1 - u0)))))}
log(f"[G0.5] {res_all['G0.5']}")

# ------------------------------------------------------------------ G0.7 closed form (no force bound)
if not a.skip_closed:
    t1 = time.time()
    s0 = A.state(u0)
    aux = A.aux(s0)

    def gpart(vv):
        return mff._lin(fields, vv, aux)[0]

    def gpartT(wg):
        return mff._linT(fields, (wg, jnp.zeros(3 * A.n_pts)), aux, n_u)

    H = np.zeros((n_u, n_u))
    E = np.eye(n_u)
    for i in range(n_u):
        H[:, i] = np.asarray(gpartT(gpart(jnp.asarray(E[i]))))
        if (i + 1) % 200 == 0:
            log(f"[G0.7] H columns {i + 1}/{n_u}")
    H = 0.5 * (H + H.T)
    bvec = np.asarray(gpartT(s0["g"]))  # gradient of (1/2)|g|^2 at u0 (g_J(u0) = 0: the energy part only)
    C = np.asarray(LC.Ct).T
    K = np.block([[H, C.T], [C, np.zeros((C.shape[0], C.shape[0]))]])
    rhs = np.concatenate([-bvec, np.zeros(C.shape[0])])
    sol = np.linalg.lstsq(K, rhs, rcond=1e-14)[0]
    du_star = sol[:n_u]
    u_star = np.asarray(u0) + du_star
    ss = A.state(jnp.asarray(u_star))
    pJ, Es_ = W.objective_parts(ss["g"])
    ev = np.linalg.eigvalsh(H)
    res_all["G0.7 closed form"] = {"phi_J": pJ, "E": Es_, "dW/W": A.energy_parts(ss)["dW/W"],
                                   "|J-J_fit|/rms": float(jnp.linalg.norm(ss["gJ"])),
                                   "KKT residual": float(np.linalg.norm(K @ sol - rhs) / np.linalg.norm(bvec)),
                                   "eig H min/max": [float(ev[0]), float(ev[-1])],
                                   "constraint |C du|": float(np.max(np.abs(C @ du_star))), "time [s]": time.time() - t1}
    np.savez(a.out + "_closed.npz", u=u_star, H=H, b=bvec, C=C)
    log(f"[G0.7] closed form: {res_all['G0.7 closed form']}")
for pth in a.compare:
    u_run = np.load(pth)["u"]
    u_star = np.load(a.out + "_closed.npz")["u"]
    du_s = u_star - np.asarray(u0)
    ss = A.state(jnp.asarray(u_run))
    res_all.setdefault("G0.7 compare", {})[pth] = {
        "|u - u*| / |u* - u0|": float(np.linalg.norm(u_run - u_star) / np.linalg.norm(du_s)),
        "objective run": sum(W.objective_parts(ss["g"])),
        "objective closed form": sum(W.objective_parts(A.state(jnp.asarray(u_star))["g"]))}
    log(f"[G0.7] {pth}: {res_all['G0.7 compare'][pth]}")

with open(a.out + ("_g0_compare.json" if a.skip_closed else "_g0.json"), "w") as fh:
    json.dump(res_all, fh, indent=1, default=float)
log("saved")
