"""Test 3 of the B.n-leak study: the (m, n) spectrum of the boundary flux leak of saved states, and the energy
error each leaked mode causes if the plasma-volume energy W_Omega is used as the free energy.

Leak: f = sqrt(g) dB . e^rho on the LCFS (normal flux per d theta d zeta), dB = B_BS[J(u) - J_fit], from direct
Biot-Savart (true) and from the annulus (what the solver sees). Expanded as f = sum f_mn sin(m theta - n NFP zeta)
(stellarator symmetry; the (0, 0) mode vanishes).

Energy error (first order in dB): with B_tan = grad_s phi_B + (circulations) on the LCFS, the free energy G and W_Omega
differ by
    dG - dW_Omega = -(1/mu0) oint phi_B (dB . n) dS + (toroidal-flux and poloidal-circulation terms),
phi_B the single-valued surface potential of the total reference field (B_DESC) on the LCFS. With phi_B = sum a_mn
sin(m theta - n NFP zeta): oint phi_B f d theta d zeta = 2 pi^2 sum a_mn f_mn, so mode (m, n) contributes
    E_mn = -(2 pi^2 / mu0) a_mn f_mn   [J].
Reported per state: E_mn summed in bands, the total, and the part above each cutoff (what a B.n constraint up to that
cutoff would leave), all relative to W_Omega(J_fit). The flux terms are reported separately (dPhi_tor).

usage: python scripts/energy/bn_spectrum.py --eq EQ --LMN 40 40 6 --frac 100 --N-phi 25 --colloc-rho 2.7 \\
           --state R1f_S=path.npz ... --out out/spec
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


def log(*s):
    print(f"[{time.time() - T0:7.1f} s]", *s, flush=True)


ap = argparse.ArgumentParser()
ap.add_argument("--eq", required=True)
ap.add_argument("--bext", default="desc-fit", choices=("desc-fit",))
ap.add_argument("--LMN", type=int, nargs=3, required=True)
ap.add_argument("--frac", type=float, default=40)
ap.add_argument("--N-phi", type=int, default=25)
ap.add_argument("--pad", type=float, default=0.02)
ap.add_argument("--colloc", type=float, default=1.5)
ap.add_argument("--colloc-rho", type=float, default=None)
ap.add_argument("--spec-modes", type=int, nargs=2, default=None, help="M, N of the spectrum (default: the LCFS grid's Nyquist)")
ap.add_argument("--direct", type=int, nargs=2, default=(32, 4))
ap.add_argument("--cutoffs", default="0/0,2/1,4/2,8/4,16/6,40/6", help="M/N cutoffs for the residual energy error")
ap.add_argument("--state", action="append", default=[])
ap.add_argument("--out", required=True)
ap.add_argument("--dump", action="store_true", help="also save <out>_<state>_dump.npz: the LCFS f (both routes), the field-line\n"
                " coefficients sqrt(g) B^theta, sqrt(g) B^zeta, |grad rho| (B_DESC), and per-shell |F|, |J - J_fit|, |J| profiles")
a = ap.parse_args()
os.makedirs(os.path.dirname(a.out) or ".", exist_ok=True)
res_all = {"args": vars(a)}

# ------------------------------------------------------------------ setup (as mf_fixed, --bext desc-fit)
inp0 = DESCInput.load(a.eq)
eq0 = inp0.eq
NFP = int(eq0.NFP)
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
                cyl_order=8)  # exactly as mf_fixed.py
u0 = jnp.asarray(cur0.u)
rho_pts = np.asarray(fr.rtz[:, 0])
shells = np.unique(rho_pts)
shell_id = np.searchsorted(shells, rho_pts)
B_D = np.asarray(fields.desc["B"])
fields.B_ext = jnp.asarray(B_D - np.asarray(fields.B_plasma(u0)))
fr.B_ext = np.asarray(fields.B_ext)
Jfit = mff._J(fields, u0, fields.I)
sw = jnp.asarray(fields.sw)
rmsJ = float(jnp.sqrt(jnp.sum(sw[:, None] ** 2 * Jfit**2)))
G = mff.gram_P(fields, cur0, pre0, log=log)
A = mff.FixedAnchor(fields, cur0, pre0, u0, rmsJ, ref["norm"], G, log=log)
del G
per = 2 * NFP if fr.sym else NFP
A.enable_energy(0.0, mf.Weights(1.0, 1.0, 1.0), float(per * np.sum(fr.dV)), log=log)
n_u = A.n_u
psi_fn = jax.jit(mff.toroidal_flux_fn(fields, fr))
psi0 = float(psi_fn(u0))

# ------------------------------------------------------------------ LCFS grid, spectrum basis, surface potential
lc = mff.LCFSField(inp0, fields, fr, 1, 1, log=log)  # only its points/geometry/interpolator are used here
rtz = lc.rtz
th_u, ze_u = np.unique(rtz[:, 1]), np.unique(rtz[:, 2])
Ms = a.spec_modes[0] if a.spec_modes else len(th_u) // 2 - 1
Ns = a.spec_modes[1] if a.spec_modes else (len(ze_u) if fr.sym else len(ze_u) // 2) - 1
modes = mff.bn_modes(Ms, Ns, bool(fr.sym))
assert all(k == "s" for _, _, k in modes), "stellarator-symmetric case only"
mm = np.array([m for m, _, _ in modes])
nn = np.array([n for _, n, _ in modes])
arg = mm[:, None] * rtz[None, :, 1] - nn[:, None] * NFP * rtz[None, :, 2]
S = 2.0 * np.sin(arg) / len(rtz)  # f_mn = S @ f  (half-period grid: the even products' mean is the full mean)
Cc = 2.0 * np.cos(arg) / len(rtz)
log(f"spectrum: |m| <= {Ms}, |n| <= {Ns}: {len(modes)} sine modes on {len(th_u)} x {len(ze_u)} LCFS points")

d = inp0.compute_points(["B", "e_theta", "e_zeta", "sqrt(g)", "B^theta", "B^zeta", "|grad(rho)|", "|B|"], rtz)
lcfs_geo = {"sgBt": np.asarray(d["sqrt(g)"]) * np.asarray(d["B^theta"]), "sgBz": np.asarray(d["sqrt(g)"]) * np.asarray(d["B^zeta"]),
            "grad_rho": np.asarray(d["|grad(rho)|"]), "absB": np.asarray(d["|B|"]), "sqrt_g": np.asarray(d["sqrt(g)"])}
Bt = np.sum(np.asarray(d["B"]) * np.asarray(d["e_theta"]), 1)  # covariant B_theta (B_DESC)
Bz = np.sum(np.asarray(d["B"]) * np.asarray(d["e_zeta"]), 1)
ct, cz = Cc @ Bt, Cc @ Bz  # cos coefficients
a_phi = np.where(mm > 0, ct / np.maximum(mm, 1), -cz / np.where(nn != 0, nn * NFP, 1))
# consistency of the two routes where both apply (m > 0, n != 0): a = c_theta / m = -c_zeta / (n NFP)
both = (mm > 0) & (nn != 0) & (mm <= 2 * eq0.M) & (np.abs(nn) <= 2 * eq0.N)  # modes DESC resolves (x2)
d_ = ct[both] / mm[both] + cz[both] / (nn[both] * NFP)
cons = float(np.sqrt(np.sum(d_**2) / np.sum((ct[both] / mm[both]) ** 2)))  # energy-weighted relative mismatch
res_all["surface potential"] = {"circulations: <B_theta>, <B_zeta> [T m]": [float(Bt.mean()), float(Bz.mean())],
                                "route consistency (weighted rel, m>0, n!=0, resolved modes)": cons,
                                "|a| by m band": {}}
log(f"phi_B: <B_theta> {Bt.mean():.4e}, <B_zeta> {Bz.mean():.4e} T m; theta/zeta route consistency {cons:.2e}")

bands = [(0, 0), (1, 1), (2, 2), (3, 3), (4, 4), (5, 8), (9, 16), (17, 40), (41, 10**6)]


def band_sum(v, m_lo, m_hi):
    sel = (mm >= m_lo) & (mm <= m_hi)
    return float(np.sum(v[sel]))


for lo, hi in bands:
    res_all["surface potential"]["|a| by m band"][f"m {lo}-{hi}"] = float(np.sqrt(band_sum(a_phi**2, lo, hi)))
cuts = [tuple(int(x) for x in c.split("/")) for c in a.cutoffs.split(",")]

disp = Displacement(eq0, 2, 2)
xi0 = jnp.zeros(disp.n)
df = DirectField(cur0, disp, (len(th_u), len(ze_u)), a.direct[0], a.direct[1], bool(fr.sym), dtype=jnp.float64, chunk=64)
R, Z = lc._R, lc._Z
phi = rtz[:, 2]
x_t = jnp.asarray(np.column_stack([R * np.cos(phi), R * np.sin(phi), Z]))
e = np.asarray(lc.sg_e)
e_c = jnp.asarray(np.column_stack([e[:, 0] * np.cos(phi) - e[:, 1] * np.sin(phi), e[:, 0] * np.sin(phi) + e[:, 1] * np.cos(phi),
                                   e[:, 2]]))
W_ref = A.W_ref_J
log(f"direct quadrature: {df.n_sources} sources; W_ref {W_ref:.6e} J")

res_all["states"] = {}
for spec in a.state:
    nm, pth = spec.split("=", 1)
    u = u0 if pth == "0" else jnp.asarray(np.load(pth)["u"])  # "NAME=0": J_fit itself
    du = u - u0
    t1 = time.time()
    f_dir = np.asarray(jnp.sum(df(du, 0.0, xi0, x_t) * e_c, axis=1))
    f_ann = np.asarray(lc.f(fields, du))
    out = {"dPhi_tor/Phi": (float(psi_fn(u)) - psi0) / psi0}
    for tag, f in (("direct", f_dir), ("annulus", f_ann)):
        fm = S @ f
        E = -(2 * np.pi**2 / MU0) * a_phi * fm / W_ref
        o = {"rms f / scale": float(np.sqrt(np.mean(f**2)) / lc.scale),
             "spectrum resolves f (rms of f - sum f_mn sin, rel)": float(np.sqrt(np.mean((f - fm @ np.sin(arg)) ** 2))
                                                                       / max(np.sqrt(np.mean(f**2)), 1e-300)),
             "|f_mn| / scale by m band": {f"m {lo}-{hi}": float(np.sqrt(band_sum(fm**2, lo, hi))) / lc.scale for lo, hi in bands},
             "E / W by m band": {f"m {lo}-{hi}": band_sum(E, lo, hi) for lo, hi in bands},
             "E / W total": float(np.sum(E)),
             "|E| / W total (no cancellation)": float(np.sum(np.abs(E))),
             "E / W left above cutoff": {f"{Mc}/{Nc}": float(np.sum(E[(mm > Mc) | (np.abs(nn) > Nc)])) for Mc, Nc in cuts},
             "top 8 modes (m, n, E/W)": [[int(mm[i]), int(nn[i]), float(E[i])] for i in np.argsort(-np.abs(E))[:8]]}
        out[tag] = o
    st = A.state(u)
    out["dW/W (Omega)"] = A.energy_parts(st)["dW/W"]
    if a.dump:
        w = np.asarray(sw) ** 2
        def shell_mean(v):
            return np.bincount(shell_id, w * v, len(shells)) / np.bincount(shell_id, w, len(shells))
        absF = np.asarray(st["absF"])
        dJ = np.linalg.norm(np.asarray(st["J"]) - np.asarray(Jfit), axis=1)
        aJ = np.linalg.norm(np.asarray(st["J"]), axis=1)
        np.savez(a.out + f"_{nm}_dump.npz", rtz=rtz, f_dir=f_dir, f_ann=f_ann, scale=lc.scale, mm=mm, nn=nn, NFP=NFP,
                 fm_dir=S @ f_dir, fm_ann=S @ f_ann, a_phi=a_phi, W_ref=W_ref, **lcfs_geo,
                 shells=shells, F_shell=shell_mean(absF), F_shell_rms=np.sqrt(shell_mean(absF**2)),
                 dJ_shell_rms=np.sqrt(shell_mean(dJ**2)), J_shell_rms=np.sqrt(shell_mean(aJ**2)),
                 shell_w=np.bincount(shell_id, w, len(shells)), M=float(st["M"]), norm=float(ref["norm"]),
                 F_desc_vol=float(ref["F_desc_vol"]), rmsJ=rmsJ)
    out["time [s]"] = time.time() - t1
    res_all["states"][nm] = out
    log(f"[{nm}] dW/W {out['dW/W (Omega)']:+.3e}; direct: rms f {out['direct']['rms f / scale']:.3e}, E/W total "
        f"{out['direct']['E / W total']:+.3e} (|.| sum {out['direct']['|E| / W total (no cancellation)']:.3e}); left above "
        + ", ".join(f"{k} {v:+.2e}" for k, v in out["direct"]["E / W left above cutoff"].items())
        + f"; annulus E/W total {out['annulus']['E / W total']:+.3e}; dPhi/Phi {out['dPhi_tor/Phi']:+.2e}")
    log(f"[{nm}] direct |f_mn| by m band: " + ", ".join(f"{k} {v:.2e}" for k, v in out["direct"]["|f_mn| / scale by m band"].items()))
    log(f"[{nm}] direct E/W by m band: " + ", ".join(f"{k} {v:+.2e}" for k, v in out["direct"]["E / W by m band"].items()))
    with open(a.out + "_spec.json", "w") as fh:
        json.dump(res_all, fh, indent=1, default=float)
with open(a.out + "_spec.json", "w") as fh:
    json.dump(res_all, fh, indent=1, default=float)
log(f"saved {a.out}_spec.json")
