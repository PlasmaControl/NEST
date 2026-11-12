"""Soft directions of DESC's fixed-boundary force residual at an axisymmetric base.

The base (N = 0) is raised to N = 1 with no re-solve, so the state is exactly axisymmetric. Build the objective eq.solve
uses (ForceBalance; fixed-boundary + self-consistency constraints, removed by LinearConstraintProjection), take the
Jacobian of the scaled residual in the reduced variables, apply the solver's x_scale='jac' column scaling, and SVD it.

For each of the smallest singular vectors: map to (dR_lmn, dZ_lmn, dL_lmn); n = 0 / n = 1 split; displacement of the
coordinate points in the poloidal plane split into the part normal to the flux surface (physical: moves the surface)
and the tangential part (relabelling within the surface); radial profile, peak and width of the normal part; its
poloidal m content (DESC theta) at the peak.

Usage: python svd_soft.py --base tokamak_island_C_L16M16.h5 --k 20 --out svd_L16.json
"""
import argparse
import json
import time

import numpy as np

ap = argparse.ArgumentParser()
ap.add_argument("--base", required=True)
ap.add_argument("--k", type=int, default=20, help="smallest singular vectors to analyse")
ap.add_argument("--chunk", type=int, default=100, help="jac_chunk_size")
ap.add_argument("--out", required=True)
a = ap.parse_args()

import desc.io  # noqa: E402
from desc.grid import LinearGrid  # noqa: E402
from desc.objectives import ForceBalance, ObjectiveFunction, get_fixed_boundary_constraints  # noqa: E402
from desc.objectives.getters import maybe_add_self_consistency  # noqa: E402
from desc.optimize._constraint_wrappers import LinearConstraintProjection  # noqa: E402
from desc.optimize.utils import compute_jac_scale  # noqa: E402

T0 = time.time()


def log(*s):
    print(f"[{time.time() - T0:7.1f} s]", *s, flush=True)


eq = desc.io.load(a.base)
if not hasattr(eq, "R_lmn"):
    eq = eq[-1]
assert eq.N == 0 and eq.NFP == 1
eq.change_resolution(N=1, N_grid=2)
log(f"{a.base}: L {eq.L} M {eq.M} N {eq.N}, grid {eq.L_grid}/{eq.M_grid}/{eq.N_grid}")

obj = ObjectiveFunction(ForceBalance(eq), jac_chunk_size=a.chunk)
con = ObjectiveFunction(maybe_add_self_consistency(eq, get_fixed_boundary_constraints(eq)))
obj.build(verbose=0)
con.build(verbose=0)
lcp = LinearConstraintProjection(obj, con)
lcp.build(verbose=0)
x = lcp.x(eq)
f = np.asarray(lcp.compute_scaled_error(x))
J = np.asarray(lcp.jac_scaled_error(x))
log(f"J {J.shape}, cost 1/2|f|^2 = {0.5 * f @ f:.4e}")
scale, _ = compute_jac_scale(J)
scale = np.asarray(scale)
Jh = J * scale[None, :]
U, s, Vt = np.linalg.svd(Jh, full_matrices=False)
log(f"SVD done: s_max {s[0]:.3e}, s_min {s[-1]:.3e}, cond {s[0] / s[-1]:.3e}")

# gradient in the solver's scaled variables and its split onto the singular vectors
g = Jh.T @ f

p0 = lcp.unpack_state(x, per_objective=False)[0]
modes = {k: getattr(eq, k[0] + "_basis").modes for k in ("R_lmn", "Z_lmn", "L_lmn")}

# geometry grid: base derivatives for the normal / tangential split
rho = np.linspace(0.04, 0.99, 96)
G = LinearGrid(rho=rho, theta=64, zeta=8, NFP=1, sym=False)
dG = eq.compute(["R_t", "Z_t", "sqrt(g)", "iota"], grid=G)
Rt, Zt, w = (np.asarray(dG[q]) for q in ("R_t", "Z_t", "sqrt(g)"))
w = np.abs(w)
et = np.hypot(Rt, Zt)
AR = np.asarray(eq.R_basis.evaluate(G))
AZ = np.asarray(eq.Z_basis.evaluate(G))
AL = np.asarray(eq.L_basis.evaluate(G))
iota = G.compress(np.asarray(dG["iota"]))
q_of = lambda r: float(1 / abs(np.interp(r, rho, iota)))  # noqa: E731
ri = G.inverse_rho_idx
th = G.nodes[:, 1]
ze = G.nodes[:, 2]
nth, nze = 64, 8

vecs, rows = {}, []
for j in range(1, a.k + 1):
    i = len(s) - j
    vh = Vt[i]
    v = scale * vh  # step in the reduced variables
    p1 = lcp.unpack_state(x + v, per_objective=False)[0]
    d = {k: np.asarray(p1[k]) - np.asarray(p0[k]) for k in ("R_lmn", "Z_lmn", "L_lmn")}
    nfr = {}
    for k in d:
        n_abs = np.abs(modes[k][:, 2])
        tot = np.linalg.norm(d[k]) or 1.0
        nfr[k] = float(np.linalg.norm(d[k][n_abs == 1]) / tot)
    dR, dZ, dL = AR @ d["R_lmn"], AZ @ d["Z_lmn"], AL @ d["L_lmn"]
    xn = (dR * Zt - dZ * Rt) / et
    xt = (dR * Rt + dZ * Zt) / et
    En, Et = np.sum(w * xn**2), np.sum(w * xt**2)
    prof = np.array([np.sqrt(np.sum((w * xn**2)[ri == r]) / np.sum(w[ri == r])) for r in range(len(rho))])
    ip = int(np.argmax(prof))
    half = prof >= prof[ip] / 2
    # contiguous half-max band around the peak
    lo = ip
    while lo > 0 and half[lo - 1]:
        lo -= 1
    hi = ip
    while hi < len(rho) - 1 and half[hi + 1]:
        hi += 1
    # poloidal m content of the normal displacement at the peak (2-D FFT over theta, zeta)
    sel = ri == ip
    X = np.zeros((nze, nth))
    ord_ = np.lexsort((th[sel], ze[sel]))
    X[:] = xn[sel][ord_].reshape(nze, nth)
    Fm = np.abs(np.fft.fft2(X)) / X.size
    mpow = np.array([Fm[:, m].sum() + (Fm[:, -m].sum() if m else 0) for m in range(9)])
    top = [int(m) for m in np.argsort(-mpow)[:3]]
    lam_rms = float(np.sqrt(np.sum(w * dL**2) / np.sum(w)))
    disp_rms = float(np.sqrt((En + Et) / np.sum(w)))
    row = {"index from smallest": j, "s": float(s[i]), "s/s_max": float(s[i] / s[0]),
           "gradient share |g.v|^2/|g|^2": float((g @ vh) ** 2 / (g @ g)),
           "n=1 fraction (R, Z, L)": [nfr["R_lmn"], nfr["Z_lmn"], nfr["L_lmn"]],
           "normal fraction of displacement": float(np.sqrt(En / (En + Et))) if En + Et > 0 else 0.0,
           "rms displacement [m]": disp_rms, "rms d lambda": lam_rms,
           "normal profile peak rho": float(rho[ip]), "q at peak": q_of(rho[ip]),
           "half-max band rho": [float(rho[lo]), float(rho[hi])],
           "normal m content at peak (top 3)": top,
           "normal m power at peak m=0..8": (mpow / (mpow.sum() or 1)).tolist(),
           "normal profile": prof.tolist()}
    rows.append(row)
    vecs[f"v{j}"] = np.concatenate([d["R_lmn"], d["Z_lmn"], d["L_lmn"]])
    log(f"#{j:2d} s {s[i]:.3e} (s/smax {s[i] / s[0]:.1e}) n=1 frac R/Z/L "
        f"{nfr['R_lmn']:.2f}/{nfr['Z_lmn']:.2f}/{nfr['L_lmn']:.2f} normal frac {row['normal fraction of displacement']:.3f} "
        f"disp {disp_rms:.2e} m dlam {lam_rms:.2e}; peak rho {rho[ip]:.3f} (q {q_of(rho[ip]):.2f}) band "
        f"[{rho[lo]:.2f}, {rho[hi]:.2f}] m {top}; grad share {row['gradient share |g.v|^2/|g|^2']:.1e}")

out = {"base": a.base, "L": eq.L, "M": eq.M, "N": eq.N, "J shape": list(J.shape), "cost": float(0.5 * f @ f),
       "s (all, descending)": s.tolist(), "rho": rho.tolist(), "vectors": rows,
       "rational n=1 surfaces": {"q=2": 0.6375, "q=3": 0.913, "q=4": 0.988}}
json.dump(out, open(a.out, "w"), indent=1)
np.savez(a.out.replace(".json", "_vecs.npz"), **vecs)
log(f"saved {a.out}")
