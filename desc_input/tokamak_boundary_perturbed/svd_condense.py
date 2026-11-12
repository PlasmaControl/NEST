"""Do the small singular values of DESC's force Jacobian all come from the poloidal-angle (theta) relabelling?

Same setup as svd_soft.py (axisymmetric base raised to N = 1, ForceBalance + fixed-boundary constraints, reduced
variables, the solver's x_scale='jac' column scaling). Append linearized theta spectral-condensation rows and SVD the
augmented matrix [J_h; c * s_max * P_hat]:

  tangential : on surfaces rho_k and zeta planes, the coordinate displacement projected onto e_theta/|e_theta|
               (both of its R, Z components), Fourier-transformed in theta, harmonic m weighted by sqrt(1 + m^2).
               Zero for displacements normal to the flux surfaces, so it lifts relabellings and leaves physical
               (surface-moving) directions alone: the discriminating test.
  full       : the same on the whole (dR, dZ) (plain spectral-width penalty; also lifts physical high-m directions).

P_hat is P scaled to unit spectral norm (in the solver's scaled variables); c in --c. Reported per case: s_min/s_max,
counts of s/s_max below thresholds, and for the smallest augmented vectors the surface-normal fraction of the
displacement, the force-row and penalty-row parts |J_h v|, |c s_max P_hat v|, and where the normal displacement peaks.

Usage: python svd_condense.py --base tokamak_island_C_L16M16.h5 --c 1e-4 1e-3 1e-2 --out cond_L16.json
"""
import argparse
import json
import time

import numpy as np

ap = argparse.ArgumentParser()
ap.add_argument("--base", required=True)
ap.add_argument("--c", type=float, nargs="+", default=[1e-4, 1e-3, 1e-2])
ap.add_argument("--kinds", nargs="+", default=["tangential", "full"])
ap.add_argument("--k", type=int, default=6, help="smallest augmented vectors to describe")
ap.add_argument("--chunk", type=int, default=100)
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
eq.change_resolution(N=1, N_grid=2)
obj = ObjectiveFunction(ForceBalance(eq), jac_chunk_size=a.chunk)
con = ObjectiveFunction(maybe_add_self_consistency(eq, get_fixed_boundary_constraints(eq)))
obj.build(verbose=0)
con.build(verbose=0)
lcp = LinearConstraintProjection(obj, con)
lcp.build(verbose=0)
x = np.asarray(lcp.x(eq))
J = np.asarray(lcp.jac_scaled_error(x))
scale = np.asarray(compute_jac_scale(J)[0])
Jh = J * scale[None, :]
nred = len(x)
log(f"L {eq.L}: J {J.shape}")

# reduced -> (R_lmn, Z_lmn) map (affine; columns by unit steps, in scaled variables)
p0 = lcp.unpack_state(x, per_objective=False)[0]
R0, Z0 = np.asarray(p0["R_lmn"]), np.asarray(p0["Z_lmn"])
MR = np.zeros((len(R0), nred))
MZ = np.zeros((len(Z0), nred))
for i in range(nred):
    e = np.zeros(nred)
    e[i] = scale[i]
    p1 = lcp.unpack_state(x + e, per_objective=False)[0]
    MR[:, i] = np.asarray(p1["R_lmn"]) - R0
    MZ[:, i] = np.asarray(p1["Z_lmn"]) - Z0
log("reduced -> R, Z map built")


def surf_eval(rho, nth, nze):
    g = LinearGrid(rho=rho, theta=nth, zeta=nze, NFP=1, sym=False)
    d = eq.compute(["R_t", "Z_t", "sqrt(g)"], grid=g)
    o = np.lexsort((g.nodes[:, 1], g.nodes[:, 2], g.nodes[:, 0]))  # rho slowest, then zeta, then theta
    Rt, Zt = np.asarray(d["R_t"])[o], np.asarray(d["Z_t"])[o]
    et = np.hypot(Rt, Zt)
    AR = np.asarray(eq.R_basis.evaluate(g))[o]
    AZ = np.asarray(eq.Z_basis.evaluate(g))[o]
    return g, o, Rt / et, Zt / et, AR, AZ, np.abs(np.asarray(d["sqrt(g)"]))[o]


# penalty rows
rho_p = np.linspace(0.05, 0.99, 24)
nth, nze = 4 * eq.M + 8, 4
_, _, tR, tZ, ARp, AZp, _ = surf_eval(rho_p, nth, nze)
dR = ARp @ MR  # (points, nred)
dZ = AZp @ MZ
st = tR[:, None] * dR + tZ[:, None] * dZ
m = np.arange(nth // 2 + 1)
wm = np.sqrt(1.0 + m**2) / np.sqrt(nth)


def fourier_rows(F):
    Fh = np.fft.rfft(F.reshape(len(rho_p), nze, nth, nred), axis=2) * wm[None, None, :, None]
    Fh = Fh.reshape(-1, nred)
    return np.vstack([Fh.real, Fh.imag])


P = {"tangential": np.vstack([fourier_rows(tR[:, None] * st), fourier_rows(tZ[:, None] * st)]),
     "full": np.vstack([fourier_rows(dR), fourier_rows(dZ)])}
del dR, dZ, st
for k in P:  # exact reduction: [J; c P] and [J; c R_P] have the same singular values and right vectors
    nrow = P[k].shape[0]
    P[k] = np.linalg.qr(P[k], mode="r")
    P[k] = P[k] / np.linalg.norm(P[k], 2)
    log(f"{k}: {nrow} rows -> R factor {P[k].shape}")
log("penalty rows built: " + ", ".join(f"{k} {v.shape}" for k, v in P.items()))

# geometry grid for describing vectors
rho_g = np.linspace(0.04, 0.99, 96)
gg, og, tRg, tZg, ARg, AZg, wg = surf_eval(rho_g, 64, 8)
rho_of = gg.nodes[og, 0]


nR_abs = np.abs(eq.R_basis.modes[:, 2])
nZ_abs = np.abs(eq.Z_basis.modes[:, 2])
ze_g, th_g = gg.nodes[og, 2], gg.nodes[og, 1]


def labels(vh):
    """n = 1 fraction of (dR, dZ) coefficients; half-max band of the normal profile; m power at the peak."""
    cR, cZ = MR @ vh, MZ @ vh
    n1 = np.sqrt(np.sum(cR[nR_abs == 1] ** 2) + np.sum(cZ[nZ_abs == 1] ** 2)) / np.sqrt(np.sum(cR**2) + np.sum(cZ**2))
    dRv, dZv = ARg @ cR, AZg @ cZ
    xn = dRv * tZg - dZv * tRg
    prof = np.array([np.sum((wg * xn**2)[rho_of == r]) / np.sum(wg[rho_of == r]) for r in rho_g]) ** 0.5
    ip = int(np.argmax(prof))
    lo, hi = ip, ip
    while lo > 0 and prof[lo - 1] >= prof[ip] / 2:
        lo -= 1
    while hi < len(rho_g) - 1 and prof[hi + 1] >= prof[ip] / 2:
        hi += 1
    X = xn[rho_of == rho_g[ip]].reshape(8, 64)  # (zeta, theta): og sorts rho, zeta, theta
    Fm = np.abs(np.fft.fft2(X))
    mp = np.array([Fm[:, mm].sum() + (Fm[:, -mm].sum() if mm else 0) for mm in range(9)])
    return float(n1), [float(rho_g[lo]), float(rho_g[hi])], [int(v) for v in np.argsort(-mp)[:3]]


def describe(vh):
    dRv, dZv = ARg @ (MR @ vh), AZg @ (MZ @ vh)
    xn = dRv * tZg - dZv * tRg
    xt = dRv * tRg + dZv * tZg
    En, Et = np.sum(wg * xn**2), np.sum(wg * xt**2)
    prof = np.array([np.sum((wg * xn**2)[rho_of == r]) / np.sum(wg[rho_of == r]) for r in rho_g])
    return float(np.sqrt(En / (En + Et))) if En + Et > 0 else 0.0, float(rho_g[np.argmax(prof)])


s0 = np.linalg.svd(Jh, compute_uv=False)
smax = s0[0]
thr = [1e-8, 1e-6, 1e-4]
out = {"base": a.base, "L": eq.L, "J shape": list(J.shape), "s_max": float(smax),
       "force only": {"s_min/s_max": float(s0[-1] / smax),
                      "count s/s_max below": {f"{t:g}": int(np.sum(s0 / smax < t)) for t in thr}},
       "cases": []}
log(f"force only: s_min/s_max {s0[-1] / smax:.2e}; counts below {thr}: "
    f"{[int(np.sum(s0 / smax < t)) for t in thr]}")
for kind in a.kinds:
    for c in a.c:
        A = np.vstack([Jh, c * smax * P[kind]])
        U, s, Vt = np.linalg.svd(A, full_matrices=False)
        case = {"penalty": kind, "c": c, "s_min/s_max": float(s[-1] / smax),
                "count s/s_max below": {f"{t:g}": int(np.sum(s / smax < t)) for t in thr}, "smallest": []}
        for j in range(1, a.k + 1):
            vh = Vt[-j]
            nf, rp = describe(vh)
            n1, band, mtop = labels(vh)
            case["smallest"].append({"s/s_max": float(s[-j] / smax), "normal fraction": nf, "peak rho": rp, "n=1 fraction": n1, "half-max band": band, "m top3": mtop,
                                     "|J_h v|/s_max": float(np.linalg.norm(Jh @ vh) / smax),
                                     "|penalty v|/s_max": float(np.linalg.norm(c * smax * P[kind] @ vh) / smax)})
        out["cases"].append(case)
        sm = case["smallest"]
        log(f"{kind:10s} c {c:.0e}: s_min/s_max {s[-1] / smax:.2e}; counts below {thr}: "
            f"{[case['count s/s_max below'][f'{t:g}'] for t in thr]}; smallest 3: "
            + "; ".join(f"s {r['s/s_max']:.1e} nf {r['normal fraction']:.2f} rho {r['peak rho']:.2f} "
                        f"n1 {r['n=1 fraction']:.2f} band [{r['half-max band'][0]:.2f},{r['half-max band'][1]:.2f}] m {r['m top3']} "
                        f"F {r['|J_h v|/s_max']:.1e} P {r['|penalty v|/s_max']:.1e}" for r in sm[:a.k]))
json.dump(out, open(a.out, "w"), indent=1)
log(f"saved {a.out}")
