"""Is a soft singular vector a pure relabelling (same field) or a different equilibrium?

For a direction v = (dR_lmn, dZ_lmn, dL_lmn) from svd_soft.py: states x +- eps v. At fixed PHYSICAL points (the
base's coordinate nodes, (R, phi, Z)), find the computational coordinates in each perturbed state and evaluate B there.
  dB_euler = [B_+(X) - B_-(X)] / (2 eps)      (field change at fixed physical points; 0 for a pure relabelling)
  dB_lagr  = [B_+(node) - B_-(node)] / (2 eps) (field change following the coordinate nodes)
Ratios are reported against |dB_lagr| and against |B| * |dx| / a. A large-singular-value vector is the control.

Usage: python svd_gauge_check.py --base tokamak_island_C_L24M24.h5 --vecs svd_L24_vecs.npz --json svd_L24.json \
       --which 1 2 4 15 --out gauge_L24.json
"""
import argparse
import json

import numpy as np

ap = argparse.ArgumentParser()
ap.add_argument("--base", required=True)
ap.add_argument("--vecs", required=True)
ap.add_argument("--json", required=True)
ap.add_argument("--which", type=int, nargs="+", default=[1, 2])
ap.add_argument("--control", action="store_true", help="also test the largest-singular-value direction")
ap.add_argument("--out", required=True)
a = ap.parse_args()

import desc.io  # noqa: E402
from desc.grid import Grid, LinearGrid  # noqa: E402

eq = desc.io.load(a.base)
eq.change_resolution(N=1, N_grid=2)
nR, nZ = len(eq.R_lmn), len(eq.Z_lmn)
V = np.load(a.vecs)
sv = json.load(open(a.json))
amin = 1.198366

G = LinearGrid(rho=np.linspace(0.1, 0.95, 18), theta=24, zeta=4, NFP=1, sym=False)
names = ["B_R", "B_phi", "B_Z"]
d0 = eq.compute(names + ["R", "phi", "Z", "|B|"], grid=G)
X = np.column_stack([np.asarray(d0["R"]), np.asarray(d0["phi"]), np.asarray(d0["Z"])])
B0 = np.column_stack([np.asarray(d0[k]) for k in names])
Bmag = np.asarray(d0["|B|"])


def state(dv, eps):
    e = eq.copy()
    e.R_lmn = eq.R_lmn + eps * dv[:nR]
    e.Z_lmn = eq.Z_lmn + eps * dv[nR:nR + nZ]
    e.L_lmn = eq.L_lmn + eps * dv[nR + nZ:]
    return e


def B_at_physical(e):
    rtz = e.map_coordinates(X, inbasis=("R", "phi", "Z"), guess=G.nodes, tol=1e-13, maxiter=60)
    rtz = np.asarray(rtz)
    g = Grid(rtz, sort=False)
    d = e.compute(names, grid=g)
    return np.column_stack([np.asarray(d[k]) for k in names]), rtz


# sanity: the base mapped onto itself
B_self, rtz_self = B_at_physical(eq)
res = {"self-map |dB|/|B| max": float(np.max(np.linalg.norm(B_self - B0, axis=1) / Bmag)),
       "self-map coord error max": float(np.max(np.abs(rtz_self - G.nodes))), "vectors": []}
print("self-map:", res["self-map |dB|/|B| max"], res["self-map coord error max"], flush=True)

tests = [(f"v{j}", V[f"v{j}"], sv["vectors"][j - 1]["s"]) for j in a.which]
if a.control:  # a stiff physical direction: the base's own R/Z (n = 0 and n = 1 blocks) mixed, unit coefficient
    rng = np.random.default_rng(0)
    dv = np.zeros(nR + nZ + len(eq.L_lmn))
    dv[: nR + nZ] = rng.standard_normal(nR + nZ) * 1e-3
    tests.append(("random R,Z control", dv, None))

for name, dv, s in tests:
    AR, AZ = np.asarray(eq.R_basis.evaluate(G)), np.asarray(eq.Z_basis.evaluate(G))
    dx = np.hypot(AR @ dv[:nR], AZ @ dv[nR:nR + nZ])
    eps = 1e-5 / max(dx.max(), 1e-30)  # max coordinate displacement 10 um
    ep, em = state(dv, eps), state(dv, -eps)
    Bp, _ = B_at_physical(ep)
    Bm, _ = B_at_physical(em)
    dp = ep.compute(names, grid=G)
    dm = em.compute(names, grid=G)
    dBe = (Bp - Bm) / (2 * eps)
    dBl = (np.column_stack([np.asarray(dp[k]) for k in names]) - np.column_stack([np.asarray(dm[k]) for k in names])) \
        / (2 * eps)
    ne, nl = np.linalg.norm(dBe, axis=1), np.linalg.norm(dBl, axis=1)
    scale = Bmag * np.maximum(dx, 1e-30) / amin
    row = {"vector": name, "s": s, "eps": eps,
           "rms |dB_euler| / rms |dB_lagr|": float(np.sqrt(np.mean(ne**2) / np.mean(nl**2))),
           "rms |dB_euler| / rms(|B| |dx| / a)": float(np.sqrt(np.mean(ne**2) / np.mean(scale**2))),
           "rms |dB_lagr| / rms(|B| |dx| / a)": float(np.sqrt(np.mean(nl**2) / np.mean(scale**2)))}
    res["vectors"].append(row)
    print(row, flush=True)
json.dump(res, open(a.out, "w"), indent=1)
