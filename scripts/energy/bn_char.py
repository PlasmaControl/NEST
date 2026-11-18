"""B.n-leak characterization from bn_spectrum --dump files (CPU, no NEST fields needed).

For each state: the LCFS normal-flux leak f = sqrt(g) dB^rho (dB = B[J] - B[J_fit]) by (m, n), and the boundary
displacement it is equivalent to. A displacement xi^rho of the LCFS changes the normal field by (to first order, with
B_DESC tangent to the LCFS)  sqrt(g) dB^rho = sqrt(g) B . grad xi^rho = sqrt(g)B^theta d_theta xi + sqrt(g)B^zeta d_zeta xi,
so xi solves that magnetic differential equation on the LCFS. It is solved by least squares in the stellarator-symmetric
cos basis cos(m theta - n NFP zeta) (the (0, 0) mode is in the null space; near-resonant modes m iota ~ n NFP are
amplified, which the per-mode table shows). xi_n = xi^rho / |grad rho| is the normal displacement [m].

Also: per-shell <|F|> of each state relative to DESC's own <|F|> on the same shells, and rms |J - J_fit| per shell.

usage: python scripts/energy/bn_char.py --dumps DIR --eq EQ.h5 --out OUT_PREFIX
"""
import argparse
import glob
import json
import os

import numpy as np

ap = argparse.ArgumentParser()
ap.add_argument("--dumps", required=True, help="directory with <prefix>_<state>_dump.npz")
ap.add_argument("--eq", required=True)
ap.add_argument("--out", required=True)
ap.add_argument("--xi-modes", type=int, nargs=2, default=(24, 6), help="M, N of the xi cos basis")
ap.add_argument("--ridge", type=float, default=1e-10, help="relative Tikhonov on the xi solve")
a = ap.parse_args()

files = sorted(glob.glob(os.path.join(a.dumps, "*_dump.npz")))
assert files, "no dumps"
res = {}


def fam(mm, nn, v):
    """Sum of squares by n family and m band."""
    out = {}
    for n in sorted(set(np.abs(nn).tolist())):
        out[f"|n|={n}"] = float(np.sqrt(np.sum(v[np.abs(nn) == n] ** 2)))
    return out


desc_shell = None
for fpath in files:
    nm = os.path.basename(fpath).split("_dump.npz")[0].split("_", 1)[1]
    d = np.load(fpath)
    rtz, NFP = d["rtz"], int(d["NFP"])
    th, ze = rtz[:, 1], rtz[:, 2]
    mm, nn = d["mm"], d["nn"]
    out = {}
    for route in ("dir", "ann"):
        f, fm = d[f"f_{route}"], d[f"fm_{route}"]
        rms = float(np.sqrt(np.mean(f**2)))
        out[f"rms f/scale ({route})"] = rms / float(d["scale"])
        out[f"f by n family /scale ({route})"] = {k: v / float(d["scale"]) for k, v in fam(mm, nn, fm).items()}
        top = np.argsort(-np.abs(fm))[:10]
        out[f"top f modes ({route}) (m, n, f_mn/scale)"] = [[int(mm[i]), int(nn[i]), float(fm[i] / d["scale"])] for i in top]
    # --- xi solve (direct route = the true leak)
    Mx, Nx = a.xi_modes
    km = [(m, n) for m in range(0, Mx + 1) for n in range(-Nx, Nx + 1) if not (m == 0 and n <= 0)]
    km = np.array(km)
    arg = km[:, 0][None] * th[:, None] - km[:, 1][None] * NFP * ze[:, None]
    Amat = d["sgBt"][:, None] * (-km[:, 0][None] * np.sin(arg)) + d["sgBz"][:, None] * (km[:, 1][None] * NFP * np.sin(arg))
    f = d["f_dir"]
    lam = a.ridge * np.trace(Amat.T @ Amat) / Amat.shape[1]
    x = np.linalg.solve(Amat.T @ Amat + lam * np.eye(Amat.shape[1]), Amat.T @ f)
    resid = float(np.linalg.norm(Amat @ x - f) / max(np.linalg.norm(f), 1e-300))
    xi_rho = np.cos(arg) @ x
    xi_n = xi_rho / d["grad_rho"]
    # xi_n by mode: project onto cos modes (half-period symmetric grid: 2 <., cos> mean)
    xin_m = 2 * np.cos(arg).T @ xi_n / len(xi_n)
    top = np.argsort(-np.abs(xin_m))[:10]
    # resonance factor of each mode: |m <sgBt> - n NFP <sgBz>| relative to |m <sgBt>| + |n NFP <sgBz>|
    bt, bz = float(np.mean(d["sgBt"])), float(np.mean(d["sgBz"]))
    out["iota_edge (<sqrt(g)B^theta>/<sqrt(g)B^zeta>)"] = bt / bz
    out["xi solve residual (rel)"] = resid
    out["xi_n rms [mm]"] = float(np.sqrt(np.mean(xi_n**2)) * 1e3)
    out["xi_n max [mm]"] = float(np.max(np.abs(xi_n)) * 1e3)
    out["xi_n by n family rms [mm]"] = {k: v * 1e3 / np.sqrt(2) for k, v in fam(km[:, 0], km[:, 1], xin_m).items()}
    out["top xi_n modes (m, n, amp mm, m*iota - n)"] = [[int(km[i, 0]), int(km[i, 1]), float(xin_m[i] * 1e3),
                                                        float(km[i, 0] * bt / bz - km[i, 1] * NFP)] for i in top]
    # --- shell profiles
    if desc_shell is None:
        from desc.grid import LinearGrid
        from desc.io import load
        eq = load(a.eq)
        eq = eq[-1] if hasattr(eq, "__len__") else eq
        desc_shell = []
        for r in d["shells"]:
            g = LinearGrid(rho=np.array([r]), M=2 * eq.M, N=2 * eq.N, NFP=eq.NFP)
            q = eq.compute(["|F|", "|J|"], grid=g)
            w = g.weights / g.weights.sum()
            desc_shell.append((float(np.sum(w * q["|F|"])), float(np.sqrt(np.sum(w * q["|J|"] ** 2)))))
        desc_shell = np.array(desc_shell)
        res["_shells"] = {"rho": d["shells"].tolist(), "DESC <|F|>": desc_shell[:, 0].tolist(), "DESC rms|J|": desc_shell[:, 1].tolist()}
    sw = d["shell_w"] / d["shell_w"].sum()
    Fs = d["F_shell"]
    out["M/M_desc"] = float(d["M"] / (d["F_desc_vol"] / d["norm"]))
    out["F_shell/DESC_shell"] = (Fs / desc_shell[:, 0]).tolist()
    out["dJ_shell/rmsJ"] = (d["dJ_shell_rms"] / d["rmsJ"]).tolist()
    out["M share from rho > 0.95"] = float(np.sum((sw * Fs)[d["shells"] > 0.95]) / np.sum(sw * Fs))
    out["M share from rho > 0.98"] = float(np.sum((sw * Fs)[d["shells"] > 0.98]) / np.sum(sw * Fs))
    res[nm] = out
    print(f"[{nm}] M/M_desc {out['M/M_desc']:.4f}; rms f/scale dir {out['rms f/scale (dir)']:.3e} ann {out['rms f/scale (ann)']:.3e};"
          f" xi_n rms {out['xi_n rms [mm]']:.3f} mm max {out['xi_n max [mm]']:.3f} mm (solve resid {resid:.1e});"
          f" M share rho>0.95 {out['M share from rho > 0.95']:.3f}, >0.98 {out['M share from rho > 0.98']:.3f}")
    print("    f by |n| (dir): " + ", ".join(f"{k} {v:.2e}" for k, v in out["f by n family /scale (dir)"].items()))
    print("    xi_n by |n| [mm]: " + ", ".join(f"{k} {v:.3f}" for k, v in out["xi_n by n family rms [mm]"].items()))
    print("    top xi modes: " + "; ".join(f"({m},{n}) {v:+.3f} mm [m i-n {r:+.2f}]" for m, n, v, r in out["top xi_n modes (m, n, amp mm, m*iota - n)"][:6]))
json.dump(res, open(a.out + "_char.json", "w"), indent=1)
print("saved", a.out + "_char.json")
