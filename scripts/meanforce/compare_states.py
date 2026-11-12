"""Field-line tracing and field-line-averaged profiles from state_tables.py output (CPU only; no NEST imports).

For each field (DESC, vacuum, each state) the field-line ODE d(rho, theta)/d zeta = (brho, bth) is integrated with RK4
on cubic splines of the tables (zeta as time, `--steps` per toroidal turn). Per line it records:
  * punctures (rho, theta* = theta + lambda_DESC) at zeta = 0 after every turn;
  * iota_line = (Delta theta + Delta lambda) / Delta zeta from the unwrapped DESC angle (exact rotation number of the
    line in DESC's straight-field-line angle);
  * field-line averages <f> = sum f / B^zeta / sum 1 / B^zeta over every RK step (dl / |B| = d zeta / B^zeta), i.e.
    flux-surface averages for lines that cover a surface; for p, lam = J.B/B^2, Jpar, Jperp, |F|, |F_par|, |F_rad|, |B|;
  * an effective flux label rho_eff = sqrt(mean of rho^2 over the punctures). In DESC's field the toroidal flux enclosed
    by a curve rho(theta*) at fixed zeta is Psi_b <rho^2>_theta*, so rho_eff^2 = enclosed flux / Psi_b for surfaces
    whose punctures are uniform in theta*; NEST's field differs from DESC's by |dB|/|B| ~ 1e-3.
  * radial spread max - min rho (island-trapped lines: iota_line locked at a rational with a large spread).

usage: python scripts/meanforce/compare_states.py --tables DIR --fields DESC vacuum S b1 --out DIR/compare
"""
import argparse
import json
import os
import time

import numpy as np
from scipy.ndimage import map_coordinates, spline_filter

ap = argparse.ArgumentParser()
ap.add_argument("--tables", required=True)
ap.add_argument("--fields", nargs="+", default=["DESC", "vacuum", "S", "b1"])
ap.add_argument("--seeds", type=float, nargs=3, default=(0.04, 0.975, 150), help="rho range and count at theta = 0")
ap.add_argument("--xseeds", type=float, nargs=3, default=(0.50, 0.78, 40), help="extra seeds at theta = pi/2 (X-points)")
ap.add_argument("--turns", type=int, default=400)
ap.add_argument("--steps", type=int, default=64)
ap.add_argument("--out", required=True)
a = ap.parse_args()
os.makedirs(os.path.dirname(a.out) or ".", exist_ok=True)
T0 = time.time()
G = np.load(os.path.join(a.tables, "grid.npz"))
rho, th, ze, NFP = G["rho"], G["theta"], G["zeta"], int(G["NFP"])
P = 3
dr, dth = rho[1] - rho[0], th[1] - th[0]
dze = ze[1] - ze[0]


def prep(f):
    f = np.asarray(f, np.float64)
    g = np.concatenate([f[:, -P:], f, f[:, :P]], axis=1)
    g = np.concatenate([g[:, :, -P:], g, g[:, :, :P]], axis=2)
    return spline_filter(g, order=3, mode="nearest")


def sample(F, r, t, z):
    z = np.broadcast_to(z, np.shape(r))
    q = np.stack([(r - rho[0]) / dr, np.mod(t, 2 * np.pi) / dth + P, np.mod(z, 2 * np.pi / NFP) / dze + P])
    return map_coordinates(F, q, order=3, prefilter=False, mode="nearest")


Flam = prep(G["lam_desc"])
QTY = ["p", "lam", "Jpar", "Jperp", "Fmag", "Fpar", "Frad", "Bmag"]
seeds_r = np.concatenate([np.linspace(*a.seeds[:2], int(a.seeds[2])), np.linspace(*a.xseeds[:2], int(a.xseeds[2]))])
seeds_t = np.concatenate([np.zeros(int(a.seeds[2])), np.full(int(a.xseeds[2]), np.pi / 2)])
# theta = 0 / pi/2 are DESC angles; convert the X-point seeds to theta* = pi/2 approximately by one fixed-point step
out = {}
for name in a.fields:
    T = np.load(os.path.join(a.tables, f"{name}.npz"))
    Fr, Ft, Fz = prep(T["brho"]), prep(T["bth"]), prep(T["Bz"])
    have = [q for q in QTY if q in T.files]
    if name == "vacuum":  # carry DESC's p along the vacuum lines (no flattening by construction)
        D = np.load(os.path.join(a.tables, "DESC.npz"))
        Fq = {"p": prep(D["p"])}
    else:
        Fq = {q: prep(np.abs(T[q]) if q in ("Fpar", "Frad") else T[q]) for q in have}
    r, t = seeds_r.copy(), seeds_t - sample(Flam, seeds_r, seeds_t, np.zeros_like(seeds_r)) * (seeds_t > 0)
    t0 = t.copy()
    lam0 = sample(Flam, r, t, np.zeros_like(r))
    h = 2 * np.pi / (NFP * a.steps)
    alive = np.ones_like(r, bool)
    acc = {q: np.zeros_like(r) for q in Fq}
    wsum = np.zeros_like(r)
    pr, pt = [], []
    z = 0.0

    def rhs(r, t, z):
        return sample(Fr, r, t, z), sample(Ft, r, t, z)

    for turn in range(a.turns):
        for _ in range(a.steps):
            w = 1.0 / sample(Fz, r, t, z)
            for q, Fq_ in Fq.items():
                acc[q] += np.where(alive, w * sample(Fq_, r, t, z), 0.0)
            wsum += np.where(alive, w, 0.0)
            k1 = rhs(r, t, z)
            k2 = rhs(r + h / 2 * k1[0], t + h / 2 * k1[1], z + h / 2)
            k3 = rhs(r + h / 2 * k2[0], t + h / 2 * k2[1], z + h / 2)
            k4 = rhs(r + h * k3[0], t + h * k3[1], z + h)
            r = r + h / 6 * (k1[0] + 2 * k2[0] + 2 * k3[0] + k4[0])
            t = t + h / 6 * (k1[1] + 2 * k2[1] + 2 * k3[1] + k4[1])
            z += h
        alive &= (r > rho[0]) & (r < rho[-1])
        pr.append(np.where(alive, r, np.nan))
        pt.append(np.where(alive, np.mod(t + sample(Flam, r, t, np.zeros_like(r)), 2 * np.pi), np.nan))
    lam1 = sample(Flam, r, t, np.zeros_like(r))
    iota_line = np.where(alive, (t - t0 + lam1 - lam0) / (2 * np.pi * a.turns / NFP), np.nan)
    PR = np.array(pr)
    res = {"seeds_rho": seeds_r, "seeds_theta": seeds_t, "punct_rho": PR.astype(np.float32),
           "punct_theta_star": np.array(pt).astype(np.float32), "iota": iota_line,
           "rho_eff": np.sqrt(np.nanmean(PR**2, 0)), "spread": np.nanmax(PR, 0) - np.nanmin(PR, 0), "alive": alive}
    for q in Fq:
        res["avg_" + q] = np.where(alive, acc[q] / np.maximum(wsum, 1e-300), np.nan)
    np.savez(a.out + f"_{name}.npz", **res)
    out[name] = res
    print(f"[{time.time() - T0:7.1f} s] {name}: {alive.sum()}/{len(alive)} lines alive; max spread "
          f"{np.nanmax(res['spread']):.4f}", flush=True)
print("done")
