"""Which radial splines carry anything? Coefficient spectrum of a spline run by (spline j, poloidal m).

For each group of coefficients with the same spline index j and poloidal number m (all n together), the group's own
contribution to the field it builds, evaluated on a (rho, theta, zeta) grid with area weights rho:
  current a, b (spline T):  rms |J(u_group)| (I = 0), for u = J_fit's (the start) and for du = u_state - u_start
                            (what the optimizer changed)
  spline pressure:          rms |grad p(c_group)| for the state's Galerkin p*
The groups are placed at the centre of their spline's support in rho, and compared with rho^|m| (a smooth m-th
harmonic must be ~ rho^|m| near the axis): splines whose whole support lies where rho^|m| < tol are the candidates
for truncation. Writes <out>.npz, <out>.json (fractions of the rms^2 in those groups) and <out>.png (maps).

    python scripts/meanforce/spline_spectrum.py --run out/day/splTP24n8.json --state out/day/splTP24n8_a0.1.npz \
        --out out/day/splTP24n8_spectrum
"""

import argparse
import json
import os
import sys
import time

import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.join(ROOT, "nest"))
from desc import set_device  # noqa: E402

set_device(os.environ.get("NEST_DEVICE", "cpu"))
import jax  # noqa: E402
import jax.numpy as jnp  # noqa: E402

jax.config.update("jax_enable_x64", True)
import matplotlib  # noqa: E402

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

import fit_cache  # noqa: E402
from boundary_potential import BoundaryPotential, _current  # noqa: E402
from desc_input import DESCInput  # noqa: E402
from pressure import Pressure  # noqa: E402
from spline_pressure import SplinePressure, _p_and_grad, trunc_kw  # noqa: E402


def support_centre(t, j):
    """rho at the centre of B_j's support (knots in s = rho^2), and the support's outer end."""
    s0, s1 = t[j], t[j + 4]
    return np.sqrt(0.5 * (s0 + s1)), np.sqrt(s1)


def grid(n_r=40, n_t=48, n_z=17, NFP=1):
    r = (np.arange(n_r) + 0.5) / n_r
    th = 2 * np.pi * np.arange(n_t) / n_t
    ze = 2 * np.pi * np.arange(n_z) / (n_z * NFP)
    R, T, Z = np.meshgrid(r, th, ze, indexing="ij")
    return [jnp.asarray(x.ravel()) for x in (R, T, Z)], R.ravel()


def group_rms(eval_batch, n_coef, groups, coef, w, batch=32):
    """rms over the grid of each group's field (eval_batch: (B, n_coef) -> (B, n_pts, k))."""
    out = np.zeros(len(groups))
    for i in range(0, len(groups), batch):
        gs = groups[i: i + batch]
        U = np.zeros((len(gs), n_coef))
        for k, idx in enumerate(gs):
            U[k, idx] = coef[idx]
        V = np.asarray(eval_batch(jnp.asarray(U)))
        out[i: i + len(gs)] = np.sqrt(np.sum(w[None, :] * np.sum(V**2, -1), 1) / w.sum())
    return out


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--run", required=True)
    ap.add_argument("--state", required=True, help="npz of the deepest state (u, c_p)")
    ap.add_argument("--tol", type=float, nargs="+", default=[1e-3, 1e-2])
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    t0 = time.time()
    ra = json.load(open(a.run))["args"]
    inp = DESCInput.load(ra["eq"])
    eq = inp.eq
    L, M, N = ra["LMN"]
    cur = fit_cache.fit(BoundaryPotential, inp, M=M, N=N, radial="spline", knots=tuple(ra["j_knots"]),
                        helicities=tuple(ra["p_rationals"]),
                            **trunc_kw(ra.get("spl_mtol", 0.0), ra.get("spl_mmax"), ra.get("spl_mfz"),
                                      ra.get("knots_fz")))
    b = cur.basis
    t = np.asarray(b.t)
    st = np.load(a.state)
    u1, u0 = np.asarray(st["u"]), np.asarray(cur.u)
    (rho, th, ze), Rg = grid(NFP=eq.NFP)
    w = Rg  # area weight rho drho dtheta (uniform in rho, theta, zeta)
    nchi, K = cur.S.n_chi, cur.S.K
    jm = b.modes[:, :2]
    keys = sorted({(int(j), int(m)) for j, m in jm})
    rep = {"basis": b.summary(), "n_u": int(cur.n_free), "state": a.state}
    jf = jax.jit(jax.vmap(lambda u: _current(cur.S, cur._RZ, u, 0.0, rho, th, ze)))
    res = {"keys": np.asarray(keys)}
    for part, off in (("a", nchi), ("b", nchi + K)):
        groups = [off + np.flatnonzero((jm[:, 0] == j) & (jm[:, 1] == m)) for j, m in keys]
        res[f"{part}_fit"] = group_rms(jf, cur.n_free, groups, u0, w)
        res[f"{part}_change"] = group_rms(jf, cur.n_free, groups, u1 - u0, w)
    print(f"current groups done ({len(keys)} (j, m) per series), {time.time() - t0:.0f} s", flush=True)
    rc, rend = np.array([support_centre(t, j) for j, _ in keys]).T
    ms = np.array([abs(m) for _, m in keys])
    res["rho_centre"], res["rho_end"], res["absm"] = rc, rend, ms
    if ra.get("pressure") == "spline":
        pM, pN = ra["p_MN"] or (M, N)
        pre = fit_cache.fit(SplinePressure, inp, M=pM, N=pN, edge_order=2, coarse=ra["p_knots"][0],
                            dense=ra["p_knots"][1], half=ra["p_knots"][2], helicities=tuple(ra["p_rationals"]),
                            **trunc_kw(ra.get("spl_mtol", 0.0), ra.get("spl_mmax"), ra.get("spl_mfz"),
                                      ra.get("knots_fz")))
        cp = np.asarray(st["c_p"])
        pm = pre.modes[pre.mode_idx]
        pkeys = sorted({(int(j), int(m)) for j, m in zip(pre.j_idx, pm[:, 0])})
        pgroups = [np.flatnonzero((pre.j_idx == j) & (pm[:, 0] == m)) for j, m in pkeys]
        RZ = cur._RZ
        tt = tuple(float(v) for v in pre.t)
        pf = jax.jit(jax.vmap(lambda c: _p_and_grad(tt, pre.NFP, pre.edge_order, RZ, jnp.asarray(pre.modes),
                                                     jnp.asarray(pre.j_idx), jnp.asarray(pre.mode_idx), c,
                                                     rho, th, ze)[:, 1:]))
        res["p_keys"] = np.asarray(pkeys)
        res["p_gradp"] = group_rms(pf, pre.n, pgroups, cp, w)
        prc, prend = np.array([support_centre(np.asarray(pre.t), j) for j, _ in pkeys]).T
        res["p_rho_centre"], res["p_rho_end"], res["p_absm"] = prc, prend, np.array([m for _, m in pkeys])
        print(f"pressure groups done ({len(pkeys)}), {time.time() - t0:.0f} s", flush=True)

    def fractions(vals, rend_, absm, label):
        tot = float(np.sum(vals**2))
        out = {}
        for tol in a.tol:
            dead = (absm > 0) & (rend_ ** np.maximum(absm, 1) < tol)
            out[f"rho_end^|m| < {tol:g}"] = {"groups": int(dead.sum()), "of": int(len(vals)),
                                             "share of rms^2": float(np.sum(vals[dead] ** 2) / tot) if tot else 0.0}
        # share by |m| band
        for lo, hi in ((0, 2), (3, 6), (7, 12), (13, 99)):
            sel = (absm >= lo) & (absm <= hi)
            out[f"|m| {lo}-{hi}"] = float(np.sum(vals[sel] ** 2) / tot) if tot else 0.0
        rep[label] = out
        print(f"{label}: {json.dumps(out)}", flush=True)

    for part in ("a", "b"):
        fractions(res[f"{part}_fit"], rend, ms, f"{part} (J_fit)")
        fractions(res[f"{part}_change"], rend, ms, f"{part} (optimizer change)")
    if "p_gradp" in res:
        fractions(res["p_gradp"], res["p_rho_end"], res["p_absm"], "p (grad p of p*)")
    np.savez(a.out + ".npz", **res)
    with open(a.out + ".json", "w") as fh:
        json.dump(rep, fh, indent=1, default=float)

    # maps: x = m, y = rho at the support centre, colour = log10 of the group's rms (relative to the max)
    panels = [("a_fit", "a: J_fit"), ("b_fit", "b: J_fit"), ("a_change", "a: optimizer change"),
              ("b_change", "b: optimizer change")] + ([("p_gradp", "p*: grad p")] if "p_gradp" in res else [])
    fig, axs = plt.subplots(1, len(panels), figsize=(3.6 * len(panels), 3.8), sharey=True)
    mm = np.linspace(0.5, max(ms.max(), 1), 200)
    for ax, (k, title) in zip(axs, panels):
        v = res[k]
        x = np.array([m for _, m in (res["p_keys"] if k.startswith("p_") else keys)])
        y = res["p_rho_centre"] if k.startswith("p_") else rc
        lv = np.log10(np.maximum(v / v.max(), 1e-8))
        sc = ax.scatter(x, y, c=lv, s=10, cmap="viridis", vmin=-6, vmax=0, marker="s")
        for tol, ls in zip(a.tol, ("-", "--")):
            ax.plot(mm, tol ** (1 / mm), ls, color="#eb6834", lw=1, label=f"rho^|m| = {tol:g}")
            ax.plot(-mm, tol ** (1 / mm), ls, color="#eb6834", lw=1)
        ax.set_title(title, fontsize=9)
        ax.set_xlabel("m")
        ax.set_ylim(0, 1)
    axs[0].set_ylabel("rho at the spline's support centre")
    axs[0].legend(fontsize=7, frameon=False, loc="lower left")
    fig.colorbar(sc, ax=axs, shrink=0.8, label="log10 (group rms / max)")
    fig.savefig(a.out + ".png", dpi=140, bbox_inches="tight")
    print(f"wrote {a.out}.png/.json/.npz; {time.time() - t0:.0f} s")


if __name__ == "__main__":
    main()
