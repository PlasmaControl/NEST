"""Hessian at a converged pointwise-bound state: is it a local minimum?

For a state of ``al_run.py`` (x, lam, rho, eps) and its tables, the exact Hessians
(``pointwise.hessians``) in the physical metric (eigenvalues of L^-1 H L^-T, L L^T = M):

    H_AL   of the augmented Lagrangian the inner solver minimizes (= H_Lag + rho sum_active
           w_i grad c_i grad c_i^T): positive definite at a strict local minimum of the
           subproblem;
    GN_AL  its Gauss-Newton part J_r^T J_r alone;
    H_Lag  of the Lagrangian Phi + sum_i lam_i w_i c_i, reduced to the tangent space of the
           active constraints (the null space of their gradients): positive semidefinite
           there is the second-order condition of the constrained minimum.

Also, for the smallest eigenvalues of H_AL, how much of the eigenvector is pressure.

    python scripts/pointwise/hessian.py --tables T.npz --state X_a0.1.npz --out X_a0.1_hess
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

jax.config.update("jax_enable_x64", True)

import pointwise as pw  # noqa: E402
import pointwise_tables  # noqa: E402


def spectrum(H, Linv):
    S = Linv @ H @ Linv.T
    S = 0.5 * (S + S.T)
    return np.linalg.eigh(S)


def summary(ev):
    top = float(np.max(np.abs(ev)))
    return {"max": float(ev.max()), "min": float(ev.min()), "n": int(len(ev)),
            "negative (< -1e-10 max)": int(np.sum(ev < -1e-10 * top)),
            "near zero (|.| <= 1e-10 max)": int(np.sum(np.abs(ev) <= 1e-10 * top)),
            "below 1e-6 max": int(np.sum(ev < 1e-6 * top)),
            "min positive / max": float(ev[ev > 1e-10 * top].min() / top) if np.any(ev > 1e-10 * top) else 0.0}


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--tables", required=True)
    ap.add_argument("--state", required=True)
    ap.add_argument("--delta", type=float, default=1e-6, help="the run's final smoothing")
    ap.add_argument("--lam-tol", type=float, default=1e-6, help="active: lam_i > this x max lam")
    ap.add_argument("--c-tol", type=float, default=1e-3, help="active: c_i > -this (on the bound)")
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    t0 = time.time()
    T = pointwise_tables.load(a.tables)
    st = np.load(a.state)
    x, lam, rho, eps = st["x"], st["lam"], float(st["rho"]), float(st["eps"])
    n_u = T["A"].shape[0]
    L, HJ = pw.metric(T)
    Linv = np.linalg.inv(L)
    H = pw.hessians(T, x, lam, rho, eps, float(T["norm"]), HJ, a.delta, a.lam_tol, a.c_tol)
    out = {"state": a.state, "eps": eps, "rho": rho, "n_u": n_u, "n": int(len(x)),
           "active points": int(np.sum(H["active"]))}
    arrays = {}
    for k in ("H_AL", "GN_AL", "H_Lag"):
        ev, V = spectrum(H[k], Linv)
        arrays[k] = ev
        out[k] = summary(ev)
        if k == "H_AL":  # pressure share of the softest eigenvectors (in the metric variables)
            ps = np.sum(V[n_u:, :20] ** 2, 0)
            out["H_AL softest 20: pressure share"] = [float(v) for v in ps]
            out["H_AL softest 20: eigenvalue / max"] = [float(v) for v in ev[:20] / np.abs(ev).max()]
    # the tangent space of the active constraints: null space of sum_active w grad c grad c^T
    eg, Vg = spectrum(H["grad_active"], Linv)
    tol = 1e-10 * max(eg.max(), 1e-300)
    Z = Vg[:, eg <= tol]
    out["tangent space dimension"] = int(Z.shape[1])
    if Z.shape[1]:
        S = Linv @ H["H_Lag"] @ Linv.T
        Hr = Z.T @ (0.5 * (S + S.T)) @ Z
        er = np.linalg.eigvalsh(Hr)
        arrays["H_Lag_reduced"] = er
        out["H_Lag reduced"] = summary(er)
    np.savez(a.out + ".npz", **arrays)
    with open(a.out + ".json", "w") as fh:
        json.dump(out, fh, indent=1)
    print(json.dumps(out, indent=1))
    print(f"{time.time() - t0:.0f} s")


if __name__ == "__main__":
    main()
