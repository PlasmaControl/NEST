"""Conditioning of HarmonicModel's LM problem with p free (joint) vs p eliminated (variable projection).

The force rows are affine in p with a constant A_p, so min_p |r(u, p)| is the weak (weighted) Poisson
equation of force balance, and the reduced problem's R is the trailing block of the QR of
R[:, p | u | r] (= the Schur complement of the joint Gram matrix). Reported, for the laptop model of
scripts/lm/lm_ab.py (L16 a/20 N = 1) at its eps 1000 stage: kappa of the joint, reduced and u-only
problems, Jacobi-scaled and whitened by the physical metric (`speed phys`), for w_J 1e-3 and 1e-7 and
for richer pressure bases (--p-factors).

run: PYTHONPATH=nest:$DESC JAX_PLATFORMS=cpu python scripts/diagnostics/pressure_elim_conditioning.py \
         [--p-factors 1.5,1.5,1 2,2,1] [--out results/lm/pressure_elim_conditioning.json]
(needs results/lm/L16a20_full_stage_eps1000_wJ0.001.npz from lm_ab.py)
"""
import argparse
import json
import os
import sys

import numpy as np
import scipy.linalg as sl

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path[:0] = [os.path.join(ROOT, "nest"), os.path.join(ROOT, "scripts", "dshape"), os.path.join(ROOT, "scripts", "lm")]
sys.argv, _argv = sys.argv[:1], sys.argv
import lm_ab  # noqa: E402

sys.argv = _argv
import jax.numpy as jnp  # noqa: E402

import fit_cache  # noqa: E402
from harmonic_dense import HarmonicModel, embed  # noqa: E402
from pressure import Pressure  # noqa: E402


def kappa(M, scale=True):
    if scale:
        cn = np.linalg.norm(M, axis=0)
        M = M / np.where(cn > 0, cn, 1.0)
    s = np.linalg.svd(M, compute_uv=False)
    return float(s[0] / s[-1])


def analyse(model, z, eps):
    R = np.asarray(model._tsqr(jnp.asarray(z), eps))
    n = model.n
    ip = np.concatenate([np.arange(b["sl"].start, b["sl"].stop) for b in model.blocks if b["kind"] == "p"])
    iu = np.concatenate([np.arange(b["sl"].start, b["sl"].stop) for b in model.blocks if b["kind"] == "u"])
    L = np.asarray(model._phys_L())
    assert np.allclose(L[np.ix_(iu, ip)], 0) and np.allclose(L[np.ix_(ip, iu)], 0), "metric not block diagonal"
    Luu, Lpp = L[np.ix_(iu, iu)], L[np.ix_(ip, ip)]
    _, R2 = np.linalg.qr(R[:, np.r_[ip, iu, [n]]])
    npp, nu = len(ip), len(iu)
    Tu = R2[npp:npp + nu, npp:npp + nu]
    white = lambda M, Lw: sl.solve_triangular(Lw, M.T, lower=True).T  # noqa: E731  M L^-T
    c, cu = R[:n, n], R2[npp:npp + nu, -1]
    f = 0.5 * float(c @ c + R[n, n] ** 2)
    return {"n_u": nu, "n_p": npp, "f": f,
            "joint_jacobi": kappa(R[:, np.r_[iu, ip]]), "reduced_jacobi": kappa(Tu),
            "joint_phys": kappa(white(R[:, np.r_[iu, ip]], sl.block_diag(Luu, Lpp)), False),
            "reduced_phys": kappa(white(Tu, Luu), False), "u_only_phys": kappa(white(R[:, iu], Luu), False),
            "gn_share_joint": float(c @ c) / (2 * f), "gn_share_reduced": float(cu @ cu) / (2 * f)}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--p-factors", nargs="*", default=["1.5,1.5,1", "2,2,1"])
    ap.add_argument("--out", default=os.path.join(ROOT, "results", "lm", "pressure_elim_conditioning.json"))
    ap.add_argument("--eq", default=None)
    ap.add_argument("--factors", type=float, nargs=3, default=None)
    ap.add_argument("--frac", type=int, default=None)
    a = ap.parse_args()
    lm_ab.configure(a.eq, a.factors, a.frac)
    stage = np.load(os.path.join(ROOT, "results", "lm", lm_ab.prefix("full") + "_stage_eps1000_wJ0.001.npz"))["x"]
    out = {}

    def report(tag, r):
        out[tag] = r
        print(f"[{tag}] n_u {r['n_u']} n_p {r['n_p']} f {r['f']:.3e} | kappa Jacobi joint {r['joint_jacobi']:.2e} "
              f"reduced {r['reduced_jacobi']:.2e} | phys joint {r['joint_phys']:.2e} reduced {r['reduced_phys']:.2e} "
              f"u-only {r['u_only_phys']:.2e} | GN share {r['gn_share_joint']:.2e} / {r['gn_share_reduced']:.2e}", flush=True)

    for wJ in (1e-3, 1e-7):
        m, c = lm_ab.build(wJ, "full")
        report(f"wJ{wJ:g}", analyse(m, m.to_z(stage), 1000.0))
    cur, pre1 = c["cur"], c["pre"]
    for pf in a.p_factors:
        fac = tuple(float(v) for v in pf.split(","))
        fac = tuple(f * g for f, g in zip(fac, lm_ab.CONFIG["factors"]))  # relative to the current factors
        pre = fit_cache.fit(Pressure, c["inp"], factors=fac, N=1, edge_order=2)
        m = HarmonicModel(c["inp"], cur, pre, c["fr"], c["run"]["frac"], 2.0, pert=c["pert"],
                          weights={"J": 1e-7, "J_rho": None, "W": 0.0, "pos": 0.0}, j_nodes="flux",
                          j_mode="full", j_target=c["jt"], verbose=False)
        m.set_source(c["x0"])
        report(f"wJ1e-07 p factors {pf}", analyse(m, m.to_z(embed(stage, cur, pre1, cur, pre)), 1000.0))
    json.dump(out, open(a.out, "w"), indent=1)
    print(f"-> {a.out}")


if __name__ == "__main__":
    main()
