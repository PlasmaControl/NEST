"""A/B test of HarmonicModel LM step options on a small (laptop) model with the production objective.

The small model: DSHAPE-like tokamak_L16M16_fb (the collaborator's profiles), basis factors 1, a/20,
N = 1 (750 unknowns). Objective as the production island runs started from hN_a120D: force balance plus
w_J (J - J_start) on the FULL 3-D J at every collocation point and every radius (j_mode full, J_rho None),
J_start = J of the axisymmetric fit; no <p> or p >= 0 terms (w_W = w_pos = 0).
(isl_scan's fresh start has no j_target, so there --j-mode full silently falls back to the toroidal
average for rho < 0.9; this script builds j_target as isl_scan.model_kwargs does for a base.)

Stages (each cached in results/lm/<tag>_*.npz):
  base   eps 0,   w_J = wJ_base, exact LM to convergence      (cf. results/gs/hN_a120D)
  stage  eps,     w_J = wJ_base, exact LM to convergence      (cf. a120Ce<eps>s3)
  A/B    eps,     w_J = wJ,      K iterations per variant from the stage state

run: PYTHONPATH=nest:$DESC JAX_PLATFORMS=cpu python scripts/lm/lm_ab.py --wJ 1e-7 --eps 1000 --iters 20 \
         --variants base phys chol,narrow phys,chol,narrow
"""
import argparse
import json
import os
import sys
import time
import types

import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.join(ROOT, "nest"))
sys.path.insert(0, os.path.join(ROOT, "scripts", "dshape"))
sys.argv, _argv = sys.argv[:1], sys.argv  # keep isl_scan's import free of our arguments
import isl_scan  # noqa: E402

sys.argv = _argv
import jax  # noqa: E402

jax.config.update("jax_enable_x64", True)
import jax.numpy as jnp  # noqa: E402

from flux_interp import TensorCurrent, TensorGrid  # noqa: E402
from harmonic_dense import HarmonicModel  # noqa: E402
from perturbation import Perturbation  # noqa: E402

TOK = os.path.join(ROOT, "desc_input", "tokamak_free_boundary")
OUT = os.path.join(ROOT, "results", "lm")


def build(wJ, j_mode, setup_cache={}, p_factors=None):  # noqa: B006
    if not setup_cache:
        a = types.SimpleNamespace(base=None, eq=os.path.join(TOK, "tokamak_L16M16_fb.h5"),
                                  coils=os.path.join(TOK, "tokamak_L16M16_field.h5"), factors=(1.0, 1.0, 1.0),
                                  edge_power=3, frac=20, N=1)
        inp, cur, pre, x0, fr, run = isl_scan.setup(a)
        g = TensorGrid(np.asarray(fr.tgrid.rho), np.asarray(fr.tgrid.theta), np.array([0.0]), inp.eq.NFP)
        tc = TensorCurrent(cur.S, cur._RZ, g, factored=True)
        jt = np.asarray(tc(jnp.asarray(x0[: cur.n_free]), cur.I))[:, :, 0]
        setup_cache.update(inp=inp, cur=cur, pre=pre, x0=x0, fr=fr, run=run, jt=jt,
                           pert=Perturbation(inp.field, 2, -1))
    c = setup_cache
    pre = c["pre"]
    if p_factors is not None:  # a richer pressure space (states of the default one are embedded: ``embed_p``)
        import fit_cache
        from pressure import Pressure
        pre = fit_cache.fit(Pressure, c["inp"], factors=tuple(p_factors), N=1, edge_order=2)
    c["pre_model"] = pre
    model = HarmonicModel(c["inp"], c["cur"], pre, c["fr"], c["run"]["frac"], 2.0, pert=c["pert"],
                          weights={"J": wJ, "J_rho": None, "W": 0.0, "pos": 0.0}, j_nodes="flux",
                          j_mode=j_mode, j_target=c["jt"], verbose=True)
    model.set_source(c["x0"])
    return model, c


def converge(model, x, eps, tag, iters=40):
    path = os.path.join(OUT, tag + ".npz")
    if os.path.exists(path):
        print(f"[{tag}] cached", flush=True)
        return np.load(path)["x"]
    x, hist = model.lm(x, eps, iters=iters, rtol=1e-10, gn_tol=1e-9)
    np.savez(path, x=x)
    json.dump(hist, open(os.path.join(OUT, tag + ".json"), "w"), indent=1)
    return x


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--wJ", type=float, default=1e-7)
    ap.add_argument("--wJ-base", type=float, default=1e-3)
    ap.add_argument("--eps", type=float, default=1000.0)
    ap.add_argument("--iters", type=int, default=20)
    ap.add_argument("--j-mode", default="full", choices=["full", "n0"])
    ap.add_argument("--variants", nargs="+", default=["base", "phys"],
                    help="comma-separated speed options per variant; 'base' = none")
    ap.add_argument("--tag", default="ab")
    ap.add_argument("--p-factors", type=float, nargs=3, default=None,
                    help="pressure basis factors of the A/B model (base and stage use the default 1 1 1)")
    ap.add_argument("--start", default="stage", choices=["stage", "base"],
                    help="A/B start: the eps stage state (default) or the eps = 0 base (solves the stage)")
    a = ap.parse_args()
    os.makedirs(OUT, exist_ok=True)
    t0 = time.time()
    pre = f"L16a20_{a.j_mode}"
    m0, c = build(a.wJ_base, a.j_mode)
    xb = converge(m0, c["x0"], 0.0, f"{pre}_base_wJ{a.wJ_base:g}")
    xs = converge(m0, xb, a.eps, f"{pre}_stage_eps{a.eps:g}_wJ{a.wJ_base:g}")
    del m0
    m, c = build(a.wJ, a.j_mode, p_factors=a.p_factors)
    if a.p_factors is not None:
        from harmonic_dense import embed
        xb, xs = (embed(x_, c["cur"], c["pre"], c["cur"], c["pre_model"]) for x_ in (xb, xs))
    res = {"args": vars(a), "runs": {}}
    for v in a.variants:
        sp = {} if v == "base" else {k: (int(val) if val.isdigit() else True)
                                     for k, _, val in (t.partition("=") for t in v.split(","))}
        m.speed = sp
        m._Lm = None
        print(f"\n=== variant {v}: speed {sp}", flush=True)
        t1 = time.time()
        x, hist = m.lm(xs if a.start == "stage" else xb, a.eps, iters=a.iters, rtol=1e-12, gn_tol=1e-12)
        res["runs"][v] = {"hist": hist, "time": time.time() - t1}
        np.savez(os.path.join(OUT, f"{a.tag}_{pre}_eps{a.eps:g}_wJ{a.wJ:g}_{v.replace(',', '+')}"
                              + ("" if a.start == "stage" else "_frombase")
                              + ("" if a.p_factors is None else "_p" + "-".join(f"{q:g}" for q in a.p_factors)) + ".npz"), x=x)
        print(f"=== {v}: f/f0 {hist[-1]['f'] / hist[0]['f']:.4f} in {len(hist) - 1} its, {time.time() - t1:.0f} s",
              flush=True)
    out = os.path.join(OUT, f"{a.tag}_{pre}_eps{a.eps:g}_wJ{a.wJ:g}" + ("" if a.start == "stage" else "_frombase")
                       + ("" if a.p_factors is None else "_p" + "-".join(f"{v:g}" for v in a.p_factors)) + ".json")
    json.dump(res, open(out, "w"), indent=1, default=float)
    print(f"\nsummary ({time.time() - t0:.0f} s): " + "  ".join(
        f"{v} {r['hist'][-1]['f'] / r['hist'][0]['f']:.4f}" for v, r in res["runs"].items()) + f"\n-> {out}")


if __name__ == "__main__":
    main()
