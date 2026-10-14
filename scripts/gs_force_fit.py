"""Force-matched start: F_NEST(x0) - F_DESC for the T fit and the force fits.

With B(x0) = B_DESC (``--bext desc``) and the pressure fit exact,
F_NEST(x0) - F_DESC = (J(u) - J_DESC) x B_DESC on the points. For the fit of T
(``BoundaryPotential.fit``) and the force-matched fits (``dense.force_matched_u``)
at each eps: <|F_NEST - F_DESC|> / norm overall and by rho on NEST's collocation
grid, the same on a finer check grid (--check x the resolution of F; the fit
cannot overfit there), <|F_NEST|> / norm, DESC's <|F|> / norm, and rms |J -
J_DESC| / rms J_DESC. No box: a minute or two on CPU.

run: sbatch --time=00:30:00 run_cpu.sbatch scripts/gs_force_fit.py [--eq ...]
    [--factors 2 2 1] [--eps 1 1e-1 1e-2 1e-3 1e-4] [--save u.npz]
"""

import argparse
import os
import sys
import time

import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, "nest"))

from desc import set_device  # noqa: E402

set_device(os.environ.get("NEST_DEVICE", "cpu"))

import jax  # noqa: E402

import fit_cache  # noqa: E402
from boundary_potential import BoundaryPotential  # noqa: E402
from cartesian_grid import cylindrical_to_cartesian  # noqa: E402
from dense import J_columns, force_matched_u  # noqa: E402
from desc_input import DESCInput  # noqa: E402
from force import ForceResidual  # noqa: E402
from pressure import Pressure  # noqa: E402

jax.config.update("jax_enable_x64", True)
TOK = os.path.join(ROOT, "desc_input", "tokamak_free_boundary", "tokamak_L24M24")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--eq", default=TOK + ".h5")
    ap.add_argument("--factors", type=float, nargs=3, default=(2, 2, 1))
    ap.add_argument("--colloc", type=float, default=2)
    ap.add_argument("--check", type=float, default=3)
    ap.add_argument("--eps", type=float, nargs="+", default=[1, 1e-1, 1e-2, 1e-3, 1e-4])
    ap.add_argument("--save", default=None, help="npz of the force-matched u for each eps")
    a = ap.parse_args()
    t0 = time.time()
    inp = DESCInput.load(a.eq)
    eq = inp.eq
    cur = fit_cache.fit(BoundaryPotential, inp, factors=tuple(a.factors))
    pre = fit_cache.fit(Pressure, inp, factors=tuple(a.factors), edge_order=2)
    b = cur.basis
    res = lambda f: tuple(int(round(f * (u + v))) for u, v in ((b.L, eq.L), (b.M, eq.M), (b.N, eq.N)))  # noqa: E731
    grids = {"collocation": ForceResidual(inp, None, resolution=res(a.colloc)),
             "check": ForceResidual(inp, None, resolution=res(a.check))}
    data = {}
    for k, fr in grids.items():
        desc = {q: np.asarray(v) for q, v in fr.desc_fields().items()}
        pg = fr._series(pre.point_evaluator(eq), fr.rtz)
        gp = np.asarray(cylindrical_to_cartesian(pg[:, 1:], fr.phi))
        data[k] = (fr, desc, J_columns(fr, cur), gp)
    fr = grids["collocation"]
    print(f"{inp}\npotential L, M, N = {b.L}, {b.M}, {b.N} ({cur.n_free} unknowns); collocation "
          f"{fr.tgrid.shape}, check {grids['check'].tgrid.shape}; setup {time.time() - t0:.0f} s\n")
    fits = {"T fit": np.asarray(cur.u)}
    for eps in a.eps:
        t1 = time.time()
        u, _ = force_matched_u(fr, cur, eps=eps, desc=data["collocation"][1],
                               cols=data["collocation"][2])
        fits[f"force fit, eps = {eps:g}"] = u
        print(f"  force fit eps = {eps:g}: {time.time() - t1:.1f} s", flush=True)

    bins = list(fr.metrics(np.zeros((len(fr.rtz), 3)))["bins"])
    print(f"\n{'<|F_NEST - F_DESC|> / norm':30s} {'all':>9s} " + " ".join(f"{k:>9s}" for k in bins)
          + f" {'check all':>10s} {'F_NEST':>9s} {'J err':>9s}")
    rows = {}
    for name, u in fits.items():
        out = {}
        for k, (g, desc, (A, J_I), gp) in data.items():
            J = np.einsum("i,ipc->pc", u, A) + J_I
            w = g.dV / g.dV.sum()
            out[k] = {"diff": g.metrics(np.cross(J - desc["J"], desc["B"])),
                      "F": g.metrics(np.cross(J, desc["B"]) - gp)["mean"],
                      "J": float(np.sqrt(np.sum(w * np.sum((J - desc["J"]) ** 2, 1))
                                         / np.sum(w * np.sum(desc["J"] ** 2, 1))))}
        m = out["collocation"]
        rows[name] = out
        print(f"{name:30s} {m['diff']['mean']:9.2e} "
              + " ".join(f"{v:9.2e}" for v in m["diff"]["bins"].values())
              + f" {out['check']['diff']['mean']:10.2e} {m['F']:9.2e} {m['J']:9.2e}")
    dF = fr.metrics(np.cross(data["collocation"][1]["J"], data["collocation"][1]["B"])
                    - data["collocation"][1]["grad(p)"])
    print(f"{'DESC <|F|> / norm (scale)':30s} {dF['mean']:9.2e} "
          + " ".join(f"{v:9.2e}" for v in dF["bins"].values()))
    if a.save:
        np.savez(a.save, **{k.replace(" ", "_").replace(",", ""): v for k, v in fits.items()})
        print(f"saved {a.save}")
    print(f"total {time.time() - t0:.0f} s")


if __name__ == "__main__":
    main()
