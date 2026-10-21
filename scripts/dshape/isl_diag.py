"""Is each state of an eps scan a minimum, and is the solver's gradient that of f?

For each saved x of results/isl/<name>_eps*.npz: g = A^T r (the solver's pass), then
  - directional derivatives by central differences of f (the trial pass) along a random
    direction in each block (u0, u1, p0, p1) and along -g, against g . d;
  - f along -D^2 g (the scaled gradient, D the solver's column scaling) for steps t over
    decades: the largest decrease a gradient step gets.

run (GPU): sbatch run.sbatch scripts/isl_diag.py --name n1
"""

import argparse
import json
import os
import sys

import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.join(ROOT, "nest"))

from desc import set_device  # noqa: E402

set_device(os.environ.get("NEST_DEVICE", "cpu"))

import jax  # noqa: E402
import jax.numpy as jnp  # noqa: E402

from harmonic_dense import HarmonicModel  # noqa: E402
from isl_scan import model_kwargs, setup  # noqa: E402
from perturbation import Perturbation  # noqa: E402

jax.config.update("jax_enable_x64", True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--name", required=True)
    ap.add_argument("--eps", type=float, nargs="*", default=None)
    a = ap.parse_args()
    summ = json.load(open(os.path.join(ROOT, "results", "isl", a.name + ".json")))
    sa = summ["args"]

    class A:
        pass
    for k, v in sa.items():
        setattr(A, k, v)
    inp, cur, pre, x_start, fr, run = setup(A)
    pert = Perturbation(inp.field, sa["m"], sa["n"])
    model = HarmonicModel(inp, cur, pre, fr, run["frac"], run.get("margin", 2.0), pert=pert,
                          chunk=sa.get("chunk", 2), batch_u=tuple(sa.get("batch", (4, 2))), **model_kwargs(A, run, cur, fr))
    model.set_source(x_start)
    rng = np.random.default_rng(0)
    out = {}
    for eps in (a.eps if a.eps else [r["eps"] for r in summ["runs"]]):
        x = np.asarray(np.load(os.path.join(ROOT, "results", "isl", f"{a.name}_eps{eps:g}.npz"))["x"])
        z = model.to_z(x)
        g, f, F = model._grad(z, eps)
        f0 = float(model._trials(z, jnp.zeros((model.n, 1)), eps)[0][0])
        res = {"f grad pass": f, "f trial pass": f0, "<|F|>/norm": F, "|g|": float(jnp.linalg.norm(g))}
        dirs = {}
        for b in model.blocks:
            d = np.zeros(model.n)
            sl = b["sl"]
            zb = np.asarray(z[sl])
            d[sl] = rng.standard_normal(sl.stop - sl.start) * (np.abs(zb).max() + 1e-30)
            dirs[f"{b['kind']}{b['n']}"] = d
        dirs["-g"] = -np.asarray(g)
        for lab, d in dirs.items():
            d = jnp.asarray(d)
            gd = float(g @ d)
            rows = []
            for t in 10.0 ** np.arange(-2, -11, -2):
                tt = t * float(jnp.linalg.norm(z)) / float(jnp.linalg.norm(d))
                fs = model._trials(z, jnp.stack([tt * d, -tt * d], 1), eps)[0]
                rows.append((t, (fs[0] - fs[1]) / (2 * tt), gd))
            res[f"dir {lab}"] = rows
            best = min(rows, key=lambda r: abs(r[1] - r[2]) / (abs(r[2]) + 1e-300))
            print(f"eps {eps:g} {lab:4s}: g.d {gd:+.6e}; central difference {best[1]:+.6e} (t {best[0]:.0e}), "
                  f"rel diff {abs(best[1] - best[2]) / (abs(best[2]) + 1e-300):.1e}", flush=True)
        # f along -g for steps over decades (absolute)
        line = []
        gn = -g * float(jnp.linalg.norm(z)) / float(jnp.linalg.norm(g))  # |gn| = |z|
        for t in 10.0 ** np.arange(-1, -16, -1):
            line.append((t, float(model._trials(z, (t * gn)[:, None], eps)[0][0])))
        res["line -g"] = line
        bt = min(line, key=lambda r: r[1])
        print(f"eps {eps:g}: f {f:.6e} (trial pass {f0:.6e}), |g| {res['|g|']:.2e}; best along -g: f "
              f"{bt[1]:.6e} at t {bt[0]:.0e} (rel decrease {(f0 - bt[1]) / f0:.1e})", flush=True)
        out[str(eps)] = res
    with open(os.path.join(ROOT, "results", "isl", a.name + "_diag.json"), "w") as fo:
        json.dump(out, fo, indent=1)


if __name__ == "__main__":
    main()
