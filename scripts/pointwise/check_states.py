"""States of several runs evaluated on one reference resolution (the finest annulus).

Runs with the same basis share the collocation points, so their states can be evaluated
with the reference tables' G (B_plasma from the reference annulus): the force each state
really has, independent of the grid it was optimized on. Also the J, B and p profiles
(averages over theta and zeta at each collocation radius) for the convergence plots.

    python scripts/pointwise/check_states.py --ref results/pointwise/tables/a80_f15.npz \
        --run qr_a20=results/pointwise/runs/qr_a20_f15.json ... --out results/pointwise/check_f15
"""

import argparse
import glob
import json
import os
import sys

import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.join(ROOT, "nest"))
from desc import set_device  # noqa: E402

set_device(os.environ.get("NEST_DEVICE", "cpu"))
import jax  # noqa: E402

jax.config.update("jax_enable_x64", True)
import jax.numpy as jnp  # noqa: E402

import pointwise as pw  # noqa: E402
import pointwise_tables  # noqa: E402


def profiles(T, base):
    """Averages over (theta, zeta) at each radius of the tensor grid: |J|, J along B,
    |B|, p and |F| / norm (weights dV)."""
    shape = tuple(int(s) for s in T["tgrid_shape"])
    w = np.asarray(T["w"]).reshape(shape)
    J, B, F, p = (np.asarray(base[k]) for k in ("J", "B", "F", "p"))
    Bn = np.linalg.norm(B, axis=1)
    q = {"|J|": np.linalg.norm(J, axis=1), "J.b": np.sum(J * B, 1) / Bn, "|B|": Bn, "p": p,
         "|F|": np.linalg.norm(F, axis=1) / float(T["norm"])}
    out = {k: np.sum(v.reshape(shape) * w, (1, 2)) / np.sum(w, (1, 2)) for k, v in q.items()}
    out["|F| max"] = np.max(q["|F|"].reshape(shape), (1, 2))
    out["rho"] = np.asarray(T["rtz"])[:, 0].reshape(shape)[:, 0, 0]
    return out


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--ref", required=True)
    ap.add_argument("--run", action="append", required=True, help="label=run.json")
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    T = pointwise_tables.load(a.ref)
    model = pw.DenseModel(T)
    x0 = jnp.concatenate([jnp.asarray(T["u0"]), jnp.asarray(T["c0"])])
    base0 = model.base(x0)
    res = {"ref": a.ref, "runs": {}}
    arrays = {"start": profiles(T, base0), "w": np.asarray(T["w"]), "norm": float(T["norm"])}
    for spec in a.run:
        label, paths = spec.split("=", 1)
        res["runs"][label] = {}
        for path in paths.split("+"):  # a run and its continuations
            if not os.path.exists(path):
                continue
            run = json.load(open(path))
            stem = path[: -len(".json")]
            for e in run["alphas"]:
                al = e["metrics"]["alpha"]
                f = f"{stem}_a{al:g}.npz"
                if not e["metrics"]["feasible"] or not os.path.exists(f):
                    continue
                st = np.load(f)
                x = jnp.asarray(st["x"])
                b = model.base(x)
                m, _ = pw.state_metrics(T, b, x, st["lam"], float(st["eps"]), base0)
                m["own max|F|"] = e["metrics"]["max|F|"]
                res["runs"][label][f"{al:g}"] = m
                for k, v in profiles(T, b).items():
                    arrays[f"{label}|{al:g}|{k}"] = v
                arrays[f"{label}|{al:g}|x"] = np.asarray(x)
                for k in ("J", "p", "F"):  # the fields at the points (shared by runs of one basis)
                    arrays[f"{label}|{al:g}|{k}pt"] = np.asarray(b[k])
        print(label, {k: f"{v['max|F|']:.3e} (own {v['own max|F|']:.3e})"
                      for k, v in res["runs"][label].items()}, flush=True)
    w_, n_ = arrays.pop("w"), arrays.pop("norm")
    arrays["weights"], arrays["normalization"] = w_, n_
    for k, v in arrays.pop("start").items():
        arrays[f"start|{k}"] = v
    np.savez(a.out + ".npz", **arrays)
    with open(a.out + ".json", "w") as fh:
        json.dump(res, fh, indent=1, default=float)
    print(f"saved {a.out}.json/.npz")


if __name__ == "__main__":
    main()
