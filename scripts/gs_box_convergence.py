"""How B_plasma on the box converges with its spacing h, at the DESC start x0.

For each box h = a / frac: rms |B_ext + B_plasma(x0) - B_DESC| / rms |B_DESC| and
<|F|> / norm at x0 on the same collocation grid (B_ext: the given field), <|F|>
with B_DESC in place of B (the fit's error alone), and the
change of B_plasma between successive boxes, from which the order p of
|B_h - B_h'| ~ h^p is estimated. Prints a table and saves --out (json).

run: sbatch run_cpu.sbatch scripts/gs_box_convergence.py [--fracs 8 10 14 20]
    [--factors 1 1 1]
"""

import argparse
import gc
import json
import os
import sys

import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, "nest"))

from desc import set_device  # noqa: E402

set_device(os.environ.get("NEST_DEVICE", "cpu"))

import jax  # noqa: E402

import fit_cache  # noqa: E402
from boundary_potential import BoundaryPotential  # noqa: E402
from desc_input import DESCInput  # noqa: E402
from force import ForceResidual  # noqa: E402
from grid import CartesianGrid  # noqa: E402
from objective import Fields  # noqa: E402
from pressure import Pressure  # noqa: E402

jax.config.update("jax_enable_x64", True)
TOK = os.path.join(ROOT, "desc_input", "tokamak_free_boundary", "tokamak_L16M16")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--eq", default=TOK + ".h5")
    ap.add_argument("--coils", default=TOK + "_field.h5")
    ap.add_argument("--fracs", type=int, nargs="+", default=[8, 10, 14, 20])
    ap.add_argument("--factors", type=float, nargs=3, default=(1, 1, 1))
    ap.add_argument("--kernel", default="truncated", choices=["hockney", "truncated"])
    ap.add_argument("--out", default=os.path.join(ROOT, "results", "gs", "box_convergence"))
    a = ap.parse_args()
    inp = DESCInput.load(a.eq, field=a.coils)
    eq = inp.eq
    cur = fit_cache.fit(BoundaryPotential, inp, factors=tuple(a.factors))
    pre = fit_cache.fit(Pressure, inp, factors=tuple(a.factors), edge_order=2)
    b = cur.basis
    res = tuple(int(round(2 * (u + v))) for u, v in ((b.L, eq.L), (b.M, eq.M), (b.N, eq.N)))
    rows, prev = [], None
    for frac in a.fracs:
        box = CartesianGrid.periodic_box(inp, inp.a / frac, odd=bool(eq.sym)).locate_plasma(inp)
        fr = ForceResidual(inp, box, resolution=res)  # the same points for every box
        fields = Fields(fr, cur, pre, kernel=a.kernel)
        f = {k: np.asarray(v) for k, v in fields.compute(fields.x0).items()}
        w = np.asarray(fields.sw) ** 2
        rms = lambda v: float(np.sqrt(np.sum(w * np.sum(v * v, 1))))  # noqa: E731
        row = {"frac": frac, "box": list(box.shape),
               "B err": rms(f["B"] - np.asarray(fields.desc["B"])) / fields.rms["B"],
               "<|F|>": float(np.sum(w * np.linalg.norm(f["F"], axis=1)) / fields.norm)}
        # the same with B = B_DESC: the fit's own error, without the box
        F_fit = np.cross(f["J"], np.asarray(fields.desc["B"])) - f["grad_p"]
        row["<|F|> with B_DESC"] = float(np.sum(w * np.linalg.norm(F_fit, axis=1)) / fields.norm)
        if prev is not None:
            row["|dB_plasma| vs previous"] = rms(f["B_plasma"] - prev) / fields.rms["B"]
        prev = f["B_plasma"]
        rows.append(row)
        print(json.dumps(row), flush=True)
        del fields, fr, box
        jax.clear_caches()
        gc.collect()
    print(f"potential factors {a.factors}; <|F|> with B_DESC (no box error): "
          f"{rows[0]['<|F|> with B_DESC']:.3e}")
    print("frac  box              B err      <|F|>     |B_h - B_prev|  order")
    for i, r in enumerate(rows):
        order = ""
        if i >= 2:  # B_h = B + C h^p: d_i = C (h_(i-1)^p - h_i^p); solve d_(i-1) / d_i for p
            d1, d2 = rows[i - 1]["|dB_plasma| vs previous"], r["|dB_plasma| vs previous"]
            h0, h1, h2 = (1 / rows[j]["frac"] for j in (i - 2, i - 1, i))
            ps = np.linspace(0.2, 12, 2000)
            g = (h0**ps - h1**ps) / (h1**ps - h2**ps) - d1 / d2
            k = np.flatnonzero(np.diff(np.sign(g)))
            order = f"{ps[k[0]]:.2f}" if len(k) else "n/a"
        print(f"a/{r['frac']:<3d} {str(r['box']):16s} {r['B err']:.3e}  {r['<|F|>']:.3e}  "
              f"{r.get('|dB_plasma| vs previous', float('nan')):.3e}      {order}")
    with open(a.out + ".json", "w") as fo:
        json.dump({"args": vars(a), "rows": rows}, fo, indent=1)
    print(f"saved {a.out}.json")


if __name__ == "__main__":
    main()
