"""Convergence of a long force-only run: does the force level off, and do the fields stop moving?

From a run with --track --keep-iterates (optimize_qa.py), with its own model (the
fits, box, dense G, B_ext = B_DESC - B_plasma(x0)), at every iterate k:
  - <|F|>/norm (the run's history);
  - the change of the fields since the previous iterate, rms over the volume and
    relative to the start's rms: B, J along B, J across B, p;
  - the distance from the start and from the last iterate.
Converged: F levels off AND the change per iteration falls toward zero.
Sliding along a flat valley: F levels off while the fields keep changing at a steady rate.
Draws figures/gs_<run>_convergence.png.

run (GPU): sbatch run.sbatch scripts/gs_long_run.py --run results/gs/long_f2_a20
"""

import argparse
import json
import os
import sys

import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, "nest"))

from desc import set_device  # noqa: E402

set_device(os.environ.get("NEST_DEVICE", "cpu"))

import jax  # noqa: E402
import jax.numpy as jnp  # noqa: E402
import matplotlib  # noqa: E402

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

import fit_cache  # noqa: E402
from boundary_potential import BoundaryPotential  # noqa: E402
from dense import DenseFields  # noqa: E402
from desc_input import DESCInput  # noqa: E402
from boxes import make_box  # noqa: E402
from force import ForceResidual  # noqa: E402
from grid import CartesianGrid  # noqa: E402
from objective import Fields  # noqa: E402
from pressure import Pressure  # noqa: E402

jax.config.update("jax_enable_x64", True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", required=True)
    ap.add_argument("--ref", default=None, help="a run whose final state is the reference (distance per iterate)")
    a = ap.parse_args()
    info = json.load(open(a.run + "_iter.json"))
    args, hist = info["args"], info["history"]
    st = np.load(a.run + "_iter.npz")
    x0 = np.asarray(st["x0"])
    xs = np.concatenate([x0[None], np.asarray(st["xs"])])
    inp = DESCInput.load(args["eq"], field=args["coils"])
    eq = inp.eq
    ep = args.get("edge_power", 2)
    cur = fit_cache.fit(BoundaryPotential, inp, factors=tuple(args["factors"]),
                        **({"edge_power": ep} if ep != 2 else {}))
    pre = fit_cache.fit(Pressure, inp, factors=tuple(args["factors"]), edge_order=2)
    b = cur.basis
    res = tuple(int(round(args["colloc"] * (u + v))) for u, v in ((b.L, eq.L), (b.M, eq.M), (b.N, eq.N)))
    box = make_box(inp, args["frac"], args.get("grid", "cartesian"), bool(eq.sym), args.get("margin", 2.0))
    fr = ForceResidual(inp, box, resolution=res)
    coils = "B_ext" not in st.files and args.get("bext") != "desc"
    if not coils:  # replaced below; else the coils' field
        fr.B_ext = np.zeros((len(fr.rtz), 3))
    fields = DenseFields.from_fields(Fields(fr, cur, pre, kernel=args["kernel"],
                                            dtype=getattr(jnp, args.get("dtype", "float64"))), cur)
    fields.x0 = jnp.asarray(x0)
    if "B_ext" in st.files:
        fields.B_ext = jnp.asarray(st["B_ext"])
    elif not coils:
        fields.B_ext = fields.desc["B"] - fields.compute(fields.x0)["B_plasma"]
    w = np.asarray(fields.sw) ** 2
    comp = jax.jit(lambda x: fields.compute(x))

    def get(x):
        f = comp(jnp.asarray(x))
        return {k: np.asarray(f[k]) for k in ("J", "B", "p")}

    f0 = get(xs[0])
    bh = f0["B"] / np.linalg.norm(f0["B"], axis=1)[:, None]
    rms = {"B": np.sqrt(np.sum(w * np.sum(f0["B"] ** 2, 1))), "J": np.sqrt(np.sum(w * np.sum(f0["J"] ** 2, 1))),
           "p": np.sqrt(np.sum(w * f0["p"] ** 2))}

    def diff(X, Y):
        dJ = X["J"] - Y["J"]
        par = np.sum(dJ * bh, 1)
        return {"B": np.sqrt(np.sum(w * np.sum((X["B"] - Y["B"]) ** 2, 1))) / rms["B"],
                "J along": np.sqrt(np.sum(w * par**2)) / rms["J"],
                "J across": np.sqrt(np.sum(w * np.sum((dJ - par[:, None] * bh) ** 2, 1))) / rms["J"],
                "p": np.sqrt(np.sum(w * (X["p"] - Y["p"]) ** 2)) / rms["p"]}

    states = [get(x) for x in xs]
    step = [diff(states[k], states[k - 1]) for k in range(1, len(states))]
    from0 = [diff(s_, states[0]) for s_ in states]
    toend = [diff(s_, states[-1]) for s_ in states]
    ref_state = get(np.load(a.ref + "_iter.npz")["x"]) if a.ref else states[-1]
    toref = [diff(s_, ref_state) for s_ in states]
    F = [e["<|F|>/norm"] for e in hist]
    its = np.arange(len(states))
    qs = ("B", "J along", "J across", "p")
    print(f"{a.run}: {len(states) - 1} iterations")
    print(f"{'it':>6s} {'F':>10s} " + " ".join(f"{'step ' + q:>15s}" for q in qs) + " "
          + " ".join(f"{'from x0 ' + q:>17s}" for q in qs))
    for k in sorted({1, 10, 30, 100, 300, 600, 1000, len(states) - 1} & set(its[1:])):
        print(f"{k:6d} {F[min(k, len(F) - 1)]:10.3e} " + " ".join(f"{step[k - 1][q]:15.2e}" for q in qs)
              + " " + " ".join(f"{from0[k][q]:17.2e}" for q in qs))
    fig, ax = plt.subplots(1, 3, figsize=(17, 4.8))
    ax[0].loglog(np.arange(len(F))[1:], F[1:], "k")
    ax[0].set_title(r"$\langle|F|\rangle$/norm, run grid (spikes: correction passes)"
                    if args.get("correct_passes") else r"$\langle|F|\rangle$/norm, run grid")
    for q, mk in zip(qs, ("C0", "C1", "C2", "C3")):
        ax[1].loglog(its[1:], [s_[q] for s_ in step], mk, label=q)
        ax[2].loglog(its[1:], [f_[q] for f_ in from0[1:]], mk, label=f"{q}: from x0")
        ax[2].loglog(its[:-1] + 1, [t[q] for t in toend[:-1]], mk + ":", lw=0.8)
    ax[1].set_title("change of the fields per iteration (rms / start's rms)")
    ax[2].set_title("distance from the start (solid) and from the last iterate (dotted)")
    for x_ in ax:
        x_.set_xlabel("iteration")
    ax[1].legend(fontsize=8)
    ax[2].legend(fontsize=8)
    terms = ", ".join([f"w_{k} {args[f'w_{k}']:g}" + (f" at rho < {args['w_J_rho']:g}" if k == "J" and
                                                      args.get("w_J_rho") else "")
                       for k in ("J", "B", "p") if (args.get(f"w_{k}") or 0) > 0])
    field = ("coils + accurate B_plasma" if args.get("bplasma") == "axisym" else
             "B_ext = B_DESC - B_plasma(x0)" if args.get("bext") == "desc" else "coils, box B_plasma")
    fig.suptitle(f"{os.path.basename(a.run)}: force balance{' + ' + terms if terms else ' only'}; {field}; "
                 f"x{args['factors'][0]:g}, box a/{args['frac']}, float64")
    fig.tight_layout()
    out = os.path.join(ROOT, "figures", f"gs_{os.path.basename(a.run)}_convergence.png")
    fig.savefig(out, dpi=140)
    np.savez(out[:-4] + ".npz", F=np.array(F), **{f"step_{q}": np.array([s_[q] for s_ in step]) for q in qs},
             **{f"from0_{q}": np.array([f_[q] for f_ in from0]) for q in qs},
             **{f"toref_{q}": np.array([t[q] for t in toref]) for q in qs},
             f=np.array([e.get("f", np.nan) for e in hist]))
    print("saved", out)


if __name__ == "__main__":
    main()
