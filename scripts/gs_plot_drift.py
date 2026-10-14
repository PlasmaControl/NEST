"""Figures of a force-only run along the iterations (optimize_qa.py --track --keep-iterates).

Rebuilds the run's model (the same fits, box and precision, the
start x0 of the run and B_ext = B_DESC - B_plasma(x0)), and draws

<out>_history.png
  a. <|F|> / norm against the iteration, DESC's for reference;
  b. <|F|> / norm by rho at a few iterations, and DESC's;
  c. the change of the coefficients, |u - u0| / |u0|, |c_p - c_p0| / |c_p0|;
  d. the change of the fields: rms |B - B_DESC| / rms B, |<p> / <p0> - 1|,
     |rms J / rms J0 - 1|, how far p is from a flux function and how much B
     crosses DESC's surfaces (``optimize_qa.drift``);
  e. Levenberg-Marquardt: lambda, the step length, |d| / |x|;
  f. which modes moved at the end: |x - x0| summed over l for each |m|, for the
     current potential and the pressure.
<out>_fields.png
  the change of J and p at the end against the start, J(x) - J(x0) split along
  and across B_DESC, and p(x) - p(x0), in the cross-section phi = 0 and as rms
  over each flux surface at a few iterations (relative to the volume rms of J0, p0).

usage: python scripts/gs_plot_drift.py --run results/gs/fdrift_L24_f2 [--eq ...]
    [--coils ...] (the run's --factors, --frac, --colloc, --kernel are read from its json)
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
from cartesian_grid import evaluate_chunked  # noqa: E402
from desc_input import DESCInput  # noqa: E402
from force import ForceResidual  # noqa: E402
from grid import CartesianGrid  # noqa: E402
from objective import Fields  # noqa: E402
from precondition import mode_labels  # noqa: E402
from pressure import Pressure  # noqa: E402

jax.config.update("jax_enable_x64", True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", required=True, help="the run's --out (reads _iter.json, _iter.npz)")
    ap.add_argument("--out", default=None, help="figure prefix (default figures/<run name>)")
    a = ap.parse_args()
    info = json.load(open(a.run + "_iter.json"))
    args, hist = info["args"], info["history"]
    st = np.load(a.run + "_iter.npz")
    x0 = np.asarray(st["x0"])
    xs = np.concatenate([x0[None], np.asarray(st["xs"])])  # iterate k = 0 .. K
    its = np.arange(len(xs))
    out = a.out or os.path.join(ROOT, "figures", "gs_" + os.path.basename(a.run))

    inp = DESCInput.load(args["eq"], field=args["coils"])
    eq = inp.eq
    cur = fit_cache.fit(BoundaryPotential, inp, factors=tuple(args["factors"]))
    pre = fit_cache.fit(Pressure, inp, factors=tuple(args["factors"]), edge_order=2)
    b = cur.basis
    res = tuple(int(round(args["colloc"] * (u + v)))
                for u, v in ((b.L, eq.L), (b.M, eq.M), (b.N, eq.N)))
    box = CartesianGrid.periodic_box(inp, inp.a / args["frac"], odd=bool(eq.sym)).locate_plasma(inp)
    fr = ForceResidual(inp, box, resolution=res)
    # plain box solves: only a few iterates are evaluated (no dense G needed)
    fields = Fields(fr, cur, pre, kernel=args["kernel"],
                    dtype=getattr(jnp, args.get("dtype", "float32")))
    fields.x0 = jnp.asarray(x0)
    if args.get("bext") == "desc":
        fields.B_ext = fields.desc["B"] - fields.compute(fields.x0)["B_plasma"]
    desc = {k: np.asarray(v) for k, v in fields.desc.items()}
    F_desc = fr.metrics(np.cross(desc["J"], desc["B"]) - desc["grad(p)"])
    n_u = fields.n_u

    # ---- history figure
    drift = [e["drift"] for e in hist if "drift" in e]
    it_d = [e["it"] for e in hist if "drift" in e]
    fig, ax = plt.subplots(2, 3, figsize=(17, 9.5))
    ax = ax.ravel()
    ax[0].semilogy(it_d, [d["<|F|>"] for d in drift], "k", label="NEST")
    ax[0].axhline(F_desc["mean"], color="C3", ls="--", label="DESC")
    ax[0].set_title(r"a. $\langle|F|\rangle$ / norm")
    ax[0].legend()
    sel = sorted({0, 5, 20, len(xs) - 1} & set(its))
    rb = list(F_desc["bins"])
    centers = [np.mean([float(v) for v in k.split("-")]) for k in rb]
    for k in sel:
        m = fr.metrics(np.asarray(fields.compute(jnp.asarray(xs[k]))["F"]))
        ax[1].semilogy(centers, list(m["bins"].values()), "o-", label=f"it {k}")
    ax[1].semilogy(centers, list(F_desc["bins"].values()), "k--", label="DESC")
    ax[1].set_xlabel(r"$\rho$ (bin centres: " + ", ".join(rb) + ")", fontsize=8)
    ax[1].set_title(r"b. $\langle|F|\rangle$ / norm by $\rho$")
    ax[1].legend(fontsize=8)
    ax[2].semilogy(it_d, [d["du"] for d in drift], label=r"$|u-u_0|/|u_0|$")
    ax[2].semilogy(it_d, [d["dc_p"] for d in drift], label=r"$|c_p-c_{p0}|/|c_{p0}|$")
    ax[2].set_title("c. change of the coefficients")
    ax[2].legend()
    for k, lab, f in (("B-B_DESC", r"rms $|B-B_{DESC}|$ / rms B", lambda v: v),
                      ("<p>", r"$|\langle p\rangle/\langle p_0\rangle-1|$", lambda v: abs(v - 1)),
                      ("rms J", r"$|$rms J / rms J$_0 - 1|$", lambda v: abs(v - 1)),
                      ("B.grad p", r"$\langle|B\cdot\nabla p|\rangle/\langle|B||\nabla p|\rangle$",
                       lambda v: v),
                      ("B.grad rho", r"$\langle|B\cdot\nabla\rho|\rangle/\langle|B||\nabla\rho|\rangle$",
                       lambda v: v)):
        ax[3].semilogy(it_d, [max(f(d[k]), 1e-18) for d in drift], label=lab)
    ax[3].set_title("d. change of the fields")
    ax[3].legend(fontsize=8)
    ih = [e["it"] for e in hist]
    ax[4].semilogy(ih, [e["lam"] for e in hist], label=r"$\lambda$")
    ax[4].semilogy(ih, [max(e["step"], 1e-18) for e in hist], label="step length")
    ax[4].semilogy(ih, [e.get("|d|/|x|", np.nan) for e in hist], label=r"$|d|/|x|$")
    ax[4].set_title("e. Levenberg-Marquardt")
    ax[4].legend(fontsize=8)
    lab = mode_labels(fields, cur, pre)  # (kind, l, m, n)
    dx = np.abs(xs[-1] - xs[0])
    for kind, name in ((0, "current potential (chi, a, b)"), (1, "pressure")):
        sel_k = lab[:, 0] == kind
        ms = np.unique(np.abs(lab[sel_k, 2]))
        rel = [np.linalg.norm(dx[sel_k & (np.abs(lab[:, 2]) == m)]) for m in ms]
        ref = np.linalg.norm(xs[0][sel_k])
        ax[5].semilogy(ms, np.maximum(np.array(rel) / ref, 1e-18), "o-", label=name)
    ax[5].set_xlabel("|m|")
    ax[5].set_title(r"f. $|x_{end}-x_0|$ by poloidal mode (over $|x_0|$ of its kind)")
    ax[5].legend(fontsize=8)
    for i in (0, 2, 3, 4):
        ax[i].set_xlabel("iteration")
    fig.suptitle(f"{os.path.basename(a.run)}: force balance only from the DESC-matched start "
                 f"({inp.name}, potential L, M = {b.L}, {b.M}, box a/{args['frac']})")
    fig.tight_layout()
    fig.savefig(out + "_history.png", dpi=130)
    print("saved", out + "_history.png", flush=True)

    # ---- fields figure: J and p of an iterate on a (rho, theta) grid at phi = 0
    rho = np.unique(np.concatenate([np.linspace(0.01, 0.95, 80), 1 - np.logspace(np.log10(0.05), -3, 30)]))
    theta = np.linspace(0, 2 * np.pi, 160, endpoint=False)
    rr, tt = np.meshgrid(rho, theta, indexing="ij")
    rtz = np.column_stack([rr.ravel(), tt.ravel(), np.zeros(rr.size)])
    dD = inp.compute_tensor(["B", "R", "Z", "sqrt(g)"], rho, theta, np.array([0.0]))
    bhat = dD["B"] / np.linalg.norm(dD["B"], axis=1)[:, None]
    shape = rr.shape
    sg = np.abs(dD["sqrt(g)"]).reshape(shape)

    def JP(x):
        c = BoundaryPotential(b, np.asarray(x[:n_u]), cur.I, eq, cur.edge_power, cur.S)
        J = np.asarray(evaluate_chunked(c.point_evaluator(eq), rtz, 50000))
        P = Pressure(pre.basis, np.asarray(x[n_u:]), eq, pre.edge_order)
        p = np.asarray(evaluate_chunked(P.point_evaluator(eq), rtz, 50000))[:, 0]
        return J, p

    J0, p0 = JP(xs[0])
    rmsJ = np.sqrt(np.sum(sg * np.sum(J0**2, 1).reshape(shape)) / np.sum(sg))
    rmsp = np.sqrt(np.sum(sg * (p0**2).reshape(shape)) / np.sum(sg))
    fs = lambda v: np.sqrt(np.sum(sg * v.reshape(shape) ** 2, 1) / np.sum(sg, 1))  # noqa: E731

    def parts(x):
        J, p = JP(x)
        dJ = J - J0
        par = np.sum(dJ * bhat, 1)
        perp = np.linalg.norm(dJ - par[:, None] * bhat, axis=1)
        return par / rmsJ, perp / rmsJ, (p - p0) / rmsp

    fig, ax = plt.subplots(2, 3, figsize=(17, 10))
    R, Z = dD["R"].reshape(shape), dD["Z"].reshape(shape)
    close = lambda v: np.concatenate([v, v[:, :1]], 1)  # noqa: E731
    end = parts(xs[-1])
    for i, (v, name) in enumerate(zip(end, (r"$(J-J_0)\cdot\hat b$", r"$|(J-J_0)_\perp|$",
                                             r"$p-p_0$"))):
        L10 = np.log10(np.abs(v).reshape(shape) + 1e-30)
        hi = np.ceil(L10.max())
        im = ax[0, i].contourf(close(R), close(Z), close(L10), np.linspace(hi - 6, hi, 25),
                               cmap="magma", extend="min")
        plt.colorbar(im, ax=ax[0, i], label=r"log$_{10}$ |.| / rms (" + ("J$_0$" if i < 2 else "p$_0$") + ")")
        ax[0, i].set_aspect("equal")
        ax[0, i].set_title(f"{name} at iteration {len(xs) - 1}")
    for k in sorted({1, 5, 20, len(xs) - 1} & set(its)):
        pk = parts(xs[k]) if k != len(xs) - 1 else end
        for i in range(3):
            ax[1, i].semilogy(rho, np.maximum(fs(pk[i]), 1e-18), label=f"it {k}")
    for i, name in enumerate(("along B", "across B", "pressure")):
        ax[1, i].set_xlabel(r"$\rho$")
        ax[1, i].set_title(f"change {name}: rms on each flux surface")
        ax[1, i].legend(fontsize=8)
    fig.suptitle(f"{os.path.basename(a.run)}: J(x) - J(x0) and p(x) - p(x0); "
                 f"rms J0 = {rmsJ / 1e3:.1f} kA/m², rms p0 = {rmsp:.0f} Pa")
    fig.tight_layout()
    fig.savefig(out + "_fields.png", dpi=130)
    print("saved", out + "_fields.png")


if __name__ == "__main__":
    main()
