"""DESC against NEST at the LCFS, down to 1 - rho = 1e-10: the current density and the
force on surfaces packed toward the edge, evaluated directly (no grid).

DESC: J, B, grad p of the equilibrium at the points (DESCInput.compute_points).
NEST: J and p from the final coefficients of each run (the point evaluators of
BoundaryPotential and Pressure), B = B_coil + B_plasma of that J: the exact axisymmetric
field (axisym_field.py, default), or with ``--bplasma cyl`` the run's own solver, the
cylindrical annulus at the run's h (Fourier in phi, Chebyshev in r and z), interpolated
to the points (CylindricalInterpolator).
Surface averages over theta weighted by |sqrt(g)|; <|J|> over max_rho <|J_DESC|>,
<|J x B - grad p|> over the force objective's norm.  Draws figures/<name>.png.

run (GPU): sbatch run.sbatch scripts/gs_edge_zoom.py --runs results/gs/cylD_a160 ...
"""

import argparse
import json
import os
import sys
import time

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
from axisym_field import plasma_field_cached  # noqa: E402
from boundary_potential import BoundaryPotential  # noqa: E402
from cartesian_grid import cylindrical_to_cartesian  # noqa: E402
from desc_input import DESCInput  # noqa: E402
from boxes import make_box  # noqa: E402
from force import ForceResidual  # noqa: E402
from objective import Fields  # noqa: E402
from pressure import Pressure  # noqa: E402

jax.config.update("jax_enable_x64", True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--runs", nargs="+", required=True)
    ap.add_argument("--labels", nargs="*", default=None)
    ap.add_argument("--n-theta", type=int, default=128)
    ap.add_argument("--decades", type=float, nargs=2, default=(-1.0, -10.0))
    ap.add_argument("--n-rho", type=int, default=37)
    ap.add_argument("--name", default="gs_edge_zoom")
    ap.add_argument("--bplasma", default="exact", choices=["exact", "cyl"])
    a = ap.parse_args()
    t0 = time.time()
    labels = a.labels or [os.path.basename(r) for r in a.runs]
    run0 = json.load(open(a.runs[0] + "_iter.json"))["args"]
    inp = DESCInput.load(run0["eq"], field=run0["coils"])
    eq = inp.eq
    ep = run0.get("edge_power", 2)
    cur = fit_cache.fit(BoundaryPotential, inp, factors=tuple(run0["factors"]),
                        **({"edge_power": ep} if ep != 2 else {}))
    pre = fit_cache.fit(Pressure, inp, factors=tuple(run0["factors"]), edge_order=2)
    # the force norm and max <|J_DESC|> from the run's own collocation grid
    b = cur.basis
    res = tuple(int(round(run0["colloc"] * (u + v))) for u, v in ((b.L, eq.L), (b.M, eq.M), (b.N, eq.N)))
    fr = ForceResidual(inp, None, resolution=res)
    norm = fr.norm
    dd = fr.desc_fields()
    r_all = np.asarray(fr.rtz)[:, 0]
    dV = np.asarray(fr.dV)
    JD = np.linalg.norm(np.asarray(dd["J"]), axis=1)
    Jmax = max(np.sum((dV * JD)[r_all == q]) / np.sum(dV[r_all == q]) for q in np.unique(r_all))
    # the edge points
    one_minus = np.logspace(a.decades[0], a.decades[1], a.n_rho)
    rho = 1.0 - one_minus
    th = 2 * np.pi * np.arange(a.n_theta) / a.n_theta
    R_, T_ = np.meshgrid(rho, th, indexing="ij")
    rtz = np.column_stack([R_.ravel(), T_.ravel(), np.zeros(R_.size)])
    X = np.asarray(cur._RZ.evaluate(jnp.asarray(rtz[:, 0]), jnp.asarray(rtz[:, 1]), jnp.asarray(rtz[:, 2])))
    Xr = np.asarray(cur._RZ.evaluate(jnp.asarray(rtz[:, 0]), jnp.asarray(rtz[:, 1]), jnp.asarray(rtz[:, 2]), dr=1))
    Xt = np.asarray(cur._RZ.evaluate(jnp.asarray(rtz[:, 0]), jnp.asarray(rtz[:, 1]), jnp.asarray(rtz[:, 2]), dt=1))
    jac = np.abs(X[:, 0] * (Xr[:, 0] * Xt[:, 1] - Xt[:, 0] * Xr[:, 1]))  # |sqrt(g)|
    phi = np.zeros(len(rtz))
    xc = np.column_stack([X[:, 0], np.zeros(len(rtz)), X[:, 1]])  # Cartesian at phi = 0
    B_coil = np.asarray(inp.B_ext(xc))

    def surf(v):
        v = (v * jac).reshape(len(rho), -1).sum(1) / jac.reshape(len(rho), -1).sum(1)
        return v

    # DESC
    d = inp.compute_points(["J", "B", "grad(p)"], rtz)
    J = np.asarray(cylindrical_to_cartesian(d["J"], phi))
    B = np.asarray(cylindrical_to_cartesian(d["B"], phi))
    gp = np.asarray(cylindrical_to_cartesian(d["grad(p)"], phi))
    curves = {"DESC": (surf(np.linalg.norm(J, axis=1)) / Jmax,
                       surf(np.linalg.norm(np.cross(J, B) - gp, axis=1)) / norm)}
    print(f"{inp}\nDESC done, {time.time() - t0:.0f} s", flush=True)
    pev = jax.jit(pre.point_evaluator(eq))
    states = []
    for run, lab in zip(a.runs, labels):
        states.append((f"NEST {lab}", np.asarray(np.load(run + "_iter.npz")["x"]),
                       json.load(open(run + "_iter.json"))["args"]))
    n_u = cur.n_free
    for lab, x, rargs in states:
        t1 = time.time()
        cu = cur.with_u(x[:n_u])
        Jn = np.asarray(cylindrical_to_cartesian(np.asarray(jax.jit(cu.point_evaluator(eq))(
            jnp.asarray(rtz[:, 0]), jnp.asarray(rtz[:, 1]), jnp.asarray(rtz[:, 2]))), phi))
        assert len(x) - n_u == len(pre.c), "pressure coefficients do not match the fit's basis"
        pr = Pressure(pre.basis, x[n_u:], eq, pre.edge_order)
        pg = np.asarray(jax.jit(pr.point_evaluator(eq))(jnp.asarray(rtz[:, 0]), jnp.asarray(rtz[:, 1]),
                                                         jnp.asarray(rtz[:, 2])))
        gpn = np.asarray(cylindrical_to_cartesian(pg[:, 1:], phi))
        if a.bplasma == "exact":
            Bpl = np.asarray(cylindrical_to_cartesian(plasma_field_cached(cur, rtz, u=x[:n_u]), phi))
        else:  # the run's own annulus solve, interpolated to the edge points
            from cylindrical_grid import CylindricalInterpolator
            box = make_box(inp, rargs["frac"], "cylindrical", bool(eq.sym), rargs.get("margin", 2.0))
            fl = Fields(ForceResidual(inp, box, resolution=res), cur, pre, kernel="cylindrical",
                        dtype=jnp.float64)
            Bp, Bz = fl.solver.field_modes(fl.J_box(jnp.asarray(x[:n_u])))
            interp = CylindricalInterpolator(box, np.column_stack([X[:, 0], phi, X[:, 1]]), dtype=jnp.float64)
            Bpl = np.asarray(cylindrical_to_cartesian(np.asarray(interp(Bp, Bz)), phi))
            del fl, box
        Fn = np.cross(Jn, B_coil + Bpl) - gpn
        curves[lab] = (surf(np.linalg.norm(Jn, axis=1)) / Jmax, surf(np.linalg.norm(Fn, axis=1)) / norm)
        print(f"{lab}: {time.time() - t1:.0f} s", flush=True)
    out = os.path.join(ROOT, "results", "gs", f"{a.name}.json")
    with open(out, "w") as fo:
        json.dump({"one_minus_rho": one_minus.tolist(), **{k: [v[0].tolist(), v[1].tolist()] for k, v in curves.items()}},
                  fo, indent=1)
    fig, ax = plt.subplots(1, 2, figsize=(14, 5.2))
    styles = {"DESC": dict(color="k", lw=2.2)}
    cols = iter(["C0", "C3", "C2", "C1", "C4"])
    for k, (Jc, Fc) in curves.items():
        st = styles.get(k) or dict(color=next(cols), lw=1.4)
        ax[0].loglog(one_minus, Jc, label=k, **st)
        ax[1].loglog(one_minus, Fc, label=k, **st)
    first = curves[states[0][0]][0]  # noqa: F841
    ax[0].loglog(one_minus, first[0] * (one_minus / one_minus[0]) ** 2, "k--", lw=0.7,
                 label=r"$\propto (1-\rho)^2$")
    ax[0].set_title(r"$\langle|J|\rangle_{surface}$ / max$\langle|J_{DESC}|\rangle$")
    ax[1].set_title(r"$\langle|J\times B-\nabla p|\rangle_{surface}$ / norm (NEST: coils + "
                    + (r"exact $B_{plasma}$)" if a.bplasma == "exact" else
                       r"$B_{plasma}$ of the run's annulus solve at its h)"))
    for x_ in ax:
        x_.set_xlabel(r"$1-\rho$")
        x_.legend(fontsize=8)
    fig.suptitle("DSHAPE L = M = 32 free boundary: DESC against NEST at the LCFS (direct evaluation, no grid)")
    fig.tight_layout()
    path = os.path.join(ROOT, "figures", f"{a.name}.png")
    fig.savefig(path, dpi=140)
    print(f"saved {path}; {out}; total {time.time() - t0:.0f} s")


if __name__ == "__main__":
    main()
