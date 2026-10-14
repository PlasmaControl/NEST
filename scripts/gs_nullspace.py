"""Rank and null space of the force-balance Jacobian (axisymmetric tokamak).

For the free-boundary tokamak of ``desc_input/tokamak_free_boundary`` (or any
equilibrium and field), forms the whole Jacobian dr/dx of the chosen objectives
at the start x0 (DESC's fits) with the current and the pressure free, and

  1. its singular values after column scaling (each column by its norm): the
     numerical rank at relative tolerances 1e-4 ... 1e-12, and the gaps;
  2. the right singular vectors of the smallest singular values, split into
     - gauge directions: dJ = 0 and dp = 0 (no change of the fields at all),
     - physical directions: the fields change but F does not, to first order;
     for each: |dJ| / |J0|, |dp| / |p0|, the parallel fraction of dJ
     (|dJ.B|^2 / |dJ|^2 |B|^2), how far dp is from a flux function
     (<|B.grad dp|> / <|B| |grad dp|>), the change of <p>, and |dF| / norm per
     unit relative change of the fields (dF/amp, amp = max(|dJ|/|J0|, |dp|/|p0|)).
     The net toroidal current is not an unknown (the potential's I), so J = 0 is
     out of reach: the trivial end state is p = 0 with a force-free J;
  3. the direction of the trivial solution (x -> 0): the share of x0 (scaled)
     in the near-null space.

Saves the singular values, the null vectors and the mode labels to --out (npz).

run (GPU): sbatch run.sbatch scripts/gs_nullspace.py [--LM 24 24] [--pLM 24 24]
    [--frac 10] [--colloc 2] [--w-J 0] ... [--out results/gs_null_L24]
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

import fit_cache  # noqa: E402
from boundary_potential import BoundaryPotential  # noqa: E402
from desc_input import DESCInput  # noqa: E402
from force import ForceResidual  # noqa: E402
from grid import CartesianGrid  # noqa: E402
from objective import (CurrentError, FieldError, Fields, ForceBalance,  # noqa: E402
                       ObjectiveFunction, PressureEnergy, PressureError,
                       PressurePositivity, PressureSquare, _cartesian)
from precondition import mode_labels  # noqa: E402
from pressure import Pressure  # noqa: E402
from dense import DenseFields  # noqa: E402

jax.config.update("jax_enable_x64", True)
TOK = os.path.join(ROOT, "desc_input", "tokamak_free_boundary")
EQ = os.path.join(TOK, "tokamak_L16M16.h5")
FIELD = os.path.join(TOK, "tokamak_L16M16_field.h5")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--eq", default=EQ)
    ap.add_argument("--field", default=FIELD)
    ap.add_argument("--LM", type=int, nargs=2, default=(24, 24), help="potential L, M")
    ap.add_argument("--pLM", type=int, nargs=2, default=None, help="pressure L, M (default LM)")
    ap.add_argument("--frac", type=int, default=10)
    ap.add_argument("--kernel", default="truncated", choices=["hockney", "truncated"])
    ap.add_argument("--colloc", type=float, default=2, help="/ resolution of F")
    ap.add_argument("--colloc-N", type=int, default=None,
                    help="toroidal resolution of the collocation grid (default: that of F, "
                         "0 in axisymmetry: one plane)")
    for k in ("force", "B", "J", "p", "W", "p2", "pos"):
        ap.add_argument(f"--w-{k}", type=float, default=1.0 if k == "force" else 0.0)
    ap.add_argument("--n-null", type=int, default=40, help="smallest singular vectors analysed")
    ap.add_argument("--run", default=None,
                    help="an optimize_qa.py --out: its model (fits, box, dtype, --bext, dense G "
                         "from the cache) and its start or end point (--at); --LM, --frac, ... ignored")
    ap.add_argument("--at", default="start", choices=["start", "end"])
    ap.add_argument("--out", default=os.path.join(ROOT, "results", "gs_null"))
    a = ap.parse_args()
    t0 = time.time()
    run = None
    if a.run:  # the run's model and point
        run = json.load(open(a.run + "_iter.json"))["args"]
        a.eq, a.field, a.frac, a.kernel, a.colloc = (run[k] for k in ("eq", "coils", "frac",
                                                                        "kernel", "colloc"))
        st = np.load(a.run + "_iter.npz")
        x_run = {"start": np.asarray(st["x0"]), "end": np.asarray(st["x"])}[a.at]
    inp = DESCInput.load(a.eq, field=a.field)
    eq = inp.eq
    if run:
        cur = fit_cache.fit(BoundaryPotential, inp, factors=tuple(run["factors"]))
        pre = fit_cache.fit(Pressure, inp, factors=tuple(run["factors"]), edge_order=2)
        L, M, pL, pM = cur.basis.L, cur.basis.M, pre.basis.L, pre.basis.M
    else:
        L, M = a.LM
        pL, pM = a.pLM or a.LM
        cur = fit_cache.fit(BoundaryPotential, inp, L=L, M=M, N=0)
        pre = fit_cache.fit(Pressure, inp, L=pL, M=pM, N=0, edge_order=2)
    box = CartesianGrid.periodic_box(inp, inp.a / a.frac, odd=True).locate_plasma(inp)
    b = cur.basis
    res = [int(round(a.colloc * (u + v))) for u, v in ((b.L, eq.L), (b.M, eq.M), (b.N, eq.N))]
    if a.colloc_N is not None:
        res[2] = a.colloc_N
    fr = ForceResidual(inp, box, resolution=tuple(res))
    fields = Fields(fr, cur, pre, kernel=a.kernel,
                    dtype=getattr(jnp, run.get("dtype", "float32")) if run else jnp.float64)
    if run:
        if run.get("dense"):
            fields = DenseFields.from_fields(fields, cur)
        st0 = jnp.asarray(np.load(a.run + "_iter.npz")["x0"])
        fields.x0 = st0  # the run's start (e.g. the force-matched fit)
        if run.get("bext") == "desc":
            fields.B_ext = fields.desc["B"] - fields.compute(st0)["B_plasma"]
    classes = {"force": ForceBalance, "B": FieldError, "J": CurrentError, "p": PressureError,
               "W": PressureEnergy, "p2": PressureSquare, "pos": PressurePositivity}
    weights = {k: getattr(a, f"w_{k}") for k in classes}
    obj = ObjectiveFunction([classes[k](w) for k, w in weights.items() if w > 0], fields).build()
    n, n_u = obj.dim_x, fields.n_u
    labels = mode_labels(fields, cur, pre)
    print(f"{inp}\npotential L, M = {L}, {M} ({n_u} unknowns), pressure {pL}, {pM} "
          f"({n - n_u}); collocation {fr.tgrid.shape}, {obj.dim_f:,} residuals; box "
          f"{box.shape}; objectives {weights}; setup {time.time() - t0:.0f} s", flush=True)

    x0 = jnp.asarray(x_run) if run else obj.x0  # the point analysed
    f0 = fields.compute(x0)
    w_vol = np.asarray(fields.sw) ** 2
    dB = np.asarray(f0["B"]) - np.asarray(fields.desc["B"])
    dB = np.sqrt(np.sum(w_vol * np.sum(dB * dB, 1))) / fields.rms["B"]
    print(f"{'run ' + a.run + ', ' + a.at if run else 'x0'}: <|F|>/norm "
          f"{obj.report(x0)['<|F|>/norm']:.3e} (DESC "
          f"{fr.F_desc_vol / fr.norm:.3e}); rms |B_ext + B_plasma(x0) - B_DESC| / rms |B_DESC| "
          f"{dB:.2e}", flush=True)

    # the whole Jacobian, in batches of J v
    t1 = time.time()
    r, Jv, _ = obj.linearize(x0)
    jv = jax.jit(jax.vmap(Jv))
    A = np.empty((obj.dim_f, n))
    for i in range(0, n, 32):  # fixed batch of 32 (one compile)
        E = np.zeros((32, n))
        k = min(32, n - i)
        E[np.arange(k), i + np.arange(k)] = 1.0
        A[:, i : i + k] = np.asarray(jv(jnp.asarray(E))).T[:, :k]
    print(f"Jacobian {A.shape} in {time.time() - t1:.0f} s", flush=True)
    s_col = np.linalg.norm(A, axis=0)
    zero_cols = np.flatnonzero(s_col == 0)
    s_col[s_col == 0] = 1.0
    # SVD through QR (the singular values to full precision, unlike J^T J; R is n x n)
    t1 = time.time()
    _, sv, Vt = np.linalg.svd(np.linalg.qr(A / s_col, mode="r"))
    sv_raw = np.linalg.svd(np.linalg.qr(A, mode="r"), compute_uv=False)
    print(f"QR + SVD in {time.time() - t1:.0f} s", flush=True)
    print(f"1. singular values of the column-scaled Jacobian: max {sv[0]:.3e}, min {sv[-1]:.3e}; "
          f"{len(zero_cols)} zero columns")
    for tol in (1e-2, 1e-4, 1e-6, 1e-8, 1e-10, 1e-12):
        print(f"   rank at {tol:.0e}: {np.sum(sv > tol * sv[0]):6d} / {n}   "
              f"(unscaled: {np.sum(sv_raw > tol * sv_raw[0]):6d})")
    lg = np.log10(sv / sv[0])
    gaps = np.argsort(np.diff(lg))[:5]  # the largest drops between neighbours
    print("   largest gaps (index: sigma_i -> sigma_i+1, relative): " + "; ".join(
        f"{i}: {sv[i] / sv[0]:.1e} -> {sv[i + 1] / sv[0]:.1e}" for i in sorted(gaps)))

    # 2. the smallest singular directions in the original unknowns
    J0, B0, p0 = (np.asarray(f0[k]) for k in ("J", "B", "p"))
    gp0 = np.asarray(f0["grad_p"])
    B2 = np.sum(B0 * B0, 1)
    rmsJ = np.sqrt(np.sum(w_vol * np.sum(J0 * J0, 1)))
    rmsp = np.sqrt(np.sum(w_vol * p0 * p0))
    avg = lambda q: float(np.sum(w_vol * q))  # noqa: E731
    # the net toroidal current through phi = const: I = int J_phi dA, from the
    # volume average of J_phi / R ~ V <J_phi / R> / 2 pi
    R = np.asarray(fields.force.geo.R).ravel()
    J_phi = lambda J: -J[:, 0] * np.sin(fr.phi) + J[:, 1] * np.cos(fr.phi)  # noqa: E731
    I0 = avg(J_phi(J0) / R)
    tang = jax.jit(lambda v: fields._tangent(v, f0["J"], f0["B"]))

    def describe(v):
        d = {k: np.asarray(x) for k, x in tang(jnp.asarray(v)).items()}
        dJ, dp, dgp = d["J"], d["p"], d["grad_p"]
        nJ = np.sqrt(avg(np.sum(dJ * dJ, 1)))
        par = np.sum(dJ * B0, 1) ** 2 / B2
        Bgp = np.abs(np.sum(B0 * dgp, 1))
        ngp = np.sqrt(np.sum(dgp * dgp, 1) * B2)
        return {"dJ/J": nJ / rmsJ, "dp/p": np.sqrt(avg(dp * dp)) / rmsp,
                "par": avg(par) / max(avg(np.sum(dJ * dJ, 1)), 1e-300),
                "B.grad dp": avg(Bgp) / max(avg(ngp), 1e-300),
                "dI/I": avg(J_phi(dJ) / R) / I0, "d<p>/<p>": avg(dp) / avg(p0),
                "dF/amp": np.sqrt(avg(np.sum(d["F"] ** 2, 1))) / fr.norm
                / max(nJ / rmsJ, np.sqrt(avg(dp * dp)) / rmsp),
                "u share": float(np.sum(v[:n_u] ** 2) / np.sum(v**2))}

    k = min(a.n_null, n)
    V = (Vt[-k:][::-1] / s_col).T  # columns: unknowns of the k smallest, smallest first
    V /= np.linalg.norm(V, axis=0)
    print(f"2. the {k} smallest singular directions (unit in x): sigma / sigma_max, "
          "|dJ|/|J0|, |dp|/|p0|, parallel fraction of dJ, <|B.grad dp|>/<|B||grad dp|>, "
          "d<p>/<p>, |dF|/norm / amp, share of u, mean l, |m| of u and of p")
    rows = []
    for i in range(k):
        v = V[:, i]
        d = describe(v)
        wu, wp = v[:n_u] ** 2, v[n_u:] ** 2
        lu = labels[:n_u][:, 1:3]
        lp = labels[n_u:][:, 1:3]
        mm = lambda w, lab: (w @ np.abs(lab)) / max(w.sum(), 1e-300)  # noqa: E731
        d.update(sigma=sv[n - 1 - i] / sv[0], lm_u=mm(wu, lu), lm_p=mm(wp, lp))
        rows.append(d)
        print(f"   {i:3d} {d['sigma']:.2e}  J {d['dJ/J']:.1e}  p {d['dp/p']:.1e}  par "
              f"{d['par']:.2f}  Bgp {d['B.grad dp']:.2f}  dpm {d['d<p>/<p>']:+.1e}  "
              f"dF/amp {d['dF/amp']:.1e}  u {d['u share']:.2f}  "
              f"l,m u {d['lm_u'][0]:4.1f},{d['lm_u'][1]:4.1f}  p {d['lm_p'][0]:4.1f},"
              f"{d['lm_p'][1]:4.1f}", flush=True)

    # 3. the trivial solution: x0 in the scaled coordinates, against the near-null space
    y0 = np.asarray(x0) * s_col
    y0 /= np.linalg.norm(y0)
    print("3. x0 (scaled) in the span of the smallest singular directions:")
    for kk in (1, 5, 10, 20, 40, 100, 200):
        if kk <= n:
            print(f"   {kk:4d}: {np.linalg.norm(Vt[-kk:] @ y0):.3f}")
    xs = np.asarray(x0)
    print(f"   |J x0| / |x0| in the scaled unknowns, / sigma_max: "
          f"{np.linalg.norm(A @ xs) / np.linalg.norm(xs * s_col) / sv[0]:.2e}")

    np.savez(a.out + ".npz", sv=sv, sv_raw=sv_raw, s_col=s_col, V=V, labels=labels,
             n_u=n_u, rows=np.array([[r[c] for c in ("sigma", "dJ/J", "dp/p", "par", "B.grad dp",
                                                     "dI/I", "d<p>/<p>", "dF/amp", "u share")]
                                     for r in rows]),
             x0=np.asarray(x0), LM=(L, M), pLM=(pL, pM), frac=a.frac, colloc=fr.tgrid.shape)
    print(f"saved {a.out}.npz; total {time.time() - t0:.0f} s")


if __name__ == "__main__":
    main()
