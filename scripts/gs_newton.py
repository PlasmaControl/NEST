"""Trust-region Newton with the exact Hessian from a run's end point (check 2).

Levenberg-Marquardt (Gauss-Newton, truncated CG) cannot follow negative curvature and
does not resolve the softest directions. Here, from the run's end point, with its own
model (fits, box, dense G, B_ext = B_DESC - B_plasma(x0), its objectives):

  each iteration
    1. A, r at x; H = A^T A + sum r d^2 r (gs_hessian.py) in the physical unknowns y
       (|y|^2 = <|dJ|^2>/rms J^2 + <dp^2>/rms p^2, so a step is an rms field change),
       in the basis of A's right singular vectors, with the k smallest as the
       near-null block: the step there from the eigendecomposition of its Schur
       complement S (accurate for the tiny and negative eigenvalues), the rest from
       the large block;
    2. the trust-region step: min g.d + d.H.d / 2 with |d| <= Delta, i.e.
       d = -(H + mu)^-1 g with mu >= max(0, -lambda_min) and |d| = Delta unless the
       Newton step is inside (the easy case; the hard case is not treated);
    3. accept if f decreases (ratio actual / predicted), update Delta.
  Records f, <|F|>/norm, |d|, lambda_min, the number of negative eigenvalues, the
  distance from the start (field change), and saves x and the history.

The model and the Jacobian come from gs_dense_solve.py (build, column tables): any grid
(box or cylindrical annulus) and B_ext (saved, B_DESC trick or coils).

run (GPU): sbatch --mem=96G run.sbatch scripts/gs_newton.py --run results/gs/fdrift64a20gpu_L24_f2
    [--iters 15] [--delta 1e-5] [--k 400]
"""

import argparse
import json
import os
import sys
import time

import numpy as np
from scipy.linalg import solve_triangular

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, "nest"))

from desc import set_device  # noqa: E402

set_device(os.environ.get("NEST_DEVICE", "cpu"))

import jax  # noqa: E402
import jax.numpy as jnp  # noqa: E402

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from gs_dense_solve import assembler, build, column_tables  # noqa: E402
import fit_cache  # noqa: E402
from boundary_potential import BoundaryPotential  # noqa: E402
from dense import DenseFields, J_columns  # noqa: E402
from desc_input import DESCInput  # noqa: E402
from force import ForceResidual  # noqa: E402
from grid import CartesianGrid  # noqa: E402
from objective import (CurrentError, FieldError, Fields, ForceBalance,  # noqa: E402
                       ObjectiveFunction, PressureEnergy, PressureError, PressurePositivity)
from pressure import Pressure  # noqa: E402

jax.config.update("jax_enable_x64", True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", required=True)
    ap.add_argument("--iters", type=int, default=15)
    ap.add_argument("--delta", type=float, default=1e-5, help="initial trust radius (rms field change)")
    ap.add_argument("--k", type=int, default=400)
    ap.add_argument("--out", default=None)
    ap.add_argument("--coils", action="store_true", help="B_ext from the coils, ignoring a saved B_ext")
    a = ap.parse_args()
    t0 = time.time()
    run = json.load(open(a.run + "_iter.json"))["args"]
    st = np.load(a.run + "_iter.npz")
    x0 = jnp.asarray(st["x0"])
    x = np.asarray(st["x"], dtype=float)
    # the run's own model (grid: box or annulus; B_ext saved, B_DESC - B_plasma(x0), or the
    # coils; all its objective terms), as gs_dense_solve.py builds it
    inp, fields, obj = build(run, st, coils=a.coils)
    fields.x0 = x0
    n, n_u, n_F = obj.dim_x, fields.n_u, obj.objectives[0].dim_f
    sw = np.asarray(fields.sw)
    w = sw**2
    tables = column_tables(fields)
    assemble = assembler(obj, fields, tables)
    AJt, Gt, PPt, _ = tables

    # the physical unknowns y = T x (as gs_hessian.py --norm physical), from the start's fields
    A_J = np.asarray(AJt, dtype=float)
    G = np.asarray(Gt, dtype=float)
    f0 = fields.compute(x0)
    rmsJ = np.sqrt(np.sum(w * np.sum(np.asarray(f0["J"]) ** 2, 1)))
    rmsp = np.sqrt(np.sum(w * np.asarray(f0["p"]) ** 2))
    n_p = n - n_u
    A_p = np.asarray(PPt, dtype=float)
    Tm = np.zeros((n, n))
    Tm[:n_u, :n_u] = np.linalg.qr((sw[None, :, None] * A_J / rmsJ).reshape(n_u, -1).T, mode="r")
    Tm[n_u:, n_u:] = np.linalg.qr(sw[:, None] * A_p.T / rmsp, mode="r")
    solveT = lambda Y: solve_triangular(Tm, Y)  # noqa: E731
    solveTt = lambda Y: solve_triangular(Tm, Y, trans="T")  # noqa: E731
    fval = lambda xx: 0.5 * float(np.sum(np.asarray(obj.compute_scaled_error(jnp.asarray(xx))) ** 2))  # noqa: E731
    Fmet = lambda xx: obj.report(jnp.asarray(xx))["<|F|>/norm"]  # noqa: E731
    y_start = Tm @ x
    print(f"{inp}\nrun {a.run}: objectives {[o.name for o in obj.objectives]}; {n} unknowns; "
          f"setup {time.time() - t0:.0f} s; start f {fval(x):.4e}, <|F|>/norm {Fmet(x):.4e}", flush=True)

    hist, delta = [], a.delta
    for it in range(1, a.iters + 1):
        t1 = time.time()
        r, blocks = assemble(jnp.asarray(x))  # the exact Jacobian from the column tables
        r = np.asarray(r)
        A = np.concatenate([np.asarray(b_) for b_ in blocks], axis=1)
        del blocks
        Ay = solveTt(A.T).T
        _, sv, Vt = np.linalg.svd(np.linalg.qr(Ay, mode="r"))
        V = Vt.T
        gy = Ay.T @ r
        # the second-order term in y and in V's basis
        R = r[:n_F].reshape(-1, 3) * (sw / fields.norm)[:, None]
        Tuu = np.cross(R[None], A_J).reshape(n_u, -1) @ G.reshape(n_u, -1).T
        H2 = np.zeros((n, n))
        H2[:n_u, :n_u] = Tuu + Tuu.T
        H2y = solveTt(solveTt(H2).T)
        Wm = V.T @ (0.5 * (H2y + H2y.T)) @ V
        gv = V.T @ gy
        k = min(a.k, n - 1)
        s, l = np.arange(n - k, n), np.arange(0, n - k)
        Hll = np.diag(sv[l] ** 2) + Wm[np.ix_(l, l)]
        Hls = Wm[np.ix_(l, s)]
        Hss = np.diag(sv[s] ** 2) + Wm[np.ix_(s, s)]
        HllinvHls = np.linalg.solve(Hll, Hls)
        S = Hss - Hls.T @ HllinvHls
        lam, U = np.linalg.eigh(0.5 * (S + S.T))
        gt = U.T @ (gv[s] - HllinvHls.T @ gv[l])  # the reduced gradient on S's eigenvectors
        n_neg = int(np.sum(lam < 0))

        def step(mu):
            ds = -U @ (gt / (lam + mu))
            dl = -np.linalg.solve(Hll + mu * np.eye(len(l)), gv[l] + Hls @ ds)
            d = np.zeros(n)
            d[s], d[l] = ds, dl
            return d  # in V's basis; |d| is the rms field change

        lo = max(0.0, -lam[0]) * (1 + 1e-9) + 1e-300
        d = step(lo if lam[0] <= 0 else 0.0)
        if lam[0] <= 0 or np.linalg.norm(d) > delta:  # on the boundary: |d(mu)| = delta
            a_, b_ = lo, lo + 1.0
            while np.linalg.norm(step(b_)) > delta:
                b_ *= 10
            for _ in range(200):
                m_ = np.sqrt(a_ * b_) if a_ > 0 else 0.5 * (a_ + b_)
                if np.linalg.norm(step(m_)) > delta:
                    a_ = m_
                else:
                    b_ = m_
            d = step(b_)
        pred = -(gv @ d + 0.5 * (d @ (sv**2 * d + Wm @ d)))  # the model's decrease
        dx = solveT(V @ d)
        f_old, f_new = fval(x), fval(x + dx)
        rho = (f_old - f_new) / pred if pred > 0 else -1.0
        accepted = f_new < f_old
        if accepted:
            x = x + dx
        delta = delta * 3 if (accepted and rho > 0.75 and np.linalg.norm(d) > 0.99 * delta) else (
            delta / 4 if rho < 0.25 else delta)
        dist = np.linalg.norm(Tm @ x - y_start)
        h = {"it": it, "f": fval(x), "F": Fmet(x), "|d|": float(np.linalg.norm(d)), "pred": float(pred),
             "actual": float(f_old - f_new), "rho": float(rho), "accepted": bool(accepted),
             "lambda_min": float(lam[0]), "n_neg": n_neg, "|g|": float(np.linalg.norm(gy)),
             "delta": float(delta), "dist from LM end": float(dist), "time": time.time() - t1}
        hist.append(h)
        print(f"it {it:3d}  f {h['f']:.4e}  <|F|>/norm {h['F']:.3e}  |d| {h['|d|']:.2e}  pred "
              f"{pred:.2e} actual {f_old - f_new:+.2e} rho {rho:+.2f} {'acc' if accepted else 'rej'}  "
              f"lambda_min {lam[0]:+.2e} neg {n_neg}  |g| {h['|g|']:.2e}  moved {dist:.2e}  Delta "
              f"{delta:.1e}  {h['time']:.0f} s", flush=True)
    out = a.out or a.run + "_newton"
    np.savez(out + ".npz", x=x, x_lm=np.asarray(st["x"]), x0=np.asarray(st["x0"]))
    with open(out + ".json", "w") as fo:
        json.dump({"args": vars(a), "history": hist}, fo, indent=1, default=float)
    print(f"saved {out}.npz/.json; total {time.time() - t0:.0f} s")


if __name__ == "__main__":
    main()
