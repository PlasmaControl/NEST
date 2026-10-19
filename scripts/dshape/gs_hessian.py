"""Exact Hessian of the force-only objective at a run's end point: valley, saddle or minimum?

f(x) = |r(x)|^2 / 2 with r = sqrt(dV/V) F / norm, F = J(u) x B(u) - grad p(c_p),
J = J_I + A_J u, B = B_ext + B_I + G u (the run's dense model). Then

    H = A^T A + sum_i r_i d^2 r_i,
    sum_i r_i d^2 r_i [a, b] = sum_points (dV/V) / norm^2 F . (A_J,a x G_b + A_J,b x G_a)

(the only curvature is the bilinear J(du) x G du; p is linear), exact and cheap. A run
with target terms (--w-J, --w-B, --w-p: J - J_DESC, ...) is analysed with them: they
are linear in x, so they add their rows to A (to A^T A) and nothing to the second term.
The gradient A^T r on the lowest eigenvectors tells whether x is stationary there.
A^T A >= 0 alone cannot show a saddle; the second term can turn its smallest
eigenvalues negative.

The smallest eigenvalues are ~1e-17 of the largest, below what a plain float64
eigensolve resolves. So H is taken in the column-scaled unknowns y = x s (s the
column norms of A) and in the basis of A's right singular vectors V, where
A^T A = diag(sigma^2) exactly. Split V into the k smallest (near-null, s) and the rest
(l): the small eigenvalues of H are those of the Schur complement

    S = H_ss - H_sl H_ll^-1 H_ls,

computed in that basis, and compared with sigma_s^2 (A^T A alone). Prints the smallest
eigenvalues of S, the number of negative ones, the size of the second term against
sigma^2 on the near-null subspace, and what the most negative / smallest eigenvectors
are (parallel fraction of dJ, |B.grad dp| / |B||grad dp|, mean |m|).

run (GPU): sbatch --mem=96G run.sbatch scripts/dshape/gs_hessian.py --run results/gs/fdrift64a20gpu_L24_f2
    [--at end|start] [--k 400]
"""

import argparse
import json
import os
import sys
import time

import numpy as np
from scipy.linalg import solve_triangular

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.join(ROOT, "nest"))

from desc import set_device  # noqa: E402

set_device(os.environ.get("NEST_DEVICE", "cpu"))

import jax  # noqa: E402
import jax.numpy as jnp  # noqa: E402

import fit_cache  # noqa: E402
from boundary_potential import BoundaryPotential  # noqa: E402
from dense import DenseFields, j_target, J_columns  # noqa: E402
from desc_input import DESCInput  # noqa: E402
from boxes import make_box  # noqa: E402
from force import ForceResidual  # noqa: E402
from grid import CartesianGrid  # noqa: E402
from objective import (CurrentError, FieldError, Fields, ForceBalance,  # noqa: E402
                       CurrentMoments, ObjectiveFunction, PressureEnergy, PressureError, PressurePositivity, _cartesian)
from precondition import mode_labels  # noqa: E402
from pressure import Pressure  # noqa: E402

jax.config.update("jax_enable_x64", True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", required=True)
    ap.add_argument("--at", default="end", choices=["start", "end"])
    ap.add_argument("--k", type=int, default=400, help="size of the near-null subspace")
    ap.add_argument("--gn", action="store_true",
                    help="also the exact Gauss-Newton step at x (direct least squares, no CG): "
                         "its size in the analysis norm and the decrease of f it predicts")
    ap.add_argument("--norm", default="coef", choices=["coef", "physical"],
                    help="coef: unknowns scaled by A's column norms; physical: z with |z|^2 = "
                         "<|dJ|^2>/rms J^2 + <dp^2>/rms p^2 (a unit step is a unit rms change of "
                         "the fields), from a QR factorization of the field-change map")
    a = ap.parse_args()
    t0 = time.time()
    run = json.load(open(a.run + "_iter.json"))["args"]
    st = np.load(a.run + "_iter.npz")
    x0 = jnp.asarray(st["x0"])
    x = jnp.asarray(st["x"] if a.at == "end" else st["x0"])
    inp = DESCInput.load(run["eq"], field=run["coils"])
    eq = inp.eq
    ep = run.get("edge_power", 2)
    cur = fit_cache.fit(BoundaryPotential, inp, factors=tuple(run["factors"]),
                        **({"edge_power": ep} if ep != 2 else {}))
    pre = fit_cache.fit(Pressure, inp, factors=tuple(run["factors"]), edge_order=2)
    b = cur.basis
    res = tuple(int(round(run["colloc"] * (u + v))) for u, v in ((b.L, eq.L), (b.M, eq.M), (b.N, eq.N)))
    box = make_box(inp, run["frac"], run.get("grid", "cartesian"), bool(eq.sym), run.get("margin", 2.0))
    fr = ForceResidual(inp, box, resolution=res)
    fr.B_ext = np.zeros((len(fr.rtz), 3))  # replaced below
    fl = Fields(fr, cur, pre, kernel=run["kernel"], dtype=getattr(jnp, run.get("dtype", "float64")),
                S_dtype=getattr(jnp, run.get("S_dtype", "float32")))
    if run.get("j_nodes") == "direct" and getattr(fl, "cyl", False):  # J at the nodes from the basis
        from node_current import NodeFields
        fl = NodeFields.from_fields(fl, cur, box)
    fields = DenseFields.from_fields(fl, cur, method=run.get("dense_g", "box"))
    fields.x0 = x0
    if "B_ext" in st.files:  # the run's own (e.g. --bplasma axisym: coils + accurate correction)
        fields.B_ext = jnp.asarray(st["B_ext"])
    elif run.get("bext") == "desc":
        fields.B_ext = fields.desc["B"] - fields.compute(x0)["B_plasma"]
    else:
        raise SystemExit("a coil-field run without a saved B_ext: rerun it with this version")
    # the run's objectives: force balance, and the linear target terms if it had them
    # (J - J_DESC, B - B_DESC, p - p_DESC: they add to A^T A, not to the curvature term)
    extra = [(k, cls) for k, cls in (("J", CurrentError), ("B", FieldError), ("p", PressureError))
             if (run.get(f"w_{k}") or 0) > 0]
    J_start = j_target(run, fields, x0, inp)  # w_J's target (None: J_DESC)
    terms = [ForceBalance()] + [cls(run[f"w_{k}"], target=J_start, rho_max=run.get("w_J_rho")) if k == "J"
                                else cls(run[f"w_{k}"]) for k, cls in extra]
    # <p> held and p >= 0, if the run had them (linear / piecewise linear: A only)
    if (run.get("w_M") or 0) > 0:
        terms.append(CurrentMoments(run["w_M"], which=tuple(run.get("moments", ("I", "R", "Z", "V")))))
    if (run.get("w_W") or 0) > 0:
        terms.append(PressureEnergy(run["w_W"], scale=run.get("p_scale", 1.0)))
    if (run.get("w_pos") or 0) > 0:
        terms.append(PressurePositivity(run["w_pos"]))
    obj = ObjectiveFunction(terms, fields).build()
    n_F = obj.objectives[0].dim_f  # the force rows come first
    n, n_u = obj.dim_x, fields.n_u
    print(f"{inp}\nrun {a.run} ({a.at}): potential L, M = {b.L}, {b.M} ({n_u} + {n - n_u} "
          f"unknowns); collocation {fr.tgrid.shape}; box a/{run['frac']}; setup "
          f"{time.time() - t0:.0f} s", flush=True)

    # A and r at x
    t1 = time.time()
    r, Jv, _ = obj.linearize(x)
    r = np.asarray(r)
    jv = jax.jit(jax.vmap(Jv))
    A = np.empty((obj.dim_f, n))
    for i in range(0, n, 32):
        E = np.zeros((32, n))
        kk = min(32, n - i)
        E[np.arange(kk), i + np.arange(kk)] = 1.0
        A[:, i : i + kk] = np.asarray(jv(jnp.asarray(E))).T[:, :kk]
    s_col = np.linalg.norm(A, axis=0)
    s_col[s_col == 0] = 1.0
    # the unknowns of the analysis: y = T x with T upper triangular (diag(s_col), or the R of
    # the field-change map for --norm physical); A_y = A T^-1
    A_J, _ = J_columns(fr, cur)  # (n_u, n_pts, 3)
    sw = np.asarray(fields.sw)
    fx = fields.compute(x)
    wv0 = sw**2
    rmsJ0 = np.sqrt(np.sum(wv0 * np.sum(np.asarray(fx["J"]) ** 2, 1)))
    rmsp0 = np.sqrt(np.sum(wv0 * np.asarray(fx["p"]) ** 2))
    if a.norm == "coef":
        Tm = np.diag(s_col)
    else:
        p_lin = jax.jit(jax.vmap(lambda c, P: P(c)[0].ravel(), in_axes=(0, None)))
        n_p = n - n_u
        A_p = np.concatenate([np.asarray(p_lin(jnp.asarray(np.eye(n_p)[i : i + 64]), fields.p_col))
                              for i in range(0, n_p, 64)])  # (n_p, n_pts)
        Cu = (sw[None, :, None] * A_J / rmsJ0).reshape(n_u, -1).T
        Cp = sw[:, None] * A_p.T / rmsp0
        Tm = np.zeros((n, n))
        Tm[:n_u, :n_u] = np.linalg.qr(Cu, mode="r")
        Tm[n_u:, n_u:] = np.linalg.qr(Cp, mode="r")
        dT = np.abs(np.diag(Tm))
        print(f"field-change map: diag of R from {dT.min():.1e} to {dT.max():.1e} (the smallest: "
              f"coefficient combinations that barely change the fields)", flush=True)
    solveT = lambda Y: solve_triangular(Tm, Y)  # noqa: E731  T^-1 Y
    solveTt = lambda Y: solve_triangular(Tm, Y, trans="T")  # noqa: E731  T^-T Y
    Ay = solveTt(A.T).T
    if a.gn:
        Qy, Ry = np.linalg.qr(Ay)
        Uy, sv, Vt = np.linalg.svd(Ry)
        c = Uy.T @ (Qy.T @ r)  # r on the left singular directions
        del Qy
    else:
        _, sv, Vt = np.linalg.svd(np.linalg.qr(Ay, mode="r"))
    V = Vt.T  # columns: right singular vectors, sigma descending
    smax = sv[0]
    f = 0.5 * float(r @ r)
    print(f"A {A.shape}: sigma/sigma_max min {sv[-1] / smax:.2e}; f = |r|^2/2 = {f:.3e} "
          f"(<|F|>/norm {obj.report(x)['<|F|>/norm']:.2e}); {time.time() - t1:.0f} s", flush=True)

    if a.gn:  # the exact Gauss-Newton step d = -V S^-1 c; its decrease c_i^2 / 2 per direction
        f_ = 0.5 * float(r @ r)
        d_comp = -c / sv
        units = "rms field change" if a.norm == "physical" else "/|y|"
        yn = 1.0 if a.norm == "physical" else np.linalg.norm(Tm @ np.asarray(x))
        print(f"exact Gauss-Newton step ({units}): |d| {np.linalg.norm(d_comp) / yn:.2e}; predicted "
              f"decrease {0.5 * np.sum(c**2):.2e} of f = {f_:.2e} ({0.5 * np.sum(c**2) / f_:.2e})")
        print("   restricted to sigma/sigma_max > cutoff: |d|, predicted decrease / f")
        for cut in (1e-2, 1e-4, 1e-6, 1e-8, 1e-10, 0.0):
            m = sv / sv[0] > cut
            print(f"     cutoff {cut:7.0e}: {m.sum():5d} directions, |d| "
                  f"{np.linalg.norm(d_comp[m]) / yn:.2e}, decrease {0.5 * np.sum(c[m]**2) / f_:.2e}")
        gn = {"d": float(np.linalg.norm(d_comp) / yn), "decrease_rel": float(0.5 * np.sum(c**2) / f_)}

    # second-order term: sum_p (dV/V) / norm^2 F_p . (A_J,a x G_b + A_J,b x G_a)
    t1 = time.time()
    G = np.asarray(fields.G, dtype=float).reshape(n_u, -1, 3)
    R = r[:n_F].reshape(-1, 3) * (sw / fields.norm)[:, None]  # = (dV/V) F / norm^2 per point
    M = np.cross(R[None], A_J)  # R x A_J,a ; R . (A_J,a x G_b) = G_b . (R x A_J,a)
    T = M.reshape(n_u, -1) @ G.reshape(n_u, -1).T
    H2 = np.zeros((n, n))
    H2[:n_u, :n_u] = T + T.T
    H2s = solveTt(solveTt(H2).T)  # T^-T H2 T^-1 (H2 symmetric)
    H2s = 0.5 * (H2s + H2s.T)
    print(f"second-order term: |H2| (scaled) {np.linalg.norm(H2s, 2):.2e} = "
          f"{np.linalg.norm(H2s, 2) / smax**2:.2e} sigma_max^2; {time.time() - t1:.0f} s", flush=True)

    # in V's basis: H = diag(sigma^2) + V^T H2 V; Schur complement on the k smallest
    W = V.T @ H2s @ V
    k = min(a.k, n - 1)
    s_idx, l_idx = np.arange(n - k, n), np.arange(0, n - k)
    Hll = np.diag(sv[l_idx] ** 2) + W[np.ix_(l_idx, l_idx)]
    Hls = W[np.ix_(l_idx, s_idx)]
    Hss = np.diag(sv[s_idx] ** 2) + W[np.ix_(s_idx, s_idx)]
    S = Hss - Hls.T @ np.linalg.solve(Hll, Hls)
    lam, U = np.linalg.eigh(0.5 * (S + S.T))
    # inertia (Haynsworth): negatives of H = those of H_ll + those of S
    neg_ll = int(np.sum(np.linalg.eigvalsh(0.5 * (Hll + Hll.T)) < 0))
    gn = np.sort(sv[s_idx] ** 2)
    W_ss = W[np.ix_(s_idx, s_idx)]
    print(f"\nnear-null subspace: the {k} smallest singular directions (sigma/sigma_max "
          f"{sv[n - k] / smax:.1e} .. {sv[-1] / smax:.1e})")
    print(f"   |V_s^T H2 V_s| = {np.linalg.norm(W_ss, 2) / smax**2:.2e} sigma_max^2, against the "
          f"smallest sigma^2 = {gn[0] / smax**2:.2e} sigma_max^2")
    print(f"   negative eigenvalues of H: {neg_ll + int(np.sum(lam < 0))} (exact inertia: "
          f"{neg_ll} in the rest, {int(np.sum(lam < 0))} in the near-null block)")
    print(f"   eigenvalues of H there (Schur complement), / sigma_max^2; smallest:")
    for i in range(min(12, k)):
        print(f"     {i:3d}  H {lam[i] / smax**2:+.3e}    A^T A alone {gn[i] / smax**2:.3e}"
              f"    (absolute: H {lam[i]:+.3e}, A^T A {gn[i]:.3e})")
    print(f"   largest of them: H {lam[-1] / smax**2:.3e}, A^T A {gn[-1] / smax**2:.3e}")

    # is x stationary? the gradient g = A^T r on the lowest eigenvectors of H (scaled
    # unknowns), and the Newton step there, -g_i / lambda_i, against |y|
    g = solveTt(A.T @ r)  # gradient in the unknowns y
    gV = V.T @ g
    gs = U.T @ gV[s_idx]  # on the eigenvectors of the near-null block
    # the Newton step: in the coefficient norm against |y|; in the physical norm it is
    # already an rms change of the fields (relative to rms J, rms p)
    ynorm = np.linalg.norm(Tm @ np.asarray(x)) if a.norm == "coef" else 1.0
    print(f"   gradient: |g| {np.linalg.norm(g):.2e}; on the near-null block {np.linalg.norm(gV[s_idx]):.2e}"
          f"; on its lowest eigenvectors (|g_i|, Newton step |g_i / lambda_i|"
          + (" / |y|):" if a.norm == "coef" else ", as an rms field change):"))
    for i in range(min(8, k)):
        print(f"     {i:3d}  lambda {lam[i] / smax**2:+.2e}  |g_i| {abs(gs[i]):.2e}  step "
              f"{abs(gs[i] / lam[i]) / ynorm:.2e}")
    stat = {"g": float(np.linalg.norm(g)), "g_null": float(np.linalg.norm(gV[s_idx])),
            "g_low": [float(abs(v)) for v in gs[:8]],
            "step_low": [float(abs(gs[i] / lam[i]) / ynorm) for i in range(min(8, k))]}

    # what the lowest eigenvectors are
    labels = mode_labels(fields, cur, pre)
    f0 = fields.compute(x)
    B0 = np.asarray(f0["B"])
    w = np.asarray(fields.sw) ** 2
    tang = jax.jit(lambda v: fields._tangent(v, f0["J"], f0["B"]))
    print("   lowest eigenvectors: parallel fraction of dJ, <|B.grad dp|>/<|B||grad dp|>, share "
          "of u, mean |m| of u and of p, |dF|/norm per unit field change")
    J0 = np.asarray(f0["J"])
    p0 = np.asarray(f0["p"])
    rmsJ, rmsp = np.sqrt(np.sum(w * np.sum(J0**2, 1))), np.sqrt(np.sum(w * p0**2))
    for i in range(min(6, k)):
        y = V[:, s_idx] @ U[:, i]
        v = solveT(y)
        d = {q: np.asarray(t) for q, t in tang(jnp.asarray(v)).items()}
        dJ, dgp = d["J"], d["grad_p"]
        par = np.sum(w * np.sum(dJ * B0, 1) ** 2 / np.sum(B0**2, 1)) / max(np.sum(w * np.sum(dJ**2, 1)), 1e-300)
        bgp = np.sum(w * np.abs(np.sum(B0 * dgp, 1))) / max(np.sum(w * np.linalg.norm(B0, axis=1) * np.linalg.norm(dgp, axis=1)), 1e-300)
        wv = v**2
        mu = (wv[:n_u] @ np.abs(labels[:n_u, 2])) / max(wv[:n_u].sum(), 1e-300)
        mp = (wv[n_u:] @ np.abs(labels[n_u:, 2])) / max(wv[n_u:].sum(), 1e-300)
        amp = max(np.sqrt(np.sum(w * np.sum(dJ**2, 1))) / rmsJ, np.sqrt(np.sum(w * d["p"] ** 2)) / rmsp)
        dF = np.sum(w * np.linalg.norm(d["F"], axis=1)) / fields.norm / amp
        print(f"     {i:3d}  lambda {lam[i] / smax**2:+.2e}  par {par:.2f}  Bgp {bgp:.3f}  u "
              f"{wv[:n_u].sum() / wv.sum():.2f}  |m| u {mu:4.1f} p {mp:4.1f}  dF/amp {dF:.1e}")
    name = os.path.join(ROOT, "results", "gs", f"hessian_{os.path.basename(a.run)}_{a.at}"
                        + ("_phys" if a.norm == "physical" else ""))
    np.savez(name + ".npz", sv=sv, lam=lam, gn=gn, k=k)
    with open(name + ".json", "w") as fo:
        json.dump({"args": vars(a), "norm": a.norm, **({"gn": gn} if a.gn else {}), "objectives": [o.name for o in obj.objectives], "stationarity": stat,
                   "f": f, "n_negative": neg_ll + int(np.sum(lam < 0)),
                   "lam_min": float(lam[0] / smax**2), "gn_min": float(gn[0] / smax**2),
                   "H2_ss": float(np.linalg.norm(W_ss, 2) / smax**2)}, fo, indent=1, default=float)
    print(f"saved {name}.npz/.json; total {time.time() - t0:.0f} s")


if __name__ == "__main__":
    main()
