"""Conditioning of the force-balance least squares (diagnostic for the optimizer).

For a BoundaryPotential of moderate size (default the equilibrium resolution),
builds the Gauss-Newton matrix A = J^T J on u exactly (one J v and one J^T w per
column) at x0 and
  1. its spectrum after Jacobi scaling (the optimizer's preconditioner):
     condition number and eigenvalue distribution
  2. the exact Gauss-Newton minimum (A^+ g, exact quartic step), repeated: the
     floor of <|F|> / norm for this basis, on the collocation grid and on a
     finer check grid (overfitting)
  3. what the slow directions are: for eigenvectors across the spectrum, the
     parallel fraction of J(v) (|J.B|^2 / |J|^2 |B|^2, volume averaged) and the
     mean |m|, |n| and l of v
  4. preconditioned CG on the exact A (Jacobi): the model decrease after k
     iterations against the exact minimum
Saves A, g, D, the mode labels and the eigen-decomposition to --out (npz).

run (GPU): sbatch --mem=96G run.sbatch scripts/diagnostics/spectrum.py [--LMN 16 16 14] [--frac 10]
    [--kernel truncated] [--pLMN 8 8 6] [--gn 3]
"""

import argparse
import os
import sys
import time

import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
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
from objective import Fields, ForceBalance, ObjectiveFunction, _cartesian  # noqa: E402
from optimize import pcg, quartic_step  # noqa: E402
from pressure import Pressure  # noqa: E402

jax.config.update("jax_enable_x64", True)
EQ = os.path.join(ROOT, "desc_input", "precise_QA_free_boundary_smooth_J",
                  "precise_QA_free_boundary_smooth_J_edgeJ_r0.3.h5")
COILS = os.path.join(ROOT, "desc_input", "precise_QA_free_boundary", "precise_QA_helical_coils.h5")


def mode_labels(cur):
    """(l, m, n) of every entry of u: chi (l = |m|), then a, then b."""
    S, modes = cur.S, np.asarray(cur.basis.modes)
    chi = np.column_stack([np.abs(S.chi_modes[:, 0]), S.chi_modes])
    return np.concatenate([chi, modes, modes]).astype(int)


def normal_matrix(J, JT, n, n_x, batch=1):
    """A = J^T J restricted to the first n unknowns, column by column."""
    A = np.zeros((n, n))
    t0 = time.time()
    for i in range(n):
        e = jnp.zeros(n_x).at[i].set(1.0)
        A[:, i] = np.asarray(JT(J(e)))[:n]
        if i == 9:
            print(f"  A: {(time.time() - t0) / 10 * n / 60:.1f} min estimated", flush=True)
    return 0.5 * (A + A.T)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--LMN", type=int, nargs=3, default=(16, 16, 14))
    ap.add_argument("--frac", type=int, default=10)
    ap.add_argument("--kernel", default="hockney", choices=["hockney", "truncated"])
    ap.add_argument("--pLMN", type=int, nargs=3, default=None,
                    help="pressure resolution; given: c_p free too (default: p fixed)")
    ap.add_argument("--colloc", type=float, default=2, help="/ resolution of F")
    ap.add_argument("--check", type=float, default=3)
    ap.add_argument("--gn", type=int, default=3, help="exact Gauss-Newton iterations")
    ap.add_argument("--out", default=os.path.join(ROOT, "results", "spectrum"))
    a = ap.parse_args()
    t0 = time.time()
    inp = DESCInput.load(EQ, field=COILS)
    eq = inp.eq
    L, M, N = a.LMN
    cur = fit_cache.fit(BoundaryPotential, inp, L=L, M=M, N=N)
    pre = fit_cache.fit(Pressure, inp, edge_order=2, **(dict(zip("LMN", a.pLMN)) if a.pLMN else {}))
    box = CartesianGrid.periodic_box(inp, inp.a / a.frac, odd=True).locate_plasma(inp)
    b = cur.basis
    res = lambda f: tuple(int(round(f * (u + v))) for u, v in ((b.L, eq.L), (b.M, eq.M), (b.N, eq.N)))
    fr = ForceResidual(inp, box, resolution=res(a.colloc))
    fields = Fields(fr, cur, pre, kernel=a.kernel, dtype=jnp.float64)
    obj = ObjectiveFunction([ForceBalance()], fields).build()
    n_x = obj.dim_x
    n = n_x if a.pLMN else fields.n_u
    labels = mode_labels(cur)
    if a.pLMN:
        labels = np.concatenate([labels, np.asarray(pre.basis.modes, int)])
    print(f"potential L, M, N = {b.L}, {b.M}, {b.N}, pressure {'free ' + str(a.pLMN) if a.pLMN else 'fixed'}, "
          f"kernel {a.kernel}: {n:,} unknowns; collocation {fr.tgrid.shape}, "
          f"{obj.dim_f:,} residuals; box {box.shape}; setup {time.time() - t0:.0f} s", flush=True)

    # B and the linear part of J at the collocation points, for the parallel fractions
    x = obj.x0
    B = np.asarray(fields.compute(x)["B"])
    w_vol = np.asarray(fields.sw) ** 2
    J_lin = jax.jit(lambda v, m: _cartesian(m(v, 0.0).reshape(-1, 3), fields.phi))

    def parallel_fraction(v):
        Jv = np.asarray(J_lin(jnp.asarray(v[: fields.n_u]), fields.J_col))
        par = np.sum(Jv * B, 1) ** 2 / np.sum(B * B, 1)
        return float(np.sum(w_vol * par) / np.sum(w_vol * np.sum(Jv * Jv, 1)))

    chk = ForceResidual(inp, box, resolution=res(a.check))
    fc = Fields(chk, cur, pre, kernel=a.kernel, dtype=jnp.float64)
    check = lambda x: chk.metrics(np.asarray(fc.compute(x)["F"]))["mean"]
    metric = lambda x: fr.metrics(np.asarray(fields.compute(x)["F"]))["mean"]
    print(f"x0: <|F|>/norm {metric(x):.3e} (check grid {chk.tgrid.shape}: {check(x):.3e}); "
          f"DESC {fr.F_desc_vol / fr.norm:.3e}", flush=True)

    save = {}
    for it in range(a.gn):
        t1 = time.time()
        r, J, JT = obj.linearize(x)
        g = np.asarray(JT(r))[:n]
        A = normal_matrix(J, JT, n, n_x)
        D = np.diag(A).copy()
        s = 1 / np.sqrt(D)
        lam, V = np.linalg.eigh(s[:, None] * A * s[None, :])
        lam = np.maximum(lam, 0)
        # exact Gauss-Newton step (pseudo-inverse, relative cutoff 1e-12)
        gs = V.T @ (s * g)
        keep = lam > 1e-12 * lam[-1]
        d = -s * (V[:, keep] @ (gs[keep] / lam[keep]))
        dx = jnp.zeros(n_x).at[:n].set(d)
        rr = float(jnp.vdot(r, r))
        model = rr + 2 * g @ d + d @ A @ d
        Jd = J(dx)
        q = obj.r(x + dx) - r - Jd
        step, f2 = quartic_step(r, Jd, q)
        if it == 0:
            save = dict(A=A, g=g, D=D, lam=lam, V=V, labels=labels, r2=rr)
            print("1. spectrum of D^-1/2 A D^-1/2 (D = diag A, the Jacobi preconditioner):")
            print(f"   lam_max {lam[-1]:.3e}, condition (lam > 1e-14 lam_max) "
                  f"{lam[-1] / lam[lam > 1e-14 * lam[-1]][0]:.2e}")
            for p in (1e-2, 1e-4, 1e-6, 1e-8, 1e-10, 1e-12):
                print(f"   eigenvalues below {p:.0e} lam_max: {np.sum(lam < p * lam[-1]):6d} "
                      f"/ {n}")
            # where the gradient sits in the spectrum: fraction of the optimal
            # decrease carried by eigenvalues below each threshold
            dec = np.where(keep, gs**2 / np.where(keep, lam, 1), 0)
            print("   share of the optimal model decrease from eigenvalues below p lam_max:")
            print("   " + ", ".join(f"{p:.0e}: {dec[lam < p * lam[-1]].sum() / dec.sum():.2f}"
                                    for p in (1e-2, 1e-4, 1e-6, 1e-8)))
            print("3. eigenvectors: lam / lam_max, parallel fraction of J(v), mean |m|, |n|, l:")
            idx = np.unique(np.concatenate([np.arange(0, n, max(n // 12, 1)), [n - 1],
                                            np.flatnonzero(keep)[:6]]))
            for i in idx:
                v = s * V[:, i]
                wv = v**2 / np.sum(v**2)
                ml = [float(wv @ np.abs(labels[:, k])) for k in (1, 2, 0)]
                print(f"   {i:6d}  {lam[i] / lam[-1]:.2e}  par {parallel_fraction(v):.3f}  "
                      f"|m| {ml[0]:5.1f}  |n| {ml[1]:5.1f}  l {ml[2]:5.1f}", flush=True)
            print("4. Jacobi-preconditioned CG on the exact A: model decrease / optimal decrease")
            opt = rr - model
            for k in (10, 50, 200, 400, 1000, 3000):
                dk, _ = pcg(lambda v: jnp.asarray(A) @ v, -jnp.asarray(g), jnp.asarray(1 / D),
                            tol=0.0, maxiter=k)
                dk = np.asarray(dk)
                print(f"   k = {k:5d}: {(-2 * g @ dk - dk @ A @ dk) / opt:.4f}", flush=True)
        x = x + step * dx
        print(f"2. exact GN iteration {it + 1}: model |r|^2 {rr:.3e} -> {model:.3e}, actual "
              f"{f2:.3e} (step {step:.3f}); <|F|>/norm {metric(x):.3e}, check grid "
              f"{check(x):.3e}; {time.time() - t1:.0f} s", flush=True)
    np.savez(a.out + ".npz", x=np.asarray(x), **save)
    print(f"saved {a.out}.npz; total {time.time() - t0:.0f} s")


if __name__ == "__main__":
    main()
