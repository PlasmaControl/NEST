"""Pointwise AL on random column tables (no DESC): the linearization and its transpose,
the exact Hessian against finite differences of the gradient, and both trust-region
solvers (Steihaug and exact QR) reaching the same constrained state."""
import os
import sys

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "nest"))
import jax  # noqa: E402

jax.config.update("jax_enable_x64", True)
import jax.numpy as jnp  # noqa: E402

import pointwise as pw  # noqa: E402


def tables(n_u=40, n_p=20, n=50, seed=0):
    r = np.random.default_rng(seed)
    w = r.random(n) + 0.5
    w /= w.sum()
    T = {"A": r.standard_normal((n_u, n, 3)), "J_I": r.standard_normal((n, 3)),
         "G": 0.3 * r.standard_normal((n_u, n, 3)), "B0": r.standard_normal((n, 3)) + [0, 3, 0],
         "gP": r.standard_normal((n_p, n, 3)), "P": r.standard_normal((n_p, n)), "w": w,
         "norm": 1.0, "u0": 0.1 * r.standard_normal(n_u), "c0": 0.1 * r.standard_normal(n_p)}
    T["J0"] = np.einsum("k,knc->nc", T["u0"], T["A"]) + T["J_I"]
    T["rmsJ0"] = float(np.sqrt(np.sum(w * np.sum(T["J0"] ** 2, 1))))
    T["rmsp0"] = float(np.sqrt(np.sum(w * (T["c0"] @ T["P"]) ** 2)))
    return T


def setup(T):
    model = pw.DenseModel(T)
    prob = pw.Problem(model, T)
    L, HJ = pw.metric(T)
    x0 = jnp.concatenate([jnp.asarray(T["u0"]), jnp.asarray(T["c0"])])
    Fn = np.linalg.norm(np.asarray(model.base(x0)["F"]), axis=1)
    return model, prob, L, HJ, x0, float(np.percentile(Fn, 90))


def test_linear_and_hessian():
    T = tables()
    model, prob, L, HJ, x0, F90 = setup(T)
    eps, rho = 0.7 * F90, 3.0
    lam = jnp.asarray(np.random.default_rng(1).random(model.n) * 0.1)
    r, m, c, b = prob.residual(x0, lam, rho, eps, 0.05)
    J, JT = prob.linear(b, m, lam, rho, eps)
    v = jnp.asarray(np.random.default_rng(2).standard_normal(model.dim))
    wr = jnp.asarray(np.random.default_rng(3).standard_normal(len(r)))
    assert abs(float(wr @ J(v)) - float(JT(wr) @ v)) < 1e-10 * abs(float(wr @ J(v)))
    h = 1e-6
    r2 = prob.residual(x0 + h * v, lam, rho, eps, 0.05)[0]
    r1 = prob.residual(x0 - h * v, lam, rho, eps, 0.05)[0]
    assert float(jnp.linalg.norm((r2 - r1) / (2 * h) - J(v))) < 1e-6 * float(jnp.linalg.norm(J(v)))
    # exact Hessian of L = |r|^2 / 2 against differences of its gradient J^T r
    H = pw.hessians(T, x0, lam, rho, eps, 1.0, HJ)["H_AL"]

    def grad(x):
        r, m, c, b = prob.residual(x, lam, rho, eps)
        return prob.linear(b, m, lam, rho, eps)[1](r)
    Hv = (grad(x0 + h * v) - grad(x0 - h * v)) / (2 * h)
    err = float(jnp.linalg.norm(Hv - H @ v) / jnp.linalg.norm(Hv))
    print(f"adjoint ok; Hessian vs differences {err:.1e}")
    assert err < 1e-5


def test_solvers_agree():
    T = tables()
    model, prob, L, HJ, x0, F90 = setup(T)
    eps = 0.6 * F90
    met = pw.Metric(L)
    s, V = np.linalg.eigh(HJ)
    Rg = (V * np.sqrt(np.maximum(s, 0))).T

    def qr_fn(rows_fn, n_rows, n_cols):
        M = rows_fn(np.arange(n_rows))
        if n_rows < n_cols:
            M = jnp.vstack([M, jnp.zeros((n_cols - n_rows, n_cols))])
        return jnp.linalg.qr(M, mode="r")[:n_cols, :n_cols]

    res = {}
    for step, d0 in (("gltr", 0.0), ("gltr", 1e-2), ("steihaug", 0.0), ("qr", 0.0), ("qr", 1e-2)):
        sol = pw.ALSolver(prob, met, x0, step=step, T=T, qr_fn=qr_fn, Rg=Rg, log=lambda *a: None,
                          cg_tol=1e-6, cg_maxiter=300)
        x, lam, rho, outer = sol.solve_alpha(x0, eps, jnp.zeros(model.n), 1.0, max_outer=30,
                                             viol_tol=1e-4, maxiter=100, delta0=d0)
        step = f"{step} delta {d0:g}"
        b = model.base(x)
        Fn = np.linalg.norm(np.asarray(b["F"]), axis=1)
        res[step] = (np.asarray(x), prob.phi(b), Fn.max() / eps, outer[-1]["viol"], len(outer))
        print(f"{step}: phi {res[step][1]:.8e}, max|F|/eps {res[step][2]:.5f}, viol {res[step][3]:.1e}, "
              f"outer {res[step][4]}, TR steps {sum(len(o['inner']) for o in outer)}")
    q = res["qr delta 0"][1]
    for k in ("gltr delta 0", "gltr delta 0.01", "qr delta 0.01"):
        assert res[k][2] < 1 + 1e-3 and abs(res[k][1] - q) < 2e-3 * q, k


if __name__ == "__main__":
    test_linear_and_hessian()
    test_solvers_agree()
