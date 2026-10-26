"""Gates of HarmonicModel speed elim (p eliminated by the weak Poisson / least-squares solve),
docs/pressure_poisson_plan.md P1, on the laptop model of scripts/lm/lm_ab.py (L16 a/20 N = 1, 750
unknowns) at its eps 1000 stage (results/lm/L16a20_full_stage_eps1000_wJ0.001.npz, made by lm_ab.py).

run: PYTHONPATH=nest:$DESC JAX_PLATFORMS=cpu python tests/test_pressure_elim.py [T1 T2 ...]
"""
import os
import sys

import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path[:0] = [os.path.join(ROOT, "nest"), os.path.join(ROOT, "scripts", "dshape"), os.path.join(ROOT, "scripts", "lm")]
sys.argv, _argv = sys.argv[:1], sys.argv
import lm_ab  # noqa: E402

sys.argv = _argv
import jax  # noqa: E402
import jax.numpy as jnp  # noqa: E402
import scipy.linalg as sl  # noqa: E402

EPS = 1000.0
_M = {}


def _model(wJ=1e-7):
    if wJ not in _M:
        m, c = lm_ab.build(wJ, "full")
        m.speed = {"elim": True, "tr": True}
        m._elim_check()
        _M[wJ] = m
    return _M[wJ]


def _state(name="stage"):
    f = {"stage": "L16a20_full_stage_eps1000_wJ0.001.npz", "base": "L16a20_full_base_wJ0.001.npz"}[name]
    return np.load(os.path.join(ROOT, "results", "lm", f))["x"]


def _split(m):
    ip = np.arange(m.n_uc, m.n)
    iu = np.arange(m.n_uc)
    return iu, ip


def _rel(a, b):
    return float(np.linalg.norm(np.asarray(a) - np.asarray(b)) / np.linalg.norm(np.asarray(b)))


def test_T1():
    """Trailing block of the [p | u | r] QR = the Schur complement of the joint Gram (with the r column)."""
    m = _model()
    z = m.to_z(_state())
    n, nu = m.n, m.n_uc
    R = np.asarray(m._tsqr(z, EPS))
    Ra = np.asarray(m._tsqr(z, EPS, cperm=m._elim_cols))
    H = R.T @ R  # (n + 1): [A | r]^T [A | r], z order u, p, r
    iu, ip = _split(m)
    ir = np.r_[iu, [n]]
    S = H[np.ix_(ir, ir)] - H[np.ix_(ir, ip)] @ np.linalg.solve(H[np.ix_(ip, ip)], H[np.ix_(ip, ir)])
    T = Ra[n - nu:, n - nu:]
    e = _rel(T.T @ T, S)
    print(f"T1 |T^T T - Schur| / |Schur| = {e:.2e}")
    assert e < 1e-10


def test_T2():
    """c_p = 0 after the start projection, and at every accepted iterate."""
    m = _model()
    z = m._elim_project(m.to_z(_state()), EPS)
    n, npc = m.n, m.n_pc
    Ra = np.asarray(m._tsqr(z, EPS, cperm=m._elim_cols))
    r = np.linalg.norm(Ra[:, n])
    e0 = np.linalg.norm(Ra[:npc, n]) / r
    m._Delta = None
    _, hist = m.lm(m.to_x(z), EPS, iters=3, rtol=0, gn_tol=0)
    es = [h["|c_p|/|r|"] for h in hist[1:]]
    print(f"T2 |c_p|/|r| after projection {e0:.2e}; at iterations {', '.join(f'{v:.1e}' for v in es)}")
    assert e0 < 1e-10 and max(es) < 1e-10


def test_T3():
    """f* from the trial pass = f at (u + du, p*(u + du)) evaluated directly, and p* is optimal there."""
    m = _model()
    z = m._elim_project(m.to_z(_state()), EPS)
    rng = np.random.default_rng(0)
    Du = jnp.asarray(rng.standard_normal((m.n_uc, 3)) * 1e-2 * float(jnp.linalg.norm(z[: m.n_uc])) / np.sqrt(m.n_uc))
    fs, _, Dz = m._elim_trials(z, Du, EPS)
    worst = wg = 0.0
    for k in range(3):
        zk = z + Dz[:, k]
        f, _ = m.evaluate(m.to_x(zk), EPS)
        g, _, _ = m._grad(zk, EPS)
        g0, _, _ = m._grad(z + jnp.concatenate([Du[:, k], jnp.zeros(m.n_pc)]), EPS)  # p not re-solved
        worst = max(worst, abs(f - fs[k]) / f)
        wg = max(wg, float(jnp.linalg.norm(g[m.n_uc:]) / jnp.linalg.norm(g0[m.n_uc:])))
    print(f"T3 |f* - f(u', p*)| / f <= {worst:.2e}; |grad_p f| at p* / without the re-solve <= {wg:.2e}")
    # f* = f(u', p) - |R_pp^-T A_p^T r|^2 / 2 cancels: the trial f is larger than f* by the removable part
    assert worst < 1e-8 and wg < 1e-8


def test_T4():
    """The undamped reduced Gauss-Newton step = the u part of the undamped joint Gauss-Newton step."""
    m = _model()
    z = m.to_z(_state())
    n, nu, npc = m.n, m.n_uc, m.n_pc
    R = np.asarray(m._tsqr(z, EPS))
    Ra = np.asarray(m._tsqr(z, EPS, cperm=m._elim_cols))
    D = 1.0 / np.linalg.norm(R[:n, :n], axis=0)  # the same column scaling for both triangular solves
    yj = -D * sl.solve_triangular(R[:n, :n] * D, R[:n, n])
    T = Ra[npc:, npc:]
    Du = D[:nu]
    yr = -Du * sl.solve_triangular(T[:nu, :nu] * Du, T[:nu, nu])
    e = _rel(yr, yj[:nu])
    print(f"T4 |reduced GN - u part of joint GN| / |.| = {e:.2e} (|y| / |u| = {np.linalg.norm(yr) / np.linalg.norm(np.asarray(z[:nu])):.2e})")
    assert e < 1e-8


def test_T5():
    """Finite differences of f*(u) match the reduced gradient T_u^T c_u (variable projection is exact)."""
    m = _model()
    z = m._elim_project(m.to_z(_state()), EPS)
    n, nu, npc = m.n, m.n_uc, m.n_pc
    Ra = np.asarray(m._tsqr(z, EPS, cperm=m._elim_cols))
    m._elim_set_Rpp(jnp.asarray(Ra[:npc, :npc]))
    T = Ra[npc:, npc:]
    gred = T[:nu, :nu].T @ T[:nu, nu]
    rng = np.random.default_rng(1)
    d = rng.standard_normal(nu)
    d *= 1e-3 * np.linalg.norm(np.asarray(z[:nu])) / np.linalg.norm(d)
    out = []
    for h in (1.0, 0.5, 0.25):
        fs, _, _ = m._elim_trials(z, jnp.asarray(np.stack([h * d, -h * d], 1)), EPS)
        out.append((fs[0] - fs[1]) / (2 * h))
    rich = (4 * out[2] - out[1]) / 3  # Richardson: O(h^4)
    e = abs(rich - gred @ d) / abs(gred @ d)
    print(f"T5 FD {rich:.10e} vs T_u^T c_u . d {gred @ d:.10e}: rel {e:.2e} (central h: {', '.join(f'{v:.6e}' for v in out)})")
    assert e < 1e-6


def test_T6():
    """A converged joint solution (lm_ab base, w_J 1e-3, eps 0) is a fixed point of the reduced iteration."""
    m = _model(1e-3)
    z = m._elim_project(m.to_z(_state("base")), 0.0)
    n, nu, npc = m.n, m.n_uc, m.n_pc
    Ra = np.asarray(m._tsqr(z, 0.0, cperm=m._elim_cols))
    T = Ra[npc:, npc:]
    D = 1.0 / np.linalg.norm(T[:nu, :nu], axis=0)
    y = -D * sl.solve_triangular(T[:nu, :nu] * D, T[:nu, nu])
    e = np.linalg.norm(y) / np.linalg.norm(np.asarray(z[:nu]))
    gn = float(T[:nu, nu] @ T[:nu, nu]) / float(T[:nu + 1, nu] @ T[:nu + 1, nu])
    print(f"T6 reduced GN step |d_u| / |u| = {e:.2e}, reduced GN share {gn:.2e}")
    assert e < 1e-6


def test_T7():
    """Manufactured Poisson: adding grad q, q in the pressure space, to the force is removed exactly
    (p*(u) does not depend on the starting p)."""
    m = _model()
    z1 = m._elim_project(m.to_z(_state()), EPS)
    rng = np.random.default_rng(2)
    q = jnp.asarray(rng.standard_normal(m.n_pc)) * float(jnp.linalg.norm(z1[m.n_uc:])) / np.sqrt(m.n_pc)
    z2 = z1.at[m.n_uc:].add(q)
    f2, _ = m.evaluate(m.to_x(z2), EPS)
    z3 = m._elim_project(z2, EPS)
    f1, _ = m.evaluate(m.to_x(z1), EPS)
    f3, _ = m.evaluate(m.to_x(z3), EPS)
    e = _rel(z3[m.n_uc:], z1[m.n_uc:])
    print(f"T7 |p*(p + q) - p*(p)| / |p*| = {e:.2e}; f {f1:.6e} -> {f2:.3e} with q -> {f3:.6e} re-solved")
    assert e < 1e-10 and abs(f3 - f1) / f1 < 1e-10


def test_T8():
    """Manufactured Poisson, q outside every V_h: the weighted least-squares (= weak Poisson) error
    |grad(p* - q)|_w / |grad q|_w falls geometrically with the pressure resolution. q = (1 - rho^2)^3 Q,
    Q = (1 + 0.2 x cos(NFP zeta) + 0.2 y sin(NFP zeta)) / (1.15 - x), x = rho cos(theta), y = rho sin(theta):
    stellarator symmetric, analytic in the disk, pole at x = 1.15 (expected rate ~0.58^L). The weights
    and gradients are those of the force rows and the pressure tables (TensorPressure, fr.geo)."""
    import fit_cache
    from flux_interp import TensorPressure
    from pressure import Pressure
    from separable import Separable
    m = _model()
    c = lm_ab.build.__defaults__[0]  # lm_ab's setup cache
    fr, cur, inp = c["fr"], c["cur"], c["inp"]
    NFP = int(inp.eq.NFP)
    g = fr.tgrid
    Rg, Tg, Zg = np.meshgrid(*(np.asarray(v) for v in (g.rho, g.theta, g.zeta)), indexing="ij")

    def q(r, t, z):
        x, y = r * jnp.cos(t), r * jnp.sin(t)
        return (1 - r**2) ** 3 * (1 + 0.2 * x * jnp.cos(NFP * z) + 0.2 * y * jnp.sin(NFP * z)) / (1.15 - x)

    d = jax.vmap(jax.grad(q, argnums=(0, 1, 2)))(*(jnp.asarray(a.ravel()) for a in (Rg, Tg, Zg)))
    gq = np.asarray(fr.geo.gradient(*(jnp.asarray(np.asarray(v).reshape(Rg.shape)) for v in d)))
    sw = np.asarray(m.sw3)
    b = (sw[..., None] * gq).ravel()
    errs, Ls = [], []
    for pf in (0.5, 0.75, 1.0, 1.5, 2.0):
        pre = fit_cache.fit(Pressure, inp, factors=(pf, pf, 1.0), N=1, edge_order=2)
        tp = TensorPressure(Separable([pre.basis], [pre.c], NFP), pre.edge_order, cur._RZ, g, fr.geo, factored=True)
        f = jax.jit(jax.vmap(lambda e: tp(e)[1]))
        n_p = pre.basis.num_modes
        A = np.empty((b.size, n_p))
        for i in range(0, n_p, 64):
            k = min(64, n_p - i)
            E = np.zeros((k, n_p))
            E[np.arange(k), i + np.arange(k)] = 1.0
            A[:, i:i + k] = (sw[None, ..., None] * np.asarray(f(jnp.asarray(E)))).reshape(k, -1).T
        cs = np.linalg.lstsq(A, b, rcond=None)[0]
        errs.append(np.linalg.norm(A @ cs - b) / np.linalg.norm(b))
        Ls.append(pre.basis.L)
    rates = [(errs[i + 1] / errs[i]) ** (4.0 / (Ls[i + 1] - Ls[i])) for i in range(len(errs) - 1)]
    print("T8 " + ", ".join(f"L{L} {e:.2e}" for L, e in zip(Ls, errs)) + "; reduction per +4 in L: "
          + ", ".join(f"{r:.3f}" for r in rates))
    assert max(rates) < 0.2 and errs[-1] < 1e-7


if __name__ == "__main__":
    jax.config.update("jax_enable_x64", True)
    names = sys.argv[1:] or [k[5:] for k in sorted(globals()) if k.startswith("test_T")]
    for nm in names:
        globals()["test_" + nm]()
    print("all passed:", " ".join(names))
