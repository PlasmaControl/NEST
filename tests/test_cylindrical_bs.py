"""Gates A1-B3 of docs/cylindrical_grid_plan.md: the cylindrical-annulus J -> B solver.

run: python tests/test_cylindrical_bs.py [name ...]   (or pytest tests/test_cylindrical_bs.py)

Each ``test_*`` prints its measured numbers and asserts the gate's pass criterion.
"""

import os
import sys
import time

import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, "nest"))

from chebyshev import ChebAxis  # noqa: E402


# ------------------------------------------------------------------ A1
def test_A1_polynomials_exact():
    """D, D2 and the quadrature are exact on polynomials of degree <= n (unmapped)."""
    worst = 0.0
    for n in (8, 16, 32, 64, 128):
        ax = ChebAxis(0.6, 1.4, n)
        t = (ax.s - 1.0) / 0.4
        c = np.random.default_rng(n).standard_normal(n + 1)
        P = np.polynomial.Polynomial(c)
        f = P(t)
        scale = np.abs(f).max()
        e1 = np.abs(ax.D @ f - P.deriv(1)(t) / 0.4).max() / (np.abs(P.deriv(1)(t)).max() / 0.4)
        e2 = np.abs(ax.D2 @ f - P.deriv(2)(t) / 0.16).max() / (np.abs(P.deriv(2)(t)).max() / 0.16)
        I = P.integ()
        eI = abs(ax.w_int @ f - 0.4 * (I(1.0) - I(-1.0))) / (0.8 * scale)
        worst = max(worst, e1 / n**2, e2 / n**4, eI)
        print(f"  A1 poly n={n:4d}: D {e1:.1e}  D2 {e2:.1e}  quad {eI:.1e}")
    assert worst < 1e-12, worst


def test_A1_smooth_functions():
    """Spectral convergence on sin(5x), e^x, plain and mapped; off-node interpolation."""
    rows = []
    for alpha in (0.0, 0.5, 0.9, 0.99):
        for n in (16, 32, 64):
            ax = ChebAxis(-1.0, 1.0, n, alpha)
            s = ax.s
            f = np.sin(5 * s) * np.exp(s)
            df = (5 * np.cos(5 * s) + np.sin(5 * s)) * np.exp(s)
            d2f = (10 * np.cos(5 * s) - 24 * np.sin(5 * s)) * np.exp(s)
            st = np.linspace(-1, 1, 1001)
            ft = np.sin(5 * st) * np.exp(st)
            e1 = np.abs(ax.D @ f - df).max() / np.abs(df).max()
            e2 = np.abs(ax.D2 @ f - d2f).max() / np.abs(d2f).max()
            ei = np.abs(ax.interp_matrix(st) @ f - ft).max() / np.abs(ft).max()
            rows.append((alpha, n, e1, e2, ei))
            print(f"  A1 smooth alpha={alpha:4.2f} n={n:3d}: D {e1:.1e}  D2 {e2:.1e}  interp {ei:.1e}")
    for alpha, n, e1, e2, ei in rows:
        if n == 64 and alpha <= 0.9:
            assert max(e1, ei) < 1e-12 and e2 < 1e-10, (alpha, e1, e2, ei)
    # interpolation reproduces the nodes
    ax = ChebAxis(0.3, 2.0, 20, 0.9)
    assert np.allclose(ax.interp_matrix(ax.s), np.eye(21), atol=1e-14)


def test_A1_spacing_rule():
    """for_spacing gives the smallest n with max spacing <= h over the plasma interval."""
    ax = ChebAxis.for_spacing(0.6, 1.4, 0.65, 1.35, 0.0173)
    assert ax.max_spacing_in(0.65, 1.35) <= 0.0173
    prev = ChebAxis(0.6, 1.4, ax.n - 1)
    assert prev.max_spacing_in(0.65, 1.35) > 0.0173
    print(f"  A1 spacing: n = {ax.n}, unmapped; "
          f"alpha 0.9 n = {ChebAxis.for_spacing(0.6, 1.4, 0.65, 1.35, 0.0173, 0.9).n}")


# ------------------------------------------------------------------ A2, A3
from annulus_poisson import AnnulusPoisson  # noqa: E402

# precise_QA_free_boundary_L16N14 extent (R 0.6435-1.3393, |Z| <= 0.4078, a = 0.17277),
# margin 2 h at h = a / 20: the a/20-equivalent sizes of the plan
QA = dict(R_min=0.6435142741280148, R_max=1.3392634816381628, Z_max=0.4078154525697382,
          a=0.17276685816749648, NFP=2)


def qa_axes(frac, alpha=0.0, margin=2.0):
    h = QA["a"] / frac
    d = margin * h
    ra = ChebAxis.for_spacing(QA["R_min"] - d, QA["R_max"] + d, QA["R_min"], QA["R_max"], h, alpha)
    za = ChebAxis.for_spacing(-QA["Z_max"] - d, QA["Z_max"] + d, -QA["Z_max"], QA["Z_max"], h, alpha)
    return ra, za, h


def test_A2_eigendecomposition():
    """Interior L_r,m and D_zz: real negative spectra, exact reconstruction, kappa(V) small."""
    # alpha >= 0.5 gives complex eigenpairs at isolated large |m| (branch crossings of the
    # non-normal mapped operator: alpha 0.5 at m = 514, 0.9 at m = 434 for the a/20 axis),
    # so the default is the unmapped axis; checked here to the |m| of a/20 and a/30
    for frac, alpha, m_max in ((20, 0.0, 495), (30, 0.0, 738), (20, 0.5, 495)):
        ra, za, _ = qa_axes(frac, alpha)
        ms = range(0, m_max + 1)
        ap = AnnulusPoisson(ra, za, ms)
        nr = ra.n
        I = slice(1, nr)
        rec = 0.0
        for k in (0, 1, 5, len(ap.ms) // 2, len(ap.ms) - 1):
            m = ap.ms[k]
            L = (ap.Lr0 - np.diag(m**2 / ap.r**2))[I, I]
            R = ap.Vr[k] @ np.diag(ap.lam_r[k]) @ ap.Vr_inv[k]
            rec = max(rec, np.abs(R - L).max() / np.abs(L).max())
        recz = np.abs(ap.Vz @ np.diag(ap.lam_z) @ ap.Vz_inv - za.D2[1:-1, 1:-1]).max() / np.abs(za.D2).max()
        print(f"  A2 a/{frac} alpha={alpha}: n_r={ra.n} n_z={za.n} |m|<= {ap.ms.max()} ({len(ap.ms)} modes); "
              f"recon r {rec:.1e} z {recz:.1e}; kappa(V_r) {ap.cond_Vr.min():.2f}..{ap.cond_Vr.max():.2f}, "
              f"kappa(V_z) {np.linalg.cond(ap.Vz):.2f}; lambda_r max {ap.lam_r.max():.3e}")
        assert rec < 1e-11 and recz < 1e-11
        assert ap.cond_Vr.max() < 1e8


def _manufactured(m, R_in, R_out, z0):
    """u = sin(pi xi) cos(9 r) e^r cos(pi z / 2 z0) sin(8 z + 0.3) (zero on the boundary) and L_m u,
    by JAX autodiff."""
    import jax
    jax.config.update("jax_enable_x64", True)
    import jax.numpy as jnp

    def u(r, z):
        xi = (r - R_in) / (R_out - R_in)
        return (jnp.sin(jnp.pi * xi) * jnp.cos(9 * r) * jnp.exp(r)
                * jnp.cos(jnp.pi * z / (2 * z0)) * jnp.sin(8 * z + 0.3))

    ur = jax.grad(u, 0)
    urr = jax.grad(ur, 0)
    uzz = jax.grad(jax.grad(u, 1), 1)

    def Lu(r, z):
        return urr(r, z) + ur(r, z) / r - m**2 * u(r, z) / r**2 + uzz(r, z)

    V = lambda f: jax.vmap(jax.vmap(f, (None, 0)), (0, None))  # noqa: E731
    return lambda R, Z: (np.asarray(V(u)(R, Z)), np.asarray(V(Lu)(R, Z)))


def test_A3_eigen_vs_dense():
    """The eigen path solves exactly the collocation system (dense Kronecker solve)."""
    ra, za = ChebAxis(0.6, 1.4, 20), ChebAxis(-0.45, 0.45, 24)
    ap = AnnulusPoisson(ra, za, [0, 1, 3, 40])
    rng = np.random.default_rng(3)
    worst = 0.0
    for k, m in enumerate(ap.ms):
        f = rng.standard_normal(ap.shape)
        g = {fa: rng.standard_normal(n) for fa, n in ap.face_sizes.items()}
        # consistent corners: in/out own the corners, caps take them from in/out
        g["bot"][0], g["bot"][-1], g["top"][0], g["top"][-1] = g["in"][0], g["out"][0], g["in"][-1], g["out"][-1]
        u = ap.solve(f, g, k)
        # dense: rows of L_m on interior nodes, identity on boundary nodes
        n1, n2 = ap.shape
        Lr = ap.Lr0 - np.diag(m**2 / ap.r**2)
        A = np.kron(Lr, np.eye(n2)) + np.kron(np.eye(n1), za.D2)
        b = f.ravel().copy()
        bnd = np.zeros(ap.shape, bool)
        bnd[0, :] = bnd[-1, :] = bnd[:, 0] = bnd[:, -1] = True
        ub = np.zeros(ap.shape)
        ub[0, :], ub[-1, :], ub[:, 0], ub[:, -1] = g["in"], g["out"], g["bot"], g["top"]
        A[bnd.ravel()] = np.eye(n1 * n2)[bnd.ravel()]
        b[bnd.ravel()] = ub[bnd]
        ud = np.linalg.solve(A, b).reshape(ap.shape)
        e = np.abs(u - ud).max() / np.abs(ud).max()
        # backward error of the eigen solution in the dense system, and kappa(A):
        # the forward difference of two backward-stable solves is ~ kappa eps
        res = np.abs(A @ u.ravel() - b).max() / (np.abs(A).sum(1).max() * np.abs(u).max())
        kap = np.linalg.cond(A)
        worst = max(worst, e / (kap * np.finfo(float).eps))
        print(f"  A3a m={m:3d}: eigen vs dense {e:.1e}; residual {res:.1e}; kappa(A) {kap:.1e}")
        assert res < 1e-14, res
    assert worst < 10, worst  # forward difference within 10 kappa eps


def test_A3_manufactured():
    """Spectral convergence of the Dirichlet core on a non-polynomial u, several m."""
    R_in, R_out, z0 = 0.6, 1.4, 0.45
    res = {}
    for alpha in (0.0,):
        for m in (0, 1, 7, 60, 400):
            errs = []
            for n in (16, 24, 32, 48, 64):
                ra, za = ChebAxis(R_in, R_out, n, alpha), ChebAxis(-z0, z0, n, alpha)
                ap = AnnulusPoisson(ra, za, [m])
                R, Z = ra.s, za.s
                uex, f = _manufactured(m, R_in, R_out, z0)(R, Z)
                u = ap.dirichlet(f, 0)
                errs.append(np.abs(u - uex).max() / np.abs(uex).max())
            res[(alpha, m)] = errs
            print(f"  A3b alpha={alpha} m={m:3d}: " + "  ".join(f"{e:.1e}" for e in errs))
    for (alpha, m), errs in res.items():
        assert errs[0] > 1e3 * errs[3] or m >= 60, (alpha, m, errs)  # convergence is exercised
        assert errs[3] < 1e-10, (alpha, m, errs)


def test_A3_bump_flux():
    """Compact f: sigma on the faces self-converges; sigma -> 0 at the corners."""
    R_in, R_out, z0 = 0.6, 1.4, 0.45
    s0 = 0.04  # 3e-17 on the nearest wall (pre-flight: printed)
    out = {}
    for m in (0, 3, 50):
        sig = []
        for n in (48, 64, 96, 128):
            ra, za = ChebAxis(R_in, R_out, n), ChebAxis(-z0, z0, n)
            ap = AnnulusPoisson(ra, za, [m])
            R, Z = np.meshgrid(ra.s, za.s, indexing="ij")
            f = np.exp(-((R - 0.95) ** 2 + (Z - 0.1) ** 2) / (2 * s0**2))
            fmax_wall = max(np.abs(f[0]).max(), np.abs(f[-1]).max(), np.abs(f[:, 0]).max(), np.abs(f[:, -1]).max())
            u = ap.dirichlet(f, 0)
            fl = ap.flux(u)
            # sigma on fixed test points of each face, by interpolation
            zt, rt = np.linspace(-z0, z0, 41), np.linspace(R_in, R_out, 41)
            sig.append(np.concatenate([za.interp_matrix(zt) @ fl["in"], za.interp_matrix(zt) @ fl["out"],
                                       ra.interp_matrix(rt) @ fl["bot"], ra.interp_matrix(rt) @ fl["top"]]))
        # scale max|u| / (R_out - R_in): at high m, sigma on the walls is itself tiny
        # ((r / r')^{+-m} decay of the radial Green's function), so max|sigma| is no scale
        scale = np.abs(u).max() / (R_out - R_in)
        d = [np.abs(sig[i] - sig[-1]).max() / scale for i in range(3)]
        corner = max(abs(fl["in"][0]), abs(fl["in"][-1]), abs(fl["bot"][0]), abs(fl["top"][-1])) / scale
        out[m] = d
        print(f"  A3c m={m:3d}: max f on wall {fmax_wall:.1e}; sigma vs n=128: "
              + "  ".join(f"{e:.1e}" for e in d) + f"; |sigma| at corners {corner:.1e}; max|sigma|/scale {np.abs(sig[-1]).max() / scale:.1e}")
        assert fmax_wall < 1e-15 and d[-1] < 1e-10 and corner < 1e-12


if __name__ == "__main__":
    names = sys.argv[1:]
    tests = [(k, v) for k, v in sorted(globals().items()) if k.startswith("test_") and callable(v)]
    for k, f in tests:
        if names and not any(n in k for n in names):
            continue
        t0 = time.time()
        print(f"{k} ...", flush=True)
        f()
        print(f"{k} passed ({time.time() - t0:.1f} s)", flush=True)
