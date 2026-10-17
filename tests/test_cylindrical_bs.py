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


# ------------------------------------------------------------------ A4
from toroidal_kernel import _Q_start, toroidal_kernel, toroidal_Q  # noqa: E402

A4_M = (0, 1, 2, 5, 20, 100, 500, 1000)
A4_X = (1e-12, 1e-8, 1e-4, 1e-2, 1.0, 10.0, 100.0)  # chi - 1


def _mp_Q(m, x, dps=40):
    import mpmath as mp
    mp.mp.dps = dps
    return mp.legenq(mp.mpf(m) - mp.mpf(1) / 2, 0, 1 + mp.mpf(x), type=3).real


def test_A4_vs_mpmath():
    """Q_{m-1/2}(chi) against mpmath on the plan's (m, chi - 1) grid, for m_max = 1000."""
    x = np.array(A4_X)
    Q = toroidal_Q(x, 1000)
    worst = {}
    for m in A4_M:
        for j, xv in enumerate(A4_X):
            ref = float(_mp_Q(m, xv))
            if ref == 0.0 or abs(ref) < 1e-290:
                continue  # underflow in double: both are 0 to double range
            e = abs(Q[m, j] - ref) / abs(ref)
            worst[(m, xv)] = e
    for m in A4_M:
        print(f"  A4 m={m:5d}: " + "  ".join(f"{worst.get((m, xv), np.nan):.0e}" for xv in A4_X))
    bad = {k: v for k, v in worst.items() if v > (1e-10 if k[1] <= 1e-8 else 1e-12)}
    assert not bad, bad
    # elliptic starting values for m = 0, 1 on a dense chi range
    xs = np.logspace(-14, 3, 40)
    Q0, Q1 = _Q_start(xs)
    e0 = max(abs(Q0[i] - float(_mp_Q(0, v))) / abs(float(_mp_Q(0, v))) for i, v in enumerate(xs))
    e1 = max(abs(Q1[i] - float(_mp_Q(1, v))) / abs(float(_mp_Q(1, v)))
             for i, v in enumerate(xs) if v < 1.13 - 1)  # the forward range (eta <= 1/2)
    print(f"  A4 elliptic: Q_-1/2 {e0:.1e} (chi-1 1e-14..1e3); Q_1/2 {e1:.1e} (forward range)")
    assert e0 < 1e-13 and e1 < 1e-13


def test_A4_switch_and_kernel():
    """Continuity across the forward/Miller switch, m_max independence, and k_m against
    the direct t-integral (normalization)."""
    import mpmath as mp
    # forward/Miller switch at m_max eta = 2: compare the same chi computed both ways
    m_max = 400
    eta = np.array([1.9, 1.99, 2.01, 2.1]) / m_max
    x = np.cosh(eta) - 1
    Qa = toroidal_Q(x, m_max)          # straddles the switch
    Qb = toroidal_Q(x, 2 * m_max)[: m_max + 1]  # all Miller-free / other branch
    e = np.abs(Qa - Qb).max(axis=0) / np.abs(Qb).max(axis=0)
    print(f"  A4 switch: max rel diff (m_max 400 vs 800) {e.max():.1e}")
    assert e.max() < 1e-12
    # k_m against (1 / 2 pi) int cos(m t) / |x - y(t)| dt, separated and close pairs
    mp.mp.dps = 60  # k_60 of a separated pair is 1e-30 of the integrand: needs > 30 digits
    for (r, z, rp, zp) in [(1.0, 0.1, 0.7, -0.3), (1.0, 0.0, 1.0 + 1e-3, 2e-3), (0.65, 0.4, 1.35, -0.4)]:
        k = toroidal_kernel(r, z, rp, zp, 60)
        for m in (0, 3, 17, 60):
            f = lambda t: mp.cos(m * t) / mp.sqrt((r - rp) ** 2 + (z - zp) ** 2  # noqa: E731
                                                  + 4 * r * rp * mp.sin(t / 2) ** 2)
            ref = float(mp.quad(f, [0, mp.mpf("1e-3"), mp.mpf("1e-2"), 0.1, mp.pi]) / mp.pi)
            e = abs(k[m] - ref) / abs(ref)
            print(f"  A4 k_m ({r}, {z}; {rp}, {zp}) m={m:3d}: {k[m]: .6e} vs quad, rel {e:.1e}")
            assert e < 1e-12


# ------------------------------------------------------------------ A5
from single_layer import single_layer_matrix  # noqa: E402

A5_BOX = dict(R_in=0.6, R_out=1.4, z0=0.45)


def _a5_setup(n_r=24, n_z=28):
    b = A5_BOX
    ra, za = ChebAxis(b["R_in"], b["R_out"], n_r), ChebAxis(-b["z0"], b["z0"], n_z)
    ap = AnnulusPoisson(ra, za, [0])

    # a smooth density per face, nonzero at the corners and at every target
    def sig(face, l):
        c = {"in": (1.0, 0.3), "out": (-0.7, 0.5), "bot": (0.4, -0.8), "top": (1.2, 0.2)}[face]
        return c[0] + c[1] * np.cos(3 * l + 0.4) + 0.2 * l**2

    axes = {"in": za, "out": za, "bot": ra, "top": ra}
    sigma = ap.pack({f: sig(f, axes[f].s) for f in ("in", "out", "bot", "top")})
    return ap, sig, sigma


def _face_param(face, l):
    b = A5_BOX
    return {"in": (b["R_in"], l), "out": (b["R_out"], l), "bot": (l, -b["z0"]), "top": (l, b["z0"])}[face]


def test_A5_normalization_3d():
    """S_m sigma at interior targets against the 3-D surface integral
    -(1/4 pi) int sigma(l') cos(m phi') / |x - y| r' dphi' dl' (scipy dblquad).

    The density here is the face interpolant itself, so only the operator is tested."""
    from scipy.integrate import dblquad
    ap, sig, sigma = _a5_setup()
    b = A5_BOX
    lims = {"in": (-b["z0"], b["z0"]), "out": (-b["z0"], b["z0"]),
            "bot": (b["R_in"], b["R_out"]), "top": (b["R_in"], b["R_out"])}
    targets = np.array([[1.0, 0.1], [0.75, -0.3], [1.3, 0.35]])
    ms = [0, 1, 2, 5]
    S = single_layer_matrix(ap, ms, targets)
    worst = 0.0
    for i, (rt, zt) in enumerate(targets):
        for k, m in enumerate(ms):
            ref = 0.0
            for f in ("in", "out", "bot", "top"):
                def g(phi, l, f=f, m=m):
                    rp, zp = _face_param(f, l)
                    d = np.sqrt(rt**2 + rp**2 - 2 * rt * rp * np.cos(phi) + (zt - zp) ** 2)
                    return sig(f, l) * np.cos(m * phi) * rp / d
                ref += dblquad(g, *lims[f], 0, 2 * np.pi, epsabs=1e-14, epsrel=1e-13)[0]
            ref *= -1 / (4 * np.pi)
            val = S[k, i] @ sigma
            e = abs(val - ref) / abs(ref)
            worst = max(worst, e)
            print(f"  A5 3-D target ({rt}, {zt}) m={m}: S sigma {val: .12e}  ref {ref: .12e}  rel {e:.1e}")
    assert worst < 1e-10, worst


def test_A5_on_surface():
    """Boundary targets (face interiors, near corners, corners) against 1-D adaptive
    quadrature of -1/2 int r' k_m sigma dl' with breakpoints; corner rows agree; the
    entries are converged in the panel rule."""
    from scipy.integrate import quad
    ap, sig, sigma = _a5_setup()
    b = A5_BOX
    pts, face_id = ap.boundary_points()
    sel = [3, 14, 0, ap.face_sizes["in"] - 1, ap.face_sizes["in"] + 1,
           2 * ap.face_sizes["in"] + 5, 2 * ap.face_sizes["in"] + ap.face_sizes["bot"] + 22]
    targets = pts[sel]
    targets = np.vstack([targets, [[b["R_in"] + 1e-4, b["z0"] - 2e-4]]])  # just inside a corner
    ms = [0, 1, 2, 3, 9, 60, 300]
    S = single_layer_matrix(ap, ms, targets)
    lims = {"in": (-b["z0"], b["z0"]), "out": (-b["z0"], b["z0"]),
            "bot": (b["R_in"], b["R_out"]), "top": (b["R_in"], b["R_out"])}
    from toroidal_kernel import toroidal_kernel
    worst = 0.0
    for i, (rt, zt) in enumerate(targets):
        vals, refs = S[:, i] @ sigma, np.zeros(len(ms))
        for f in ("in", "out", "bot", "top"):
            a, c = lims[f]
            along = zt if f in ("in", "out") else rt
            brk = [x for x in (along, a, c) if a < x < c]
            for k, m in enumerate(ms):
                def g(l, f=f, m=m):
                    rp, zp = _face_param(f, l)
                    if rp == rt and zp == zt:
                        return 0.0
                    return -0.5 * rp * toroidal_kernel(rt, zt, rp, zp, m)[m] * sig(f, l)
                refs[k] += quad(g, a, c, points=brk or None, limit=400, epsabs=1e-15, epsrel=1e-13)[0]
        scale = np.abs(refs).max()  # the m = 0 value: high-m potentials are small
        e = np.abs(vals - refs) / np.maximum(np.abs(refs), 1e-3 * scale)
        worst = max(worst, e.max())
        print(f"  A5 surface target ({rt:.4f}, {zt:.4f}): rel err by m " + " ".join(f"{x:.0e}" for x in e))
    assert worst < 1e-10, worst
    # the corner (R_in, -z0) is node 0 of 'in' and node 0 of 'bot': identical rows
    nin = ap.face_sizes["in"]
    Sc = single_layer_matrix(ap, [0, 4], pts[[0, 2 * nin]])
    assert np.allclose(pts[0], pts[2 * nin])
    dc = np.abs(Sc[:, 0] - Sc[:, 1]).max() / np.abs(Sc).max()
    # panel-rule convergence
    S2 = single_layer_matrix(ap, ms, targets, n_gl=24, ratio=0.1)
    dp = (np.abs(S2 - S).max(axis=(1, 2)) / np.abs(S2).max(axis=(1, 2)))
    print(f"  A5 corner rows differ by {dc:.1e}; panel rule (16, 0.15) vs (24, 0.1) by m: "
          + " ".join(f"{x:.0e}" for x in dp))
    # entries: the floor at high m is the recurrence's accuracy near chi = 1 (gate A4, ~1e-12)
    assert dc < 1e-13 and dp.max() < 1e-11


# ------------------------------------------------------------------ A6
def test_A6_gaussian_charges():
    """Free-space scalar solve against NFP-periodic, z-asymmetric Gaussian charges, both
    mode shifts (the full sweep, to 1e-12, is tests/cyl_a6.py)."""
    sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
    from cyl_a6 import run_case
    r, _ = run_case(64, 151)
    for s in (0, 1):
        e = r[f"shift{s}"]
        print(f"  A6 n=64 N_phi=151 shift {s}: g {e['boundary_g']:.1e} nodes {e['nodes']:.1e} "
              f"off-node {e['off_node']:.1e}")
        assert max(e[k] for k in ('boundary_g', 'nodes', 'off_node')) < 1e-6
    assert r["wall_tail"] < 1e-16


# ------------------------------------------------------------------ B1
def _jax64():
    import jax
    jax.config.update("jax_enable_x64", True)
    return jax


def test_B1_jax_vs_numpy():
    """The JAX potential equals the NumPy reference (FreeSpaceAnnulus) mode for mode,
    for A_z (shift 0) and A_+ (shift 1), with a chunk that does not divide the modes."""
    _jax64()
    import jax.numpy as jnp
    from scipy.constants import mu_0
    from cylindrical_bs import CylindricalSolver, potential
    ra, za = ChebAxis(0.6, 1.4, 20), ChebAxis(-0.45, 0.45, 22)
    sol = CylindricalSolver(ra, za, NFP=3, N_phi=15, chunk=4, keep_numpy=True)
    rng = np.random.default_rng(5)
    J = rng.standard_normal(sol.shape + (3,))
    A = np.asarray(potential(sol, jnp.asarray(J)))
    Az = sol.fs.solve(-mu_0 * J[..., 2], shift=0)
    Ac = sol.fs.solve(-mu_0 * (J[..., 0] + 1j * J[..., 1]), shift=1)
    ez = np.abs(A[..., 2] - Az).max() / np.abs(Az).max()
    ec = np.abs(A[..., 0] + 1j * A[..., 1] - Ac).max() / np.abs(Ac).max()
    print(f"  B1 JAX vs NumPy: A_z {ez:.1e}, A_r + i A_phi {ec:.1e}")
    assert ez < 1e-13 and ec < 1e-13


# ------------------------------------------------------------------ B2
def _b2_case(n, N_phi, NFP=3):
    _jax64()
    import jax.numpy as jnp
    sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
    from cyl_manufactured import fields, wall_values
    from cylindrical_bs import CylindricalSolver, field, potential
    ra, za = ChebAxis(0.6, 1.4, n), ChebAxis(-0.45, 0.45, n)
    t0 = time.time()
    sol = CylindricalSolver(ra, za, NFP=NFP, N_phi=N_phi)
    t_pre = time.time() - t0
    ex = fields(sol.nodes(), NFP)
    t0 = time.time()
    B = np.asarray(field(sol, jnp.asarray(ex["J"])))
    t_call = time.time() - t0
    A = np.asarray(potential(sol, jnp.asarray(ex["J"])))
    eB = np.abs(B - ex["B"]).max() / np.abs(ex["B"]).max()
    eA = np.abs(A - ex["A"]).max() / np.abs(ex["A"]).max()
    # interior only (rho of the support): where B matters for NEST
    return dict(n=n, N_phi=N_phi, eB=eB, eA=eA, wall_J=wall_values(ex["J"]) / np.abs(ex["J"]).max(),
                pre=t_pre, call=t_call), sol, ex, B


def test_B2_manufactured():
    """B from J = curl curl curl psi / mu0 against the exact curl curl psi, everywhere."""
    rows = []
    for n, N_phi in ((40, 31), (56, 45), (72, 61), (88, 75), (104, 91)):
        r, *_ = _b2_case(n, N_phi)
        rows.append(r)
        print(f"  B2 n={n} N_phi={N_phi}: |B - B_exact|/max {r['eB']:.1e}, A {r['eA']:.1e}; "
              f"J on walls / max {r['wall_J']:.0e}; precompute {r['pre']:.1f} s, first call {r['call']:.1f} s",
              flush=True)
    assert rows[-1]["wall_J"] < 1e-15
    assert rows[0]["eB"] > 1e2 * rows[-1]["eB"]  # convergence exercised
    assert rows[-1]["eB"] < 1e-9 and rows[-1]["eA"] < 1e-9


# ------------------------------------------------------------------ B3
def test_B3_identities():
    """div B = 0; a gradient added to J leaves B unchanged and appears as
    curl B - mu0 J' = -mu0 grad chi; curl B = mu0 J for the solenoidal J."""
    import jax.numpy as jnp
    from scipy.constants import mu_0
    from cylindrical_bs import field
    r, sol, ex, B = _b2_case(104, 91)  # B accurate to ~1e-12 here (gate B2)
    J = ex["J"]
    gc = ex["grad_chi"] * (0.3 * np.abs(J).max() / np.abs(ex["grad_chi"]).max())  # |grad chi| = 0.3 max|J|
    Bs = np.abs(B).max()
    # derivative scale: max |grad B| ~ max |mu0 J|
    dscale = mu_0 * np.abs(J).max()
    divB = np.abs(np.asarray(sol.div_grid(jnp.asarray(B)))).max() / dscale
    curl_res = np.abs(np.asarray(sol.curl_grid(jnp.asarray(B))) - mu_0 * J).max() / dscale
    J2 = J + gc
    B2 = np.asarray(field(sol, jnp.asarray(J2)))
    dB = np.abs(B2 - B).max() / Bs
    res2 = np.asarray(sol.curl_grid(jnp.asarray(B2))) - mu_0 * J2
    pred = np.abs(res2 + mu_0 * gc).max() / dscale
    print(f"  B3 n=104 N_phi=91: div B {divB:.1e}; curl B - mu0 J {curl_res:.1e} (of mu0 max|J|); "
          f"B(J + grad chi) - B(J) {dB:.1e} (|grad chi|/|J| = {np.abs(gc).max() / np.abs(J).max():.1e}); "
          f"curl B - mu0 J' + mu0 grad chi {pred:.1e}")
    assert divB < 1e-10 and curl_res < 1e-8 and dB < 1e-10 and pred < 1e-8


def test_B3_adjoint_and_grads():
    """Adjoint identity <L J, b> = <J, L^T b> (jax.vjp) and check_grads on a small grid."""
    import jax
    import jax.numpy as jnp
    from jax.test_util import check_grads
    from cylindrical_bs import CylindricalSolver, field
    _jax64()
    sol = CylindricalSolver(ChebAxis(0.6, 1.4, 16), ChebAxis(-0.45, 0.45, 18), NFP=2, N_phi=9, chunk=5)
    rng = np.random.default_rng(11)
    J = jnp.asarray(rng.standard_normal(sol.shape + (3,)))
    b = jnp.asarray(rng.standard_normal(sol.shape + (3,)))
    LJ, vjp = jax.vjp(lambda x: field(sol, x), J)
    lhs, rhs = float(jnp.sum(LJ * b)), float(jnp.sum(J * vjp(b)[0]))
    e = abs(lhs - rhs) / abs(lhs)
    J2 = jnp.asarray(rng.standard_normal(sol.shape + (3,)))
    lin = float(jnp.abs(field(sol, 2 * J - 3 * J2) - 2 * field(sol, J) + 3 * field(sol, J2)).max()
                / jnp.abs(field(sol, J)).max())
    print(f"  B3 adjoint: <LJ, b> {lhs:.12e} vs <J, L^T b> {rhs:.12e}, rel {e:.1e}; linearity {lin:.1e}")
    assert e < 1e-12 and lin < 1e-13
    check_grads(lambda x: field(sol, x), (J,), order=1, modes=("fwd", "rev"))
    print("  B3 check_grads (fwd, rev): passed")


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
