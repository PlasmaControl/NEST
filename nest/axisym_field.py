"""Accurate plasma field of an axisymmetric BoundaryPotential current (no box).

For an axisymmetric J = curl T (BoundaryPotential, N = 0) the plasma field splits:

  toroidal  B_phi = mu0 T_zeta / R exactly.  Ampere on the toroidal circle through
            the point, with Stokes for J = curl T on the flat annulus between the
            LCFS and that circle: the poloidal current through it is
            2 pi (T_zeta(point) - T_zeta(LCFS)), and T_zeta = 0 on the LCFS
            (f(1) = 0, d_zeta chi = 0).  Needs a convex cross-section (one crossing
            of the LCFS on the inner side), true here.
  poloidal  (B_R, B_Z) from the toroidal current, as a sum of coaxial current rings
            (Simpson et al. 2001, with K, E by the AGM), dI = J_phi dA =
            -sign(R_rho Z_theta - R_theta Z_rho) sqrt(g) J^zeta drho dtheta.  The
            area integral over the cross-section is done for each target in polar
            coordinates about it in the plane (x, y) = rho (cos theta, sin theta):
            p = p_t + r (cos g, sin g), r in [0, r_b(g)] up to the unit circle, with
            r = r_b s^2 (Gauss-Legendre in s, trapezoid in g).  The 1/d singularity
            of the ring kernel times r dr is then smooth, and its log part s^3 log s.
            dx dy = rho drho dtheta, so dI = (...) dx dy / rho.

Accuracy is spectral in (n_gamma, n_s) for targets inside the plasma; check it by
doubling them (``plasma_field`` with two resolutions).  The radial rule sets it near the edge:
(256, 48) is off by ~7e-9 of rms B at rho > 0.8, (512, 48) the same, (256, 96) at round-off
against (1024, 192) (tests/floor_fixes.py); hence the default n_s = 96.

Speed: the quadrature nodes depend on the target (n_gamma n_s = 12288 per target), and
evaluating the Fourier-Zernike bases of J and of the equilibrium there dominated
(~25 targets/s for L = 64 on an A100).  The three fields the integrand needs,
G = sign sqrt(g) J^zeta / rho, R and Z, are polynomials in rho for each poloidal mode
(G: T_theta = rho^2 x a regular scalar, so d_rho T_theta / rho is regular; R, Z:
Zernike series), so ``_tables`` represents them exactly, once, as Chebyshev series in
rho on [0, 1] times Fourier series in theta (coefficients from the bases at
Gauss-Chebyshev x uniform nodes, checked against the bases at random points to
round-off).  At the nodes they are then one Chebyshev table and one matrix product:
same quadrature, same result to round-off (``method="direct"`` keeps the basis
evaluation, for checks).
"""

import functools
import hashlib
import inspect
import os
import sys

import jax
import jax.numpy as jnp
import numpy as np

from boundary_potential import _potential, _sqrtg_J
from current_potential import MU_0


def _ellipke(m1, n_iter=16):
    """K(m), E(m) from the complementary parameter m1 = 1 - m (accurate as m -> 1)."""
    a, g = jnp.ones_like(m1), jnp.sqrt(m1)
    c2 = 1.0 - m1  # c_0^2 = m
    s, p = 0.5 * c2, 0.5
    for _ in range(n_iter):
        a, g, c = 0.5 * (a + g), jnp.sqrt(a * g), 0.5 * (a - g)
        p = 2 * p
        s = s + p * c * c
    K = jnp.pi / (2 * a)
    return K, K * (1 - s)


def _ring_field(Rs, Zs, I, R, Z):
    """(B_R, B_Z) at (R, Z) of rings of radius Rs at height Zs with current I (broadcast)."""
    zeta = Z - Zs
    a2 = (R - Rs) ** 2 + zeta**2  # alpha^2
    b2 = (R + Rs) ** 2 + zeta**2  # beta^2
    K, E = _ellipke(a2 / b2)
    C = MU_0 * I / jnp.pi
    b = jnp.sqrt(b2)
    BR = C * zeta / (2 * a2 * b * R) * ((Rs**2 + R**2 + zeta**2) * E - a2 * K)
    BZ = C / (2 * a2 * b) * ((Rs**2 - R**2 - zeta**2) * E + a2 * K)
    return BR, BZ


@functools.partial(jax.jit, static_argnums=(0, 1, 5, 6))
def _poloidal_chunk(S, RZ, u, I, pt, n_gamma, n_s, sign):
    """(B_R, B_Z) at the targets with parameter-plane points pt (n_t, 2)."""
    s, ws = np.polynomial.legendre.leggauss(n_s)
    s, ws = (s + 1) / 2, ws / 2
    gam = 2 * np.pi * np.arange(n_gamma) / n_gamma
    e = jnp.asarray(np.stack([np.cos(gam), np.sin(gam)], 1))  # (g, 2)
    pu = pt @ e.T  # (t, g)
    rb = -pu + jnp.sqrt(pu**2 + 1 - jnp.sum(pt**2, 1)[:, None])  # (t, g)
    r = rb[:, :, None] * jnp.asarray(s**2)[None, None]  # (t, g, s)
    x = pt[:, 0, None, None] + r * e[None, :, 0, None]
    y = pt[:, 1, None, None] + r * e[None, :, 1, None]
    rho = jnp.sqrt(x**2 + y**2).ravel()
    theta = jnp.arctan2(y, x).ravel()
    zeta = jnp.zeros_like(rho)
    c = _sqrtg_J(S, u, I, rho, theta, zeta)[2]
    X = RZ.evaluate(rho, theta, zeta)
    w = (2 * rb[:, :, None] ** 2 * jnp.asarray(s**3 * ws)[None, None] * (2 * np.pi / n_gamma)).ravel()
    dI = sign * c * w / jnp.maximum(rho, 1e-300)
    Rt = RZ.evaluate(jnp.sqrt(jnp.sum(pt**2, 1)), jnp.arctan2(pt[:, 1], pt[:, 0]),
                     jnp.zeros(len(pt)))
    shape = r.shape
    BR, BZ = _ring_field(X[:, 0].reshape(shape), X[:, 1].reshape(shape), dI.reshape(shape),
                         Rt[:, 0, None, None], Rt[:, 1, None, None])
    return jnp.sum(BR, axis=(1, 2)), jnp.sum(BZ, axis=(1, 2))


def _tables(current, u, sign, check=2000, tol=1e-12):
    """C[k, f, j]: G (= sign sqrt(g) J^zeta / rho), R, Z (f = 0, 1, 2) as
    sum_{k, j} C[k, f, j] T_k(2 rho - 1) F_j(theta), F_j = 1, cos m theta (m = 1..M),
    sin m theta (m = 1..M); exact for the polynomials they are (checked)."""
    S, RZ, I = current.S, current._RZ, current.I
    eq = current.eq
    M = int(max(current.basis.M, eq.M))
    n = int(max(current.basis.L + 2 * current.edge_power + 4, eq.L)) + 9  # Chebyshev terms
    t = np.cos(np.pi * (np.arange(n) + 0.5) / n)  # Gauss-Chebyshev nodes
    rho = (1 + t) / 2
    n_th = 2 * M + 2
    th = 2 * np.pi * np.arange(n_th) / n_th
    mm = np.arange(-M, M + 1)
    m_order = np.concatenate([[0], np.arange(1, M + 1), -np.arange(1, M + 1)])

    def fields(r, q):
        r, q = jnp.asarray(r), jnp.asarray(q)
        z = jnp.zeros_like(r)
        c = _sqrtg_J(S, u, I, r, q, z)[2]
        X = RZ.evaluate(r, q, z)
        return np.stack([sign * np.asarray(c) / np.asarray(r), np.asarray(X[:, 0]), np.asarray(X[:, 1])], 1)

    def four(q):
        q = np.asarray(q)[:, None]
        return np.where(m_order >= 0, np.cos(np.abs(m_order) * q), np.sin(np.abs(m_order) * q))

    R_, T_ = np.meshgrid(rho, th, indexing="ij")
    V = fields(R_.ravel(), T_.ravel()).reshape(n, n_th, 3)
    Fm = four(th)  # (n_th, 2M + 1), orthogonal columns
    A = np.einsum("jtf,tm->jfm", V, Fm) / np.sum(Fm**2, 0)  # Fourier coefficients at each rho_j
    Tm = np.cos(np.arange(n)[None, :] * np.arccos(t)[:, None])  # T_k(t_j)
    C = np.linalg.solve(Tm, A.reshape(n, -1)).reshape(n, 3, len(mm))
    # exact? against the bases at random points
    rng = np.random.default_rng(0)
    rc, qc = np.sqrt(rng.random(check)), 2 * np.pi * rng.random(check)
    ref = fields(rc, qc)
    Tc = np.polynomial.chebyshev.chebvander(2 * rc - 1, n - 1)
    got = np.einsum("pk,kfm,pm->pf", Tc, C, four(qc))
    err = np.max(np.abs(got - ref), 0) / np.max(np.abs(ref), 0)
    if np.max(err) > tol:
        raise RuntimeError(f"axisym_field tables not exact: rel. errors (G, R, Z) {err}")
    return C, M


def _cheb(t, n):
    """T_0..T_{n-1}(t), shape (len(t), n), by the three-term recurrence."""
    T = [jnp.ones_like(t), t]
    for _ in range(2, n):
        T.append(2 * t * T[-1] - T[-2])
    return jnp.stack(T[:n], 1)


@functools.partial(jax.jit, static_argnums=(3, 4, 5))
def _poloidal_chunk_tab(C, pt, Rt, n_gamma, n_s, M):
    """(B_R, B_Z) at the targets (parameter-plane points pt, positions Rt) from the tables."""
    s, ws = np.polynomial.legendre.leggauss(n_s)
    s, ws = (s + 1) / 2, ws / 2
    gam = 2 * np.pi * np.arange(n_gamma) / n_gamma
    e = jnp.asarray(np.stack([np.cos(gam), np.sin(gam)], 1))  # (g, 2)
    pu = pt @ e.T  # (t, g)
    rb = -pu + jnp.sqrt(pu**2 + 1 - jnp.sum(pt**2, 1)[:, None])  # (t, g)
    r = rb[:, :, None] * jnp.asarray(s**2)[None, None]  # (t, g, s)
    x = pt[:, 0, None, None] + r * e[None, :, 0, None]
    y = pt[:, 1, None, None] + r * e[None, :, 1, None]
    rho = jnp.sqrt(x**2 + y**2).ravel()
    theta = jnp.arctan2(y, x).ravel()
    mt = theta[:, None] * jnp.arange(1, M + 1, dtype=rho.dtype)
    F = jnp.concatenate([jnp.ones_like(theta)[:, None], jnp.cos(mt), jnp.sin(mt)], 1)  # 1, cos, sin
    n = C.shape[0]
    V = (_cheb(2 * rho - 1, n) @ C.reshape(n, -1)).reshape(len(rho), 3, -1)
    G, Rs, Zs = jnp.sum(V * F[:, None, :], axis=2).T  # elementwise: no batched tiny products
    w = (2 * rb[:, :, None] ** 2 * jnp.asarray(s**3 * ws)[None, None] * (2 * np.pi / n_gamma)).ravel()
    shape = r.shape
    BR, BZ = _ring_field(Rs.reshape(shape), Zs.reshape(shape), (G * w).reshape(shape),
                         Rt[:, 0, None, None], Rt[:, 1, None, None])
    return jnp.sum(BR, axis=(1, 2)), jnp.sum(BZ, axis=(1, 2))


def _chunk(chunk, n_gamma, n_s):
    """Targets per chunk: ``chunk`` at (256, 48), fewer for finer quadratures (same node count)."""
    return max(1, int(chunk * 256 * 48 // (n_gamma * n_s)))


def plasma_field(current, rtz, u=None, n_gamma=256, n_s=96, chunk=32, method="tables"):
    """The plasma field of ``current`` (a BoundaryPotential with N = 0) at the flux
    points rtz (n, 3; rho < 1), in (R, phi, Z) components, shape (n, 3).
    ``method``: "tables" (default) or "direct" (the bases at every node; slow).
    ``chunk``: targets at a time at (256, 48), scaled down for finer quadratures."""
    chunk = _chunk(chunk, n_gamma, n_s)
    S, RZ, I = current.S, current._RZ, current.I
    u = jnp.asarray(current.u if u is None else u)
    rtz = np.asarray(rtz)
    rho, theta = rtz[:, 0], rtz[:, 1]
    # B_phi exactly from T_zeta
    T = np.asarray(jax.jit(functools.partial(_potential, S))(u, I, jnp.asarray(rho), jnp.asarray(theta),
                                                             jnp.zeros(len(rho))))
    Xt_all = np.asarray(RZ.evaluate(jnp.asarray(rho), jnp.asarray(theta), jnp.zeros(len(rho))))
    R = Xt_all[:, 0]
    Bphi = MU_0 * T[:, 1] / R
    # the orientation of (rho, theta) in the (R, Z) plane, once
    r0, t0, z0 = jnp.asarray([0.5]), jnp.asarray([0.3]), jnp.asarray([0.0])
    Xr, Xt = np.asarray(RZ.evaluate(r0, t0, z0, dr=1)), np.asarray(RZ.evaluate(r0, t0, z0, dt=1))
    sign = -float(np.sign(Xr[0, 0] * Xt[0, 1] - Xt[0, 0] * Xr[0, 1]))
    pts = np.stack([rho * np.cos(theta), rho * np.sin(theta)], 1)
    if method == "tables":
        C, M = _tables(current, u, sign)
        C = jnp.asarray(C)
    elif method != "direct":
        raise ValueError(f"method {method!r}")
    out = []
    for i in range(0, len(pts), chunk):
        p, Xp = pts[i : i + chunk], Xt_all[i : i + chunk]
        n = len(p)
        if n < chunk:  # one compiled shape
            p = np.concatenate([p, np.repeat(p[-1:], chunk - n, 0)])
            Xp = np.concatenate([Xp, np.repeat(Xp[-1:], chunk - n, 0)])
        if method == "tables":
            BR, BZ = _poloidal_chunk_tab(C, jnp.asarray(p), jnp.asarray(Xp), n_gamma, n_s, M)
        else:
            BR, BZ = _poloidal_chunk(S, RZ, u, I, jnp.asarray(p), n_gamma, n_s, sign)
        out.append(np.stack([np.asarray(BR)[:n], np.asarray(BZ)[:n]], 1))
    BRZ = np.concatenate(out)
    return np.stack([BRZ[:, 0], Bphi, BRZ[:, 1]], 1)


def _transform(V, Fm, Tm):
    """Values V[..., j_rho, i_theta] on the table grid -> C[..., k, m] (Chebyshev x Fourier)."""
    A = np.einsum("...jt,tm->...jm", V, Fm) / np.sum(Fm**2, 0)
    return np.linalg.solve(Tm, A)  # along j_rho (Tm is square, batched over the leading axes)


@functools.partial(jax.jit, static_argnums=(6, 7, 8, 9))
def _matrix_chunk(Cgeo, Cvec, e_j, e_k, pt, Rt, n_gamma, n_s, M, n_u):
    """d(B_R, B_Z)/du at the targets, (n_u, t, 2), from each target's quadrature moments
    M_t[f, c, j] = sum_q K_f(t, q) w_q T_c(rho_q) F_j(theta_q), then
    dB_f/du_k = sum_{entries e of k} C_e . M_t[f, :, j_e]."""
    s, ws = np.polynomial.legendre.leggauss(n_s)
    s, ws = (s + 1) / 2, ws / 2
    gam = 2 * np.pi * np.arange(n_gamma) / n_gamma
    e = jnp.asarray(np.stack([np.cos(gam), np.sin(gam)], 1))
    pu = pt @ e.T
    rb = -pu + jnp.sqrt(pu**2 + 1 - jnp.sum(pt**2, 1)[:, None])
    r = rb[:, :, None] * jnp.asarray(s**2)[None, None]
    x = pt[:, 0, None, None] + r * e[None, :, 0, None]
    y = pt[:, 1, None, None] + r * e[None, :, 1, None]
    t_n, Q = pt.shape[0], n_gamma * n_s
    rho = jnp.sqrt(x**2 + y**2).reshape(t_n, Q)
    theta = jnp.arctan2(y, x).reshape(t_n, Q)
    mt = theta[..., None] * jnp.arange(1, M + 1, dtype=rho.dtype)
    F = jnp.concatenate([jnp.ones_like(theta)[..., None], jnp.cos(mt), jnp.sin(mt)], -1)  # (t, Q, J)
    n = Cgeo.shape[0]
    Tq = _cheb(2 * rho.ravel() - 1, n).reshape(t_n, Q, n)
    RZ = jnp.sum((Tq @ Cgeo.reshape(n, -1)).reshape(t_n, Q, 2, -1) * F[:, :, None, :], axis=-1)
    w = (2 * rb[:, :, None] ** 2 * jnp.asarray(s**3 * ws)[None, None] * (2 * np.pi / n_gamma)).reshape(t_n, Q)
    BR, BZ = _ring_field(RZ[..., 0], RZ[..., 1], w, Rt[:, 0, None], Rt[:, 1, None])  # per unit G
    KT = jnp.concatenate([BR[..., None] * Tq, BZ[..., None] * Tq], -1)  # (t, Q, 2n)
    Mt = jnp.einsum("tqc,tqj->tcj", KT, F).reshape(t_n, 2, n, -1)  # the moments
    out = jnp.einsum("tfce,ec->etf", Mt[..., e_j], Cvec)  # (E, t, 2)
    return jax.ops.segment_sum(out, e_k, num_segments=n_u)


def plasma_field_matrix(current, rtz, n_gamma=256, n_s=96, chunk=32, batch=512, drop=1e-14,
                        verbose=False):
    """B_plasma per unit coefficient, at the flux points rtz: (n_u, n, 3) in (R, phi, Z)
    components, so that plasma_field(current, rtz, u) = einsum("k,knc->nc", u, .) +
    plasma_field(current, rtz, u=0) (the net-current part), to round-off.

    The same quadrature as ``plasma_field``, for every coefficient at once:
      - each coefficient's G_k = sign sqrt(g) J^zeta_k / rho (I = 0) as Chebyshev x
        Fourier coefficients on the table grid (exact; checked), kept as the entries
        (k, Fourier column j, Chebyshev vector) above ``drop`` x the coefficient's largest
        (one entry per coefficient: each basis function is one Fourier mode);
      - per target, the moments of its quadrature against T_c F_j (one batched matrix
        product), contracted with those entries;
      - B_phi = mu0 T_zeta / R from the T_zeta basis matrix at the targets.
    """
    import time

    t0 = time.time()
    chunk = _chunk(chunk, n_gamma, n_s)
    S, RZ, I = current.S, current._RZ, current.I
    n_u = current.n_free
    rtz = np.asarray(rtz)
    rho_t, th_t = rtz[:, 0], rtz[:, 1]
    r0, a0, z0 = jnp.asarray([0.5]), jnp.asarray([0.3]), jnp.asarray([0.0])
    Xr, Xt = np.asarray(RZ.evaluate(r0, a0, z0, dr=1)), np.asarray(RZ.evaluate(r0, a0, z0, dt=1))
    sign = -float(np.sign(Xr[0, 0] * Xt[0, 1] - Xt[0, 0] * Xr[0, 1]))
    Cg, M = _tables(current, jnp.zeros(n_u), sign)  # R, Z rows: the geometry
    n = Cg.shape[0]
    # the table grid (as _tables)
    tch = np.cos(np.pi * (np.arange(n) + 0.5) / n)
    rho = (1 + tch) / 2
    n_th = 2 * M + 2
    th = 2 * np.pi * np.arange(n_th) / n_th
    m_order = np.concatenate([[0], np.arange(1, M + 1), -np.arange(1, M + 1)])
    Fm = np.where(m_order >= 0, np.cos(np.abs(m_order) * th[:, None]), np.sin(np.abs(m_order) * th[:, None]))
    Tm = np.cos(np.arange(n)[None, :] * np.arccos(tch)[:, None])
    R_, T_ = np.meshgrid(rho, th, indexing="ij")
    rg, tg = jnp.asarray(R_.ravel()), jnp.asarray(T_.ravel())

    def gfield(u, r, q):  # G of u (no net current) at points
        return sign * _sqrtg_J(S, u, 0.0, r, q, jnp.zeros_like(r))[2] / r

    gv = jax.jit(jax.vmap(gfield, in_axes=(0, None, None)))
    eye = np.eye(n_u)
    V = np.concatenate([np.asarray(gv(jnp.asarray(eye[i:i + batch]), rg, tg))
                        for i in range(0, n_u, batch)]).reshape(n_u, n, n_th)
    Cb = _transform(V, Fm, Tm)  # (n_u, n, J)
    del V
    # exact? a random combination against the basis at random points
    rng = np.random.default_rng(1)
    uc = rng.standard_normal(n_u)
    rc, qc = np.sqrt(rng.random(2000)), 2 * np.pi * rng.random(2000)
    ref = np.asarray(gfield(jnp.asarray(uc), jnp.asarray(rc), jnp.asarray(qc)))
    Fc = np.where(m_order >= 0, np.cos(np.abs(m_order) * qc[:, None]), np.sin(np.abs(m_order) * qc[:, None]))
    got = np.einsum("pc,cj,pj->p", np.polynomial.chebyshev.chebvander(2 * rc - 1, n - 1),
                    np.einsum("k,kcj->cj", uc, Cb), Fc)
    err_tab = np.max(np.abs(got - ref)) / np.max(np.abs(ref))
    if err_tab > 1e-12:
        raise RuntimeError(f"plasma_field_matrix: basis tables not exact ({err_tab:.1e})")
    # the entries: (k, j) with a non-negligible Chebyshev vector
    nrm = np.max(np.abs(Cb), axis=1)  # (n_u, J)
    big = nrm.max(1, keepdims=True)
    keep = nrm > drop * np.where(big > 0, big, 1.0)
    keep &= big > 0
    e_k, e_j = np.nonzero(keep)
    dropped = float(np.max(np.where(keep, 0.0, nrm) / np.where(big > 0, big, 1.0)))
    Cvec = Cb[e_k, :, e_j]  # (E, n)
    del Cb
    if verbose:
        print(f"    exact G: tables {n} x {2 * M + 1}, {len(e_k)} entries for {n_u} coefficients "
              f"(largest dropped {dropped:.1e}), exact to {err_tab:.1e}; {time.time() - t0:.0f} s",
              flush=True)
    # poloidal field: moments per target
    Cgeo = jnp.asarray(Cg[:, 1:3])
    Cv, ej, ek = jnp.asarray(Cvec), jnp.asarray(e_j), jnp.asarray(e_k)
    pts = np.stack([rho_t * np.cos(th_t), rho_t * np.sin(th_t)], 1)
    Xt_all = np.asarray(RZ.evaluate(jnp.asarray(rho_t), jnp.asarray(th_t), jnp.zeros(len(rho_t))))
    out = np.empty((n_u, len(pts), 3))
    for i in range(0, len(pts), chunk):
        p, Xp = pts[i:i + chunk], Xt_all[i:i + chunk]
        m = len(p)
        if m < chunk:
            p = np.concatenate([p, np.repeat(p[-1:], chunk - m, 0)])
            Xp = np.concatenate([Xp, np.repeat(Xp[-1:], chunk - m, 0)])
        D = np.asarray(_matrix_chunk(Cgeo, Cv, ej, ek, jnp.asarray(p), jnp.asarray(Xp), n_gamma, n_s, M, n_u))
        out[:, i:i + m, 0], out[:, i:i + m, 2] = D[:, :m, 0], D[:, :m, 1]
    if verbose:
        print(f"    exact G: poloidal field at {len(pts)} targets, {time.time() - t0:.0f} s", flush=True)
    # toroidal field: mu0 T_zeta / R per coefficient
    Tz = jax.jit(jax.vmap(lambda u, r, q: _potential(S, u, 0.0, r, q, jnp.zeros_like(r))[:, 1],
                          in_axes=(0, None, None)))
    for j in range(0, len(pts), 8192):
        r_, q_ = jnp.asarray(rho_t[j:j + 8192]), jnp.asarray(th_t[j:j + 8192])
        for i in range(0, n_u, batch):
            out[i:i + batch, j:j + 8192, 1] = MU_0 * np.asarray(Tz(jnp.asarray(eye[i:i + batch]), r_, q_)) \
                / Xt_all[j:j + 8192, 0]
    if verbose:
        print(f"    exact G: done, {time.time() - t0:.0f} s", flush=True)
    return out


def plasma_field_cached(current, rtz, u=None, n_gamma=256, n_s=96, where=None):
    """``plasma_field`` stored on disk (<repository>/.cache/AxisymB_<hash>.npz), keyed
    by the coefficients, the points, the resolution and this module's source."""
    u = np.asarray(current.u if u is None else u, dtype=float)
    rtz = np.asarray(rtz, dtype=float)
    where = where or os.environ.get("NEST_CACHE", os.path.join(
        os.path.dirname(os.path.dirname(os.path.abspath(__file__))), ".cache"))
    eq = current.eq
    h = hashlib.sha256()
    for part in (u.tobytes(), rtz.tobytes(), np.asarray(eq.R_lmn).tobytes(), np.asarray(eq.Z_lmn).tobytes(),
                 repr((current.basis.L, current.basis.M, current.basis.N, current.edge_power,
                       float(current.I), n_gamma, n_s)).encode(),
                 inspect.getsource(sys.modules[__name__]).encode()):
        h.update(part)
    path = os.path.join(where, f"AxisymB_{h.hexdigest()[:24]}.npz")
    if os.path.exists(path):
        return np.load(path)["B"]
    B = plasma_field(current, rtz, u=u, n_gamma=n_gamma, n_s=n_s)
    os.makedirs(where, exist_ok=True)
    tmp = path + f".{os.getpid()}.tmp.npz"
    np.savez(tmp, B=B)
    os.replace(tmp, path)
    return B
