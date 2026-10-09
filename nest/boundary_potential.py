"""Boundary-adapted current potential, J = curl T with J = 0 on the LCFS.

The representation of J_potential_notes.pdf, in the flux coordinates
(rho, theta, zeta) of the fixed DESC equilibrium and the radial gauge T_rho = 0,
T = T_theta grad(theta) + T_zeta grad(zeta):

    T_theta = I / 2pi h_0(rho) + d_theta chi + rho^2 f(rho) a(rho, theta, zeta)
    T_zeta  =                    d_zeta chi  +       f(rho) b(rho, theta, zeta)

    sqrt(g) J^rho   = d_theta T_zeta - d_zeta T_theta = f (d_theta b - rho^2 d_zeta a)
    sqrt(g) J^theta = - d_rho T_zeta
    sqrt(g) J^zeta  =   d_rho T_theta

with
    f(rho)  = (1 - rho^2)^s                    (s = ``edge_power``, the notes' s = 2)
    a, b    Fourier-Zernike series (one basis, ``basis``)
    chi     = sum_{m != 0, all n} chi_mn h_|m|(rho) F_m(theta) F_n(zeta)
    h_|m|   = rho^q sum_{j<s} binom(-q/2, j) (rho^2 - 1)^j,   q = |m| + 2,

F_m, F_n DESC's real Fourier functions (cos for m >= 0, sin for m < 0). For
s = 2, h_|m| = rho^q ((q + 2) - q rho^2) / 2 is the notes' h_m and
h_0 = rho^2 (2 - rho^2).

Properties, for every coefficient vector u = (chi, a, b) (no constraint to keep):
  - div J = 0: J is a curl.
  - LCFS: h_q(1) = 1 and h_q^(j)(1) = 0 for j = 1..s-1, f = O((1 - rho)^s), and
    the chi terms cancel in sqrt(g) J^rho, so every sqrt(g) J^i, hence J, is
    O((1 - rho)^(s-1)) at rho = 1 (zero on the LCFS for s >= 2).
  - Axis: J is regular if T is a smooth 1-form in (x, y) = rho (cos, sin)(theta),
    i.e. T_zeta is a regular scalar (Fourier mode m ~ rho^|m| P(rho^2), true for
    any Zernike series) and T_theta = rho^2 x (regular scalar) (mode m ~
    rho^(|m|+2)). h_0, h_|m| ~ rho^(|m|+2) and the factor rho^2 on a (page 4 of
    the notes) give exactly that; a and b are unrestricted.
  - Net toroidal current: int sqrt(g) J^zeta drho dtheta = 2pi <T_theta(1) -
    T_theta(0)>_theta = I through every cross-section zeta = const.

Gauge: in the radial gauge the only freedom left is T_zeta -> T_zeta + alpha(zeta)
(curl-free). It is not in the representation (the notes drop alpha and the
m = 0 modes of chi, which are of that form), so u -> J is one to one. The DESC
target potential, defined up to alpha, is fitted with alpha free (``fit``).

J = sum_i sqrt(g) J^i e_i / sqrt(g) in (R, phi, Z) components. At rho = 0,
sqrt(g) = 0 and N = sum_i sqrt(g) J^i e_i = 0, so J there is the limit
d_rho N / d_rho sqrt(g) along the ray, computed from second derivatives; for
rho > 0 every term of N is O(rho) on its own, so N / sqrt(g) has no cancellation.
"""

import functools

import jax
import jax.numpy as jnp
import numpy as np
from numpy.polynomial import Polynomial

from desc.basis import FourierZernikeBasis, fourier, zernike_radial
from desc.grid import LinearGrid

from current_potential import MU_0, _radial_integral_matrix
from desc_input import covariant_basis, current_from_sqrtg
from separable import Separable


def _binom(a, j):
    """Generalized binomial coefficient a (a - 1) ... (a - j + 1) / j!, any real a."""
    return np.prod([(a - i) / (i + 1) for i in range(j)])


def h_polynomial(q, s):
    """h_q(rho) = rho^q sum_{j<s} binom(-q/2, j) (rho^2 - 1)^j as a Polynomial in rho."""
    t = Polynomial([-1, 0, 1])
    return Polynomial.basis(q) * sum(_binom(-q / 2, j) * t**j for j in range(s))


class _Structure:
    """What is fixed: the bases, mode lists and radial polynomial tables."""

    def __init__(self, basis, s):
        self.basis, self.s, self.NFP = basis, s, basis.NFP
        K = basis.num_modes
        self.ab = Separable([basis, basis], [np.zeros(K)] * 2, basis.NFP)
        M, N = basis.M, basis.N
        self.chi_m = np.array([m for m in range(-M, M + 1) if m != 0])
        self.chi_n = np.arange(-N, N + 1)
        mm, nn = np.meshgrid(self.chi_m, self.chi_n, indexing="ij")
        self.chi_modes = np.column_stack([mm.ravel(), nn.ravel()])
        self.n_chi, self.K = len(self.chi_modes), K
        # h_|m| and its first two derivatives as rows of monomial coefficients
        polys = [h_polynomial(m + 2, s) for m in range(M + 1)]
        D = max(p.degree() for p in polys)
        self.h_coef = np.zeros((3, M + 1, D + 1))
        for i, p in enumerate(polys):
            for d in range(3):
                c = p.deriv(d).coef if d else p.coef
                self.h_coef[d, i, : len(c)] = c
        self.chi_abs = np.abs(self.chi_m)

    def split(self, u):
        """u -> (C_chi[m, n], C_ab) as JAX arrays."""
        chi, a, b = u[: self.n_chi], u[self.n_chi : self.n_chi + self.K], u[self.n_chi + self.K :]
        return chi.reshape(len(self.chi_m), len(self.chi_n)), self.ab.scatter([a, b])


def _h(S, rho, d):
    """h_|m|^(d)(rho), shape (n_points, M + 1); column 0 is h_0."""
    powers = rho[:, None] ** jnp.arange(S.h_coef.shape[2])
    return powers @ jnp.asarray(S.h_coef[d]).T


def _window(S, rho):
    """f, f', f'' and g = rho^2 f, g', g''."""
    s, u = S.s, 1 - rho**2
    f = u**s
    f1 = -2 * s * rho * u ** (s - 1)
    f2 = -2 * s * u ** (s - 1) + 4 * s * (s - 1) * rho**2 * u ** (s - 2)
    return f, f1, f2, rho**2 * f, 2 * rho * f + rho**2 * f1, 2 * f + 4 * rho * f1 + rho**2 * f2


def _chi(S, Cchi, rho, theta, zeta, dr=0, dt=0, dz=0):
    P = _h(S, rho, dr)[:, S.chi_abs] * fourier(theta[:, None], S.chi_m, dt=dt)
    F = fourier(zeta[:, None], S.chi_n, NFP=S.NFP, dt=dz)
    return jnp.einsum("im,mn,in->i", P, Cchi, F)


def _potential(S, u, I, rho, theta, zeta):
    """(T_theta, T_zeta) at the points, shape (n_points, 2)."""
    Cchi, Cab = S.split(u)
    f, _, _, g, _, _ = _window(S, rho)
    AB = S.ab.evaluate(rho, theta, zeta, C=Cab)
    T_t = I / (2 * np.pi) * _h(S, rho, 0)[:, 0] + _chi(S, Cchi, rho, theta, zeta, dt=1) + g * AB[:, 0]
    T_z = _chi(S, Cchi, rho, theta, zeta, dz=1) + f * AB[:, 1]
    return jnp.stack([T_t, T_z], axis=-1)


def _sqrtg_J(S, u, I, rho, theta, zeta, deriv=False):
    """sqrt(g) (J^rho, J^theta, J^zeta), or with ``deriv`` their rho-derivatives."""
    Cchi, Cab = S.split(u)
    f, f1, f2, g, g1, g2 = _window(S, rho)
    ev = functools.partial(S.ab.evaluate, rho, theta, zeta, C=Cab)
    ch = functools.partial(_chi, S, Cchi, rho, theta, zeta)
    AB, ABr, ABt, ABz = ev(), ev(dr=1), ev(dt=1), ev(dz=1)
    A, B, Ar, Br, Bt, Az = AB[:, 0], AB[:, 1], ABr[:, 0], ABr[:, 1], ABt[:, 1], ABz[:, 0]
    i2pi = I / (2 * np.pi)
    if not deriv:
        a = f * (Bt - rho**2 * Az)
        b = -(ch(dr=1, dz=1) + f1 * B + f * Br)
        c = i2pi * _h(S, rho, 1)[:, 0] + ch(dr=1, dt=1) + g1 * A + g * Ar
        return a, b, c
    ABrr, ABrt, ABrz = ev(dr=2), ev(dr=1, dt=1), ev(dr=1, dz=1)
    a = f1 * (Bt - rho**2 * Az) + f * (ABrt[:, 1] - 2 * rho * Az - rho**2 * ABrz[:, 0])
    b = -(ch(dr=2, dz=1) + f2 * B + 2 * f1 * Br + f * ABrr[:, 1])
    c = (i2pi * _h(S, rho, 2)[:, 0] + ch(dr=2, dt=1) + g2 * A + 2 * g1 * Ar
         + g * ABrr[:, 0])
    return a, b, c


@functools.partial(jax.jit, static_argnums=(0, 1))
def _current(S, RZ, u, I, rho, theta, zeta):
    """(J_R, J_phi, J_Z) at the points, shape (n_points, 3); linear in (u, I)."""
    a, b, c = _sqrtg_J(S, u, I, rho, theta, zeta)
    X, E, sqrt_g = covariant_basis(RZ, rho, theta, zeta)
    R = X[:, 0]
    axis = rho == 0
    sJ = jnp.stack([a, b, c], axis=-1)
    J = current_from_sqrtg(sJ, E, jnp.where(axis, 1.0, sqrt_g))

    def on_axis():  # J = d_rho N / d_rho sqrt(g) along the ray (0 / 0 at rho = 0)
        # N = a e_rho + b e_theta + c e_zeta, e_rho = (R_r, 0, Z_r), e_theta = (R_t, 0, Z_t),
        # e_zeta = (R_z, R, Z_z) in (R, phi, Z) components
        Xr, Xt, Xz = E[:, 0, ::2], E[:, 1, ::2], E[:, 2, ::2]
        ar, br, cr = _sqrtg_J(S, u, I, rho, theta, zeta, deriv=True)
        Xrr = RZ.evaluate(rho, theta, zeta, dr=2)
        Xrt = RZ.evaluate(rho, theta, zeta, dr=1, dt=1)
        Xrz = RZ.evaluate(rho, theta, zeta, dr=1, dz=1)
        Nr = jnp.stack([
            ar * Xr[:, 0] + a * Xrr[:, 0] + br * Xt[:, 0] + b * Xrt[:, 0] + cr * Xz[:, 0] + c * Xrz[:, 0],
            cr * R + c * Xr[:, 0],
            ar * Xr[:, 1] + a * Xrr[:, 1] + br * Xt[:, 1] + b * Xrt[:, 1] + cr * Xz[:, 1] + c * Xrz[:, 1],
        ], axis=-1)
        sqrt_g_r = (Xr[:, 0] * (Xt[:, 0] * Xr[:, 1] - Xr[:, 0] * Xt[:, 1])
                    + R * (Xrt[:, 0] * Xr[:, 1] + Xt[:, 0] * Xrr[:, 1]
                           - Xrr[:, 0] * Xt[:, 1] - Xr[:, 0] * Xrt[:, 1]))
        return Nr / jnp.where(axis, sqrt_g_r, 1.0)[:, None]

    J_axis = jax.lax.cond(jnp.any(axis), on_axis, lambda: jnp.zeros_like(J))
    return jnp.where(axis[:, None], J_axis, J)


def _sqrtg_J_columns(S, kind, l, m, n, rho, theta, zeta):
    """sqrt(g) (J^rho, J^theta, J^zeta) of single unit coefficients, (n_points, n_cols, 3).

    Column j is the coefficient (kind[j], l[j], m[j], n[j]), kind 0 = chi, 1 = a,
    2 = b (``BoundaryPotential.coefficient_modes``), with I = 0. Closed form from
    one radial factor and two Fourier factors per column, so the cost does not
    grow with the number of modes; equal to ``_sqrtg_J`` of the unit vector:
        a:   f (-rho^2 d_zeta A),  0,                     g' A + g d_rho A
        b:   f d_theta B,          -(f' B + f d_rho B),   0
        chi: 0,                    -d_rho d_zeta chi,     d_rho d_theta chi
    with A, B = Z_l^|m|(rho) F_m(theta) F_n(zeta) and chi = h_|m|(rho) F_m F_n.
    """
    f, f1, _, g, g1, _ = (v[:, None] for v in _window(S, rho))
    am = jnp.abs(m)
    chi = (kind == 0)[None]
    P = jnp.where(chi, _h(S, rho, 0)[:, am], zernike_radial(rho[:, None], l[None], am[None]))
    Pr = jnp.where(chi, _h(S, rho, 1)[:, am],
                   zernike_radial(rho[:, None], l[None], am[None], dr=1))
    Ft, Ftt = fourier(theta[:, None], m[None]), fourier(theta[:, None], m[None], dt=1)
    Fz = fourier(zeta[:, None], n[None], NFP=S.NFP)
    Fzz = fourier(zeta[:, None], n[None], NFP=S.NFP, dt=1)
    a_, b_ = (kind == 1)[None], (kind == 2)[None]
    r2 = rho[:, None] ** 2
    sJr = jnp.where(a_, -f * r2 * P * Ft * Fzz, 0) + jnp.where(b_, f * P * Ftt * Fz, 0)
    sJt = jnp.where(b_, -(f1 * P + f * Pr) * Ft * Fz, 0) + jnp.where(chi, -Pr * Ft * Fzz, 0)
    sJz = jnp.where(a_, (g1 * P + g * Pr) * Ft * Fz, 0) + jnp.where(chi, Pr * Ftt * Fz, 0)
    return jnp.stack([sJr, sJt, sJz], axis=-1)


_sqrtg_J_columns_jit = jax.jit(_sqrtg_J_columns, static_argnums=0)
_sqrtg_J_jit = jax.jit(_sqrtg_J, static_argnums=0)


class BoundaryPotential:
    """The boundary-adapted potential: J = curl T, J = 0 on the LCFS.

    Parameters
    ----------
    basis : FourierZernikeBasis
        Basis of a and b (sym=False); chi uses its M, N.
    u : ndarray, shape (n_free,)
        Free coefficients (chi_mn in ``chi_modes`` order, then a, then b, each in
        ``basis.modes`` order) in A. Any u is admissible.
    I : float
        Net toroidal current [A], fixed.
    eq : Equilibrium
        The equilibrium whose flux coordinates and geometry T refers to.
    edge_power : int
        s >= 2: J = O((1 - rho)^(s - 1)) at the LCFS.
    """

    def __init__(self, basis, u, I, eq, edge_power=2, _structure=None):
        self.S = _structure or _Structure(basis, edge_power)
        self.basis, self.eq, self.edge_power = basis, eq, edge_power
        self.u = np.asarray(u, dtype=float)
        self.I = float(I)
        assert self.u.shape == (self.n_free,)
        self._RZ = Separable([eq.R_basis, eq.Z_basis], [eq.R_lmn, eq.Z_lmn], eq.NFP)

    @property
    def n_free(self):
        return self.S.n_chi + 2 * self.S.K

    @property
    def chi_modes(self):
        """(m, n) of the chi coefficients, the first part of u."""
        return self.S.chi_modes

    def with_u(self, u):
        """The same representation with other free coefficients (no recompilation)."""
        return BoundaryPotential(self.basis, u, self.I, self.eq, self.edge_power, self.S)

    def coefficient_modes(self):
        """(kind, l, m, n) of every free coefficient, in u order; kind 0 = chi, 1 = a,
        2 = b (l = 0 for chi)."""
        S, md = self.S, self.basis.modes
        kind = np.repeat([0, 1, 2], [S.n_chi, S.K, S.K])
        l = np.concatenate([np.zeros(S.n_chi, int), md[:, 0], md[:, 0]])
        m = np.concatenate([S.chi_modes[:, 0], md[:, 1], md[:, 1]])
        n = np.concatenate([S.chi_modes[:, 1], md[:, 2], md[:, 2]])
        return kind, l.astype(int), m.astype(int), n.astype(int)

    def sqrtg_J_columns(self, cols, rho, theta, zeta):
        """sqrt(g) J^i of the unit coefficients ``cols`` (indices into u) at the points,
        (n_points, len(cols), 3), I = 0; see ``_sqrtg_J_columns``. Jitted."""
        kind, l, m, n = (jnp.asarray(v[np.asarray(cols)]) for v in self.coefficient_modes())
        return _sqrtg_J_columns_jit(self.S, kind, l, m, n, rho, theta, zeta)

    def sqrtg_J(self, u, rho, theta, zeta, I=None):
        """sqrt(g) (J^rho, J^theta, J^zeta) for coefficients u and net current I
        (default ``self.I``), shape (n_points, 3). Jitted, linear in (u, I)."""
        I = self.I if I is None else I
        return jnp.stack(_sqrtg_J_jit(self.S, jnp.asarray(u), I, rho, theta, zeta), axis=-1)

    def current_function(self):
        """Jitted f(u, rho, theta, zeta) -> (J_R, J_phi, J_Z), linear in u, for JAX
        differentiation with respect to the coefficients (I fixed)."""
        return lambda u, rho, theta, zeta: _current(self.S, self._RZ, u, self.I, rho, theta, zeta)

    def point_evaluator(self, eq=None):
        """Jitted f(rho, theta, zeta) -> J = curl T in (R, phi, Z) components."""
        assert eq is None or eq is self.eq, "the geometry is that of self.eq"
        return functools.partial(_current, self.S, self._RZ, jnp.asarray(self.u), self.I)

    def evaluate(self, grid, chunk=50000):
        """J at the nodes of ``grid``, (J_R, J_phi, J_Z), shape (num_nodes, 3)."""
        return self._chunked(self.point_evaluator(), grid, chunk)

    def T(self, grid, chunk=50000):
        """(T_theta, T_zeta) at the nodes of ``grid``, shape (num_nodes, 2)."""
        f = jax.jit(functools.partial(_potential, self.S))
        return self._chunked(lambda r, t, z: f(jnp.asarray(self.u), self.I, r, t, z), grid, chunk)

    @staticmethod
    def _chunked(f, grid, chunk):
        nodes = np.asarray(grid.nodes if hasattr(grid, "nodes") else grid)
        out = [np.asarray(f(*(jnp.asarray(nodes[i : i + chunk, j]) for j in range(3))))
               for i in range(0, len(nodes), chunk)]
        return np.concatenate(out)

    @classmethod
    def fit(cls, eq, L=None, M=None, N=None, factors=(3, 3, 2.5), n_rho=None,
            edge_power=2, I=None):
        """Fit the free coefficients to DESC's potential, T itself (not J).

        DESC's sqrt(g) J^zeta, sqrt(g) J^theta (Ampere's law from B_theta_r -
        B_rho_t, B_rho_z - B_zeta_r) are integrated from the axis on a tensor grid
        (n_rho Chebyshev-Gauss points, default 2 L; 4 M + 1 in theta, 4 N + 1 in
        zeta per period) into the radial-gauge target T_theta = int_0^rho
        sqrt(g) J^zeta, T_zeta = -int_0^rho sqrt(g) J^theta (zero on the axis).
        The coefficients minimize sum_p w_p |T(u) - T_target|^2 (w_p uniform in
        theta, zeta and rho drho in rho) with the gauge alpha(zeta) of T_zeta
        free (the target is only defined up to it); alpha is then dropped. I is
        DESC's net toroidal current I(rho = 1) unless given.

        The problem separates in the Fourier modes: each chi_mn couples the
        T_theta mode (-m, n) (via d_theta) with the T_zeta mode (m, -n) (via
        d_zeta), so it is solved as one small least-squares problem per chi_mn,
        plus one per m = 0 mode of T_theta (a) and of T_zeta (b and alpha).
        """
        L = int(round(factors[0] * eq.L)) if L is None else L
        M = int(round(factors[1] * eq.M)) if M is None else M
        N = int(round(factors[2] * eq.N)) if N is None else N
        basis = FourierZernikeBasis(L, M, N, NFP=eq.NFP, sym=False,
                                    spectral_indexing=eq.spectral_indexing)
        S = _Structure(basis, edge_power)
        if I is None:
            g1 = LinearGrid(rho=1.0, M=eq.M_grid, N=eq.N_grid, NFP=eq.NFP)
            I = float(g1.compress(eq.compute("current", grid=g1)["current"])[0])

        # target potential in the radial gauge, then its real Fourier
        # coefficients C[rho, m, n, component] (uniform theta, zeta)
        rho, theta, zeta, T = target_potential(eq, 2 * L if n_rho is None else n_rho,
                                               4 * M + 1, 4 * N + 1)
        ms, ns = np.arange(-M, M + 1), np.arange(-N, N + 1)
        Pt = np.linalg.pinv(np.asarray(fourier(theta[:, None], ms[None, :])))
        Pz = np.linalg.pinv(np.asarray(fourier(zeta[:, None], ns[None, :], NFP=eq.NFP)))
        C = np.einsum("aj,ijkc,bk->iabc", Pt, T, Pz)
        x = 2 * rho - 1
        q = np.sqrt(np.sqrt(1 - x**2) * rho)  # sqrt of the weights of int . rho drho
        u, alpha = _fit_blocks(S, I, rho, q, C, ms, ns)
        pot = cls(basis, u, I, eq, edge_power, S)
        pot.fit_info = {"alpha": alpha, "alpha_n": ns}  # the dropped gauge, sum alpha_n F_n(zeta)
        return pot

    def gauge(self, zeta):
        """alpha(zeta) of the fit: T_target ~ T(u) + (0, alpha) up to the misfit."""
        a = self.fit_info
        return np.asarray(fourier(np.asarray(zeta)[:, None], a["alpha_n"], NFP=self.eq.NFP)) @ a["alpha"]

    def save(self, path):
        b = self.basis
        np.savez_compressed(path, u=self.u, I=self.I, edge_power=self.edge_power,
                            modes=b.modes, chi_modes=self.chi_modes, L=b.L, M=b.M, N=b.N,
                            NFP=b.NFP, spectral_indexing=b.spectral_indexing)

    @classmethod
    def load(cls, path, eq):
        with np.load(path) as f:
            basis = FourierZernikeBasis(int(f["L"]), int(f["M"]), int(f["N"]), NFP=int(f["NFP"]),
                                        sym=False, spectral_indexing=str(f["spectral_indexing"]))
            assert np.array_equal(basis.modes, f["modes"]), "basis mode mismatch"
            pot = cls(basis, f["u"], float(f["I"]), eq, int(f["edge_power"]))
            assert np.array_equal(pot.chi_modes, f["chi_modes"]), "chi mode mismatch"
            return pot


def target_potential(eq, n_rho, n_theta, n_zeta):
    """DESC's potential in the radial gauge on a tensor grid.

    T_theta = int_0^rho sqrt(g) J^zeta, T_zeta = -int_0^rho sqrt(g) J^theta, with
    sqrt(g) J from Ampere's law (B_theta_r - B_rho_t, B_rho_z - B_zeta_r), at
    n_rho Chebyshev-Gauss points in rho and n_theta, n_zeta uniform points in
    theta and zeta (one field period). Returns rho, theta, zeta and T of shape
    (n_rho, n_theta, n_zeta, 2).
    """
    rho, Q, _ = _radial_integral_matrix(n_rho)
    grid = LinearGrid(rho=rho, theta=n_theta, zeta=n_zeta, NFP=eq.NFP, sym=False)
    d = eq.compute(["B_theta_r", "B_rho_t", "B_rho_z", "B_zeta_r"], grid=grid)
    sJ_zeta = (np.asarray(d["B_theta_r"]) - np.asarray(d["B_rho_t"])) / MU_0
    sJ_theta = (np.asarray(d["B_rho_z"]) - np.asarray(d["B_zeta_r"])) / MU_0
    idx = np.asarray(grid.meshgrid_reshape(np.arange(grid.num_nodes), "rtz"))
    T = np.stack([np.einsum("ij,jtz->itz", Q, sJ_zeta[idx]),
                  -np.einsum("ij,jtz->itz", Q, sJ_theta[idx])], axis=-1)
    nodes = np.asarray(grid.nodes)
    return rho, nodes[idx[0, :, 0], 1], nodes[idx[0, 0, :], 2], T


def _fit_blocks(S, I, rho, q, C, ms, ns):
    """Mode-by-mode least squares of T(u) against the target modes C (see ``fit``)."""
    b, NFP = S.basis, S.NFP
    f, _, _, g, _, _ = (np.asarray(v) for v in _window(S, jnp.asarray(rho)))
    H = np.asarray(_h(S, jnp.asarray(rho), 0))  # h_|m|(rho), column |m|
    index = {tuple(mode): i for i, mode in enumerate(b.modes)}
    am, an = {m: i for i, m in enumerate(ms)}, {n: i for i, n in enumerate(ns)}
    Z = {}  # radial columns per |m|: (l values, Z_l^|m|(rho))
    for m in np.unique(np.abs(b.modes[:, 1])):
        ls = np.unique(b.modes[np.abs(b.modes[:, 1]) == m, 0])
        Z[m] = (ls, np.asarray(zernike_radial(rho[:, None], ls, np.full(len(ls), m))))
    a_c, b_c = np.zeros(S.K), np.zeros(S.K)
    chi = np.zeros((len(S.chi_m), len(S.chi_n)))
    nr = len(rho)

    def solve(A, y):  # weighted least squares, rows in blocks of n_rho
        w = np.tile(q, A.shape[0] // nr)
        return np.linalg.lstsq(w[:, None] * A, w * y, rcond=None)[0]

    def put(coef, mode_m, mode_n, sol):
        ls = Z[abs(mode_m)][0]
        for l, v in zip(ls, sol):
            if (l, mode_m, mode_n) in index:
                coef[index[(l, mode_m, mode_n)]] = v

    # chi_mn with the T_theta mode (-m, n) [a] and the T_zeta mode (m, -n) [b]
    for i, m in enumerate(S.chi_m):
        ls, Zm = Z[abs(m)]
        for j, n in enumerate(S.chi_n):
            nl = len(ls)
            A = np.zeros((2 * nr, 1 + 2 * nl))
            A[:nr, 0] = -m * H[:, abs(m)]  # d_theta F_m = -m F_-m
            A[:nr, 1 : 1 + nl] = g[:, None] * Zm
            A[nr:, 0] = -n * NFP * H[:, abs(m)]  # d_zeta F_n = -n NFP F_-n
            A[nr:, 1 + nl :] = f[:, None] * Zm
            y = np.concatenate([C[:, am[-m], an[n], 0], C[:, am[m], an[-n], 1]])
            sol = solve(A, y)
            chi[i, j] = sol[0]
            put(a_c, -m, n, sol[1 : 1 + nl])
            put(b_c, m, -n, sol[1 + nl :])
    # m = 0: a alone (T_theta, minus the net-current term), b with the gauge alpha_n
    ls, Z0 = Z[0]
    alpha = np.zeros(len(ns))
    for n in ns:
        y = C[:, am[0], an[n], 0] - (I / (2 * np.pi) * H[:, 0] if n == 0 else 0)
        put(a_c, 0, n, solve(g[:, None] * Z0, y))
        sol = solve(np.column_stack([f[:, None] * Z0, np.ones(nr)]), C[:, am[0], an[n], 1])
        put(b_c, 0, n, sol[:-1])
        alpha[an[n]] = sol[-1]
    return np.concatenate([chi.ravel(), a_c, b_c]), alpha
