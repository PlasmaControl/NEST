"""Pressure as its own Fourier-Zernike series p(rho, theta, zeta).

p is written in the flux coordinates of the starting DESC equilibrium, like
the current potential T, and is optimized together with T: once the surfaces
break, p is no longer a flux function. Its gradient uses the geometry of the
fixed equilibrium,

    grad p = p_rho grad(rho) + p_theta grad(theta) + p_zeta grad(zeta),
    grad(rho) = e_theta x e_zeta / sqrt(g),  grad(theta) = e_zeta x e_rho / sqrt(g),
    grad(zeta) = e_rho x e_theta / sqrt(g).

The starting p is DESC's p(rho): only the m = n = 0 modes, an exact fit when
p(rho) is a polynomial in rho^2 of degree <= L.

With no surface current B is continuous across the plasma boundary, so
pressure balance there needs p = 0 on rho = 1. Since every Zernike radial
polynomial is 1 at rho = 1, p(1, theta, zeta) = sum_mn (sum_l c_lmn) F_mn, and
p = 0 on the boundary means sum_l c_lmn = 0 for each (m, n);
``impose_zero_edge`` sets the lowest-l coefficient of each (m, n) from the
others, the smoothest correction (rho^|m| F_mn).

Smooth edge (``edge_order`` k, e.g. 2, as for J): p = w(rho) Q(rho, theta,
zeta) with w = (1 - rho^2)^(k+1) and Q the Fourier-Zernike series (the
coefficients ``c``). Then p = O((1 - rho^2)^(k+1)) and grad p =
O((1 - rho^2)^k), like J x B, for any coefficients; p = 0 on the edge holds by
construction.
"""

import functools

import jax
import jax.numpy as jnp
import numpy as np

from desc.basis import FourierZernikeBasis, fourier, zernike_radial
from desc.grid import LinearGrid

from desc_input import covariant_basis, dual_basis
from separable import Separable


def window(rho, k, dr=0):
    """w = (1 - rho^2)^(k+1) (dr = 0) or dw/drho (dr = 1)."""
    if dr == 0:
        return (1 - rho**2) ** (k + 1)
    return -2 * (k + 1) * rho * (1 - rho**2) ** k


@functools.partial(jax.jit, static_argnums=(0, 1, 2))
def _p_and_grad(P, RZ, k, rho, theta, zeta, C=None):
    """(p, dp/dR, dp/(R dphi), dp/dZ) at the points, shape (n_points, 4).

    ``C`` (``Separable.scatter``) replaces the stored coefficients; linear in C.
    """
    p = P.evaluate(rho, theta, zeta, C=C)[:, 0]
    pr = P.evaluate(rho, theta, zeta, dr=1, C=C)[:, 0]
    pt = P.evaluate(rho, theta, zeta, dt=1, C=C)[:, 0]
    pz = P.evaluate(rho, theta, zeta, dz=1, C=C)[:, 0]
    if k is not None:  # p = w Q
        w = window(rho, k)
        p, pr, pt, pz = w * p, window(rho, k, dr=1) * p + w * pr, w * pt, w * pz
    _, E, sqrt_g = covariant_basis(RZ, rho, theta, zeta)
    grad = jnp.einsum("ni,nia->na", jnp.stack([pr, pt, pz], axis=-1), dual_basis(E, sqrt_g))
    return jnp.concatenate([p[:, None], grad], axis=-1)


def _p_columns(k, l, m, n, l0, NFP, rho, theta, zeta):
    """(p, d_rho p, d_theta p, d_zeta p) of single free coefficients, (n_points, n_cols, 4).

    Column j is the function w(rho) [Z_l^|m| - Z_l0^|m|](rho) F_m(theta) F_n(zeta):
    with ``edge_order`` k, w = (1 - rho^2)^(k+1) and no l0 term (l0 < 0); without it,
    w = 1 and l0 the lowest l of the (m, n) group, whose coefficient is fixed by
    p = 0 on rho = 1 (``impose_zero_edge``). Closed form, cost independent of the
    number of modes.
    """
    am = jnp.abs(m)
    R = zernike_radial(rho[:, None], l[None], am[None])
    Rr = zernike_radial(rho[:, None], l[None], am[None], dr=1)
    sub = (l0 >= 0)[None]
    l0c = jnp.maximum(l0, 0)
    R = R - jnp.where(sub, zernike_radial(rho[:, None], l0c[None], am[None]), 0)
    Rr = Rr - jnp.where(sub, zernike_radial(rho[:, None], l0c[None], am[None], dr=1), 0)
    if k is not None:
        w, w1 = window(rho, k)[:, None], window(rho, k, dr=1)[:, None]
        R, Rr = w * R, w1 * R + w * Rr
    Ft, Ftt = fourier(theta[:, None], m[None]), fourier(theta[:, None], m[None], dt=1)
    Fz = fourier(zeta[:, None], n[None], NFP=NFP)
    Fzz = fourier(zeta[:, None], n[None], NFP=NFP, dt=1)
    return jnp.stack([R * Ft * Fz, Rr * Ft * Fz, R * Ftt * Fz, R * Ft * Fzz], axis=-1)


_p_columns_jit = jax.jit(_p_columns, static_argnums=(0, 5))


class Pressure:
    """p as a Fourier-Zernike series in the equilibrium's flux coordinates.

    Parameters
    ----------
    basis : FourierZernikeBasis
    c : ndarray, shape (basis.num_modes,)
        Coefficients of p in Pa, or of Q with ``edge_order``.
    eq : Equilibrium
        The equilibrium whose flux coordinates and geometry p refers to.
    edge_order : int or None
        k of the smooth edge, p = (1 - rho^2)^(k+1) Q; None for p itself a
        Fourier-Zernike series.
    """

    def __init__(self, basis, c, eq, edge_order=None):
        self.basis = basis
        self.c = np.asarray(c)
        self.eq = eq
        self.edge_order = edge_order

    @classmethod
    def fit(cls, eq, L=None, M=None, N=None, factors=(3, 3, 2.5), edge_order=None):
        """DESC's p(rho) on a basis of resolution (L, M, N), default that of T.

        Only the m = n = 0 modes are fitted (least squares on 4 L + 8 points
        in rho); the others start at zero. Then p = 0 is imposed on rho = 1.
        With ``edge_order`` the fit is of p by w Q (exact when p(rho) is w
        times a polynomial in rho^2).
        """
        L = int(round(factors[0] * eq.L)) if L is None else L
        M = int(round(factors[1] * eq.M)) if M is None else M
        N = int(round(factors[2] * eq.N)) if N is None else N
        basis = FourierZernikeBasis(
            L, M, N, NFP=eq.NFP, sym=False, spectral_indexing=eq.spectral_indexing
        )
        rho = np.linspace(0, 1, 4 * L + 8)
        p = np.asarray(eq.compute("p", grid=LinearGrid(rho=rho, NFP=eq.NFP))["p"])
        k = np.flatnonzero((basis.modes[:, 1] == 0) & (basis.modes[:, 2] == 0))
        l = basis.modes[k, 0]
        Z = np.asarray(zernike_radial(rho[:, None], l, np.zeros_like(l)))
        if edge_order is not None:
            Z = window(rho, edge_order)[:, None] * Z
        c = np.zeros(basis.num_modes)
        c[k] = np.linalg.lstsq(Z, p, rcond=None)[0]
        return cls(basis, c, eq, edge_order).impose_zero_edge()

    def free_coefficients(self):
        """Indices of the free coefficients of p, and for each the dependent lowest-l
        coefficient of its (m, n) group (-1 if none).

        With ``edge_order`` every coefficient is free. Without it, p = 0 on rho = 1
        fixes the lowest-l coefficient of each (m, n) group (``impose_zero_edge``):
        those are not free, and each free coefficient c_lmn carries -c_lmn into it.
        """
        modes = self.basis.modes
        if self.edge_order is not None:
            return np.arange(len(modes)), np.full(len(modes), -1)
        _, g = np.unique(modes[:, 1:], axis=0, return_inverse=True)
        g = g.ravel()
        order = np.lexsort((modes[:, 0], g))
        first = order[np.r_[0, np.flatnonzero(np.diff(g[order])) + 1]]
        lowest = {gi: fi for gi, fi in zip(g[first], first)}
        free = np.array([i for i in range(len(modes)) if lowest[g[i]] != i], dtype=int)
        return free, np.array([lowest[g[i]] for i in free], dtype=int)

    def to_coefficients(self, x):
        """All coefficients c from the free ones x (``free_coefficients`` order)."""
        free, dep = self.free_coefficients()
        c = jnp.zeros(self.basis.num_modes).at[free].set(x)
        return c.at[dep[dep >= 0]].add(-jnp.asarray(x)[dep >= 0])

    def free_values(self):
        """The free coefficients of ``self.c`` (which must satisfy the edge constraint)."""
        return np.asarray(self.c)[self.free_coefficients()[0]]

    def p_columns(self, cols, rho, theta, zeta):
        """(p, d_rho p, d_theta p, d_zeta p) of the free coefficients ``cols`` (indices
        into ``free_coefficients()``), (n_points, len(cols), 4); see ``_p_columns``."""
        free, dep = self.free_coefficients()
        cols = np.asarray(cols)
        md = self.basis.modes
        l, m, n = (jnp.asarray(md[free[cols], i]) for i in range(3))
        l0 = np.where(dep[cols] >= 0, md[np.maximum(dep[cols], 0), 0], -1)
        return _p_columns_jit(self.edge_order, l, m, n, jnp.asarray(l0), self.basis.NFP,
                              rho, theta, zeta)

    def impose_zero_edge(self):
        """Copy with p = 0 on rho = 1: sum_l c_lmn = 0 for each (m, n).

        The edge value of each (m, n) group, sum_l c_lmn, is subtracted from
        its lowest-l coefficient (l = |m|): the correction is that value times
        rho^|m| F_mn, smooth and regular at the axis. As a parametrization for
        the optimizer, c_|m|mn is a linear function of the other coefficients.
        With ``edge_order`` p = 0 on the edge already: returned unchanged.
        """
        if self.edge_order is not None:
            return self
        modes = self.basis.modes
        _, g = np.unique(modes[:, 1:], axis=0, return_inverse=True)
        g = g.ravel()
        edge = np.bincount(g, self.c)
        order = np.lexsort((modes[:, 0], g))  # by group, then by l
        first = order[np.r_[0, np.flatnonzero(np.diff(g[order])) + 1]]  # lowest l
        c = self.c.copy()
        c[first] -= edge[g[first]]
        return Pressure(self.basis, c, self.eq)

    @functools.cached_property
    def _separable(self):
        """(P, RZ): the series of p and of the geometry, built once (jit static args)."""
        eq = self.eq
        return (Separable([self.basis], [self.c], self.basis.NFP),
                Separable([eq.R_basis, eq.Z_basis], [eq.R_lmn, eq.Z_lmn], eq.NFP))

    def point_evaluator(self, eq=None):
        """Jitted f(rho, theta, zeta) -> (p, grad p in (R, phi, Z) components)."""
        P, RZ = self._separable
        if eq is not None and eq is not self.eq:  # another geometry
            RZ = Separable([eq.R_basis, eq.Z_basis], [eq.R_lmn, eq.Z_lmn], eq.NFP)
        return functools.partial(_p_and_grad, P, RZ, self.edge_order)

    def pressure_function(self):
        """Jitted f(c, rho, theta, zeta) -> (p, grad p), linear in the coefficients c
        (shape (basis.num_modes,)), for JAX differentiation; c replaces ``self.c``.

        Without ``edge_order`` p = 0 on the edge only if c satisfies the constraint of
        ``impose_zero_edge``; with it, for every c."""
        P, RZ = self._separable
        k = self.edge_order
        return lambda c, rho, theta, zeta: _p_and_grad(P, RZ, k, rho, theta, zeta, P.scatter([c]))

    def evaluate(self, grid, chunk=50000):
        """(p, grad p) at the nodes of ``grid``, shape (num_nodes, 4)."""
        f = self.point_evaluator()
        nodes = np.asarray(grid.nodes)
        out = [np.asarray(f(*(jnp.asarray(nodes[i : i + chunk, j]) for j in range(3))))
               for i in range(0, len(nodes), chunk)]
        return np.concatenate(out)

    def save(self, path):
        b = self.basis
        np.savez_compressed(path, c=self.c, modes=b.modes, L=b.L, M=b.M, N=b.N,
                            NFP=b.NFP, spectral_indexing=b.spectral_indexing,
                            edge_order=-1 if self.edge_order is None else self.edge_order)

    @classmethod
    def load(cls, path, eq):
        with np.load(path) as f:
            basis = FourierZernikeBasis(
                int(f["L"]), int(f["M"]), int(f["N"]), NFP=int(f["NFP"]),
                sym=False, spectral_indexing=str(f["spectral_indexing"]),
            )
            assert np.array_equal(basis.modes, f["modes"]), "basis mode mismatch"
            k = int(f["edge_order"]) if "edge_order" in f else -1
            return cls(basis, f["c"], eq, None if k < 0 else k)
