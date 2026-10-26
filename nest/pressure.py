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

from desc.basis import FourierZernikeBasis, zernike_radial
from desc.grid import LinearGrid

from separable import Separable


def window(rho, k, dr=0):
    """w = (1 - rho^2)^(k+1) (dr = 0) or dw/drho (dr = 1)."""
    if dr == 0:
        return (1 - rho**2) ** (k + 1)
    return -2 * (k + 1) * rho * (1 - rho**2) ** k


@functools.partial(jax.jit, static_argnums=(0, 1, 2))
def _p_and_grad(P, RZ, k, rho, theta, zeta, C=None):
    """(p, dp/dR, dp/(R dphi), dp/dZ) at the points, shape (n_points, 4).

    ``C`` (``P.scatter``) replaces the stored coefficients, e.g. traced ones.
    """
    p = P.evaluate(rho, theta, zeta, C=C)[:, 0]
    pr = P.evaluate(rho, theta, zeta, dr=1, C=C)[:, 0]
    pt = P.evaluate(rho, theta, zeta, dt=1, C=C)[:, 0]
    pz = P.evaluate(rho, theta, zeta, dz=1, C=C)[:, 0]
    if k is not None:  # p = w Q
        w = window(rho, k)
        p, pr, pt, pz = w * p, window(rho, k, dr=1) * p + w * pr, w * pt, w * pz
    R = RZ.evaluate(rho, theta, zeta)[:, 0]  # columns: R, Z
    Rr, Zr = RZ.evaluate(rho, theta, zeta, dr=1).T
    Rt, Zt = RZ.evaluate(rho, theta, zeta, dt=1).T
    Rz, Zz = RZ.evaluate(rho, theta, zeta, dz=1).T
    # e_rho = (R_r, 0, Z_r), e_theta = (R_t, 0, Z_t), e_zeta = (R_z, R, Z_z) in
    # (R, phi, Z) components (phi = zeta); sqrt(g) = e_rho . e_theta x e_zeta
    sqrt_g = R * (Rt * Zr - Rr * Zt)
    g_R = -pr * R * Zt + pt * R * Zr
    g_phi = pr * (Zt * Rz - Rt * Zz) + pt * (Zz * Rr - Rz * Zr) + pz * (Zr * Rt - Rr * Zt)
    g_Z = pr * R * Rt - pt * R * Rr
    return jnp.stack([p, g_R / sqrt_g, g_phi / sqrt_g, g_Z / sqrt_g], axis=-1)


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
    def fit(cls, eq, L=None, M=None, N=None, factors=(3, 3, 2.5), edge_order=None, sym=None):
        """DESC's p(rho) on a basis of resolution (L, M, N), default that of T.

        Only the m = n = 0 modes are fitted (least squares on 4 L + 8 points
        in rho); the others start at zero. Then p = 0 is imposed on rho = 1.
        With ``edge_order`` the fit is of p by w Q (exact when p(rho) is w
        times a polynomial in rho^2). ``sym`` (default: the equilibrium's):
        stellarator symmetry, a cos-type basis.
        """
        L = int(round(factors[0] * eq.L)) if L is None else L
        M = int(round(factors[1] * eq.M)) if M is None else M
        N = int(round(factors[2] * eq.N)) if N is None else N
        sym = bool(eq.sym) if sym is None else sym
        basis = FourierZernikeBasis(L, M, N, NFP=eq.NFP, sym="cos" if sym else False,
                                    spectral_indexing=eq.spectral_indexing)
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

    # the interface of Fields (shared with ``label_pressure.LabelPressure``)
    @property
    def n(self):
        return self.basis.num_modes

    def with_c(self, c, eq=None):
        return Pressure(self.basis, c, self.eq if eq is None else eq, self.edge_order)

    def labels(self):
        """(kind, l, m, n) per unknown (``precondition.mode_labels``): kind 1."""
        p = np.asarray(self.basis.modes, int)
        return np.column_stack([np.ones(len(p), int), p])

    def tensor(self, grid, geo, RZ, factored=False):
        from flux_interp import TensorPressure
        P = Separable([self.basis], [self.c], self.basis.NFP)
        return TensorPressure(P, self.edge_order, RZ, grid, geo, factored=factored)

    def point_evaluator(self, eq=None):
        """Jitted f(rho, theta, zeta) -> (p, grad p in (R, phi, Z) components)."""
        eq = self.eq if eq is None else eq
        P = Separable([self.basis], [self.c], self.basis.NFP)
        RZ = Separable([eq.R_basis, eq.Z_basis], [eq.R_lmn, eq.Z_lmn], eq.NFP)
        return functools.partial(_p_and_grad, P, RZ, self.edge_order)

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
                            edge_order=-1 if self.edge_order is None else self.edge_order,
                            sym=str(b.sym) if b.sym else "")

    @classmethod
    def load(cls, path, eq):
        with np.load(path) as f:
            sym = str(f["sym"]) if "sym" in f and str(f["sym"]) else False
            basis = FourierZernikeBasis(
                int(f["L"]), int(f["M"]), int(f["N"]), NFP=int(f["NFP"]),
                sym=sym, spectral_indexing=str(f["spectral_indexing"]),
            )
            assert np.array_equal(basis.modes, f["modes"]), "basis mode mismatch"
            k = int(f["edge_order"]) if "edge_order" in f else -1
            return cls(basis, f["c"], eq, None if k < 0 else k)
