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
"""

import functools

import jax
import jax.numpy as jnp
import numpy as np

from desc.basis import FourierZernikeBasis, zernike_radial
from desc.grid import LinearGrid

from separable import Separable


@functools.partial(jax.jit, static_argnums=(0, 1))
def _p_and_grad(P, RZ, rho, theta, zeta):
    """(p, dp/dR, dp/(R dphi), dp/dZ) at the points, shape (n_points, 4)."""
    p = P.evaluate(rho, theta, zeta)[:, 0]
    pr = P.evaluate(rho, theta, zeta, dr=1)[:, 0]
    pt = P.evaluate(rho, theta, zeta, dt=1)[:, 0]
    pz = P.evaluate(rho, theta, zeta, dz=1)[:, 0]
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
        Coefficients of p in Pa.
    eq : Equilibrium
        The equilibrium whose flux coordinates and geometry p refers to.
    """

    def __init__(self, basis, c, eq):
        self.basis = basis
        self.c = np.asarray(c)
        self.eq = eq

    @classmethod
    def fit(cls, eq, L=None, M=None, N=None, factors=(3, 3, 2.5)):
        """DESC's p(rho) on a basis of resolution (L, M, N), default that of T.

        Only the m = n = 0 modes are fitted (least squares on 4 L + 8 points
        in rho); the others start at zero. Then p = 0 is imposed on rho = 1.
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
        c = np.zeros(basis.num_modes)
        c[k] = np.linalg.lstsq(Z, p, rcond=None)[0]
        return cls(basis, c, eq).impose_zero_edge()

    def impose_zero_edge(self):
        """Copy with p = 0 on rho = 1: sum_l c_lmn = 0 for each (m, n).

        The edge value of each (m, n) group, sum_l c_lmn, is subtracted from
        its lowest-l coefficient (l = |m|): the correction is that value times
        rho^|m| F_mn, smooth and regular at the axis. As a parametrization for
        the optimizer, c_|m|mn is a linear function of the other coefficients.
        """
        modes = self.basis.modes
        _, g = np.unique(modes[:, 1:], axis=0, return_inverse=True)
        g = g.ravel()
        edge = np.bincount(g, self.c)
        order = np.lexsort((modes[:, 0], g))  # by group, then by l
        first = order[np.r_[0, np.flatnonzero(np.diff(g[order])) + 1]]  # lowest l
        c = self.c.copy()
        c[first] -= edge[g[first]]
        return Pressure(self.basis, c, self.eq)

    def point_evaluator(self, eq=None):
        """Jitted f(rho, theta, zeta) -> (p, grad p in (R, phi, Z) components)."""
        eq = self.eq if eq is None else eq
        P = Separable([self.basis], [self.c], self.basis.NFP)
        RZ = Separable([eq.R_basis, eq.Z_basis], [eq.R_lmn, eq.Z_lmn], eq.NFP)
        return functools.partial(_p_and_grad, P, RZ)

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
                            NFP=b.NFP, spectral_indexing=b.spectral_indexing)

    @classmethod
    def load(cls, path, eq):
        with np.load(path) as f:
            basis = FourierZernikeBasis(
                int(f["L"]), int(f["M"]), int(f["N"]), NFP=int(f["NFP"]),
                sym=False, spectral_indexing=str(f["spectral_indexing"]),
            )
            assert np.array_equal(basis.modes, f["modes"]), "basis mode mismatch"
            return cls(basis, f["c"], eq)
