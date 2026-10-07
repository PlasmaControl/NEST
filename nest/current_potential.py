"""Current density as the curl of a potential, J = curl T.

T is written in the flux coordinates (rho, theta, zeta) of the starting DESC
equilibrium, in the gauge T_rho = 0:

    T = T_theta grad(theta) + T_zeta grad(zeta),

with T_theta and T_zeta Fourier-Zernike series. Then, in any coordinates,

    sqrt(g) J^rho   = d_theta T_zeta - d_zeta T_theta
    sqrt(g) J^theta = - d_rho T_zeta
    sqrt(g) J^zeta  =   d_rho T_theta

so div J = 0 for any coefficients. J is linear in the coefficients; the
geometry (e_rho, e_theta, e_zeta, sqrt(g)) is that of the fixed equilibrium.

The starting T comes from DESC's J by integrating along rho from the axis,

    T_theta = int_0^rho sqrt(g) J^zeta drho',   T_zeta = - int_0^rho sqrt(g) J^theta drho',

which reproduces all three components of J (the rho one through div J = 0)
and makes T_theta, T_zeta regular at the axis.

No current through the boundary, J.n = 0 at rho = 1, means d_theta T_zeta =
d_zeta T_theta there: the boundary values of (T_theta, T_zeta) must be a
surface gradient plus constants. ``impose_no_normal_current`` projects them
onto that form; the constants, kept, are the net currents
(int_0^2pi T_theta(rho=1) dtheta = toroidal current I).
"""

import functools

import jax
import jax.numpy as jnp
import numpy as np
from numpy.polynomial import chebyshev as cheb

from desc.basis import FourierZernikeBasis, zernike_radial
from desc.grid import LinearGrid
from desc.transform import Transform

from separable import Separable

MU_0 = 4e-7 * np.pi
COMPONENTS = ("T_theta", "T_zeta")


def _radial_integral_matrix(n):
    """Chebyshev-Gauss nodes rho in (0, 1) and Q with (Q f)_i = int_0^rho_i f drho."""
    x = np.cos(np.pi * (np.arange(n) + 0.5) / n)[::-1]
    V = cheb.chebvander(x, n - 1)
    # coefficients -> coefficients of the integral from x = -1 (rho = 0), drho = dx / 2
    I = np.stack([cheb.chebint(np.eye(n)[k], lbnd=-1, scl=0.5) for k in range(n)], axis=1)
    Q = cheb.chebvander(x, n) @ I @ np.linalg.inv(V)
    return (x + 1) / 2, Q


@functools.partial(jax.jit, static_argnums=(0, 1))
def _J_from_T(T, RZ, rho, theta, zeta):
    """(J_R, J_phi, J_Z) = curl T at the points, shape (n_points, 3)."""
    Tr = T.evaluate(rho, theta, zeta, dr=1)  # columns: T_theta, T_zeta
    Tt = T.evaluate(rho, theta, zeta, dt=1)
    Tz = T.evaluate(rho, theta, zeta, dz=1)
    a = Tt[:, 1] - Tz[:, 0]  # sqrt(g) J^rho
    b = -Tr[:, 1]  # sqrt(g) J^theta
    c = Tr[:, 0]  # sqrt(g) J^zeta
    X = RZ.evaluate(rho, theta, zeta)  # columns: R, Z
    Xr = RZ.evaluate(rho, theta, zeta, dr=1)
    Xt = RZ.evaluate(rho, theta, zeta, dt=1)
    Xz = RZ.evaluate(rho, theta, zeta, dz=1)
    R = X[:, 0]
    # e_rho = (R_r, 0, Z_r), e_theta = (R_t, 0, Z_t), e_zeta = (R_z, R, Z_z) in
    # (R, phi, Z) components (phi = zeta); sqrt(g) = e_rho . e_theta x e_zeta
    sqrt_g = R * (Xt[:, 0] * Xr[:, 1] - Xr[:, 0] * Xt[:, 1])
    J_R = a * Xr[:, 0] + b * Xt[:, 0] + c * Xz[:, 0]
    J_Z = a * Xr[:, 1] + b * Xt[:, 1] + c * Xz[:, 1]
    return jnp.stack([J_R, c * R, J_Z], axis=-1) / sqrt_g[:, None]


class CurrentPotential:
    """T_theta, T_zeta as Fourier-Zernike series sharing one basis, J = curl T.

    Parameters
    ----------
    basis : FourierZernikeBasis
    c : ndarray, shape (basis.num_modes, 2)
        Coefficients of (T_theta, T_zeta) in A (A/m^2 times m^2 per radian).
    eq : Equilibrium
        The equilibrium whose flux coordinates and geometry T refers to.
    """

    def __init__(self, basis, c, eq):
        self.basis = basis
        self.c = np.asarray(c)
        self.eq = eq

    @classmethod
    def fit(cls, eq, L=None, M=None, N=None, factors=(3, 3, 2.5), n_rho=None,
            no_normal_current=True):
        """Fit T to DESC's J of ``eq`` (radial integral, then least squares).

        T is sampled on a tensor grid with n_rho Chebyshev-Gauss points in rho
        (default 2 L), 4 M + 1 in theta and 4 N + 1 in zeta per field period.
        With ``no_normal_current``, J.n = 0 is then imposed on the boundary.
        """
        L = int(round(factors[0] * eq.L)) if L is None else L
        M = int(round(factors[1] * eq.M)) if M is None else M
        N = int(round(factors[2] * eq.N)) if N is None else N
        basis = FourierZernikeBasis(
            L, M, N, NFP=eq.NFP, sym=False, spectral_indexing=eq.spectral_indexing
        )
        rho, Q = _radial_integral_matrix(2 * L if n_rho is None else n_rho)
        grid = LinearGrid(rho=rho, theta=4 * M + 1, zeta=4 * N + 1, NFP=eq.NFP, sym=False)
        d = eq.compute(["B_theta_r", "B_rho_t", "B_rho_z", "B_zeta_r"], grid=grid)
        sJ_zeta = (np.asarray(d["B_theta_r"]) - np.asarray(d["B_rho_t"])) / MU_0
        sJ_theta = (np.asarray(d["B_rho_z"]) - np.asarray(d["B_zeta_r"])) / MU_0

        # integrate along rho on the (rho, theta, zeta) layout, back to node order
        idx = np.asarray(grid.meshgrid_reshape(np.arange(grid.num_nodes), "rtz"))
        T = np.empty((grid.num_nodes, 2))
        for k, (f, sign) in enumerate([(sJ_zeta, 1), (sJ_theta, -1)]):
            T[idx.ravel(), k] = sign * np.einsum("ij,jtz->itz", Q, f[idx]).ravel()

        tr = Transform(grid, basis, build=False, build_pinv=True)
        c = np.stack([np.asarray(tr.fit(T[:, k])) for k in range(2)], axis=-1)
        pot = cls(basis, c, eq)
        return pot.impose_no_normal_current() if no_normal_current else pot

    def impose_no_normal_current(self, power=None):
        """Copy of the potential with J.n = 0 on rho = 1, net currents unchanged.

        On the boundary, each Fourier mode (m, n NFP) of (T_theta, T_zeta) is
        replaced by its component along the wavevector (m, n NFP), which makes
        d_theta T_zeta = d_zeta T_theta; the (0, 0) mode (net currents) is kept.
        The correction d(theta, zeta) is extended inside as d rho^p, regular at
        the axis: p >= |m| + 2 for T_theta (which vanishes on the axis like
        rho^(|m|+2) in the gauge T_rho = 0) and p >= |m| for T_zeta. With
        ``power=None`` p is that minimum (the smoothest extension, spread over
        the cross-section); a larger ``power`` confines the change to a layer
        of width ~ 1/power at the edge. p is capped by the basis resolution.
        """
        b = self.basis
        grid = LinearGrid(rho=1.0, theta=2 * b.M + 1, zeta=2 * b.N + 1, NFP=b.NFP,
                          sym=False)
        idx = np.asarray(grid.meshgrid_reshape(np.arange(grid.num_nodes), "rtz"))[0]
        T = self.T(grid)[idx]  # (n_theta, n_zeta, 2) on the boundary
        Th = np.fft.fft2(T, axes=(0, 1))
        m = np.fft.fftfreq(idx.shape[0], 1 / idx.shape[0])[:, None]
        k = b.NFP * np.fft.fftfreq(idx.shape[1], 1 / idx.shape[1])[None, :]
        k2 = m**2 + k**2
        k2[0, 0] = 1
        proj = (m * Th[..., 0] + k * Th[..., 1]) / k2
        Th_new = np.stack([m * proj, k * proj], axis=-1)
        Th_new[0, 0] = Th[0, 0]  # (0, 0): the net currents, unchanged
        delta = np.fft.ifft2(Th_new - Th, axes=(0, 1)).real
        delta_nodes = np.empty((grid.num_nodes, 2))
        delta_nodes[idx.ravel()] = delta.reshape(-1, 2)

        # boundary correction in the real Fourier basis T_m(theta) F_n(zeta):
        # the Zernike modes with l = |m| equal it on rho = 1 (Z_l^m(1) = 1)
        mn = b.modes[b.modes[:, 0] == np.abs(b.modes[:, 1])]
        A = np.asarray(b.evaluate(grid.nodes, modes=mn))
        d = np.linalg.lstsq(A, delta_nodes, rcond=None)[0]  # (n_mn, 2)

        # extend inside: d rho^p = d sum_l a_l Z_l^|m|(rho), l = |m|, |m|+2, .., p
        index = {tuple(mode): i for i, mode in enumerate(b.modes)}
        r = np.linspace(0, 1, 4 * b.L + 8)
        c = self.c.copy()
        for comp, p_min in ((0, 2), (1, 0)):
            for (_, mm, nn), dk in zip(mn, d[:, comp]):
                am = abs(mm)
                p = am + p_min if power is None else max(am + p_min, power)
                p -= (p - am) % 2  # same parity as m
                ls = [l for l in range(am, p + 1, 2) if (l, mm, nn) in index]
                Z = np.asarray(zernike_radial(r[:, None], np.array(ls), np.full(len(ls), am)))
                a_l = np.linalg.lstsq(Z, r ** max(ls[-1], am), rcond=None)[0]
                for l, al in zip(ls, a_l):
                    c[index[(l, mm, nn)], comp] += dk * al
        return CurrentPotential(b, c, self.eq)

    def T(self, grid):
        """(T_theta, T_zeta) at the nodes of ``grid``, shape (num_nodes, 2)."""
        tr = Transform(grid, self.basis, build=True)
        return np.stack([np.asarray(tr.transform(self.c[:, k])) for k in range(2)], axis=-1)

    def point_evaluator(self, eq=None):
        """Jitted f(rho, theta, zeta) -> J = curl T in (R, phi, Z) components."""
        eq = self.eq if eq is None else eq
        T = Separable([self.basis] * 2, [self.c[:, k] for k in range(2)], self.basis.NFP)
        RZ = Separable([eq.R_basis, eq.Z_basis], [eq.R_lmn, eq.Z_lmn], eq.NFP)
        return functools.partial(_J_from_T, T, RZ)

    def evaluate(self, grid, chunk=50000):
        """J = curl T at the nodes of ``grid``, (J_R, J_phi, J_Z), shape (num_nodes, 3)."""
        f = self.point_evaluator()
        nodes = np.asarray(grid.nodes)
        out = [np.asarray(f(*(jnp.asarray(nodes[i : i + chunk, j]) for j in range(3))))
               for i in range(0, len(nodes), chunk)]
        return np.concatenate(out)

    def save(self, path):
        b = self.basis
        np.savez_compressed(path, c=self.c, modes=b.modes, L=b.L, M=b.M, N=b.N,
                            NFP=b.NFP, spectral_indexing=b.spectral_indexing,
                            components=np.array(COMPONENTS))

    @classmethod
    def load(cls, path, eq):
        with np.load(path) as f:
            basis = FourierZernikeBasis(
                int(f["L"]), int(f["M"]), int(f["N"]), NFP=int(f["NFP"]),
                sym=False, spectral_indexing=str(f["spectral_indexing"]),
            )
            assert np.array_equal(basis.modes, f["modes"]), "basis mode mismatch"
            return cls(basis, f["c"], eq)
