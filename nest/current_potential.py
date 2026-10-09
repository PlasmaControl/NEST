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

Smooth edge (``edge_order`` k, e.g. 2): J and its first k - 1 radial
derivatives vanish at rho = 1. T stays a Fourier-Zernike series, with the
linear constraints, for each (m, n) and component,

    d_rho^j T(rho = 1) = 0,  j = 1..k,

which make sqrt(g) J^theta and sqrt(g) J^zeta (and their derivatives) vanish
there. The boundary values of T stay free: they carry the net currents
(int_0^2pi T_theta(rho=1) dtheta = toroidal current I) and set
sqrt(g) J^rho = d_theta T_zeta - d_zeta T_theta on the boundary, which is not
constrained (no J.n = 0 condition for now).
"""

import functools

import jax
import jax.numpy as jnp
import numpy as np
from numpy.polynomial import chebyshev as cheb
from scipy.linalg import null_space

from desc.basis import FourierZernikeBasis, fourier, zernike_radial
from desc.grid import LinearGrid
from desc.transform import Transform

from separable import Separable

MU_0 = 4e-7 * np.pi
COMPONENTS = ("T_theta", "T_zeta")


def _radial_integral_matrix(n):
    """Chebyshev-Gauss nodes rho in (0, 1), Q with (Q f)_i = int_0^rho_i f drho,
    and the row q1 with q1 f = int_0^1 f drho."""
    x = np.cos(np.pi * (np.arange(n) + 0.5) / n)[::-1]
    V = cheb.chebvander(x, n - 1)
    # coefficients -> coefficients of the integral from x = -1 (rho = 0), drho = dx / 2
    I = np.stack([cheb.chebint(np.eye(n)[k], lbnd=-1, scl=0.5) for k in range(n)], axis=1)
    Vi = np.linalg.inv(V)
    return (x + 1) / 2, cheb.chebvander(x, n) @ I @ Vi, cheb.chebvander(1.0, n)[0] @ I @ Vi


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
    edge_order : int or None
        k of the smooth edge (sqrt(g) J^theta, sqrt(g) J^zeta = O((1 - rho)^k)
        at rho = 1), kept by ``impose_smooth_edge``; None for none.
    """

    def __init__(self, basis, c, eq, edge_order=None):
        self.basis = basis
        self.c = np.asarray(c)
        self.eq = eq
        self.edge_order = edge_order

    @classmethod
    def fit(cls, eq, L=None, M=None, N=None, factors=(3, 3, 2.5), n_rho=None,
            edge_order=None):
        """Fit T to DESC's J of ``eq`` (radial integral, then least squares).

        T is sampled on a tensor grid with n_rho Chebyshev-Gauss points in rho
        (default 2 L), 4 M + 1 in theta and 4 N + 1 in zeta per field period.
        With ``edge_order`` k, the fit is instead a least-squares fit of
        d_rho T = (sqrt(g) J^zeta, -sqrt(g) J^theta), i.e. of J, under the
        constraints d_rho^j T(1) = 0 (j = 1..k), T = 0 on the axis and T(1)
        equal to DESC's (so the net currents are DESC's).
        """
        L = int(round(factors[0] * eq.L)) if L is None else L
        M = int(round(factors[1] * eq.M)) if M is None else M
        N = int(round(factors[2] * eq.N)) if N is None else N
        basis = FourierZernikeBasis(
            L, M, N, NFP=eq.NFP, sym=False, spectral_indexing=eq.spectral_indexing
        )
        rho, Q, q1 = _radial_integral_matrix(2 * L if n_rho is None else n_rho)
        grid = LinearGrid(rho=rho, theta=4 * M + 1, zeta=4 * N + 1, NFP=eq.NFP, sym=False)
        d = eq.compute(["B_theta_r", "B_rho_t", "B_rho_z", "B_zeta_r"], grid=grid)
        sJ_zeta = (np.asarray(d["B_theta_r"]) - np.asarray(d["B_rho_t"])) / MU_0
        sJ_theta = (np.asarray(d["B_rho_z"]) - np.asarray(d["B_zeta_r"])) / MU_0

        # integrate along rho on the (rho, theta, zeta) layout, back to node order
        idx = np.asarray(grid.meshgrid_reshape(np.arange(grid.num_nodes), "rtz"))
        T = np.empty((grid.num_nodes, 2))
        for k, (f, sign) in enumerate([(sJ_zeta, 1), (sJ_theta, -1)]):
            T[idx.ravel(), k] = sign * np.einsum("ij,jtz->itz", Q, f[idx]).ravel()

        if edge_order is None:
            tr = Transform(grid, basis, build=False, build_pinv=True)
            c = np.stack([np.asarray(tr.fit(T[:, k])) for k in range(2)], axis=-1)
        else:
            nodes = np.asarray(grid.nodes)
            dT = np.stack([sJ_zeta[idx], -sJ_theta[idx]], axis=-1)
            T1 = np.einsum("j,jtzc->tzc", q1, dT)  # T on rho = 1
            c = _fit_smooth_edge(basis, edge_order, rho, nodes[idx[0, :, 0], 1],
                                 nodes[idx[0, 0, :], 2], dT, T1)
        return cls(basis, c, eq, edge_order)

    def impose_smooth_edge(self, edge_order=None):
        """Copy with d_rho^j T(1) = 0 (j = 1..k): sqrt(g) J^theta, J^zeta = O((1 - rho)^k).

        For each (m, n) and component, the k + 1 lowest-l coefficients are
        corrected (a smooth, low-degree change) so that the derivatives vanish
        and the boundary value is unchanged; (m, n) with fewer than k + 2 radial
        terms (the highest |m|) are set to zero.
        As a parametrization for the optimizer, these coefficients are linear
        functions of the others.
        """
        k = self.edge_order if edge_order is None else edge_order
        b = self.basis
        index = {tuple(mode): i for i, mode in enumerate(b.modes)}
        c = self.c.copy()
        for m in np.unique(b.modes[:, 1]):
            ls = np.unique(b.modes[b.modes[:, 1] == m, 0])
            if len(ls) < k + 2:
                c[b.modes[:, 1] == m] = 0
                continue
            # rows: boundary value and d_rho^j at rho = 1, j = 1..k
            A = np.stack([np.asarray(zernike_radial(np.ones((1, 1)), ls, np.full(len(ls), abs(m)),
                                                    dr=j))[0] for j in range(k + 1)])
            for n in np.unique(b.modes[b.modes[:, 1] == m, 2]):
                rows = np.array([index[(l, m, n)] for l in ls])
                r = A @ c[rows]  # (k + 1, 2)
                r[0] = 0  # keep the boundary value
                c[rows[: k + 1]] -= np.linalg.solve(A[:, : k + 1], r)
        return CurrentPotential(b, c, self.eq, k)

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
                            components=np.array(COMPONENTS),
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


def _fit_smooth_edge(basis, k, rho, theta, zeta, dT, T1):
    """Constrained least-squares T from d_rho T samples on a tensor grid.

    dT = d_rho (T_theta, T_zeta) = (sqrt(g) J^zeta, -sqrt(g) J^theta) has shape
    (n_rho, n_theta, n_zeta, 2) at the tensor product of the Chebyshev-Gauss
    ``rho``, ``theta`` (uniform) and ``zeta`` (uniform, one field period); T1,
    shape (n_theta, n_zeta, 2), is T on rho = 1. The Fourier coefficients in
    (theta, zeta) are projected first; then for each m (all n at once) the
    radial coefficients minimize the error in J, i.e. in d_rho T with weight
    1 / sqrt(g) ~ 1 / rho in the integral over rho, under d_rho^j T(1) = 0
    (j = 1..k), T(1) = T1 and, for m = 0, T(0) = 0. Returns the coefficients,
    shape (basis.num_modes, 2).
    """
    modes = basis.modes
    ms, ns = np.arange(-basis.M, basis.M + 1), np.arange(-basis.N, basis.N + 1)
    Pt = np.linalg.pinv(np.asarray(fourier(theta[:, None], ms[None, :])))
    Pz = np.linalg.pinv(np.asarray(fourier(zeta[:, None], ns[None, :], NFP=basis.NFP)))
    # sqrt of the Chebyshev-Gauss weights of int drho / rho
    q = ((1 - (2 * rho - 1) ** 2) ** 0.5 / rho) ** 0.5
    C = q[:, None, None, None] * np.einsum("aj,ijkc,bk->iabc", Pt, dT, Pz)
    C1 = np.einsum("aj,jkc,bk->abc", Pt, T1, Pz)
    index = {tuple(mode): i for i, mode in enumerate(modes)}
    c = np.zeros((basis.num_modes, 2))
    for a, m in enumerate(ms):
        ls = np.unique(modes[modes[:, 1] == m, 0])
        am = np.full(len(ls), abs(m))
        # constraint rows E c = e: T(1) = T1, d_rho^j T(1) = 0, T(0) = 0 for m = 0
        E = [np.asarray(zernike_radial(np.ones((1, 1)), ls, am, dr=j))[0] for j in range(k + 1)]
        e = [C1[a].reshape(-1)] + [np.zeros(C1[a].size)] * k
        if m == 0:
            E.append(np.asarray(zernike_radial(np.zeros((1, 1)), ls, am))[0])
            e.append(np.zeros(C1[a].size))
        if len(ls) <= len(E):  # too few terms for the constraints: mode left at zero
            continue
        E, e = np.array(E), np.array(e)
        c0 = np.linalg.lstsq(E, e, rcond=None)[0]  # (n_l, n_n * 2)
        Zr = q[:, None] * np.asarray(zernike_radial(rho[:, None], ls, am, dr=1))
        y = C[:, a].reshape(len(rho), -1) - Zr @ c0
        Nul = null_space(E)
        coef = c0 + (Nul @ np.linalg.lstsq(Zr @ Nul, y, rcond=None)[0] if Nul.shape[1] else 0)
        coef = coef.reshape(len(ls), len(ns), 2)
        for il, l in enumerate(ls):
            for b, n in enumerate(ns):
                if (l, m, n) in index:
                    c[index[(l, m, n)]] = coef[il, b]
    return c
