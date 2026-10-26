"""Free boundary: the plasma boundary moves with the solution, with B.n = 0 on it.

NEST solves force balance in the flux coordinates of a DESC equilibrium whose
boundary rho = 1 is fixed. Here the boundary is an unknown too, solved at the
same time as the current and the pressure (``FreeBoundaryFields``,
``FreeBoundaryObjective``).

Conditions at the boundary
--------------------------
B = B_ext + B_plasma is the total field everywhere (B_plasma the Biot-Savart
field of the volume current), and J = 0, p = 0 on rho = 1 for any unknowns
(``BoundaryPotential``, smooth-edge ``Pressure``). So there is no surface
current, B is continuous across the boundary, and pressure balance

    [[ |B|^2 / 2 mu0 + p ]] = 0

holds identically. What is left of the free-boundary problem is that the
boundary be a magnetic surface of the total field,

    B . n = 0 on the boundary                         (``BoundaryNormalField``),

with the toroidal flux inside it held (``ToroidalFlux``; DESC holds Psi in its
free-boundary solve, and without it the boundary could slide to any nearby
surface of the field).

Unknowns
--------
x = (u, c_p, xi). xi(theta, zeta) = sum_k xi_k F_k(theta, zeta) is the normal
displacement of the boundary [m] (``Displacement``): in each plane phi = zeta,
along the outward normal of the cross-section. It is extended into the plasma
as rho^|m| per Fourier mode, so the coordinates move with it,

    X(rho, theta, zeta) = X_a(rho, theta, zeta) + D(rho, theta, zeta; xi),

X_a the geometry of the anchor (the starting DESC equilibrium, then the last
re-anchored one). u and c_p keep their meaning in the moving coordinates:
sqrt(g) J^i and p(rho, theta, zeta) move with them (J is carried as a flux), so
div J = 0, J = 0 and p = 0 on the boundary and the net current hold for any xi.

Exact in xi:      the geometry at the collocation, boundary and cross-section
                  points, hence J, grad p, the volume, the boundary normal, the
                  flux element, and the positions at which B_plasma is
                  interpolated (``MovingInterpolator``).
First order in D: B_ext at the moved points (its gradient, once per anchor), and
                  the coordinates of the box points: J on the box is the moved
                  J (exact, on the flux grid) at s + ds, ds = -(dX/ds)^-1 D,
                  taken as J(s) + ds . grad_s J(s) with the derivatives of the
                  Lagrange interpolation (``BoxShift``). This is the derivative
                  of what the box samples, so it agrees with re-anchoring to
                  second order in D (the spectral curl(D x J) does not: the box
                  does not resolve a steep edge current).

The first-order parts become exact by re-anchoring: X_a <- X_a + D(xi), xi <- 0
(``Displacement.apply``), and everything is rebuilt on the new geometry. At
convergence xi -> 0 at the last anchor, where the residual is exact.

The residual is no longer quadratic in x: ``levenberg_marquardt`` must be run
with ``quadratic=False`` (each step is checked with the true residual).
"""

import functools

import jax
import jax.numpy as jnp
import numpy as np
from desc.basis import DoubleFourierSeries, fourier
from desc.grid import Grid

from batching import map_chunks
from flux_interp import (StencilInterpolator, TensorGeometry, TensorGrid, _lagrange, _Pytree,
                         separable_factors)
from fourier import fft_J
from objective import (VECTOR_PARITY, CurrentError, CurrentMoments, Fields, ObjectiveFunction, _cartesian,
                       _Objective, _symmetric_box)
from separable import Separable


class _Modes:
    """A list of (l, m, n) modes, as ``Separable`` reads a basis."""

    def __init__(self, modes):
        self.modes = np.asarray(modes)
        self.num_modes = len(self.modes)


# ---------------------------------------------------------------- displacement
class Displacement:
    """The normal displacement xi of the boundary [m] and its extension D inside.

    xi(theta, zeta) = sum_k xi_k F_k, a ``DoubleFourierSeries`` with |m| <= M,
    |n| <= N (even, cos-type, with stellarator symmetry). On rho = 1 the
    boundary moves in each plane phi = zeta along its outward in-plane normal,

        (dR, dZ) = xi (Z_theta, -R_theta) / |X_theta|   (sign: outward),

    projected on the boundary Fourier modes of the equilibrium's own bases
    (|m| <= eq.M, |n| <= eq.N) and extended as rho^|m|: the l = |m| Zernike
    modes of R_basis and Z_basis,

        D(rho, theta, zeta) = sum_mn (dR_mn, dZ_mn) rho^|m| F_m(theta) F_n(zeta).

    The displaced geometry is again a series in the equilibrium's bases (only
    its l = |m| coefficients change), linear in xi: ``A`` (n_xi, n_lm, n_n, 2)
    in the layout of ``sep`` (a ``Separable`` with components dR, dZ).
    """

    def __init__(self, eq, M=8, N=6, n_theta=None, n_zeta=None):
        self.eq, self.NFP, self.sym = eq, int(eq.NFP), bool(eq.sym)
        M, N = min(M, eq.M), min(N, eq.N)
        self.M, self.N = M, N
        self.basis = DoubleFourierSeries(M, N, NFP=eq.NFP, sym="cos" if self.sym else False)
        self.modes = self.basis.modes[:, 1:]
        self.n = self.basis.num_modes
        mR, mZ = np.asarray(eq.R_basis.modes), np.asarray(eq.Z_basis.modes)
        self.iR = np.flatnonzero(mR[:, 0] == np.abs(mR[:, 1]))
        self.iZ = np.flatnonzero(mZ[:, 0] == np.abs(mZ[:, 1]))
        self.sep = Separable([_Modes(mR[self.iR]), _Modes(mZ[self.iZ])],
                             [np.zeros(len(self.iR)), np.zeros(len(self.iZ))], eq.NFP)
        # outward in-plane normal of the boundary on a uniform grid over one period
        n_t = n_theta or 4 * eq.M + 2 * M + 1
        n_z = n_zeta or 4 * eq.N + 2 * N + 1
        T, Zt = (a.ravel() for a in np.meshgrid(2 * np.pi * np.arange(n_t) / n_t,
                                                2 * np.pi * np.arange(n_z) / (n_z * eq.NFP),
                                                indexing="ij"))
        RZ = Separable([eq.R_basis, eq.Z_basis], [eq.R_lmn, eq.Z_lmn], eq.NFP)
        one, Tj, Zj = jnp.ones(T.size), jnp.asarray(T), jnp.asarray(Zt)
        X = np.asarray(RZ.evaluate(one, Tj, Zj))
        Xt = np.asarray(RZ.evaluate(one, Tj, Zj, dt=1))
        X_axis = np.asarray(RZ.evaluate(0 * one, Tj, Zj))
        nrm = np.hypot(Xt[:, 0], Xt[:, 1])
        nR, nZ = Xt[:, 1] / nrm, -Xt[:, 0] / nrm
        s = np.sign(np.mean(nR * (X[:, 0] - X_axis[:, 0]) + nZ * (X[:, 1] - X_axis[:, 1])))
        nR, nZ = s * nR, s * nZ
        self.normal_sign = s
        F = np.asarray(self.basis.evaluate(Grid(np.column_stack([T * 0 + 1, T, Zt]), sort=False)))
        BR = np.asarray(fourier(T[:, None], mR[self.iR, 1]) * fourier(Zt[:, None], mR[self.iR, 2],
                                                                       NFP=eq.NFP))
        BZ = np.asarray(fourier(T[:, None], mZ[self.iZ, 1]) * fourier(Zt[:, None], mZ[self.iZ, 2],
                                                                       NFP=eq.NFP))
        # (n_R, n_xi), (n_Z, n_xi): the boundary modes of dR, dZ of each xi_k
        self.cR = np.linalg.lstsq(BR, F * nR[:, None], rcond=None)[0]
        self.cZ = np.linalg.lstsq(BZ, F * nZ[:, None], rcond=None)[0]
        # what the projection on the equilibrium's modes misses, relative
        err = max(np.linalg.norm(BR @ self.cR - F * nR[:, None]),
                  np.linalg.norm(BZ @ self.cZ - F * nZ[:, None])) / np.linalg.norm(F)
        self.projection_error = float(err)
        self.A = np.stack([np.asarray(self.sep.scatter([self.cR[:, k], self.cZ[:, k]]))
                           for k in range(self.n)])

    def coefficients(self, xi):
        """(dR_lmn, dZ_lmn) of xi on the equilibrium's full bases."""
        xi = np.asarray(xi, dtype=float)
        dR, dZ = np.zeros(self.eq.R_basis.num_modes), np.zeros(self.eq.Z_basis.num_modes)
        dR[self.iR], dZ[self.iZ] = self.cR @ xi, self.cZ @ xi
        return dR, dZ

    def apply(self, xi, eq=None):
        """A copy of the equilibrium with the geometry X + D(xi) (re-anchoring).

        Only R_lmn, Z_lmn (and the boundary surface and axis that follow from
        them) change: profiles, Psi and lambda are kept, so the copy is a
        coordinate system for NEST, not a DESC equilibrium."""
        eq = self.eq if eq is None else eq
        dR, dZ = self.coefficients(xi)
        new = eq.copy()
        new.R_lmn = np.asarray(eq.R_lmn) + dR
        new.Z_lmn = np.asarray(eq.Z_lmn) + dZ
        new.surface = new.get_surface_at(rho=1.0)
        new.axis = new.get_axis()
        return new


class MovingGeometry(_Pytree):
    """The geometry X_a + D(xi) on a tensor grid: X_a cached (``TensorGeometry``),
    D from its factor tables and the map A of a ``Displacement``."""

    _dynamic = ("geo", "f", "A")

    def __init__(self, geo, disp, grid):
        self.geo, self.grid = geo, grid
        self.f = separable_factors(disp.sep, grid)
        self.A = jnp.asarray(disp.A)

    def at(self, xi):
        """(geometry at xi, D = (dR, dZ) with shape grid.shape + (2,))."""
        C = jnp.einsum("k,kpqc->pqc", xi, self.A)
        D, Dr, Dt, Dz = self.f(C), self.f(C, dr=1), self.f(C, dt=1), self.f(C, dz=1)
        g = object.__new__(TensorGeometry)
        g.R, g.Z = self.geo.R + D[..., 0], self.geo.Z + D[..., 1]
        g.Xr, g.Xt, g.Xz = self.geo.Xr + Dr, self.geo.Xt + Dt, self.geo.Xz + Dz
        g.sqrt_g = g.R * (g.Xt[..., 0] * g.Xr[..., 1] - g.Xr[..., 0] * g.Xt[..., 1])
        return g, D


def _chunked(f, x, step=200000):
    """f(x) on the rows of x, ``step`` at a time (NumPy)."""
    return np.concatenate([np.asarray(f(x[i : i + step])) for i in range(0, len(x), step)])


def _with_geo(obj, geo):
    """A copy of a ``TensorCurrent`` or ``TensorPressure`` on another geometry."""
    new = object.__new__(type(obj))
    new.__dict__.update(obj.__dict__)
    new.geo = geo
    return new


def boundary_normal(g):
    """e_theta x e_zeta on a tensor grid (R, phi, Z components): the boundary normal
    times the area element per d theta d zeta."""
    R, (Rt, Zt), (Rz, Zz) = g.R, (g.Xt[..., 0], g.Xt[..., 1]), (g.Xz[..., 0], g.Xz[..., 1])
    return jnp.stack([-R * Zt, Zt * Rz - Rt * Zz, R * Rt], axis=-1)


# ---------------------------------------------------------------- interpolation
class MovingInterpolator(_Pytree):
    """``objective.SpectralInterpolator`` with points that move: the same type-2
    non-uniform FFT (and the same ES kernel, stencils and fine grid, so the values
    at the anchor points are identical), but the kernel weights are computed at
    each call from t + dt, so the interpolant is differentiable in the positions
    (its derivative is the spectral gradient of the box field).

    Parameters
    ----------
    t : (n, 3) array
        The anchor points as angles in [-pi, pi) of the box period (x, y, z).
    shape, dtype, eps, upsampfac, chunk : as ``SpectralInterpolator``
    """

    _dynamic = ("t0",)

    def __init__(self, t, shape, dtype=jnp.float32, eps=1e-6, upsampfac=1.25, chunk=16384):
        from nufftax.core.kernel import compute_kernel_params, kernel_fourier_series
        from nufftax.utils.grid import compute_grid_size
        self.shape, self.dtype = tuple(shape), dtype
        self.cdtype = jnp.complex64 if dtype == jnp.float32 else jnp.complex128
        kp = compute_kernel_params(eps, upsampfac)
        self.ns, self.beta, self.c = int(kp.nspread), float(kp.beta), float(kp.c)
        self.nf = tuple(compute_grid_size(n, upsampfac, kp.nspread) for n in self.shape)
        self.phihat = tuple(np.asarray(kernel_fourier_series(nf, kp.nspread, kp.beta, kp.c,
                                                             dtype=dtype)) for nf in self.nf)
        t = jnp.asarray(t, dtype)
        self.n = int(t.shape[0])
        self.chunk = chunk = min(chunk, self.n)
        self.m = -(-self.n // chunk) * chunk
        self.t0 = jnp.concatenate([t, jnp.zeros((self.m - self.n, 3), dtype)]).reshape(
            self.m // chunk, chunk, 3)

    def _weights(self, x, nf):
        """Stencil indices (c, ns) and ES weights (c, ns) of the angles x along an axis
        of fine size nf (nufftax's conventions: fold_rescale, i0 = ceil(x - ns / 2)).
        The square root is floored so that its derivative stays finite at the end
        of the support."""
        s = x / (2 * np.pi) + 0.5
        s = (s - jnp.floor(s)) * nf
        i0 = jnp.ceil(s - self.ns / 2).astype(jnp.int32)
        j = i0[:, None] + jnp.arange(self.ns)
        z = j.astype(x.dtype) - s[:, None]
        arg = 1 - self.c * z * z
        w = jnp.where(arg > 0, jnp.exp(self.beta * (jnp.sqrt(jnp.maximum(arg, 1e-7)) - 1)), 0)
        return j % nf, w.astype(self.dtype)

    def __call__(self, B_box, dt=None):
        """B_box (Nx, Ny, Nz, 3) -> (n, 3) at t0 + dt (angles), float64."""
        from nufftax.core.deconvolve import _build_deconv_factors_1d
        N = np.prod(self.shape)
        nf1, nf2, nf3 = self.nf
        t = self.t0
        if dt is not None:
            dt = jnp.concatenate([dt.astype(self.dtype),
                                  jnp.zeros((self.m - self.n, 3), self.dtype)])
            t = t + dt.reshape(t.shape)
        F = jnp.fft.fftshift(jnp.fft.fftn(B_box.astype(self.cdtype), axes=(0, 1, 2)),
                             axes=(0, 1, 2))
        F = jnp.transpose(F, (3, 2, 1, 0))
        fac = [_build_deconv_factors_1d(jnp.asarray(ph), n, 0).astype(F.dtype)
               for n, ph in zip(self.shape, self.phihat)]

        @functools.partial(jax.checkpoint, policy=jax.checkpoint_policies.nothing_saveable)
        def chunk(fw, tc):
            ix, wx = self._weights(tc[:, 0], nf1)
            iy, wy = self._weights(tc[:, 1], nf2)
            iz, wz = self._weights(tc[:, 2], nf3)
            flat = (iz[:, :, None, None] * (nf1 * nf2) + iy[:, None, :, None] * nf1
                    + ix[:, None, None, :])
            wt = wz[:, :, None, None] * wy[:, None, :, None] * wx[:, None, None, :]
            return fw, jnp.sum(fw[flat] * wt, axis=(1, 2, 3))

        def component(Fc):
            for ax, (n, nf) in enumerate(zip(self.shape, self.nf)):
                axis = 2 - ax
                Fc = Fc * fac[ax].reshape((-1,) + (1,) * (2 - axis))
                left = nf // 2 - n // 2
                pad = [(0, 0)] * 3
                pad[axis] = (left, nf - n - left)
                Fc = jnp.fft.ifftshift(jnp.pad(Fc, pad), axes=axis)
            fw = (jnp.real(jnp.fft.ifftn(Fc)) * (nf1 * nf2 * nf3)).reshape(-1)
            return jax.lax.scan(chunk, fw, t)[1].reshape(-1)

        @functools.partial(jax.checkpoint, policy=jax.checkpoint_policies.nothing_saveable)
        def step(F, c):
            return F, component(jax.lax.dynamic_index_in_dim(F, c, keepdims=False))

        out = jax.lax.scan(step, F, jnp.arange(3))[1]
        return (out.T[: self.n] / N).astype(jnp.float64)


class _Rows(_Pytree):
    """The first n rows of an interpolator's output (the collocation points)."""

    _dynamic = ("interp",)

    def __init__(self, interp, n):
        self.interp, self.n = interp, n

    def __call__(self, *B):
        return self.interp(*B)[: self.n]


def _lagrange_weights(x):
    """Lagrange weights on nodes 0..o-1 and their derivatives in s, from the offsets
    x[:, k] = s - node_k (in node spacings): (w, dw / ds), each (n, o)."""
    n, o = x.shape
    w, dw = np.ones((n, o)), np.zeros((n, o))
    for j in range(o):
        for k in range(o):
            if k != j:
                w[:, j] *= x[:, k] / (j - k)
        for l in range(o):
            if l != j:
                t = np.full(n, 1.0 / (j - l))
                for k in range(o):
                    if k not in (j, l):
                        t = t * x[:, k] / (j - k)
                dw[:, j] += t
    return w, dw


class BoxShift(_Pytree):
    """J at fixed box points when the coordinates move by D: the box point keeps
    its position x, so its coordinates move by ds = -(dX/ds)^-1 D (rho and theta;
    zeta = phi does not move), to first order. With the flux-grid field f (on the
    moved geometry), f(s + ds) = f(s) + d_rho f d rho + d_theta f d theta, the
    derivatives being those of the Lagrange interpolation of ``PointInterpolator``
    (the same stencils, weights differentiated): ``__call__`` returns the three in
    one gather.

    Parameters
    ----------
    grid : TensorGrid
        The uniform flux grid of J.
    rtz : (n, 3)
        The box points' anchor coordinates (those of ``Fields.to_box``).
    RZ : Separable
        The anchor geometry.
    disp : Displacement
    order : int
        As the ``PointInterpolator`` it differentiates.
    """

    _dynamic = ("idx", "w", "wr", "wt", "sign", "jinv", "D_P", "D_F", "A")

    def __init__(self, grid, rtz, RZ, disp, order=4, dtype=jnp.float32, chunk=65536):
        n_r, n_t, n_z = grid.shape
        true = np.array(rtz, dtype=float)
        rtz = true.copy()
        d_r, d_t = grid.rho[1] - grid.rho[0], 2 * np.pi / n_t
        sign_pt = np.ones(len(rtz))
        if grid.sym:
            period = 2 * np.pi / grid.NFP
            z = np.mod(rtz[:, 2], period)
            other = z > period / 2
            rtz[:, 1] = np.where(other, -rtz[:, 1], rtz[:, 1])
            rtz[:, 2] = np.where(other, period - z, z)
            sign_pt = np.where(other, -1.0, 1.0)
            d_z = (period / 2) / (n_z - 1)
        else:
            d_z = 2 * np.pi / (n_z * grid.NFP)
        o = order
        sr = rtz[:, 0] / d_r - 0.5
        ir, _ = _lagrange(sr, o)
        ir -= np.maximum(ir + o - n_r, 0)  # one-sided at the top, as PointInterpolator
        wr, dwr = _lagrange_weights(sr[:, None] - (ir[:, None] + np.arange(o)))
        st = np.mod(rtz[:, 1], 2 * np.pi) / d_t
        it, _ = _lagrange(st, o)
        wt, dwt = _lagrange_weights(st[:, None] - (it[:, None] + np.arange(o)))
        zeta = rtz[:, 2] if grid.sym else np.mod(rtz[:, 2], 2 * np.pi / grid.NFP)
        _, wz = _lagrange(zeta / d_z, o)
        R = ir[:, None] + np.arange(o)
        T = it[:, None] + np.arange(o)
        Zi = _lagrange(zeta / d_z, o)[0][:, None] + np.arange(o)
        Rb = np.broadcast_to(R[:, :, None, None], (len(rtz), o, o, o))
        Tb = np.broadcast_to(T[:, None, :, None], (len(rtz), o, o, o))
        Zb = np.broadcast_to(Zi[:, None, None, :], (len(rtz), o, o, o))
        sign = np.ones((len(rtz), o, o, o))
        if grid.sym:
            top = n_z - 1
            out = (Zb < 0) | (Zb > top)
            Zb = np.where(Zb < 0, -Zb, np.where(Zb > top, 2 * top - Zb, Zb))
            Tb = np.where(out, -Tb, Tb)
            sign = np.where(out, -1.0, 1.0)
        else:
            Zb = np.mod(Zb, n_z)
        refl = Rb < 0
        Rb = np.where(refl, -Rb - 1, Rb)
        Tb = np.mod(Tb + np.where(refl, n_t // 2, 0), n_t)
        idx = (Rb * n_t + Tb) * n_z + Zb
        prod = lambda a, b, c: (a[:, :, None, None] * b[:, None, :, None]  # noqa: E731
                                * c[:, None, None, :]).reshape(len(rtz), -1)
        self.idx = jnp.asarray(idx.reshape(len(rtz), -1).astype(np.int32))
        self.w = jnp.asarray(prod(wr, wt, wz), dtype)
        self.wr = jnp.asarray(prod(dwr / d_r, wt, wz), dtype)
        # d / d theta of the true theta: minus that of the mapped one
        self.wt = jnp.asarray(prod(wr, dwt / d_t, wz) * sign_pt[:, None], dtype)
        self.sign = jnp.asarray((sign * sign_pt[:, None, None, None]).reshape(len(rtz), -1), dtype)
        self.sym, self.chunk = grid.sym, chunk
        # (rho, theta) of the moved coordinates: -(dX/d(rho, theta))^-1 D at the points
        r, t, z = (jnp.asarray(true[:, i]) for i in range(3))
        Xr, Xt = np.asarray(RZ.evaluate(r, t, z, dr=1)), np.asarray(RZ.evaluate(r, t, z, dt=1))
        jac = np.stack([np.stack([Xr[:, 0], Xt[:, 0]], -1), np.stack([Xr[:, 1], Xt[:, 1]], -1)], 1)
        self.jinv = jnp.asarray(np.linalg.inv(jac))
        self.D_P = jnp.asarray(disp.sep.poloidal(r, t))
        self.D_F = jnp.asarray(disp.sep.toroidal(z))
        self.A = jnp.asarray(disp.A)

    def shift(self, xi):
        """(d rho, d theta) of the box points for the displacement xi, each (n,)."""
        C = jnp.einsum("k,kpqc->pqc", xi, self.A)
        D = jnp.einsum("nqc,nq->nc", jnp.einsum("np,pqc->nqc", self.D_P, C), self.D_F)
        d = -jnp.einsum("nij,nj->ni", self.jinv, D)
        return d[:, 0], d[:, 1]

    def __call__(self, field, parity):
        """field (n_rho, n_theta, n_zeta, c) on the flux grid -> its value and its rho
        and theta derivatives at the points, each (n, c) (parity as PointInterpolator)."""
        flat = field.reshape(-1, field.shape[-1])
        par = jnp.asarray(parity, self.w.dtype)

        def f(fl, i, w, wr, wt, s):
            g = fl[i]
            if self.sym:
                g = g * jnp.where(par < 0, s[:, :, None], 1.0)
            return tuple(jnp.einsum("nk,nkc->nc", v, g) for v in (w, wr, wt))

        return map_chunks(f, self.idx, self.w, self.wr, self.wt, self.sign, chunk=self.chunk,
                          carry=flat)


class StencilShift(_Pytree):
    """``BoxShift`` for the inside nodes of the cylindrical grid, with the stencils
    of a ``StencilInterpolator`` (formed at each call): J at the nodes for the
    displacement D is J(s) + ds . grad_s J(s), ds = -(dX/d(rho, theta))^-1 D, with
    D itself interpolated from the flux grid (``MovingGeometry.at``) by the same
    stencils. The cache: the nodes' flux coordinates (in the interpolator) and the
    inverse Jacobians (n, 2, 2) of the anchor geometry at them.

    Parameters
    ----------
    interp : StencilInterpolator
        The nodes and stencils (``Fields.to_box`` with ``cyl_stencil``).
    RZ : Separable
        The anchor geometry.
    """

    _dynamic = ("interp", "jinv")

    def __init__(self, interp, RZ, chunk=200000):
        from batching import host_chunks
        rtz = np.asarray(interp.rtz)

        @jax.jit
        def jinv(r):
            a, b, c = r[:, 0], r[:, 1], r[:, 2]
            Xr, Xt = RZ.evaluate(a, b, c, dr=1), RZ.evaluate(a, b, c, dt=1)
            jac = jnp.stack([jnp.stack([Xr[:, 0], Xt[:, 0]], -1),
                             jnp.stack([Xr[:, 1], Xt[:, 1]], -1)], 1)
            return jnp.linalg.inv(jac)

        self.interp = interp
        self.jinv = jnp.asarray(host_chunks(jinv, rtz, chunk=chunk))

    def __call__(self, J_flux, D_flux):
        """J (n, 3) at the nodes (parity of a symmetric vector, cylindrical
        components), shifted by the displacement D_flux (flux grid, (dR, dZ))."""
        f = jnp.concatenate([J_flux, D_flux.astype(J_flux.dtype)], axis=-1)
        v, vr, vt = self.interp(f, tuple(VECTOR_PARITY) + (1, -1), derivs=True)
        d = -jnp.einsum("nij,nj->ni", self.jinv.astype(v.dtype), v[:, 3:])
        return v[:, :3] + d[:, :1] * vr[:, :3] + d[:, 1:] * vt[:, :3]

    def tangent(self, J0, D0, dJ, dD):
        """The derivative of ``__call__`` at (J0, D0) along (dJ, dD), linear in (dJ, dD).
        Written out (not ``jax.jvp``) so that J0, D0 are constants of the chunked map
        and only (dJ, dD) its carry: the jvp of the carried map would keep a copy of the
        flux-grid field per chunk for its transpose."""
        it = self.interp
        dt = J0.dtype
        f0 = jnp.concatenate([J0, D0.astype(dt)], axis=-1).reshape(-1, 5)
        par = jnp.asarray([p < 0 for p in tuple(VECTOR_PARITY) + (1, -1)])

        def f(ft, rtz, jinv):
            i, w, s, wr, wt = it.stencil(rtz, True)
            sg = jnp.where(par[None, None, :], s[:, :, None], 1.0).astype(dt)
            g0, gt = f0[i] * sg, ft[i] * sg
            v0, vr0, vt0 = (jnp.einsum("nk,nkc->nc", q.astype(dt), g0) for q in (w, wr, wt))
            v1, vr1, vt1 = (jnp.einsum("nk,nkc->nc", q.astype(dt), gt) for q in (w, wr, wt))
            ji = jinv.astype(dt)
            d0 = -jnp.einsum("nij,nj->ni", ji, v0[:, 3:])
            d1 = -jnp.einsum("nij,nj->ni", ji, v1[:, 3:])
            return (v1[:, :3] + d0[:, :1] * vr1[:, :3] + d0[:, 1:] * vt1[:, :3]
                    + d1[:, :1] * vr0[:, :3] + d1[:, 1:] * vt0[:, :3])

        ft = jnp.concatenate([dJ, dD.astype(dt)], axis=-1).reshape(-1, 5)
        return map_chunks(f, it.rtz, self.jinv, chunk=it.chunk, carry=ft)


class DirectField(_Pytree):
    """B_plasma at points on or outside the plasma by direct Biot-Savart quadrature
    of the volume current,

        B(x) = mu0 / 4 pi sum_s (J dV)_s x (x - x_s) / |x - x_s|^3,

    with no box. The sources are a tensor grid on the moving geometry:
    Gauss-Legendre in rho, uniform in theta (offset by half a step from the
    boundary grid), and zeta midpoints over half a field period with stellarator
    symmetry (else over one period). They are copied by the mirror (x, -y, -z),
    which takes J to (-J_x, J_y, J_z), and by the NFP rotations. The field is
    then exactly symmetric and periodic, and independent of the box resolution.
    On NCSX the box's B.n at the boundary is off by ~1.5e-3 |B| at a/10 (5e-4 at
    a/20), and it changes from one field period to the next.

    J vanishes linearly at the boundary, so the integrand is bounded near a
    target on it; Gauss nodes cluster at rho = 1. Linear in u, differentiable in
    xi and in the target positions.

    The targets are the boundary grid of ``FreeBoundaryFields`` (theta_i = 2 pi i /
    n_theta, zeta midpoints of n_zeta cells per half period, or nodes over one
    period). The sources are q times finer in theta and zeta (q even) and offset
    by half a source step, so that, with their mirror and rotated copies, no
    source column meets a target: the nearest source sits a quarter of a target
    step away. Both move with the coordinates, so this holds for any xi.

    Parameters
    ----------
    current : BoundaryPotential
        On the anchor geometry.
    disp : Displacement
    boundary : (n_theta, n_zeta)
        The target grid (``FreeBoundaryFields``).
    n_rho : int
        Gauss-Legendre points in rho.
    q : int
        Sources per target step in theta and zeta (even).
    sym : bool
    dtype : float32 or float64
        Of the pair sums.
    chunk : int
        Targets per chunk (memory: chunk x sources x 3 values).
    """

    _dynamic = ("tc", "mg", "w", "zeta")

    def __init__(self, current, disp, boundary, n_rho, q, sym, dtype=jnp.float32, chunk=8):
        if q % 2:
            raise ValueError("q must be even (else sources meet the zeta midpoints)")
        n_tb, n_zb = boundary
        NFP = int(current.basis.NFP)
        xg, wg = np.polynomial.legendre.leggauss(n_rho)
        rho, w_r = (xg + 1) / 2, wg / 2
        n_t = q * n_tb
        theta = 2 * np.pi * (np.arange(n_t) + 0.5) / n_t
        # target cells: zeta midpoints of d (symmetric) or nodes of d (full period)
        d = (np.pi if sym else 2 * np.pi) / (NFP * n_zb)
        n_z = q * n_zb
        zeta = (np.arange(n_z) + 0.5) * d / q
        if not sym:  # targets on the nodes j d: sources at (j + 1/2) d / q never meet them
            zeta = (np.arange(n_z) + 0.5) * d / q
        d_z = d / q
        grid = TensorGrid(rho, theta, zeta, NFP)
        from flux_interp import TensorCurrent
        geo = TensorGeometry(current._RZ, grid)
        self.tc = TensorCurrent(current.S, current._RZ, grid, geo)
        self.mg = MovingGeometry(geo, disp, grid)
        self.w = jnp.asarray(w_r[:, None, None] * (2 * np.pi / n_t) * d_z * np.ones(grid.shape))
        self.zeta = jnp.asarray(zeta)
        self.NFP, self.sym, self.dtype, self.chunk = NFP, bool(sym), dtype, chunk
        self.n_sources = grid.size * (2 if sym else 1) * NFP

    def sources(self, u, I, xi):
        """Positions and J dV of all the sources (n, 3) each, Cartesian."""
        g, _ = self.mg.at(xi)
        J = _with_geo(self.tc, g)(u, I) * (jnp.abs(g.sqrt_g) * self.w)[..., None]
        c, s = jnp.cos(self.zeta), jnp.sin(self.zeta)
        P = jnp.stack([g.R * c, g.R * s, g.Z], axis=-1).reshape(-1, 3)
        Q = jnp.stack([J[..., 0] * c - J[..., 1] * s, J[..., 0] * s + J[..., 1] * c, J[..., 2]],
                      axis=-1).reshape(-1, 3)
        if self.sym:
            P = jnp.concatenate([P, P * jnp.asarray([1.0, -1.0, -1.0])])
            Q = jnp.concatenate([Q, Q * jnp.asarray([-1.0, 1.0, 1.0])])
        Ps, Qs = [], []
        for k in range(self.NFP):
            a = 2 * np.pi * k / self.NFP
            Rk = jnp.asarray([[np.cos(a), -np.sin(a), 0.0], [np.sin(a), np.cos(a), 0.0],
                              [0.0, 0.0, 1.0]])
            Ps.append(P @ Rk.T)
            Qs.append(Q @ Rk.T)
        return jnp.concatenate(Ps), jnp.concatenate(Qs)

    def __call__(self, u, I, xi, targets):
        """B_plasma at the targets (n, 3), float64."""
        from scipy.constants import mu_0
        P, Q = self.sources(u, I, xi)
        P, Q = P.astype(self.dtype), Q.astype(self.dtype)

        def f(carry, t):
            P, Q = carry
            d = t[:, None, :] - P[None]
            k = jnp.sum(d * d, axis=-1) ** -1.5
            B = jnp.stack([jnp.sum((Q[None, :, (i + 1) % 3] * d[..., (i + 2) % 3]
                                    - Q[None, :, (i + 2) % 3] * d[..., (i + 1) % 3]) * k, axis=1)
                           for i in range(3)], axis=-1)
            return B

        B = map_chunks(f, targets.astype(self.dtype), chunk=self.chunk, carry=(P, Q))
        return (mu_0 / (4 * np.pi) * B).astype(jnp.float64)


# ---------------------------------------------------------------- reference
def reference(inp, resolution, sym=None, current=None):
    """What stays DESC's across anchors, from the starting equilibrium on the
    collocation grid of ``resolution``: the force normalization and DESC's own
    force error, rms |J|, |B|, p, and DESC's current as sqrt(g) J^i (``N_desc``,
    carried with the coordinates as the target of ``TransportedCurrentError``).

    ``current``: a ``BoundaryPotential`` on the same equilibrium (the start fit) whose
    sqrt(g) J^i is the target instead of DESC's. Unlike DESC's (J != 0 on NCSX's LCFS),
    it is in the basis, so the current term does not pull against force balance."""
    from force import ForceResidual
    fr = ForceResidual(inp, None, resolution=resolution, sym=sym)
    g = fr.tgrid
    d = inp.compute_tensor(["sqrt(g)", "J^rho", "J^theta", "J^zeta", "J", "B", "p"],
                           g.rho, g.theta, g.zeta)
    if current is None:
        N = np.stack([d["sqrt(g)"] * d[k] for k in ("J^rho", "J^theta", "J^zeta")], axis=-1)
    else:
        from flux_interp import TensorCurrent
        tc = TensorCurrent(current.S, current._RZ, g, fr.geo)
        N = np.stack([np.asarray(v) for v in tc.contravariant(jnp.asarray(current.u), current.I)],
                     axis=-1)
    w = fr.dV / fr.dV.sum()
    rms = {k: float(np.sqrt(np.sum(w * np.sum(np.reshape(d[k], (len(w), -1)) ** 2, 1))))
           for k in ("J", "B", "p")}
    return {"norm": fr.norm, "F_desc_vol": fr.F_desc_vol, "rms": rms,
            "N_desc": N.reshape(g.shape + (3,)), "shape": g.shape, "a": inp.a}


# ---------------------------------------------------------------- fields
class FreeBoundaryFields(Fields):
    """``Fields`` with the boundary displacement xi as unknowns, x = (u, c_p, xi).

    Besides J, B, p, grad p and F at the collocation points, the fields hold
    B.n on the boundary (``Bn``, unit normal), the toroidal flux through the
    cross-section phi = 0 (``Psi``), the stored energy int p dV over the covered
    part of the plasma (``W``) and DESC's current carried with the coordinates
    (``J_desc``). See the module docstring for what is exact in xi.

    Parameters
    ----------
    force : ForceResidual
        On the anchor geometry; its ``norm`` should be the reference's.
    current, pressure : BoundaryPotential, Pressure
        On the anchor geometry (their u, c_p are the start).
    disp : Displacement
        On the anchor geometry.
    ref : dict
        ``reference`` of the starting equilibrium (``rms``, ``N_desc``).
    boundary : (n_theta, n_zeta)
        Boundary grid of B.n: n_theta uniform, and n_zeta midpoints over half a
        field period with stellarator symmetry (else n_zeta over one period).
    section : (n_rho, n_theta)
        Gauss-Legendre x uniform grid of the cross-section phi = 0 (Psi).
    gradient_step : float
        Of the forward differences of B_ext [m].
    grad_B_ext : (n_points, 3, 3) array, optional
        The gradient of B_ext at the points from a nearby anchor (same grids):
        reused instead of three more coil sums. B_ext itself is always exact.
    direct : (n_rho, q), optional
        Source grid of ``DirectField`` (n_rho Gauss points, q times the boundary
        grid in theta and zeta): B_plasma on the boundary (hence B.n) by direct
        Biot-Savart instead of from the box.
    direct_tangent : bool or (n_rho, q)
        Also differentiate the direct field: True with the same sources, or a
        coarser ``DirectField`` (n_rho, q) for the derivative only. Each product
        J v, J^T w then holds a direct sum. By default the derivative of B on the
        boundary is the box's: an inexact Newton step, whose length is checked
        with the true residual (``levenberg_marquardt(quadratic=False)``). The
        box's B.n is off by ~3 % of B_plasma.n at a/10, so once B.n/|B| is
        ~2e-4 its steps stop decreasing the true residual. Memory: JAX's
        transpose of the differentiated chunked sum keeps the target x source
        array (53 GB for NCSX at n_rho 16, q 2), so this is for small grids
        until the sum gets an explicit transpose. Recommended: the box's error
        there is above the B.n wanted, and differs between field periods, so the
        half-period residual would be fitted to it.
    others : as ``Fields``.
    """

    _dynamic = Fields._dynamic + ("mg_col", "mg_bnd", "mg_sec", "mg_flux", "box_shift",
                                  "N_desc", "w_vol", "w_bnd", "w_sec", "B_ext_all",
                                  "grad_B_ext", "phi_all", "move", "box_scale", "direct",
                                  "direct_d", "x_bnd")

    def __init__(self, force, current, pressure, disp, ref, boundary=None, section=None,
                 kernel="truncated", flux_grid=None, order=4, dtype=jnp.float32, chunk=65536,
                 gradient_step=1e-5, grad_B_ext=None, direct=None, direct_tangent=False,
                 verbose=True, **fields_kw):
        self.disp, self.n_xi, self.ref = disp, disp.n, ref
        self.xi_scale = 1.0  # the unknowns are xi / xi_scale (``scale_xi``)
        b = current.basis
        eq = force.inp.eq
        self._boundary = boundary or (2 * (b.M + eq.M) + 1,
                                      (b.N + eq.N) + 1 if force.sym else 2 * (b.N + eq.N) + 1)
        self._section = section or (eq.L + 1, 4 * eq.M + 1)
        self._gradient_step, self._verbose = gradient_step, verbose
        self._grad = grad_B_ext
        if "B_ext" not in vars(force):  # the coil field in chunks (one call: ~18 GB for NCSX)
            force.B_ext = _chunked(force.inp.B_ext, force.x)
        super().__init__(force, current, pressure, kernel=kernel, flux_grid=flux_grid,
                         order=order, dtype=dtype, chunk=chunk, **fields_kw)
        self.x0 = jnp.concatenate([self.x0, jnp.zeros(self.n_xi)])
        if tuple(ref["shape"]) != force.tgrid.shape:
            raise ValueError(f"collocation grid {force.tgrid.shape} differs from the "
                             f"reference's {tuple(ref['shape'])}")
        self.N_desc = jnp.asarray(ref["N_desc"])
        self.norm = ref["norm"]
        # quadrature weights of the collocation grid (dV / |sqrt(g)|): W = sum p |sqrt(g)| w
        self.w_vol = jnp.asarray((force.dV / np.abs(np.asarray(force.geo.sqrt_g)).ravel())
                                 .reshape(force.tgrid.shape))
        # J on the box: the flux grid on the moving geometry, the box points' moving
        # coordinates
        self.mg_flux = MovingGeometry(self.J_flux.geo, disp, self.J_flux.grid)
        if self.cyl:  # the inside nodes of the cylindrical grid, cylindrical components
            if not isinstance(self.to_box, StencilInterpolator):
                self.to_box = StencilInterpolator(self.J_flux.grid, self.rows_rtz, order,
                                                  chunk=chunk, dtype=self.cyl_dtype)
            self.box_shift = StencilShift(self.to_box, current._RZ)
        else:
            self.box_shift = BoxShift(self.J_flux.grid, self._box_rtz(force.box), current._RZ,
                                      disp, order, dtype, chunk)
        # B_plasma on the boundary by direct Biot-Savart (the box's is too coarse there)
        self.direct = None if direct is None else DirectField(
            current, disp, self._boundary, direct[0], direct[1], self.sym, dtype=dtype)
        self.direct_d = None  # the direct field of the derivative
        if direct is not None and direct_tangent is not False and direct_tangent is not None:
            self.direct_d = self.direct if direct_tangent is True else DirectField(
                current, disp, self._boundary, direct_tangent[0], direct_tangent[1], self.sym,
                dtype=dtype)
        self.to_mesh()  # with the annulus solver on several devices (``cyl_mesh``)

    # ------------------------------------------------------------ setup
    def _box_rtz(self, box):
        """(rho, theta, zeta) of the box points J is interpolated at (as in ``Fields``)."""
        inside, rtz = np.asarray(box.inside), np.asarray(box.rtz)
        if self.sym and _symmetric_box(box):
            ix, iy, iz = np.unravel_index(inside, box.shape)
            keep = (box.ys[iy] > 0) | ((box.ys[iy] == 0) & (box.zs[iz] >= 0))
            rtz = rtz[keep]
        return rtz

    def _points(self, force, box, dtype, chunk):
        """Collocation, boundary and cross-section points: their moving geometry,
        weights, B_ext and its gradient, and one interpolator for all of them."""
        import time
        t0 = time.time()
        eq, inp, disp = force.inp.eq, force.inp, self.disp
        RZ = Separable([eq.R_basis, eq.Z_basis], [eq.R_lmn, eq.Z_lmn], eq.NFP)
        n_t, n_z = self._boundary
        theta = 2 * np.pi * np.arange(n_t) / n_t
        if self.sym:
            zeta = (np.arange(n_z) + 0.5) * np.pi / (eq.NFP * n_z)
            d_z = np.pi / (eq.NFP * n_z)
        else:
            zeta = 2 * np.pi * np.arange(n_z) / (eq.NFP * n_z)
            d_z = 2 * np.pi / (eq.NFP * n_z)
        bgrid = TensorGrid([1.0], theta, zeta, eq.NFP)
        n_r, n_ts = self._section
        xg, wg = np.polynomial.legendre.leggauss(n_r)
        sgrid = TensorGrid((xg + 1) / 2, 2 * np.pi * np.arange(n_ts) / n_ts, [0.0], eq.NFP)
        bgeo, sgeo = TensorGeometry(RZ, bgrid), TensorGeometry(RZ, sgrid)
        self.mg_col = MovingGeometry(force.geo, disp, force.tgrid)
        self.mg_bnd = MovingGeometry(bgeo, disp, bgrid)
        self.mg_sec = MovingGeometry(sgeo, disp, sgrid)
        dA = np.linalg.norm(np.asarray(boundary_normal(bgeo)), axis=-1).ravel() * (
            2 * np.pi / n_t) * d_z
        self.area = float(dA.sum())
        self.w_bnd = jnp.asarray(np.sqrt(dA / dA.sum()))
        self.w_sec = jnp.asarray((wg[:, None] / 2 * (2 * np.pi / n_ts) * np.ones((n_r, n_ts)))
                                 .ravel())
        cyl = lambda g, phi: np.column_stack([np.asarray(g.R).ravel() * np.cos(phi),  # noqa: E731
                                              np.asarray(g.R).ravel() * np.sin(phi),
                                              np.asarray(g.Z).ravel()])
        phi_b = bgrid.nodes()[:, 2]
        phi_s = np.zeros(sgrid.size)
        x_all = np.concatenate([force.x, cyl(bgeo, phi_b), cyl(sgeo, phi_s)])
        self.n_col, self.n_bnd, self.n_sec = len(force.x), bgrid.size, sgrid.size
        self.x_bnd = jnp.asarray(x_all[self.n_col : self.n_col + self.n_bnd])
        self.phi_all = jnp.asarray(np.concatenate([force.phi, phi_b, phi_s]))
        self.bgrid, self.sgrid = bgrid, sgrid
        # B_ext at all points and its gradient by forward differences (coil
        # Biot-Savart sums, in chunks)
        h = self._gradient_step

        bext = functools.partial(_chunked, inp.B_ext)
        B0 = np.concatenate([np.asarray(force.B_ext), bext(x_all[self.n_col :])])
        if self._grad is not None and np.shape(self._grad) == (len(x_all), 3, 3):
            G = np.asarray(self._grad)  # a previous anchor's (it only enters the model in xi)
        else:
            G = np.stack([(bext(x_all + h * np.eye(3)[j]) - B0) / h for j in range(3)], axis=-1)
        self.B_ext_all = jnp.asarray(B0)
        self.grad_B_ext = jnp.asarray(G)  # [n, i, j] = d B_i / d x_j
        if self.cyl:  # one moving interpolator on the annulus for all points
            from cylindrical_grid import CylindricalInterpolator
            rpz = np.column_stack([np.hypot(x_all[:, 0], x_all[:, 1]),
                                   np.asarray(self.phi_all), x_all[:, 2]])
            self.move = CylindricalInterpolator(self.cyl_grid, rpz, dtype=dtype,
                                                order=self.cyl_order)
            self.t = self.box_scale = None
            self.to_points = _Rows(self.move, self.n_col)
            if self._verbose:
                print(f"  free boundary: {self.n_xi} boundary modes (M, N = {disp.M}, {disp.N}), "
                      f"boundary grid {bgrid.shape[1:]}, section {sgrid.shape[:2]}, B_ext "
                      f"gradient at {len(x_all):,} points; cylindrical grid {box.shape}; "
                      f"{time.time() - t0:.0f} s", flush=True)
            return
        # the points as angles of the box period, one moving interpolator
        x0 = np.array([a[0] for a in box.axes])
        L = np.array([len(a) * (a[1] - a[0]) for a in box.axes])
        t = 2 * np.pi * (x_all - x0) / L
        t = (t + np.pi) % (2 * np.pi) - np.pi
        self.box_scale = jnp.asarray(2 * np.pi / L)
        self.move = MovingInterpolator(t, box.shape, dtype, self.eps, chunk=4 * chunk)
        self.t = jnp.asarray(t[: self.n_col], dtype=dtype)
        self.to_points = _Rows(self.move, self.n_col)
        if self._verbose:
            print(f"  free boundary: {self.n_xi} boundary modes (M, N = {disp.M}, {disp.N}), "
                  f"boundary grid {bgrid.shape[1:]}, section {sgrid.shape[:2]}, B_ext gradient "
                  f"at {len(x_all):,} points; {time.time() - t0:.0f} s", flush=True)

    # ------------------------------------------------------------ interface
    @property
    def n(self):
        return self.n_u + self.n_p + self.n_xi

    def split(self, x):
        return x[: self.n_u], x[self.n_u : self.n_u + self.n_p], x[self.n_u + self.n_p :]

    @property
    def rms(self):
        return self.ref["rms"]

    @property
    def desc(self):
        raise NotImplementedError("DESC's fields on a moved geometry: use the reference "
                                  "(TransportedCurrentError for J)")

    def scale_xi(self, s):
        """Unknowns xi / s instead of xi (s in metres per unit): xi's columns of the
        Jacobian are then s times larger. To bring them to the size of the
        current's, so that the optimizer's damping floor (a fraction of the mean
        of diag(J^T J)) does not swamp the steps of u and c_p."""
        for mg in (self.mg_col, self.mg_bnd, self.mg_sec, self.mg_flux, self.box_shift):
            if hasattr(mg, "A"):  # (StencilShift takes D from mg_flux)
                mg.A = mg.A * s
        for d in {id(f): f for f in (self.direct, self.direct_d) if f is not None}.values():
            d.mg.A = d.mg.A * s
        self.xi_scale *= s
        return self

    def xi_metres(self, x):
        """The displacement coefficients of x in metres."""
        return np.asarray(self.split(x)[2]) * self.xi_scale

    def fixed_view(self):
        """A plain ``Fields`` on the anchor geometry sharing these caches (x = (u, c_p)),
        e.g. for the block preconditioner of the current and pressure."""
        v = object.__new__(Fields)
        v.__dict__.update({k: w for k, w in self.__dict__.items() if k != "_aux"})
        v.x0 = self.x0[: self.n_u + self.n_p]
        v._desc = {"rms": self.ref["rms"], "fields": {}}
        return v

    # ------------------------------------------------------------ evaluation
    def J_half(self, u, xi, I=None):
        """J at the box points it is interpolated at (Cartesian) for the displacement
        xi (``BoxShift``); linear in u for I = 0."""
        g, D = self.mg_flux.at(xi)
        if self.cyl:  # cylindrical components on the annulus, in its precision
            J_flux = _with_geo(self.J_flux, g)(u, self.I if I is None else I)
            return self.box_shift(J_flux.astype(self.cyl_dtype), D)
        J_flux = _with_geo(self.J_flux, g)(u, self.I if I is None else I).astype(self.dtype)
        J, Jr, Jt = self.box_shift(J_flux, VECTOR_PARITY)
        dr, dt = self.box_shift.shift(xi)
        J = J + dr.astype(self.dtype)[:, None] * Jr + dt.astype(self.dtype)[:, None] * Jt
        return _cartesian(J, self.box_phi.astype(self.dtype))

    def _assemble(self, V):
        """A field at the J points onto the box (mirror rows, zero outside); on the
        cylindrical grid, the rows of all inside nodes (``Fields._cyl_rows``)."""
        if self.cyl:
            return self._cyl_rows(V)
        if self.half is not None:
            V = jnp.concatenate([V, V[self.half_rows] * jnp.asarray([-1.0, 1.0, 1.0], V.dtype)])
        V = jnp.pad(V, ((0, 1), (0, 0)))
        return V[self.src].reshape(self.shape + (3,))

    def _box_solve(self, J_hat):
        solve = jax.checkpoint(lambda J_hat, K_hat: self._solve(J_hat, self.axes, K_hat),
                               policy=jax.checkpoint_policies.nothing_saveable)
        return solve(J_hat, self.K_hat)

    def _grid_field(self, J_grid):
        """B_plasma of J on the grid (``_assemble``): the box field, or on the annulus
        the FFT-index slabs (B_+, B_z) of ``CylindricalSolver.field_modes_F``."""
        if self.cyl and self.cyl_fused:  # B on the points' planes
            return self._cyl_planes(J_grid, self.move.Ep)
        if self.cyl:
            return self.solver.field_modes_F(*self._cyl_sources(J_grid))
        return self._box_solve(fft_J(J_grid))

    def _interp(self, B_box, dx):
        if self.cyl:  # the points move in their planes: (dR, dZ)
            ph = self.phi_all
            d = jnp.stack([dx[:, 0] * jnp.cos(ph) + dx[:, 1] * jnp.sin(ph), dx[:, 2]], axis=-1)
            B = self.move.from_planes(*B_box, d) if self.cyl_fused else self.move(*B_box, d)
            return _cartesian(B, ph)
        return self.move(B_box, dx * self.box_scale)

    def _interp_d(self, B_box, dx, ddx):
        """The derivative of ``_interp(B_box, dx)`` in dx along ddx (cylindrical grid):
        ``CylindricalInterpolator.from_planes_dd``, the field on the planes held."""
        ph = self.phi_all
        cyl = lambda v: jnp.stack([v[:, 0] * jnp.cos(ph) + v[:, 1] * jnp.sin(ph), v[:, 2]],  # noqa: E731
                                  axis=-1)
        planes = B_box if self.cyl_fused else self.move.planes(*B_box)
        return _cartesian(self.move.from_planes_dd(*planes, cyl(dx), cyl(ddx)), ph)

    def _local(self, x):
        """Everything but B_plasma: J, J_desc, p, grad p at the collocation points, the
        displacement of every point, the boundary normal, the flux element and W;
        exact in xi."""
        u, c, xi = self.split(x)
        gc, Dc = self.mg_col.at(xi)
        gb, Db = self.mg_bnd.at(xi)
        gs, Ds = self.mg_sec.at(xi)
        J = _cartesian(_with_geo(self.J_col, gc)(u, self.I).reshape(-1, 3), self.phi)
        p, gp = _with_geo(self.p_col, gc)(c)
        N = self.N_desc
        J_desc = _cartesian(gc.vector(N[..., 0], N[..., 1], N[..., 2]).reshape(-1, 3), self.phi)
        d = jnp.concatenate([Dc.reshape(-1, 2), Db.reshape(-1, 2), Ds.reshape(-1, 2)])
        ph = self.phi_all
        dx = jnp.stack([d[:, 0] * jnp.cos(ph), d[:, 0] * jnp.sin(ph), d[:, 1]], axis=-1)
        nb = _cartesian(boundary_normal(gb).reshape(-1, 3), ph[self.n_col : self.n_col + self.n_bnd])
        nhat = nb / jnp.linalg.norm(nb, axis=1, keepdims=True)
        fac = (gs.Xr[..., 1] * gs.Xt[..., 0] - gs.Xr[..., 0] * gs.Xt[..., 1]).ravel()
        W = jnp.sum(p * jnp.abs(gc.sqrt_g) * self.w_vol)[None]
        return {"J": J, "J_desc": J_desc, "p": p.ravel(), "grad_p": _cartesian(gp.reshape(-1, 3),
                                                                                self.phi),
                "dx": dx, "nhat": nhat, "fac": fac, "W": W, "x_p": c}

    def _fields(self, loc, B_all):
        """The fields from the local part and B at all points (bilinear)."""
        n_c, n_b = self.n_col, self.n_bnd
        Bc, Bb, Bs = B_all[:n_c], B_all[n_c : n_c + n_b], B_all[n_c + n_b :]
        return {"J": loc["J"], "J_desc": loc["J_desc"], "B": Bc, "p": loc["p"],
                "grad_p": loc["grad_p"], "F": jnp.cross(loc["J"], Bc) - loc["grad_p"],
                "Bn": jnp.sum(Bb * loc["nhat"], axis=1), "B_bnd": Bb,
                "Psi": jnp.sum(Bs[:, 1] * loc["fac"] * self.w_sec)[None], "W": loc["W"],
                "x_p": loc["x_p"]}

    def _J_half_x(self, x):
        u, _, xi = self.split(x)
        return self.J_half(u, xi)

    def _flux_JD(self, x):
        """J (cylindrical precision) and D on the flux grid at x (cylindrical grid)."""
        u, _, xi = self.split(x)
        g, D = self.mg_flux.at(xi)
        return _with_geo(self.J_flux, g)(u, self.I).astype(self.cyl_dtype), D

    def _bnd_direct(self, x, dxb, field=None):
        """B_plasma on the boundary moved by dxb, by ``DirectField``."""
        u, _, xi = self.split(x)
        return (field or self.direct)(u, self.I, xi, self.x_bnd + dxb)

    def _state(self, x):
        loc = self._local(x)
        B_box = self._grid_field(self._assemble(self._J_half_x(x)))
        B_ext = self.B_ext_all + jnp.einsum("nij,nj->ni", self.grad_B_ext, loc["dx"])
        B_all = B_ext + self._interp(B_box, loc["dx"])
        if self.direct is not None:
            b = slice(self.n_col, self.n_col + self.n_bnd)
            B_all = B_all.at[b].set(B_ext[b] + self._bnd_direct(x, loc["dx"][b]))
        return loc, B_box, B_all

    def _compute(self, x):
        loc, _, B_all = self._state(x)
        return self._fields(loc, B_all)

    def _base(self, x):
        """What the derivative at x needs: x, B on the box and at all points, and the
        fields (``_tangent``)."""
        loc, B_box, B_all = self._state(x)
        f = self._fields(loc, B_all)
        return {"x": x, "B_box": B_box, "B_all": B_all, "J": f["J"], "B": f["B"],
                "p": f["p"], "grad_p": f["grad_p"]}, f

    def _tangent(self, v, base):
        """The derivative of the fields at base["x"] along v (linear in v)."""
        loc0, dloc = jax.jvp(self._local, (base["x"],), (v,))
        if self.cyl:  # explicit: the large primal arrays as constants (``StencilShift.tangent``)
            (J0, D0), (dJ, dD) = jax.jvp(self._flux_JD, (base["x"],), (v,))
            dJ_half = self.box_shift.tangent(J0, D0, dJ, dD)
            dB_box = self._grid_field(self._assemble(dJ_half))
            dB_pl = (self._interp(dB_box, loc0["dx"])
                     + self._interp_d(base["B_box"], loc0["dx"], dloc["dx"]))
        else:
            dJ_half = jax.jvp(self._J_half_x, (base["x"],), (v,))[1]
            dB_box = self._grid_field(self._assemble(dJ_half))
            _, dB_pl = jax.jvp(self._interp, (base["B_box"], loc0["dx"]), (dB_box, dloc["dx"]))
        dB_ext = jnp.einsum("nij,nj->ni", self.grad_B_ext, dloc["dx"])
        dB_all = dB_ext + dB_pl
        if self.direct_d is not None:
            b = slice(self.n_col, self.n_col + self.n_bnd)
            dB_d = jax.jvp(lambda x, dxb: self._bnd_direct(x, dxb, self.direct_d),
                           (base["x"], loc0["dx"][b]), (v, dloc["dx"][b]))[1]
            dB_all = dB_all.at[b].set(dB_ext[b] + dB_d)
        return jax.jvp(self._fields, (loc0, base["B_all"]), (dloc, dB_all))[1]


# ---------------------------------------------------------------- objectives
class TransportedCurrentError(CurrentError):
    """J - J_DESC, with DESC's current carried with the coordinates (its sqrt(g) J^i
    fixed, ``FreeBoundaryFields.J_desc``); the same as ``CurrentError`` at xi = 0
    of the first anchor."""

    def compute(self, f):
        return f["J"] - f["J_desc"]

    def _default_target(self, fields):
        return 0.0


class TransportedCurrentMoments(CurrentMoments):
    """``CurrentMoments`` of the start fit's own J (not DESC's, which differs where the fit's
    edge factor forces J -> 0); pass ``target`` to keep it across anchors."""

    def _start_J(self, fields):
        return fields.compute(fields.x0)["J"]


class StoredEnergy(_Objective):
    """int p dV (over the covered part of the plasma) held: one residual,
    weight (W - target) / target. The volume moves with the boundary; the target
    (default: W at the start) is kept across anchors."""

    name, quantity = "W", "W"

    def build(self, fields):
        if self.target is None:
            self.target = float(fields.compute(fields.x0)[self.quantity][0])
        if self.normalization is None:
            self.normalization = abs(float(self.target))
        self.target = jnp.asarray(self.target)
        self.dim_f = 1
        self._sw = fields.sw
        return self

    def compute(self, f):
        return f[self.quantity]

    def compute_scaled_tangent(self, e):
        return self.weight * e / self.normalization


class ToroidalFlux(StoredEnergy):
    """The toroidal flux through the cross-section phi = 0 held (as DESC holds Psi)."""

    name, quantity = "Psi", "Psi"


class BoundaryNormalField(_Objective):
    """B.n -> 0 on the boundary (unit normal), normalized by rms |B| there:
    |scaled error|^2 = weight^2 <(B.n)^2>_area / <|B|^2>_area."""

    name, quantity = "B.n", "Bn"

    def build(self, fields):
        if self.normalization is None:
            B = np.asarray(fields.compute(fields.x0)["B_bnd"])
            w = np.asarray(fields.w_bnd) ** 2
            self.normalization = float(np.sqrt(np.sum(w * np.sum(B**2, 1))))
        self.target = jnp.asarray(0.0 if self.target is None else self.target)
        self.dim_f = fields.n_bnd
        self._sw = fields.w_bnd
        return self


class FreeBoundaryObjective(ObjectiveFunction):
    """``ObjectiveFunction`` on ``FreeBoundaryFields``: the derivative comes from
    ``FreeBoundaryFields._tangent`` (the residual is not quadratic in xi)."""

    def linearize(self, x):
        r, base = _fb_base(self, x, self.fields)
        return (r, (lambda v: _fb_lin(self, v, base, self.fields)),
                (lambda w: _fb_lin_T(self, w, base, self.fields)))

    def _lin_impl(self, v, base, fields):
        d = fields._tangent(v, base)
        return jnp.concatenate([o.compute_scaled_tangent(o.tangent(d, base))
                                for o in self.objectives])

    def report(self, x):
        f = self.fields.compute(x)
        w = np.asarray(self.fields.sw) ** 2
        out = {"<|F|>/norm": float(np.sum(w * np.linalg.norm(np.asarray(f["F"]), axis=1)))
               / self.fields.norm}
        scalar = 0.0
        for o in self.objectives:
            e = float(jnp.linalg.norm(o.compute_scaled_error(f)))
            if o.weight:  # a weight-0 term (B.n with --fixed-boundary) is only reported below
                out[o.name] = e / o.weight
            scalar += 0.5 * e**2
        out["scalar"] = scalar
        Bb = np.asarray(f["B_bnd"])
        w = np.asarray(self.fields.w_bnd) ** 2
        rel = np.asarray(f["Bn"]) / np.linalg.norm(Bb, axis=1)
        out["Bn/|B| rms"] = float(np.sqrt(np.sum(w * rel**2)))
        out["Bn/|B| max"] = float(np.abs(rel).max())
        out["max |xi| [m]"] = float(np.abs(self.fields.xi_metres(x)).max()) if \
            self.fields.n_xi else 0.0
        return out


@functools.partial(jax.jit, static_argnums=0)
def _fb_base(objfun, x, fields):
    base, f = fields._base(x)
    return jnp.concatenate([o.compute_scaled_error(f) for o in objfun.objectives]), base


@functools.partial(jax.jit, static_argnums=0)
def _fb_lin(objfun, v, base, fields):
    return objfun._lin_impl(v, base, fields)


@functools.partial(jax.jit, static_argnums=0)
def _fb_lin_T(objfun, w, base, fields):
    x = jax.ShapeDtypeStruct((objfun.dim_x,), w.dtype)
    return jax.linear_transpose(lambda v: objfun._lin_impl(v, base, fields), x)(w)[0]


def release(*objs):
    """Delete the device arrays owned by NEST objects (fields, objectives, force
    residual, their caches), recursively through their attributes, lists and dicts;
    DESC objects are not entered. JAX's caches keep the static part of the pytrees
    passed to jitted functions alive after the objects are gone, and with it some
    of their arrays: without this, every anchor leaves its caches on the GPU (about
    4 GB each for NCSX at a/10). Returns the bytes released."""
    mods = ("flux_interp", "objective", "free_boundary", "force", "grid", "precondition",
            "boundary_potential", "pressure", "separable", "dense", "cylindrical_grid")
    seen, freed = set(), [0]

    def visit(o):
        if id(o) in seen:
            return
        seen.add(id(o))
        if isinstance(o, jax.Array):
            if not o.is_deleted():
                freed[0] += o.nbytes
                o.delete()
        elif isinstance(o, dict):
            for v in o.values():
                visit(v)
        elif isinstance(o, (list, tuple)):
            for v in o:
                visit(v)
        elif hasattr(o, "__dict__") and type(o).__module__ in mods:
            for v in vars(o).values():
                visit(v)
        elif type(o).__name__ == "_Static":
            visit(o.d)

    for o in objs:
        visit(o)
    return freed[0]


# ---------------------------------------------------------------- preconditioner
class FreeBoundaryPreconditioner:
    """Preconditioner of (J^T J + lam D) for x = (u, c_p, xi).

    The (u, c_p) part is ``precondition.BlockPreconditioner`` B (on
    ``FreeBoundaryFields.fixed_view``). The xi columns of J^T J are formed exactly,
    C = J^T J E_xi (n_xi products), which gives the xi block G and its coupling
    C_u to (u, c_p). With ``coupled`` the solve is the block LDL^T one with B in
    place of the (u, c_p) block,

        y_u = B^-1 r_u,  y_xi = S^-1 (r_xi - C_u^T y_u),  x_u = y_u - B^-1 C_u y_xi,
        S = G + lam D_xi - C_u^T B^-1 C_u   (n_xi x n_xi),

    so that a step of the boundary comes with the change of current and pressure
    it needs (the two are strongly coupled at NCSX's edge). B^-1 C_u and S are
    recomputed when lam changes. Without ``coupled`` (default): B and G alone. On
    NCSX the coupled form converges worse: B is too rough an A_uu for the
    complement."""

    def __init__(self, block, obj, x, floor=1e-10, coupled=False, verbose=True, G=None):
        import time
        t0 = time.time()
        n, n_xi = obj.dim_x, obj.fields.n_xi
        self.n_up = n - n_xi
        C = np.zeros((n, n_xi), dtype=np.float32 if coupled else np.float64)
        if G is not None:  # given (e.g. formed on a coarser grid: it only preconditions)
            if coupled or np.shape(G) != (n_xi, n_xi):
                raise ValueError("G: (n_xi, n_xi), without coupled")
            G = np.asarray(G, dtype=float)
        else:
            _, J, JT = obj.linearize(jnp.asarray(x))
            G = np.zeros((n_xi, n_xi))
            for k in range(n_xi):
                e = jnp.zeros(n).at[self.n_up + k].set(1.0)
                col = np.asarray(JT(J(e)))
                G[:, k] = col[self.n_up :]
                if coupled:
                    C[:, k] = col
        G = (G + G.T) / 2
        self.G, self.floor, self.coupled = G, floor, coupled
        self.Cu = jnp.asarray(C[: self.n_up]) if coupled else None
        self.dbar = float(np.mean(np.diag(G)))
        self.block = block
        self.diag = jnp.concatenate([block.diag, jnp.asarray(np.diag(G))])
        self._lam = None
        lam = np.linalg.eigvalsh(G)
        if verbose:
            print(f"  xi block ({n_xi}{', coupled' if coupled else ''}): condition "
                  f"{lam[-1] / max(lam[0], 1e-300):.1e}, {time.time() - t0:.0f} s", flush=True)

    def rescale(self, s):
        """For xi / s as the unknowns (``FreeBoundaryFields.scale_xi``)."""
        self.G = self.G * s**2
        self.dbar = self.dbar * s**2
        self.diag = self.diag.at[self.n_up :].multiply(s**2)
        if self.Cu is not None:
            self.Cu = self.Cu * s
        self._lam = None
        return self

    def scale(self):
        """s that brings the mean of diag(G) to that of the (u, c_p) blocks."""
        return float(np.sqrt(np.mean(np.asarray(self.block.diag)) / np.mean(np.diag(self.G))))

    def _factor(self, lam):
        """B^-1 C_u and the eigen-decomposition of S (or of G) at this lam."""
        if self._lam == lam:
            return
        S = self.G + lam * self.dbar * np.eye(len(self.G))
        self.W = None
        if self.coupled:
            W = jax.vmap(lambda c: self.block(c, lam), in_axes=1, out_axes=1)(
                self.Cu.astype(jnp.float64))
            S = S - np.asarray(self.Cu.astype(jnp.float64).T @ W)
            self.W = W
        S = (S + S.T) / 2
        w, U = np.linalg.eigh(S)
        top = max(w[-1], 1e-300)
        # S can lose definiteness where B is not A_uu: floor its eigenvalues
        self.U, self.w = jnp.asarray(U), jnp.asarray(np.maximum(w, self.floor * top))
        self._lam = lam

    def __call__(self, r, lam=0.0):
        self._factor(float(lam))
        y = self.block(r[: self.n_up], lam)
        rx = r[self.n_up :]
        if self.W is not None:
            rx = rx - self.Cu.astype(y.dtype).T @ y
        z = self.U @ ((self.U.T @ rx) / self.w)
        if self.W is not None:
            y = y - self.W @ z
        return jnp.concatenate([y, z])
