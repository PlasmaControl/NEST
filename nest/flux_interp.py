"""Evaluation on flux-coordinate tensor grids, and interpolation to fixed points.

The series of NEST (the current potential, the pressure, the geometry R, Z)
are sums of P_p(rho, theta) F_q(zeta) (``Separable``). On a tensor grid rho x
theta x zeta the factors P (one matrix per derivative, (n_rho n_theta) x n_p)
and F (n_zeta x n_q) depend only on the grid, so they are computed once
(``TensorFactors``) and evaluating any coefficients C[p, q, c] is two matrix
products, P C F^T, instead of rebuilding the basis at every point.

``TensorGeometry`` caches R, Z and their first derivatives (and sqrt(g)) on a
tensor grid; ``tensor_current`` and ``tensor_pressure`` give J (cylindrical
components) and p, grad p there for any coefficients.

``PointInterpolator`` brings a field on a uniform tensor grid to fixed points
given by their flux coordinates (e.g. the box points, whose (rho, theta, zeta)
never change): tensor-product Lagrange interpolation of order ``order`` per
direction (4: cubic), with indices and weights computed once and applied as a
gather and a weighted sum. In theta and zeta the grid is periodic (one field
period in zeta: cylindrical components are periodic there); in rho the
stencils cross the axis by the reflection (-rho, theta) = (rho, theta + pi),
under which any smooth field in (x, y) is continuous, so no stencil is
one-sided at the axis. Uniform rho nodes are cell-centred, rho_i = (i + 1/2) d
with d = 1 / (n_rho - 1/2), the last one at rho = 1, and no node on the axis.
"""

import jax
import jax.numpy as jnp
import numpy as np

from batching import map_chunks
from boundary_potential import _h, _window
from desc.basis import fourier


class _Static:
    """Static (non-array) attributes of a pytree node, hashed by identity."""

    def __init__(self, d):
        self.d = d


class _Pytree:
    """Register a class of cached arrays as a JAX pytree.

    The attributes named in ``_dynamic`` (arrays, or pytrees of them) are the
    leaves; every other attribute is static. Passing such an object as an
    argument of a jitted function keeps its arrays as inputs on the device,
    instead of embedding them as constants in the compiled program (which costs
    host memory and compile time for large caches).
    """

    _dynamic = ()

    def __init_subclass__(cls, **kw):
        super().__init_subclass__(**kw)
        jax.tree_util.register_pytree_node(cls, cls._flatten, cls._unflatten)

    def _flatten(self):
        if "_aux" not in vars(self):
            stat = {k: v for k, v in vars(self).items() if k not in self._dynamic}
            object.__setattr__(self, "_aux", _Static(stat))
        return tuple(getattr(self, k, None) for k in self._dynamic), self._aux

    @classmethod
    def _unflatten(cls, aux, children):
        obj = object.__new__(cls)
        for k, v in aux.d.items():
            object.__setattr__(obj, k, v)
        object.__setattr__(obj, "_aux", aux)
        for k, v in zip(cls._dynamic, children):
            object.__setattr__(obj, k, v)
        return obj


class TensorGrid:
    """rho x theta x zeta (one field period); fields have shape (n_rho, n_theta, n_zeta, ...)."""

    def __init__(self, rho, theta, zeta, NFP, sym=False):
        self.rho, self.theta, self.zeta = (np.asarray(a, dtype=float) for a in (rho, theta, zeta))
        self.NFP, self.sym = int(NFP), bool(sym)

    @classmethod
    def uniform(cls, n_rho, n_theta, n_zeta, NFP, sym=False):
        """Cell-centred rho (last node at 1), uniform periodic theta, and zeta
        uniform over one field period (n_zeta points) or, with stellarator
        symmetry, over half of it including both ends (n_zeta + 1 points from 0
        to pi / NFP)."""
        assert n_theta % 2 == 0, "n_theta must be even (axis reflection theta + pi)"
        d = 1 / (n_rho - 0.5)
        if sym:
            zeta = np.pi * np.arange(n_zeta + 1) / (n_zeta * NFP)
        else:
            zeta = 2 * np.pi * np.arange(n_zeta) / (n_zeta * NFP)
        return cls((np.arange(n_rho) + 0.5) * d, 2 * np.pi * np.arange(n_theta) / n_theta,
                   zeta, NFP, sym)

    @property
    def shape(self):
        return (len(self.rho), len(self.theta), len(self.zeta))

    @property
    def size(self):
        return int(np.prod(self.shape))

    def nodes(self):
        """(rho, theta, zeta) of all nodes, in the grid's (C) order, (size, 3)."""
        R, T, Z = np.meshgrid(self.rho, self.theta, self.zeta, indexing="ij")
        return np.column_stack([R.ravel(), T.ravel(), Z.ravel()])


class TensorFactors(_Pytree):
    """Cached factors of a separable series on a ``TensorGrid``.

    ``poloidal(rho, theta, dr, dt)`` -> (n, n_p) and ``toroidal(zeta, dz)`` ->
    (n, n_q), as ``Separable.poloidal`` and ``Separable.toroidal``. ``pol`` and
    ``tor`` list the derivatives to cache.
    """

    _dynamic = ("P", "F")

    def __init__(self, poloidal, toroidal, grid, pol=((0, 0),), tor=(0,)):
        self.grid = grid
        R, T = np.meshgrid(grid.rho, grid.theta, indexing="ij")
        R, T = jnp.asarray(R.ravel()), jnp.asarray(T.ravel())
        self.P = {d: jnp.asarray(poloidal(R, T, *d)) for d in pol}
        self.F = {d: jnp.asarray(toroidal(jnp.asarray(grid.zeta), d)) for d in tor}

    def __call__(self, C, dr=0, dt=0, dz=0):
        """sum_pq C[p, q, c] P_p F_q (or a derivative), shape grid.shape + (n_c,)."""
        n_r, n_t, n_z = self.grid.shape
        PC = jnp.einsum("ip,pqc->iqc", self.P[(dr, dt)], C)
        return jnp.einsum("iqc,kq->ikc", PC, self.F[dz]).reshape(n_r, n_t, n_z, -1)


class FactoredFactors(_Pytree):
    """``TensorFactors`` of a ``Separable`` without the (n_rho n_theta) x n_p tables.

    On a tensor grid the poloidal functions P_p = Z_l^|m|(rho) T_m(theta) are products
    themselves, so sum_p C[p, q, c] P_p is a radial contraction over l (per m) followed
    by an angular one over m, with tables (n_rho, n_m, n_l) and (n_theta, n_m) only:
    O(n_rho n_m n_l + n_rho n_theta n_m) memory instead of n_rho n_theta n_p (at L = M = 96
    with sin and cos in theta, 50 GB on a 384 x 576 grid against ~60 MB). Same values
    and call as ``TensorFactors`` (up to rounding).
    """

    _dynamic = ("Z", "T", "F", "mi", "li")

    def __init__(self, sep, grid, pol=((0, 0), (1, 0), (0, 1)), tor=(0, 1)):
        from desc.basis import zernike_radial
        self.grid = grid
        lm = np.asarray(sep.lm)
        ms, mi = np.unique(lm[:, 1], return_inverse=True)
        ls, li = np.unique(lm[:, 0], return_inverse=True)
        self.n_m, self.n_l = len(ms), len(ls)
        self.mi, self.li = jnp.asarray(mi.ravel()), jnp.asarray(li.ravel())
        rho = jnp.asarray(grid.rho)
        self.Z, self.T = {}, {}
        for dr in sorted({d[0] for d in pol}):
            z = zernike_radial(rho[:, None], lm[:, 0], lm[:, 1], dr=dr)  # (n_rho, n_p)
            self.Z[dr] = jnp.zeros((len(grid.rho), self.n_m, self.n_l)).at[:, self.mi, self.li].set(z)
        for dt in sorted({d[1] for d in pol}):
            self.T[dt] = jnp.asarray(fourier(jnp.asarray(grid.theta)[:, None], ms, dt=dt))
        self.F = {d: jnp.asarray(sep.toroidal(jnp.asarray(grid.zeta), d)) for d in tor}

    def __call__(self, C, dr=0, dt=0, dz=0):
        """sum_pq C[p, q, c] P_p F_q (or a derivative), shape grid.shape + (n_c,)."""
        Cg = jnp.zeros((self.n_m, self.n_l) + C.shape[1:], C.dtype).at[self.mi, self.li].set(C)
        A = jnp.einsum("iml,mlqc->imqc", self.Z[dr], Cg)
        PC = jnp.einsum("jm,imqc->ijqc", self.T[dt], A)
        return jnp.einsum("ijqc,kq->ijkc", PC, self.F[dz])


def separable_factors(sep, grid, pol=((0, 0), (1, 0), (0, 1)), tor=(0, 1), factored=False):
    if factored:
        return FactoredFactors(sep, grid, pol, tor)
    return TensorFactors(sep.poloidal, sep.toroidal, grid, pol, tor)


class TensorGeometry(_Pytree):
    """R, Z, their first derivatives and sqrt(g) on a tensor grid (cached)."""

    _dynamic = ("R", "Xr", "Xt", "Xz", "sqrt_g")

    def __init__(self, RZ, grid):
        f = separable_factors(RZ, grid)
        C = RZ.C
        X, Xr, Xt, Xz = f(C), f(C, dr=1), f(C, dt=1), f(C, dz=1)
        self.R, self.Z, self.Xr, self.Xt, self.Xz = X[..., 0], X[..., 1], Xr, Xt, Xz
        # sqrt(g) = e_rho . e_theta x e_zeta, e_rho = (R_r, 0, Z_r), e_theta = (R_t, 0, Z_t),
        # e_zeta = (R_z, R, Z_z) in (R, phi, Z) (phi = zeta)
        self.sqrt_g = self.R * (Xt[..., 0] * Xr[..., 1] - Xr[..., 0] * Xt[..., 1])

    def vector(self, a, b, c):
        """(a e_rho + b e_theta + c e_zeta) / sqrt(g) in (R, phi, Z) components."""
        Xr, Xt, Xz = self.Xr, self.Xt, self.Xz
        V = jnp.stack([a * Xr[..., 0] + b * Xt[..., 0] + c * Xz[..., 0], c * self.R,
                       a * Xr[..., 1] + b * Xt[..., 1] + c * Xz[..., 1]], axis=-1)
        return V / self.sqrt_g[..., None]

    def gradient(self, fr, ft, fz):
        """grad f from d_rho f, d_theta f, d_zeta f, in (R, phi, Z) components."""
        R, (Rr, Zr), (Rt, Zt), (Rz, Zz) = (self.R, self.Xr.transpose(3, 0, 1, 2),
                                           self.Xt.transpose(3, 0, 1, 2), self.Xz.transpose(3, 0, 1, 2))
        g = jnp.stack([-fr * R * Zt + ft * R * Zr,
                       fr * (Zt * Rz - Rt * Zz) + ft * (Zz * Rr - Rz * Zr) + fz * (Zr * Rt - Rr * Zt),
                       fr * R * Rt - ft * R * Rr], axis=-1)
        return g / self.sqrt_g[..., None]


class TensorCurrent(_Pytree):
    """J = curl T of a ``BoundaryPotential`` structure on a tensor grid (rho > 0)."""

    _dynamic = ("geo", "ab", "chi", "w", "h0r", "rho2")

    def __init__(self, S, RZ, grid, geometry=None, factored=False):
        self.S, self.grid = S, grid
        self.geo = geometry or TensorGeometry(RZ, grid)
        self.ab = separable_factors(S.ab, grid, factored=factored)

        def chi_pol(R, T, dr, dt):
            return _h(S, R, dr)[:, S.chi_abs] * fourier(T[:, None], S.chi_m, dt=dt)

        def chi_tor(z, dz):
            return fourier(z[:, None], S.chi_n, NFP=S.NFP, dt=dz)

        self.chi = TensorFactors(chi_pol, chi_tor, grid, pol=((1, 0), (1, 1)), tor=(0, 1))
        rho = jnp.asarray(grid.rho)
        self.w = [v[:, None, None] for v in _window(S, rho)]
        self.h0r = _h(S, rho, 1)[:, 0][:, None, None]
        self.rho2 = (rho**2)[:, None, None]

    def __call__(self, u, I):
        """J in (R, phi, Z) components, shape grid.shape + (3,); linear in (u, I)."""
        # unbound: precondition.py calls this on a stand-in (``_View``) with the same fields
        return self.geo.vector(*TensorCurrent.contravariant(self, u, I))

    def contravariant(self, u, I):
        """(sqrt(g) J^rho, sqrt(g) J^theta, sqrt(g) J^zeta), each of shape grid.shape."""
        Cchi, Cab = self.S.split(u)
        f, f1, _, g, g1, _ = self.w
        AB, ABr, ABt, ABz = self.ab(Cab), self.ab(Cab, dr=1), self.ab(Cab, dt=1), self.ab(Cab, dz=1)
        Cc = Cchi[:, :, None]
        chi_rz, chi_rt = self.chi(Cc, dr=1, dz=1)[..., 0], self.chi(Cc, dr=1, dt=1)[..., 0]
        a = f * (ABt[..., 1] - self.rho2 * ABz[..., 0])
        b = -(chi_rz + f1 * AB[..., 1] + f * ABr[..., 1])
        c = I / (2 * np.pi) * self.h0r + chi_rt + g1 * AB[..., 0] + g * ABr[..., 0]
        return a, b, c


class TensorPressure(_Pytree):
    """p and grad p of a ``Pressure`` (``P``: its Separable, ``k``: edge order) on a tensor grid."""

    _dynamic = ("geo", "f", "wk", "wkr")

    def __init__(self, P, k, RZ, grid, geometry=None, factored=False):
        self.P, self.k, self.grid = P, k, grid
        self.wk = self.wkr = None
        self.geo = geometry or TensorGeometry(RZ, grid)
        self.f = separable_factors(P, grid, factored=factored)
        if k is not None:
            rho = grid.rho[:, None, None]
            self.wk = jnp.asarray((1 - rho**2) ** (k + 1))
            self.wkr = jnp.asarray(-2 * (k + 1) * rho * (1 - rho**2) ** k)

    def __call__(self, c_p):
        """(p, grad p in (R, phi, Z)), shapes grid.shape and grid.shape + (3,)."""
        C = self.P.scatter([c_p])
        p, pr, pt, pz = (self.f(C, *d)[..., 0] for d in ((0, 0, 0), (1, 0, 0), (0, 1, 0), (0, 0, 1)))
        if self.k is not None:  # p = w Q
            p, pr, pt, pz = self.wk * p, self.wkr * p + self.wk * pr, self.wk * pt, self.wk * pz
        return p, self.geo.gradient(pr, pt, pz)


def _lagrange(s, order):
    """Stencil start i0 (n,) and weights (n, order) for node coordinates s (in units
    of the spacing): nodes i0 .. i0 + order - 1, centred on s."""
    i0 = np.floor(s).astype(int) - (order // 2 - 1)
    x = s[:, None] - (i0[:, None] + np.arange(order))
    w = np.ones((len(s), order))
    for j in range(order):
        for k in range(order):
            if k != j:
                w[:, j] *= x[:, k] / (j - k)
    return i0, w


class PointInterpolator(_Pytree):
    """Cached tensor-product Lagrange interpolation from a uniform ``TensorGrid``
    to fixed points (rho, theta, zeta).

    With a stellarator-symmetric grid (zeta over half a field period), a point
    with zeta in the other half is mapped by (theta, zeta) -> (-theta, -zeta),
    and stencil nodes beyond zeta = 0 or pi / NFP are reflected the same way;
    the reflected values carry the parity of each component (``parity`` in
    ``__call__``: -1 for V_R of a symmetric vector field, +1 for V_phi, V_Z and
    for scalars such as p)."""

    _dynamic = ("idx", "w", "sign", "s", "sign_pt")

    def __init__(self, grid, rtz, order=4, chunk=65536, dtype=jnp.float64, compact=False):
        """``compact``: store only the stencil coordinates (s_rho, s_theta, s_zeta) and the
        point's symmetry sign (25 bytes per point, against 64 indices, weights and signs,
        1280 bytes in float64) and form the stencil per chunk inside the apply, with the
        same arithmetic in the same order (bitwise the cached result)."""
        n_r, n_t, n_z = grid.shape
        rtz = np.array(rtz, dtype=float)
        d_r = grid.rho[1] - grid.rho[0]
        d_t = 2 * np.pi / n_t
        sign_pt = np.ones(len(rtz))
        if grid.sym:
            period = 2 * np.pi / grid.NFP
            z = np.mod(rtz[:, 2], period)
            other = z > period / 2  # map to the covered half
            rtz[:, 1] = np.where(other, -rtz[:, 1], rtz[:, 1])
            rtz[:, 2] = np.where(other, period - z, z)
            sign_pt = np.where(other, -1.0, 1.0)
            d_z = (period / 2) / (n_z - 1)
        else:
            d_z = 2 * np.pi / (n_z * grid.NFP)
        self.sym, self.chunk, self.order, self.grid_shape = grid.sym, chunk, int(order), (n_r, n_t, n_z)
        s_r = rtz[:, 0] / d_r - 0.5
        s_t = np.mod(rtz[:, 1], 2 * np.pi) / d_t
        s_z = (rtz[:, 2] if grid.sym else np.mod(rtz[:, 2], 2 * np.pi / grid.NFP)) / d_z
        self.compact = bool(compact)
        if self.compact:  # (attributes of the cached form are None)
            self.idx = self.w = self.sign = None
            self.s = jnp.asarray(np.stack([s_r, s_t, s_z], axis=1), jnp.float64)
            self.sign_pt = jnp.asarray(sign_pt.astype(np.int8))
            self.dtype = jnp.dtype(dtype)
            return
        self.s = self.sign_pt = None
        # rho: s with rho_i = (i + 1/2) d; the top of the stencil stays below n_r
        ir, wr = _lagrange(s_r, order)
        shift = np.maximum(ir + order - n_r, 0)
        ir -= shift
        if shift.any():  # recompute the weights of the shifted (one-sided) stencils
            x = rtz[:, 0:1] / d_r - 0.5 - (ir[:, None] + np.arange(order))
            wr = np.ones_like(x)
            for j in range(order):
                for k in range(order):
                    if k != j:
                        wr[:, j] *= x[:, k] / (j - k)
        it, wt = _lagrange(s_t, order)
        iz, wz = _lagrange(s_z, order)
        R = ir[:, None] + np.arange(order)  # (n, order); < 0: reflected through the axis
        T = it[:, None] + np.arange(order)
        Zi = iz[:, None] + np.arange(order)
        o = order
        # all combinations (n, r, t, z)
        Rb = np.broadcast_to(R[:, :, None, None], (len(rtz), o, o, o))
        Tb = np.broadcast_to(T[:, None, :, None], (len(rtz), o, o, o))
        Zb = np.broadcast_to(Zi[:, None, None, :], (len(rtz), o, o, o))
        sign = np.ones((len(rtz), o, o, o))
        if grid.sym:  # zeta nodes beyond 0 or pi / NFP: (theta, zeta) -> (-theta, -zeta)
            top = n_z - 1
            out = (Zb < 0) | (Zb > top)
            Zb = np.where(Zb < 0, -Zb, np.where(Zb > top, 2 * top - Zb, Zb))
            Tb = np.where(out, -Tb, Tb)
            sign = np.where(out, -1.0, 1.0)
        else:
            Zb = np.mod(Zb, n_z)
        refl = Rb < 0  # through the axis: (-rho, theta) = (rho, theta + pi), no parity
        Rb = np.where(refl, -Rb - 1, Rb)
        Tb = np.mod(Tb + np.where(refl, n_t // 2, 0), n_t)
        idx = (Rb * n_t + Tb) * n_z + Zb
        w = wr[:, :, None, None] * wt[:, None, :, None] * wz[:, None, None, :]
        self.idx = jnp.asarray(idx.reshape(len(rtz), -1).astype(np.int32))
        self.w = jnp.asarray(w.reshape(len(rtz), -1), dtype)
        self.sign = jnp.asarray((sign * sign_pt[:, None, None, None]).reshape(len(rtz), -1), dtype)

    def _stencil(self, s, sign_pt):
        """(idx, w, sign) of a chunk of points from their stencil coordinates: the
        construction above in JAX, operation for operation."""
        n_r, n_t, n_z = self.grid_shape
        o = self.order
        k = jnp.arange(o)

        def lag(x0, i0):  # weights of nodes i0 .. i0 + o - 1 at x0, as _lagrange's loop
            x = x0[:, None] - (i0[:, None] + k).astype(x0.dtype)
            cols = []
            for j in range(o):
                wj = jnp.ones_like(x0)
                for q in range(o):
                    if q != j:
                        wj = wj * (x[:, q] / (j - q))
                cols.append(wj)
            return jnp.stack(cols, axis=1)

        def start(x0):
            return jnp.floor(x0).astype(jnp.int32) - (o // 2 - 1)

        ir = start(s[:, 0])
        ir = ir - jnp.maximum(ir + o - n_r, 0)  # one-sided at the top of rho
        wr = lag(s[:, 0], ir)
        it, iz = start(s[:, 1]), start(s[:, 2])
        wt, wz = lag(s[:, 1], it), lag(s[:, 2], iz)
        n = s.shape[0]
        Rb = jnp.broadcast_to((ir[:, None] + k)[:, :, None, None], (n, o, o, o))
        Tb = jnp.broadcast_to((it[:, None] + k)[:, None, :, None], (n, o, o, o))
        Zb = jnp.broadcast_to((iz[:, None] + k)[:, None, None, :], (n, o, o, o))
        sign = jnp.ones((n, o, o, o), self.dtype)
        if self.sym:
            top = n_z - 1
            out = (Zb < 0) | (Zb > top)
            Zb = jnp.where(Zb < 0, -Zb, jnp.where(Zb > top, 2 * top - Zb, Zb))
            Tb = jnp.where(out, -Tb, Tb)
            sign = jnp.where(out, -1.0, 1.0).astype(self.dtype)
        else:
            Zb = jnp.mod(Zb, n_z)
        refl = Rb < 0
        Rb = jnp.where(refl, -Rb - 1, Rb)
        Tb = jnp.mod(Tb + jnp.where(refl, n_t // 2, 0), n_t)
        idx = ((Rb * n_t + Tb) * n_z + Zb).reshape(n, -1)
        w = (wr[:, :, None, None] * wt[:, None, :, None] * wz[:, None, None, :]).reshape(n, -1)
        sign = (sign * sign_pt.astype(self.dtype)[:, None, None, None]).reshape(n, -1)
        return idx, w.astype(self.dtype), sign

    @property
    def n_points(self):
        return int((self.s if self.compact else self.idx).shape[0])

    @property
    def nbytes(self):
        """Bytes held per the interpolator (the stencil cache, or its compact form)."""
        return sum(int(a.size * a.dtype.itemsize) for a in (self.idx, self.w, self.sign, self.s, self.sign_pt)
                   if a is not None)

    def __call__(self, field, parity=None):
        """field (n_rho, n_theta, n_zeta, c) on the grid -> (n_points, c) at the points.

        ``parity``: per component, +1 or -1 under stellarator symmetry (only used
        with a symmetric grid; default all +1).
        """
        flat = field.reshape(-1, field.shape[-1])
        if self.compact:
            use_sign = self.sym and parity is not None and not all(p > 0 for p in parity)
            par = jnp.asarray(parity if use_sign else [1] * flat.shape[-1], self.dtype)

            def fc(fl, s, sp):
                i, w, sg = self._stencil(s, sp)
                if not use_sign:
                    return jnp.einsum("nk,nkc->nc", w, fl[i])
                ws = w[:, :, None] * jnp.where(par < 0, sg[:, :, None], 1.0)
                return jnp.einsum("nkc,nkc->nc", ws, fl[i])
            return map_chunks(fc, self.s, self.sign_pt, chunk=self.chunk, carry=flat)
        if not self.sym or parity is None or all(p > 0 for p in parity):
            return map_chunks(lambda fl, i, w: jnp.einsum("nk,nkc->nc", w, fl[i]),
                              self.idx, self.w, chunk=self.chunk, carry=flat)
        par = jnp.asarray(parity, self.w.dtype)  # (c,): components with -1 take the sign
        def f(fl, i, w, s):
            ws = w[:, :, None] * jnp.where(par < 0, s[:, :, None], 1.0)
            return jnp.einsum("nkc,nkc->nc", ws, fl[i])
        return map_chunks(f, self.idx, self.w, self.sign, chunk=self.chunk, carry=flat)


def _lagrange_jnp(x, order, deriv=False):
    """Lagrange weights (n, order) on nodes 0..order-1 from the offsets x[:, k] = s - k
    (in node spacings), and with ``deriv`` their derivatives in s (JAX)."""
    w = []
    for j in range(order):
        t = jnp.ones_like(x[:, 0])
        for k in range(order):
            if k != j:
                t = t * x[:, k] / (j - k)
        w.append(t)
    w = jnp.stack(w, axis=1)
    if not deriv:
        return w
    dw = []
    for j in range(order):
        s = jnp.zeros_like(x[:, 0])
        for l in range(order):
            if l == j:
                continue
            t = jnp.full_like(x[:, 0], 1.0 / (j - l))
            for k in range(order):
                if k not in (j, l):
                    t = t * x[:, k] / (j - k)
            s = s + t
        dw.append(s)
    return w, jnp.stack(dw, axis=1)


class StencilInterpolator(_Pytree):
    """``PointInterpolator`` with the stencils formed on the fly from the points'
    coordinates: the cache is the points (rho, theta, zeta) in float64, 24 bytes per
    point against 64 indices, weights and signs (~770 bytes at order 4), at the
    cost of forming the weights (a few hundred flops per point) at every call. The
    same nodes and weights as ``PointInterpolator`` (stellarator reflection,
    reflection through the axis, one-sided stencils at the top in rho).

    ``__call__(field, parity, derivs=True)`` also returns the derivatives of the
    interpolant in rho and in the points' own theta (as ``free_boundary.BoxShift``).
    """

    _dynamic = ("rtz",)

    def __init__(self, grid, rtz, order=4, chunk=65536, dtype=jnp.float64):
        n_r, n_t, n_z = grid.shape
        rho = np.asarray(grid.rho)
        self.d_r = float(rho[1] - rho[0])
        if abs(rho[0] - self.d_r / 2) > 1e-12 * self.d_r:
            raise ValueError("StencilInterpolator needs rho_i = (i + 1/2) d (TensorGrid.uniform)")
        self.d_t = 2 * np.pi / n_t
        self.period = 2 * np.pi / grid.NFP
        self.d_z = (self.period / 2) / (n_z - 1) if grid.sym else self.period / n_z
        self.shape, self.sym, self.order = (n_r, n_t, n_z), bool(grid.sym), int(order)
        self.chunk, self.dtype = int(chunk), jnp.dtype(dtype)
        self.rtz = jnp.asarray(np.asarray(rtz, dtype=float))
        self.n = int(self.rtz.shape[0])

    def stencil(self, rtz, derivs=False):
        """Flat node indices (n, o^3), weights and parity signs (and the rho and theta
        derivative weights) of the points rtz (n, 3), float64."""
        n_r, n_t, n_z = self.shape
        o = self.order
        r, t, z = rtz[:, 0], rtz[:, 1], rtz[:, 2]
        sign_pt = jnp.ones_like(r)
        if self.sym:
            zm = jnp.mod(z, self.period)
            other = zm > self.period / 2
            t = jnp.where(other, -t, t)
            z = jnp.where(other, self.period - zm, zm)
            sign_pt = jnp.where(other, -1.0, 1.0)
        else:
            z = jnp.mod(z, self.period)
        k = jnp.arange(o)
        sr = r / self.d_r - 0.5
        ir = jnp.floor(sr).astype(jnp.int32) - (o // 2 - 1)
        ir = ir - jnp.maximum(ir + o - n_r, 0)
        st = jnp.mod(t, 2 * np.pi) / self.d_t
        it = jnp.floor(st).astype(jnp.int32) - (o // 2 - 1)
        sz = z / self.d_z
        iz = jnp.floor(sz).astype(jnp.int32) - (o // 2 - 1)
        xr = sr[:, None] - (ir[:, None] + k)
        xt = st[:, None] - (it[:, None] + k)
        xz = sz[:, None] - (iz[:, None] + k)
        if derivs:
            wr, dwr = _lagrange_jnp(xr, o, True)
            wt, dwt = _lagrange_jnp(xt, o, True)
        else:
            wr, wt = _lagrange_jnp(xr, o), _lagrange_jnp(xt, o)
        wz = _lagrange_jnp(xz, o)
        R = (ir[:, None] + k)[:, :, None, None]
        T = (it[:, None] + k)[:, None, :, None]
        Z = (iz[:, None] + k)[:, None, None, :]
        R, T, Z = jnp.broadcast_arrays(R, T, Z)
        sign = jnp.ones(R.shape, r.dtype)
        if self.sym:
            top = n_z - 1
            out = (Z < 0) | (Z > top)
            Z = jnp.where(Z < 0, -Z, jnp.where(Z > top, 2 * top - Z, Z))
            T = jnp.where(out, -T, T)
            sign = jnp.where(out, -1.0, 1.0)
        else:
            Z = jnp.mod(Z, n_z)
        refl = R < 0
        R = jnp.where(refl, -R - 1, R)
        T = jnp.mod(T + jnp.where(refl, n_t // 2, 0), n_t)
        n = r.shape[0]
        idx = ((R * n_t + T) * n_z + Z).reshape(n, -1)
        prod = lambda a, b, c: (a[:, :, None, None] * b[:, None, :, None]  # noqa: E731
                                * c[:, None, None, :]).reshape(n, -1)
        sign = (sign * sign_pt[:, None, None, None]).reshape(n, -1)
        if not derivs:
            return idx, prod(wr, wt, wz), sign
        # d / d theta of the points' own theta: minus that of the mapped one
        return (idx, prod(wr, wt, wz), sign, prod(dwr / self.d_r, wt, wz),
                prod(wr, dwt / self.d_t, wz) * sign_pt[:, None])

    def __call__(self, field, parity=None, derivs=False):
        """field (n_rho, n_theta, n_zeta, c) on the grid -> (n_points, c) at the points;
        with ``derivs``, (value, d/d rho, d/d theta), each (n_points, c)."""
        c = field.shape[-1]
        flat = field.reshape(-1, c)
        if parity is None or not self.sym:
            parity = (1.0,) * c
        neg = jnp.asarray([p < 0 for p in parity])
        dt = self.dtype

        def f(fl, rtz):
            if derivs:
                i, w, s, wr, wt = self.stencil(rtz, True)
            else:
                i, w, s = self.stencil(rtz)
            g = fl[i]  # (n, o^3, c)
            g = g * jnp.where(neg[None, None, :], s[:, :, None], 1.0).astype(g.dtype)
            if not derivs:
                return jnp.einsum("nk,nkc->nc", w.astype(dt), g)
            return tuple(jnp.einsum("nk,nkc->nc", v.astype(dt), g) for v in (w, wr, wt))

        return map_chunks(f, self.rtz, chunk=self.chunk, carry=flat)
