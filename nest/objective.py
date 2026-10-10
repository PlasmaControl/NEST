"""Objectives of NEST, organized like DESC's: ``_Objective`` terms combined in an
``ObjectiveFunction``.

The unknowns x = (u, c_p) are the free coefficients of the current potential
(``BoundaryPotential``) and of the pressure (``Pressure``, smooth edge
recommended so that every c_p is admissible). ``Fields`` evaluates, for any x,
the physical fields at the collocation points of a ``ForceResidual`` (a tensor
grid in flux coordinates with volume weights dV):

    J, p, grad p    tensor evaluation of the series (``flux_interp``)
    B = B_ext + B_plasma(J)
                    B_ext from the coils (once); B_plasma from J at the box points
                    (tensor evaluation on a uniform flux grid, cached Lagrange
                    interpolation), the FFT and a free-space kernel (Hockney-
                    Eastwood or the truncated Green's function of Vico-Greengard,
                    ``kernel``), and a type-2 non-uniform FFT to the points
    F = J x B - grad p

With stellarator symmetry the collocation grid, the flux grid and the box
points cover half the domain; the rest follows from the parity of each field.

Each objective, as in DESC, has ``compute`` (its quantity at the points), a
``target``, a ``normalization`` and a ``weight``; its scaled error is

    weight * sqrt(dV / V) * (compute - target) / normalization,

so |scaled error|^2 = weight^2 <|compute - target|^2>_vol / normalization^2.

    ForceBalance   F                   target 0         normalization <|grad |B|^2 / 2 mu0|>_vol
    FieldError     B                   target B_desc    normalization rms |B_desc|
    CurrentError   J                   target J_desc    normalization rms |J_desc|
    PressureError  p                   target p_desc    normalization rms p_desc

``ObjectiveFunction`` concatenates the scaled errors (``compute_scaled_error``),
gives the scalar |f|^2 / 2 (``compute_scalar``), J v and J^T w without forming
the Jacobian (``jvp_scaled_error``, ``vjp_scaled_error``), explicit columns
(``jac_scaled_error``) and a printout (``print_value``).
"""

import functools

import jax
import jax.numpy as jnp
import numpy as np

from flux_interp import (PointInterpolator, TensorCurrent, TensorGeometry, TensorGrid,
                         TensorPressure, _Pytree)
from fourier import fft_J
from magnetic_field import b_plasma_free, biot_savart_kernel_hat
from separable import Separable

VECTOR_PARITY = (-1, 1, 1)  # (V_R, V_phi, V_Z) of a stellarator-symmetric field


def _cartesian(V, phi):
    c, s = jnp.cos(phi), jnp.sin(phi)
    return jnp.stack([V[..., 0] * c - V[..., 1] * s, V[..., 0] * s + V[..., 1] * c, V[..., 2]], axis=-1)


class Fields(_Pytree):
    """J, B, p, grad p and F at the collocation points, as a JAX function of x.

    Parameters
    ----------
    force : ForceResidual
        Collocation grid, weights, normalization, B_ext, box (located).
    current : BoundaryPotential
        Fixed structure, geometry and I; its u is the start.
    pressure : Pressure
        Fixed basis and edge order; its coefficients are the start.
    kernel : "hockney" or "truncated"
    flux_grid : TensorGrid, optional
        Uniform flux grid for J at the box points; default (4 L, 6 M, 4 N) of
        the potential's basis (half the zeta range with symmetry).
    order : int
        Lagrange order of the interpolation to the box points.
    dtype : float32 or float64
        Of J and B on the box and of the non-uniform FFT.
    chunk : int
        Points per chunk of the interpolations.
    """

    # arrays passed to the jitted functions as inputs (not compiled in as constants)
    _dynamic = ("J_col", "p_col", "phi", "sw", "B_ext", "J_flux", "to_box", "box_phi", "src",
                "K_hat", "t", "to_points")

    def __init__(self, force, current, pressure, kernel="hockney", flux_grid=None, order=4,
                 dtype=jnp.float32, chunk=65536):
        eq, box = force.inp.eq, force.box
        self.force, self.kernel, self.dtype, self.chunk = force, kernel, dtype, chunk
        self.sym = force.sym
        self.cdtype = jnp.complex64 if dtype == jnp.float32 else jnp.complex128
        # transform precision (1e-5 would give ~2e-4 with nufftax's kernel width rule)
        self.eps = 1e-6 if dtype == jnp.float32 else 1e-10
        self.I = current.I
        self.n_u, self.n_p = current.n_free, pressure.basis.num_modes
        self.x0 = jnp.asarray(np.concatenate([current.u, pressure.c]))

        # collocation points (force.tgrid, C order)
        cgrid = force.tgrid
        geo = force.geo  # the same R, Z series on the same grid
        self.J_col = TensorCurrent(current.S, current._RZ, cgrid, geo)
        P = Separable([pressure.basis], [pressure.c], pressure.basis.NFP)
        self.p_col = TensorPressure(P, pressure.edge_order, current._RZ, cgrid, geo)
        self.phi = jnp.asarray(force.phi)
        self.sw = jnp.asarray(np.sqrt(force.dV / force.dV.sum()))  # sqrt(dV / V)
        self.B_ext = jnp.asarray(force.B_ext)
        self._desc = {}  # DESC's fields at the points and their rms, on first use
        self.norm = force.norm

        # J at the box points: uniform flux grid, cached interpolation
        b = current.basis
        fg = flux_grid or TensorGrid.uniform(4 * b.L, 2 * (3 * b.M), max(4 * b.N, 8), eq.NFP,
                                             sym=self.sym)
        self.J_flux = TensorCurrent(current.S, current._RZ, fg)
        self.half = self.half_rows = None
        inside, rtz = np.asarray(box.inside), np.asarray(box.rtz)
        if self.sym and _symmetric_box(box):
            # interpolate at y > 0 (and y = 0, z >= 0) only; the mirror (x, -y, -z)
            # takes (-J_x, J_y, J_z)
            ix, iy, iz = np.unravel_index(inside, box.shape)
            keep = (box.ys[iy] > 0) | ((box.ys[iy] == 0) & (box.zs[iz] >= 0))
            inside, rtz = inside[keep], rtz[keep]
            ix, iy, iz = ix[keep], iy[keep], iz[keep]
            mirror = np.ravel_multi_index((ix, box.shape[1] - 1 - iy, box.shape[2] - 1 - iz),
                                          box.shape)
            other = mirror != inside  # points on y = z = 0 are their own mirror
            self.half, self.half_rows = mirror[other], np.flatnonzero(other)
        # in the precision of the box fields (J on the box is stored in it)
        self.to_box = PointInterpolator(fg, rtz, order, chunk=chunk, dtype=dtype)
        self.box_phi = jnp.asarray(rtz[:, 2])
        # J on the box as one gather: each box point takes a row of [J at the
        # interpolated points; their mirrors; 0] (a scatter onto a zero box would
        # make the reverse pass keep that box)
        n, n_h = len(inside), 0 if self.half is None else len(self.half)
        src = np.full(int(np.prod(box.shape)), n + n_h, np.int32)
        src[inside] = np.arange(n)
        if self.half is not None:
            src[self.half] = n + np.arange(n_h)
        self.src = jnp.asarray(src)
        self.shape, self.axes = box.shape, box.axes
        if kernel == "hockney":
            self.K_hat = biot_savart_kernel_hat(box.axes, dtype)
            self._solve = b_plasma_free
        elif kernel == "truncated":
            from magnetic_field import b_plasma_truncated, truncated_green_hat
            self.K_hat = truncated_green_hat(box.axes, dtype=dtype)
            self._solve = b_plasma_truncated
        else:
            raise ValueError(f"kernel {kernel!r}")
        x0 = np.array([a[0] for a in box.axes])
        L = np.array([len(a) * (a[1] - a[0]) for a in box.axes])
        t = 2 * np.pi * (force.x - x0) / L
        self.t = jnp.asarray((t + np.pi) % (2 * np.pi) - np.pi, dtype=dtype)
        self.to_points = SpectralInterpolator(self.t, box.shape, dtype, self.eps, chunk=4 * chunk)

    def compute(self, x):
        """The fields at x (jitted, with the caches as inputs)."""
        return _fields_compute(x, self)

    @property
    def desc(self):
        """DESC's J, B, grad p and p at the collocation points (the default targets)."""
        if not self._desc:
            d = self.force.desc_fields()
            w = self.force.dV / self.force.dV.sum()
            self._desc["fields"] = {k: jnp.asarray(v) for k, v in d.items()}
            ms = {k: np.sum(w * np.sum(np.reshape(d[k], (len(w), -1)) ** 2, 1)) for k in ("J", "B", "p")}
            self._desc["rms"] = {k: float(np.sqrt(v)) for k, v in ms.items()}
        return self._desc["fields"]

    @property
    def rms(self):
        """Volume rms of DESC's J, B and p (the default normalizations)."""
        self.desc
        return self._desc["rms"]

    @property
    def n(self):
        return self.n_u + self.n_p

    def split(self, x):
        return x[: self.n_u], x[self.n_u :]

    def _interpolate(self, B_box):
        return self.to_points(B_box)

    def J_box(self, u, I=None):
        """J on the box (Cartesian components), shape box.shape + (3,); I = 0 gives
        the part linear in u."""
        J_flux = self.J_flux(u, self.I if I is None else I).astype(self.dtype)
        J = _cartesian(self.to_box(J_flux, VECTOR_PARITY), self.box_phi.astype(self.dtype))
        if self.half is not None:
            J = jnp.concatenate([J, J[self.half_rows] * jnp.asarray([-1.0, 1.0, 1.0], self.dtype)])
        J = jnp.pad(J, ((0, 1), (0, 0)))  # the zero row
        return J[self.src].reshape(self.shape + (3,))

    def B_plasma(self, u, I=None):
        """B_plasma at the collocation points, (n, 3); I = 0 gives the part linear in u."""
        # checkpointed: the reverse pass recomputes the kernel slices of the solve
        # from K_hat (cached) instead of storing them
        solve = jax.checkpoint(lambda J_hat, K_hat: self._solve(J_hat, self.axes, K_hat),
                               policy=jax.checkpoint_policies.nothing_saveable)
        B_box = solve(fft_J(self.J_box(u, I)), self.K_hat)
        return self._interpolate(B_box)

    def _compute(self, x):
        u, c_p = self.split(x)
        J = _cartesian(self.J_col(u, self.I).reshape(-1, 3), self.phi)
        p, gp = self.p_col(c_p)
        grad_p = _cartesian(gp.reshape(-1, 3), self.phi)
        B_pl = self.B_plasma(u)
        B = self.B_ext + B_pl
        return {"J": J, "B": B, "B_plasma": B_pl, "p": p.ravel(), "grad_p": grad_p,
                "F": jnp.cross(J, B) - grad_p}


    def _tangent(self, v, J0, B0):
        """The derivative of the fields at x along v, from J0 = J(x) and B0 = B(x):
        J and B_plasma are affine in u (their linear parts: I = 0), p and grad p
        linear in c_p, so

            dJ = J_lin(v_u),  dB = B_lin(v_u),  dp = p(v_c),
            dF = dJ x B0 + J0 x dB - grad p(v_c)

        exactly. Linear in v: J v is one evaluation, J^T w its transpose."""
        u, c_p = self.split(v)
        dJ = _cartesian(self.J_col(u, 0.0).reshape(-1, 3), self.phi)
        p, gp = self.p_col(c_p)
        grad_p = _cartesian(gp.reshape(-1, 3), self.phi)
        dB = self.B_plasma(u, 0.0)
        return {"J": dJ, "B": dB, "B_plasma": dB, "p": p.ravel(), "grad_p": grad_p,
                "F": jnp.cross(dJ, B0) + jnp.cross(J0, dB) - grad_p}


_fields_compute = jax.jit(lambda x, fields: fields._compute(x))


class SpectralInterpolator(_Pytree):
    """Trigonometric interpolation of box fields at fixed points: a type-2
    non-uniform FFT (the method of ``nufftax.nufft3d2``) split so that what
    depends on the points only is computed once.

    Per evaluation: the FFT of the box field (3 components together), the
    deconvolution and zero-padding to the upsampled grid, one inverse FFT, and
    a gather of nspread^3 fine-grid values per point with the cached kernel
    weights (in chunks, ``lax.scan``). A plain JAX linear map, so its reverse
    mode and the transpose of that are JAX's own (gather <-> scatter-add).
    Only the real part is kept: B is real.

    Parameters
    ----------
    t : (n, 3) array
        The points as angles in [-pi, pi) of the box period (x, y, z).
    shape : tuple
        The box shape (Nx, Ny, Nz).
    dtype : float32 or float64
    eps : float
        Requested precision of the transform.
    upsampfac : float
    chunk : int
        Points per chunk of the gather.
    """

    _dynamic = ("idx", "w", "order")

    def __init__(self, t, shape, dtype=jnp.float32, eps=1e-6, upsampfac=1.25, chunk=16384,
                 sort=False):
        from nufftax.core.kernel import compute_kernel_params, kernel_fourier_series
        from nufftax.core.spread import compute_kernel_weights_3d, fold_rescale
        from nufftax.utils.grid import compute_grid_size
        self.shape, self.dtype, self.chunk = tuple(shape), dtype, chunk
        self.cdtype = jnp.complex64 if dtype == jnp.float32 else jnp.complex128
        kp = compute_kernel_params(eps, upsampfac)
        self.ns = kp.nspread
        self.nf = tuple(compute_grid_size(n, upsampfac, kp.nspread) for n in self.shape)
        self.phihat = tuple(np.asarray(kernel_fourier_series(nf, kp.nspread, kp.beta, kp.c,
                                                             dtype=dtype)) for nf in self.nf)
        t = jnp.asarray(t, dtype)
        self.n = t.shape[0]
        # points sorted by fine-grid cell (z, y, x): neighbours in a chunk share
        # grid values, which makes the gather and the scatter-add of the reverse
        # pass local; the output is put back in the original order
        self.order = None
        if sort:
            cells = [np.floor((np.asarray(t[:, i], float) / (2 * np.pi) + 0.5) * nf) % nf
                     for i, nf in enumerate(
                         compute_grid_size(n, upsampfac, kp.nspread) for n in self.shape)]
            order = np.lexsort((cells[0], cells[1], cells[2]))
            t = t[order]
            self.order = jnp.asarray(np.argsort(order))
        chunk = min(chunk, self.n)
        self.chunk = chunk
        m = -(-self.n // chunk) * chunk
        t = jnp.concatenate([t, jnp.zeros((m - self.n, 3), dtype)])
        nf1, nf2, nf3 = self.nf
        ix, iy, iz, wx, wy, wz = compute_kernel_weights_3d(
            fold_rescale(t[:, 0], nf1), fold_rescale(t[:, 1], nf2), fold_rescale(t[:, 2], nf3),
            nf1, nf2, nf3, kp)
        # per point: the fine-grid offsets iz nf1 nf2 + iy nf1 + ix and the weights per axis
        idx = jnp.stack([iz * (nf1 * nf2), iy * nf1, ix], axis=1).astype(jnp.int32)
        w = jnp.stack([wz, wy, wx], axis=1).astype(dtype)
        k = m // chunk
        self.idx = idx.reshape(k, chunk, 3, self.ns)
        self.w = w.reshape(k, chunk, 3, self.ns)

    def __call__(self, B_box):
        """B_box (Nx, Ny, Nz, 3) -> (n, 3) at the points, float64."""
        from nufftax.core.deconvolve import _build_deconv_factors_1d
        N = np.prod(self.shape)
        nf1, nf2, nf3 = self.nf
        F = jnp.fft.fftshift(jnp.fft.fftn(B_box.astype(self.cdtype), axes=(0, 1, 2)), axes=(0, 1, 2))
        F = jnp.transpose(F, (3, 2, 1, 0))  # (3, Nz, Ny, Nx), centred modes
        ns = self.ns
        fac = [_build_deconv_factors_1d(jnp.asarray(ph), n, 0).astype(F.dtype)
               for n, ph in zip(self.shape, self.phihat)]

        # checkpointed: the reverse pass recomputes the nspread^3 indices and weights
        # of a chunk from the cached per-axis ones instead of storing them
        # (fw as the scan's carry: JAX transposes that form, not a closed-over fw)
        @functools.partial(jax.checkpoint, policy=jax.checkpoint_policies.nothing_saveable)
        def chunk(fw, iw):
            idx, w = iw
            flat = (idx[:, 0, :, None, None] + idx[:, 1, None, :, None]
                    + idx[:, 2, None, None, :])  # (c, ns, ns, ns)
            wt = w[:, 0, :, None, None] * w[:, 1, None, :, None] * w[:, 2, None, None, :]
            return fw, jnp.sum(fw[flat] * wt, axis=(1, 2, 3))

        def component(Fc):
            # deconvolution, then zero-padding to the fine grid in FFT order: centred
            # padding and ifftshift (``deconvolve_pad_3d`` of nufftax writes into a
            # zero array instead, which the reverse pass would keep)
            for ax, (n, nf) in enumerate(zip(self.shape, self.nf)):
                axis = 2 - ax  # x is the last axis
                Fc = Fc * fac[ax].reshape((-1,) + (1,) * (2 - axis))
                left = nf // 2 - n // 2
                pad = [(0, 0)] * 3
                pad[axis] = (left, nf - n - left)
                Fc = jnp.fft.ifftshift(jnp.pad(Fc, pad), axes=axis)
            fw = (jnp.real(jnp.fft.ifftn(Fc)) * (nf1 * nf2 * nf3)).reshape(-1)
            return jax.lax.scan(chunk, fw, (self.idx, self.w))[1].reshape(-1)

        # one component at a time: a third of the fine-grid memory (F as the carry
        # of a checkpointed scan over the component index, the form JAX transposes)
        @functools.partial(jax.checkpoint, policy=jax.checkpoint_policies.nothing_saveable)
        def step(F, c):
            return F, component(jax.lax.dynamic_index_in_dim(F, c, keepdims=False))

        out = jax.lax.scan(step, F, jnp.arange(3))[1]  # (3, n padded)
        out = out.T[: self.n]
        if self.order is not None:
            out = out[self.order]
        return (out / N).astype(jnp.float64)


def _symmetric_box(box):
    ys, zs = np.asarray(box.ys), np.asarray(box.zs)
    return (len(ys) % 2 == 1 and len(zs) % 2 == 1 and np.allclose(ys, -ys[::-1])
            and np.allclose(zs, -zs[::-1]))


# ---------------------------------------------------------------- objectives
class _Objective:
    """A term of the objective: weight sqrt(dV/V) (compute - target) / normalization.

    ``target`` and ``normalization`` default to the class's choice (``build``).
    """

    name = None
    quantity = None  # key of ``Fields.compute``

    def __init__(self, weight=1.0, target=None, normalization=None):
        self.weight, self.target, self.normalization = float(weight), target, normalization

    def build(self, fields):
        if self.normalization is None:
            self.normalization = self._default_normalization(fields)
        if self.target is None:
            self.target = self._default_target(fields)
        self.target = jnp.asarray(self.target)
        n = len(fields.force.rtz)
        self.dim_f = n * (1 if self.quantity == "p" else 3)
        self._sw = fields.sw
        return self

    def compute(self, f):
        return f[self.quantity]

    def compute_scaled_error(self, f):
        return self.compute_scaled_tangent(self.compute(f) - self.target)

    def compute_scaled_tangent(self, e):
        """The scaling of the error (weight sqrt(dV/V) / normalization) applied to e:
        the derivative of the scaled error from that of the quantity."""
        sw = self._sw if e.ndim == 1 else self._sw[:, None]
        return (self.weight * sw * e / self.normalization).ravel()

    def _default_target(self, fields):
        return fields.desc[self.quantity]


class ForceBalance(_Objective):
    """F = J x B - grad p -> 0, normalized by DESC's <|grad |B|^2 / 2 mu0|>_vol."""

    name, quantity = "force", "F"

    def _default_normalization(self, fields):
        return fields.norm

    def _default_target(self, fields):
        return 0.0


class FieldError(_Objective):
    """B - B_desc, normalized by rms |B_desc|."""

    name, quantity = "B", "B"

    def _default_normalization(self, fields):
        return fields.rms["B"]


class CurrentError(_Objective):
    """J - J_desc, normalized by rms |J_desc|."""

    name, quantity = "J", "J"

    def _default_normalization(self, fields):
        return fields.rms["J"]


class PressureError(_Objective):
    """p - p_desc, normalized by rms p_desc."""

    name, quantity = "p", "p"

    def _default_normalization(self, fields):
        return fields.rms["p"]


class ObjectiveFunction:
    """Weighted objectives on shared ``Fields``, mirroring DESC's ObjectiveFunction.

    f(x) = concatenation of the objectives' scaled errors; the scalar is |f|^2 / 2.
    """

    def __init__(self, objectives, fields):
        self.objectives = tuple(objectives)
        self.fields = fields
        self.built = False

    def build(self):
        for o in self.objectives:
            o.build(self.fields)
        self.dim_f = sum(o.dim_f for o in self.objectives)
        self.built = True
        return self

    # --- DESC-like interface
    @property
    def x0(self):
        return self.fields.x0

    @property
    def dim_x(self):
        return self.fields.n

    def _impl(self, x, fields):
        f = fields._compute(x)
        return jnp.concatenate([o.compute_scaled_error(f) for o in self.objectives])

    def compute_scaled_error(self, x):
        return _scaled(self, x, self.fields)

    def compute_scalar(self, x):
        e = self.compute_scaled_error(x)
        return 0.5 * jnp.sum(e**2)

    def jvp_scaled_error(self, v, x):
        """J(x) v (``linearize``)."""
        return self.linearize(x)[1](v)

    def vjp_scaled_error(self, v, x):
        """J(x)^T v (reverse mode, ``linearize``)."""
        return self.linearize(x)[2](v)

    def linearize(self, x):
        """f(x), v -> J v and w -> J^T w at x.

        The derivative is explicit (``Fields._tangent``): it needs only J and B at
        the collocation points at x, kept here. J v evaluates the linear map, J^T w
        is its transpose (``jax.linear_transpose``: a backward pass without a
        forward one). Nothing derived from the caches is stored."""
        r, J0, B0 = _base(self, x, self.fields)
        return (r, (lambda v: _lin(self, v, J0, B0, self.fields)),
                (lambda w: _lin_T(self, w, J0, B0, self.fields)))

    def _lin_impl(self, v, J0, B0, fields):
        d = fields._tangent(v, J0, B0)
        return jnp.concatenate([o.compute_scaled_tangent(o.compute(d)) for o in self.objectives])

    def jac_scaled_error(self, x, cols=None, batch=16):
        """Explicit Jacobian columns (all by default: small problems only)."""
        from jacobian import jacobian_columns
        return jacobian_columns(self, x, np.arange(self.dim_x) if cols is None else cols, batch)

    def print_value(self, x, print_fn=print):
        r = self.report(x)
        print_fn("  ".join(f"{k} {v:.3e}" for k, v in r.items()))
        return r

    def report(self, x):
        """Per objective: |scaled error| and the rms of compute - target over its
        normalization; also DESC's force metric <|F|>_vol / norm and the scalar."""
        f = self.fields.compute(x)
        w = np.asarray(self.fields.sw) ** 2
        out = {"<|F|>/norm": float(np.sum(w * np.linalg.norm(np.asarray(f["F"]), axis=1)))
               / self.fields.norm}
        for o in self.objectives:
            e = o.compute_scaled_error(f)
            out[f"{o.name}"] = float(jnp.linalg.norm(e)) / o.weight
        out["scalar"] = float(self.compute_scalar(x))
        return out

    # --- names used by the optimizer
    @property
    def n(self):
        return self.dim_x

    def r(self, x):
        return self.compute_scaled_error(x)

    def jvp(self, x, v):
        return self.jvp_scaled_error(v, x)

    def vjp(self, x, w):
        return self.vjp_scaled_error(w, x)


# the objective function is static (its objectives' targets are small); the
# Fields caches are inputs
@functools.partial(jax.jit, static_argnums=0)
def _scaled(objfun, x, fields):
    return objfun._impl(x, fields)


@functools.partial(jax.jit, static_argnums=0)
def _base(objfun, x, fields):
    f = fields._compute(x)
    r = jnp.concatenate([o.compute_scaled_error(f) for o in objfun.objectives])
    return r, f["J"], f["B"]


@functools.partial(jax.jit, static_argnums=0)
def _lin(objfun, v, J0, B0, fields):
    return objfun._lin_impl(v, J0, B0, fields)


@functools.partial(jax.jit, static_argnums=0)
def _lin_T(objfun, w, J0, B0, fields):
    x = jax.ShapeDtypeStruct((objfun.dim_x,), w.dtype)
    return jax.linear_transpose(lambda v: objfun._lin_impl(v, J0, B0, fields), x)(w)[0]
