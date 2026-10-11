"""Step 5 of NEST: the plasma's magnetic field B_plasma from J on the box G3.

Three ways to invert curl B = mu0 J, div B = 0 with the FFT of J (``fourier.fft_J``):

periodic        B_hat = i mu0 k x J_hat / |k|^2.
                Spectral, but it is the field of the infinite lattice of copies of
                the box, with zero mean: the neighbouring plasmas add an error that
                decays with the padding (roughly (plasma size / period)^3).

free space, two kernels (``b_hat_free``, ``b_plasma_free``):

vico_greengard  (default) B_hat = i G(k) k x J_hat, G the Fourier transform of the
                truncated Green's function mu0 / (4 pi r), r < R_cut (Vico, Greengard
                and Ferrando, J. Comput. Phys. 323, 2016): G_R(k) = mu0 (1 - cos k
                R_cut) / k^2 sampled on a finer precompute grid, transformed to real
                space, cut to the minimum-image displacements of the box and
                transformed back (``vico_greengard_kernel_hat``). B is the continuous
                free-space Biot-Savart integral of the sampled J, spectrally
                accurate; the error left is that of sampling J (its smoothness at
                the boundary).

hockney         Hockney-Eastwood: the discrete Biot-Savart sum
                    B(x) = mu0 / 4 pi sum_x' J(x') x (x - x') / |x - x'|^3 h^3
                over the grid points x' (the self term x' = x is zero), done as a
                circular convolution with the kernel K(d) = d / (4 pi |d|^3) at the
                minimum-image displacements d (``biot_savart_kernel_hat``); second
                order in h.

Both free-space kernels give the field with no images wherever every source-target
separation is its own minimum image (at most half a period in each direction). With a
period at least twice the extent of the current (factor 2 in ``cartesian_grid``) that
holds at least in the plasma's bounding box; beyond it, only within period / 2 -
extent of it (about zero in z), and further away the images come back. R_cut defaults
to the diagonal of the half-period box, which bounds every such separation. On an even
axis, Hockney's kernel is zero at the ambiguous displacement +-N h / 2, and G is zero
on the Nyquist plane of k (the first derivative i k there is not real), so the
Vico-Greengard B_hat has no Nyquist content.

All use the unnormalized rfft of ``fourier.fft_J`` (k_z >= 0 only). ``b_plasma_*``
return B on the box, shape (Nx, Ny, Nz, 3), in Cartesian components [T];
``interpolate_hat`` evaluates the box's Fourier series at any points straight from
the half spectrum (type-2 non-uniform FFT); ``BiotSavart`` chains these into one
linear, differentiable map from J at the inside points to B at fixed targets.
"""

import functools
import hashlib
import os

import jax
import jax.numpy as jnp
import numpy as np
from nufftax import nufft3d2
from scipy.constants import mu_0

from fourier import fft_J, wavevectors
from grid import scatter

KERNELS = ("vico_greengard", "hockney")


def _not_nyquist(n, half=False):
    """1 except at the Nyquist index of an even-length FFT axis."""
    m = np.ones(n // 2 + 1 if half else n)
    if n % 2 == 0:
        m[n // 2] = 0
    return m


def _broadcast(v):
    """Three 1-D arrays as (n, 1, 1), (1, n, 1), (1, 1, n)."""
    return [v[0][:, None, None], v[1][None, :, None], v[2][None, None, :]]


def b_plasma_periodic(J_hat, axes):
    """B_hat = i mu0 k x J_hat / |k|^2, back to real space.

    The k = 0 mode (the mean of B over the box) is zero. The Nyquist planes of
    even axes are dropped: a first derivative there is not real.
    """
    shape = tuple(len(a) for a in axes)
    dt = J_hat.real.dtype
    k = _broadcast([jnp.asarray(ki, dt) for ki in wavevectors(axes)])
    k2 = k[0] ** 2 + k[1] ** 2 + k[2] ** 2
    nyq = [jnp.asarray(_not_nyquist(n, i == 2), dt) for i, n in enumerate(shape)]
    nyq = _broadcast(nyq)
    g = jnp.where(k2 > 0, mu_0 / jnp.where(k2 > 0, k2, 1), 0) * nyq[0] * nyq[1] * nyq[2]

    def component(i):
        j, l = (i + 1) % 3, (i + 2) % 3
        B_hat = 1j * g * (k[j] * J_hat[..., l] - k[l] * J_hat[..., j])
        return jnp.fft.irfftn(B_hat, s=shape)

    return jnp.stack([component(i) for i in range(3)], axis=-1)


def biot_savart_kernel_hat(axes, dtype=jnp.float32):
    """rfft of h^3 K(d), K(d) = d / (4 pi |d|^3), at the minimum-image displacements.

    K(0) = 0 (the self term of the odd kernel), and K = 0 on the Nyquist plane
    of an even axis, where the displacement +-N h / 2 is ambiguous; with the
    factor-2 padding no pair of current-carrying points is that far apart.
    Shape (Nx, Ny, Nz // 2 + 1, 3).
    """
    shape = tuple(len(a) for a in axes)
    h = [a[1] - a[0] for a in axes]
    d = _broadcast([jnp.asarray(hi * np.fft.fftfreq(n) * n, dtype) for hi, n in zip(h, shape)])
    nyq = _broadcast([jnp.asarray(_not_nyquist(n), dtype) for n in shape])
    r2 = d[0] ** 2 + d[1] ** 2 + d[2] ** 2
    w = jnp.where(r2 > 0, 1 / (4 * np.pi * jnp.where(r2 > 0, r2, 1) ** 1.5), 0)
    w = w * nyq[0] * nyq[1] * nyq[2] * float(np.prod(h))  # Python float: keeps dtype
    return jnp.stack([jnp.fft.rfftn(w * d[i]) for i in range(3)], axis=-1)


def half_box_diagonal(axes):
    """Diagonal of the half-period box [m]: the default R_cut of the Vico-Greengard kernel."""
    return float(np.linalg.norm([len(a) * (a[1] - a[0]) / 2 for a in axes]))


def precompute_shape(axes, R_cut, factor=4):
    """Points per axis of the Vico-Greengard precompute grid (spacing h of the box).

    The precompute period must hold the truncated kernel's support plus the source
    window and resolve the oscillation of (1 - cos k R) / k^2 in k (Vico et al. 2016).
    With the source window half the box period L_i / 2:
    P_i = max(factor L_i / 2, (factor / 2) R_cut), rounded up to an even count.
    """
    out = []
    for a in axes:
        h = a[1] - a[0]
        P = max(factor * len(a) * h / 2, factor / 2 * R_cut)
        m = int(np.ceil(P / h - 1e-9))
        out.append(m + m % 2)
    return tuple(out)


def octant_folds(shape):
    """Per axis, the octant index |m| of each fft position (min(i, n - i)), (x, y)."""
    return tuple(np.minimum(np.arange(n), n - np.arange(n)) for n in shape[:2])


def unfold_octant(G, shape):
    """The rfft-layout kernel (Nx, Ny, Nz // 2 + 1) from its octant
    (Nx // 2 + 1, Ny // 2 + 1, Nz // 2 + 1): G(k_x, k_y, k_z) = G(|k_x|, |k_y|, k_z)
    for a kernel even in k_x and k_y. A gather, fused by XLA into its consumer under
    jit (no full-size copy); NumPy in, NumPy out."""
    fx, fy = octant_folds(shape)
    if isinstance(G, np.ndarray):
        return G[np.ix_(fx, fy, np.arange(G.shape[2]))]
    return G[jnp.asarray(fx)[:, None], jnp.asarray(fy)[None, :], :]


def vico_greengard_kernel_hat(axes, R_cut=None, factor=4, cache_dir=None, dtype=jnp.float64,
                              octant=False):
    """G(k) of the truncated Green's function on the rfft modes, real, (Nx, Ny, Nz // 2 + 1),
    or with ``octant`` only its k_x, k_y >= 0 octant, (Nx // 2 + 1, Ny // 2 + 1, Nz // 2 + 1).

    1. G_R(k) = mu0 (1 - cos k R) / k^2 (k = 0: mu0 R^2 / 2) on the non-negative
       octant of the precompute grid (``precompute_shape``); G_R is even.
    2. Real-space kernel g = inverse transform: an even real sequence, so a 3-D
       DCT-I of the octant, divided by the precompute volume.
    3. g at the minimum-image displacements of the box, zero on the Nyquist plane
       of an even axis.
    4. G = h^3 rfft(g), real since g is even, zero on the Nyquist plane of k of an
       even axis (B_hat = i G k x J_hat: a first derivative, not real there).
    5. G is even in k_x and k_y, so only the octant is kept; the full rfft layout is
       its unfolding (``unfold_octant``), 4x the memory. ``b_hat_free`` takes
       either.
    On the host (scipy.fft); the octant is cached as .npy in ``cache_dir`` keyed by
    the box and R_cut.
    """
    import scipy.fft as sfft

    shape = tuple(len(a) for a in axes)
    h = np.array([a[1] - a[0] for a in axes])
    R_cut = half_box_diagonal(axes) if R_cut is None else float(R_cut)
    Mp = np.array(precompute_shape(axes, R_cut, factor))
    path = None
    if cache_dir is not None:
        key = hashlib.sha256(repr((shape, np.round(h, 12).tolist(), round(R_cut, 12),
                                   int(factor))).encode()).hexdigest()[:16]
        os.makedirs(cache_dir, exist_ok=True)
        path = os.path.join(cache_dir, f"vg_kernel_hat_{key}_octant.npy")
        if os.path.exists(path):
            G = np.load(path)
            return jnp.asarray(G if octant else unfold_octant(G, shape), dtype)
    # 1. octant of G_R, modes 0 .. M / 2
    ks = [2 * np.pi * np.arange(m // 2 + 1) / (m * hi) for m, hi in zip(Mp, h)]
    k2 = ks[0][:, None, None] ** 2 + ks[1][None, :, None] ** 2 + ks[2][None, None, :] ** 2
    kR = np.sqrt(k2) * R_cut
    with np.errstate(invalid="ignore", divide="ignore"):
        GR = np.where(kR < 1e-3, (0.5 - kR**2 / 24 + kR**4 / 720) * R_cut**2,
                      (1 - np.cos(kR)) / np.where(k2 == 0, 1.0, k2))
    GR *= mu_0
    del k2, kR
    # 2. g at displacements 0 .. M / 2 (DCT-I of the even sequence)
    g = sfft.dctn(GR, type=1, workers=-1) / float(np.prod(Mp * h))
    del GR
    # 3. minimum-image displacements of the box, Nyquist planes zero
    idx = [np.abs(np.round(np.fft.fftfreq(n) * n).astype(int)) for n in shape]
    gb = g[np.ix_(*idx)]
    del g
    for m in _broadcast([_not_nyquist(n) for n in shape]):
        gb = gb * m
    # 4. back to the box modes
    G = sfft.rfftn(gb, workers=-1).real * float(np.prod(h))
    nyq = _broadcast([_not_nyquist(n, half=i == 2) for i, n in enumerate(shape)])
    G = np.ascontiguousarray((G * nyq[0] * nyq[1] * nyq[2])[: shape[0] // 2 + 1,
                                                              : shape[1] // 2 + 1])
    if path is not None:
        np.save(path, G)
    return jnp.asarray(G if octant else unfold_octant(G, shape), dtype)


def kernel_hat(axes, kernel="vico_greengard", dtype=jnp.float64, **kw):
    """The free-space kernel on the rfft modes: ``vico_greengard_kernel_hat`` (real,
    (Nx, Ny, Nz // 2 + 1), or its octant with ``octant=True``) or
    ``biot_savart_kernel_hat`` (complex, (..., 3))."""
    if kernel == "vico_greengard":
        return vico_greengard_kernel_hat(axes, dtype=dtype, **kw)
    if kernel == "hockney":
        return biot_savart_kernel_hat(axes, dtype)
    raise ValueError(f"kernel must be one of {KERNELS}")


def b_hat_free(J_hat, K_hat, k, i):
    """Component i of the free-space B_hat on the rfft modes.

    ``K_hat`` from ``kernel_hat``: complex (..., 3) is Hockney's, B_hat = mu0 J_hat x
    K_hat; real (Nx, Ny, Nz // 2 + 1) or its octant is the Vico-Greengard G,
    B_hat = i G k x J_hat, with ``k`` the broadcast rfft wavevectors.
    """
    j, l = (i + 1) % 3, (i + 2) % 3
    if K_hat.ndim == 4:
        return mu_0 * (J_hat[..., j] * K_hat[..., l] - J_hat[..., l] * K_hat[..., j])
    if K_hat.shape[:2] != J_hat.shape[:2]:
        K_hat = unfold_octant(K_hat, J_hat.shape)
    c = k[j] * J_hat[..., l] - k[l] * J_hat[..., j]
    return jax.lax.complex(-K_hat * c.imag, K_hat * c.real)  # i G c; G is real


def b_plasma_free(J_hat, axes, K_hat=None, kernel="vico_greengard"):
    """Free-space B on the box from J_hat, (Nx, Ny, Nz, 3).

    ``K_hat`` (``kernel_hat``) depends only on the grid, so it can be computed once
    and reused while J changes; otherwise it is built for ``kernel``.
    """
    shape = tuple(len(a) for a in axes)
    dt = J_hat.real.dtype
    K_hat = kernel_hat(axes, kernel, dt) if K_hat is None else K_hat
    k = _broadcast([jnp.asarray(ki, dt) for ki in wavevectors(axes)])
    return jnp.stack([jnp.fft.irfftn(b_hat_free(J_hat, K_hat, k, i), s=shape)
                      for i in range(3)], axis=-1)


def nufft_angles(axes, x):
    """2 pi (x - x_0) / L per axis, wrapped to [-pi, pi): the NUFFT coordinates of x."""
    x0 = np.array([a[0] for a in axes])
    L = np.array([len(a) * (a[1] - a[0]) for a in axes])
    t = 2 * np.pi * (np.asarray(x, dtype=float) - x0) / L
    return (t + np.pi) % (2 * np.pi) - np.pi


def full_spectrum(F_hat, nz):
    """The fftn of a real field from its rfft half spectrum (Hermitian reflection):
    F(-k) = conj F(k) fills k_z < 0. (Nx, Ny, Nz). No transform, exact."""
    nh = F_hat.shape[2]
    R = jnp.conj(F_hat[:, :, 1 : nz - nh + 1][:, :, ::-1])
    R = jnp.roll(jnp.flip(R, (0, 1)), 1, axis=(0, 1))  # index i -> (-i) mod n
    return jnp.concatenate([F_hat, R], axis=2)


@functools.partial(jax.jit, static_argnames=("nz", "eps", "upsampfac", "chunk", "half"))
def interpolate_hat(F_hat, t, nz, eps=1e-10, upsampfac=1.25, chunk=16384, half=False):
    """Real field of the rfft half spectrum ``F_hat`` (Nx, Ny, Nz // 2 + 1) at angles t.

    f(x) = sum_k F(k) exp(i k . (x - x_0)) / N over the centred modes, evaluated
    with a type-2 non-uniform FFT (``nufftax``, relative precision ``eps``), real
    part. ``t`` from ``nufft_angles``; ``chunk`` points per NUFFT call. Pure JAX,
    linear in F_hat. Equal to the field at the grid nodes.

    half=False  the full spectrum is rebuilt by Hermitian reflection
                (``full_spectrum``) and handed to the NUFFT: the interpolation of
                ``interpolate`` before the half-spectrum option, for any field.
    half=True   the NUFFT works on the half spectrum itself, f = Re sum c F_hat
                exp(i k . (x - x_0)) / N, c = 1 for k_z = 0 and the k_z Nyquist
                index of an even Nz, 2 otherwise, the k_z range centred by the
                factor exp(i s t_z), s = (Nz // 2 + 1) // 2: half the memory of the
                spectrum and of the NUFFT's upsampled grid. The same as half=False
                for fields without Nyquist content (the Vico-Greengard B). But
                the k_z = 0 modes, where a smooth field's content is, then sit at
                the edge of the NUFFT's band, where its error is largest: at
                upsampfac 1.25 the error is about 100x that of half=False at the
                same eps (Gaussian blob: eps 1e-10 gives 3e-8 vs 7e-10; eps 1e-14
                gives 9e-11, for about twice the interpolation time). Use with
                eps <= 1e-14.
    """
    single = F_hat.dtype == jnp.complex64
    t = t.astype(jnp.float32 if single else jnp.float64)
    eps = max(eps, 1e-6) if single else eps
    if half:
        nh = F_hat.shape[2]
        c = np.full(nh, 2.0)
        c[0] = 1
        if nz % 2 == 0:
            c[-1] = 1
        F = jnp.fft.fftshift(F_hat * jnp.asarray(c, F_hat.real.dtype), axes=(0, 1))
    else:
        F = jnp.fft.fftshift(full_spectrum(F_hat, nz))
    F = jnp.transpose(F, (2, 1, 0))  # axes (z, y, x), the x mode last
    # a Python loop over chunks, not lax.map: JAX cannot transpose a scan whose body
    # is not linear in the carried value
    v = jnp.concatenate([nufft3d2(t[i : i + chunk, 0], t[i : i + chunk, 1], t[i : i + chunk, 2],
                                  F, eps=eps, isign=1, upsampfac=upsampfac)
                         for i in range(0, t.shape[0], chunk)])
    if half:
        v = v * jnp.exp(1j * (F_hat.shape[2] // 2) * t[:, 2])
    return v.real / (F_hat.shape[0] * F_hat.shape[1] * nz)


def interpolate(field, axes, x, eps=1e-10, upsampfac=1.25, chunk=16384, half=False):
    """Spectral (trigonometric) interpolation of a periodic field on the box.

    ``field`` has shape (Nx, Ny, Nz) or (Nx, Ny, Nz, n) on the grid ``axes``; the
    result, at the Cartesian points x (n_points, 3), is the box's Fourier series
    evaluated with a type-2 non-uniform FFT from the rfft (``interpolate_hat``, with
    ``half``). It equals the field at the grid nodes. Shape (n_points,) or
    (n_points, n).

    Memory: one component at a time, on a grid ``upsampfac`` times finer per axis
    (1.25: about 2 times the spectrum), ``chunk`` points per call. A float32 field
    is interpolated in single precision (precision max(eps, 1e-6)).
    """
    vector = np.ndim(field) == 4
    f = field if vector else field[..., None]
    t = jnp.asarray(nufft_angles(axes, x))
    nz = len(axes[2])
    out = [np.asarray(interpolate_hat(jnp.fft.rfftn(jnp.asarray(f[..., c])), t, nz, eps,
                                      upsampfac, chunk, half)).astype(float)
           for c in range(f.shape[3])]
    out = np.stack(out, axis=-1)
    return out if vector else out[:, 0]


@functools.partial(jax.jit, static_argnames=("shape", "eps", "upsampfac", "chunk", "half"))
def _biot_savart(J, C, shape, eps, upsampfac, chunk, half):
    J_hat = fft_J(scatter(J, C["inside"], shape))  # (Nx, Ny, Nz // 2 + 1, 3)
    k = _broadcast(C["k"])
    return jnp.stack([interpolate_hat(b_hat_free(J_hat, C["K_hat"], k, i), C["t"], shape[2], eps,
                                      upsampfac, chunk, half) for i in range(3)], axis=-1)


class BiotSavart:
    """Linear map J (inside points of the box) -> free-space B_plasma (target points).

        B(x_t) = interpolate_hat(b_hat_free(fft_J(scatter(J))))

    All in JAX: linear in J, with jvp, vjp and linear_transpose; the Jacobian columns
    of B are ``apply`` of one coefficient's current. Constants (inside indices,
    kernel, wavevectors, target angles) are a pytree, ``consts()``, to pass as a jit
    argument: arrays closed over by a jitted function become compile-time constants
    of the XLA program (GB-sized at production resolutions). Memory per call, in
    units of one complex field on the box: J on the box (real, 3 components: 1.5),
    its half spectrum (1.5), one component of B_hat (0.5), the full spectrum handed
    to the NUFFT (1, or 0.5 with ``half``) and the NUFFT's upsampled grid (about 2,
    or 1 with ``half``, at upsampfac 1.25). The Vico-Greengard G is stored as its
    octant (1 / 16 of a complex field; the rfft layout would be 1 / 4).

    Parameters
    ----------
    box : CartesianGrid
        Periodic box with the plasma located (``locate_plasma``).
    targets : ndarray, shape (n_targets, 3)
        Cartesian points where B is wanted [m].
    kernel : {"vico_greengard", "hockney"}
    dtype : np.float64 or np.float32
        Precision of the FFTs and the NUFFT (eps is raised to 1e-6 for float32).
        The solve path needs float64.
    eps, upsampfac, chunk : NUFFT precision, oversampling and targets per call.
    half : bool
        Type-2 NUFFT on the half spectrum (``interpolate_hat``): less memory, needs
        eps <= 1e-14 at upsampfac 1.25.
    kernel_kw : arguments of ``vico_greengard_kernel_hat`` (R_cut, factor, cache_dir).
    """

    def __init__(self, box, targets, kernel="vico_greengard", dtype=np.float64, eps=1e-10,
                 upsampfac=1.25, chunk=65536, half=False, **kernel_kw):
        if box.inside is None:
            raise RuntimeError("locate the plasma on the box first (box.locate_plasma)")
        self.box, self.kernel, self.dtype = box, kernel, np.dtype(dtype)
        self.eps, self.upsampfac, self.half = eps, upsampfac, bool(half)
        self.n_targets = len(targets)
        self.chunk = int(min(chunk, max(self.n_targets, 1)))
        kw = dict(kernel_kw, octant=True) if kernel == "vico_greengard" else {}
        self._C = {"inside": jnp.asarray(box.inside, dtype=jnp.int32),
                   "t": jnp.asarray(nufft_angles(box.axes, targets), dtype=self.dtype),
                   "k": tuple(jnp.asarray(ki, self.dtype) for ki in wavevectors(box.axes)),
                   "K_hat": kernel_hat(box.axes, kernel, self.dtype, **kw)}

    @property
    def n_inside(self):
        return len(self.box.inside)

    def consts(self):
        """Every array the map reads, as a pytree (pass to ``apply`` as a jit argument)."""
        return self._C

    def apply(self, J, C=None):
        """B_plasma at the targets, (n_targets, 3) [T], from Cartesian J at the inside
        points, (n_inside, 3) [A/m^2]. Linear in J; pure JAX (jit, jvp, vjp)."""
        C = self._C if C is None else C
        return _biot_savart(jnp.asarray(J, dtype=self.dtype), C, self.box.shape, self.eps,
                            self.upsampfac, self.chunk, self.half)

    __call__ = apply
