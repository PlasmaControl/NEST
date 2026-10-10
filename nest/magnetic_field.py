"""Step 5 of NEST: the plasma's magnetic field B_plasma from J on the box G3.

Three ways to invert curl B = mu0 J, div B = 0 with the FFT of J (``fourier.fft_J``):

periodic     B_hat = i mu0 k x J_hat / |k|^2.
             Spectral, but it is the field of the infinite lattice of copies of
             the box, with zero mean: the neighbouring plasmas add an error that
             decays with the padding (roughly (plasma size / period)^3).

free space   Hockney-Eastwood: the discrete Biot-Savart sum
                 B(x) = mu0 / 4 pi sum_x' J(x') x (x - x') / |x - x'|^3 h^3
             over the grid points x' (the self term x' = x is zero), done as a
             circular convolution with the kernel K(d) = d / (4 pi |d|^3) at the
             minimum-image displacements d. The result is the free-space sum,
             with no images, wherever every source-target separation is its own
             minimum image (at most half a period in each direction). With a
             period at least twice the extent of the current (factor 2 in
             ``cartesian_grid``) that holds at least in the plasma's bounding
             box; beyond it, only within period / 2 - extent of it (about zero
             in z), and further away the images come back. The kernel is the
             singular Biot-Savart kernel sampled at the grid points, so the
             sum is a low-order quadrature of the continuous integral.

truncated    Vico, Greengard and Ferrando (J. Comput. Phys. 323, 2016): the
             free-space Green's function cut off at a radius L, G_L(r) = 1 /
             (4 pi r) for r < L, 0 beyond, whose Fourier transform is known,
                 G_L_hat(k) = (1 - cos(L |k|)) / |k|^2  (L^2 / 2 at k = 0).
             With L = half the diagonal of the box (default), G_L = G at every
             minimum-image displacement, so the region without images is the
             same as for Hockney-Eastwood, but the kernel is the band-limited
             one of the grid instead of point samples: B = mu0 curl (G_L * J)
             is spectrally accurate for a J resolved by the grid.
             G_L_hat oscillates on the scale 2 pi / L, finer than the box's
             wavevector spacing 2 pi / L_box in the short direction, so the
             discrete kernel is computed once (``truncated_green_hat``) from
             G_L_hat on a finer wavevector lattice (an s times longer period
             in each direction), the minimum-image cell of its inverse
             transform being the kernel on the box. B_hat = i mu0 k x J_hat
             K_hat, as in the periodic method with 1 / |k|^2 -> K_hat.

All three use the unnormalized rfft of ``fourier.fft_J`` and return B on the box,
shape (Nx, Ny, Nz, 3), in Cartesian components [T].
"""

import itertools

import jax
import jax.numpy as jnp
import numpy as np
from nufftax import nufft3d2
from scipy.constants import mu_0

from fourier import wavevectors


def _not_nyquist(n, half=False):
    """1 except at the Nyquist index of an even-length FFT axis."""
    m = np.ones(n // 2 + 1 if half else n)
    if n % 2 == 0:
        m[n // 2] = 0
    return m


def _broadcast(v):
    """Three 1-D arrays as (n, 1, 1), (1, n, 1), (1, 1, n)."""
    return [v[0][:, None, None], v[1][None, :, None], v[2][None, None, :]]


def _curl_solve(J_hat, axes, g):
    """B = irfft of i k x J_hat g, with g(k) on the rfft lattice (mu0 included).

    The Nyquist planes of even axes are dropped: a first derivative there is
    not real.
    """
    shape = tuple(len(a) for a in axes)
    dt = J_hat.real.dtype
    k = _broadcast([jnp.asarray(ki, dt) for ki in wavevectors(axes)])
    nyq = _broadcast([jnp.asarray(_not_nyquist(n, i == 2), dt) for i, n in enumerate(shape)])
    g = g * nyq[0] * nyq[1] * nyq[2]

    def component(i):
        j, l = (i + 1) % 3, (i + 2) % 3
        B_hat = 1j * g * (k[j] * J_hat[..., l] - k[l] * J_hat[..., j])
        return jnp.fft.irfftn(B_hat, s=shape)

    return jnp.stack([component(i) for i in range(3)], axis=-1)


def b_plasma_periodic(J_hat, axes):
    """B_hat = i mu0 k x J_hat / |k|^2, back to real space.

    The k = 0 mode (the mean of B over the box) is zero.
    """
    dt = J_hat.real.dtype
    k = _broadcast([jnp.asarray(ki, dt) for ki in wavevectors(axes)])
    k2 = k[0] ** 2 + k[1] ** 2 + k[2] ** 2
    return _curl_solve(J_hat, axes, jnp.where(k2 > 0, mu_0 / jnp.where(k2 > 0, k2, 1), 0))


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


def b_plasma_free(J_hat, axes, K_hat=None):
    """Free-space B = mu0 sum J(x') x K(x - x') h^3 by circular convolution.

    ``K_hat`` from ``biot_savart_kernel_hat`` depends only on the grid, so it
    can be computed once and reused while J changes.
    """
    shape = tuple(len(a) for a in axes)
    K_hat = biot_savart_kernel_hat(axes, J_hat.real.dtype) if K_hat is None else K_hat

    def component(i):
        j, l = (i + 1) % 3, (i + 2) % 3
        B_hat = mu_0 * (J_hat[..., j] * K_hat[..., l] - J_hat[..., l] * K_hat[..., j])
        return jnp.fft.irfftn(B_hat, s=shape)

    return jnp.stack([component(i) for i in range(3)], axis=-1)


def interpolate(field, axes, x, eps=1e-10, upsampfac=1.25, chunk=16384):
    """Spectral (trigonometric) interpolation of a periodic field on the box.

    ``field`` has shape (Nx, Ny, Nz) or (Nx, Ny, Nz, n) on the grid ``axes``;
    the result, at the Cartesian points x (n_points, 3), is the box's Fourier
    series sum_k F_k exp(i k . (x - x_0)) / N, evaluated with a type-2
    non-uniform FFT (``nufftax``, relative precision ``eps``). It equals the
    field at the grid nodes. Shape (n_points,) or (n_points, n).

    Memory: the transform works on a grid ``upsampfac`` times finer per axis
    (1.25: about 2 times the box, against 8 times for the usual 2), one
    component at a time (a NumPy ``field`` is sent to the GPU one component
    at a time), and on ``chunk`` points at a time. A float32 field is
    interpolated in single precision (complex64, precision max(eps, 1e-6)),
    half the memory of double.
    """
    vector = np.ndim(field) == 4
    f = field if vector else field[..., None]
    shape = f.shape[:3]
    # full spectrum per component, centred modes (-N/2 .. N/2 - 1), axes (z, y, x)
    x0 = np.array([a[0] for a in axes])
    L = np.array([len(a) * (a[1] - a[0]) for a in axes])
    t = 2 * np.pi * (np.asarray(x) - x0) / L  # angle of each point along each axis
    t = jnp.asarray((t + np.pi) % (2 * np.pi) - np.pi)  # in [-pi, pi)
    single = f.dtype == jnp.float32
    ctype, eps = (jnp.complex64, max(eps, 1e-6)) if single else (jnp.complex128, eps)
    t = t.astype(jnp.float32) if single else t
    out = []
    for c in range(f.shape[3]):
        F = jnp.fft.fftshift(jnp.fft.fftn(jnp.asarray(f[..., c]).astype(ctype)))
        F = jnp.transpose(F, (2, 1, 0))  # axes (z, y, x), the x mode last
        v = [np.asarray(jnp.real(nufft3d2(t[i : i + chunk, 0], t[i : i + chunk, 1],
                                          t[i : i + chunk, 2], F, eps=eps, isign=1,
                                          upsampfac=upsampfac)))
             for i in range(0, len(t), chunk)]
        out.append(np.concatenate(v).astype(float) / np.prod(shape))
        del F
    out = np.stack(out, axis=-1)
    return out if vector else out[:, 0]


@jax.jit
def _truncated_residue(w, kx, ky, kz, px, py, pz, L):
    """w + Re(phase x ifft of G_L_hat) for one residue class of the fine lattice."""
    kk = jnp.sqrt(kx**2 + ky**2 + kz**2)
    G = jnp.where(kk > 0, 2 * (jnp.sin(L * kk / 2) / jnp.where(kk > 0, kk, 1)) ** 2, L**2 / 2)
    return w + jnp.real(px * py * pz * jnp.fft.ifftn(G.astype(px.dtype)))


def truncated_green_hat(axes, L=None, dtype=jnp.float32):
    """rfft of the discrete kernel h^3 G_L(d) of the truncated Green's function.

    G_L(r) = 1 / (4 pi r) for r < L (default: half the box diagonal) and 0
    beyond. Its band-limited values at the minimum-image displacements d = j h
    (j_i = -N_i / 2 .. N_i / 2 - 1) are the inverse DFT of G_L_hat on a lattice
    with an s_i times longer period, s_i L_i >= L + L_i / 2, so that no image of
    G_L reaches the cell. That DFT, of size s N, is done as s_x s_y s_z DFTs of
    size N, one per residue r of the fine wavenumber index m = s n + r:

        w(j) = 1 / (s_x s_y s_z) sum_r exp(2 pi i r . j / (s N)) ifft_N[G_L_hat(k_{s n + r})](j)

    so only arrays of the box size are needed. The kernel is zero on the
    Nyquist plane of an even axis (ambiguous displacement +-N h / 2), as in
    ``biot_savart_kernel_hat``; it is then even, so K_hat is real. Computed in
    the default JAX float precision (float64 with JAX_ENABLE_X64) and returned
    as ``dtype``, shape (Nx, Ny, Nz // 2 + 1).
    """
    shape = tuple(len(a) for a in axes)
    h = np.array([a[1] - a[0] for a in axes])
    P = h * np.array(shape)
    L = float(np.linalg.norm(P)) / 2 if L is None else float(L)
    s = [int(np.ceil((L + p / 2) / p)) for p in P]
    j = [np.fft.fftfreq(n) * n for n in shape]  # displacement indices, FFT order
    cdt = jnp.complex128 if jnp.result_type(float) == jnp.float64 else jnp.complex64
    w = jnp.zeros(shape, jnp.result_type(float))
    for r in itertools.product(*(range(si) for si in s)):
        k, phase = [], []
        for i in range(3):
            M = s[i] * shape[i]
            m = (s[i] * j[i] + r[i] + M // 2) % M - M // 2  # centred fine index
            k.append(jnp.asarray(2 * np.pi * m / (M * h[i])))
            phase.append(jnp.asarray(np.exp(2j * np.pi * r[i] * j[i] / M), cdt))
        w = _truncated_residue(w, *_broadcast(k), *_broadcast(phase), L)
    nyq = _broadcast([jnp.asarray(_not_nyquist(n)) for n in shape])
    w = w * nyq[0] * nyq[1] * nyq[2] / np.prod(s)
    return jnp.fft.rfftn(w).real.astype(dtype)


def b_plasma_truncated(J_hat, axes, K_hat=None):
    """Free-space B = mu0 curl (G_L * J) with the truncated Green's function.

    ``K_hat`` from ``truncated_green_hat`` depends only on the grid, so it can
    be computed once and reused while J changes.
    """
    K_hat = truncated_green_hat(axes, dtype=J_hat.real.dtype) if K_hat is None else K_hat
    return _curl_solve(J_hat, axes, mu_0 * K_hat)
