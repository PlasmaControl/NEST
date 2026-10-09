"""Step 5 of NEST: the plasma's magnetic field B_plasma from J on the box G3.

Two ways to invert curl B = mu0 J, div B = 0 with the FFT of J (``fourier.fft_J``):

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
             in z), and further away the images come back.

Both use the unnormalized rfft of ``fourier.fft_J`` and return B on the box,
shape (Nx, Ny, Nz, 3), in Cartesian components [T].
"""

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
