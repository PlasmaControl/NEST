"""Step 4 of NEST: Fourier transform of J on the periodic box G3.

For each Cartesian component,

    J_hat[a, b, c] = sum_{p, q, r} J[p, q, r] exp(-2 pi i (a p / Nx + b q / Ny + c r / Nz)),

unnormalized (the inverse carries 1/N). J is real, so only half the spectrum
is kept along z (rfft): c = 0 .. Nz // 2, the other half being the complex
conjugate. The wavevector of index (a, b, c) is k = 2 pi (a~ / Lx, b~ / Ly,
c / Lz), with the centred index a~ = a for a <= Nx / 2 and a - Nx otherwise;
the largest |k_i| is the Nyquist wavenumber pi / dx_i.

Indices are counted from the first grid point x_0, so J_hat differs from
sum J exp(-i k . x) by the constant phase exp(-i k . x_0); the inverse
transform returns values on the same grid, so the phase never matters.
"""

import jax.numpy as jnp
import numpy as np


def wavevectors(axes, full=False):
    """(kx, ky, kz) of the rfft of a field on the grid ``axes``.

    kx and ky have the full length Nx, Ny in FFT order (0, 1, ..., -1); kz
    has Nz // 2 + 1 non-negative values, or all Nz in FFT order with ``full``
    (the full fftn). Units 1/m.
    """
    k = []
    for i, a in enumerate(axes):
        n, h = len(a), a[1] - a[0]
        half = i == 2 and not full
        k.append(2 * np.pi * (np.fft.rfftfreq(n, d=h) if half else np.fft.fftfreq(n, d=h)))
    return tuple(k)


def nyquist(axes):
    """Nyquist wavenumbers pi / dx_i and the largest |k|."""
    k = np.array([np.pi / (a[1] - a[0]) for a in axes])
    return k, float(np.linalg.norm(k))


def fft_J(J):
    """rfft of each Cartesian component of J, shape (Nx, Ny, Nz // 2 + 1, 3).

    Computed with JAX (on the GPU when DESC/JAX use it), one component at a
    time; float32 J gives complex64 J_hat.
    """
    return jnp.stack([jnp.fft.rfftn(jnp.asarray(J[..., i])) for i in range(3)], axis=-1)


def ifft_J(J_hat, shape):
    """Inverse of ``fft_J`` back to real space, shape ``shape`` + (3,)."""
    return jnp.stack(
        [jnp.fft.irfftn(J_hat[..., i], s=shape) for i in range(3)], axis=-1
    )
