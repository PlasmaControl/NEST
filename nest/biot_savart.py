"""B_plasma as one linear, differentiable map of J (for the Jacobian and the optimizer).

``BiotSavart`` maps J at the box points inside the plasma (``CartesianGrid.inside``)
to B_plasma at fixed target points x_t (e.g. the collocation points of the force
residual), all in JAX:

    J_box    J at the inside points, zero elsewhere (scatter on fixed indices)
    J_hat    FFT of each Cartesian component on the periodic box (unnormalized)
    B_hat    = J_hat x Q(k),   Q = i q(k) purely imaginary (the kernel, below)
    B(x_t)   = sum_k B_hat(k) exp(i k . (x_t - x_0)) / N   (type-2 NUFFT, ``nufftax``)

The last step is the spectral interpolation of ``magnetic_field.interpolate``,
applied to B_hat directly: there is no inverse FFT to the box and back. The map is
linear in J and contains only scatter, FFT and the NUFFT, so jvp, vjp and
linear_transpose are available; the B_plasma columns of the Jacobian are
``apply`` of the current of one coefficient.

Two kernels:

hockney          q = mu0 Im FFT[h^3 K(d)],  K(d) = d / (4 pi |d|^3) at the minimum-image
                 displacements, K(0) = 0. Q x J_hat is the discrete Biot-Savart sum
                 of ``magnetic_field.b_plasma_free`` (second order in h).
vico_greengard   q = -G(k) k,  G the Fourier transform of the truncated Green's
                 function mu0 / (4 pi r) for r < R_cut, computed as in Vico,
                 Greengard and Ferrando (J. Comput. Phys. 323, 2016): sampled
                 G_R(k) = mu0 (1 - cos k R_cut) / k^2 on a finer precompute
                 grid, transformed to real space, cut to the minimum-image
                 displacements of the box, transformed back. B is then the
                 continuous free-space Biot-Savart integral of the sampled J
                 (spectrally accurate); the error left is that of the sampling of
                 J, i.e. its smoothness at the boundary.

Both are free-space fields, with no periodic images, wherever every source-target
displacement is at most half a period in each direction. The box of
``CartesianGrid.periodic_box`` (period at least twice the plasma extent) gives that
for targets inside the plasma's bounding box. R_cut defaults to the diagonal of
the half-period box, which bounds every such displacement. Even-length axes have
their Nyquist plane removed (q = 0 there), which keeps the operator exactly
symmetric under the reflections of the box.

Constants (inside indices, kernel, target angles) are a pytree, ``consts()``, to
pass as a jit argument: arrays closed over by a jitted function become compile-
time constants of the XLA program (GB-sized at production resolutions).
"""

import functools
import hashlib
import os

import jax
import jax.numpy as jnp
import numpy as np
from nufftax import nufft3d2
from scipy.constants import mu_0

from fourier import wavevectors

KERNELS = ("hockney", "vico_greengard")


def _box(axes):
    """Shape, spacings h, periods N h and first point x_0 of the box ``axes``."""
    shape = tuple(len(a) for a in axes)
    h = np.array([a[1] - a[0] for a in axes])
    return shape, h, np.array(shape) * h, np.array([a[0] for a in axes])


def _min_image(n):
    """Integer minimum-image displacements of an n-point periodic axis (FFT order)."""
    return np.round(np.fft.fftfreq(n) * n).astype(int)


def _nyquist_mask(shape):
    """1 except on the Nyquist plane (index n / 2) of each even-length axis, (n,1,1) etc."""
    out = []
    for i, n in enumerate(shape):
        m = np.ones(n)
        if n % 2 == 0:
            m[n // 2] = 0
        out.append(m.reshape([n if j == i else 1 for j in range(3)]))
    return out


def hockney_kernel(axes):
    """q = mu0 Im FFT[h^3 K(d)] of the discrete Biot-Savart kernel, (3, Nx, Ny, Nz).

    K(d) = d / (4 pi |d|^3) at the minimum-image displacements, K(0) = 0 and
    K = 0 on the Nyquist plane of an even axis (as in ``biot_savart_kernel_hat``).
    The kernel is real and odd, so its FFT is purely imaginary. Full spectrum
    (the type-2 interpolation needs all modes), on the host.
    """
    shape, h, _, _ = _box(axes)
    d = [(hi * _min_image(n)).reshape([n if j == i else 1 for j in range(3)])
         for i, (hi, n) in enumerate(zip(h, shape))]
    r2 = d[0] ** 2 + d[1] ** 2 + d[2] ** 2
    with np.errstate(divide="ignore"):
        w = np.where(r2 > 0, 1 / (4 * np.pi * np.where(r2 > 0, r2, 1) ** 1.5), 0)
    for m in _nyquist_mask(shape):
        w = w * m
    w = w * float(np.prod(h))
    return np.stack([mu_0 * np.fft.fftn(w * d[i]).imag for i in range(3)])


def precompute_shape(axes, R_cut, factor=4):
    """Points per axis of the Vico-Greengard precompute grid (spacing h of the box).

    The precompute period must hold the truncated kernel's support plus the
    source window and resolve the oscillation of (1 - cos k R) / k^2 in k
    (Vico et al. 2016). With the source window half the box period L_i / 2:
    P_i = max(factor L_i / 2, (factor / 2) R_cut), rounded up to an even count.
    """
    _, h, period, _ = _box(axes)
    P = np.maximum(factor * period / 2, factor / 2 * R_cut)
    M = np.ceil(P / h - 1e-9).astype(int)
    return tuple(int(m + m % 2) for m in M)


def half_box_diagonal(axes):
    """Diagonal of the half-period box [m]: the default R_cut."""
    return float(np.linalg.norm(_box(axes)[2] / 2))


def vico_greengard_kernel(axes, R_cut=None, factor=4, cache_dir=None, verbose=False):
    """G(k) of the truncated Green's function on the box modes (FFT order), real.

    1. G_R(k) = mu0 (1 - cos k R) / k^2 (k = 0: mu0 R^2 / 2) on the non-negative
       octant of the precompute grid (``precompute_shape``); G_R is even.
    2. Real-space kernel g = inverse transform: an even real sequence, so a 3-D
       DCT-I of the octant, divided by the precompute volume.
    3. g at the minimum-image displacements of the box, zero on the Nyquist
       plane of an even axis.
    4. G = h^3 FFT(g), real since g is even.
    Host computation (scipy.fft), cached as .npy in ``cache_dir`` keyed by the box
    and R_cut. Shape (Nx, Ny, Nz), units T m / A (times m^2).
    """
    import scipy.fft as sfft

    shape, h, _, _ = _box(axes)
    R_cut = half_box_diagonal(axes) if R_cut is None else float(R_cut)
    Mp = np.array(precompute_shape(axes, R_cut, factor))
    path = None
    if cache_dir is not None:
        key = hashlib.sha256(repr((shape, np.round(h, 12).tolist(), round(R_cut, 12),
                                   int(factor))).encode()).hexdigest()[:16]
        os.makedirs(cache_dir, exist_ok=True)
        path = os.path.join(cache_dir, f"vg_kernel_{key}.npy")
        if os.path.exists(path):
            return np.load(path)
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
    # 3. minimum-image displacements of the box
    idx = [np.abs(_min_image(n)) for n in shape]
    if np.any([i.max() > m // 2 for i, m in zip(idx, Mp)]):
        raise ValueError("precompute grid smaller than the box")
    gb = g[np.ix_(*idx)]
    del g
    for m in _nyquist_mask(shape):
        gb = gb * m
    # 4. back to the box modes
    G = np.ascontiguousarray(sfft.fftn(gb, workers=-1).real * float(np.prod(h)))
    if verbose:
        print(f"VG kernel: box {shape}, R_cut {R_cut:.4f} m, precompute {tuple(Mp)}", flush=True)
    if path is not None:
        np.save(path, G)
    return G


def _angles(axes, x):
    """2 pi (x - x_0) / L per axis, wrapped to [-pi, pi): NUFFT coordinates of x."""
    _, _, period, x0 = _box(axes)
    t = 2 * np.pi * (np.asarray(x, dtype=float) - x0) / period
    return (t + np.pi) % (2 * np.pi) - np.pi


@functools.partial(jax.jit, static_argnames=("shape", "kernel", "eps", "upsampfac", "chunk"))
def _apply(J, C, shape, kernel, eps, upsampfac, chunk):
    """B (n_targets, 3) from J (n_inside, 3); see ``BiotSavart``."""
    real = J.dtype
    ctype = jnp.complex64 if real == jnp.float32 else jnp.complex128
    n = int(np.prod(shape))
    J_hat = [jnp.fft.fftn(jnp.zeros(n, real).at[C["inside"]].set(J[:, c], unique_indices=True)
                          .reshape(shape)) for c in range(3)]
    if kernel == "vico_greengard":
        k = C["k"]
        k = [k[0][:, None, None], k[1][None, :, None], k[2][None, None, :]]
        q = [-C["G"] * ki for ki in k]
    else:
        q = [C["q"][i] for i in range(3)]
    t = C["t"]
    out = []
    for b in range(3):
        j, l = (b + 1) % 3, (b + 2) % 3
        c = J_hat[j] * q[l] - J_hat[l] * q[j]          # B_hat = i c
        B_hat = jax.lax.complex(-c.imag, c.real).astype(ctype)
        F = jnp.transpose(jnp.fft.fftshift(B_hat), (2, 1, 0))   # axes (z, y, x), x mode last
        # a Python loop over target chunks, not lax.map: JAX cannot transpose a scan
        # whose body is not linear in the carried value, and linear_transpose is wanted
        v = [nufft3d2(t[i : i + chunk, 0], t[i : i + chunk, 1], t[i : i + chunk, 2], F,
                      eps=eps, isign=1, upsampfac=upsampfac).real
             for i in range(0, t.shape[0], chunk)]
        out.append(jnp.concatenate(v) / n)
    return jnp.stack(out, axis=-1)


class BiotSavart:
    """Linear map J (inside points of the box) -> B_plasma (target points).

    Parameters
    ----------
    box : CartesianGrid
        Periodic box with the plasma located (``locate_plasma``).
    targets : ndarray, shape (n_targets, 3)
        Cartesian points where B is wanted [m].
    kernel : {"hockney", "vico_greengard"}
    dtype : np.float64 or np.float32
        Precision of the FFTs and the NUFFT (eps is raised to 1e-6 for float32).
        The solve path needs float64.
    eps, upsampfac : NUFFT precision and oversampling (1.25: about 2x the box
        in memory, against 8x for 2.0).
    chunk : int
        Targets per NUFFT call (memory).
    R_cut, vg_factor, cache_dir : Vico-Greengard parameters (``vico_greengard_kernel``).
    """

    def __init__(self, box, targets, kernel="vico_greengard", dtype=np.float64, eps=1e-10,
                 upsampfac=1.25, chunk=65536, R_cut=None, vg_factor=4, cache_dir=None):
        if box.inside is None:
            raise RuntimeError("locate the plasma on the box first (box.locate_plasma)")
        if kernel not in KERNELS:
            raise ValueError(f"kernel must be one of {KERNELS}")
        self.box, self.kernel, self.dtype = box, kernel, np.dtype(dtype)
        self.eps = max(eps, 1e-6) if self.dtype == np.float32 else eps
        self.upsampfac = upsampfac
        targets = np.asarray(targets, dtype=float)
        self.n_targets = len(targets)
        self.chunk = int(min(chunk, max(self.n_targets, 1)))
        self.shape = box.shape
        C = {"inside": jnp.asarray(box.inside, dtype=jnp.int32),
             "t": jnp.asarray(_angles(box.axes, targets), dtype=self.dtype)}
        if kernel == "vico_greengard":
            self.R_cut = half_box_diagonal(box.axes) if R_cut is None else float(R_cut)
            C["G"] = jnp.asarray(vico_greengard_kernel(box.axes, self.R_cut, vg_factor, cache_dir),
                                 dtype=self.dtype)
            C["k"] = tuple(jnp.asarray(k, dtype=self.dtype)
                           for k in wavevectors(box.axes, full=True))
        else:
            C["q"] = jnp.asarray(hockney_kernel(box.axes), dtype=self.dtype)
        self._C = C

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
        return _apply(jnp.asarray(J, dtype=self.dtype), C, self.shape, self.kernel, self.eps,
                      self.upsampfac, self.chunk)

    __call__ = apply
