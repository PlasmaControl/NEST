"""Free-space Biot-Savart J -> B on one field period of a toroidal annulus (JAX).

Grid: N_phi (odd) planes phi_k = 2 pi k / (NFP N_phi), Chebyshev nodes in r on
[R_in, R_out] and in z on [-z0, z0]; fields have shape (N_phi, n_r + 1, n_z + 1)
with cylindrical components (V_r, V_phi, V_z) last.

With A the Coulomb-gauge vector potential, Delta A = -mu0 J, B = curl A, and the
complex combination A_+ = A_x + i A_y = e^{i phi} (A_r + i A_phi):

    A_z    real, modes m = j NFP        (rfft of -mu0 J_z over the period)
    A_+    modes m = j NFP + 1          (fft of -mu0 (J_r + i J_phi) over the period)

each mode a 2-D free-space problem in (r, z) solved with James' correction
(``annulus_freespace``, gate A6). The curl, per FFT index j:

    B_+^(j) = i d_z A_+^(j) - i (d_r - j NFP / r) A_z^(j)
    psi^(j) = (d_r + (j NFP + 1) / r) A_+^(j),     B_z^(j) = (psi^(j) - conj psi^(-j)) / 2i

(A_z^(-j) = conj A_z^(j)); the inverse FFT of B_+^ is B_r + i B_phi and the
inverse rfft of B_z^ is B_z.

The precomputation (eigendecompositions, single-layer matrices) is NumPy
(``FreeSpaceAnnulus``); ``CylindricalSolver`` holds it as a pytree, so that a
jitted call takes the operators as inputs. The mode solves run as a
``lax.map`` over chunks of ``chunk`` modes: transient memory is that of one
chunk (its V_r, V_r^-1 and S_m gathered from the distinct-|m| tables).
Real operators act on complex data as two real products.
"""

import functools

import jax
import jax.numpy as jnp
import numpy as np
from scipy.constants import mu_0

from annulus_freespace import FreeSpaceAnnulus, mode_indices
from flux_interp import _Pytree


def _lmul(V, X):
    """Batched real V (c, a, b) times complex X (c, b, d)."""
    return (jnp.einsum("kab,kbd->kad", V, X.real) + 1j * jnp.einsum("kab,kbd->kad", V, X.imag))


def _rmul(X, M):
    """Complex X (..., a, b) times real M (b, d)."""
    return X.real @ M + 1j * (X.imag @ M)


def _along(D, X, axis):
    """Real matrix D applied along ``axis`` of complex or real X."""
    if jnp.iscomplexobj(X):
        return _along(D, X.real, axis) + 1j * _along(D, X.imag, axis)
    return jnp.moveaxis(jnp.tensordot(D, X, axes=(1, axis)), 0, axis)


class CylindricalSolver(_Pytree):
    """J (cylindrical components on the annulus grid) -> A or B, free space.

    Parameters
    ----------
    r_axis, z_axis : ChebAxis
    NFP, N_phi : int (N_phi odd)
    chunk : int
        Modes per step of the mode loop.
    fs : FreeSpaceAnnulus, optional
        A precomputation to reuse (shifts 0 and 1).
    """

    _dynamic = ("Vr", "Vr_inv", "lam_r", "ar", "Vz", "Vz_inv", "lam_z", "ez", "S",
                "Dr", "Dz", "r", "kz", "kp", "mz", "mp")

    def __init__(self, r_axis, z_axis, NFP, N_phi, chunk=16, fs=None, verbose=False):
        fs = fs or FreeSpaceAnnulus(r_axis, z_axis, NFP, N_phi, shifts=(0, 1), verbose=verbose)
        ap = fs.ap
        self.NFP, self.N_phi, self.chunk = int(NFP), int(N_phi), int(chunk)
        self.shape = (self.N_phi,) + ap.shape
        self.nr, self.nz = ap.nr, ap.nz
        self.face_sizes = ap.face_sizes
        f = lambda a: jnp.asarray(a, jnp.float64)  # noqa: E731
        self.Vr, self.Vr_inv, self.lam_r, self.ar = f(ap.Vr), f(ap.Vr_inv), f(ap.lam_r), f(ap.ar)
        self.Vz, self.Vz_inv, self.lam_z, self.ez = f(ap.Vz), f(ap.Vz_inv), f(ap.lam_z), f(ap.ez)
        self.S = f(fs.S)
        self.Dr, self.Dz, self.r = f(r_axis.D), f(z_axis.D), f(r_axis.s)
        jz = mode_indices(N_phi, real=True)
        jp = mode_indices(N_phi, real=False)
        self.jz, self.jp = jz, jp
        self.mz, self.mp = f(jz * NFP), f(jp * NFP + 1)
        self.kz = jnp.asarray(ap.k_of(jz * NFP), jnp.int32)
        self.kp = jnp.asarray(ap.k_of(jp * NFP + 1), jnp.int32)
        self.r_axis, self.z_axis = r_axis, z_axis
        self.fs = fs

    # ------------------------------------------------------------ per mode
    def _dirichlet(self, FI, k):
        W = _rmul(_lmul(self.Vr_inv[k], FI), self.Vz_inv.T)
        W = W / (self.lam_r[k][:, :, None] + self.lam_z[None, None, :])
        return _rmul(_lmul(self.Vr[k], W), self.Vz.T)

    def _solve_chunk(self, F, k):
        """Free-space u for one chunk of modal slabs F (c, n_r + 1, n_z + 1), indices k."""
        FI = F[:, 1:-1, 1:-1]
        uI = self._dirichlet(FI, k)
        u = jnp.pad(uI, ((0, 0), (1, 1), (1, 1)))
        # outward normal derivative on the faces (in, out, bot, top), packed
        ur = _along(self.Dr[jnp.array([0, -1])], u, 1)  # (c, 2, n_z + 1)
        uz = _along(self.Dz[jnp.array([0, -1])], u, 2)  # (c, n_r + 1, 2)
        sigma = jnp.concatenate([-ur[:, 0], ur[:, 1], -uz[:, :, 0], uz[:, :, 1]], axis=-1)
        S = self.S[k]
        g = (jnp.einsum("kij,kj->ki", S, sigma.real) + 1j * jnp.einsum("kij,kj->ki", S, sigma.imag))
        n1, n2 = self.nz + 1, self.nr + 1
        g_in, g_out = g[:, :n1], g[:, n1:2 * n1]
        g_bot, g_top = g[:, 2 * n1:2 * n1 + n2], g[:, 2 * n1 + n2:]
        ar = self.ar[k]
        rhs = (FI - ar[:, :, 0, None] * g_in[:, None, 1:-1] - ar[:, :, 1, None] * g_out[:, None, 1:-1]
               - g_bot[:, 1:-1, None] * self.ez[None, :, 0] - g_top[:, 1:-1, None] * self.ez[None, :, 1])
        uI = self._dirichlet(rhs, k)
        u = jnp.pad(uI, ((0, 0), (1, 1), (1, 1)))
        u = u.at[:, 0, :].set(g_in).at[:, -1, :].set(g_out)
        u = u.at[:, 1:-1, 0].set(g_bot[:, 1:-1]).at[:, 1:-1, -1].set(g_top[:, 1:-1])
        return u

    def solve_modes(self, F, k):
        """Free-space u for modal slabs F (K, n_r + 1, n_z + 1) with distinct-|m| indices k."""
        K, c = F.shape[0], self.chunk
        n = -(-K // c)
        pad = n * c - K
        Fp = jnp.pad(F, ((0, pad), (0, 0), (0, 0))).reshape((n, c) + F.shape[1:])
        kp = jnp.pad(k, (0, pad)).reshape(n, c)
        out = jax.lax.map(lambda a: self._solve_chunk(a[0], a[1]), (Fp, kp))
        return out.reshape((n * c,) + F.shape[1:])[:K]

    # ------------------------------------------------------------ fields
    def potential_modes(self, J):
        """FFT-index slabs of A_z (rfft half) and of A_+'s periodic factor (full)."""
        Fz = jnp.fft.rfft(-mu_0 * J[..., 2], axis=0)
        Fp = jnp.fft.fft(-mu_0 * (J[..., 0] + 1j * J[..., 1]), axis=0)
        return self.solve_modes(Fz, self.kz), self.solve_modes(Fp, self.kp)

    def potential(self, J):
        """A on the grid: (A_r, A_phi, A_z), shape J.shape."""
        Az, Ap = self.potential_modes(J)
        Ac = jnp.fft.ifft(Ap, axis=0)
        return jnp.stack([Ac.real, Ac.imag, jnp.fft.irfft(Az, n=self.N_phi, axis=0)], axis=-1)

    def field_modes(self, J):
        """FFT-index slabs of B_+ / e^{i phi} (full) and of B_z (rfft half)."""
        Az, Ap = self.potential_modes(J)
        N = self.N_phi
        jz = np.asarray(self.jz)
        # A_z at every signed j (FFT order): A_z^(-j) = conj A_z^(j)
        Az_full = jnp.zeros((N,) + Az.shape[1:], Az.dtype).at[jz].set(Az)
        Az_full = Az_full.at[(-jz[1:]) % N].set(jnp.conj(Az[1:]))
        inv_r = (1.0 / self.r)[None, :, None]
        Bp = (1j * _along(self.Dz, Ap, 2)
              - 1j * (_along(self.Dr, Az_full, 1) - self.mz_full()[:, None, None] * inv_r * Az_full))
        psi = _along(self.Dr, Ap, 1) + self.mp[:, None, None] * inv_r * Ap
        psi_neg = psi[(-jz) % N]  # psi at -j for j >= 0
        Bz = (psi[jz] - jnp.conj(psi_neg)) / 2j
        return Bp, Bz

    def mz_full(self):
        return jnp.asarray(np.asarray(self.jp) * self.NFP, jnp.float64)

    def field(self, J):
        """B on the grid: (B_r, B_phi, B_z), shape J.shape [T] for J [A/m^2]."""
        Bp, Bz = self.field_modes(J)
        Bc = jnp.fft.ifft(Bp, axis=0)
        return jnp.stack([Bc.real, Bc.imag, jnp.fft.irfft(Bz, n=self.N_phi, axis=0)], axis=-1)

    # ------------------------------------------------------------ diagnostics
    def _dphi(self, f):
        """d/dphi of a real field on the planes (spectral, N_phi odd)."""
        k = 1j * self.NFP * jnp.asarray(np.fft.fftfreq(self.N_phi, 1.0 / self.N_phi))
        return jnp.fft.ifft(k[:, None, None] * jnp.fft.fft(f, axis=0), axis=0).real

    def div_grid(self, V):
        """div V of cylindrical components V on the grid (spectral in all directions)."""
        r = self.r[None, :, None]
        return (_along(self.Dr, r * V[..., 0], 1) / r + self._dphi(V[..., 1]) / r
                + _along(self.Dz, V[..., 2], 2))

    def curl_grid(self, V):
        """curl V of cylindrical components V on the grid."""
        r = self.r[None, :, None]
        Vr, Vp, Vz = V[..., 0], V[..., 1], V[..., 2]
        return jnp.stack([self._dphi(Vz) / r - _along(self.Dz, Vp, 2),
                          _along(self.Dz, Vr, 2) - _along(self.Dr, Vz, 1),
                          (_along(self.Dr, r * Vp, 1) - self._dphi(Vr)) / r], axis=-1)

    # ------------------------------------------------------------ grid
    @property
    def phi(self):
        return 2 * np.pi * np.arange(self.N_phi) / (self.NFP * self.N_phi)

    def nodes(self):
        """(r, phi, z) of every node, shape (N_phi, n_r + 1, n_z + 1, 3)."""
        P, R, Z = np.meshgrid(self.phi, self.r_axis.s, self.z_axis.s, indexing="ij")
        return np.stack([R, P, Z], axis=-1)


@functools.partial(jax.jit, static_argnums=())
def field(solver, J):
    """Jitted B = solver.field(J), the operators passed as inputs."""
    return solver.field(J)


@jax.jit
def potential(solver, J):
    return solver.potential(J)
