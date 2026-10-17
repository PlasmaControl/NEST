"""Free-space Poisson solve for an NFP-periodic source in a toroidal annulus (NumPy).

The annulus is Omega = [R_in, R_out] x [-z0, z0] in (r, z) and one field period
0 <= phi < 2 pi / NFP, with N_phi (odd) uniform planes phi_k = 2 pi k / (NFP N_phi).
For a source f supported inside Omega (and vanishing near its boundary), the
free-space solution of Delta u = f, u -> 0 at infinity, is computed in Omega
mode by mode (James' method):

    u_D     Dirichlet solution (u_D = 0 on the boundary), ``AnnulusPoisson``
    sigma   outward normal derivative of u_D on the four faces
    g       S_m sigma on the boundary nodes, ``single_layer``
    u       L_m u = f inside, u = g on the boundary

since u_free = u_D + S[sigma] in Omega and S[sigma] is harmonic there.

Fields carry a mode shift s: a field whose physical modes are m = j NFP + s,
with j the FFT index over the period. s = 0 is a scalar such as A_z (real, rfft);
s = 1 is A_+ = A_x + i A_y, whose FFT is that of the NFP-periodic A_r + i A_phi
(see docs/cylindrical_grid_plan.md, D5).

This is the reference implementation of stage A (gate A6); the JAX apply path of
stage B is checked against it.
"""

import time

import numpy as np

from annulus_poisson import AnnulusPoisson
from single_layer import single_layer_matrix


def mode_indices(N_phi, real):
    """FFT indices j of a period with N_phi (odd) planes: 0..(N-1)/2 (rfft) or all
    j in FFT order (0, 1, .., (N-1)/2, -(N-1)/2, .., -1)."""
    if N_phi % 2 == 0:
        raise ValueError("N_phi must be odd")
    if real:
        return np.arange(N_phi // 2 + 1)
    return np.fft.fftfreq(N_phi, 1.0 / N_phi).astype(int)


class FreeSpaceAnnulus:
    """Precomputed per-mode operators for the shifts in ``shifts`` (0, 1 or both).

    Parameters
    ----------
    r_axis, z_axis : ChebAxis
    NFP, N_phi : int
    shifts : tuple of int
    """

    def __init__(self, r_axis, z_axis, NFP, N_phi, shifts=(0,), verbose=False, S_dtype=np.float64,
                 **sl_kw):
        self.NFP, self.N_phi, self.shifts = int(NFP), int(N_phi), tuple(shifts)
        ms = set()
        for s in self.shifts:
            ms |= set(np.abs(mode_indices(N_phi, real=(s == 0)) * NFP + s).tolist())
        t0 = time.time()
        self.ap = AnnulusPoisson(r_axis, z_axis, sorted(ms))
        t1 = time.time()
        pts, _ = self.ap.boundary_points()
        self.S = single_layer_matrix(self.ap, self.ap.ms, pts, verbose=verbose, dtype=S_dtype, **sl_kw)
        self.timing = {"eig": t1 - t0, "S": time.time() - t1}
        if verbose:
            print(f"  FreeSpaceAnnulus: {len(self.ap.ms)} distinct |m| <= {self.ap.ms.max()}, "
                  f"N_b = {self.ap.n_boundary}; eig {self.timing['eig']:.1f} s, "
                  f"S {self.timing['S']:.1f} s ({self.S.nbytes / 1e6:.0f} MB)", flush=True)

    @property
    def phi(self):
        return 2 * np.pi * np.arange(self.N_phi) / (self.NFP * self.N_phi)

    def solve_modes(self, F, m):
        """Free-space u for modal slabs F (n_modes, n_r + 1, n_z + 1) with modes m."""
        ap = self.ap
        k = ap.k_of(m)
        uD = ap.dirichlet(F, k)
        sigma = ap.pack(ap.flux(uD))
        g = np.einsum("kij,kj->ki", self.S[k], sigma)
        return ap.solve(F, g, k), {"uD": uD, "sigma": sigma, "g": g}

    def solve(self, f, shift=0, return_parts=False):
        """Free-space u of f on the grid (N_phi, n_r + 1, n_z + 1).

        shift 0: f real, rfft in phi. shift s != 0: f is the NFP-periodic factor of a
        field with modes j NFP + s (complex), full FFT.
        """
        real = shift == 0
        j = mode_indices(self.N_phi, real)
        F = np.fft.rfft(f, axis=0) if real else np.fft.fft(f, axis=0)
        U, parts = self.solve_modes(F, j * self.NFP + shift)
        u = np.fft.irfft(U, n=self.N_phi, axis=0) if real else np.fft.ifft(U, axis=0)
        return (u, parts) if return_parts else u
