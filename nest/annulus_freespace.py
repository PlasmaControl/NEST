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


def z_reflection(ap, tol=1e-12):
    """The permutation P of the packed boundary nodes under z -> -z (the lateral faces
    reversed, the caps exchanged), or None if the z axis is not symmetric about 0."""
    z = ap.z_axis.s
    if np.abs(z[::-1] + z).max() > tol * np.abs(z).max():
        return None
    from single_layer import FACES
    sizes = ap.face_sizes
    offs = dict(zip(FACES, np.cumsum([0] + [sizes[f] for f in FACES])[:-1]))
    perm = np.empty(ap.n_boundary, int)
    for f in FACES:
        n, o = sizes[f], offs[f]
        if f in ("in", "out"):
            perm[o:o + n] = o + np.arange(n)[::-1]
        else:
            perm[o:o + n] = offs["top" if f == "bot" else "bot"] + np.arange(n)
    return perm


class ZFold:
    """S_m in the even and odd parts of the boundary under z -> -z.

    On a z-symmetric annulus S_m = P S_m P (P the node reflection), so in the
    orthonormal basis u_a = c_a (e_i + e_Pi) (c = 1/sqrt 2 for a pair, 1/2 for a node
    on z = 0) and v_a = (e_i - e_Pi) / sqrt 2 it is block diagonal:

        S_e[a, b] = 2 c_a c_b (S[i, j] + S[i, Pj]),   S_o[a, b] = S[i, j] - S[i, Pj]

    with i, j the representatives (i <= Pi). Only the rows S[rep, :] are computed, and
    S_e, S_o together hold half of S_m. ``apply`` forms S_m sigma with gathers only
    (transposable by JAX).
    """

    def __init__(self, perm):
        n = len(perm)
        idx = np.arange(n)
        self.rep = idx[idx <= perm]
        self.partner = perm[self.rep]
        fixed = self.rep == self.partner
        self.c = np.where(fixed, 0.5, 1 / np.sqrt(2))
        self.odd_rows = np.nonzero(~fixed)[0]  # rows of rep that have an odd part
        self.odd, self.podd = self.rep[~fixed], self.partner[~fixed]
        n_e, n_o = len(self.rep), len(self.odd)
        a_of = np.empty(n, int)
        a_of[self.rep] = np.arange(n_e)
        a_of[self.partner] = np.arange(n_e)
        self.a_of = a_of
        self.ce = np.where(fixed, 1.0, 1 / np.sqrt(2))[a_of]
        b_of = np.full(n, n_o)  # a zero slot for the nodes on z = 0
        b_of[self.odd] = np.arange(n_o)
        b_of[self.podd] = np.arange(n_o)
        self.b_of = b_of
        so = np.zeros(n)
        so[self.odd], so[self.podd] = 1 / np.sqrt(2), -1 / np.sqrt(2)
        self.so = so
        self.n_e, self.n_o = n_e, n_o

    def fold(self, S_rows, dtype=None):
        """(K, n_e, N_b) rows S[:, rep, :] -> S_e (K, n_e, n_e), S_o (K, n_o, n_o)."""
        K = S_rows.shape[0]
        dtype = dtype or S_rows.dtype
        S_e = np.empty((K, self.n_e, self.n_e), dtype)
        S_o = np.empty((K, self.n_o, self.n_o), dtype)
        cc = 2 * self.c[:, None] * self.c[None, :]
        for k in range(K):  # mode by mode: no full-size temporaries
            R = np.asarray(S_rows[k], np.float64)
            S_e[k] = cc * (R[:, self.rep] + R[:, self.partner])
            S_o[k] = (R[self.odd_rows][:, self.odd] - R[self.odd_rows][:, self.podd])
        return S_e, S_o

    def apply(self, S_e, S_o, sigma, mv, xp=np):
        """S_m sigma for packed sigma (..., N_b); mv(S, x) the batched product."""
        rdt = sigma.real.dtype  # the coefficients in sigma's precision (no promotion)
        c, ce, so = (xp.asarray(a, rdt) for a in (self.c, self.ce, self.so))
        s_e = c * (sigma[..., self.rep] + sigma[..., self.partner])
        s_o = (sigma[..., self.odd] - sigma[..., self.podd]) * xp.asarray(1 / np.sqrt(2), rdt)
        y_e, y_o = mv(S_e, s_e), mv(S_o, s_o)
        y_o = xp.concatenate([y_o, xp.zeros(y_o.shape[:-1] + (1,), y_o.dtype)], axis=-1)
        return ce * y_e[..., self.a_of] + so * y_o[..., self.b_of]


class FreeSpaceAnnulus:
    """Precomputed per-mode operators for the shifts in ``shifts`` (0, 1 or both).

    Parameters
    ----------
    r_axis, z_axis : ChebAxis
    NFP, N_phi : int
    shifts : tuple of int
    """

    def __init__(self, r_axis, z_axis, NFP, N_phi, shifts=(0,), verbose=False, S_dtype=np.float64,
                 zsym="auto", **sl_kw):
        self.NFP, self.N_phi, self.shifts = int(NFP), int(N_phi), tuple(shifts)
        ms = set()
        for s in self.shifts:
            ms |= set(np.abs(mode_indices(N_phi, real=(s == 0)) * NFP + s).tolist())
        t0 = time.time()
        self.ap = AnnulusPoisson(r_axis, z_axis, sorted(ms))
        t1 = time.time()
        pts, _ = self.ap.boundary_points()
        perm = z_reflection(self.ap) if zsym else None
        if zsym is True and perm is None:
            raise ValueError("zsym=True needs a z axis symmetric about 0")
        self.zfold = None if perm is None else ZFold(perm)
        if self.zfold is None:
            self.S = single_layer_matrix(self.ap, self.ap.ms, pts, verbose=verbose, dtype=S_dtype,
                                         **sl_kw)
            self.S_o = None
        else:  # half the rows, folded into the even and odd blocks
            rows = single_layer_matrix(self.ap, self.ap.ms, pts[self.zfold.rep], verbose=verbose,
                                       dtype=S_dtype, **sl_kw)
            self.S, self.S_o = self.zfold.fold(rows)
            del rows
        self.timing = {"eig": t1 - t0, "S": time.time() - t1}
        if verbose:
            print(f"  FreeSpaceAnnulus: {len(self.ap.ms)} distinct |m| <= {self.ap.ms.max()}, "
                  f"N_b = {self.ap.n_boundary}{' (z-folded)' if self.zfold else ''}; eig "
                  f"{self.timing['eig']:.1f} s, S {self.timing['S']:.1f} s "
                  f"({self.S_nbytes / 1e6:.0f} MB)", flush=True)

    @property
    def S_nbytes(self):
        return sum(a.nbytes for a in (self.S, self.S_o) if a is not None)

    def apply_S(self, k, sigma):
        """S_m sigma for packed sigma (n_modes, N_b), distinct-|m| indices k."""
        if self.zfold is None:
            return np.einsum("kij,kj->ki", self.S[k], sigma)
        return self.zfold.apply(self.S[k], self.S_o[k], sigma,
                                lambda S, x: np.einsum("kij,kj->ki", S, x))

    @property
    def phi(self):
        return 2 * np.pi * np.arange(self.N_phi) / (self.NFP * self.N_phi)

    def solve_modes(self, F, m):
        """Free-space u for modal slabs F (n_modes, n_r + 1, n_z + 1) with modes m."""
        ap = self.ap
        k = ap.k_of(m)
        uD = ap.dirichlet(F, k)
        sigma = ap.pack(ap.flux(uD))
        g = self.apply_S(k, sigma)
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
