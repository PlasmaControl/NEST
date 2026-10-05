"""Fourier-Zernike representation of the plasma current density.

DESC computes J pointwise from the R, Z, lambda series, so to store J as a
Fourier-Zernike series it is sampled on a quadrature grid and fitted by least
squares in DESC's real basis

    J_a(rho, theta, zeta) = sum_k c_k^a  Z_l^|m|(rho)  T_m(theta)  T_n(NFP zeta),
    T_m(x) = cos(|m| x) for m >= 0,  sin(|m| x) for m < 0
    
"""

import numpy as np

from desc.basis import FourierZernikeBasis
from desc.grid import QuadratureGrid
from desc.transform import Transform

COMPONENTS = ("J_R", "J_phi", "J_Z")


class CurrentFZ:
    """J_R, J_phi, J_Z as Fourier-Zernike series sharing one basis.

    Parameters
    ----------
    basis : FourierZernikeBasis
    c : ndarray, shape (basis.num_modes, 3)
        Coefficients of (J_R, J_phi, J_Z) in A/m^2.
    """

    def __init__(self, basis, c):
        self.basis = basis
        self.c = np.asarray(c)

    @classmethod
    def fit(cls, eq, L=None, M=None, N=None, oversample=2):
        """Least-squares fit of the DESC current density of ``eq``.

        Defaults to twice the equilibrium resolution: J contains second
        derivatives and products of the R, Z, lambda series, so its spectrum is
        wider than theirs.
        """
        L = 2 * eq.L if L is None else L
        M = 2 * eq.M if M is None else M
        N = 2 * eq.N if N is None else N
        basis = FourierZernikeBasis(
            L, M, N, NFP=eq.NFP, sym=False, spectral_indexing=eq.spectral_indexing
        )
        grid = QuadratureGrid(
            L=oversample * L, M=oversample * M, N=oversample * N, NFP=eq.NFP
        )
        J = np.asarray(eq.compute("J", grid=grid)["J"])
        tr = Transform(grid, basis, build=False, build_pinv=True)
        c = np.stack([np.asarray(tr.fit(J[:, i])) for i in range(3)], axis=-1)
        return cls(basis, c)

    def evaluate(self, grid):
        """(J_R, J_phi, J_Z) at the nodes of ``grid``, shape (num_nodes, 3)."""
        tr = Transform(grid, self.basis, build=True)
        return np.stack(
            [np.asarray(tr.transform(self.c[:, i])) for i in range(3)], axis=-1
        )

    def save(self, path):
        b = self.basis
        np.savez_compressed(
            path,
            c=self.c,
            modes=b.modes,
            L=b.L,
            M=b.M,
            N=b.N,
            NFP=b.NFP,
            spectral_indexing=b.spectral_indexing,
            components=np.array(COMPONENTS),
        )

    @classmethod
    def load(cls, path):
        with np.load(path) as f:
            basis = FourierZernikeBasis(
                int(f["L"]), int(f["M"]), int(f["N"]), NFP=int(f["NFP"]),
                sym=False, spectral_indexing=str(f["spectral_indexing"]),
            )
            assert np.array_equal(basis.modes, f["modes"]), "basis mode mismatch"
            return cls(basis, f["c"])
