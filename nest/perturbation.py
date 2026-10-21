"""A resonant error field for the tokamak: one helical surface current on the coils'
own winding surface.

The tokamak's external field is a ``FourierCurrentPotentialField`` (an axisymmetric
winding surface 0.8 m outside the LCFS, ``desc_input/tokamak_free_boundary``). The
perturbation is a second current potential on the same surface,

    Phi(theta, zeta) = sin(m theta - n zeta)     [A per unit amplitude],

with no net currents (I = G = 0). Its field is curl- and divergence-free inside the
winding surface and has the single toroidal harmonic n; it is stellarator symmetric
(Phi odd), as is the tokamak. B_ext = B_coils + eps B_pert, eps in A.

The resonant helicity: field lines follow theta* - iota zeta = const, so (m, n)
resonates where iota = n / m (iota < 0 for this tokamak: n = -1 for q = 2).
"""

import numpy as np
from desc.grid import LinearGrid


def helical_potential(field, m, n):
    """A copy of the surface-current field ``field`` with Phi = sin(m theta - n zeta)
    and I = G = 0 (coefficients found by least squares on the field's own basis, so
    DESC's sign conventions for the modes do not matter)."""
    p = field.copy()
    p.change_Phi_resolution(M=max(abs(m), int(field.M_Phi)), N=max(abs(n), int(field.N_Phi)))
    th = 2 * np.pi * np.arange(4 * abs(m) + 8) / (4 * abs(m) + 8)
    ze = 2 * np.pi * np.arange(4 * abs(n) + 8) / (4 * abs(n) + 8) / p.NFP
    T, Z = np.meshgrid(th, ze, indexing="ij")
    nodes = np.column_stack([np.zeros(T.size), T.ravel(), Z.ravel()])
    A = np.asarray(p.Phi_basis.evaluate(nodes))
    c, *_ = np.linalg.lstsq(A, np.sin(m * T.ravel() - n * p.NFP * Z.ravel()), rcond=None)
    if np.abs(A @ c - np.sin(m * T.ravel() - n * p.NFP * Z.ravel())).max() > 1e-12:
        raise ValueError(f"sin({m} theta - {n} zeta) is not in the field's Phi basis ({p.sym_Phi})")
    c[np.abs(c) < 1e-14] = 0.0
    p.Phi_mn = c
    p.I, p.G = 0.0, 0.0
    return p


class Perturbation:
    """B_pert per unit amplitude at Cartesian points (chunked, as ``DESCInput.B_ext``)."""

    def __init__(self, field, m=2, n=-1):
        self.m, self.n = int(m), int(n)
        self.field = helical_potential(field, m, n)
        f = self.field
        self.source_grid = LinearGrid(M=max(64, 4 * f.M_Phi, 4 * f.M), N=max(128, 4 * f.N_Phi, 4 * f.N),
                                      NFP=f.NFP)

    def __call__(self, x, chunk_size=2000):
        """(B_x, B_y, B_z) [T per A] at Cartesian points x (n, 3)."""
        return np.asarray(self.field.compute_magnetic_field(
            np.asarray(x, dtype=float), basis="xyz", source_grid=self.source_grid, chunk_size=chunk_size))

    def __repr__(self):
        return f"Perturbation(Phi = sin({self.m} theta - ({self.n}) zeta) A on the coils' winding surface)"
