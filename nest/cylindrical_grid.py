"""The annulus grid of the cylindrical J -> B solver, and the plasma on it.

Nodes (r_i, phi_k, z_l): Chebyshev in r on [R_in, R_out] and in z on
[z_lo, z_hi], N_phi (odd) uniform planes phi_k = 2 pi k / (NFP N_phi) over one
field period. ``around`` sizes the grid for a spacing h: the Chebyshev spacing
is at most h over the plasma's (R, Z) extent, the walls are ``margin`` h beyond
it, and the arc spacing at R_out is at most h.

``locate_plasma`` finds, once, the nodes inside the plasma and their flux
coordinates (``grid.map_to_flux``, 2-D Newton on each plane); fields known there
are put on the grid with ``scatter`` (zero elsewhere).

``evaluate`` takes the FFT-index slabs of a field (``CylindricalSolver.field_modes``)
to arbitrary points (r, phi, z): exact Fourier synthesis in phi (on each distinct
phi of the points) and barycentric Chebyshev interpolation in (r, z).
"""

import numpy as np

from chebyshev import ChebAxis
from grid import map_to_flux, plasma_extent


class CylindricalGrid:
    """Annulus grid over one field period."""

    def __init__(self, r_axis, z_axis, NFP, N_phi):
        if N_phi % 2 == 0:
            raise ValueError("N_phi must be odd")
        self.r_axis, self.z_axis, self.NFP, self.N_phi = r_axis, z_axis, int(NFP), int(N_phi)
        self.inside = self.rtz = None

    @classmethod
    def around(cls, eq, h, margin=2.0, alpha=0.0, extent=None):
        """Grid with spacing at most h over the plasma, walls margin h beyond it."""
        R_min, R_max, Z_min, Z_max = extent or plasma_extent(eq)
        d = margin * h
        ra = ChebAxis.for_spacing(R_min - d, R_max + d, R_min, R_max, h, alpha)
        za = ChebAxis.for_spacing(Z_min - d, Z_max + d, Z_min, Z_max, h, alpha)
        NFP = int(eq.NFP if hasattr(eq, "NFP") else eq.eq.NFP)
        N_phi = int(np.ceil(2 * np.pi * (R_max + d) / (NFP * h)))
        N_phi += 1 - N_phi % 2
        g = cls(ra, za, NFP, N_phi)
        g.h, g.margin = h, margin
        return g

    @property
    def shape(self):
        return (self.N_phi, self.r_axis.n + 1, self.z_axis.n + 1)

    @property
    def size(self):
        return int(np.prod(self.shape))

    @property
    def phi(self):
        return 2 * np.pi * np.arange(self.N_phi) / (self.NFP * self.N_phi)

    def nodes(self):
        """(r, phi, z) of every node, shape shape + (3,)."""
        P, R, Z = np.meshgrid(self.phi, self.r_axis.s, self.z_axis.s, indexing="ij")
        return np.stack([R, P, Z], axis=-1)

    def cartesian(self, flat_index=None):
        n = self.nodes().reshape(-1, 3)
        n = n if flat_index is None else n[flat_index]
        return np.column_stack([n[:, 0] * np.cos(n[:, 1]), n[:, 0] * np.sin(n[:, 1]), n[:, 2]])

    def locate_plasma(self, eq, verbose=False, **kw):
        rtz = map_to_flux(eq, self.cartesian(), verbose=verbose, **kw)
        ok = np.flatnonzero(np.isfinite(rtz[:, 0]))
        self.inside, self.rtz = ok, rtz[ok]
        return self

    def scatter(self, values, dtype=np.float64):
        values = np.asarray(values)
        out = np.zeros((self.size,) + values.shape[1:], dtype=dtype)
        out[self.inside] = values
        return out.reshape(self.shape + values.shape[1:])

    def __repr__(self):
        return (f"CylindricalGrid(r [{self.r_axis.a:.4f}, {self.r_axis.b:.4f}] n {self.r_axis.n + 1}, "
                f"z [{self.z_axis.a:.4f}, {self.z_axis.b:.4f}] n {self.z_axis.n + 1}, "
                f"N_phi {self.N_phi}, NFP {self.NFP}: {self.size / 1e6:.2f}M nodes)")


def evaluate(grid, Bp, Bz, pts, plane_tol=1e-12, chunk=4096):
    """Field at points pts (n, 3) = (r, phi, z) from the FFT-index slabs.

    Bp: (N_phi, n_r + 1, n_z + 1) slabs of V_r + i V_phi (full FFT order);
    Bz: (N_phi // 2 + 1, ...) rfft slabs of V_z. Returns (n, 3) cylindrical
    components. Points sharing a phi share the Fourier synthesis.
    """
    Bp, Bz = np.asarray(Bp), np.asarray(Bz)
    N, NFP = grid.N_phi, grid.NFP
    jp = np.fft.fftfreq(N, 1.0 / N)
    jz = np.arange(N // 2 + 1)
    wz = np.where(jz == 0, 1.0, 2.0)
    pts = np.asarray(pts, dtype=float)
    out = np.empty((len(pts), 3))
    key = np.round(np.mod(pts[:, 1], 2 * np.pi / NFP) / plane_tol) * plane_tol
    uniq, inv = np.unique(key, return_inverse=True)
    for u, phi in enumerate(uniq):
        sel = np.flatnonzero(inv == u)
        ep = np.exp(1j * jp * NFP * phi) / N
        ez = wz * np.exp(1j * jz * NFP * phi) / N
        Pc = np.tensordot(ep, Bp, axes=(0, 0))          # V_r + i V_phi on the plane
        Pz = np.real(np.tensordot(ez, Bz, axes=(0, 0)))
        for i in range(0, len(sel), chunk):
            s = sel[i:i + chunk]
            Ir = grid.r_axis.interp_matrix(pts[s, 0])
            Iz = grid.z_axis.interp_matrix(pts[s, 2])
            c = np.einsum("pb,pb->p", Ir @ Pc, Iz)
            out[s, 0], out[s, 1] = c.real, c.imag
            out[s, 2] = np.einsum("pb,pb->p", Ir @ Pz, Iz)
    return out
