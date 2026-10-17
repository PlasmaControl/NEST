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
        """Cartesian positions of all nodes, or of the given flat indices (without
        forming the full node array)."""
        if flat_index is None:
            n = self.nodes().reshape(-1, 3)
        else:
            k, i, l = np.unravel_index(flat_index, self.shape)
            n = np.column_stack([self.r_axis.s[i], self.phi[k], self.z_axis.s[l]])
        return np.column_stack([n[:, 0] * np.cos(n[:, 1]), n[:, 0] * np.sin(n[:, 1]), n[:, 2]])

    def locate_plasma(self, eq, verbose=False, planes=32, **kw):
        """Inside nodes and their flux coordinates, ``planes`` phi-planes per call of
        ``map_to_flux`` (bounded memory), with one shared Newton guess grid."""
        import time
        from grid import FluxGrid, _as_input
        inp = _as_input(eq)
        L, M, N = inp.resolution
        guess = kw.pop("guess_grid", None) or FluxGrid.concentric(inp, 2 * L, 2 * M, 2 * N)
        per_plane = self.size // self.N_phi
        inside, rtz, t0 = [], [], time.time()
        for k0 in range(0, self.N_phi, planes):
            k1 = min(k0 + planes, self.N_phi)
            idx = np.arange(k0 * per_plane, k1 * per_plane)
            r = map_to_flux(inp, self.cartesian(idx), guess_grid=guess, **kw)
            ok = np.flatnonzero(np.isfinite(r[:, 0]))
            inside.append(idx[ok])
            rtz.append(r[ok])
            if verbose:
                n_in = sum(len(i) for i in inside)
                print(f"    locate: planes {k1}/{self.N_phi}, {n_in:,} inside, {time.time() - t0:.0f} s",
                      flush=True)
        self.inside, self.rtz = np.concatenate(inside), np.concatenate(rtz)
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


# ---------------------------------------------------------------- JAX interpolator
import jax  # noqa: E402
import jax.numpy as jnp  # noqa: E402

from flux_interp import _Pytree  # noqa: E402


def _bary_rows(y, nodes, w):
    """Barycentric Lagrange rows l_j(y) (n_points, n_nodes), exact at the nodes."""
    d = y[:, None] - nodes[None, :]
    hit = d == 0
    c = w[None, :] / jnp.where(hit, 1.0, d)
    c = c / jnp.sum(c, axis=1, keepdims=True)
    return jnp.where(jnp.any(hit, axis=1, keepdims=True), hit.astype(c.dtype), c)


class CylindricalInterpolator(_Pytree):
    """The field of ``CylindricalSolver.field_modes`` at fixed points (r, phi, z).

    Points are grouped by plane (their phi modulo the period): on each plane the
    exact Fourier synthesis of the modal slabs gives the (r, z) node values, and
    each point takes the barycentric Chebyshev interpolant of them. The barycentric
    rows are formed on the fly, plane by plane (``lax.map``), so the cache is only
    the points' Chebyshev coordinates and the synthesis weights. Linear in the
    slabs, so JAX's reverse mode (and its transpose) apply. Returns cylindrical
    components (V_r, V_phi, V_z), shape (n, 3).
    """

    _dynamic = ("Ep", "Ez", "yr", "yz", "unsort", "nodes_r", "nodes_z", "w_r", "w_z")

    def __init__(self, grid, pts, plane_tol=1e-12):
        pts = np.asarray(pts, dtype=float)
        N, NFP = grid.N_phi, grid.NFP
        period = 2 * np.pi / NFP
        key = np.round(np.mod(pts[:, 1], period) / plane_tol) * plane_tol
        uniq, inv = np.unique(key, return_inverse=True)
        counts = np.bincount(inv)
        P = int(counts.max())
        order = np.argsort(inv, kind="stable")
        # padded slots: plane k, slot s; padding repeats the plane's last point
        starts = np.concatenate([[0], np.cumsum(counts)[:-1]])
        slot = np.empty(len(pts), int)
        slot[order] = np.arange(len(pts)) - np.repeat(starts, counts)
        pad_idx = np.empty((len(uniq), P), int)
        for k in range(len(uniq)):
            idx = order[starts[k]:starts[k] + counts[k]]
            pad_idx[k] = np.concatenate([idx, np.full(P - counts[k], idx[-1])])
        self.unsort = jnp.asarray(inv * P + slot, jnp.int32)  # point -> flat padded slot
        jp = np.fft.fftfreq(N, 1.0 / N)
        jz = np.arange(N // 2 + 1)
        wz = np.where(jz == 0, 1.0, 2.0)
        self.Ep = jnp.asarray(np.exp(1j * np.outer(uniq, jp * NFP)) / N)
        self.Ez = jnp.asarray(wz * np.exp(1j * np.outer(uniq, jz * NFP)) / N)
        self.yr = jnp.asarray(grid.r_axis.to_y(pts[pad_idx, 0]))
        self.yz = jnp.asarray(grid.z_axis.to_y(pts[pad_idx, 2]))
        self.nodes_r, self.w_r = jnp.asarray(grid.r_axis.y), jnp.asarray(grid.r_axis.w_bary)
        self.nodes_z, self.w_z = jnp.asarray(grid.z_axis.y), jnp.asarray(grid.z_axis.w_bary)
        self.n, self.n_planes, self.P = len(pts), len(uniq), P

    def __call__(self, Bp, Bz):
        planes_c = jnp.einsum("kj,jab->kab", self.Ep, Bp)
        planes_z = jnp.real(jnp.einsum("kj,jab->kab", self.Ez, Bz))

        # planes as the scan's carry and the points' coordinates scanned over (the
        # form JAX transposes, as in objective.SpectralInterpolator)
        import functools

        @functools.partial(jax.checkpoint, policy=jax.checkpoint_policies.nothing_saveable)
        def one(carry, args):
            Pcs, Pzs = carry
            k, yr, yz = args
            Pc = jax.lax.dynamic_index_in_dim(Pcs, k, keepdims=False)
            Pz = jax.lax.dynamic_index_in_dim(Pzs, k, keepdims=False)
            Ir = _bary_rows(yr, self.nodes_r, self.w_r)
            Iz = _bary_rows(yz, self.nodes_z, self.w_z)
            c = jnp.sum((Ir @ Pc.real + 1j * (Ir @ Pc.imag)) * Iz, axis=1)
            z = jnp.sum((Ir @ Pz) * Iz, axis=1)
            return carry, jnp.stack([c.real, c.imag, z], axis=-1)

        out = jax.lax.scan(one, (planes_c, planes_z),
                           (jnp.arange(self.n_planes), self.yr, self.yz))[1]  # (n_planes, P, 3)
        return out.reshape(-1, 3)[self.unsort]
