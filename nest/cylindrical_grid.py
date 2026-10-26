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
    def around(cls, eq, h, margin=2.0, alpha=0.0, extent=None, axisym=None, N_phi=None):
        """Grid with spacing at most h over the plasma, walls margin h beyond it.

        ``N_phi`` (odd): that many planes instead of the arc spacing h, for a J whose
        toroidal harmonics are known to be |n| <= (N_phi - 1) / 2 (the solve is then exact
        mode by mode, as for ``axisym``), e.g. a 3-D perturbation of an axisymmetric
        equilibrium.

        ``axisym`` (default: the equilibrium has N = 0): one plane, N_phi = 1. The solve is
        then exactly two 2-D problems (|m| = 0 for A_z, 1 for A_r + i A_phi), the same as on
        N_phi > 1 planes for an axisymmetric J (gate D3a), at a cost independent of phi.
        Valid only when J is axisymmetric (a current basis with N = 0)."""
        R_min, R_max, Z_min, Z_max = extent or plasma_extent(eq)
        d = margin * h
        ra = ChebAxis.for_spacing(R_min - d, R_max + d, R_min, R_max, h, alpha)
        za = ChebAxis.for_spacing(Z_min - d, Z_max + d, Z_min, Z_max, h, alpha)
        e = eq.eq if hasattr(eq, "eq") else eq
        NFP = int(e.NFP)
        if axisym is None:
            axisym = int(e.N) == 0
        if N_phi is not None:
            N_phi = int(N_phi)
        elif axisym:
            N_phi = 1
        else:
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

    def locate_plasma(self, eq, verbose=False, planes=32, sym=False, **kw):
        """Inside nodes and their flux coordinates, ``planes`` phi-planes per call of
        ``map_to_flux`` (bounded memory), with one shared Newton guess grid.

        ``sym`` (stellarator symmetry, z-symmetric axis): Newton on planes
        0..(N_phi - 1) / 2 only; the nodes of the other planes are the mirrors
        ((-k) mod N_phi, i, n_z - l) of those, with flux coordinates (rho, -theta, -zeta).
        """
        zs = self.z_axis
        if sym and abs(zs.a + zs.b) > 1e-12 * max(abs(zs.a), abs(zs.b)):
            raise ValueError("sym needs a z-symmetric annulus")
        # cached on disk (<repository>/.cache/CylLocate_<hash>.npz, NEST_CACHE=off disables):
        # keyed by the equilibrium file and the grid
        import hashlib
        import os
        cache = None
        where = os.environ.get("NEST_CACHE", os.path.join(os.path.dirname(os.path.dirname(
            os.path.abspath(__file__))), ".cache"))
        path_eq = getattr(eq, "path", None)
        if where != "off" and path_eq is not None and not kw:
            import fit_cache
            key = repr((fit_cache._file_hash(path_eq), round(self.r_axis.a, 12), round(self.r_axis.b, 12),
                        self.r_axis.n, round(self.z_axis.a, 12), round(self.z_axis.b, 12), self.z_axis.n,
                        self.NFP, self.N_phi, bool(sym)))
            cache = os.path.join(where, "CylLocate_" + hashlib.sha256(key.encode()).hexdigest()[:24] + ".npz")
            if os.path.exists(cache):
                with np.load(cache) as f:
                    self.inside, self.rtz = f["inside"], f["rtz"]
                if verbose:
                    print(f"    locate: {len(self.inside):,} inside (cached)", flush=True)
                return self
        import time
        from grid import FluxGrid, _as_input
        inp = _as_input(eq)
        L, M, N = inp.resolution
        guess = kw.pop("guess_grid", None) or FluxGrid.concentric(inp, 2 * L, 2 * M, 2 * N)
        per_plane = self.size // self.N_phi
        inside, rtz, t0 = [], [], time.time()
        n_planes = (self.N_phi - 1) // 2 + 1 if sym else self.N_phi
        for k0 in range(0, n_planes, planes):
            k1 = min(k0 + planes, n_planes)
            idx = np.arange(k0 * per_plane, k1 * per_plane)
            r = map_to_flux(inp, self.cartesian(idx), guess_grid=guess, **kw)
            ok = np.flatnonzero(np.isfinite(r[:, 0]))
            inside.append(idx[ok])
            rtz.append(r[ok])
            if verbose:
                n_in = sum(len(i) for i in inside)
                print(f"    locate: planes {k1}/{n_planes}, {n_in:,} inside, {time.time() - t0:.0f} s",
                      flush=True)
        inside, rtz = np.concatenate(inside), np.concatenate(rtz)
        if sym:  # the mirrors of the nodes on planes 1..(N_phi - 1) / 2
            k, i, l = np.unravel_index(inside, self.shape)
            m = k >= 1
            mi = np.ravel_multi_index(((-k[m]) % self.N_phi, i[m], self.z_axis.n - l[m]), self.shape)
            mr = np.column_stack([rtz[m, 0], np.mod(-rtz[m, 1], 2 * np.pi),
                                  np.mod(-rtz[m, 2], 2 * np.pi / self.NFP)])
            order = np.argsort(np.concatenate([inside, mi]), kind="stable")
            inside = np.concatenate([inside, mi])[order]
            rtz = np.concatenate([rtz, mr])[order]
        self.inside, self.rtz = inside, rtz
        if cache is not None:
            os.makedirs(where, exist_ok=True)
            tmp = cache + f".{os.getpid()}.tmp.npz"
            np.savez(tmp, inside=inside, rtz=rtz)
            os.replace(tmp, cache)
        return self

    def locate_plasma_planes(self, eq, sym=False, rho_out=1.03, n_theta=720, chunk=1 << 20,
                             atol=1e-8, maxiter=16, verbose=False):
        """``locate_plasma`` for grids of many planes, plane by plane: the candidates of
        a plane are its (r, z) nodes inside the surface rho = rho_out there (a polygon
        of n_theta points), their first guess the nearest point of a (rho, theta) mesh
        of that plane (a 2-D tree), then Newton (``grid._newton``) on the device in
        chunks. The same nodes and coordinates as ``locate_plasma`` (inside: Newton
        converged to atol with rho <= 1), without its 3-D tree over every node of the
        plasma's (R, Z) extent. ``sym`` as ``locate_plasma``."""
        import time

        from matplotlib.path import Path
        from scipy.spatial import cKDTree

        import jax.numpy as jnp
        from grid import _as_input, _newton
        zs = self.z_axis
        if sym and abs(zs.a + zs.b) > 1e-12 * max(abs(zs.a), abs(zs.b)):
            raise ValueError("sym needs a z-symmetric annulus")
        inp = _as_input(eq)
        RZ = inp.geometry
        n_planes = (self.N_phi - 1) // 2 + 1 if sym else self.N_phi
        phis = self.phi[:n_planes]
        R2, Z2 = np.meshgrid(self.r_axis.s, self.z_axis.s, indexing="ij")
        pts2 = np.column_stack([R2.ravel(), Z2.ravel()])
        per_plane = len(pts2)
        t0 = time.time()
        th = 2 * np.pi * np.arange(n_theta) / n_theta
        P, T = np.meshgrid(phis, th, indexing="ij")
        poly = np.asarray(RZ.evaluate(jnp.full(P.size, rho_out), jnp.asarray(T.ravel()),
                                      jnp.asarray(P.ravel()))).reshape(n_planes, n_theta, 2)
        rg, tg = np.linspace(0.05, 1.0, 24), 2 * np.pi * np.arange(96) / 96
        Pm, Rm, Tm = np.meshgrid(phis, rg, tg, indexing="ij")
        mesh = np.asarray(RZ.evaluate(jnp.asarray(Rm.ravel()), jnp.asarray(Tm.ravel()),
                                      jnp.asarray(Pm.ravel()))).reshape(n_planes, -1, 2)
        rt_mesh = np.column_stack([Rm[0].ravel(), Tm[0].ravel()])
        inside, rtz = [], []
        buf = {"idx": [], "tgt": [], "z": [], "r0": [], "t0": []}

        def flush(final=False):
            n = sum(len(v) for v in buf["idx"])
            if n == 0 or (n < chunk and not final):
                return
            cat = {k: np.concatenate(v) for k, v in buf.items()}
            for k in buf:
                buf[k] = []
            for i in range(0, n, chunk):
                m = min(chunk, n - i)
                size = min(chunk, 1 << int(np.ceil(np.log2(max(m, 4096)))))  # few compiled sizes
                pad = np.pad(np.arange(i, i + m), (0, size - m), mode="edge")
                rho, theta, res = _newton(RZ, jnp.asarray(cat["tgt"][pad]), jnp.asarray(cat["z"][pad]),
                                          jnp.asarray(cat["r0"][pad]), jnp.asarray(cat["t0"][pad]),
                                          maxiter=maxiter, rho_max=1.3)
                rho, theta, res = (np.asarray(v)[:m] for v in (rho, theta, res))
                ok = (res < atol) & (rho <= 1)
                inside.append(cat["idx"][i : i + m][ok])
                rtz.append(np.column_stack([rho, theta, cat["z"][i : i + m]])[ok])

        for k in range(n_planes):
            b = poly[k]
            lo, hi = b.min(0), b.max(0)
            box = np.flatnonzero((pts2[:, 0] >= lo[0]) & (pts2[:, 0] <= hi[0])
                                 & (pts2[:, 1] >= lo[1]) & (pts2[:, 1] <= hi[1]))
            cand = box[Path(b).contains_points(pts2[box])]
            nn = cKDTree(mesh[k]).query(pts2[cand])[1]
            buf["idx"].append(k * per_plane + cand)
            buf["tgt"].append(pts2[cand])
            buf["z"].append(np.full(len(cand), phis[k]))
            buf["r0"].append(rt_mesh[nn, 0])
            buf["t0"].append(rt_mesh[nn, 1])
            flush()
            if verbose and (k + 1) % max(1, n_planes // 8) == 0:
                print(f"    locate: planes {k + 1}/{n_planes}, "
                      f"{sum(len(i) for i in inside):,} inside, {time.time() - t0:.0f} s", flush=True)
        flush(final=True)
        inside, rtz = np.concatenate(inside), np.concatenate(rtz)
        order = np.argsort(inside)
        inside, rtz = inside[order], rtz[order]
        if sym:  # the mirrors of the nodes on planes 1..(N_phi - 1) / 2 (as locate_plasma)
            k, i, l = np.unravel_index(inside, self.shape)
            m = k >= 1
            mi = np.ravel_multi_index(((-k[m]) % self.N_phi, i[m], self.z_axis.n - l[m]), self.shape)
            mr = np.column_stack([rtz[m, 0], np.mod(-rtz[m, 1], 2 * np.pi),
                                  np.mod(-rtz[m, 2], 2 * np.pi / self.NFP)])
            order = np.argsort(np.concatenate([inside, mi]), kind="stable")
            inside = np.concatenate([inside, mi])[order]
            rtz = np.concatenate([rtz, mr])[order]
        self.inside, self.rtz = inside, rtz
        if verbose:
            print(f"    locate: {len(inside):,} inside, {time.time() - t0:.0f} s", flush=True)
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


def _local_rows(y, nodes, p):
    """Local Lagrange interpolation on the p nodes around each y (ascending nodes):
    indices (n_points, p) and weights (n_points, p), exact at the nodes and
    differentiable in y."""
    n = nodes.shape[0]
    i0 = jnp.clip(jnp.searchsorted(nodes, y) - p // 2, 0, n - p)
    idx = i0[:, None] + jnp.arange(p)
    x = nodes[idx]
    d = y[:, None] - x
    eye = jnp.eye(p, dtype=bool)[None]
    num = jnp.prod(jnp.where(eye, 1.0, d[:, None, :]), axis=2)
    den = jnp.prod(jnp.where(eye, 1.0, x[:, :, None] - x[:, None, :]), axis=2)
    return idx, num / den


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

    _dynamic = ("Ep", "Ez", "yr", "yz", "unsort", "nodes_r", "nodes_z", "w_r", "w_z", "pad_idx",
                "rp", "zp")

    def __init__(self, grid, pts, plane_tol=1e-12, dtype=jnp.float64, order=None):
        pts = np.asarray(pts, dtype=float)
        # ``order`` p: local Lagrange interpolation on the p x p Chebyshev nodes around each
        # point (p^2 per point instead of the global barycentric rows' n_r + n_z and their
        # n_r n_z product per plane); None: global
        self.order = order
        dt = jnp.dtype(dtype)
        cdt = jnp.complex64 if dt == jnp.float32 else jnp.complex128
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
        self.Ep = jnp.asarray(np.exp(1j * np.outer(uniq, jp * NFP)) / N, cdt)
        self.Ez = jnp.asarray(wz * np.exp(1j * np.outer(uniq, jz * NFP)) / N, cdt)
        self.yr = jnp.asarray(grid.r_axis.to_y(pts[pad_idx, 0]), dt)
        self.yz = jnp.asarray(grid.z_axis.to_y(pts[pad_idx, 2]), dt)
        # for moving points (``__call__(..., d)``): the slots' points and positions
        self.pad_idx = jnp.asarray(pad_idx, jnp.int32)
        self.rp = jnp.asarray(pts[pad_idx, 0])
        self.zp = jnp.asarray(pts[pad_idx, 2])
        self.maps = tuple((ax.a, ax.b, ax.alpha) for ax in (grid.r_axis, grid.z_axis))
        self.nodes_r = jnp.asarray(grid.r_axis.y, dt)
        self.w_r = jnp.asarray(grid.r_axis.w_bary, dt)
        self.nodes_z = jnp.asarray(grid.z_axis.y, dt)
        self.w_z = jnp.asarray(grid.z_axis.w_bary, dt)
        self.n, self.n_planes, self.P = len(pts), len(uniq), P

    def _on_plane(self, Pc, Pz, yr, yz):
        """The field of one plane (Pc = B_r + i B_phi, Pz = B_z on the (r, z) nodes) at
        points of Chebyshev coordinates (yr, yz): (P, 3)."""
        if getattr(self, "order", None):
            ir, wr = _local_rows(yr, self.nodes_r, self.order)
            iz, wz = _local_rows(yz, self.nodes_z, self.order)
            gi = (ir[:, :, None], iz[:, None, :])
            c = jnp.einsum("pi,pij,pj->p", wr, Pc.real[gi], wz) + 1j * jnp.einsum(
                "pi,pij,pj->p", wr, Pc.imag[gi], wz)
            z = jnp.einsum("pi,pij,pj->p", wr, Pz[gi], wz)
            return jnp.stack([c.real, c.imag, z], axis=-1)
        Ir = _bary_rows(yr, self.nodes_r, self.w_r)
        Iz = _bary_rows(yz, self.nodes_z, self.w_z)
        c = jnp.sum((Ir @ Pc.real + 1j * (Ir @ Pc.imag)) * Iz, axis=1)
        z = jnp.sum((Ir @ Pz) * Iz, axis=1)
        return jnp.stack([c.real, c.imag, z], axis=-1)

    def _to_y(self, s, axis):
        a, b, alpha = self.maps[axis]
        x = 2 * (s - a) / (b - a) - 1
        return x if alpha == 0.0 else jnp.sin(x * np.arcsin(alpha)) / alpha

    def planes(self, Bp, Bz):
        """The field on the points' planes from the modal slabs (B_+, B_z)."""
        return (jnp.einsum("kj,jab->kab", self.Ep, Bp),
                jnp.real(jnp.einsum("kj,jab->kab", self.Ez, Bz)))

    def __call__(self, Bp, Bz, d=None):
        """The field at the points, or with ``d`` (n, 2) at the points moved by
        (dR, dZ) in their planes (differentiable in d)."""
        return self.from_planes(*self.planes(Bp, Bz), d)

    def from_planes_dd(self, planes_c, planes_z, d, dd):
        """The derivative of ``from_planes(planes_c, planes_z, d)`` in d along dd (n, 2),
        the planes held: they are constants of the scan over planes, not its carry (the
        jvp of the carried scan would keep a copy of all planes per plane for its
        transpose). Linear in dd."""
        import functools
        dp, ddp = d[self.pad_idx], dd[self.pad_idx]
        yr, dyr = jax.jvp(lambda s: self._to_y(s, 0), (self.rp + dp[..., 0],), (ddp[..., 0],))
        yz, dyz = jax.jvp(lambda s: self._to_y(s, 1), (self.zp + dp[..., 1],), (ddp[..., 1],))
        dt = self.yr.dtype
        yr, yz, dyr, dyz = (a.astype(dt) for a in (yr, yz, dyr, dyz))

        @functools.partial(jax.checkpoint, policy=jax.checkpoint_policies.nothing_saveable)
        def one(carry, args):  # the linear inputs as the carry (JAX transposes that form)
            k, yr, yz = args
            dyr = jax.lax.dynamic_index_in_dim(carry[0], k, keepdims=False)
            dyz = jax.lax.dynamic_index_in_dim(carry[1], k, keepdims=False)
            Pc = jax.lax.dynamic_index_in_dim(planes_c, k, keepdims=False)
            Pz = jax.lax.dynamic_index_in_dim(planes_z, k, keepdims=False)

            def val(yr, yz):
                return self._on_plane(Pc, Pz, yr, yz)

            return carry, jax.jvp(val, (yr, yz), (dyr, dyz))[1]

        out = jax.lax.scan(one, (dyr, dyz), (jnp.arange(self.n_planes), yr, yz))[1]
        return out.reshape(-1, 3)[self.unsort].astype(jnp.float64)

    def from_planes(self, planes_c, planes_z, d=None):
        """``__call__`` from the field on the points' planes (B_r + i B_phi, B_z on the
        (r, z) nodes, in the order of ``Ep``; e.g. ``CylindricalSolver.planes_F``)."""
        if d is None:
            yr, yz = self.yr, self.yz
        else:
            dp = d[self.pad_idx]  # (n_planes, P, 2)
            yr = self._to_y(self.rp + dp[..., 0], 0).astype(self.yr.dtype)
            yz = self._to_y(self.zp + dp[..., 1], 1).astype(self.yz.dtype)

        # planes as the scan's carry and the points' coordinates scanned over (the
        # form JAX transposes, as in objective.SpectralInterpolator)
        import functools

        @functools.partial(jax.checkpoint, policy=jax.checkpoint_policies.nothing_saveable)
        def one(carry, args):
            Pcs, Pzs = carry
            k, yr, yz = args
            Pc = jax.lax.dynamic_index_in_dim(Pcs, k, keepdims=False)
            Pz = jax.lax.dynamic_index_in_dim(Pzs, k, keepdims=False)
            return carry, self._on_plane(Pc, Pz, yr, yz)

        out = jax.lax.scan(one, (planes_c, planes_z),
                           (jnp.arange(self.n_planes), yr, yz))[1]  # (n_planes, P, 3)
        return out.reshape(-1, 3)[self.unsort].astype(jnp.float64)
