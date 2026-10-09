"""The two grids of NEST: the flux grid G1 and the Cartesian box G3.

G1 (``FluxGrid``) is a DESC ``ConcentricGrid`` in the flux coordinates
(rho, theta, zeta) of the starting equilibrium over one field period, with its
copies around the torus and their positions x = X(rho, theta, zeta).

G3 (``CartesianGrid``) is a uniform grid x_p = x_0 + p h, p = 0..N - 1 in each
direction; built by ``periodic_box`` it is a periodic box with period L = N h
about ``factor`` times the plasma, for the FFT. ``locate_plasma`` finds, once,
the points inside the plasma Omega_p = {rho(x) <= 1} and their flux
coordinates, by inverting X with ``map_to_flux``; fields known at those points
are then put on the box with ``scatter`` (zero outside: the zero padding) and
read back with ``gather``.

Every function takes either a ``DESCInput`` or a bare DESC ``Equilibrium``.
"""

import functools
import time

import jax
import jax.numpy as jnp
import numpy as np
from scipy.fft import next_fast_len
from scipy.spatial import cKDTree

from desc.grid import ConcentricGrid, LinearGrid

from desc_input import DESCInput, equilibrium
from fourier import nyquist, wavevectors


def _as_input(obj):
    return obj if isinstance(obj, DESCInput) else DESCInput(obj, check=False)


def plasma_extent(eq, M=None, N=None):
    """Bounds of the plasma boundary: (R_min, R_max, Z_min, Z_max)."""
    return _as_input(eq).extent(M, N)


def progress(stage, k, n, done, total, t0, extra="", unit="points"):
    """One line of progress: chunk k of n, items (``unit``) done, elapsed time and ETA."""
    dt = time.time() - t0
    eta = dt / done * (total - done)
    extra = f"{extra}, " if extra else ""
    print(f"  {stage} chunk {k}/{n}: {done:,}/{total:,} {unit}, {extra}"
          f"{dt:.0f}s elapsed, ETA {eta:.0f}s", flush=True)


# ----------------------------------------------------------------------- G1
class FluxGrid:
    """A DESC grid over one field period and its NFP copies around the torus.

    Parameters
    ----------
    grid : desc.grid.Grid
        Nodes over one field period (``grid.NFP`` = the equilibrium's NFP).
    """

    def __init__(self, grid):
        self.desc = grid
        self.NFP = int(grid.NFP)

    @classmethod
    def concentric(cls, eq, L, M, N, node_pattern="cheb1", axis=False):
        """``ConcentricGrid`` over one period.

        "cheb1" puts surfaces at rho = 1 and, with axis=False, just off the axis
        (DESC's J is undefined at rho = 0), clustered toward both ends.
        """
        return cls(ConcentricGrid(L, M, N, NFP=equilibrium(eq).NFP, sym=False,
                                  axis=axis, node_pattern=node_pattern))

    @property
    def num_nodes(self):
        """Nodes per field period."""
        return self.desc.num_nodes

    def tile(self, f):
        """Values at the one-period nodes copied to the NFP periods (period-major)."""
        f = np.asarray(f)
        return np.tile(f, (self.NFP,) + (1,) * (f.ndim - 1))

    def _shift(self):
        return np.repeat(2 * np.pi * np.arange(self.NFP) / self.NFP, self.num_nodes)

    def torus_nodes(self):
        """(rho, theta, zeta) over the full torus, shape (NFP num_nodes, 3)."""
        nodes = self.tile(self.desc.nodes)
        nodes[:, 2] += self._shift()
        return nodes

    def torus_weights(self):
        """Quadrature weights over the full torus (they sum to those of one period)."""
        return self.tile(self.desc.weights) / self.NFP

    def positions(self, eq):
        """R, phi, Z and Cartesian x = (R cos phi, R sin phi, Z) over the full torus."""
        d = _as_input(eq).compute(["R", "phi", "Z"], self.desc)
        R, Z, phi = self.tile(d["R"]), self.tile(d["Z"]), self.tile(d["phi"]) + self._shift()
        x = np.column_stack([R * np.cos(phi), R * np.sin(phi), Z])
        return {"R": R, "phi": phi, "Z": Z, "x": x}


# -------------------------------------------------------------- inverse map
@functools.partial(jax.jit, static_argnames=("RZ", "maxiter", "rho_max"))
def _newton(RZ, target, zeta, rho, theta, maxiter, rho_max):
    """Solve (R, Z)(rho, theta, zeta) = target for (rho, theta) at fixed zeta.

    Newton in u = rho cos(theta), v = rho sin(theta), which, unlike (rho,
    theta), are regular at the magnetic axis. A fixed number of steps on all
    points at once; rho may exceed 1 (up to rho_max) so that points just
    outside converge instead of stalling at the boundary. Returns rho, theta
    and the residual |(R, Z) - target| in metres.
    """
    a = RZ.collapse(zeta)

    def residual(rho, theta):
        return jnp.einsum("ip,ipc->ic", RZ.poloidal(rho, theta), a) - target

    def step(_, uv):
        u, v = uv
        rho, theta = jnp.hypot(u, v), jnp.arctan2(v, u)
        c, s = jnp.cos(theta), jnp.sin(theta)
        f = residual(rho, theta)
        f_r = jnp.einsum("ip,ipc->ic", RZ.poloidal(rho, theta, dr=1), a)
        # (1/rho) df/dtheta is finite at the axis: every m != 0 term has rho^|m|
        f_t = jnp.einsum("ip,ipc->ic", RZ.poloidal(rho, theta, dt=1), a)
        f_t = f_t / jnp.maximum(rho, 1e-100)[:, None]
        f_u = f_r * c[:, None] - f_t * s[:, None]
        f_v = f_r * s[:, None] + f_t * c[:, None]
        det = f_u[:, 0] * f_v[:, 1] - f_v[:, 0] * f_u[:, 1]
        det = jnp.where(jnp.abs(det) > 1e-14, det, 1e-14)
        du = (f_v[:, 1] * f[:, 0] - f_v[:, 0] * f[:, 1]) / det
        dv = (-f_u[:, 1] * f[:, 0] + f_u[:, 0] * f[:, 1]) / det
        scale = jnp.minimum(1, 0.2 / jnp.maximum(jnp.hypot(du, dv), 1e-300))
        u, v = u - scale * du, v - scale * dv
        r = jnp.hypot(u, v)
        shrink = jnp.minimum(1, rho_max / jnp.maximum(r, 1e-300))
        return u * shrink, v * shrink

    u, v = jax.lax.fori_loop(0, maxiter, step, (rho * jnp.cos(theta), rho * jnp.sin(theta)))
    rho, theta = jnp.hypot(u, v), jnp.mod(jnp.arctan2(v, u), 2 * jnp.pi)
    return rho, theta, jnp.linalg.norm(residual(rho, theta), axis=1)


def map_to_flux(eq, x, guess_grid=None, atol=1e-8, maxiter=20, rho_max=1.3,
                chunk=200000, verbose=False):
    """Flux coordinates (rho, theta, zeta) of Cartesian points x, shape (n, 3).

    Rows are NaN for points outside the plasma. Since zeta = phi = atan2(y, x)
    is known, only (rho, theta) are solved for, by Newton on
    (R, Z)(rho, theta, phi) = (R*, Z*) with the equilibrium series collapsed
    to 2-D at each point's phi. Points outside the boundary's (R, Z) extent
    are skipped; the rest start from the nearest node of ``guess_grid`` (a
    DESC grid or ``FluxGrid`` over one period, default a flux grid at twice the
    equilibrium resolution). A point is inside when Newton converges,
    |(R, Z) - (R*, Z*)| < atol metres, with rho <= 1.

    Points are solved in chunks of ``chunk``; the last chunk is padded so the
    solver compiles only once.
    """
    inp = _as_input(eq)
    inp.check()
    x = np.asarray(x, dtype=float)
    R = np.hypot(x[:, 0], x[:, 1])
    phi = np.mod(np.arctan2(x[:, 1], x[:, 0]), 2 * np.pi)
    R_min, R_max, Z_min, Z_max = inp.extent()
    cand = np.flatnonzero(
        (R >= R_min) & (R <= R_max) & (x[:, 2] >= Z_min) & (x[:, 2] <= Z_max)
    )
    if verbose:
        print(f"map_to_flux: {cand.size:,} of {len(x):,} points within the plasma's "
              "(R, Z) extent", flush=True)
    rtz = np.full_like(x, np.nan)
    if cand.size == 0:
        return rtz

    # initial guess: nearest node of a flux grid over the full torus
    L, M, N = inp.resolution
    if guess_grid is None:
        guess_grid = FluxGrid.concentric(inp, 2 * L, 2 * M, 2 * N)
    elif not isinstance(guess_grid, FluxGrid):
        guess_grid = FluxGrid(guess_grid)
    nodes = guess_grid.torus_nodes()
    _, nn = cKDTree(guess_grid.positions(inp)["x"]).query(x[cand])

    RZ = inp.geometry
    chunk = min(chunk, cand.size)
    n_chunks = -(-cand.size // chunk)
    t0 = time.time()
    for k, i in enumerate(range(0, cand.size, chunk)):
        idx = cand[i : i + chunk]
        pad = np.pad(np.arange(idx.size), (0, chunk - idx.size), mode="edge")
        sel = idx[pad]
        rho, theta, res = _newton(
            RZ, jnp.column_stack([R[sel], x[sel, 2]]), jnp.asarray(phi[sel]),
            jnp.asarray(nodes[nn[i + pad], 0]), jnp.asarray(nodes[nn[i + pad], 1]),
            maxiter=maxiter, rho_max=rho_max,
        )
        rho, theta, res = (np.asarray(v)[: idx.size] for v in (rho, theta, res))
        ok = (res < atol) & (rho <= 1)
        rtz[idx[ok]] = np.column_stack([rho, theta, phi[idx]])[ok]
        if verbose:
            progress("Newton", k + 1, n_chunks, i + idx.size, cand.size, t0,
                     f"{ok.sum():,} of {idx.size:,} inside")
    return rtz


# ----------------------------------------------------------------------- G3
class CartesianGrid:
    """Uniform Cartesian grid with axes (xs, ys, zs), and the plasma inside it.

    Parameters
    ----------
    axes : tuple of three 1-D arrays
        Equally spaced coordinates [m]; an axis may have a single point.
    periodic : bool
        Whether the grid is a period of the box used by the FFT.
    """

    def __init__(self, axes, periodic=False):
        self.axes = tuple(np.asarray(a, dtype=float) for a in axes)
        if len(self.axes) != 3:
            raise ValueError("axes must be (xs, ys, zs)")
        for a in self.axes:
            if len(a) > 2 and not np.allclose(np.diff(a), a[1] - a[0], rtol=1e-9, atol=0):
                raise ValueError("axes must be equally spaced")
        self.periodic = periodic
        self.inside = self.rtz = None

    @classmethod
    def periodic_box(cls, eq, h, factor=2.0):
        """Periodic box G3 with spacing h around the plasma.

        In each direction the period L = N h is at least ``factor`` times the
        plasma extent (2 R_max in x and y, Z_max - Z_min in z), with N rounded
        up to an FFT-friendly size (small prime factors). The box is centred on
        the plasma, x_p = c + h (p - N // 2), so it has a point at the centre,
        e.g. the plane y = 0.
        """
        _, R_max, Z_min, Z_max = plasma_extent(eq)
        c = np.array([0.0, 0.0, (Z_min + Z_max) / 2])
        extent = np.array([2 * R_max, 2 * R_max, Z_max - Z_min])
        n = [next_fast_len(int(np.ceil(factor * e / h))) for e in extent]
        return cls(tuple(c[i] + h * (np.arange(n[i]) - n[i] // 2) for i in range(3)),
                   periodic=True)

    # ------------------------------------------------------------ geometry
    @property
    def xs(self):
        return self.axes[0]

    @property
    def ys(self):
        return self.axes[1]

    @property
    def zs(self):
        return self.axes[2]

    @property
    def shape(self):
        return tuple(len(a) for a in self.axes)

    @property
    def size(self):
        return int(np.prod(self.shape))

    @property
    def dx(self):
        """Spacings (dx, dy, dz) [m]; NaN along a single-point axis."""
        return np.array([a[1] - a[0] if len(a) > 1 else np.nan for a in self.axes])

    @property
    def dV(self):
        """Volume of a cell [m^3] (NaN if an axis has a single point)."""
        return float(np.prod(self.dx))

    @property
    def period(self):
        """(Lx, Ly, Lz) = N h of the periodic box [m]."""
        return np.array([len(a) * (a[1] - a[0]) for a in self.axes])

    def points(self, flat_index=None):
        """Cartesian points (x, y, z) of the grid, or of the given flat indices."""
        if flat_index is None:
            flat_index = np.arange(self.size)
        i = np.unravel_index(flat_index, self.shape)
        return np.column_stack([a[k] for a, k in zip(self.axes, i)])

    def wavevectors(self):
        """(kx, ky, kz) of the rfft on this grid (``fourier.wavevectors``)."""
        return wavevectors(self.axes)

    def nyquist(self):
        """Nyquist wavenumbers pi / dx_i and the largest |k|."""
        return nyquist(self.axes)

    # -------------------------------------------------------------- plasma
    def locate_plasma(self, eq, guess_grid=None, newton_chunk=200000, verbose=False,
                      **kw):
        """Points inside the plasma and their flux coordinates (once per grid).

        Only the block of the grid within the plasma's (R, Z) extent is solved
        for (``map_to_flux``); the rest is never formed as points. Sets
        ``inside`` (flat indices of the points with rho <= 1) and ``rtz``
        (their (rho, theta, zeta)), and returns the grid.
        """
        _, R_max, Z_min, Z_max = plasma_extent(eq)
        block = [np.flatnonzero(np.abs(self.xs) <= R_max),
                 np.flatnonzero(np.abs(self.ys) <= R_max),
                 np.flatnonzero((self.zs >= Z_min) & (self.zs <= Z_max))]
        x = np.stack(np.meshgrid(*(a[b] for a, b in zip(self.axes, block)), indexing="ij"),
                     axis=-1).reshape(-1, 3)
        rtz = map_to_flux(eq, x, guess_grid=guess_grid, chunk=newton_chunk,
                          verbose=verbose, **kw)
        ok = np.flatnonzero(np.isfinite(rtz[:, 0]))
        self.inside = np.ravel_multi_index(
            [b[i] for b, i in zip(block, np.unravel_index(ok, [len(b) for b in block]))],
            self.shape)
        self.rtz = rtz[ok]
        return self

    def _require_plasma(self):
        if self.inside is None:
            raise RuntimeError("call locate_plasma first")

    @property
    def mask(self):
        """chi_{Omega_p} on the grid: True for rho <= 1."""
        self._require_plasma()
        m = np.zeros(self.shape, dtype=bool)
        m.ravel()[self.inside] = True
        return m

    @property
    def phi_inside(self):
        """Cylindrical angle of the inside points (= zeta)."""
        self._require_plasma()
        return self.rtz[:, 2]

    @property
    def plasma_volume(self):
        """Number of inside points times dV [m^3] (a staircase estimate)."""
        self._require_plasma()
        return len(self.inside) * self.dV

    def scatter(self, values, dtype=np.float32):
        """Values at the inside points onto the full grid, zero elsewhere.

        ``values`` has shape (n_inside, ...); the result (Nx, Ny, Nz, ...).
        """
        self._require_plasma()
        values = np.asarray(values)
        out = np.zeros(self.shape + values.shape[1:], dtype=dtype)
        out.reshape((-1,) + values.shape[1:])[self.inside] = values
        return out

    def gather(self, field):
        """A field on the grid, shape (Nx, Ny, Nz, ...), at the inside points."""
        self._require_plasma()
        field = np.asarray(field)
        return field.reshape((-1,) + field.shape[3:])[self.inside]
