"""Steps 2-3 of NEST: the current density on the periodic Cartesian box G3.

G3 is a uniform periodic grid, x_p = x_0 + p h with p = 0..N-1 and period
L = N h, about ``factor`` times larger than the plasma in each direction.
For each point x of the block of G3 that covers the plasma, the flux
coordinates (rho, theta, zeta) with X(rho, theta, zeta) = x are found by
Newton iteration and J is evaluated there from its Fourier-Zernike series.
Points with no solution rho <= 1 are outside the plasma, Omega_p =
{rho(x) <= 1}, and, like the rest of the box, get J = 0 (zero padding).
"""

import functools
import time

import jax
import jax.numpy as jnp
import numpy as np
from scipy.fft import next_fast_len
from scipy.spatial import cKDTree

from desc.grid import LinearGrid

from desc_grid import flux_grid
from separable import Separable


def plasma_extent(eq, M=None, N=None):
    """Bounds of the plasma boundary: (R_min, R_max, Z_min, Z_max)."""
    M = 4 * eq.M + 1 if M is None else M
    N = 4 * eq.N + 1 if N is None else N
    g = LinearGrid(rho=1.0, theta=M, zeta=N, NFP=eq.NFP, sym=False)
    d = eq.compute(["R", "Z"], grid=g)
    R, Z = np.asarray(d["R"]), np.asarray(d["Z"])
    return R.min(), R.max(), Z.min(), Z.max()


def cartesian_grid(eq, h, factor=2.0):
    """Periodic Cartesian box G3 with spacing h around the plasma.

    In each direction the period L = N h is at least ``factor`` times the
    plasma extent (2 R_max in x and y, Z_max - Z_min in z), with N rounded up
    to an FFT-friendly size (small prime factors). The box is centred on the
    plasma, x_p = c + h (p - N // 2), so it has a point at the centre, e.g.
    the plane y = 0. Returns the axes (xs, ys, zs).
    """
    _, R_max, Z_min, Z_max = plasma_extent(eq)
    c = np.array([0.0, 0.0, (Z_min + Z_max) / 2])
    extent = np.array([2 * R_max, 2 * R_max, Z_max - Z_min])
    n = [next_fast_len(int(np.ceil(factor * e / h))) for e in extent]
    return tuple(c[i] + h * (np.arange(n[i]) - n[i] // 2) for i in range(3))


def _check_phi_is_zeta(eq):
    """The 2-D inverse map uses zeta = phi; true when DESC's omega = 0."""
    g = LinearGrid(rho=np.array([0.5, 1.0]), M=4, N=4, NFP=eq.NFP, sym=False)
    d = eq.compute(["phi", "zeta"], grid=g)
    err = np.max(np.abs(np.asarray(d["phi"]) - np.asarray(d["zeta"])))
    if err > 1e-12:
        raise NotImplementedError(f"phi != zeta (max diff {err:.1e}): omega != 0")


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
    flux grid G1, default one at twice the equilibrium resolution). A point
    is inside when Newton converges, |(R, Z) - (R*, Z*)| < atol metres, with
    rho <= 1.

    Points are solved in chunks of ``chunk``; the last chunk is padded so the
    solver compiles only once.
    """
    _check_phi_is_zeta(eq)
    x = np.asarray(x, dtype=float)
    R = np.hypot(x[:, 0], x[:, 1])
    phi = np.mod(np.arctan2(x[:, 1], x[:, 0]), 2 * np.pi)
    R_min, R_max, Z_min, Z_max = plasma_extent(eq)
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
    if guess_grid is None:
        guess_grid = flux_grid(eq, 2 * eq.L, 2 * eq.M, 2 * eq.N)
    nodes, xg = _torus_nodes(eq, guess_grid)
    _, nn = cKDTree(xg).query(x[cand])

    RZ = Separable([eq.R_basis, eq.Z_basis], [eq.R_lmn, eq.Z_lmn], eq.NFP)
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
            _progress("Newton", k + 1, n_chunks, i + idx.size, cand.size, t0,
                      f"{ok.sum():,} of {idx.size:,} inside")
    return rtz


def _progress(stage, k, n, done, total, t0, extra=""):
    """One line of progress: chunk k of n, points done, elapsed time and ETA."""
    dt = time.time() - t0
    eta = dt / done * (total - done)
    print(f"  {stage} chunk {k}/{n}: {done:,}/{total:,} points, {extra}, "
          f"{dt:.0f}s elapsed, ETA {eta:.0f}s", flush=True)


def _torus_nodes(eq, grid):
    """Nodes of a one-period ``grid`` copied around the torus, with positions."""
    shift = np.repeat(2 * np.pi * np.arange(eq.NFP) / eq.NFP, grid.num_nodes)
    nodes = np.tile(np.asarray(grid.nodes), (eq.NFP, 1))
    nodes[:, 2] += shift
    d = eq.compute(["R", "phi", "Z"], grid=grid)
    R, Z = np.tile(np.asarray(d["R"]), eq.NFP), np.tile(np.asarray(d["Z"]), eq.NFP)
    phi = np.tile(np.asarray(d["phi"]), eq.NFP) + shift
    return nodes, np.column_stack([R * np.cos(phi), R * np.sin(phi), Z])


def current_on_cartesian(eq, current, axes, guess_grid=None, newton_chunk=200000,
                         eval_chunk=50000, dtype=np.float32, verbose=False, **kw):
    """J on the Cartesian grid ``axes = (xs, ys, zs)``, zero outside the plasma.

    Only the block of the grid inside the plasma's (R, Z) extent is solved
    for; the rest of the grid (the zero padding of the box) is never formed
    as points. Returns a dict with
        J       Cartesian components of J, shape (Nx, Ny, Nz, 3)    [A/m^2]
        mask    chi_{Omega_p}, shape (Nx, Ny, Nz): True for rho <= 1
        inside  flat indices of the points inside the plasma
        rtz     their flux coordinates (rho, theta, zeta), (n_inside, 3)
    plus the axes xs, ys, zs and spacings dx. ``newton_chunk`` and
    ``eval_chunk`` are the numbers of points per call of the inverse map and of
    the series evaluation; ``verbose`` prints progress. The inverse map and the
    series are computed in float64; J on the box is stored as ``dtype``
    (float32 by default, half the memory of the largest array).
    """
    shape = tuple(len(a) for a in axes)
    _, R_max, Z_min, Z_max = plasma_extent(eq)
    block = [np.flatnonzero(np.abs(axes[0]) <= R_max),
             np.flatnonzero(np.abs(axes[1]) <= R_max),
             np.flatnonzero((axes[2] >= Z_min) & (axes[2] <= Z_max))]
    x = np.stack(np.meshgrid(*(a[b] for a, b in zip(axes, block)), indexing="ij"),
                 axis=-1).reshape(-1, 3)
    rtz = map_to_flux(eq, x, guess_grid=guess_grid, chunk=newton_chunk,
                      verbose=verbose, **kw)
    ok = np.flatnonzero(np.isfinite(rtz[:, 0]))
    x, rtz = x[ok], rtz[ok]
    inside = np.ravel_multi_index(
        [b[i] for b, i in zip(block, np.unravel_index(ok, [len(b) for b in block]))],
        shape)

    evaluate = current.point_evaluator(eq)
    J_in = np.empty_like(x)
    phi = np.arctan2(x[:, 1], x[:, 0])
    eval_chunk = min(eval_chunk, max(len(x), 1))
    n_chunks, t0 = -(-len(x) // eval_chunk), time.time()
    for k, i in enumerate(range(0, len(x), eval_chunk)):
        sl = slice(i, i + eval_chunk)
        n = len(x[sl])
        pad = np.pad(np.arange(i, i + n), (0, eval_chunk - n), mode="edge")
        J_cyl = np.asarray(evaluate(*(jnp.asarray(rtz[pad, j]) for j in range(3))))[:n]
        c, s = np.cos(phi[sl]), np.sin(phi[sl])
        J_in[sl] = np.column_stack([J_cyl[:, 0] * c - J_cyl[:, 1] * s,
                                    J_cyl[:, 0] * s + J_cyl[:, 1] * c,
                                    J_cyl[:, 2]])
        if verbose and ((k + 1) % 20 == 0 or k + 1 == n_chunks):
            _progress("J series", k + 1, n_chunks, i + n, len(x), t0,
                      f"{current.basis.num_modes} modes")

    J = np.zeros(shape + (3,), dtype=dtype)
    J.reshape(-1, 3)[inside] = J_in
    mask = np.zeros(shape, dtype=bool)
    mask.ravel()[inside] = True
    return {
        "xs": axes[0],
        "ys": axes[1],
        "zs": axes[2],
        "dx": np.array([a[1] - a[0] if len(a) > 1 else np.nan for a in axes]),
        "J": J,
        "mask": mask,
        "inside": inside,
        "rtz": rtz,
    }
