"""Step 2 of NEST: the current density on a uniform Cartesian grid G2.

For each Cartesian point x the flux coordinates (rho, theta, zeta) with
X(rho, theta, zeta) = x are found by Newton iteration, and J is evaluated 
there from its Fourier-Zernike series. Points where no solution with 
rho <= 1 exists are outside the plasma, Omega_p = {rho(x) <= 1},
and get J = 0.
"""

import functools
import time

import jax
import jax.numpy as jnp
import numpy as np
from scipy.spatial import cKDTree

from desc.basis import fourier, zernike_radial
from desc.grid import LinearGrid

from desc_grid import flux_grid


def plasma_extent(eq, M=None, N=None):
    """Bounds of the plasma boundary: (R_min, R_max, Z_min, Z_max)."""
    M = 4 * eq.M + 1 if M is None else M
    N = 4 * eq.N + 1 if N is None else N
    g = LinearGrid(rho=1.0, theta=M, zeta=N, NFP=eq.NFP, sym=False)
    d = eq.compute(["R", "Z"], grid=g)
    R, Z = np.asarray(d["R"]), np.asarray(d["Z"])
    return R.min(), R.max(), Z.min(), Z.max()


def cartesian_grid(eq, N=None, h=None, pad=0.0):
    """Uniform grid G2 covering the plasma.

    x and y span [-R_max, R_max], z spans [Z_min, Z_max], each widened by
    ``pad`` metres on both sides. Give either
      N = (Nx, Ny, Nz): points including both ends, x_p = x_min + p dx with
          dx = (x_max - x_min) / (Nx - 1), or
      h: the same spacing dx = dy = dz = h in all directions; the number of
          points is the smallest odd number covering the span (odd, so the
          grid has a plane through the box centre, e.g. y = 0), centred on it.
    """
    _, R_max, Z_min, Z_max = plasma_extent(eq)
    lo = np.array([-R_max, -R_max, Z_min]) - pad
    hi = np.array([R_max, R_max, Z_max]) + pad
    if (N is None) == (h is None):
        raise ValueError("give exactly one of N and h")
    if N is not None:
        return tuple(np.linspace(lo[i], hi[i], N[i]) for i in range(3))
    n = np.ceil((hi - lo) / h).astype(int) // 2 * 2 + 1  # intervals, odd points
    n = np.where((n - 1) * h < hi - lo, n + 2, n)
    c = (lo + hi) / 2
    return tuple(c[i] + h * (np.arange(n[i]) - (n[i] - 1) / 2) for i in range(3))


class _Separable:
    """Fourier-Zernike series regrouped by the toroidal factor.

    A series f(rho, theta, zeta) = sum_k c_k Z_k(rho) T_k(theta) F_k(zeta) in
    DESC's basis is stored as a tensor C[p, q, c] over the distinct (l, m)
    pairs p, the distinct n values q and the components c, so that

        f_c = sum_p P_p(rho, theta) sum_q C[p, q, c] F_q(zeta),

    with P_p = Z_l^|m|(rho) T_m(theta). At fixed zeta the inner sum collapses
    to a 2-D Zernike series with coefficients a[p, c] = sum_q C[p, q, c] F_q.
    """

    def __init__(self, bases, coefs, NFP):
        modes = np.concatenate([b.modes for b in bases])
        self.lm, p = np.unique(modes[:, :2], axis=0, return_inverse=True)
        self.n, q = np.unique(modes[:, 2], return_inverse=True)
        self.NFP = NFP
        C = np.zeros((len(self.lm), len(self.n), len(bases)))
        k0 = 0
        for c, (b, coef) in enumerate(zip(bases, coefs)):
            k = slice(k0, k0 + b.num_modes)
            C[p.ravel()[k], q.ravel()[k], c] = np.asarray(coef)
            k0 += b.num_modes
        self.C = jnp.asarray(C)

    def poloidal(self, rho, theta, dr=0, dt=0):
        """P_p(rho, theta) and derivatives, shape (n_points, n_lm)."""
        l, m = self.lm[:, 0], self.lm[:, 1]
        return zernike_radial(rho[:, None], l, m, dr=dr) * fourier(theta[:, None], m, dt=dt)

    def toroidal(self, zeta):
        """F_q(zeta), shape (n_points, n_n)."""
        return fourier(zeta[:, None], self.n, NFP=self.NFP)

    def collapse(self, zeta):
        """2-D coefficients a[i, p, c] at each point's zeta."""
        return jnp.einsum("iq,pqc->ipc", self.toroidal(zeta), self.C)

    def evaluate(self, rho, theta, zeta):
        """f_c at the points, shape (n_points, n_components)."""
        # (P C) first, then the toroidal factor: never forms (points, p, q, c)
        PC = jnp.einsum("ip,pqc->iqc", self.poloidal(rho, theta), self.C)
        return jnp.einsum("iqc,iq->ic", PC, self.toroidal(zeta))


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

    RZ = _Separable([eq.R_basis, eq.Z_basis], [eq.R_lmn, eq.Z_lmn], eq.NFP)
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


@functools.partial(jax.jit, static_argnums=0)
def _evaluate(series, rho, theta, zeta):
    return series.evaluate(rho, theta, zeta)


def current_on_cartesian(eq, current, axes, guess_grid=None, newton_chunk=200000,
                         eval_chunk=20000, verbose=False, **kw):
    """J on the Cartesian grid ``axes = (xs, ys, zs)``, zero outside the plasma.

    Returns a dict with arrays of shape (Nx, Ny, Nz[, 3]):
        J     Cartesian components of J                             [A/m^2]
        mask  chi_{Omega_p}: True inside the plasma (rho <= 1)
        rtz   flux coordinates (rho, theta, zeta), NaN outside
    plus the axes xs, ys, zs and spacings dx. ``newton_chunk`` and
    ``eval_chunk`` are the numbers of points per call of the inverse map and of
    the series evaluation; ``verbose`` prints progress.
    """
    xs, ys, zs = axes
    X = np.stack(np.meshgrid(xs, ys, zs, indexing="ij"), axis=-1)
    x = X.reshape(-1, 3)
    rtz = map_to_flux(eq, x, guess_grid=guess_grid, chunk=newton_chunk,
                      verbose=verbose, **kw)
    inside = np.flatnonzero(np.isfinite(rtz[:, 0]))

    J = np.zeros_like(x)
    series = _Separable([current.basis] * 3, [current.c[:, i] for i in range(3)],
                        current.basis.NFP)
    phi = np.arctan2(x[inside, 1], x[inside, 0])
    eval_chunk = min(eval_chunk, max(inside.size, 1))
    n_chunks, t0 = -(-inside.size // eval_chunk), time.time()
    for k, i in enumerate(range(0, inside.size, eval_chunk)):
        sl = inside[i : i + eval_chunk]
        pad = np.pad(np.arange(sl.size), (0, eval_chunk - sl.size), mode="edge")
        y = rtz[sl[pad]]
        J_cyl = np.asarray(_evaluate(series, *(jnp.asarray(y[:, j]) for j in range(3))))
        J_cyl = J_cyl[: sl.size]
        c, s = np.cos(phi[i : i + sl.size]), np.sin(phi[i : i + sl.size])
        J[sl] = np.column_stack(
            [J_cyl[:, 0] * c - J_cyl[:, 1] * s,
             J_cyl[:, 0] * s + J_cyl[:, 1] * c,
             J_cyl[:, 2]]
        )
        if verbose and ((k + 1) % 20 == 0 or k + 1 == n_chunks):
            _progress("J series", k + 1, n_chunks, i + sl.size, inside.size, t0,
                      f"{current.basis.num_modes} modes")

    shape = X.shape[:3]
    return {
        "xs": xs,
        "ys": ys,
        "zs": zs,
        "dx": np.array([a[1] - a[0] if len(a) > 1 else np.nan for a in axes]),
        "J": J.reshape(shape + (3,)),
        "mask": np.isfinite(rtz[:, 0]).reshape(shape),
        "rtz": rtz.reshape(shape + (3,)),
    }
