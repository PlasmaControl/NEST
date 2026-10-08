"""Steps 2-3 of NEST: the current density on the periodic Cartesian box G3.

G3 is a uniform periodic grid, x_p = x_0 + p h with p = 0..N-1 and period
L = N h, about ``factor`` times larger than the plasma in each direction.
For each point x of the block of G3 that covers the plasma, the flux
coordinates (rho, theta, zeta) with X(rho, theta, zeta) = x are found by
Newton iteration and J is evaluated there from its Fourier-Zernike series.
Points with no solution rho <= 1 are outside the plasma, Omega_p =
{rho(x) <= 1}, and, like the rest of the box, get J = 0 (zero padding).
"""

import time

import jax.numpy as jnp
import numpy as np

from grid import CartesianGrid, map_to_flux, plasma_extent, progress  # noqa: F401
from desc_input import equilibrium

# re-exported: ``plasma_extent`` and ``map_to_flux`` live in grid.py


def cartesian_grid(eq, h, factor=2.0):
    """Axes (xs, ys, zs) of the periodic box G3 with spacing h around the plasma.

    See ``CartesianGrid.periodic_box``, which this wraps.
    """
    return CartesianGrid.periodic_box(eq, h, factor).axes


def current_on_cartesian(eq, current, axes, guess_grid=None, newton_chunk=200000,
                         eval_chunk=50000, dtype=np.float32, verbose=False, **kw):
    """J on the Cartesian grid ``axes``, zero outside the plasma.

    ``axes`` is (xs, ys, zs) or a ``CartesianGrid``; the plasma is located on it
    (``CartesianGrid.locate_plasma``) unless that was already done. Returns a
    dict with
        J       Cartesian components of J, shape (Nx, Ny, Nz, 3)    [A/m^2]
        mask    chi_{Omega_p}, shape (Nx, Ny, Nz): True for rho <= 1
        inside  flat indices of the points inside the plasma
        rtz     their flux coordinates (rho, theta, zeta), (n_inside, 3)
        grid    the ``CartesianGrid``
    plus the axes xs, ys, zs and spacings dx. ``newton_chunk`` and
    ``eval_chunk`` are the numbers of points per call of the inverse map and of
    the series evaluation; ``verbose`` prints progress. The inverse map and the
    series are computed in float64; J on the box is stored as ``dtype``
    (float32 by default, half the memory of the largest array).
    """
    grid = axes if isinstance(axes, CartesianGrid) else CartesianGrid(axes)
    if grid.inside is None:
        grid.locate_plasma(eq, guess_grid=guess_grid, newton_chunk=newton_chunk,
                           verbose=verbose, **kw)
    J_cyl = evaluate_chunked(current.point_evaluator(equilibrium(eq)), grid.rtz, eval_chunk,
                             verbose and f"J series ({current.basis.num_modes} modes)")
    J = grid.scatter(cylindrical_to_cartesian(J_cyl, grid.phi_inside), dtype)
    return {
        "xs": grid.xs,
        "ys": grid.ys,
        "zs": grid.zs,
        "dx": grid.dx,
        "J": J,
        "mask": grid.mask,
        "inside": grid.inside,
        "rtz": grid.rtz,
        "grid": grid,
    }


def evaluate_chunked(evaluate, rtz, chunk=50000, label=None):
    """``evaluate(rho, theta, zeta)`` on the rows of ``rtz``, in chunks.

    Every call gets exactly ``chunk`` points (the last chunk is padded), so a
    jitted ``evaluate`` compiles once. ``label`` (a string) prints progress.
    Returns the stacked outputs, shape (len(rtz), ...).
    """
    chunk = min(chunk, max(len(rtz), 1))
    n_chunks, t0, out = -(-len(rtz) // chunk), time.time(), []
    for k, i in enumerate(range(0, len(rtz), chunk)):
        n = min(chunk, len(rtz) - i)
        pad = np.pad(np.arange(i, i + n), (0, chunk - n), mode="edge")
        out.append(np.asarray(evaluate(*(jnp.asarray(rtz[pad, j]) for j in range(3))))[:n])
        if label and ((k + 1) % 20 == 0 or k + 1 == n_chunks):
            progress(label, k + 1, n_chunks, i + n, len(rtz), t0, "")
    return np.concatenate(out)


def cylindrical_to_cartesian(V, phi):
    """(V_R, V_phi, V_Z) at angles phi -> (V_x, V_y, V_z)."""
    c, s = np.cos(phi), np.sin(phi)
    return np.column_stack([V[:, 0] * c - V[:, 1] * s, V[:, 0] * s + V[:, 1] * c, V[:, 2]])


def pressure_on_cartesian(eq, pressure, d, eval_chunk=50000, dtype=np.float32):
    """p and grad p on the box of ``d = current_on_cartesian(...)``, zero outside.

    Reuses the flux coordinates of the inside points from ``d``, so only the
    series is evaluated. Returns p, shape (Nx, Ny, Nz) [Pa], and grad p in
    Cartesian components, shape (Nx, Ny, Nz, 3) [Pa/m].
    """
    shape = d["mask"].shape
    X = [a[i] for a, i in zip((d["xs"], d["ys"], d["zs"]), np.unravel_index(d["inside"], shape))]
    out = evaluate_chunked(pressure.point_evaluator(equilibrium(eq)), d["rtz"], eval_chunk)
    p = np.zeros(shape, dtype=dtype)
    p.ravel()[d["inside"]] = out[:, 0]
    grad_p = np.zeros(shape + (3,), dtype=dtype)
    grad_p.reshape(-1, 3)[d["inside"]] = cylindrical_to_cartesian(out[:, 1:], np.arctan2(X[1], X[0]))
    return p, grad_p
