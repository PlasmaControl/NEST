"""Step 1 of NEST: evaluate the current density on the flux-aligned grid G1.

J comes from any current with ``evaluate(grid)`` (e.g. ``boundary_potential.
BoundaryPotential``); the geometry X(rho, theta, zeta) comes from the DESC
equilibrium.

"""

import numpy as np

from desc_input import DESCInput, equilibrium
from grid import FluxGrid


def flux_grid(eq, L, M, N, node_pattern="cheb1", axis=False):
    """Concentric grid G1 in (rho, theta, zeta) over one field period (a DESC grid).

    "cheb1" puts surfaces at rho = 1 and, with axis=False, just off the axis
    (DESC's J is undefined at rho = 0), clustered toward both ends. See
    ``grid.FluxGrid.concentric``, which this wraps.
    """
    return FluxGrid.concentric(eq, L, M, N, node_pattern, axis).desc


def evaluate_on_flux_grid(eq, grid, current):
    """Positions from ``eq`` and J from the series ``current``, full torus.

    ``grid`` is a one-period DESC grid or a ``FluxGrid``.
    """
    fg = grid if isinstance(grid, FluxGrid) else FluxGrid(grid)
    NFP = equilibrium(eq).NFP
    if fg.NFP != NFP:
        raise ValueError(f"grid.NFP={fg.NFP} must equal eq.NFP={NFP}")
    inp = eq if isinstance(eq, DESCInput) else DESCInput(eq, check=False)
    d = inp.compute(["p", "sqrt(g)"], fg.desc)
    pos = fg.positions(inp)
    nodes, phi = fg.torus_nodes(), pos["phi"]
    J_cyl = fg.tile(current.evaluate(fg.desc))

    c, s = np.cos(phi), np.sin(phi)
    J = np.stack(
        [J_cyl[:, 0] * c - J_cyl[:, 1] * s,
         J_cyl[:, 0] * s + J_cyl[:, 1] * c,
         J_cyl[:, 2]],
        axis=-1,
    )

    return {
        "rho": nodes[:, 0],
        "theta": nodes[:, 1],
        "zeta": nodes[:, 2],
        "x": pos["x"],
        "J": J,
        "J_cyl": J_cyl,
        "R": pos["R"],
        "Z": pos["Z"],
        "phi": phi,
        "p": fg.tile(d["p"]),
        "sqrt_g": fg.tile(d["sqrt(g)"]),
        "weights": fg.torus_weights(),
        "NFP": np.asarray(NFP),
    }


def save(data, path):
    np.savez_compressed(path, **data)


def load(path):
    with np.load(path) as f:
        return {k: f[k] for k in f.files}
