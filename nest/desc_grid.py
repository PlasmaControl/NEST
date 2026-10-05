"""Step 1 of NEMO: evaluate the current density on the flux-aligned grid G1.

J is stored as a Fourier-Zernike series (see ``current_fz.CurrentFZ``) and
evaluated from it; the geometry X(rho, theta, zeta) comes from the DESC
equilibrium.

"""

import numpy as np

from desc.grid import ConcentricGrid


def flux_grid(eq, L, M, N, node_pattern="jacobi", axis=False):
    """Concentric grid G1 in (rho, theta, zeta) over one field period.

    """
    return ConcentricGrid(
        L, M, N, NFP=eq.NFP, sym=False, axis=axis, node_pattern=node_pattern
    )


def evaluate_on_flux_grid(eq, grid, current):
    """Positions from ``eq`` and J from the series ``current``, full torus.

    """
    NFP = eq.NFP
    if grid.NFP != NFP:
        raise ValueError(f"grid.NFP={grid.NFP} must equal eq.NFP={NFP}")
    names = ["R", "phi", "Z", "p", "sqrt(g)"]
    d = {k: np.asarray(v) for k, v in eq.compute(names, grid=grid).items()}
    J_cyl = current.evaluate(grid)

    shift = np.repeat(2 * np.pi * np.arange(NFP) / NFP, grid.num_nodes)
    tile = lambda f: np.tile(f, (NFP,) + (1,) * (np.ndim(f) - 1))
    nodes = tile(np.asarray(grid.nodes))
    nodes[:, 2] += shift
    R, Z, phi, J_cyl = tile(d["R"]), tile(d["Z"]), tile(d["phi"]) + shift, tile(J_cyl)

    c, s = np.cos(phi), np.sin(phi)
    x = np.stack([R * c, R * s, Z], axis=-1)
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
        "x": x,
        "J": J,
        "J_cyl": J_cyl,
        "R": R,
        "Z": Z,
        "phi": phi,
        "p": tile(d["p"]),
        "sqrt_g": tile(d["sqrt(g)"]),
        "weights": tile(np.asarray(grid.weights)) / NFP,
        "NFP": np.asarray(NFP),
    }


def save(data, path):
    np.savez_compressed(path, **data)


def load(path):
    with np.load(path) as f:
        return {k: f[k] for k in f.files}
