"""The external field a DESC equilibrium implies for NEST's discrete Biot-Savart.

    B_ext = B_DESC - BS_h(J_DESC)

B_DESC is DESC's total field, J_DESC its current density sampled on the box and
BS_h the box Biot-Savart operator (``biot_savart.BiotSavart``) with the same box
and kernel as the solve. With this B_ext, B = B_ext + BS_h(J) equals B_DESC exactly
for J = J_DESC at every box resolution, so the discretization of B_plasma acts only
on J - J_DESC: a verification setting (e.g. fixed-boundary equilibria, or checking
that the starting point is stationary), not a physical external field. It is
defined only inside the plasma, where B_DESC is.

``DESCExternalField`` has the interface NEST uses for an external field
(``compute_magnetic_field``, as DESC's fields), so it is passed as the ``field``
of a ``DESCInput``:

    box = CartesianGrid.periodic_box(inp, h).locate_plasma(inp)
    inp_v = DESCInput(inp.eq, field=DESCExternalField(inp, box), name=...)
"""

import numpy as np

from magnetic_field import BiotSavart
from cartesian_grid import cylindrical_to_cartesian
from grid import map_to_flux


def _xyz_to_rpz(V, phi):
    c, s = np.cos(phi), np.sin(phi)
    return np.column_stack([V[:, 0] * c + V[:, 1] * s, -V[:, 0] * s + V[:, 1] * c, V[:, 2]])


class DESCExternalField:
    """B_DESC - BS_h(J_DESC) at points inside the plasma.

    Parameters
    ----------
    inp : DESCInput
        The equilibrium (its field, if any, is not used).
    box : CartesianGrid
        Periodic box with the plasma located; the same box as the solve.
    current : object with ``point_evaluator()`` or None
        J sampled on the box; None for DESC's own J (``DESCInput.compute_points``).
    kernel, dtype, eps, upsampfac, cache_dir : passed to ``BiotSavart``.
    """

    def __init__(self, inp, box, current=None, kernel="vico_greengard", dtype=np.float64,
                 eps=1e-10, upsampfac=1.25, cache_dir=None, chunk=20000):
        self.inp, self.box = inp, box
        self.bs_kw = dict(kernel=kernel, dtype=dtype, eps=eps, upsampfac=upsampfac,
                          cache_dir=cache_dir)
        if current is None:
            J = inp.compute_points(["J"], box.rtz, chunk=chunk)["J"]
        else:
            from cartesian_grid import evaluate_chunked
            J = evaluate_chunked(current.point_evaluator(inp.eq), box.rtz, chunk)
        self.J_box = cylindrical_to_cartesian(J, box.phi_inside)  # (n_inside, 3)

    def at_flux(self, rtz, x):
        """B_ext at points with known flux coordinates ``rtz`` and positions ``x``,
        in Cartesian components, shape (n, 3)."""
        rtz, x = np.asarray(rtz, dtype=float), np.asarray(x, dtype=float)
        phi = np.arctan2(x[:, 1], x[:, 0])
        B_desc = cylindrical_to_cartesian(self.inp.compute_points(["B"], rtz)["B"], phi)
        return B_desc - np.asarray(BiotSavart(self.box, x, **self.bs_kw).apply(self.J_box))

    def compute_magnetic_field(self, coords, params=None, basis="rpz", source_grid=None,
                               transforms=None, chunk_size=None):
        """B_ext at points ``coords`` (n, 3) in ``basis`` "xyz" or "rpz", in the same basis.

        The flux coordinates of the points are found by ``grid.map_to_flux``; points
        outside the plasma raise, since B_DESC is not defined there.
        """
        coords = np.asarray(coords, dtype=float)
        if basis == "rpz":
            x = np.column_stack([coords[:, 0] * np.cos(coords[:, 1]),
                                 coords[:, 0] * np.sin(coords[:, 1]), coords[:, 2]])
        else:
            x = coords
        rtz = map_to_flux(self.inp, x)
        outside = ~np.isfinite(rtz[:, 0])
        if outside.any():
            raise ValueError(f"DESCExternalField: {outside.sum()} of {len(x)} points are "
                             "outside the plasma, where B_DESC is not defined")
        B = self.at_flux(rtz, x)
        return _xyz_to_rpz(B, np.arctan2(x[:, 1], x[:, 0])) if basis == "rpz" else B
