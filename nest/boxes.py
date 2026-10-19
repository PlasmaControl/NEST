"""The B_plasma grid of a run: main's periodic Cartesian box or the cylindrical
annulus (``cylindrical_grid``), with the plasma located on it (as optimize_qa.py)."""

from grid import CartesianGrid


def make_box(inp, frac, grid="cartesian", sym=True, margin=2.0, verbose=False, N_phi=None):
    """The grid at h = a / frac, plasma located (``N_phi``: the annulus' planes, see
    ``CylindricalGrid.around``)."""
    if grid == "cylindrical":
        from cylindrical_grid import CylindricalGrid
        g = CylindricalGrid.around(inp, inp.a / frac, margin, N_phi=N_phi)
        zs = g.z_axis
        sym_z = abs(zs.a + zs.b) <= 1e-12 * max(abs(zs.a), abs(zs.b))
        return g.locate_plasma(inp, verbose=verbose, sym=sym and sym_z)
    return CartesianGrid.periodic_box(inp, inp.a / frac, odd=sym).locate_plasma(inp)
