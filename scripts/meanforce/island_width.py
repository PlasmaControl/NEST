"""Width of the 2/1 island chain from Poincare crossings (no DESC / JAX: usable on a login node)."""

import numpy as np


def width(r, ts, m=2, rho_c=None, band=None):
    """rho extent of the lines trapped in the m/1 chain (isl_islands.py's rule) and the flags. With
    rho_c, band: only lines whose mean rho is within band of rho_c count (a line elsewhere whose m theta*
    happens to stay within 2 pi, e.g. near the axis, made the floor state's width 0.43)."""
    trapped = np.zeros(r.shape[1], bool)
    for j in range(r.shape[1]):
        al = m * ts[:, j][np.isfinite(ts[:, j])]
        if len(al) < 20:
            continue
        if rho_c is not None and abs(np.nanmean(r[:, j]) - rho_c) > band:
            continue
        un = np.concatenate([[0.0], np.cumsum(np.mod(np.diff(al) + np.pi, 2 * np.pi) - np.pi)])
        trapped[j] = un.max() - un.min() < 2 * np.pi
    if not trapped.any():
        return 0.0, trapped
    rt = r[:, trapped]
    return float(np.nanmax(rt) - np.nanmin(rt)), trapped
