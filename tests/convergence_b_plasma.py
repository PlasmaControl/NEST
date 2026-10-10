"""Convergence of B_plasma with h: periodic FFT, Hockney-Eastwood, truncated kernel.

J = curl T (``BoundaryPotential``, J = 0 on the LCFS) on the box G3 with
h = a / frac for several fracs, all multiples of the first, so that every point
of the coarsest box is a point of every box (the boxes are centred on the same
point). B_plasma by the three methods of ``magnetic_field``, compared at fixed
points of space:

  exterior  max |B - B_ref| / max |B_ref| at n_ext points of the coarsest box
            0.3 a to 0.6 a from the plasma and inside the region where every
            box sees no images (B_ref: continuous Biot-Savart of the same J,
            Gauss quadrature in flux coordinates at max(potential resolution,
            3 times the equilibrium's); its difference with 2/3 of that is
            printed and must be well below the errors).
  interior  the same error at the plasma points of the coarsest box, against
            the truncated-kernel B of the finest box (self-convergence; the
            continuous integral is singular there), also in bands of rho.

Errors are max |B - B_ref| / max |B_ref| and the rms of |B - B_ref| / max |B_ref|.
Prints a table per method and saves the errors and the sampled B to ``--out``
(npz).

run: python tests/convergence_b_plasma.py [--eq qa|w7x] [--edge-power 2] [--fracs 5 10 20]
     [--factors 3 3 2.5] [--out errors.npz]
"""

import argparse
import os
import sys
import time

import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, "nest"))
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

# sets the DESC device (NEST_DEVICE) before any other DESC import
from test_magnetic_field import EQ_FILES, continuous_biot_savart, rel, timed  # noqa: E402

import desc.io  # noqa: E402
from scipy.spatial import cKDTree  # noqa: E402

from boundary_potential import BoundaryPotential  # noqa: E402
from cartesian_grid import current_on_cartesian  # noqa: E402
from fourier import fft_J  # noqa: E402
from grid import CartesianGrid, plasma_extent  # noqa: E402
from magnetic_field import (  # noqa: E402
    b_plasma_free,
    b_plasma_periodic,
    b_plasma_truncated,
    biot_savart_kernel_hat,
    truncated_green_hat,
)

METHODS = ("periodic", "hockney", "truncated")
RHO_BANDS = [(0, 0.5), (0.5, 0.8), (0.8, 0.95), (0.95, 1.0)]


def rms_rel(B, B_ref):
    return np.sqrt(np.mean(np.sum((B - B_ref) ** 2, axis=1))) / np.abs(B_ref).max()


def print_table(title, fracs, rows):
    """rows: {name: errors per frac}; with the observed order between columns."""
    print(f"\n{title:>22} " + "".join(f"{'a/' + str(f):>18}" for f in fracs))
    for name, e in rows.items():
        order = [""] + [f"({np.log(e[i - 1] / e[i]) / np.log(fracs[i] / fracs[i - 1]):.1f})"
                        if e[i] > 0 and e[i - 1] > 0 else "" for i in range(1, len(e))]
        print(f"{name:>22} " + "".join(f"{v:>11.2e} {o:>6}" for v, o in zip(e, order)))


def box_indices(grid, x):
    """Flat indices of the points x in ``grid`` (which must contain them)."""
    idx = [np.rint((x[:, i] - a[0]) / (a[1] - a[0])).astype(int) for i, a in enumerate(grid.axes)]
    flat = np.ravel_multi_index(idx, grid.shape)
    assert np.allclose(grid.points(flat), x, atol=1e-9 * np.abs(x).max()), "grids not nested"
    return flat


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--eq", default="qa", choices=["qa", "w7x"])
    ap.add_argument("--edge-power", type=int, default=2)
    ap.add_argument("--fracs", type=int, nargs="+", default=[5, 10, 20])
    ap.add_argument("--factors", type=float, nargs=3, default=(3, 3, 2.5),
                    help="L, M, N of the potential as multiples of the equilibrium's")
    ap.add_argument("--n-ext", type=int, default=200)
    ap.add_argument("--out", default=None)
    args = ap.parse_args()
    fracs = sorted(args.fracs)
    assert all(f % fracs[0] == 0 for f in fracs), "fracs must be multiples of the first"

    path = [p for p in EQ_FILES if ("W7-X" in p) == (args.eq == "w7x")][0]
    eq = desc.io.load(path)
    eq = eq[-1] if hasattr(eq, "__len__") else eq
    a = float(eq.compute("a")["a"])
    t0 = time.time()
    current = BoundaryPotential.fit(eq, factors=args.factors, edge_power=args.edge_power)
    print(f"{os.path.basename(path)}: a = {a:.4f} m, BoundaryPotential L, M, N = "
          f"{current.basis.L}, {current.basis.M}, {current.basis.N}, s = {args.edge_power} "
          f"(fit {time.time() - t0:.0f} s)", flush=True)

    # region with no images for every box: within P / 2 of every plasma point
    _, R_max, Z_min, Z_max = plasma_extent(eq)
    lo_p, hi_p = np.array([-R_max, -R_max, Z_min]), np.array([R_max, R_max, Z_max])
    periods = np.array([CartesianGrid.periodic_box(eq, a / f).period for f in fracs])
    lo, hi = hi_p - periods.min(axis=0) / 2, lo_p + periods.min(axis=0) / 2

    rng = np.random.default_rng(0)
    B = {m: [] for m in METHODS}
    for k, frac in enumerate(fracs):
        grid = CartesianGrid.periodic_box(eq, a / frac)
        t0 = time.time()
        G_hat, t_G = timed(truncated_green_hat, grid.axes)
        d = current_on_cartesian(eq, current, grid)
        t_J = time.time() - t0 - t_G
        if k == 0:  # the fixed sample points, from the coarsest box
            X_in, rho_in = grid.points(d["inside"]), d["rtz"][:, 0]
            box = np.flatnonzero(np.all((grid.points() > lo) & (grid.points() < hi), axis=1))
            dist, _ = cKDTree(X_in).query(grid.points(box), distance_upper_bound=0.65 * a)
            shell = box[(dist >= 0.3 * a) & (dist <= 0.6 * a)]
            X_ext = grid.points(rng.choice(shell, min(args.n_ext, len(shell)), replace=False))
            t1 = time.time()
            res = np.maximum([current.basis.L, current.basis.M, current.basis.N],
                             3 * np.array([eq.L, eq.M, eq.N]))
            res2 = tuple(int(v) for v in np.ceil(2 * res / 3))
            res = tuple(int(v) for v in res)
            B_ref2 = continuous_biot_savart(eq, current, X_ext, LMN=res2)
            B_ref = continuous_biot_savart(eq, current, X_ext, LMN=res)
            print(f"{len(X_in):,} interior and {len(X_ext)} exterior points "
                  f"(of {len(shell):,}); continuous Biot-Savart at L, M, N = {res} "
                  f"{time.time() - t1:.0f} s, vs {res2}: max {rel(B_ref2, B_ref):.1e}",
                  flush=True)
        pts = box_indices(grid, np.concatenate([X_in, X_ext]))
        J_hat = fft_J(d["J"])
        del d
        times = {}
        # truncated first, so that its kernel is freed before Hockney-Eastwood's
        # (three complex arrays of the box) is built
        for name in ("truncated", "periodic", "hockney"):
            if name == "truncated":
                out, times[name] = timed(b_plasma_truncated, J_hat, grid.axes, G_hat)
                del G_hat
            elif name == "periodic":
                out, times[name] = timed(b_plasma_periodic, J_hat, grid.axes)
            else:
                K_hat = biot_savart_kernel_hat(grid.axes)
                out, times[name] = timed(b_plasma_free, J_hat, grid.axes, K_hat)
                del K_hat
            B[name].append(out.reshape(-1, 3)[pts].astype(np.float64))
            del out
        del J_hat
        print(f"h = a/{frac}: box {grid.shape} ({grid.size / 1e6:.0f}M); J on the box "
              f"{t_J:.0f} s, truncated kernel {t_G:.1f} s; B: "
              + ", ".join(f"{m} {t:.2f} s" for m, t in times.items()), flush=True)

    n_in = len(X_in)
    ref_in = B["truncated"][-1][:n_in]
    scale_in = np.abs(ref_in).max()
    err = {}
    for m in METHODS:
        err[f"{m}_exterior_max"] = np.array([rel(b[n_in:], B_ref) for b in B[m]])
        err[f"{m}_exterior_rms"] = np.array([rms_rel(b[n_in:], B_ref) for b in B[m]])
        err[f"{m}_interior_max"] = np.array([rel(b[:n_in], ref_in) for b in B[m]])
        err[f"{m}_interior_rms"] = np.array([rms_rel(b[:n_in], ref_in) for b in B[m]])
        for lo_r, hi_r in RHO_BANDS:
            band = (rho_in >= lo_r) & (rho_in <= hi_r)
            err[f"{m}_interior_max_rho{lo_r}-{hi_r}"] = np.array(
                [np.abs(b[:n_in][band] - ref_in[band]).max() / scale_in for b in B[m]])
    print(f"\nerrors / max |B_ref|; exterior: vs continuous Biot-Savart, interior: vs "
          f"truncated at h = a/{fracs[-1]}; (order) between columns")
    for where in ("exterior", "interior"):
        for norm in ("max", "rms"):
            print_table(f"{where} {norm}", fracs, {m: err[f"{m}_{where}_{norm}"] for m in METHODS})
    for m in METHODS:
        print_table(f"interior max, {m}", fracs,
                    {f"rho {lo_r}-{hi_r}": err[f"{m}_interior_max_rho{lo_r}-{hi_r}"]
                     for lo_r, hi_r in RHO_BANDS})
    if args.out:
        np.savez(args.out, fracs=fracs, a=a, edge_power=args.edge_power, X_in=X_in,
                 rho_in=rho_in, X_ext=X_ext, B_ref_ext=B_ref,
                 **{f"B_{m}": np.array(B[m]) for m in METHODS}, **err)


if __name__ == "__main__":
    main()
