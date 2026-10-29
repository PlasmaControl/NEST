"""Rotational transform of traced field lines (poincare_fixed.py sections): per line, the
poloidal angle of its phi = 0 crossings about the field's own magnetic axis, unwrapped per
field period; iota = total poloidal angle / total toroidal angle (n_periods 2 pi / NFP). The
axis is the centroid of the innermost line's crossings (refined once with the next lines).
Each line is labelled by the mean DESC rho of its first K crossings.
usage: python fieldline_iota.py EQ.h5 OUT.npz name=sections.npz:key ..."""
import sys
import numpy as np
from desc import set_device
set_device("cpu")
import desc.io  # noqa: E402
from desc.grid import LinearGrid  # noqa: E402


def iota_lines(PR, PZ, NFP, K=40, eq=None, phi=0.0, seeds_row=True):
    R, Z = (PR[1:], PZ[1:]) if seeds_row else (PR, PZ)  # row 0 of the phi = 0 sections: the seeds
    ok = np.isfinite(R).all(0)
    # axis: centroid of the line with the smallest crossing spread
    spread = np.where(ok, np.nanstd(R, 0) + np.nanstd(Z, 0), np.inf)
    i0 = int(np.argmin(spread))
    Ra, Za = np.nanmean(R[:, i0]), np.nanmean(Z[:, i0])
    th = np.arctan2(Z - Za, R - Ra)
    d = np.diff(th, axis=0)
    d = np.mod(d + np.pi, 2 * np.pi) - np.pi
    n = d.shape[0]
    iota = d.sum(0) / (n * 2 * np.pi / NFP)
    big = np.nanmax(np.abs(d), 0)  # largest per-period step: near pi means the unwrap is unsafe
    # rho label (DESC coordinates)
    pts = np.column_stack([R[:K].ravel(), np.full(R[:K].size, phi), Z[:K].ravel()])
    good = np.isfinite(pts).all(1)
    rr = np.full(len(pts), np.nan)
    rr[good] = np.asarray(eq.map_coordinates(pts[good], inbasis=("R", "phi", "Z"),
                                             outbasis=("rho", "theta", "zeta"), maxiter=60, tol=1e-10))[:, 0]
    rho = np.nanmean(rr.reshape(K, -1), 0)
    # half-sample estimate of the error: the two halves of the trace
    h = n // 2
    i1 = d[:h].sum(0) / (h * 2 * np.pi / NFP)
    i2 = d[h:].sum(0) / ((n - h) * 2 * np.pi / NFP)
    return rho[ok], iota[ok], np.abs(i1 - i2)[ok], (Ra, Za), big[ok]


if __name__ == "__main__":
    eq = desc.io.load(sys.argv[1])
    out = {}
    for spec in sys.argv[3:]:
        name, rest = spec.split("=", 1)
        path, key = rest.split(":")
        d = np.load(path)
        # the phi = pi / NFP section (near-triangular; the phi = 0 bean wraps around the axis and
        # its geometric angle can jump by more than pi between crossings)
        rho, io, err, ax, big = iota_lines(d[f"{key}_Rh"], d[f"{key}_Zh"], eq.NFP, eq=eq, phi=np.pi / eq.NFP,
                                           seeds_row=False)
        out[f"{name}_rho"], out[f"{name}_iota"], out[f"{name}_err"] = rho, io, err
        out[f"{name}_maxstep"] = big
        print(f"   {name}: largest per-period poloidal step {big.max():.2f} rad", flush=True)
        out[f"{name}_axis"] = np.array(ax)
        print(f"{name}: {len(rho)} lines, axis (R, Z) = ({ax[0]:.4f}, {ax[1]:.4f}), "
              f"median half-trace difference {np.median(err):.1e}", flush=True)
    r = np.linspace(0, 1, 401)
    g = LinearGrid(rho=r, NFP=eq.NFP)
    out["desc_rho"] = r
    out["desc_iota"] = np.asarray(g.compress(eq.compute("iota", grid=g)["iota"]))
    np.savez(sys.argv[2], **out)
    print("saved", sys.argv[2])
