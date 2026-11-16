"""The current/pressure basis options of mf_fixed.py for the helper scripts (island_seed.py,
start_check_bext.py), so seeds and diagnostics are built in the same (Fourier-Zernike or spline) basis
and on the same radial collocation as a run. Same flag names and meanings as mf_fixed.py, except
--p-basis for its --pressure."""

import numpy as np

import fit_cache
from boundary_potential import BoundaryPotential
from pressure import Pressure


def add_basis_args(ap):
    ap.add_argument("--current", choices=("fz", "spline", "fz+spline"), default="fz")
    ap.add_argument("--p-basis", choices=("fz", "spline", "fz+spline"), default="fz", help="mf_fixed.py's --pressure")
    ap.add_argument("--j-knots", type=float, nargs=3, default=(0.05, 0.01, 0.08))
    ap.add_argument("--p-knots", type=float, nargs=3, default=(0.05, 0.01, 0.08))
    ap.add_argument("--p-MN", type=int, nargs=2, default=None)
    ap.add_argument("--spl-mtol", type=float, default=0.0)
    ap.add_argument("--spl-mmax", type=int, default=None)
    ap.add_argument("--spl-mfz", type=int, default=None)
    ap.add_argument("--knots-fz", type=float, nargs="+", default=None)
    ap.add_argument("--spl-band", type=float, nargs=3, action="append", default=None)
    ap.add_argument("--spl-reg", choices=("drop3", "rhom"), default="drop3")
    ap.add_argument("--spl-qcap", type=int, default=6)
    ap.add_argument("--p-colloc", type=int, default=4)
    ap.add_argument("--p-rationals", nargs="*", default=["2/1"])
    ap.add_argument("--colloc-knots", action="store_true")


def build(inp, a, LMN, colloc=1.5, colloc_rho=None, log=print):
    """(cur, pre, res, rho_nodes) as mf_fixed.py builds them for these options and --LMN."""
    from spline_pressure import trunc_kw
    eq = inp.eq
    tk = trunc_kw(a.spl_mtol, a.spl_mmax, a.spl_mfz, a.knots_fz, a.spl_band, a.spl_reg, a.spl_qcap)
    if a.current == "spline":
        cur = fit_cache.fit(BoundaryPotential, inp, M=LMN[1], N=LMN[2], radial="spline", knots=tuple(a.j_knots),
                            helicities=tuple(a.p_rationals), **tk)
        log(f"[current] spline a, b: {cur.basis.summary()}")
    elif a.current == "fz+spline":
        cur = fit_cache.fit(BoundaryPotential, inp, L=LMN[0], M=LMN[1], N=LMN[2], radial="fz+spline",
                            knots=tuple(a.j_knots), helicities=tuple(a.p_rationals),
                            **trunc_kw(a.spl_mtol, a.spl_mmax, a.spl_mfz, a.knots_fz, a.spl_band))
        log(f"[current] FZ + band splines: {cur.basis.summary()}")
    else:
        cur = fit_cache.fit(BoundaryPotential, inp, L=LMN[0], M=LMN[1], N=LMN[2])
    if a.p_basis == "spline":
        from spline_pressure import SplinePressure
        pM, pN = a.p_MN if a.p_MN else (cur.basis.M, cur.basis.N)
        pre = fit_cache.fit(SplinePressure, inp, M=pM, N=pN, edge_order=2, coarse=a.p_knots[0], dense=a.p_knots[1],
                            half=a.p_knots[2], helicities=tuple(a.p_rationals), **tk)
        log(f"[pressure] spline: {pre.summary()}")
    elif a.p_basis == "fz+spline":
        spd = {"bands": tuple(tuple(float(v) for v in b_) for b_ in a.spl_band), "knots": tuple(a.p_knots)}
        spd.update(trunc_kw(a.spl_mtol, a.spl_mmax, a.spl_mfz, a.knots_fz))
        pre = fit_cache.fit(Pressure, inp, edge_order=2, L=LMN[0], M=LMN[1], N=LMN[2], spline=tuple(sorted(spd.items())))
        log(f"[pressure] FZ + band splines: {pre.basis.summary()}")
    else:
        pre = fit_cache.fit(Pressure, inp, edge_order=2, L=LMN[0], M=LMN[1], N=LMN[2])
    b = cur.basis
    res = tuple(int(round(colloc * (u + v))) for u, v in ((b.L, eq.L), (b.M, eq.M), (b.N, eq.N)))
    if colloc_rho:
        res = (int(round(colloc_rho * (b.L + eq.L))),) + res[1:]
    bps = ([pre.bp_rho] if a.p_basis == "spline" else []) + ([pre.basis.bp_rho] if a.p_basis == "fz+spline" else [])
    bps += [b.bp_rho] if a.current in ("spline", "fz+spline") else []
    if a.colloc_knots:
        from spline_pressure import breakpoints, rational_centres
        cen = rational_centres(eq, tuple(a.p_rationals)) if a.p_rationals else []
        bps.append(breakpoints(0.0, 1.0, a.j_knots[0], a.j_knots[1], cen, a.j_knots[2], tk.get("fz_knots"),
                               a.spl_band))
    rho_nodes = None
    if bps:
        from spline_pressure import composite_gauss
        rho_nodes = composite_gauss(np.unique(np.concatenate(bps)), a.p_colloc, min_nodes=res[0] // 2 + 1)
        log(f"[spline] collocation: {len(rho_nodes[0])} radial nodes ({a.p_colloc} per knot interval)")
    return cur, pre, res, rho_nodes
