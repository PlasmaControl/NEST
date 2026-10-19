"""Gate B4: B_plasma of NEST's J (smooth-current precise_QA) on the cylindrical
annulus against main's Vico-Greengard box, at fixed points.

J = curl T, ``BoundaryPotential`` at factors (3, 3, 2.5), edge power 2 (J ~ (1 - rho)
at the edge: a kink on rho = 1), main's B-convergence case (``EQ_FILES`` "qa" of
tests/test_magnetic_field.py). Each step runs in its own process, so its peak
memory (``ru_maxrss``) is its own:

  targets           interior points: the force collocation grid (Gauss-Legendre rho,
                    uniform theta, zeta over one period) of ForceResidual's default
                    resolution; exterior points: inside the finest annulus,
                    ``--ext-lo``..``--ext-hi`` a from the LCFS
  ref               B at the exterior points by continuous Biot-Savart (Gauss quadrature
                    in flux coordinates), two resolutions
  cyl --frac F      cylindrical solve, h = a / F; J exact at the nodes (series at the
                    Newton-mapped flux coordinates), or ``--J lagrange`` (main's
                    cached flux-grid interpolation, check (d))
  box --frac F      main's pipeline: periodic box h = a / F, J exact at the box points,
                    truncated (VG) kernel, type-2 NUFFT to the points; ``--dtype``
  analyze           error tables (interior by rho band against the finest cylindrical
                    solve, exterior against continuous Biot-Savart), memory, time

Outputs go to ``--dir`` (default results/cyl/b4).

run: python tests/cyl_b4.py targets|ref|cyl|box|analyze [--frac F] [...]
"""

import argparse
import json
import os
import resource
import sys
import time

import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, "nest"))
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

EQ = os.path.join(ROOT, "desc_input", "precise_QA_free_boundary_smooth_J",
                  "precise_QA_free_boundary_smooth_J_edgeJ_r0.3.h5")
FACTORS = (3, 3, 2.5)
RHO_BANDS = [(0, 0.5), (0.5, 0.8), (0.8, 0.95), (0.95, 1.0)]


def peak_gb():
    r = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    return r / 1e9 if sys.platform == "darwin" else r / 1e6  # bytes on macOS, kB on Linux


def setup():
    import jax
    jax.config.update("jax_enable_x64", True)
    import fit_cache
    from boundary_potential import BoundaryPotential
    from desc_input import DESCInput
    inp = DESCInput.load(EQ)
    t0 = time.time()
    cur = fit_cache.fit(BoundaryPotential, inp, factors=FACTORS)
    print(f"{os.path.basename(EQ)}: a = {inp.a:.5f} m; BoundaryPotential L, M, N = "
          f"{cur.basis.L}, {cur.basis.M}, {cur.basis.N} ({cur.n_free} free); fit/load "
          f"{time.time() - t0:.0f} s", flush=True)
    return inp, cur


def cyl_to_cart(V, phi):
    c, s = np.cos(phi), np.sin(phi)
    return np.column_stack([V[:, 0] * c - V[:, 1] * s, V[:, 0] * s + V[:, 1] * c, V[:, 2]])


def to_rpz(X):
    return np.column_stack([np.hypot(X[:, 0], X[:, 1]), np.arctan2(X[:, 1], X[:, 0]), X[:, 2]])


# ------------------------------------------------------------------ steps
def step_targets(a):
    from scipy.spatial import cKDTree
    from cylindrical_grid import CylindricalGrid
    from force import ForceResidual
    inp, cur = setup()
    fr = ForceResidual(inp, None, sym=False)
    X_in, rtz_in, dV = fr.x, fr.rtz, fr.dV
    # exterior: inside the finest annulus (smallest margin), ext_lo..ext_hi a from the LCFS
    g = CylindricalGrid.around(inp, inp.a / max(a.fracs), a.margin)
    from desc.grid import LinearGrid
    lg = LinearGrid(rho=1.0, theta=256, zeta=256, NFP=1, sym=False)
    d = inp.compute(["R", "phi", "Z"], lg)
    lcfs = np.column_stack([d["R"] * np.cos(d["phi"]), d["R"] * np.sin(d["phi"]), d["Z"]])
    rng = np.random.default_rng(0)
    n_try = 400000
    P = np.column_stack([rng.uniform(g.r_axis.a, g.r_axis.b, n_try), rng.uniform(0, 2 * np.pi, n_try),
                         rng.uniform(g.z_axis.a, g.z_axis.b, n_try)])
    Xc = np.column_stack([P[:, 0] * np.cos(P[:, 1]), P[:, 0] * np.sin(P[:, 1]), P[:, 2]])
    dist, _ = cKDTree(lcfs).query(Xc)
    # outside the plasma: the inverse map finds no rho <= 1 (checked below)
    sel = np.flatnonzero((dist >= a.ext_lo * inp.a) & (dist <= a.ext_hi * inp.a))
    from grid import map_to_flux
    sel = sel[: 4 * a.n_ext]
    rr = map_to_flux(inp, Xc[sel])
    sel = sel[~np.isfinite(rr[:, 0])][: a.n_ext]
    X_ext = Xc[sel]
    np.savez(os.path.join(a.dir, "targets.npz"), X_in=X_in, rtz_in=rtz_in, dV=dV, X_ext=X_ext,
             dist_ext=dist[sel], a=inp.a)
    print(f"targets: {len(X_in):,} interior (collocation grid {fr.tgrid.shape}), {len(X_ext)} exterior "
          f"{a.ext_lo}-{a.ext_hi} a from the LCFS inside {g}", flush=True)


def step_ref(a):
    from test_magnetic_field import continuous_biot_savart
    inp, cur = setup()
    t = np.load(os.path.join(a.dir, "targets.npz"))
    eq = inp.eq
    res = np.maximum([cur.basis.L, cur.basis.M, cur.basis.N], 3 * np.array([eq.L, eq.M, eq.N]))
    res2 = tuple(int(v) for v in np.ceil(2 * res / 3))
    res = tuple(int(v) for v in res)
    if a.lmn:  # explicit (fine, comparison) resolutions; the 2/3 rule's comparison level
        res, res2 = tuple(a.lmn[:3]), tuple(a.lmn[3:])  # can under-resolve the basis
    out_name = a.ref_out
    t0 = time.time()
    X = t["X_ext"]
    B1, B2 = np.zeros_like(X), np.zeros_like(X)
    for i in range(0, len(X), a.ext_chunk):  # in chunks of targets, with progress
        sl = slice(i, i + a.ext_chunk)
        B2[sl] = continuous_biot_savart(eq, cur, X[sl], LMN=res2, block=a.block)
        B1[sl] = continuous_biot_savart(eq, cur, X[sl], LMN=res, block=a.block)
        print(f"  ref: {min(i + a.ext_chunk, len(X))}/{len(X)} targets, {time.time() - t0:.0f} s, "
              f"peak {peak_gb():.2f} GB", flush=True)
    dB = np.abs(B1 - B2).max() / np.abs(B1).max()
    np.savez(os.path.join(a.dir, out_name), B=B1, B_coarse=B2, LMN=res, LMN2=res2)
    print(f"continuous Biot-Savart at {len(B1)} exterior points, L, M, N = {res} vs {res2}: "
          f"max diff {dB:.1e}; {time.time() - t0:.0f} s; peak {peak_gb():.2f} GB", flush=True)


def step_cyl(a):
    import jax
    import jax.numpy as jnp
    from cartesian_grid import evaluate_chunked
    from cylindrical_bs import CylindricalSolver
    from cylindrical_grid import CylindricalGrid, evaluate
    inp, cur = setup()
    base = peak_gb()
    t = np.load(os.path.join(a.dir, "targets.npz"))
    h = inp.a / a.frac
    g = CylindricalGrid.around(inp, h, a.margin, alpha=a.alpha)
    print(f"[cyl a/{a.frac}] {g}; baseline peak {base:.2f} GB", flush=True)
    tm, stage_peak = {}, {}

    def done(name, t0):
        tm[name], stage_peak[name] = time.time() - t0, peak_gb()
        print(f"[cyl a/{a.frac}] {name}: {tm[name]:.1f} s, peak so far {stage_peak[name]:.2f} GB", flush=True)

    t0 = time.time()
    g.locate_plasma(inp, verbose=True, planes=a.planes)
    done("locate", t0)
    t0 = time.time()
    if a.J == "exact":
        Jc = evaluate_chunked(cur.point_evaluator(), g.rtz, 20000, f"J series ({len(g.rtz):,} points)")
    else:  # main's path for the box: uniform flux grid + cached Lagrange interpolation
        from flux_interp import PointInterpolator, TensorCurrent, TensorGrid
        b = cur.basis
        fg = TensorGrid.uniform(4 * b.L, 2 * (3 * b.M), max(4 * b.N, 8), inp.eq.NFP)
        Jf = TensorCurrent(cur.S, cur._RZ, fg)(jnp.asarray(cur.u), cur.I)
        Jc = np.asarray(PointInterpolator(fg, g.rtz, 4)(Jf))
    done("J", t0)
    J = g.scatter(Jc)
    del Jc
    t0 = time.time()
    S_dtype = np.float32 if a.S_dtype == "float32" else np.float64
    sol = CylindricalSolver(g.r_axis, g.z_axis, g.NFP, g.N_phi, chunk=a.chunk, verbose=True,
                            S_dtype=S_dtype)
    done("precompute", t0)
    fm = jax.jit(lambda s, x: s.field_modes(x))
    Jd = jnp.asarray(J)
    del J
    t0 = time.time()
    Bp, Bz = jax.block_until_ready(fm(sol, Jd))
    done("solve_first", t0)
    t0 = time.time()
    Bp, Bz = jax.block_until_ready(fm(sol, Jd))
    done("solve", t0)
    t0 = time.time()
    out = {}
    for k in ("X_in", "X_ext"):
        P = to_rpz(t[k])
        out[k] = cyl_to_cart(evaluate(g, Bp, Bz, P), P[:, 1])
    done("interp", t0)
    info = dict(frac=a.frac, h=h, grid=repr(g), shape=g.shape, nodes=g.size, inside=len(g.inside),
                n_modes=len(sol.fs.ap.ms), N_b=sol.fs.ap.n_boundary, S_GB=sol.S_nbytes / 1e9,
                S_dtype=a.S_dtype, peak_GB=peak_gb(), base_GB=base, stage_peak_GB=stage_peak,
                times=tm, J=a.J, margin=a.margin, alpha=a.alpha)
    tag = (f"cyl_a{a.frac}" + ("" if a.J == "exact" else "_lagrange")
           + ("" if a.margin == 2.0 else f"_m{a.margin:g}") + ("" if a.alpha == 0 else f"_al{a.alpha:g}"))
    np.savez(os.path.join(a.dir, tag + ".npz"), B_in=out["X_in"], B_ext=out["X_ext"])
    json.dump(info, open(os.path.join(a.dir, tag + ".json"), "w"), indent=1)
    print(f"{tag}: inside {len(g.inside):,}; |m| {len(sol.fs.ap.ms)}, N_b {sol.fs.ap.n_boundary}, "
          f"S {info['S_GB']:.2f} GB ({a.S_dtype}); peak {info['peak_GB']:.2f} GB", flush=True)


def step_box(a):
    import jax.numpy as jnp
    from cartesian_grid import current_on_cartesian
    from fourier import fft_J
    from grid import CartesianGrid
    from magnetic_field import b_plasma_truncated, interpolate, truncated_green_hat
    inp, cur = setup()
    base = peak_gb()
    t = np.load(os.path.join(a.dir, "targets.npz"))
    dt = np.float64 if a.dtype == "float64" else np.float32
    jdt = jnp.float64 if a.dtype == "float64" else jnp.float32
    h = inp.a / a.frac
    tm = {}
    grid = CartesianGrid.periodic_box(inp, h)
    t0 = time.time()
    G = truncated_green_hat(grid.axes, dtype=jdt)
    tm["kernel"] = time.time() - t0
    t0 = time.time()
    d = current_on_cartesian(inp.eq, cur, grid, dtype=dt)
    tm["locate+J"] = time.time() - t0
    t0 = time.time()
    B = b_plasma_truncated(fft_J(d["J"]), grid.axes, G)
    B = np.asarray(B)
    tm["solve"] = time.time() - t0
    n_in = int(len(d["inside"]))
    del d, G
    t0 = time.time()
    eps = 1e-12 if a.dtype == "float64" else 1e-6
    X = np.concatenate([t["X_in"], t["X_ext"]])
    Bt = interpolate(B, grid.axes, X, eps=eps)
    tm["interp"] = time.time() - t0
    n = len(t["X_in"])
    info = dict(frac=a.frac, h=h, shape=grid.shape, nodes=grid.size, inside=n_in, dtype=a.dtype,
                peak_GB=peak_gb(), base_GB=base, times=tm)
    tag = f"box_a{a.frac}_{a.dtype}"
    np.savez(os.path.join(a.dir, tag + ".npz"), B_in=Bt[:n], B_ext=Bt[n:])
    json.dump(info, open(os.path.join(a.dir, tag + ".json"), "w"), indent=1)
    print(f"{tag}: box {grid.shape} ({grid.size / 1e6:.1f}M), inside {n_in:,}; times "
          + ", ".join(f"{k} {v:.1f} s" for k, v in tm.items())
          + f"; peak {info['peak_GB']:.2f} GB (baseline {base:.2f})", flush=True)


def step_analyze(a):
    t = np.load(os.path.join(a.dir, "targets.npz"))
    rho = t["rtz_in"][:, 0]
    w = t["dV"] / t["dV"].sum()
    runs = {}
    for f in sorted(os.listdir(a.dir)):
        if f.endswith(".json") and (f.startswith("cyl_") or f.startswith("box_")):
            tag = f[:-5]
            runs[tag] = (json.load(open(os.path.join(a.dir, f))), np.load(os.path.join(a.dir, tag + ".npz")))
    ref_tag = a.ref or max((k for k in runs if k.startswith("cyl_a") and k.count("_") == 1),
                           key=lambda k: runs[k][0]["frac"])
    Bref = runs[ref_tag][1]["B_in"]
    scale = np.abs(Bref).max()
    ext = np.load(os.path.join(a.dir, "ref_ext.npz")) if os.path.exists(os.path.join(a.dir, "ref_ext.npz")) else None
    rows = []
    for tag, (info, B) in sorted(runs.items(), key=lambda kv: (kv[0].split("_")[0], kv[1][0]["frac"])):
        dB = np.linalg.norm(B["B_in"] - Bref, axis=1) / scale
        row = dict(tag=tag, nodes=info["nodes"], peak_GB=info["peak_GB"],
                   rms=float(np.sqrt(np.sum(w * dB**2))), max=float(dB.max()))
        for lo, hi in RHO_BANDS:
            m = (rho > lo) & (rho <= hi)
            row[f"max_{lo}-{hi}"] = float(dB[m].max())
        if ext is not None:
            de = np.linalg.norm(B["B_ext"] - ext["B"], axis=1) / np.abs(ext["B"]).max()
            row["ext_max"] = float(de.max())
        rows.append(row)
    print(f"interior: |B - B_ref| / max|B_ref|, B_ref = {ref_tag} (volume-weighted rms, max, max by rho band);"
          f" exterior: against continuous Biot-Savart")
    hdr = ["tag", "nodes", "peak_GB", "rms", "max"] + [f"max_{lo}-{hi}" for lo, hi in RHO_BANDS] + ["ext_max"]
    print("  ".join(f"{x:>14}" for x in hdr))
    for r in rows:
        print("  ".join(f"{r.get(k, float('nan')):>14.3g}" if k != "tag" else f"{r[k]:>14}" for k in hdr))
    if ext is not None:
        print(f"continuous Biot-Savart self-difference: "
              f"{np.abs(ext['B'] - ext['B_coarse']).max() / np.abs(ext['B']).max():.1e}")
    json.dump(dict(ref=ref_tag, rows=rows), open(os.path.join(a.dir, "analysis.json"), "w"), indent=1)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("step", choices=["targets", "ref", "cyl", "box", "analyze"])
    ap.add_argument("--frac", type=int, default=10)
    ap.add_argument("--fracs", type=int, nargs="+", default=[10, 15, 20])
    ap.add_argument("--margin", type=float, default=2.0)
    ap.add_argument("--alpha", type=float, default=0.0, help="KTE map of the r and z axes")
    ap.add_argument("--chunk", type=int, default=16)
    ap.add_argument("--J", default="exact", choices=["exact", "lagrange"])
    ap.add_argument("--dtype", default="float64", choices=["float64", "float32"])
    ap.add_argument("--planes", type=int, default=32, help="phi-planes per inverse-map call")
    ap.add_argument("--S-dtype", default="float32", choices=["float64", "float32"])
    ap.add_argument("--ext-lo", type=float, default=0.15)
    ap.add_argument("--ext-hi", type=float, default=0.4)
    ap.add_argument("--n-ext", type=int, default=200)
    ap.add_argument("--ext-chunk", type=int, default=25)
    ap.add_argument("--lmn", type=int, nargs=6, default=None,
                    help="L M N of the reference quadrature, then of its comparison")
    ap.add_argument("--ref-out", default="ref_ext.npz")
    ap.add_argument("--block", type=int, default=50000)
    ap.add_argument("--ref", default=None)
    ap.add_argument("--dir", default=os.path.join(ROOT, "results", "cyl", "b4"))
    a = ap.parse_args()
    os.makedirs(a.dir, exist_ok=True)
    {"targets": step_targets, "ref": step_ref, "cyl": step_cyl, "box": step_box,
     "analyze": step_analyze}[a.step](a)


if __name__ == "__main__":
    main()
