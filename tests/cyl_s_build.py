"""S_m reference builds for the production cylindrical grids (CPU).

  grids   (local, needs DESC) the production CylindricalGrid axes of the smooth-current
          precise_QA, or --eq/--coils (``optimize_qa``'s make_box: around(inp, inp.a / frac), margin 2) for
          --fracs, written to --grids JSON.
  build   (no DESC) for one --frac of that JSON, the S_m of FreeSpaceAnnulus(shifts=(0, 1),
          z-folded) in float64, stored in the S_m cache --cache (the NEST_S_CACHE layout, so
          production runs load it), with
          1. a pool check: the first rows serial against the process pool (NEST_S_JOBS), bitwise;
          2. the build (timed, per-10% progress);
          3. a quadrature spot check: --spot rows with the default rule (16 points, ratio 0.15)
             against a tightened one (32 points, ratio 0.08), max |difference| relative to the
             row's maximum, per |m| band.
          Writes --out/s_ref_a<frac>.json.

run: python tests/cyl_s_build.py grids --fracs 20 30 40 50 60 70 --grids s_grids.json [--eq EQ.h5 --coils C.h5]
     python tests/cyl_s_build.py build --grids s_grids.json --frac 40 --cache DIR --jobs 32 --out DIR
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
T0 = time.time()


def log(msg):
    r = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    rss = r / 1e9 if sys.platform == "darwin" else r / 1e6
    print(f"[{time.time() - T0:7.1f} s, rss peak {rss:5.1f} GB] {msg}", flush=True)


def step_grids(a):
    from cylindrical_grid import CylindricalGrid
    from desc_input import DESCInput
    eq = a.eq or os.path.join(ROOT, "desc_input", "precise_QA_free_boundary_smooth_J",
                              "precise_QA_free_boundary_smooth_J_edgeJ_r0.3.h5")
    coils = a.coils or os.path.join(ROOT, "desc_input", "precise_QA_free_boundary",
                                    "precise_QA_helical_coils.h5")
    inp = DESCInput.load(eq, field=coils)
    out = {"equilibrium": os.path.basename(eq), "a": float(inp.a), "grids": {}}
    for f in a.fracs:
        g = CylindricalGrid.around(inp, inp.a / f)
        ax = lambda x: [x.a, x.b, x.n, x.alpha]  # noqa: E731
        out["grids"][str(f)] = dict(r_axis=ax(g.r_axis), z_axis=ax(g.z_axis), NFP=g.NFP, N_phi=g.N_phi,
                                    shape=list(g.shape))
        print(f"a/{f}: {g}", flush=True)
    json.dump(out, open(a.grids, "w"), indent=1)


def band_errors(S_def, S_tight, ms):
    rel = np.abs(S_tight - S_def).max(axis=(1, 2)) / np.maximum(np.abs(S_def).max(axis=(1, 2)), 1e-300)
    bands = [(0, 100), (100, 300), (300, 600), (600, 1000), (1000, 10 ** 6)]
    return {f"{lo}-{hi if hi < 10 ** 6 else 'max'}": float(rel[(ms >= lo) & (ms < hi)].max())
            for lo, hi in bands if ((ms >= lo) & (ms < hi)).any()}


def step_build(a):
    from annulus_freespace import FreeSpaceAnnulus, S_key, ZFold, z_reflection
    from annulus_poisson import AnnulusPoisson
    from chebyshev import ChebAxis
    from single_layer import single_layer_matrix
    G = json.load(open(a.grids))["grids"][str(a.frac)]
    ra, za = ChebAxis(*G["r_axis"]), ChebAxis(*G["z_axis"])
    NFP, N = G["NFP"], G["N_phi"]
    info = dict(frac=a.frac, shape=G["shape"], cpus=os.cpu_count(), jobs=a.jobs)
    log(f"a/{a.frac}: grid {G['shape']}, {os.cpu_count()} CPUs, jobs {a.jobs}")
    # the |m| set and boundary as FreeSpaceAnnulus builds them
    from annulus_freespace import mode_indices
    ms = set()
    for s in (0, 1):
        ms |= set(np.abs(mode_indices(N, real=(s == 0)) * NFP + s).tolist())
    ap = AnnulusPoisson(ra, za, sorted(ms))
    pts, _ = ap.boundary_points()
    fold = ZFold(z_reflection(ap))
    rows_pts = pts[fold.rep]
    info.update(n_modes=len(ap.ms), m_max=int(ap.ms.max()), N_b=ap.n_boundary, n_rows=len(rows_pts),
                key=S_key(ap, True))
    log(f"{len(ap.ms)} modes (|m| <= {ap.ms.max()}), N_b {ap.n_boundary}, {len(rows_pts)} rows; key {info['key']}")
    # 1. pool check
    n_chk = min(len(rows_pts), max(16, 2 * a.jobs))
    t = time.time()
    s1 = single_layer_matrix(ap, ap.ms, rows_pts[:n_chk], n_jobs=1)
    t_ser = time.time() - t
    t = time.time()
    sp = single_layer_matrix(ap, ap.ms, rows_pts[:n_chk], n_jobs=a.jobs)
    t_par = time.time() - t
    ok = bool(np.array_equal(s1, sp))
    info["pool_check"] = dict(rows=n_chk, bitwise=ok, serial_s=t_ser, pool_s=t_par)
    log(f"pool check on {n_chk} rows: bitwise {ok}; serial {t_ser:.1f} s, pool {t_par:.1f} s ({t_ser / t_par:.1f}x)")
    os.environ["NEST_S_JOBS"] = str(a.jobs if ok else 1)
    del s1, sp
    # 2. the build, into the cache
    t = time.time()
    fs = FreeSpaceAnnulus(ra, za, NFP, N, shifts=(0, 1), verbose=True, S_dtype=np.float64, zsym=True,
                          cache=a.cache)
    info.update(build_s=time.time() - t, cache=list(fs.S_cache), S_GB_float64=fs.S_nbytes / 1e9)
    log(f"build: {info['build_s']:.0f} s, cache {fs.S_cache}, {fs.S_nbytes / 1e9:.2f} GB float64")
    del fs
    # 3. quadrature spot check
    rng = np.random.default_rng(0)
    n = len(rows_pts)
    pick = np.unique(np.concatenate([[0, n - 1, n // 4, n // 2, 3 * n // 4],
                                     rng.choice(n, min(n, a.spot), replace=False)]))[: a.spot]
    os.environ["NEST_S_JOBS"] = "1"
    S_def = single_layer_matrix(ap, ap.ms, rows_pts[pick])
    S_tight = single_layer_matrix(ap, ap.ms, rows_pts[pick], n_gl=32, ratio=0.08)
    info["spot_check"] = dict(rows=int(len(pick)), rule="32 points, ratio 0.08 vs 16, 0.15",
                              max_rel_by_m=band_errors(S_def, S_tight, np.asarray(ap.ms)))
    log(f"spot check ({len(pick)} rows): {info['spot_check']['max_rel_by_m']}")
    info["total_s"] = time.time() - T0
    os.makedirs(a.out, exist_ok=True)
    out = os.path.join(a.out, f"s_ref_a{a.frac}.json")
    with open(out, "w") as f:
        json.dump(info, f, indent=1)
        f.flush()
        os.fsync(f.fileno())
    log(f"saved {out}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("step", choices=["grids", "build"])
    ap.add_argument("--fracs", type=int, nargs="+", default=[20, 30, 40, 50, 60, 70])
    ap.add_argument("--eq", help="DESC equilibrium .h5 (grids; default: smooth-current precise_QA)")
    ap.add_argument("--coils", help="coil set .h5 for --eq (grids)")
    ap.add_argument("--frac", type=int)
    ap.add_argument("--grids", default="s_grids.json")
    ap.add_argument("--cache", default=os.environ.get("NEST_S_CACHE"))
    ap.add_argument("--jobs", type=int, default=os.cpu_count())
    ap.add_argument("--spot", type=int, default=24)
    ap.add_argument("--out", default=".")
    a = ap.parse_args()
    (step_grids if a.step == "grids" else step_build)(a)


if __name__ == "__main__":
    main()
