"""Device (JAX) S_m build against the stored float64 references.

For each --fracs of the grids JSON (``tests/cyl_s_build.py grids``):
  1. AnnulusPoisson (the per-mode eigendecompositions, threaded), timed;
  2. the z-folded rows by ``single_layer_jax.single_layer_matrix_jax``, timed (a/--warm first,
     to separate the compilation);
  3. folded mode by mode and compared with the reference S_e, S_o at --ref/S_<key>/ (memory-mapped;
     the key from --refjson/s_ref_a<frac>.json): max |difference| / max |reference row|, per |m| band;
  4. with --rekey and a pass (<= --tol), the reference directory renamed to the current key
     (``annulus_freespace.S_key``) when its meta matches the key inputs.
Also times loading one reference (``load_S``) with --load-frac. Writes --out/s_gpu.json.

run: python tests/cyl_s_gpu.py --grids s_grids.json --fracs 20 30 40 50 60 --ref /results/cyl/S_cache \\
         --refjson /results/cyl/s_ref --out DIR [--rekey]
"""
import argparse
import json
import os
import resource
import sys
import time

import numpy as np

T0 = time.time()


def log(msg):
    r = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    r = r / 1e9 if sys.platform == "darwin" else r / 1e6
    print(f"[{time.time() - T0:7.1f} s, rss peak {r:5.1f} GB] {msg}", flush=True)


def setup(G):
    from annulus_freespace import ZFold, mode_indices, z_reflection
    from annulus_poisson import AnnulusPoisson
    from chebyshev import ChebAxis
    ra, za, N, NFP = ChebAxis(*G["r_axis"]), ChebAxis(*G["z_axis"]), G["N_phi"], G["NFP"]
    ms = sorted(set(np.abs(mode_indices(N, real=True) * NFP).tolist())
                | set(np.abs(mode_indices(N, real=False) * NFP + 1).tolist()))
    t = time.time()
    ap = AnnulusPoisson(ra, za, ms)
    t_eig = time.time() - t
    pts, _ = ap.boundary_points()
    zf = ZFold(z_reflection(ap))
    return ap, zf, pts[zf.rep], t_eig


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--grids", default="s_grids.json")
    p.add_argument("--fracs", type=int, nargs="+", default=[20, 30, 40, 50, 60])
    p.add_argument("--warm", type=int, default=10)
    p.add_argument("--ref", default="/results/cyl/S_cache")
    p.add_argument("--refjson", default="/results/cyl/s_ref")
    p.add_argument("--mem-bytes", type=float, default=None)
    # tolerance: the forward-recurrence lanes (near-coincident points) amplify rounding by up to
    # e^4 m eps, so two correct implementations differ by ~1e-11 there in Q and ~1e-12 in S_m
    # (a/30 subset: 6.6e-13), below the quadrature error (<= 3.4e-11)
    p.add_argument("--tol", type=float, default=5e-12)
    p.add_argument("--rekey", action="store_true")
    p.add_argument("--load-frac", type=int, default=60)
    p.add_argument("--out", default=".")
    a = p.parse_args()
    import jax
    jax.config.update("jax_enable_x64", True)
    from annulus_freespace import S_key, load_S
    from annulus_poisson import _n_threads
    from single_layer_jax import single_layer_matrix_jax
    grids = json.load(open(a.grids))["grids"]
    res = dict(backend=jax.default_backend(), devices=[str(d) for d in jax.devices()],
               cpus=len(os.sched_getaffinity(0)) if hasattr(os, "sched_getaffinity") else os.cpu_count(),
               runs={})
    log(f"jax {jax.__version__} {res['backend']} {res['devices']}; {res['cpus']} CPUs")
    out_json = os.path.join(a.out, "s_gpu.json")
    for frac in ([a.warm] if a.warm else []) + a.fracs:
        G = grids[str(frac)]
        ap, zf, rows_pts, t_eig = setup(G)
        n_th = _n_threads(len(ap.ms))
        log(f"a/{frac}: {len(ap.ms)} modes, m_max {ap.ms.max()}, N_b {ap.n_boundary}, rows {len(rows_pts)}; "
            f"eig {t_eig:.1f} s ({n_th} threads)")
        t = time.time()
        rows = single_layer_matrix_jax(ap, ap.ms, rows_pts, verbose=True, mem_bytes=a.mem_bytes)
        t_b = time.time() - t
        n_nan = int(np.isnan(rows).sum())
        log(f"a/{frac}: device build {t_b:.1f} s ({rows.nbytes / 1e9:.2f} GB rows; NaN entries {n_nan})")
        run = dict(eig_s=t_eig, eig_threads=n_th, build_s=t_b, rows_GB=rows.nbytes / 1e9, nan=n_nan)
        rj = os.path.join(a.refjson, f"s_ref_a{frac}.json")
        if os.path.exists(rj):
            key_old = json.load(open(rj))["key"]
            path = os.path.join(a.ref, f"S_{key_old}")
            if not os.path.isdir(path):  # already re-keyed
                path = os.path.join(a.ref, f"S_{S_key(ap, True)}")
            Se, So = (np.load(os.path.join(path, n), mmap_mode="r") for n in ("S.npy", "S_o.npy"))
            cc = 2 * zf.c[:, None] * zf.c[None, :]
            err = np.zeros(len(ap.ms))
            t = time.time()
            for k in range(len(ap.ms)):
                R = rows[k]
                e = cc * (R[:, zf.rep] + R[:, zf.partner])
                o = R[zf.odd_rows][:, zf.odd] - R[zf.odd_rows][:, zf.podd]
                re, ro = np.asarray(Se[k]), np.asarray(So[k])
                err[k] = max((np.abs(e - re).max(axis=1) / np.abs(re).max(axis=1)).max(),
                             (np.abs(o - ro).max(axis=1) / np.abs(ro).max(axis=1)).max())
            bands = {}
            for lo, hi in ((0, 100), (100, 300), (300, 600), (600, 1000), (1000, 10**6)):
                sel = (ap.ms >= lo) & (ap.ms < hi)
                if sel.any():
                    bands[f"{lo}-{hi}"] = float(err[sel].max())
            run.update(ref=path, max_rowrel=float(err.max()), bands=bands, compare_s=time.time() - t)
            ok = err.max() <= a.tol
            log(f"a/{frac}: vs reference max row-relative {err.max():.2e} ({'pass' if ok else 'FAIL'}; tol "
                f"{a.tol:.0e}); bands {', '.join(f'{b} {v:.1e}' for b, v in bands.items())}")
            if a.rekey and ok:
                key_new = S_key(ap, True)
                meta = json.load(open(os.path.join(path, "meta.json")))
                same = (meta["ms"] == [int(m) for m in ap.ms] and meta["zsym"] and meta["n_gl"] == 16
                        and meta["ratio"] == 0.15
                        and meta["r_axis"] == [ap.r_axis.a, ap.r_axis.b, ap.r_axis.n, ap.r_axis.alpha]
                        and meta["z_axis"] == [ap.z_axis.a, ap.z_axis.b, ap.z_axis.n, ap.z_axis.alpha])
                dst = os.path.join(a.ref, f"S_{key_new}")
                if same and not os.path.exists(dst):
                    meta.update(key=key_new, key_old=key_old, key_scheme="S_NUMERICS 1")
                    with open(os.path.join(path, "meta.json"), "w") as f:
                        json.dump(meta, f, indent=1)
                    os.rename(path, dst)
                    run.update(rekeyed=dst)
                    log(f"a/{frac}: re-keyed S_{key_old} -> S_{key_new}")
                else:
                    log(f"a/{frac}: not re-keyed (meta match {same}, target exists {os.path.exists(dst)})")
            del Se, So
        res["runs"][str(frac) if frac not in res["runs"] else f"{frac}b"] = run
        del rows
        json.dump(res, open(out_json, "w"), indent=1)
    if a.load_frac:
        ap, zf, _, _ = setup(grids[str(a.load_frac)])
        key = S_key(ap, True)
        for where in (a.ref,):
            t = time.time()
            got = load_S(where, key, np.float64)
            if got is None:
                rj = json.load(open(os.path.join(a.refjson, f"s_ref_a{a.load_frac}.json")))["key"]
                got = load_S(where, rj, np.float64)
            t_l = time.time() - t
            if got is not None:
                res["load"] = dict(frac=a.load_frac, s=t_l, GB=(got[0].nbytes + got[1].nbytes) / 1e9)
                log(f"load a/{a.load_frac}: {t_l:.1f} s for {res['load']['GB']:.1f} GB")
        json.dump(res, open(out_json, "w"), indent=1)
    log("done")


if __name__ == "__main__":
    main()
