"""Memory and time audit of the NEST objective (one-off setup, caches, evaluations).

For the smooth-current precise_QA in its coils, with the full-resolution
potential and pressure, at box spacings h = a / frac:
  - time of each one-off stage (load, fits, box location, collocation setup,
    Fields setup) and the host memory
  - the size of every cached array of ``Fields`` (basis factors, geometry,
    interpolation weights, kernel, B_ext, DESC targets), by group
  - compile and run time of f, J v and J^T w, and the GPU peak memory after
    each (``memory_stats``; run without XLA_PYTHON_CLIENT_ALLOCATOR=platform)
  - run time of each component of f: J and p at the collocation points, J on
    the flux grid, interpolation to the box, FFT + kernel, non-uniform FFT

run (GPU): sbatch --export=ALL,XLA_PYTHON_CLIENT_ALLOCATOR=default --mem=64G run.sbatch \
    scripts/audit.py [frac] [--kernel truncated] [--full]   (the pool allocator: memory_stats)
    [--colloc c (collocation c x the resolution of F; default the equilibrium's 2 L, 2 M, 2 N)]
    [--profile (cProfile of the setup stages)]
One box per job: the arrays and FFT plans of a previous box stay on the GPU.
"""

import os
import resource
import sys
import time

import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, "nest"))

from desc import set_device  # noqa: E402

set_device(os.environ.get("NEST_DEVICE", "cpu"))

import jax  # noqa: E402
import jax.numpy as jnp  # noqa: E402

import fit_cache  # noqa: E402
from boundary_potential import BoundaryPotential  # noqa: E402
from desc_input import DESCInput  # noqa: E402
from force import ForceResidual  # noqa: E402
from fourier import fft_J  # noqa: E402
from grid import CartesianGrid  # noqa: E402
from objective import (CurrentError, FieldError, Fields, ForceBalance,  # noqa: E402
                       ObjectiveFunction, PressureError)
from pressure import Pressure  # noqa: E402

jax.config.update("jax_enable_x64", True)
EQ = os.path.join(ROOT, "desc_input", "precise_QA_free_boundary_smooth_J",
                  "precise_QA_free_boundary_smooth_J_edgeJ_r0.3.h5")
COILS = os.path.join(ROOT, "desc_input", "precise_QA_free_boundary", "precise_QA_helical_coils.h5")


def gpu():
    s = jax.devices()[0].memory_stats() or {}
    return s.get("bytes_in_use", 0) / 1e9, s.get("peak_bytes_in_use", 0) / 1e9


def host():
    return resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / (1e9 if sys.platform == "darwin" else 1e6)


def stage(name, t0):
    use, peak = gpu()
    print(f"  {name:38s} {time.time() - t0:8.1f} s   GPU in use {use:6.2f} GB, peak {peak:6.2f} GB,"
          f" host peak {host():6.2f} GB", flush=True)


def nbytes(obj, seen=None, depth=0):
    """Total size of the arrays held by obj (recursing into attributes, dicts, lists)."""
    seen = set() if seen is None else seen
    if id(obj) in seen or depth > 4:
        return 0
    seen.add(id(obj))
    if isinstance(obj, (np.ndarray, jax.Array)):
        return obj.nbytes
    if isinstance(obj, dict):
        return sum(nbytes(v, seen, depth + 1) for v in obj.values())
    if isinstance(obj, (list, tuple)):
        return sum(nbytes(v, seen, depth + 1) for v in obj)
    if hasattr(obj, "__dict__"):
        return sum(nbytes(v, seen, depth + 1) for v in vars(obj).values())
    return 0


def timed(f, *a):
    t0 = time.time()
    jax.block_until_ready(f(*a))
    t_c = time.time() - t0
    t0 = time.time()
    out = jax.block_until_ready(f(*a))
    return out, t_c, time.time() - t0


def profiled(name, f, *a, **k):
    """f(*a, **k) under cProfile; the slowest calls (cumulative) are printed."""
    import cProfile
    import pstats
    pr = cProfile.Profile()
    t0 = time.time()
    out = pr.runcall(f, *a, **k)
    if PROFILE:
        print(f"  --- profile of {name} ({time.time() - t0:.1f} s):")
        st = pstats.Stats(pr)
        st.sort_stats("cumulative")
        rows = sorted(st.stats.items(), key=lambda kv: -kv[1][3])
        shown = 0
        for (fn, line, func), (_, nc, tt, ct, _) in rows:
            if ct < 1.0 or "site-packages/jax" in fn or "<" in fn:
                continue
            print(f"      {ct:8.1f} s  {nc:7d} calls  {os.path.basename(fn)}:{line} {func}")
            shown += 1
            if shown == 18:
                break
    return out


PROFILE = "--profile" in sys.argv


def memory(obj, x0, v, w):
    """Compiled memory of f, the linearization, J v and J^T w (XLA's analysis), their
    largest buffers, and what the linearization keeps."""
    import re
    from objective import _base, _lin, _lin_T, _scaled
    nb = {"f32": 4, "f64": 8, "c64": 8, "c128": 16, "s32": 4, "s64": 8, "u32": 4, "pred": 1}
    r, base = _base(obj, x0, obj.fields)
    print("  compiled memory [GB]: arguments, outputs, temporaries; largest buffers")
    for name, low in (("f", _scaled.lower(obj, x0, obj.fields)),
                      ("linearize (f, J, B)", _base.lower(obj, x0, obj.fields)),
                      ("J v", _lin.lower(obj, v, base, obj.fields)),
                      ("J^T w", _lin_T.lower(obj, w, base, obj.fields))):
        c = low.compile()
        ma = c.memory_analysis()
        sizes = {}
        for m in re.finditer(r"(f32|f64|c64|c128|s32|s64|u32|pred)\[([0-9,]+)\]", c.as_text()):
            sizes[m.group(0)] = int(np.prod([int(d) for d in m.group(2).split(",")])) * nb[m.group(1)]
        big = sorted(sizes.items(), key=lambda t: -t[1])[:5]
        print(f"    {name:26s} {ma.argument_size_in_bytes / 1e9:6.2f} {ma.output_size_in_bytes / 1e9:6.2f} "
              f"{ma.temp_size_in_bytes / 1e9:6.2f};  " + ", ".join(f"{k} {b / 1e9:.2f}" for k, b in big),
              flush=True)
    kept = r.nbytes + sum(v.nbytes for v in base.values())
    print(f"    linearization keeps {kept / 1e9:.3f} GB (f, J, B and p at the collocation points)",
          flush=True)
    del r, base


def audit(frac, kernel="hockney", sym=True, colloc=None):
    print(f"\n=== h = a/{frac}, kernel {kernel}, stellarator symmetry {sym} ===", flush=True)
    t0 = time.time()
    inp = DESCInput.load(EQ, field=COILS)
    stage("load", t0)
    t0 = time.time()
    small = dict(L=12, M=8, N=6) if os.environ.get("AUDIT_SMALL") else {}  # quick local check
    cur = profiled("potential fit", fit_cache.fit, BoundaryPotential, inp, sym=sym, **small)
    pre = profiled("pressure fit", fit_cache.fit, Pressure, inp, edge_order=2, sym=sym, **small)
    stage(f"fits (potential {cur.n_free:,}, pressure {pre.basis.num_modes:,})", t0)
    t0 = time.time()
    box = CartesianGrid.periodic_box(inp, inp.a / frac, odd=sym).locate_plasma(inp)
    stage(f"box {box.shape}, {len(box.inside):,} inside", t0)
    t0 = time.time()
    eq, b = inp.eq, cur.basis
    res = None if colloc is None else tuple(
        int(round(colloc * (u + v))) for u, v in ((b.L, eq.L), (b.M, eq.M), (b.N, eq.N)))
    fr = profiled("ForceResidual", ForceResidual, inp, box, resolution=res, sym=sym)
    stage(f"collocation {fr.tgrid.shape} = {len(fr.rtz):,}", t0)
    t0 = time.time()
    fields = profiled("Fields", Fields, fr, cur, pre, kernel=kernel)
    obj = ObjectiveFunction([ForceBalance(), FieldError(0.3), CurrentError(0.1),
                             PressureError(0.1)], fields).build()
    stage(f"Fields + objective (flux grid {fields.J_flux.grid.shape})", t0)

    print("  cached arrays [GB]:")
    groups = {"collocation: J factors + geometry": fields.J_col,
              "collocation: p factors": fields.p_col.f,
              "flux grid: J factors + geometry": fields.J_flux,
              f"box interpolation ({fields.to_box.idx.shape[0]:,} points x "
              f"{fields.to_box.idx.shape[1]})": fields.to_box,
              "kernel K_hat": fields.K_hat, "B_ext, angles, weights": [fields.B_ext, fields.t, fields.sw],
              "DESC targets": fields.desc}
    for k, v in groups.items():
        print(f"    {k:55s} {nbytes(v) / 1e9:8.3f}")
    print(f"    box fields per evaluation (J, J_hat, B: float32)            "
          f"{3 * 3 * np.prod(box.shape) * 4 / 1e9:8.3f}")

    x0 = obj.x0
    rng = np.random.default_rng(0)
    v = jnp.asarray(rng.normal(size=obj.dim_x)) * 1e-3
    w = jnp.asarray(rng.normal(size=obj.dim_f))
    for name, f, args in [("f", obj.compute_scaled_error, (x0,)),
                          ("J v", obj.jvp_scaled_error, (v, x0)),
                          ("J^T w", obj.vjp_scaled_error, (w, x0))]:
        t0 = time.time()
        _, t_c, t_r = timed(f, *args)
        use, peak = gpu()
        print(f"  {name:6s} compile {t_c:6.1f} s, run {t_r:6.2f} s; GPU in use {use:.2f} GB, "
              f"peak {peak:.2f} GB", flush=True)

    u, c_p = fields.split(x0)
    I = fields.I
    # the caches are passed as arguments (pytrees), as in the objective
    comps = [("J at collocation (tensor)", jax.jit(lambda u, m: m(u, I)), (u, fields.J_col)),
             ("p, grad p at collocation (tensor)", jax.jit(lambda c, m: m(c)), (c_p, fields.p_col)),
             ("J on the flux grid (tensor)", jax.jit(lambda u, m: m(u, I)), (u, fields.J_flux)),
             ("J at the box points (interpolation)", jax.jit(lambda u, fl: fl.J_box(u)), (u, fields)),
             ("B on the box (FFT + kernel)",
              jax.jit(lambda J, fl: fl._solve(fft_J(J), fl.axes, fl.K_hat)),
              (fields.J_box(u), fields)),
             ("B to the collocation points (NUFFT)", jax.jit(lambda B, fl: fl._interpolate(B)),
              (jnp.zeros(box.shape + (3,), fields.dtype), fields))]
    print("  components of f: forward and reverse (J^T) run time, after compilation:")
    for name, f, args in comps:
        out, t_c, t_r = timed(f, *args)
        ct = jax.tree_util.tree_map(jnp.ones_like, out)
        back = jax.jit(lambda a, c, ct, f=f: jax.vjp(lambda a: f(a, c), a)[1](ct)[0])
        _, _, t_b = timed(back, args[0], args[1], ct)
        print(f"    {name:42s} {t_r:7.3f} s, reverse {t_b:7.3f} s (compile {t_c:.1f} s)",
              flush=True)
    memory(obj, x0, v, w)
    # one CG iteration of the optimizer: J v and J^T w from the cached linearization
    t0 = time.time()
    r, J, JT = obj.linearize(x0)
    jax.block_until_ready(r)
    t_lin = time.time() - t0
    _, t_c1, t_J = timed(J, v)
    _, t_c2, t_JT = timed(JT, w)
    print(f"  optimizer: linearize {t_lin:.2f} s (with compile), J v {t_J:.3f} s, J^T w "
          f"{t_JT:.3f} s per CG iteration (compile {t_c1:.1f}, {t_c2:.1f} s)", flush=True)


if __name__ == "__main__":
    args = sys.argv[1:]
    kernel = "truncated" if "--kernel" in args and "truncated" in args else "hockney"
    sym = "--full" not in args
    colloc = float(args[args.index("--colloc") + 1]) if "--colloc" in args else None
    fracs = [int(a) for i, a in enumerate(args) if a.isdigit() and args[i - 1] != "--colloc"]
    for frac in fracs or [10]:
        audit(frac, kernel, sym, colloc)
