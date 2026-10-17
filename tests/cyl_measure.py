"""B5: measured peak memory and time of the J -> B_plasma pipeline (one configuration
per process), to validate the XLA estimate of tests/cyl_memory.py.

The same pipeline as the model, built from the real objects at production sizes on a
synthetic setup (no equilibrium: the cost does not depend on J's values):
the smooth-current precise_QA extent; inside nodes a random subset of the measured
fraction; 1.65M collocation points (99 zeta planes of 65 x 257) at random (r, z)
in an ellipse within the plasma extent.

  cylindrical  CylindricalSolver (S_m float32, z-folded unless --no-zfold) +
               CylindricalInterpolator, in --dtype; --sym gathers the mirror half of J
  box          main's VG box (truncated kernel, float32): gather onto the box, rfft,
               b_plasma_truncated, SpectralInterpolator

Stages: build, fwd (f), rev (J^T w by jax.linear_transpose, as the optimizer); the
peak (device peak_bytes_in_use on a GPU, process ru_maxrss on the CPU) is printed after
each, with the time of a second call.

run: python tests/cyl_measure.py --method cylindrical --frac 20 [--dtype float32] [--sym]
     [--out results/cyl/measure/<name>.json]
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

import jax  # noqa: E402

jax.config.update("jax_enable_x64", True)
import jax.numpy as jnp  # noqa: E402

from cyl_memory import A, BOX_INSIDE, CYL_INSIDE, EXTENT, N_RHO, N_THETA, N_ZETA, NFP  # noqa: E402

T0 = time.time()


def peak():
    """The device peak on an accelerator (XLA memory_stats), else the process peak RSS."""
    d = jax.devices()[0]
    if d.platform != "cpu":
        st = d.memory_stats() or {}
        if "peak_bytes_in_use" in st:
            return st["peak_bytes_in_use"] / 1e9
    r = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    return r / 1e9 if sys.platform == "darwin" else r / 1e6


def log(msg):
    print(f"[{time.time() - T0:7.1f} s, peak {peak():5.2f} GB] {msg}", flush=True)


def points(rng):
    """(r, phi, z) of 1.65M points: N_ZETA planes over the period, P per plane."""
    P = N_RHO * N_THETA
    R0, a_r = 0.5 * (EXTENT[0] + EXTENT[1]), 0.45 * (EXTENT[1] - EXTENT[0])
    a_z = 0.45 * (EXTENT[3] - EXTENT[2])
    rho, th = np.sqrt(rng.random((N_ZETA, P))), 2 * np.pi * rng.random((N_ZETA, P))
    phi = np.repeat(2 * np.pi * np.arange(N_ZETA) / (NFP * N_ZETA), P).reshape(N_ZETA, P)
    return np.stack([R0 + a_r * rho * np.cos(th), phi, a_z * rho * np.sin(th)], -1).reshape(-1, 3)


def timed(f, *args):
    out = jax.block_until_ready(f(*args))  # compile + first call
    t0 = time.time()
    out = jax.block_until_ready(f(*args))
    return out, time.time() - t0


def cylindrical(frac, dtype, sym, zfold, rng):
    from cylindrical_bs import CylindricalSolver
    from cylindrical_grid import CylindricalGrid, CylindricalInterpolator

    class _Inp:
        pass
    inp = _Inp()
    inp.NFP = NFP
    g = CylindricalGrid.around(inp, A / frac, 2.0, extent=EXTENT)
    dt = jnp.dtype(dtype)
    log(f"cylindrical a/{frac}: grid {g.shape} ({g.size / 1e6:.1f}M nodes), {dtype}, "
        f"sym {sym}, zfold {zfold}")
    from annulus_freespace import FreeSpaceAnnulus
    fs = FreeSpaceAnnulus(g.r_axis, g.z_axis, NFP, g.N_phi, shifts=(0, 1), verbose=True,
                          S_dtype=np.float32, zsym="auto" if zfold else False)
    sol = CylindricalSolver(g.r_axis, g.z_axis, NFP, g.N_phi, fs=fs, dtype=dt)
    del fs
    log(f"solver built: S_m {sol.S_nbytes / 1e9:.2f} GB")
    itp = CylindricalInterpolator(g, points(rng), dtype=dt)
    log("interpolator built")
    n_all = int(CYL_INSIDE * g.size)
    inside = np.sort(rng.choice(g.size, n_all, replace=False))
    n = n_all // 2 if sym else n_all
    src = np.full(g.size, n_all, np.int32)
    src[inside] = np.arange(n_all)
    src = jnp.asarray(src)
    rows = jnp.asarray(rng.choice(n, n_all - n, replace=n_all - n > n)) if sym else None
    shape = g.shape

    def fwd(sol, itp, src, J):
        if sym:
            J = jnp.concatenate([J, J[rows] * jnp.asarray([-1.0, 1.0, 1.0], J.dtype)])
        Jg = jnp.pad(J, ((0, 1), (0, 0)))[src].reshape(shape + (3,))
        return itp(*sol.field_modes(Jg))

    J = jnp.asarray(rng.standard_normal((n, 3)), dt)
    return dict(grid=list(shape), nodes=g.size, inside=n, S_GB=sol.S_nbytes / 1e9), fwd, (sol, itp, src), J


def box(frac, rng):
    from scipy.fft import next_fast_len
    from fourier import fft_J
    from magnetic_field import b_plasma_truncated, truncated_green_hat
    from objective import SpectralInterpolator
    h = A / frac
    R_max = EXTENT[1]
    ext = [2 * R_max, 2 * R_max, EXTENT[3] - EXTENT[2]]
    n = []
    for e in ext:
        k = next_fast_len(int(np.ceil(2.0 * e / h)))
        while k % 2 == 0:
            k = next_fast_len(k + 1)
        n.append(k)
    shape = tuple(n)
    axes = tuple(h * (np.arange(m) - m // 2) for m in shape)
    size = int(np.prod(shape))
    log(f"box a/{frac}: {shape} ({size / 1e6:.1f}M points), float32")
    f32 = jnp.float32
    K_hat = truncated_green_hat(axes, dtype=f32)
    log("K_hat built")
    rpz = points(rng)
    x = np.column_stack([rpz[:, 0] * np.cos(rpz[:, 1]), rpz[:, 0] * np.sin(rpz[:, 1]), rpz[:, 2]])
    x0 = np.array([a[0] for a in axes])
    L = np.array([len(a) * (a[1] - a[0]) for a in axes])
    t = 2 * np.pi * (x - x0) / L
    sp = SpectralInterpolator(jnp.asarray((t + np.pi) % (2 * np.pi) - np.pi, f32), shape, f32, 1e-6,
                              chunk=4 * 65536)
    log("SpectralInterpolator built")
    n_all = int(BOX_INSIDE * size)
    n_in = n_all // 2
    inside = np.sort(rng.choice(size, n_all, replace=False))
    src = np.full(size, n_all, np.int32)
    src[inside] = np.arange(n_all)
    src = jnp.asarray(src)
    rows = jnp.asarray(rng.choice(n_in, n_all - n_in, replace=False))

    def fwd(K_hat, sp, src, J):
        J = jnp.concatenate([J, J[rows] * jnp.asarray([-1.0, 1.0, 1.0], f32)])
        J = jnp.pad(J, ((0, 1), (0, 0)))[src].reshape(shape + (3,))
        solve = jax.checkpoint(lambda Jh, K: b_plasma_truncated(Jh, axes, K),
                               policy=jax.checkpoint_policies.nothing_saveable)
        return sp(solve(fft_J(J), K_hat))

    J = jnp.asarray(rng.standard_normal((n_in, 3)), f32)
    return dict(grid=list(shape), nodes=size, inside=n_in), fwd, (K_hat, sp, src), J


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--method", choices=["cylindrical", "box"], required=True)
    ap.add_argument("--frac", type=int, required=True)
    ap.add_argument("--dtype", default="float32", choices=["float32", "float64"])
    ap.add_argument("--sym", action="store_true")
    ap.add_argument("--no-zfold", action="store_true")
    ap.add_argument("--stages", nargs="+", default=["fwd", "rev"])
    ap.add_argument("--out")
    a = ap.parse_args()
    rng = np.random.default_rng(0)
    log(f"start (JAX {jax.__version__}, {jax.devices()[0].platform})")
    if a.method == "cylindrical":
        info, fwd, ops, J = cylindrical(a.frac, a.dtype, a.sym, not a.no_zfold, rng)
    else:
        info, fwd, ops, J = box(a.frac, rng)
    info.update(method=a.method, frac=a.frac, dtype=a.dtype if a.method == "cylindrical" else "float32",
                sym=a.sym, zfold=not a.no_zfold, build_peak_GB=peak(), build_s=time.time() - T0)
    log(f"built: {info['grid']}, {info['inside']:,} J rows")
    f = jax.jit(fwd)
    if "fwd" in a.stages:
        B, t = timed(f, *ops, J)
        info.update(fwd_s=t, fwd_peak_GB=peak())
        log(f"fwd: {t:.2f} s per call, B {B.shape} {B.dtype}")
        del B
    if "rev" in a.stages:
        w = jnp.asarray(rng.standard_normal((N_RHO * N_THETA * N_ZETA, 3)))
        r = jax.jit(lambda ops, w: jax.linear_transpose(lambda J: fwd(*ops, J), J)(w)[0])
        g, t = timed(r, ops, w)
        info.update(rev_s=t, rev_peak_GB=peak())
        log(f"J^T w: {t:.2f} s per call, {g.shape} {g.dtype}")
    print(json.dumps(info), flush=True)
    if a.out:
        os.makedirs(os.path.dirname(a.out), exist_ok=True)
        json.dump(info, open(a.out, "w"), indent=1)


if __name__ == "__main__":
    main()
