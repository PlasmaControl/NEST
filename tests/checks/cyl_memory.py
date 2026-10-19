"""B5: memory of the J -> B_plasma program, cylindrical annulus against main's box,
from XLA's compiled memory analysis (abstract shapes, nothing is run).

For each h = a / frac, the jitted programs of ``Fields.B_plasma``'s data path are
lowered with abstract operands of the production sizes and compiled; XLA's
``memory_analysis()`` gives the bytes of the arguments (the cached operators and
the input), the output and the temporaries; the peak used is their sum (an upper
bound: XLA's own CPU peak_memory_in_bytes omits the temporaries):

  cylindrical   J at the inside nodes (n_in, 3) f64 -> gather onto the annulus ->
                ``CylindricalSolver.field_modes`` (S_m float32, the rest float64) ->
                ``CylindricalInterpolator`` -> (n_pts, 3)
  box           J at the half-box inside points (n_in, 3) f32 -> mirror -> gather onto
                the periodic box -> rfft -> VG kernel (``b_plasma_truncated``, K_hat
                f32) -> ``SpectralInterpolator`` (type-2 NUFFT, eps 1e-6) -> (n_pts, 3),
                the float32 production path of main

both forward (f, J v) and transposed (J^T w: ``jax.linear_transpose``, as the
optimizer). Not included: what both share (the flux-grid J factors and geometry,
the collocation-side caches), and J sampling (``PointInterpolator``: 64 stencil
indices and weights per inside point), which is reported separately per method.

Points: main's production collocation grid (``--colloc 2`` with factors 3, 3, 2.5 on
L, M, N = 16, 16, 14): 65 x 257 x 99 = 1.65M points on 99 zeta planes. Inside
fractions are the measured ones (cylindrical ~8% of the nodes, the box 1.25%,
halved by stellarator symmetry). Sizes from the smooth-current precise_QA extent.

run: python tests/checks/cyl_memory.py [--fracs 10 15 20 30 40 60] [--out results/cyl/memory.json]
"""

import argparse
import json
import os
import sys
import time

import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.join(ROOT, "nest"))

import jax  # noqa: E402

jax.config.update("jax_enable_x64", True)
import jax.numpy as jnp  # noqa: E402

# smooth-current precise_QA (tests/checks/cyl_b4.py), extent (R_min, R_max, Z_min, Z_max) and a
EXTENT = (0.6435142741280147, 1.3392634816381634, -0.4078154525697383, 0.4078154525697381)
A = 0.17285
NFP = 2
N_RHO, N_THETA, N_ZETA = 65, 257, 99  # production collocation (sym half)
CYL_INSIDE, BOX_INSIDE = 0.08, 0.0125
SDS = jax.ShapeDtypeStruct


def analyse(fn, *args):
    t0 = time.time()
    c = jax.jit(fn).lower(*args).compile()
    m = c.memory_analysis()
    out = {"args": m.argument_size_in_bytes / 1e9, "out": m.output_size_in_bytes / 1e9,
           "temp": m.temp_size_in_bytes / 1e9, "xla_peak": m.peak_memory_in_bytes / 1e9,
           "compile_s": time.time() - t0}
    # XLA's CPU peak_memory_in_bytes leaves out the temporaries (it equals args + out);
    # the estimate used here is the upper bound args + out + temp
    out["peak"] = out["args"] + out["out"] + out["temp"]
    return out


def _abstract(cls, static, leaves):
    """An instance of a _Pytree class with abstract leaves (no data)."""
    obj = object.__new__(cls)
    for k, v in static.items():
        object.__setattr__(obj, k, v)
    for k in cls._dynamic:
        object.__setattr__(obj, k, leaves.get(k))
    return obj


# ------------------------------------------------------------------ cylindrical
def cylindrical(frac, chunk=16, dtype="float64", sym=False, zfold=False):
    from types import SimpleNamespace
    from annulus_freespace import ZFold, mode_indices, z_reflection
    from cylindrical_bs import CylindricalSolver
    from cylindrical_grid import CylindricalGrid, CylindricalInterpolator

    class _Inp:  # what CylindricalGrid.around needs
        pass
    inp = _Inp()
    inp.NFP = NFP
    g = CylindricalGrid.around(inp, A / frac, 2.0, extent=EXTENT)
    N, nr, nz = g.N_phi, g.r_axis.n, g.z_axis.n
    jz, jp = mode_indices(N, True), mode_indices(N, False)
    Kd = len(set(np.abs(jz * NFP)) | set(np.abs(jp * NFP + 1)))
    Nb = 2 * (nz + 1) + 2 * (nr + 1)
    fold = None
    if zfold:  # the z -> -z fold of S_m (FreeSpaceAnnulus zsym)
        fold = ZFold(z_reflection(SimpleNamespace(
            z_axis=g.z_axis, n_boundary=Nb,
            face_sizes={"in": nz + 1, "out": nz + 1, "bot": nr + 1, "top": nr + 1})))
    S_shapes = [(Kd, Nb, Nb)] if fold is None else [(Kd, fold.n_e, fold.n_e), (Kd, fold.n_o, fold.n_o)]
    f32, i32 = jnp.float32, jnp.int32
    f64 = jnp.dtype(dtype)  # the solver's arithmetic (float32: complex64)
    c128 = jnp.complex64 if f64 == jnp.float32 else jnp.complex128
    leaves = dict(Vr=SDS((Kd, nr - 1, nr - 1), f64), Vr_inv=SDS((Kd, nr - 1, nr - 1), f64),
                  lam_r=SDS((Kd, nr - 1), f64), ar=SDS((Kd, nr - 1, 2), f64),
                  Vz=SDS((nz - 1, nz - 1), f64), Vz_inv=SDS((nz - 1, nz - 1), f64),
                  lam_z=SDS((nz - 1,), f64), ez=SDS((nz - 1, 2), f64), S=SDS(S_shapes[0], f32),
                  S_o=None if fold is None else SDS(S_shapes[1], f32),
                  Dr=SDS((nr + 1, nr + 1), f64), Dz=SDS((nz + 1, nz + 1), f64), r=SDS((nr + 1,), f64),
                  kz=SDS((len(jz),), i32), kp=SDS((N,), i32), mz=SDS((len(jz),), f64),
                  mp=SDS((N,), f64))
    static = dict(NFP=NFP, N_phi=N, chunk=chunk, shape=(N, nr + 1, nz + 1), nr=nr, nz=nz,
                  face_sizes=None, jz=jz, jp=jp, zfold=fold, dtype=f64)
    sol = _abstract(CylindricalSolver, static, leaves)
    P = N_RHO * N_THETA
    n_pts = P * N_ZETA
    ileaves = dict(Ep=SDS((N_ZETA, N), c128), Ez=SDS((N_ZETA, N // 2 + 1), c128),
                   yr=SDS((N_ZETA, P), f64), yz=SDS((N_ZETA, P), f64), unsort=SDS((n_pts,), i32),
                   nodes_r=SDS((nr + 1,), f64), nodes_z=SDS((nz + 1,), f64),
                   w_r=SDS((nr + 1,), f64), w_z=SDS((nz + 1,), f64))
    itp = _abstract(CylindricalInterpolator, dict(n=n_pts, n_planes=N_ZETA, P=P), ileaves)
    size = N * (nr + 1) * (nz + 1)
    n_all = int(CYL_INSIDE * size)
    n_in = n_all // 2 if sym else n_all  # sampled rows (Fields cyl_half: mirrors gathered)
    shape = (N, nr + 1, nz + 1)

    def fwd(sol, itp, src, rows, J):
        if sym:
            J = jnp.concatenate([J, J[rows] * jnp.asarray([-1.0, 1.0, 1.0], J.dtype)])
        Jg = jnp.pad(J, ((0, 1), (0, 0)))[src].reshape(shape + (3,))
        return itp(*sol.field_modes(Jg))

    def rev(sol, itp, src, rows, w):
        return jax.linear_transpose(lambda J: fwd(sol, itp, src, rows, J), SDS((n_in, 3), f64))(w)[0]

    src, rows = SDS((size,), i32), SDS((n_all - n_in,), i32)
    isz = jnp.dtype(f64).itemsize
    info = dict(frac=frac, grid=f"{nr + 1}x{nz + 1}x{N}", nodes=size, inside=n_in, modes=Kd, N_b=Nb,
                dtype=str(jnp.dtype(f64)), sym=sym, zfold=zfold,
                S_GB=sum(np.prod(sh) for sh in S_shapes) * 4 / 1e9, J_sampling_GB=n_in * 64 * (4 + 2 * isz) / 1e9)
    info["fwd"] = analyse(fwd, sol, itp, src, rows, SDS((n_in, 3), f64))
    info["rev"] = analyse(rev, sol, itp, src, rows, SDS((n_pts, 3), jnp.float64))
    return info


# ------------------------------------------------------------------ box
def box(frac, chunk=4 * 65536):
    from scipy.fft import next_fast_len
    from fourier import fft_J
    from magnetic_field import b_plasma_truncated
    from nufftax.core.kernel import compute_kernel_params, kernel_fourier_series
    from nufftax.utils.grid import compute_grid_size
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
    n_in = int(BOX_INSIDE * size / 2)  # the half box (stellarator symmetry)
    n_pts = N_RHO * N_THETA * N_ZETA
    f32, i32 = jnp.float32, jnp.int32
    eps, up = 1e-6, 1.25
    kp = compute_kernel_params(eps, up)
    nf = tuple(compute_grid_size(m, up, kp.nspread) for m in shape)
    phihat = tuple(np.asarray(kernel_fourier_series(m, kp.nspread, kp.beta, kp.c, dtype=np.float32))
                   for m in nf)
    ch = min(chunk, n_pts)
    k = -(-n_pts // ch)
    sp = _abstract(SpectralInterpolator,
                   dict(shape=shape, dtype=f32, chunk=ch, cdtype=jnp.complex64, ns=kp.nspread, nf=nf,
                        phihat=phihat, n=n_pts, order=None),
                   dict(idx=SDS((k, ch, 3, kp.nspread), i32), w=SDS((k, ch, 3, kp.nspread), f32),
                        order=None))
    half = n_in  # mirrored rows ~ the half count

    def fwd(K_hat, sp, src, rows, J):
        J = jnp.concatenate([J, J[rows] * jnp.asarray([-1.0, 1.0, 1.0], f32)])
        J = jnp.pad(J, ((0, 1), (0, 0)))[src].reshape(shape + (3,))
        solve = jax.checkpoint(lambda Jh, K: b_plasma_truncated(Jh, axes, K),
                               policy=jax.checkpoint_policies.nothing_saveable)
        return sp(solve(fft_J(J), K_hat))

    def rev(K_hat, sp, src, rows, w):
        return jax.linear_transpose(lambda J: fwd(K_hat, sp, src, rows, J), SDS((n_in, 3), f32))(w)[0]

    K = SDS((shape[0], shape[1], shape[2] // 2 + 1), f32)
    src, rows = SDS((size,), i32), SDS((half,), i32)
    info = dict(frac=frac, grid="x".join(map(str, shape)), nodes=size, inside=n_in,
                J_sampling_GB=n_in * 64 * (4 + 4 + 4) / 1e9)
    info["fwd"] = analyse(fwd, K, sp, src, rows, SDS((n_in, 3), f32))
    info["rev"] = analyse(rev, K, sp, src, rows, SDS((n_pts, 3), jnp.float64))
    return info


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--fracs", type=int, nargs="+", default=[10, 15, 20, 30, 40, 60])
    ap.add_argument("--methods", nargs="+", default=["cylindrical", "box"])
    ap.add_argument("--out", default=os.path.join(ROOT, "results", "cyl", "memory.json"))
    ap.add_argument("--cyl-dtype", default="float64", choices=["float64", "float32"])
    ap.add_argument("--cyl-sym", action="store_true", help="Fields cyl_half (sample half)")
    ap.add_argument("--cyl-zfold", action="store_true", help="S_m z-fold (half storage)")
    a = ap.parse_args()
    rows = []
    for frac in a.fracs:
        for meth in a.methods:
            t0 = time.time()
            r = (cylindrical(frac, dtype=a.cyl_dtype, sym=a.cyl_sym, zfold=a.cyl_zfold)
                 if meth == "cylindrical"
                 else box(frac))
            r["method"] = meth
            rows.append(r)
            print(f"a/{frac:<3d} {meth:12s} {r['grid']:>16s} ({r['nodes'] / 1e6:7.1f}M): "
                  f"fwd peak {r['fwd']['peak']:6.2f} GB (args {r['fwd']['args']:.2f}, temp "
                  f"{r['fwd']['temp']:.2f}); J^T w peak {r['rev']['peak']:6.2f} GB (temp "
                  f"{r['rev']['temp']:.2f}); J sampling {r['J_sampling_GB']:.2f} GB"
                  + (f"; S_m {r['S_GB']:.2f} GB" if meth == "cylindrical" else "")
                  + f"  [{time.time() - t0:.0f} s]", flush=True)
            os.makedirs(os.path.dirname(a.out), exist_ok=True)
            json.dump(rows, open(a.out, "w"), indent=1)


if __name__ == "__main__":
    main()
