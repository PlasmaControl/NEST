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

run: python tests/cyl_memory.py [--fracs 10 15 20 30 40 60] [--out results/cyl/memory.json]

NCSX free boundary, the fused product per device with the annulus solve split over n
devices (``CylindricalSolver(mesh=...)``; on the CPU, n fake host devices):

     python tests/cyl_memory.py --ncsx 240 481 --devices 4
"""

import argparse
import json
import os
import sys
import time

import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, "nest"))
if "--devices" in sys.argv:  # n fake host devices (read by XLA when JAX starts)
    os.environ["XLA_FLAGS"] = (os.environ.get("XLA_FLAGS", "") + " --xla_force_host_platform_"
                               f"device_count={sys.argv[sys.argv.index('--devices') + 1]}")

import jax  # noqa: E402

jax.config.update("jax_enable_x64", True)
import jax.numpy as jnp  # noqa: E402

# smooth-current precise_QA (tests/cyl_b4.py), extent (R_min, R_max, Z_min, Z_max) and a
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


# ------------------------------------------------------------------ NCSX, several devices
# the free-boundary annulus of NCSX (walls 2 cm beyond the plasma, then 2 h), the
# measured inside fraction at a/240, the points: 65 x 257 per plane on 85 collocation
# and 43 boundary planes (as the a/240 runs)
NCSX_EXTENT = (1.0256443673030686 - 0.02, 1.7864047374222176 + 0.02,
               -0.631909948918055 - 0.02, 0.631909948918055 + 0.02)
NCSX_A, NCSX_NFP, NCSX_INSIDE = 0.32262239992943775, 3, 0.134
NCSX_P, NCSX_PLANES = 65 * 257, 85 + 43


def ncsx(frac, N_phi, devices=1, chunk=16, rblock=32, n_os=1, order=8):
    """The fused J -> B of the NCSX free boundary per device: J's rows (half sampled,
    mirrored) -> ``Fields._cyl_planes`` (the sources rblock rows of r at a time, z-folded
    float32 S_m) -> ``CylindricalInterpolator.from_planes`` at the moved points (local
    order-8 interpolation), and its transpose; the solve split over ``devices`` devices
    (everything else replicated on them)."""
    from types import SimpleNamespace
    from jax.sharding import Mesh, NamedSharding, PartitionSpec
    from annulus_freespace import ZFold, mode_indices, z_reflection
    from cylindrical_bs import CylindricalSolver, pair_blocks
    from cylindrical_grid import CylindricalGrid, CylindricalInterpolator
    from objective import Fields

    inp = SimpleNamespace(NFP=NCSX_NFP, N=12)
    g = CylindricalGrid.around(inp, NCSX_A / frac, 2.0, extent=NCSX_EXTENT, N_phi=N_phi)
    N, nr, nz, NFP = g.N_phi, g.r_axis.n, g.z_axis.n, NCSX_NFP
    jz, jp = mode_indices(N, True), mode_indices(N, False)
    ms = np.array(sorted(set(np.abs(jz * NFP)) | set(np.abs(jp * NFP + 1))))
    kz, kp = np.searchsorted(ms, jz * NFP), np.searchsorted(ms, np.abs(jp * NFP + 1))
    Nb = 2 * (nz + 1) + 2 * (nr + 1)
    fold = ZFold(z_reflection(SimpleNamespace(
        z_axis=g.z_axis, n_boundary=Nb,
        face_sizes={"in": nz + 1, "out": nz + 1, "bot": nr + 1, "top": nr + 1})))
    f32, f64, i32, c128 = jnp.float32, jnp.float64, jnp.int32, jnp.complex128
    mesh = Mesh(np.array(jax.devices()[:devices]), ("modes",)) if devices > 1 else None
    rep = lambda shape, dt: SDS(shape, dt, sharding=None if mesh is None else  # noqa: E731
                                NamedSharding(mesh, PartitionSpec()))
    static = dict(NFP=NFP, N_phi=N, chunk=chunk, shape=(N, nr + 1, nz + 1), nr=nr, nz=nz,
                  face_sizes=None, jz=jz, jp=jp, zfold=fold, dtype=jnp.dtype(f64), mesh=mesh,
                  mesh_axis=None if mesh is None else "modes")
    leaves = dict(Vz=rep((nz - 1, nz - 1), f64), Vz_inv=rep((nz - 1, nz - 1), f64),
                  lam_z=rep((nz - 1,), f64), ez=rep((nz - 1, 2), f64),
                  Dr=rep((nr + 1, nr + 1), f64), Dz=rep((nz + 1, nz + 1), f64),
                  r=rep((nr + 1,), f64), kz=rep((len(jz),), i32), kp=rep((N,), i32),
                  mz=rep((len(jz),), f64), mp=rep((N,), f64))
    per_mode = dict(Vr=((nr - 1, nr - 1), f64), Vr_inv=((nr - 1, nr - 1), f64),
                    lam_r=((nr - 1,), f64), ar=((nr - 1, 2), f64),
                    S=((fold.n_e, fold.n_e), f32), S_o=((fold.n_o, fold.n_o), f32))
    if mesh is None:
        K = len(ms)
        leaves.update({k: SDS((K,) + sh, dt) for k, (sh, dt) in per_mode.items()})
    else:  # the device's rows on a leading device axis (``CylindricalSolver._distribute``)
        jv, _, use = pair_blocks(N, chunk, devices, kz, kp)
        K, L = max(len(u) for u in use), jv.shape[1]
        c = max(1, chunk // 2)
        dev = lambda shape, dt: SDS(shape, dt, sharding=NamedSharding(  # noqa: E731
            mesh, PartitionSpec("modes")))
        leaves.update({k: dev((devices, K) + sh, dt) for k, (sh, dt) in per_mode.items()})
        pairs = [(devices, L // c, c)] * 8
        leaves["pairs"] = tuple(dev(sh, dt) for sh, dt in zip(pairs, [i32] * 3 + [f64] * 3
                                                                  + [i32] * 2))
        leaves["sel"] = (dev((devices, L), i32), dev((devices, 2 * L), i32))
        static["n_pairs"] = L
    sol = _abstract(CylindricalSolver, static, leaves)
    n_pts = NCSX_P * NCSX_PLANES
    ileaves = dict(Ep=rep((NCSX_PLANES, N), c128), Ez=rep((NCSX_PLANES, N // 2 + 1), c128),
                   yr=rep((NCSX_PLANES, NCSX_P), f64), yz=rep((NCSX_PLANES, NCSX_P), f64),
                   unsort=rep((n_pts,), i32), nodes_r=rep((nr + 1,), f64),
                   nodes_z=rep((nz + 1,), f64), w_r=rep((nr + 1,), f64), w_z=rep((nz + 1,), f64),
                   pad_idx=rep((NCSX_PLANES, NCSX_P), i32), rp=rep((NCSX_PLANES, NCSX_P), f64),
                   zp=rep((NCSX_PLANES, NCSX_P), f64))
    itp = _abstract(CylindricalInterpolator,
                    dict(n=n_pts, n_planes=NCSX_PLANES, P=NCSX_P, order=order,
                         maps=tuple((ax.a, ax.b, ax.alpha) for ax in (g.r_axis, g.z_axis))),
                    ileaves)
    Nf = N * n_os
    size = Nf * (nr + 1) * (nz + 1)
    n_all = int(NCSX_INSIDE * size)
    n_in = n_all // 2
    F = object.__new__(Fields)
    F.n_os, F.shape, F.cyl_rblock, F.cyl_dtype = n_os, (Nf, nr + 1, nz + 1), rblock, jnp.dtype(f64)
    ph = (rep((n_os, N // 2 + 1), c128), rep((n_os, N), c128)) if n_os > 1 else (None, None)

    def fwd(sol, itp, src, rows, ph, J, d):
        F.solver, F.src, (F.ph_z, F.ph_p) = sol, src, ph
        J = jnp.concatenate([J, J[rows] * jnp.asarray([-1.0, 1.0, 1.0], J.dtype)])
        Jr = jnp.pad(J, ((0, 1), (0, 0)))
        return itp.from_planes(*F._cyl_planes(Jr, itp.Ep), d)

    def rev(sol, itp, src, rows, ph, w, d):
        return jax.linear_transpose(lambda J: fwd(sol, itp, src, rows, ph, J, d),
                                    SDS((n_in, 3), f64))(w)[0]

    args = (sol, itp, rep((size,), i32), rep((n_all - n_in,), i32), ph)
    d = rep((n_pts, 2), f64)
    n_dev = max(devices, 1)
    S_GB = len(ms) * (fold.n_e ** 2 + fold.n_o ** 2) * 4 / 1e9
    V_GB = 2 * len(ms) * (nr - 1) ** 2 * 8 / 1e9
    src_GB = (N // 2 + 1 + N) * (nr + 1) * (nz + 1) * 16 / 1e9
    info = dict(frac=frac, N_phi=N, grid=f"{N}x{nr + 1}x{nz + 1}", nodes=size, inside_rows=n_in,
                modes=len(ms), devices=n_dev, rows_per_device=K, S_GB=S_GB, V_GB=V_GB,
                sources_GB=src_GB, planes_GB=NCSX_PLANES * (nr + 1) * (nz + 1) * 24 / 1e9,
                S_V_sources_per_device_GB=(S_GB + V_GB) * K / len(ms) + src_GB / n_dev,
                n_os=n_os, rblock=rblock, chunk=chunk)
    info["fwd"] = analyse(fwd, *args, rep((n_in, 3), f64), d)
    info["rev"] = analyse(rev, *args, rep((n_pts, 3), f64), d)
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
    ap.add_argument("--ncsx", type=float, nargs=2, metavar=("FRAC", "N_PHI"),
                    help="the NCSX free-boundary product per device instead (``ncsx``)")
    ap.add_argument("--devices", type=int, default=1, help="--ncsx: the solve split over n")
    ap.add_argument("--chunk", type=int, default=16, help="--ncsx: modes per step")
    ap.add_argument("--rblock", type=int, default=32, help="--ncsx: cyl_rblock")
    a = ap.parse_args()
    if a.ncsx:
        t0 = time.time()
        r = ncsx(a.ncsx[0], int(a.ncsx[1]), a.devices, chunk=a.chunk, rblock=a.rblock)
        print(json.dumps(r, indent=1, default=str))
        print(f"NCSX a/{a.ncsx[0]:g}, {r['grid']} on {r['devices']} device(s), "
              f"{r['rows_per_device']} of {r['modes']} |m| each: per device J -> B "
              f"{r['fwd']['peak']:.1f} GB, transpose {r['rev']['peak']:.1f} GB  "
              f"[{time.time() - t0:.0f} s]")
        return
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
