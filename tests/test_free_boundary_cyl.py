"""The free boundary on the cylindrical annulus (``FreeBoundaryFields`` with a
``CylindricalGrid``): the same checks as ``test_free_boundary.py`` for the box.

NCSX (DESC re-solve in its coils) at a low resolution, annulus h = a / frac with
N_phi planes:
  0. ``StencilInterpolator`` against ``PointInterpolator`` (values) and against
     ``BoxShift`` (rho and theta derivatives) at the grid's inside nodes
  1. xi = 0: ``FreeBoundaryFields`` against ``Fields`` at x0 (F, B, J, p)
  2. J v against central differences along u, c_p, xi and all; the adjoint
     identity <w, J v> = <J^T w, v>
  3. first order in D: the fields at xi against those rebuilt exactly on the
     displaced geometry (``Displacement.apply``, xi = 0, the same annulus), for xi
     and 2 xi; the difference must scale as |xi|^2
  4. B.n / |B| of the start on the boundary: annulus against direct Biot-Savart
  5. (--devices n > 1) the annulus solve split over n devices (``cyl_mesh``):
     B_plasma and its transpose, and the free-boundary fields, J v and J^T w, against
     the single-device ones (on the CPU, n fake host devices)

run: python tests/test_free_boundary_cyl.py [frac] [--N-phi 15] [--eq file] [--devices 4]
"""

import argparse
import os
import sys
import time

import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, "nest"))

if "--devices" in sys.argv and os.environ.get("NEST_DEVICE", "cpu") == "cpu":
    # n fake host devices for check 5 (read by XLA when JAX starts)
    n_dev = sys.argv[sys.argv.index("--devices") + 1]
    os.environ["XLA_FLAGS"] = (os.environ.get("XLA_FLAGS", "")
                               + f" --xla_force_host_platform_device_count={n_dev}")

from desc import set_device  # noqa: E402

_visible = os.environ.get("CUDA_VISIBLE_DEVICES")
set_device(os.environ.get("NEST_DEVICE", "cpu"))
if "--devices" in sys.argv and _visible and os.environ.get("NEST_DEVICE", "cpu") == "gpu":
    os.environ["CUDA_VISIBLE_DEVICES"] = _visible  # set_device keeps only one GPU

import jax  # noqa: E402
import jax.numpy as jnp  # noqa: E402

import fit_cache  # noqa: E402
from boundary_potential import BoundaryPotential  # noqa: E402
from cylindrical_grid import CylindricalGrid  # noqa: E402
from desc_input import DESCInput  # noqa: E402
from flux_interp import PointInterpolator, StencilInterpolator  # noqa: E402
from force import ForceResidual  # noqa: E402
from free_boundary import (BoundaryNormalField, BoxShift, Displacement,  # noqa: E402
                           FreeBoundaryFields, FreeBoundaryObjective, StoredEnergy, ToroidalFlux,
                           TransportedCurrentError, reference)
from grid import plasma_extent  # noqa: E402
from objective import Fields, ForceBalance, PressurePositivity  # noqa: E402
from pressure import Pressure  # noqa: E402

jax.config.update("jax_enable_x64", True)
D = os.path.join(ROOT, "desc_input", "vmec_free_boundary", "NCSX")
EQ, COILS = os.path.join(D, "NCSX_L12M12N8.h5"), os.path.join(D, "NCSX_coils.h5")


def annulus(inp, h, N_phi, pad):
    """The annulus around the plasma's (R, Z) extent widened by pad [m] (room for the
    boundary to move), N_phi planes."""
    R0, R1, Z0, Z1 = plasma_extent(inp)
    Z = max(abs(Z0), abs(Z1)) + pad
    return CylindricalGrid.around(inp, h, extent=(R0 - pad, R1 + pad, -Z, Z), N_phi=N_phi)


FUSED = False  # --fused: B by CylindricalSolver.planes_F
RBLOCK = 0  # --rblock: the sources rb rows of r at a time
ORDER = None  # --order: local interpolation to the points


def build(inp, cur, pre, ref, grid, res, M=4, N=3, solver=None, direct=None, n_os=1,
          fused=None):
    g = CylindricalGrid(grid.r_axis, grid.z_axis, grid.NFP, grid.N_phi)
    g.locate_plasma(inp, sym=bool(inp.eq.sym))
    fr = ForceResidual(inp, g, resolution=res)
    disp = Displacement(inp.eq, M, N)
    fb = FreeBoundaryFields(fr, cur, pre, disp, ref, dtype=jnp.float64, cyl_stencil=True,
                            cyl_solver=solver, direct=direct, verbose=False,
                            cyl_oversample=n_os, cyl_fused=FUSED if fused is None else fused,
                            cyl_rblock=RBLOCK, cyl_order=ORDER)
    return fb, fr, g


def rebase(cur, pre, eq):
    """The fits' structure and coefficients on another geometry."""
    return (BoundaryPotential(cur.basis, cur.u, cur.I, eq, cur.edge_power, cur.S),
            Pressure(pre.basis, pre.c, eq, pre.edge_order))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("frac", type=float, nargs="?", default=6)
    ap.add_argument("--N-phi", type=int, default=15)
    ap.add_argument("--fused", action="store_true", help="cyl_fused (CylindricalSolver.planes_F)")
    ap.add_argument("--rblock", type=int, default=0, help="cyl_rblock")
    ap.add_argument("--order", type=int, default=0, help="cyl_order (0: global)")
    ap.add_argument("--oversample", type=int, default=1,
                    help="n_os: J on n_os N_phi planes, the solve on N_phi (``cyl_oversample``)")
    ap.add_argument("--eq", default=EQ)
    ap.add_argument("--coils", default=COILS)
    ap.add_argument("--factors", type=float, nargs=3, default=(1, 1, 1))
    ap.add_argument("--xi", type=float, default=1e-3, help="largest displacement of check 3 [m]")
    ap.add_argument("--devices", type=int, default=1,
                    help="check 5: the annulus solve on this many devices (CPU: fake ones)")
    ap.add_argument("--only-devices", action="store_true", help="checks 0 and 5 only")
    a = ap.parse_args()
    t0 = time.time()
    inp = DESCInput.load(a.eq, field=a.coils)
    eq = inp.eq
    cur = fit_cache.fit(BoundaryPotential, inp, factors=tuple(a.factors))
    pre = fit_cache.fit(Pressure, inp, factors=tuple(a.factors), edge_order=2)
    b = cur.basis
    res = tuple(int(u + v) for u, v in ((b.L, eq.L), (b.M, eq.M), (b.N, eq.N)))
    ref = reference(inp, res)
    global FUSED, RBLOCK, ORDER
    FUSED, RBLOCK, ORDER = a.fused, a.rblock, a.order or None
    n_os = a.oversample
    grid = annulus(inp, inp.a / a.frac, a.N_phi * n_os, 0.02)
    print(f"{inp}\nresolution of F {res}, {grid}; setup {time.time() - t0:.0f} s", flush=True)
    rng = np.random.default_rng(0)

    # 1. xi = 0 against the fixed-boundary fields
    fb, fr, g = build(inp, cur, pre, ref, grid, res, n_os=n_os)
    print(f"   built: {len(g.inside):,} inside nodes, {fb.to_box.n:,} interpolated; "
          f"{time.time() - t0:.0f} s", flush=True)

    # 0. the stencils
    pi = PointInterpolator(fb.J_flux.grid, fb.rows_rtz, 4, dtype=jnp.float64)
    si = fb.to_box
    assert isinstance(si, StencilInterpolator)
    f = jnp.asarray(rng.normal(size=fb.J_flux.grid.shape + (5,)))
    par = (-1, 1, 1, 1, -1)
    v_p, v_s = np.asarray(pi(f, par)), si(f, par, derivs=True)
    e0 = float(np.abs(np.asarray(v_s[0]) - v_p).max() / np.abs(v_p).max())
    bs = BoxShift(fb.J_flux.grid, fb.rows_rtz, cur._RZ, fb.disp, 4, jnp.float64)
    v_b = bs(f, par)
    e0d = max(float(np.abs(np.asarray(v_s[i]) - np.asarray(v_b[i])).max()
                    / np.abs(np.asarray(v_b[i])).max()) for i in (1, 2))
    print(f"0. stencils: values vs PointInterpolator {e0:.1e}, rho and theta derivatives vs "
          f"BoxShift {e0d:.1e}", flush=True)
    assert e0 < 1e-13 and e0d < 1e-12
    del pi, bs

    if n_os > 1:  # 0b. the oversampled solve against the full-grid solve of J's low modes
        from cylindrical_grid import CylindricalInterpolator
        from objective import _cartesian
        full = Fields(fr, cur, pre, dtype=jnp.float64, cyl_stencil=True, cyl_order=ORDER)
        u0 = jnp.asarray(cur.u)
        Fz, Fp = full.solver.sources(full.J_box(u0))
        jz, jp, jm = np.asarray(full.solver.jz), np.asarray(full.solver.jp), (a.N_phi - 1) // 2
        Bp, Bz = full.solver.field_modes_F(Fz * jnp.asarray(np.abs(jz) <= jm)[:, None, None],
                                           Fp * jnp.asarray(np.abs(jp) <= jm)[:, None, None])
        B_mask = np.asarray(_cartesian(full.to_points(Bp, Bz), full.phi))
        B_os = np.asarray(fb.fixed_view().B_plasma(u0))
        B_all = np.asarray(_cartesian(full.to_points(*full.solver.field_modes(full.J_box(u0))),
                                      full.phi))
        e0b = float(np.abs(B_os - B_mask).max() / np.abs(B_mask).max())
        print(f"0b. oversampled ({a.N_phi} planes of {a.N_phi * n_os}) vs the full grid's J "
              f"modes |j| <= {jm}: {e0b:.1e}; the modes above change B_plasma by "
              f"{np.abs(B_all - B_mask).max() / np.abs(B_mask).max():.1e}", flush=True)
        assert e0b < 1e-12
        del full
    plain = Fields(fr, cur, pre, dtype=jnp.float64, cyl_stencil=True, cyl_solver=fb.solver,
                   cyl_oversample=n_os, cyl_fused=FUSED, cyl_rblock=RBLOCK, cyl_order=ORDER)
    # 0c. B on the planes chunk by chunk (planes_F) against the full modal slabs
    other = Fields(fr, cur, pre, dtype=jnp.float64, cyl_stencil=True, cyl_solver=fb.solver,
                   cyl_oversample=n_os, cyl_fused=not FUSED, cyl_rblock=0 if RBLOCK else 7,
                   cyl_order=ORDER)
    u0 = jnp.asarray(cur.u)
    B1, B2 = np.asarray(plain.B_plasma(u0)), np.asarray(other.B_plasma(u0))
    w = jnp.asarray(rng.normal(size=B1.shape))
    T1 = jax.linear_transpose(lambda u: plain.B_plasma(u, 0.0), u0)(w)[0]
    T2 = jax.linear_transpose(lambda u: other.B_plasma(u, 0.0), u0)(w)[0]
    e0c = float(np.abs(B1 - B2).max() / np.abs(B2).max())
    e0t = float(jnp.abs(T1 - T2).max() / jnp.abs(T2).max())
    print(f"0c. fused {FUSED} rblock {RBLOCK} vs fused {not FUSED} rblock {0 if RBLOCK else 7}: "
          f"B_plasma {e0c:.1e}, its transpose {e0t:.1e}", flush=True)
    assert e0c < 1e-12 and e0t < 1e-12
    del other
    # 0d. local interpolation to the points (order 8) against the global one
    loc = Fields(fr, cur, pre, dtype=jnp.float64, cyl_stencil=True, cyl_solver=fb.solver,
                 cyl_oversample=n_os, cyl_fused=FUSED, cyl_rblock=RBLOCK,
                 cyl_order=None if ORDER else 8)
    B3 = np.asarray(loc.B_plasma(u0))
    print(f"0d. interpolation order {ORDER} vs {None if ORDER else 8}: B_plasma "
          f"{np.abs(B1 - B3).max() / np.abs(B3).max():.1e} max, "
          f"{np.sqrt(np.mean((B1 - B3) ** 2)) / np.sqrt(np.mean(B3 ** 2)):.1e} rms", flush=True)
    del loc
    if a.devices > 1:
        devices(a, inp, cur, pre, ref, grid, res, fr, fb, plain, n_os)
        if a.only_devices:
            return
    f0 = {k: np.asarray(v) for k, v in fb.compute(fb.x0).items()}
    p0 = {k: np.asarray(v) for k, v in plain.compute(plain.x0).items()}
    e1 = {k: float(np.abs(f0[k] - p0[k]).max() / np.abs(p0[k]).max()) for k in ("F", "B", "J", "p")}
    Jd = np.asarray(plain.desc["J"])
    e1["J_desc"] = float(np.abs(f0["J_desc"] - Jd).max() / np.abs(Jd).max())
    print("1. xi = 0 vs Fields: " + ", ".join(f"{k} {v:.1e}" for k, v in e1.items())
          + f"; {time.time() - t0:.0f} s", flush=True)

    # 2. derivative
    objs = [ForceBalance(), TransportedCurrentError(0.1), StoredEnergy(1.0),
            PressurePositivity(1.0), BoundaryNormalField(1.0), ToroidalFlux(1.0)]
    obj = FreeBoundaryObjective(objs, fb).build()
    x0 = obj.x0
    n_u, n_p, n_xi = fb.n_u, fb.n_p, fb.n_xi
    rms = lambda v: np.sqrt(np.mean(np.asarray(v) ** 2))  # noqa: E731
    scale = np.concatenate([np.full(n_u, rms(x0[:n_u])), np.full(n_p, rms(x0[n_u : n_u + n_p])),
                            np.full(n_xi, 1e-3 / np.sqrt(n_xi))])
    idx = np.arange(obj.dim_x)
    e2 = {}
    for name, mask in [("u", idx < n_u), ("c_p", (idx >= n_u) & (idx < n_u + n_p)),
                       ("xi", idx >= n_u + n_p), ("all", idx >= 0)]:
        v = jnp.asarray(1e-4 * scale * rng.normal(size=obj.dim_x) * mask)
        Jv = obj.jvp_scaled_error(v, x0)
        fd = (obj.compute_scaled_error(x0 + v) - obj.compute_scaled_error(x0 - v)) / 2
        e2[name] = float(jnp.linalg.norm(Jv - fd) / jnp.linalg.norm(fd))
    xi = rng.normal(size=n_xi) * 1e-3 / np.sqrt(n_xi)
    x1 = x0.at[n_u + n_p :].set(jnp.asarray(xi))
    v = jnp.asarray(1e-4 * scale * rng.normal(size=obj.dim_x))
    Jv = obj.jvp_scaled_error(v, x1)
    fd = (obj.compute_scaled_error(x1 + v) - obj.compute_scaled_error(x1 - v)) / 2
    e2["all at xi != 0"] = float(jnp.linalg.norm(Jv - fd) / jnp.linalg.norm(fd))
    w = jnp.asarray(rng.normal(size=obj.dim_f))
    lhs, rhs = float(jnp.vdot(w, Jv)), float(jnp.vdot(obj.vjp_scaled_error(w, x1), v))
    e_adj = abs(lhs - rhs) / abs(lhs)
    print("2. J v vs central differences: " + ", ".join(f"{k} {e:.1e}" for k, e in e2.items())
          + f"; adjoint {e_adj:.1e}; {time.time() - t0:.0f} s", flush=True)

    # 3. first order in D: against the fields rebuilt on the displaced geometry
    from desc.grid import LinearGrid
    xi_n = rng.normal(size=n_xi)
    Fb = np.asarray(fb.disp.basis.evaluate(LinearGrid(rho=1.0, M=2 * eq.M, N=2 * eq.N,
                                                      NFP=eq.NFP)))
    e3 = []
    for s in (1.0, 2.0):
        xs = 0.5 * s * a.xi * xi_n / np.abs(Fb @ xi_n).max()
        f_model = fb.compute(x0.at[n_u + n_p :].set(jnp.asarray(xs)))
        eq_s = fb.disp.apply(xs)
        inp_s = DESCInput(eq_s, inp.field, name="displaced")
        cur_s, pre_s = rebase(cur, pre, eq_s)
        fb_s, _, _ = build(inp_s, cur_s, pre_s, ref, grid, res, solver=fb.solver, n_os=n_os)
        f_ex = fb_s.compute(fb_s.x0)
        d = {k: float(np.abs(np.asarray(f_model[k] - f_ex[k])).max()
                      / np.abs(np.asarray(f_ex[k])).max()) for k in ("F", "B", "J", "Bn", "Psi", "W")}
        e3.append(d)
        print(f"3. max |D| {0.5 * s * a.xi:.1e} m: model vs rebuilt " +
              ", ".join(f"{k} {v:.1e}" for k, v in d.items()), flush=True)
    print("   ratio (2 xi / xi): " + ", ".join(f"{k} {e3[1][k] / e3[0][k]:.1f}" for k in e3[0]
                                             if e3[0][k] > 1e-12), flush=True)

    # 4. the start's boundary: annulus against direct Biot-Savart
    fbd, _, _ = build(inp, cur, pre, ref, grid, res, solver=fb.solver, direct=(24, 2), n_os=n_os)
    Bn_cyl, Bn_dir = np.asarray(fb.compute(x0)["Bn"]), np.asarray(fbd.compute(x0)["Bn"])
    Bm = np.linalg.norm(np.asarray(fbd.compute(x0)["B_bnd"]), axis=1)
    r = obj.report(x0)
    print(f"4. start: B.n/|B| rms {r['Bn/|B| rms']:.2e} (annulus), direct "
          f"{np.sqrt(np.mean((Bn_dir / Bm) ** 2)):.2e}; annulus - direct rms "
          f"{np.sqrt(np.mean(((Bn_cyl - Bn_dir) / Bm) ** 2)):.1e}; <|F|>/norm "
          f"{r['<|F|>/norm']:.2e}; total {time.time() - t0:.0f} s")
    assert e1["F"] < 1e-8 and e1["J_desc"] < 1e-8
    assert max(e2.values()) < 1e-5 and e_adj < 1e-8


def devices(a, inp, cur, pre, ref, grid, res, fr, fb, plain, n_os):
    """5. The annulus solve split over a.devices devices against one device."""
    from jax.sharding import Mesh

    from cylindrical_bs import CylindricalSolver
    t0 = time.time()
    rng = np.random.default_rng(5)  # its own draws: checks 1-4 as without it
    if len(jax.devices()) < a.devices:
        raise SystemExit(f"--devices {a.devices}: JAX sees {len(jax.devices())}")
    mesh = Mesh(np.array(jax.devices()[: a.devices]), ("modes",))
    one = fb.solver
    sh = CylindricalSolver(one.r_axis, one.z_axis, one.NFP, one.N_phi, chunk=one.chunk,
                           S_dtype=one.S.dtype, dtype=one.dtype, mesh=mesh)
    per = sum(x.nbytes for x in (sh.S, sh.S_o) if x is not None) / a.devices
    print(f"5. {a.devices} devices: {sh.n_pairs} pairs and {sh.S.shape[1]} of "
          f"{one.S.shape[0]} |m| per device, S_m {per / 1e6:.1f} of {one.S_nbytes / 1e6:.1f} MB",
          flush=True)
    # B_plasma and its transpose
    u0 = jnp.asarray(cur.u)
    fsh = Fields(fr, cur, pre, dtype=jnp.float64, cyl_stencil=True, cyl_solver=sh,
                 cyl_oversample=n_os, cyl_fused=True, cyl_rblock=RBLOCK, cyl_order=ORDER)
    B1, B2 = np.asarray(plain.B_plasma(u0)), np.asarray(fsh.B_plasma(u0))
    w = jnp.asarray(rng.normal(size=B1.shape))
    T1 = jax.linear_transpose(lambda u: plain.B_plasma(u, 0.0), u0)(w)[0]
    T2 = jax.linear_transpose(lambda u: fsh.B_plasma(u, 0.0), u0)(w)[0]
    e5b = float(np.abs(B1 - B2).max() / np.abs(B1).max())
    e5t = float(jnp.abs(T1 - T2).max() / jnp.abs(T1).max())
    # the free boundary: f, J v and J^T w at xi != 0
    fbs, _, _ = build(inp, cur, pre, ref, grid, res, solver=sh, n_os=n_os, fused=True)
    objs = lambda: [ForceBalance(), TransportedCurrentError(0.1), StoredEnergy(1.0),  # noqa: E731
                    PressurePositivity(1.0), BoundaryNormalField(1.0), ToroidalFlux(1.0)]
    o1 = FreeBoundaryObjective(objs(), fb).build()
    o2 = FreeBoundaryObjective(objs(), fbs).build()
    n_u, n_p = fb.n_u, fb.n_p
    x = o1.x0.at[n_u + n_p :].set(jnp.asarray(rng.normal(size=fb.n_xi) * 1e-3 / np.sqrt(fb.n_xi)))
    v = jnp.asarray(rng.normal(size=o1.dim_x)) * jnp.abs(x).mean()
    wf = jnp.asarray(rng.normal(size=o1.dim_f))
    rel = lambda p, q: float(jnp.linalg.norm(p - q) / jnp.linalg.norm(p))  # noqa: E731
    e5 = {"f": rel(o1.compute_scaled_error(x), o2.compute_scaled_error(x)),
          "J v": rel(o1.jvp_scaled_error(v, x), o2.jvp_scaled_error(v, x)),
          "J^T w": rel(o1.vjp_scaled_error(wf, x), o2.vjp_scaled_error(wf, x))}
    planes = fbs._grid_field(fbs._assemble(fbs._J_half_x(x)))
    print(f"   vs 1 device: B_plasma {e5b:.1e}, its transpose {e5t:.1e}; free boundary "
          + ", ".join(f"{k} {e:.1e}" for k, e in e5.items())
          + f"; planes on {sorted(d.id for d in planes[0].sharding.device_set)}; "
          f"{time.time() - t0:.0f} s", flush=True)
    assert e5b < 1e-12 and e5t < 1e-12 and max(e5.values()) < 1e-12


if __name__ == "__main__":
    main()
