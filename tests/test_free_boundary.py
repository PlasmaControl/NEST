"""The free boundary (``free_boundary``): moving geometry, fields and derivative.

NCSX (DESC re-solve in its coils) at a low resolution, box h = a / frac:
  1. the displacement: the boundary of ``Displacement.apply(xi)`` against
     X + xi n (the projection on the equilibrium's modes), and the outward sign
  2. xi = 0: ``FreeBoundaryFields`` against ``Fields`` at x0 (F, B, J, p), and
     DESC's current carried with the coordinates against DESC's J
  3. J v against central differences along u, c_p, xi and all (box in float64);
     the adjoint identity <w, J v> = <J^T w, v>; with B on the boundary by direct
     Biot-Savart (``DirectField``), its derivative and its difference from the box's
  4. first order in D: the fields at xi against those rebuilt exactly on the
     displaced geometry (``Displacement.apply``, xi = 0), for xi and 2 xi; the
     difference must scale as |xi|^2
  5. B.n / |B| of the start on the boundary, and the toroidal flux against DESC's Psi

run: python tests/test_free_boundary.py [frac] [--eq file] [--factors 1 1 1]
"""

import argparse
import os
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
from free_boundary import (BoundaryNormalField, Displacement, FreeBoundaryFields,  # noqa: E402
                           FreeBoundaryObjective, StoredEnergy, ToroidalFlux,
                           TransportedCurrentError, reference)
from grid import CartesianGrid  # noqa: E402
from objective import Fields, ForceBalance, PressurePositivity  # noqa: E402
from pressure import Pressure  # noqa: E402

jax.config.update("jax_enable_x64", True)
D = os.path.join(ROOT, "desc_input", "vmec_free_boundary", "NCSX")
EQ, COILS = os.path.join(D, "NCSX_L12M12N8.h5"), os.path.join(D, "NCSX_coils.h5")


def build(inp, cur, pre, ref, frac, a0, res, dtype, M=4, N=3, axes=None):
    sym = bool(inp.eq.sym)
    box = (CartesianGrid.periodic_box(inp, a0 / frac, odd=sym) if axes is None else
           CartesianGrid(axes, periodic=True)).locate_plasma(inp)
    fr = ForceResidual(inp, box, resolution=res)
    disp = Displacement(inp.eq, M, N)
    return FreeBoundaryFields(fr, cur, pre, disp, ref, dtype=dtype), fr, box


def rebase(cur, pre, eq):
    """The fits' structure and coefficients on another geometry."""
    return (BoundaryPotential(cur.basis, cur.u, cur.I, eq, cur.edge_power, cur.S),
            Pressure(pre.basis, pre.c, eq, pre.edge_order))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("frac", type=int, nargs="?", default=6)
    ap.add_argument("--eq", default=EQ)
    ap.add_argument("--coils", default=COILS)
    ap.add_argument("--factors", type=float, nargs=3, default=(1, 1, 1))
    ap.add_argument("--xi", type=float, default=1e-3, help="largest displacement of check 4 [m]")
    a = ap.parse_args()
    t0 = time.time()
    inp = DESCInput.load(a.eq, field=a.coils)
    eq = inp.eq
    cur = fit_cache.fit(BoundaryPotential, inp, factors=tuple(a.factors))
    pre = fit_cache.fit(Pressure, inp, factors=tuple(a.factors), edge_order=2)
    b = cur.basis
    res = tuple(int(u + v) for u, v in ((b.L, eq.L), (b.M, eq.M), (b.N, eq.N)))
    ref = reference(inp, res)
    print(f"{inp}\nresolution of F {res}, setup {time.time() - t0:.0f} s", flush=True)

    # 1. displacement
    disp = Displacement(eq, 4, 3)
    rng = np.random.default_rng(0)
    xi = rng.normal(size=disp.n) * 1e-3 / np.sqrt(disp.n)
    new = disp.apply(xi)
    from desc.grid import LinearGrid
    g = LinearGrid(rho=1.0, M=2 * eq.M, N=2 * eq.N, NFP=eq.NFP, sym=False)
    d0 = eq.compute(["R", "Z", "e_theta"], grid=g)
    d1 = new.compute(["R", "Z"], grid=g)
    Rt, Zt = np.asarray(d0["e_theta"])[:, 0], np.asarray(d0["e_theta"])[:, 2]
    nR, nZ = disp.normal_sign * Zt / np.hypot(Rt, Zt), -disp.normal_sign * Rt / np.hypot(Rt, Zt)
    xi_g = np.asarray(disp.basis.evaluate(g)) @ xi
    dR, dZ = np.asarray(d1["R"] - d0["R"]), np.asarray(d1["Z"] - d0["Z"])
    e1 = np.sqrt(np.mean((dR - xi_g * nR) ** 2 + (dZ - xi_g * nZ) ** 2)) / np.sqrt(np.mean(xi_g**2))
    out = np.mean(dR * nR + dZ * nZ) / np.mean(xi_g)  # 1 when xi > 0 moves outward
    print(f"1. displacement: projection error {disp.projection_error:.1e} (unit modes), "
          f"boundary vs X + xi n {e1:.1e} rms, outward component / xi {out:.3f}", flush=True)

    # 2. xi = 0 against the fixed-boundary fields
    fb, fr, box = build(inp, cur, pre, ref, a.frac, inp.a, res, jnp.float64)
    plain = Fields(fr, cur, pre, kernel="truncated", dtype=jnp.float64)
    f0 = {k: np.asarray(v) for k, v in fb.compute(fb.x0).items()}
    p0 = {k: np.asarray(v) for k, v in plain.compute(plain.x0).items()}
    e2 = {k: float(np.abs(f0[k] - p0[k]).max() / np.abs(p0[k]).max()) for k in ("F", "B", "J", "p")}
    Jd = np.asarray(plain.desc["J"])
    e2["J_desc"] = float(np.abs(f0["J_desc"] - Jd).max() / np.abs(Jd).max())
    sh = fb.box_shift  # the same stencils as the box interpolation it differentiates
    assert np.array_equal(np.asarray(sh.idx), np.asarray(fb.to_box.idx))
    e2["box weights"] = float(np.abs(np.asarray(sh.w) - np.asarray(fb.to_box.w)).max())
    print("2. xi = 0 vs Fields: " + ", ".join(f"{k} {v:.1e}" for k, v in e2.items()), flush=True)

    # 3. derivative
    objs = [ForceBalance(), TransportedCurrentError(0.1), StoredEnergy(1.0),
            PressurePositivity(1.0), BoundaryNormalField(1.0), ToroidalFlux(1.0)]
    obj = FreeBoundaryObjective(objs, fb).build()
    x0 = obj.x0
    n_u, n_p, n_xi = fb.n_u, fb.n_p, fb.n_xi
    rms = lambda v: np.sqrt(np.mean(np.asarray(v) ** 2))  # noqa: E731
    scale = np.concatenate([np.full(n_u, rms(x0[:n_u])), np.full(n_p, rms(x0[n_u : n_u + n_p])),
                            np.full(n_xi, 1e-3 / np.sqrt(n_xi))])
    idx = np.arange(obj.dim_x)
    e3 = {}
    for name, mask in [("u", idx < n_u), ("c_p", (idx >= n_u) & (idx < n_u + n_p)),
                       ("xi", idx >= n_u + n_p), ("all", idx >= 0)]:
        v = jnp.asarray(1e-4 * scale * rng.normal(size=obj.dim_x) * mask)
        Jv = obj.jvp_scaled_error(v, x0)
        fd = (obj.compute_scaled_error(x0 + v) - obj.compute_scaled_error(x0 - v)) / 2
        e3[name] = float(jnp.linalg.norm(Jv - fd) / jnp.linalg.norm(fd))
    # around a displaced point too (xi != 0: the cross terms of the box current)
    x1 = x0.at[n_u + n_p :].set(jnp.asarray(xi))
    v = jnp.asarray(1e-4 * scale * rng.normal(size=obj.dim_x))
    Jv = obj.jvp_scaled_error(v, x1)
    fd = (obj.compute_scaled_error(x1 + v) - obj.compute_scaled_error(x1 - v)) / 2
    e3["all at xi != 0"] = float(jnp.linalg.norm(Jv - fd) / jnp.linalg.norm(fd))
    w = jnp.asarray(rng.normal(size=obj.dim_f))
    lhs, rhs = float(jnp.vdot(w, Jv)), float(jnp.vdot(obj.vjp_scaled_error(w, x1), v))
    e_adj = abs(lhs - rhs) / abs(lhs)
    print("3. J v vs central differences: " + ", ".join(f"{k} {e:.1e}" for k, e in e3.items())
          + f"; adjoint {e_adj:.1e}", flush=True)

    # 3b. with B on the boundary by direct Biot-Savart: its derivative (direct_tangent)
    # against central differences, and B.n against the box's
    fbd = FreeBoundaryFields(fr, cur, pre, fb.disp, ref, dtype=jnp.float64, direct=(24, 2),
                             direct_tangent=True)
    assert fbd.direct_d is fbd.direct
    objd = FreeBoundaryObjective([BoundaryNormalField(1.0)], fbd).build()
    v = jnp.asarray(1e-4 * scale * rng.normal(size=objd.dim_x))
    Jv = objd.jvp_scaled_error(v, x1)
    fd = (objd.compute_scaled_error(x1 + v) - objd.compute_scaled_error(x1 - v)) / 2
    e3d = float(jnp.linalg.norm(Jv - fd) / jnp.linalg.norm(fd))
    Bn_box, Bn_dir = np.asarray(fb.compute(x0)["Bn"]), np.asarray(fbd.compute(x0)["Bn"])
    Bm = np.linalg.norm(np.asarray(fbd.compute(x0)["B_bnd"]), axis=1)
    print(f"3b. direct B on the boundary: J v vs central differences {e3d:.1e}; B.n/|B| box - "
          f"direct rms {np.sqrt(np.mean(((Bn_box - Bn_dir) / Bm) ** 2)):.1e} (the box's error)",
          flush=True)
    assert e3d < 1e-5

    # 4. first order in D: against the fields rebuilt on the displaced geometry
    xi_n = rng.normal(size=n_xi)
    e4 = []
    for s in (1.0, 2.0):
        xs = 0.5 * s * a.xi * xi_n / np.abs(np.asarray(fb.disp.basis.evaluate(
            LinearGrid(rho=1.0, M=2 * eq.M, N=2 * eq.N, NFP=eq.NFP)))@ xi_n).max()
        f_model = fb.compute(x0.at[n_u + n_p :].set(jnp.asarray(xs)))
        eq_s = fb.disp.apply(xs)
        inp_s = DESCInput(eq_s, inp.field, name="displaced")
        cur_s, pre_s = rebase(cur, pre, eq_s)
        # on the same box: the difference is that of the model, not of the box
        fb_s, _, _ = build(inp_s, cur_s, pre_s, ref, a.frac, inp.a, res, jnp.float64,
                           axes=box.axes)
        f_ex = fb_s.compute(fb_s.x0)
        d = {k: float(np.abs(np.asarray(f_model[k] - f_ex[k])).max()
                      / np.abs(np.asarray(f_ex[k])).max()) for k in ("F", "B", "J", "Bn", "Psi", "W")}
        e4.append(d)
        print(f"4. max |D| {0.5 * s * a.xi:.1e} m: model vs rebuilt " +
              ", ".join(f"{k} {v:.1e}" for k, v in d.items()), flush=True)
    print("   ratio (2 xi / xi): " + ", ".join(f"{k} {e4[1][k] / e4[0][k]:.1f}" for k in e4[0]
                                             if e4[0][k] > 1e-12))

    # 5. the start's boundary
    r = obj.report(x0)
    f = fb.compute(x0)
    print(f"5. start: B.n/|B| rms {r['Bn/|B| rms']:.2e}, max {r['Bn/|B| max']:.2e}; "
          f"Psi {float(f['Psi'][0]):.5f} Wb (DESC {float(eq.Psi):.5f}); "
          f"<|F|>/norm {r['<|F|>/norm']:.2e}; total {time.time() - t0:.0f} s")
    assert e2["F"] < 1e-6 and e2["J_desc"] < 1e-8
    assert max(e3.values()) < 1e-5 and e_adj < 1e-8


if __name__ == "__main__":
    main()
