"""Gates of the spline T and p on the ripple equilibrium (bp2/L32N4 x0.001), before any spline run.

  R1  radial functions of reg="rhom" (rho^q_m B_j(rho^2) x scale): d/drho, d2/drho2 against central
      differences; unit maximum of every column
  R2  axis: (a) an m = 1 radial function ~ rho (a shifted axis) fitted on rho <= 0.2 by each basis' m = 1
      columns (rhom represents it, drop3 cannot near the axis); (b) J and p, grad p of random coefficients at
      rho -> 0: the spread over theta divided by the size must fall with rho (a single limit at the axis)
  R3  boundary conditions for random coefficients: J = 0 and p = 0 on rho = 1, J = O(1 - rho) (the same edge
      structure as the Fourier-Zernike runs: BoundaryPotential edge_power 2, p = (1 - rho^2)^3 Q)
  R4  fits: rms |J_fit - J_DESC| / rms J_DESC by rho band, spline (rhom, bands, FZ edge spacing) against
      Fourier-Zernike 32/32/6; p fit of DESC's p(rho)
  R5  tensor evaluators (TensorCurrent, TensorSplinePressure) against the point evaluators, random coefficients
  R6  save / load round trips (reg, q_cap, bands' breakpoints kept)
  R7  unknown counts of candidate spline settings against FZ 32/32/6 and 40/40/6

    python tests/test_ripple_spline_gates.py [--eq PATH]
"""

import argparse
import os
import sys
import tempfile
import time

import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, "nest"))
from desc import set_device  # noqa: E402

set_device(os.environ.get("NEST_DEVICE", "cpu"))
import jax  # noqa: E402
import jax.numpy as jnp  # noqa: E402

jax.config.update("jax_enable_x64", True)

from boundary_potential import BoundaryPotential  # noqa: E402
from desc_input import DESCInput  # noqa: E402
from flux_interp import TensorCurrent, TensorGeometry, TensorGrid  # noqa: E402
from spline_basis import SplineBasis  # noqa: E402
from spline_pressure import SplinePressure  # noqa: E402

EQ = os.path.join(ROOT, "desc_input", "tokamak_boundary_perturbed", "bp2", "L32N4",
                  "tokamak_ripple_C_L32M32N4_x0.001.h5")
BANDS = ((0.45, 0.76, 0.01), (0.86, 0.95, 0.015))


def pts(rho, th, ze):
    R_, T_, Z_ = np.meshgrid(rho, th, ze, indexing="ij")
    return np.column_stack([R_.ravel(), T_.ravel(), Z_.ravel()])


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--eq", default=EQ)
    ap.add_argument("--MN", type=int, nargs=2, default=(32, 6))
    ap.add_argument("--m-fz", type=int, default=32)
    a = ap.parse_args()
    t0 = time.time()
    inp = DESCInput.load(a.eq)
    eq = inp.eq
    M, N = a.MN
    ok = True
    kw = dict(bands=BANDS, fz_knots=(32, 1.0), reg="rhom", q_cap=6, m_fz=a.m_fz)

    # R1
    sb = SplineBasis.from_knots(eq, M, N, helicities=("2/1",), **kw)
    print(f"spline basis: {sb.summary()}", flush=True)
    print(f"  breakpoints: {np.round(sb.bp_rho, 4).tolist()}")
    rho = np.linspace(0.02, 0.98, 301)
    js = np.arange(sb.n_s)
    worst = 0.0
    for m in (0, 1, 2, 3, 7, 20):
        ms = np.full(len(js), m)
        h = 1e-5
        r0, rp, rm = (jnp.asarray(x) for x in (rho, rho + h, rho - h))
        f0, fp, fm = (np.asarray(sb.radial(js, ms, x, 0)) for x in (r0, rp, rm))
        d1, d2 = np.asarray(sb.radial(js, ms, r0, 1)), np.asarray(sb.radial(js, ms, r0, 2))
        e1 = np.abs(d1 - (fp - fm) / (2 * h)).max() / max(np.abs(d1).max(), 1e-300)
        e2 = np.abs(d2 - (fp - 2 * f0 + fm) / h**2).max() / max(np.abs(d2).max(), 1e-300)
        mx = np.asarray(sb.radial(js, ms, jnp.asarray(np.linspace(0, 1, 4001)), 0)).max(0)
        worst = max(worst, e1, e2 / 1e2)
        print(f"R1 m {m:2d} (q {sb.q(m)}): d/drho {e1:.1e}, d2/drho2 {e2:.1e} (FD), column max in "
              f"[{mx.min():.4f}, {mx.max():.4f}]")
        ok &= e1 < 1e-6 and e2 < 1e-3 and abs(mx.min() - 1) < 1e-3 and abs(mx.max() - 1) < 1e-3

    # R2a: a shifted axis, f = rho on [0, 0.2], by the m = 1 columns
    r = np.linspace(0, 0.2, 201)
    for reg in ("rhom", "drop3"):
        b_ = SplineBasis.from_knots(eq, M, N, helicities=("2/1",), **dict(kw, reg=reg))
        _, C, _ = b_.radial_columns(1, r)
        cf = np.linalg.lstsq(C, r, rcond=None)[0]
        err = np.abs(C @ cf - r)
        print(f"R2a {reg}: m = 1 fit of rho on [0, 0.2]: max err {err.max():.2e}, at rho <= 0.05 {err[r <= 0.05].max():.2e}")
        if reg == "rhom":
            ok &= err.max() < 1e-10

    # fits (R4) and objects for R2b, R3, R5
    t1 = time.time()
    sp = BoundaryPotential.fit(eq, M=M, N=N, radial="spline", helicities=("2/1",), **kw)
    print(f"spline current fit: n_u {sp.n_free}, {time.time() - t1:.0f} s", flush=True)
    fz = BoundaryPotential.fit(eq, L=32, M=32, N=6)
    print(f"FZ 32/32/6 current: n_u {fz.n_free}", flush=True)
    pr = SplinePressure.fit(eq, M=M, N=N, edge_order=2, helicities=("2/1",), **kw)
    print(f"spline pressure: {pr.summary()}", flush=True)

    # R2b
    rng = np.random.default_rng(3)
    u = rng.normal(size=sp.n_free) * np.abs(sp.u).max()
    Jf = jax.jit(sp.with_u(u).point_evaluator())
    c = rng.normal(size=pr.n) * np.abs(pr.c).max()
    Pf = jax.jit(pr.with_c(c).point_evaluator())
    th = 2 * np.pi * np.arange(16) / 16
    # a regular field has one limit at the axis: its spread over theta falls ~ rho (J, grad p: the theta-dependent
    # part of a smooth vector field is O(rho)); an irregular one keeps an O(1) spread
    sprJ, sprP, sprG = [], [], []
    rs = (1e-3, 1e-4, 1e-5)
    for r_ in rs:
        x = pts([r_], th, [0.3])
        Jv = np.asarray(Jf(*(jnp.asarray(x[:, i]) for i in range(3))))
        Pv = np.asarray(Pf(*(jnp.asarray(x[:, i]) for i in range(3))))
        sprJ.append(np.abs(Jv - Jv.mean(0)).max())
        sprP.append(np.abs(Pv[:, 0] - Pv[:, 0].mean()).max())
        sprG.append(np.abs(Pv[:, 1:] - Pv[:, 1:].mean(0)).max())
    rJ, rP, rG = (v[-1] / v[0] for v in (sprJ, sprP, sprG))
    print(f"R2b absolute theta spread at rho {rs}: J {[f'{v:.2e}' for v in sprJ]} (ratio {rJ:.1e}), p "
          f"{[f'{v:.2e}' for v in sprP]} (ratio {rP:.1e}), grad p {[f'{v:.2e}' for v in sprG]} (ratio {rG:.1e}); "
          f"regular: ratios ~ 1e-2 or below")
    ok &= rJ < 3e-2 and rP < 3e-2 and rG < 3e-2

    # R3
    x = pts([1.0, 1 - 1e-3, 1 - 2e-3], th, [0.3, 1.1])
    Jv = np.linalg.norm(np.asarray(Jf(*(jnp.asarray(x[:, i]) for i in range(3)))), axis=1).reshape(3, -1)
    Pv = np.asarray(Pf(*(jnp.asarray(x[:, i]) for i in range(3))))[:, 0].reshape(3, -1)
    xin = pts([0.5], th, [0.3])
    Jscale = np.abs(np.asarray(Jf(*(jnp.asarray(xin[:, i]) for i in range(3))))).max()
    ratio = Jv[2].max() / max(Jv[1].max(), 1e-300)
    print(f"R3 random coefficients: max |J| on rho = 1 / |J| at 0.5 {Jv[0].max() / Jscale:.1e}; |J|(1-2e-3)/|J|(1-1e-3) "
          f"{ratio:.2f} (2 for O(1 - rho)); max |p| on rho = 1 {np.abs(Pv[0]).max():.1e}")
    ok &= Jv[0].max() / Jscale < 1e-10 and np.abs(Pv[0]).max() < 1e-12 and 1.5 < ratio < 2.5

    # R4
    g_r = np.linspace(0.01, 0.99, 99)
    x = pts(g_r, 2 * np.pi * np.arange(33) / 33, 2 * np.pi * np.arange(9) / 9)
    Jd = np.asarray(inp.compute_points(["J"], x)["J"])
    bands = ((0, 0.4), (0.4, 0.55), (0.55, 0.72), (0.72, 0.85), (0.85, 0.95), (0.95, 1.0))
    rmsD = np.sqrt(np.mean(np.sum(Jd**2, 1)))
    for name, pot in (("spline", sp), ("FZ 32/32/6", fz)):
        Jp = np.asarray(jax.jit(pot.point_evaluator())(*(jnp.asarray(x[:, i]) for i in range(3))))
        e = np.sum((Jp - Jd) ** 2, 1)
        tot = np.sqrt(e.mean()) / rmsD
        by = [np.sqrt(e[(x[:, 0] >= lo) & (x[:, 0] < hi)].mean()) / rmsD for lo, hi in bands]
        print(f"R4 {name:10s} rms |J - J_DESC| / rms J_DESC {tot:.2e}; by band "
              + ", ".join(f"{lo}-{hi}: {v:.1e}" for (lo, hi), v in zip(bands, by)))
        if name == "spline":
            tot_s = tot
        else:
            tot_f = tot
    ok &= tot_s <= 1.5 * tot_f
    from desc.grid import LinearGrid
    rr = np.linspace(0, 1, 201)
    pD = np.asarray(eq.compute("p", grid=LinearGrid(rho=rr, NFP=eq.NFP))["p"])
    pS = np.asarray(jax.jit(pr.point_evaluator())(jnp.asarray(rr), jnp.zeros(201), jnp.zeros(201)))[:, 0]
    ep = np.abs(pS - pD).max() / np.abs(pD).max()
    print(f"R4 spline p fit of DESC's p(rho): max err / max p {ep:.1e}")
    ok &= ep < 1e-3

    # R5
    g = TensorGrid(np.linspace(0.01, 0.99, 23), 2 * np.pi * np.arange(17) / 17, 2 * np.pi * np.arange(7) / 7, eq.NFP)
    geo = TensorGeometry(sp._RZ, g)
    sp2 = sp.with_u(u)
    Jt = np.asarray(TensorCurrent(sp2.S, sp2._RZ, g, geo)(jnp.asarray(u), sp2.I)).reshape(-1, 3)
    nodes = g.nodes()
    Jp = np.asarray(Jf(*(jnp.asarray(nodes[:, i]) for i in range(3))))
    e5 = float(np.abs(Jt - Jp).max() / np.abs(Jp).max())
    pt, gp = pr.with_c(c).tensor(g, geo, sp._RZ)(jnp.asarray(c))
    Pp = np.asarray(Pf(*(jnp.asarray(nodes[:, i]) for i in range(3))))
    gpt = np.stack([np.asarray(v).ravel() for v in gp], -1) if isinstance(gp, (tuple, list)) else np.asarray(gp).reshape(-1, 3)
    e5p = float(np.abs(np.asarray(pt).ravel() - Pp[:, 0]).max() / np.abs(Pp[:, 0]).max())
    e5g = float(np.abs(gpt - Pp[:, 1:]).max() / np.abs(Pp[:, 1:]).max())
    print(f"R5 tensor vs point: J {e5:.1e}, p {e5p:.1e}, grad p {e5g:.1e}")
    ok &= e5 < 1e-9 and e5p < 1e-9 and e5g < 1e-8

    # R6
    with tempfile.TemporaryDirectory() as d:
        sp.save(os.path.join(d, "t.npz"))
        pr.save(os.path.join(d, "p.npz"))
        sp3 = BoundaryPotential.load(os.path.join(d, "t.npz"), eq)
        pr3 = SplinePressure.load(os.path.join(d, "p.npz"), eq)
    e6 = (np.array_equal(sp3.basis.modes, sp.basis.modes) and sp3.basis.reg == "rhom" and np.array_equal(sp3.u, sp.u)
          and pr3.reg == "rhom" and np.array_equal(pr3.c, pr.c) and np.array_equal(pr3.scale, pr.scale))
    print(f"R6 save/load round trip: {e6}")
    ok &= bool(e6)

    # R7
    from desc.basis import FourierZernikeBasis
    from spline_pressure import count_fz
    print("R7 unknowns (a + b; chi adds the same in both):")
    for L_ in (32, 40):
        fzb = FourierZernikeBasis(L_, L_, 6, NFP=eq.NFP, sym="cos" if eq.sym else False,
                                  spectral_indexing=eq.spectral_indexing)
        print(f"   FZ {L_}/{L_}/6: a + b {2 * fzb.num_modes}; p ~{count_fz(L_, L_, 6, bool(eq.sym))}")
    for mfz in (None, 40, 32, 24):
        for mmax in (None, 24, 16):
            for M_ in (32, 24):
                b_ = SplineBasis.from_knots(eq, M_, 6, helicities=("2/1",), **dict(kw, m_fz=mfz, m_max=mmax))
                print(f"   spline M {M_} m_fz {mfz} m_max {mmax}: n_s {b_.n_s}, a + b {2 * b_.num_modes}")
    print(f"ALL PASSED ({time.time() - t0:.0f} s)" if ok else f"FAILED ({time.time() - t0:.0f} s)")
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
