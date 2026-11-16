"""Gates of the FZ + band-localized spline basis ("fz+spline", spline_basis.CompositeBasis) on the ripple equilibrium.

  H1  J_fit of the hybrid = the FZ 32/32/6 J_fit (spline coefficients 0), p fit = FZ p fit
  H2  the spline part is local: J of spline-only coefficients vanishes outside the bands (and is not 0 inside);
      p of spline-only coefficients likewise; the edge (J = 0, p = 0 on rho = 1) and the axis untouched
  H3  tensor evaluators against point evaluators (random coefficients, current and pressure)
  H4  save / load round trips; FZ states embed exactly (embed_truncate: no coefficient dropped)
  H5  unknown counts

    python tests/test_hybrid_gates.py [--mmax 16]
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
from pressure import Pressure  # noqa: E402

EQ = os.path.join(ROOT, "desc_input", "tokamak_boundary_perturbed", "bp2", "L32N4", "tokamak_ripple_C_L32M32N4_x0.001.h5")
BANDS = ((0.45, 0.76, 0.01), (0.86, 0.95, 0.015))


def pts(rho, th, ze):
    R_, T_, Z_ = np.meshgrid(rho, th, ze, indexing="ij")
    return np.column_stack([R_.ravel(), T_.ravel(), Z_.ravel()])


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--eq", default=EQ)
    ap.add_argument("--mmax", type=int, default=16)
    a = ap.parse_args()
    t0 = time.time()
    eq = DESCInput.load(a.eq).eq
    ok = True
    fz = BoundaryPotential.fit(eq, L=32, M=32, N=6)
    hy = BoundaryPotential.fit(eq, L=32, M=32, N=6, radial="fz+spline", bands=BANDS, m_max=a.mmax)
    pf = Pressure.fit(eq, L=32, M=32, N=6, edge_order=2)
    ph = Pressure.fit(eq, L=32, M=32, N=6, edge_order=2,
                      spline=(("bands", BANDS), ("knots", (0.05, 0.01, 0.08)), ("m_max", a.mmax)))
    print(f"hybrid current: {hy.basis.summary()['n_fz']} FZ + {hy.basis.summary()['n_spline']} spline modes, n_u "
          f"{hy.n_free} (FZ {fz.n_free}); pressure n {ph.n} (FZ {pf.n}); {time.time() - t0:.0f} s", flush=True)
    print(f"  spline breakpoints: {np.round(hy.basis.bp_rho, 4).tolist()}")
    # H1
    x = pts(np.linspace(0.02, 0.98, 49), 2 * np.pi * np.arange(17) / 17, 2 * np.pi * np.arange(7) / 7)
    X = [jnp.asarray(x[:, i]) for i in range(3)]
    Jf, Jh = (np.asarray(jax.jit(p_.point_evaluator())(*X)) for p_ in (fz, hy))
    Pf, Ph = (np.asarray(jax.jit(p_.point_evaluator())(*X)) for p_ in (pf, ph))
    e1 = np.abs(Jf - Jh).max() / np.abs(Jf).max()
    e1p = np.abs(Pf - Ph).max() / np.abs(Pf).max()
    print(f"H1 J_fit hybrid vs FZ: {e1:.1e}; p fit hybrid vs FZ: {e1p:.1e}")
    ok &= e1 < 1e-12 and e1p < 1e-12
    # H2: spline-only coefficients
    rng = np.random.default_rng(5)
    S = hy.S
    nc, K, Kf = S.n_chi, S.K, hy.basis.n_fz
    u = np.zeros(hy.n_free)
    u[nc + Kf:nc + K] = rng.normal(size=K - Kf)
    u[nc + K + Kf:] = rng.normal(size=K - Kf)
    u *= np.abs(hy.u).max()
    hsf_u = jax.jit(hy.with_u(u).point_evaluator())
    hsf_0 = jax.jit(hy.with_u(np.zeros(hy.n_free)).point_evaluator())  # the net-current term I h_0 alone

    def hsf(*X):  # J of the spline coefficients alone (J is affine in u: J(u) - J(0))
        return hsf_u(*X) - hsf_0(*X)
    rr = np.array([0.0, 0.2, 0.4, 0.449, 0.5, 0.6, 0.7, 0.759, 0.8, 0.85, 0.9, 0.951, 0.98, 1.0])
    xs = pts(rr, 2 * np.pi * np.arange(13) / 13, [0.4, 1.7])
    Jv = np.linalg.norm(np.asarray(hsf(*(jnp.asarray(xs[:, i]) for i in range(3)))), axis=1).reshape(len(rr), -1).max(1)
    inside = np.array([any(lo <= r_ <= hi for lo, hi, _ in BANDS) for r_ in rr])
    c = np.zeros(ph.n)
    c[ph.basis.n_fz:] = rng.normal(size=ph.n - ph.basis.n_fz)
    Pv = np.abs(np.asarray(jax.jit(ph.with_c(c).point_evaluator())(*(jnp.asarray(xs[:, i]) for i in range(3))))).reshape(len(rr), -1, 4).max(axis=(1, 2))
    print("H2 spline-only |J| by rho: " + " ".join(f"{r_:.3f}:{v:.1e}" for r_, v in zip(rr, Jv)))
    print("   spline-only max(|p|, |grad p|) by rho: " + " ".join(f"{r_:.3f}:{v:.1e}" for r_, v in zip(rr, Pv)))
    ax = rr > 0  # the point evaluator's grad p divides by sqrt(g) = 0 on the axis (nan there, for FZ too)
    ok &= Jv[~inside].max() <= 1e-12 * Jv[inside].max() and Pv[~inside & ax].max() <= 1e-12 * Pv[inside].max()
    ok &= Jv[inside].min() > 0
    # H3
    g = TensorGrid(np.linspace(0.02, 0.98, 31), 2 * np.pi * np.arange(17) / 17, 2 * np.pi * np.arange(7) / 7, eq.NFP)
    geo = TensorGeometry(hy._RZ, g)
    ur = hy.u + 0.1 * u
    h2 = hy.with_u(ur)
    Jt = np.asarray(TensorCurrent(h2.S, h2._RZ, g, geo)(jnp.asarray(ur), h2.I)).reshape(-1, 3)
    nodes = g.nodes()
    Jp = np.asarray(jax.jit(h2.point_evaluator())(*(jnp.asarray(nodes[:, i]) for i in range(3))))
    cr = ph.c + 0.1 * np.abs(ph.c).max() * c
    pt, gp = ph.with_c(cr).tensor(g, geo, hy._RZ)(jnp.asarray(cr))
    Pp = np.asarray(jax.jit(ph.with_c(cr).point_evaluator())(*(jnp.asarray(nodes[:, i]) for i in range(3))))
    e3 = np.abs(Jt - Jp).max() / np.abs(Jp).max()
    e3p = np.abs(np.asarray(pt).ravel() - Pp[:, 0]).max() / np.abs(Pp[:, 0]).max()
    e3g = np.abs(np.asarray(gp).reshape(-1, 3) - Pp[:, 1:]).max() / np.abs(Pp[:, 1:]).max()
    print(f"H3 tensor vs point: J {e3:.1e}, p {e3p:.1e}, grad p {e3g:.1e}")
    ok &= e3 < 1e-9 and e3p < 1e-9 and e3g < 1e-8
    # H4
    with tempfile.TemporaryDirectory() as d:
        hy.save(os.path.join(d, "t.npz"))
        ph.save(os.path.join(d, "p.npz"))
        h3 = BoundaryPotential.load(os.path.join(d, "t.npz"), eq)
        p3 = Pressure.load(os.path.join(d, "p.npz"), eq)
    sys.path.insert(0, os.path.join(ROOT, "nest"))
    import meanforce_fixed as mff
    ue, frac, nd = mff.embed_truncate(fz, hy, fz.u)
    e4 = (np.array_equal(h3.u, hy.u) and np.array_equal(h3.basis.modes, hy.basis.modes) and np.array_equal(p3.c, ph.c)
          and np.array_equal(p3.basis.modes, ph.basis.modes))
    print(f"H4 save/load {e4}; FZ J_fit embedded: dropped {nd}, |u_embed - u_hybrid| {np.abs(ue - hy.u).max():.1e}")
    ok &= bool(e4) and nd == 0 and np.abs(ue - hy.u).max() == 0
    print(f"H5 n_u: FZ {fz.n_free}, hybrid {hy.n_free}; n_p: FZ {pf.n}, hybrid {ph.n}")
    print(f"ALL PASSED ({time.time() - t0:.0f} s)" if ok else f"FAILED ({time.time() - t0:.0f} s)")
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
