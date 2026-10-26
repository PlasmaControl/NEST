"""Gates of the reduced Poisson pressure on the free boundary (``nest/meanforce.py``),
plan sections 3, 5 and 6, on a small NCSX case (CPU):

  T4  manufactured: a = P c_q (J x B a gradient in the space) gives p* = c_q and f = 0
  T1  the trailing block of the QR of [P | C | r] is the explicit projection (I - Q Q^T)[C | r]
  T2  the Galerkin identity P^T f = 0 at a state with xi != 0
  T3  the linearization (with dPi/dxi) against central differences: u, xi, all;
      the adjoint identity <w, J v> = <J^T w, v>
  T6  envelope: d/du |f|^2 / 2 = <f, da/du> (p held at p*: the joint (u, c) gradient)
  D   the dense rows (u columns from the tables) against J v at xi = 0
  R   the AL rows: grad L_A = J^T r against central differences of the plain merit
  I   iota of DESC's start against DESC's profile
  T5  (short) GLTR and QR trust-region steps from the same start: both decrease L_A

run: python tests/test_meanforce.py [--frac 6] [--N-phi 15]
"""

import argparse
import os
import sys
import time

import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, "nest"))
sys.path.insert(0, os.path.join(ROOT, "tests"))

from desc import set_device  # noqa: E402

set_device(os.environ.get("NEST_DEVICE", "cpu"))
import jax  # noqa: E402
import jax.numpy as jnp  # noqa: E402

jax.config.update("jax_enable_x64", True)

import fit_cache  # noqa: E402
import meanforce as mf  # noqa: E402
import pointwise as pw  # noqa: E402
from boundary_potential import BoundaryPotential  # noqa: E402
from desc_input import DESCInput  # noqa: E402
from free_boundary import reference  # noqa: E402
from pressure import Pressure  # noqa: E402
from test_free_boundary_cyl import annulus, build  # noqa: E402

D = os.path.join(ROOT, "desc_input", "vmec_free_boundary", "NCSX")


def rel(a, b):
    return float(jnp.linalg.norm(a - b) / max(float(jnp.linalg.norm(b)), 1e-300))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--frac", type=float, default=6)
    ap.add_argument("--N-phi", type=int, default=15)
    ap.add_argument("--eq", default=os.path.join(D, "NCSX_L12M12N8.h5"))
    ap.add_argument("--coils", default=os.path.join(D, "NCSX_coils.h5"))
    ap.add_argument("--factors", type=float, nargs=3, default=(0.5, 0.5, 0.5))
    ap.add_argument("--no-t5", action="store_true")
    a = ap.parse_args()
    t0 = time.time()
    inp = DESCInput.load(a.eq, field=a.coils)
    eq = inp.eq
    cur = fit_cache.fit(BoundaryPotential, inp, factors=tuple(a.factors))
    pre = fit_cache.fit(Pressure, inp, factors=tuple(a.factors), edge_order=2)
    b = cur.basis
    res = tuple(int(u + v) for u, v in ((b.L, eq.L), (b.M, eq.M), (b.N, eq.N)))
    ref = reference(inp, res, current=cur)
    grid = annulus(inp, inp.a / a.frac, a.N_phi, 0.02)
    fb, fr, g = build(inp, cur, pre, ref, grid, res, M=4, N=3)
    A = mf.Anchor(fb, cur.u, 1.0, ref["norm"], chunk=32)
    print(f"{inp}\nbasis {b.L}/{b.M}/{b.N}: n_u {A.n_u}, n_p {A.n_p} ({len(A.keep)} kept), "
          f"n_xi {A.n_xi}, points {A.n_pts:,}, boundary {A.n_b}; {time.time() - t0:.0f} s",
          flush=True)
    rng = np.random.default_rng(0)
    ok = True

    # T4: manufactured gradient
    xi0 = jnp.zeros(A.n_xi)
    sw0 = mf._sw(fb, xi0)
    P = A._pcols(xi0, sw0, A.keep)
    cq = jnp.asarray(rng.normal(size=P.shape[1]))
    Q, R, c, f = mf._reduce(P, P @ cq)
    e4 = (rel(c, cq), float(jnp.linalg.norm(f) / jnp.linalg.norm(P @ cq)))
    print(f"T4 manufactured: |p* - q| / |q| {e4[0]:.1e}, |f| / |a| {e4[1]:.1e}")
    ok &= max(e4) < 1e-8

    # T1: the trailing block against the explicit projection
    C = jnp.asarray(rng.normal(size=(P.shape[0], 20)))
    r = jnp.asarray(rng.normal(size=(P.shape[0], 1)))
    Rf = jnp.linalg.qr(jnp.concatenate([P, C, r], 1), mode="r")
    k = P.shape[1]
    Tb = Rf[k:, k:]
    X = jnp.concatenate([C, r], 1)
    Xp = X - Q @ (Q.T @ X)
    e1 = rel(Tb.T @ Tb, Xp.T @ Xp)
    print(f"T1 trailing block vs projection (Gram): {e1:.1e}")
    ok &= e1 < 1e-10

    # a state with xi != 0
    z = jnp.concatenate([jnp.asarray(cur.u) * (1 + 0.01 * rng.normal(size=A.n_u)),
                         jnp.asarray(1e-4 * rng.normal(size=A.n_xi))])
    s = A.state(z)
    gal = A.galerkin(s)
    print(f"T2 Galerkin max_j |P_j^T f| / (|P_j||f|): {gal:.1e}; M {float(s['M']):.3e}, "
          f"B.n/|B| rms {float(s['Bn_rms']):.3e}")
    ok &= gal < 1e-10

    # T3: linearization against central differences
    lin, linT = A.raw_lin(s), A.raw_lin_T(s)

    def raw(zz):
        ss = A.state(zz)
        return (ss["g"], ss["z"].ravel(), ss["rel"], jnp.atleast_1d(ss["Psi"]))

    names = ("g", "z (projected f)", "rel", "Psi")
    for tag in ("u", "xi", "all"):
        v = rng.normal(size=A.n)
        if tag == "u":
            v[A.n_u :] = 0
            v *= 1e-2 * np.abs(np.asarray(cur.u)).mean() / np.abs(v).mean()
        elif tag == "xi":
            v[: A.n_u] = 0
            v *= 1e-4 / np.abs(v).max()
        else:
            v[: A.n_u] *= 1e-2 * np.abs(np.asarray(cur.u)).mean() / np.abs(v[: A.n_u]).mean()
            v[A.n_u :] *= 1e-4 / np.abs(v[A.n_u :]).max()
        v = jnp.asarray(v)
        Jv = lin(v)
        errs = {}
        for h in (1e-1, 1e-2, 1e-3):
            fp, fm = raw(z + h * v), raw(z - h * v)
            errs[h] = [rel((p_ - m_) / (2 * h), j_) for p_, m_, j_ in zip(fp, fm, Jv)]
        best = [min(errs[h][i] for h in errs) for i in range(4)]
        print(f"T3 J v vs central differences ({tag}): " +
              ", ".join(f"{n} {e:.1e}" for n, e in zip(names, best)))
        ok &= max(best[i] for i in range(4) if float(jnp.linalg.norm(Jv[i])) > 0) < 1e-6
    v = jnp.asarray(rng.normal(size=A.n))
    w = tuple(jnp.asarray(rng.normal(size=x.shape)) for x in lin(v))
    a1 = sum(float(jnp.sum(wi * ji)) for wi, ji in zip(w, lin(v)))
    a2 = float(linT(w) @ v)
    print(f"T3 adjoint: {abs(a1 - a2) / abs(a1):.1e}")
    ok &= abs(a1 - a2) / abs(a1) < 1e-10

    # T6: envelope (the pressure held at p* gives the reduced gradient in u)
    v = rng.normal(size=A.n)
    v[A.n_u :] = 0
    v = jnp.asarray(v * 1e-2 * np.abs(np.asarray(cur.u)).mean() / np.abs(v).mean())
    xf = jnp.concatenate([z[: A.n_u], jnp.zeros(A.n_p), z[A.n_u :]])
    base, _ = mf._fb_base(fb, xf)
    d = fb._tangent(jnp.concatenate([v[: A.n_u], jnp.zeros(A.n_p), v[A.n_u :]]), base)
    da = (s["sw"][:, None] * (jnp.cross(d["J"], s["B"]) + jnp.cross(s["J"], d["B"]))).ravel()
    pred = float(s["f"] @ da)
    h = 1e-2
    fd = (0.5 * float(jnp.sum(A.state(z + h * v)["f"] ** 2))
          - 0.5 * float(jnp.sum(A.state(z - h * v)["f"] ** 2))) / (2 * h)
    print(f"T6 envelope d|f|^2/2 along u: {abs(fd - pred) / abs(pred):.1e}")
    ok &= abs(fd - pred) / abs(pred) < 1e-6

    # D: dense rows against J v at xi = 0
    zu = jnp.concatenate([z[: A.n_u], jnp.zeros(A.n_xi)])
    A.build_tables(z[: A.n_u])
    s0 = A.state(zu)
    W = mf.Weights(0.7 * float(s0["M"]), 0.9 * float(s0["Bn_rms"]), float(s0["Psi"]) * 1.001,
                   nu=(0.1, 0.05, 0.01), rho=1.0)
    rows = mf.Rows(A, s0, W, ref["norm"])
    Lu = np.linalg.cholesky(np.eye(A.n_u) * 2.0 + 0.1 * np.ones((A.n_u, A.n_u)))
    Lx = np.linalg.cholesky(np.eye(A.n_xi) * 3.0)
    Cd = rows.dense(jnp.asarray(Lu), jnp.asarray(np.linalg.inv(Lx).T))
    L = np.zeros((A.n, A.n))
    L[: A.n_u, : A.n_u], L[A.n_u :, A.n_u :] = Lu, Lx
    met = pw.Metric(L)
    ycols = rng.normal(size=(A.n, 3))
    eD = max(rel(Cd[:, :-1] @ jnp.asarray(ycols[:, i]), rows.J(met.back(jnp.asarray(ycols[:, i]))))
             for i in range(3))
    eR = rel(Cd[:, -1], rows.r)
    print(f"D dense [J L^-T | r] vs J L^-T y: {eD:.1e}, r {eR:.1e}")
    ok &= eD < 1e-10 and eR < 1e-14

    # R: the model's gradient is the merit's
    gL = rows.grad()
    v = rng.normal(size=A.n)
    v[: A.n_u] *= 1e-2 * np.abs(np.asarray(cur.u)).mean() / np.abs(v[: A.n_u]).mean()
    v[A.n_u :] *= 1e-4 / np.abs(v[A.n_u :]).max()
    v = jnp.asarray(v)
    errs = []
    for h in (1e-1, 1e-2, 1e-3):
        fp, fm = W.merit(A.state(zu + h * v))[0], W.merit(A.state(zu - h * v))[0]
        errs.append(abs((fp - fm) / (2 * h) - float(gL @ v)) / abs(float(gL @ v)))
    print(f"R grad L_A = J^T r vs central differences of the plain merit: {min(errs):.1e}")
    ok &= min(errs) < 1e-5

    # I: iota of the start against DESC's
    zs = jnp.concatenate([jnp.asarray(cur.u), jnp.zeros(A.n_xi)])
    ss = A.state(zs)
    io = np.asarray(mf.iota_profile(fb, jnp.zeros(A.n_xi), ss["B"]))
    rho = np.asarray(fr.tgrid.rho)
    io_desc = np.asarray(inp.iota(rho)[0])
    print("I iota (NEST start / DESC): " + ", ".join(f"{r_:.2f}: {i:+.4f}/{j:+.4f}"
                                                   for r_, i, j in zip(rho[::3], io[::3], io_desc[::3])))
    core = rho > 0.1  # (the averages of B^theta are ill-conditioned at the axis)
    ok &= np.max(np.abs(io - io_desc)[core]) < 0.05 * np.max(np.abs(io_desc))

    # T5: a few steps of each method from the same start
    if not a.no_t5:
        Af, kap = A, 1.0
        L, HJ = mf.metric(A.A_table(), np.asarray(sw0), 1.0, np.eye(A.n_xi) * 1e4, kap)
        met = pw.Metric(L)
        z0 = zs
        s0 = A.state(z0)
        out = {}
        for step in ("gltr", "qr"):
            Wt = mf.Weights(0.7 * float(s0["M"]), 1.5 * float(s0["Bn_rms"]), float(s0["Psi"]),
                            rho=10.0)
            sol = mf.Solver(Af, met, Wt, ref["norm"], step=step, max_disp=1.0, cg_maxiter=60,
                            log=lambda *x: None)
            if step == "qr":
                A.build_tables(z0[: A.n_u])
            f0 = Wt.merit(s0)[0]
            zz, hist, _ = sol.inner(z0, maxiter=6)
            out[step] = (f0, Wt.merit(A.state(zz))[0], sum(h_["accepted"] for h_ in hist))
            print(f"T5 {step}: L_A {out[step][0]:.6e} -> {out[step][1]:.6e} "
                  f"({out[step][2]} accepted of {len(hist)}); "
                  f"GN share {[round(float(h_['GN share']), 3) for h_ in hist]}")
        dif = abs(out["gltr"][1] - out["qr"][1]) / abs(out["gltr"][0] - out["qr"][1])
        print(f"T5 after 6 steps: |L_gltr - L_qr| / decrease {dif:.2e}")
        ok &= out["gltr"][1] < out["gltr"][0] and out["qr"][1] < out["qr"][0]
    print(f"{'ALL PASSED' if ok else 'FAILED'}; {time.time() - t0:.0f} s")
    sys.exit(0 if ok else 1)


if __name__ == "__main__":
    main()
