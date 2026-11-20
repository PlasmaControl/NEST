"""Resolution ladder for the (2,1)-rippled candidate-C tokamak: does DESC's shielding current become a sheet?

Two stages, run as separate jobs:

  --stage base    re-solve the axisymmetric base (N = 0) at L = M = --L and save base_L{L}.h5. Raising L, M
                  from the L32 base is a zero-padded truncation, not an equilibrium at the new resolution.
  --stage ripple  start from base_L{L}.h5 (or --init, an already rippled state, e.g. the same L at lower N),
                  raise N, put the (2,1) ripple xi = amp * a on the boundary (the same least-squares fit of a
                  normal displacement cos(2 theta + zeta) * nhat as make_bdry_ripple.py), perturb to first
                  order and solve force balance. Saves ripple_L{L}N{N}.h5 and a JSON with the cost history.

The ripple goes on in one step (no 1e-4 / 3e-4 continuation): at xi/a = 1e-3 the response is linear, and the
equilibrium does not depend on the path (checked against bp2/L32N4/..._x0.001.h5).
"""
import argparse
import json
import os
import time

import numpy as np

ap = argparse.ArgumentParser()
ap.add_argument("--stage", choices=["base", "ripple"], required=True)
ap.add_argument("--L", type=int, required=True, help="L = M")
ap.add_argument("--N", type=int, default=0)
ap.add_argument("--base", default="../tokamak_island_C_L32M32.h5", help="axisymmetric source for --stage base")
ap.add_argument("--init", default=None, help="--stage ripple: start from this rippled state instead of the base")
ap.add_argument("--amp", type=float, default=1e-3, help="xi / a")
ap.add_argument("--ftol", type=float, default=1e-6)
ap.add_argument("--maxiter", type=int, default=400)
ap.add_argument("--tr", default="qr", choices=["qr", "cho", "svd"], help="lsq-exact trust-region subproblem")
ap.add_argument("--chunk", type=int, default=None, help="jac_chunk_size (default: DESC's auto)")
ap.add_argument("--start", choices=["direct", "perturb"], default="direct",
                help="direct: set the rippled boundary and let the solve's linear constraints absorb it; "
                     "perturb: first-order perturbation (a bad start here: the resonant response is near-singular)")
ap.add_argument("--seg", type=int, default=0,
                help="ripple stage: solve in segments of this many iterations, saving after each (0: one solve)")
ap.add_argument("--budget", type=float, default=3.6, help="hours: stop starting segments past this")
ap.add_argument("--jblock", type=int, default=0,
                help="build the projected Jacobian in blocks of this many columns straight into an (m, n) buffer "
                     "(DESC's _jac makes (n, m) then transposes: two J-sized arrays at once)")
ap.add_argument("--snap", type=int, default=0, help="also keep a copy every this many iterations (_it<k>.h5)")
ap.add_argument("--tag", default="", help="suffix for the output names")
ap.add_argument("--out", default="out")
a = ap.parse_args()
os.makedirs(a.out, exist_ok=True)

from desc import set_device  # noqa: E402

if os.environ.get("NEST_DEVICE") == "gpu":
    set_device("gpu")
import desc.io  # noqa: E402
from desc.equilibrium import Equilibrium  # noqa: E402
from desc.grid import LinearGrid, QuadratureGrid  # noqa: E402

T0 = time.time()

if a.jblock > 0:
    import jax
    import jax.numpy as jnp
    from desc.optimize._constraint_wrappers import LinearConstraintProjection

    _put = jax.jit(lambda J, blk, i: jax.lax.dynamic_update_slice(J, blk.T, (0, i)), donate_argnums=0)

    def _jac_lowmem(self, x_reduced, constants=None, op="scaled"):
        x = self.recover(x_reduced)
        v = self._feasible_tangents
        n = v.shape[1]
        jvp = getattr(self._objective, "jvp_" + op)
        J = None
        for i in range(0, n, a.jblock):
            blk = jvp(v[:, i:i + a.jblock].T, x, constants)
            if J is None:
                J = jnp.zeros((blk.shape[1], n), dtype=blk.dtype)
            J = _put(J, blk, i)
            del blk
        return J

    LinearConstraintProjection._jac = _jac_lowmem

    # lsqtr's cho step does jnp.dot(J_a.T, J_a) eagerly: J_a.T is a full copy. Jitted, the transpose folds into
    # the GEMM. Re-exec least_squares.py with that one line changed (here only; the shared DESC is untouched).
    import inspect

    import desc.optimize._desc_wrappers as _dw
    import desc.optimize.least_squares as _ls

    _src = inspect.getsource(_ls)
    _line = "B_h = jnp.dot(J_a.T, J_a)"
    assert _src.count(_line) == 1, "lsqtr source changed: re-check the cho Gram line"
    _ls.__dict__["_gram"] = jax.jit(lambda A: A.T @ A)
    # g = f @ J and the column norms of J also made J-sized temporaries (XLA transposes for the gemv; A**2):
    # do both over column blocks
    _KB = 4096

    def _fJ(f, J):
        return jnp.concatenate([f @ J[:, i:i + _KB] for i in range(0, J.shape[1], _KB)])

    def _jac_scale_blocked(A, prev_scale_inv=None):
        si = jnp.concatenate([jnp.sum(A[:, i:i + _KB] ** 2, axis=0) for i in range(0, A.shape[1], _KB)]) ** 0.5
        si = jnp.where(si < jnp.finfo(A.dtype).eps * max(A.shape), 1, si)
        if prev_scale_inv is not None:
            si = jnp.maximum(si, prev_scale_inv)
        return 1 / si, si

    _ls.__dict__["_fJ"] = _fJ
    _src2 = _src.replace(_line, "B_h = _gram(J_a)")
    assert _src2.count("g = jnp.dot(f, J)") == 2, "lsqtr source changed: re-check the f @ J lines"
    _src2 = _src2.replace("g = jnp.dot(f, J)", "g = _fJ(f, J)")
    exec(compile(_src2, _ls.__file__, "exec"), _ls.__dict__)
    _ls.compute_jac_scale = _jac_scale_blocked
    _dw.lsqtr = _ls.lsqtr


def log(*s):
    print(f"[{time.time() - T0:8.1f} s]", *s, flush=True)


def load(path):
    eq = desc.io.load(path)
    return eq if isinstance(eq, Equilibrium) else eq[-1]


def force_ratio(eq, fac=1):
    g = QuadratureGrid(L=fac * eq.L_grid, M=fac * eq.M_grid, N=fac * eq.N_grid, NFP=eq.NFP)
    d = eq.compute(["<|F|>_vol", "<|grad(|B|^2)|/2mu0>_vol"], grid=g)
    return float(d["<|F|>_vol"] / d["<|grad(|B|^2)|/2mu0>_vol"])


def solve(eq, info, maxiter=None):
    maxiter = maxiter or a.maxiter
    t = time.time()
    from desc.objectives import ForceBalance, ObjectiveFunction
    obj = ObjectiveFunction(ForceBalance(eq=eq), jac_chunk_size=a.chunk) if a.chunk else "force"
    _, res = eq.solve(objective=obj, ftol=a.ftol, xtol=1e-14, gtol=1e-14, maxiter=maxiter, verbose=3,
                      options={"tr_method": a.tr})
    info["solve wall [s]"] = time.time() - t
    for k in ("nfev", "njev", "nit", "status", "message"):
        if k in res:
            v = res[k]
            info["solve " + k] = v if isinstance(v, str) else int(v)
    if "allcost" in res:
        info["cost history"] = [float(c) for c in np.asarray(res["allcost"]).ravel()]


info = {"args": vars(a)}
if a.stage == "base":
    eq = load(a.base)
    assert eq.N == 0 and eq.NFP == 1 and eq.sym
    log(f"source {a.base}: L {eq.L} M {eq.M}; force {force_ratio(eq):.3e}")
    eq.change_resolution(L=a.L, M=a.L, N=0, L_grid=2 * a.L, M_grid=2 * a.L, N_grid=0)
    info["force before"] = force_ratio(eq)
    log(f"-> L {eq.L} M {eq.M} (grid {eq.L_grid}/{eq.M_grid}); force {info['force before']:.3e}")
    solve(eq, info)
    path = os.path.join(a.out, f"base_L{a.L}.h5")
    eq.save(path)
    info["force"] = force_ratio(eq)
    info["force x2"] = force_ratio(eq, 2) if a.L <= 64 else None  # the 2x quadrature grid OOMs host RAM above
    log(f"saved {path}: force {info['force']} (x2 grid {info['force x2']})")
    with open(os.path.join(a.out, f"base_L{a.L}.json"), "w") as f:
        json.dump(info, f, indent=1, default=float)
    raise SystemExit

# ---------------------------------------------------------------- ripple stage
# the base supplies the (fixed, L-independent) boundary and minor radius; any resolution will do
bp = os.path.join(a.out, f"base_L{a.L}.h5")
if not os.path.exists(bp):
    import glob
    bp = sorted(glob.glob(os.path.join(a.out, "base_L*.h5")))[0]
base = load(bp)
info["base"] = bp
minor_a = float(base.compute("a")["a"])
base.change_resolution(L=a.L, M=a.L, N=a.N, L_grid=2 * a.L, M_grid=2 * a.L, N_grid=2 * a.N)
log(f"base L{a.L} -> N {a.N}; a = {minor_a:.6f} m")

# the ripple: normal displacement xi cos(2 theta + zeta) nhat, fitted onto the base's boundary basis
m_r, k_r = 2, 1  # cos(m theta + k zeta), k = -m iota at q = 2 (iota = -1/2): resonant helicity
srf = base.surface
gs = LinearGrid(rho=np.array([1.0]), M=4 * srf.M + 8, N=4 * max(srf.N, 1) + 8, NFP=1, sym=False)
ds = srf.compute(["R", "Z", "R_t", "Z_t"], grid=gs)
R, Z, Rt, Zt = (np.asarray(ds[q]) for q in ("R", "Z", "R_t", "Z_t"))
nR, nZ = Zt, -Rt
nn = np.hypot(nR, nZ)
nR, nZ = nR / nn, nZ / nn
sgn = np.sign(np.mean(nR * (R - R.mean()) + nZ * (Z - Z.mean())))
nR, nZ = sgn * nR, sgn * nZ
shape = np.cos(m_r * gs.nodes[:, 1] + k_r * gs.nodes[:, 2])
AR, AZ = np.asarray(srf.R_basis.evaluate(gs)), np.asarray(srf.Z_basis.evaluate(gs))
cR, *_ = np.linalg.lstsq(AR, shape * nR, rcond=None)
cZ, *_ = np.linalg.lstsq(AZ, shape * nZ, rcond=None)
fit_err = max(np.abs(AR @ cR - shape * nR).max(), np.abs(AZ @ cZ - shape * nZ).max())
info["ripple fit error"] = float(fit_err)
log(f"ripple fit error {fit_err:.2e} per unit xi")
xi = a.amp * minor_a
Rb_t = np.asarray(base.Rb_lmn) + xi * cR
Zb_t = np.asarray(base.Zb_lmn) + xi * cZ

if a.init:
    eq = load(a.init)
    log(f"init {a.init}: L {eq.L} M {eq.M} N {eq.N}")
    eq.change_resolution(L=a.L, M=a.L, N=a.N, L_grid=2 * a.L, M_grid=2 * a.L, N_grid=2 * a.N)
    eq.Rb_lmn, eq.Zb_lmn = Rb_t, Zb_t  # same ripple (it lives in n = +-1 only; the fit is exact)
    info["init"] = a.init
elif a.start == "direct":
    eq = base
    eq.Rb_lmn, eq.Zb_lmn = Rb_t, Zb_t
else:
    eq = base
    t = time.time()
    eq.perturb(deltas={"Rb_lmn": Rb_t - np.asarray(eq.Rb_lmn), "Zb_lmn": Zb_t - np.asarray(eq.Zb_lmn)},
               order=1, verbose=2, copy=False)
    info["perturb wall [s]"] = time.time() - t
if a.L <= 64:  # the volume force on the quadrature grid takes ~15 min at L = 96
    info["force before"] = force_ratio(eq)
    log(f"start force {info['force before']:.3e}")
path = os.path.join(a.out, f"ripple_L{a.L}N{a.N}{a.tag}.h5")
info["a [m]"], info["xi [m]"] = minor_a, xi


def checkpoint(final):
    eq.save(path)
    # the volume force on the quadrature grid costs ~15 min at L = 96: only at the end there
    info["force"] = force_ratio(eq) if (final or a.L <= 64) else info.get("force")
    info["force x2"] = force_ratio(eq, 2) if (final and a.L <= 64) else None
    with open(os.path.join(a.out, f"ripple_L{a.L}N{a.N}{a.tag}.json"), "w") as f:
        json.dump(info, f, indent=1, default=float)
    log(f"saved {path}: force {info['force']} (x2 grid {info['force x2']})")


if a.seg <= 0:
    solve(eq, info)
    checkpoint(True)
else:
    info["segments"], done, t_seg = [], 0, 0.0
    while done < a.maxiter:
        t1 = time.time()
        seg = {}
        solve(eq, seg, maxiter=min(a.seg, a.maxiter - done))
        nit = seg.get("solve nit", a.seg)
        done += nit
        t_seg = time.time() - t1
        seg["wall [s]"] = t_seg
        info["segments"].append(seg)
        info["solve nit"] = done
        stop = nit < a.seg or (time.time() - T0 + 1.2 * t_seg) / 3600 > a.budget
        checkpoint(stop or done >= a.maxiter)
        if a.snap and (done // a.snap) > ((done - nit) // a.snap):
            sp = path[:-3] + f"_it{done}.h5"
            eq.save(sp)
            log(f"snapshot {sp}")
        log(f"segment: {nit} its ({done} total) in {t_seg:.0f} s; {seg.get('solve message', '')}")
        if stop:
            break
