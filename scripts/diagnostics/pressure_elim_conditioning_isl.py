"""Conditioning of an isl_scan HarmonicModel state with p free (joint) vs p eliminated (speed elim), at production size.

The model is built exactly as scripts/dshape/isl_scan.py builds it (same setup / model_kwargs, same flags), the state
x is read from a scan's npz (embedded in these bases when its N or pressure space differs), and one streamed TSQR of
[A | r] gives R. Reported (as scripts/diagnostics/pressure_elim_conditioning.py): kappa of the joint, reduced and
u-only problems, Jacobi-scaled and whitened by the physical metric, and the GN shares. SVDs and QRs run on the
device (jax.numpy); n = 16k (N = 2) needs ~2 GB per n x n float64 matrix.

example (Stellar, from the N = 2 state at 1000 A):
  python scripts/diagnostics/pressure_elim_conditioning_isl.py --base results/gs/hN_a120D --N 2 --eps 1000 \
      --w-J 1e-7 --j-mode full --w-W 0 --w-pos 0 --batch 16 8 --state results/isl/a120Cn2d1000_eps1000.npz \
      [--p-factors 3 3 1] [--speed resident,gpus=4] --out results/isl/cond_a120Cn2d1000.json
"""
import argparse
import json
import os
import sys
import time
import types

import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path[:0] = [os.path.join(ROOT, "nest"), os.path.join(ROOT, "scripts", "dshape")]
sys.argv, _argv = sys.argv[:1], sys.argv
import isl_scan  # noqa: E402

sys.argv = _argv
import jax  # noqa: E402
import jax.numpy as jnp  # noqa: E402

import fit_cache  # noqa: E402
from boundary_potential import BoundaryPotential  # noqa: E402
from harmonic_dense import HarmonicModel, embed  # noqa: E402
from perturbation import Perturbation  # noqa: E402
from pressure import Pressure  # noqa: E402


def kappa(M, scale=True):
    if scale:
        cn = jnp.linalg.norm(M, axis=0)
        M = M / jnp.where(cn > 0, cn, 1.0)
    s = jnp.linalg.svd(M, compute_uv=False)
    return float(s[0] / s[-1])


def analyse(model, z, eps, log):
    t0 = time.time()
    R = model._tsqr(jnp.asarray(z), eps)
    n, nu = model.n, model.n_uc
    log(f"  TSQR of [A | r]: {R.shape}, {time.time() - t0:.0f} s")
    iu, ip = jnp.arange(nu), jnp.arange(nu, n)
    L = model._phys_L()  # block diagonal (u, p): its blocks are the Cholesky factors of the two metrics
    Luu, Lpp = L[:nu, :nu], L[nu:, nu:]
    R2 = jnp.linalg.qr(R[:, jnp.concatenate([ip, iu, jnp.array([n])])], mode="r")
    npp = n - nu
    Tu = R2[npp:npp + nu, npp:npp + nu]
    white = lambda M, Lw: jax.scipy.linalg.solve_triangular(Lw, M.T, lower=True).T  # noqa: E731
    Rj = R[:, jnp.concatenate([iu, ip])]
    c, cu = R[:n, n], R2[npp:npp + nu, -1]
    f = 0.5 * float(c @ c + R[n, n] ** 2)
    out = {"n_u": int(nu), "n_p": int(npp), "f": f}
    for k, fn in (("joint_jacobi", lambda: kappa(Rj)), ("reduced_jacobi", lambda: kappa(Tu)),
                  ("joint_phys", lambda: kappa(white(Rj, jax.scipy.linalg.block_diag(Luu, Lpp)), False)),
                  ("reduced_phys", lambda: kappa(white(Tu, Luu), False)),
                  ("u_only_phys", lambda: kappa(white(R[:, iu], Luu), False))):
        t1 = time.time()
        out[k] = fn()
        log(f"  kappa {k}: {out[k]:.3e} ({time.time() - t1:.0f} s)")
    out["gn_share_joint"] = float(c @ c) / (2 * f)
    out["gn_share_reduced"] = float(cu @ cu) / (2 * f)
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--base", default=None, help="the base run (isl_scan --base); without it a fresh start from --eq")
    ap.add_argument("--eq", default=os.path.join(isl_scan.TOK, "tokamak_L32M32_fb.h5"))
    ap.add_argument("--coils", default=os.path.join(isl_scan.TOK, "tokamak_L32M32_field.h5"))
    ap.add_argument("--factors", type=float, nargs=3, default=(3, 3, 1))
    ap.add_argument("--frac", type=int, default=320)
    ap.add_argument("--state", required=True, help="a scan's <name>_eps<eps>.npz (x, N)")
    ap.add_argument("--N", type=int, default=2)
    ap.add_argument("--eps", type=float, default=1000.0)
    ap.add_argument("--w-J", type=float, default=None)
    ap.add_argument("--w-W", type=float, default=None)
    ap.add_argument("--w-pos", type=float, default=None)
    ap.add_argument("--j-mode", default="full", choices=["full", "n0"])
    ap.add_argument("--p-factors", type=float, nargs=3, default=None)
    ap.add_argument("--m", type=int, default=2)
    ap.add_argument("--n", type=int, default=-1)
    ap.add_argument("--chunk", type=int, default=2)
    ap.add_argument("--batch", type=int, nargs=2, default=(4, 2))
    ap.add_argument("--speed", default=None, help="device options only (resident, gpus=G, merge=M)")
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    log = lambda s: print(s, flush=True)  # noqa: E731
    t0 = time.time()
    ns = types.SimpleNamespace(base=a.base, eq=a.eq, coils=a.coils, factors=a.factors, edge_power=3, frac=a.frac, N=a.N, p_factors=a.p_factors, w_J=a.w_J, w_W=a.w_W, w_pos=a.w_pos,
                               j_mode=a.j_mode, speed=a.speed, transport=None, par=None, moments=None, rough=None)
    inp, cur, pre, x_base, fr, run = isl_scan.setup(ns)
    model = HarmonicModel(inp, cur, pre, fr, run["frac"], run.get("margin", 2.0), pert=Perturbation(inp.field, a.m, a.n),
                          chunk=a.chunk, batch_u=tuple(a.batch), **isl_scan.model_kwargs(ns, run, cur, fr))
    model.set_source(x_base)
    d = np.load(a.state)
    N0 = int(d["N"]) if "N" in d else a.N
    ep = run.get("edge_power", 2)
    kw = {"edge_power": ep} if ep != 2 else {}
    c0 = fit_cache.fit(BoundaryPotential, inp, factors=tuple(run["factors"]), N=N0, **kw)
    p0 = fit_cache.fit(Pressure, inp, factors=tuple(run["factors"]), N=N0, edge_order=2)
    x = embed(np.asarray(d["x"]), c0, p0, cur, pre)
    log(f"model ready ({time.time() - t0:.0f} s); state {a.state} (N = {N0}) embedded; n = {model.n:,}")
    f, F = model.evaluate(x, a.eps)
    log(f"  f {f:.6e}, <|F|>/norm {F:.4e} at eps {a.eps:g}")
    out = {"args": vars(a), "f": f, "<|F|>/norm": F, "kappa": analyse(model, model.to_z(x), a.eps, log),
           "time": time.time() - t0}
    json.dump(out, open(a.out, "w"), indent=1)
    log(f"-> {a.out} ({out['time']:.0f} s)")


if __name__ == "__main__":
    main()
