"""The accurate axisymmetric plasma field (axisym_field.py) against the box, at x0.

At the J-fitted start x0 (J fitted to J_DESC on the run's collocation points) of
the L = M = 32 free-boundary tokamak in its coils' field:
  - convergence of axisym_field.plasma_field in (n_gamma, n_s) on a subset of points;
  - rms |B_coil + B_plasma(J(x0)) - B_DESC| / rms |B_DESC| by rho, toroidal and
    poloidal, for the accurate field (and the box at a/20 from gs_box_convergence:
    3.1e-5);
  - <|J x B - grad p|> / norm at x0 with that B, against DESC's own.
If the accurate field brings B(x0) to B_DESC within DESC's own consistency, the
coils-only NEST start equals DESC's state, with no B_DESC in B_ext.

run (GPU): sbatch run.sbatch scripts/gs_axisym_check.py
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
from axisym_field import plasma_field  # noqa: E402
from boundary_potential import BoundaryPotential  # noqa: E402
from cartesian_grid import cylindrical_to_cartesian  # noqa: E402
from dense import force_matched_u  # noqa: E402
from desc_input import DESCInput  # noqa: E402
from force import ForceResidual  # noqa: E402
from pressure import Pressure  # noqa: E402

jax.config.update("jax_enable_x64", True)
TOK = os.path.join(ROOT, "desc_input", "tokamak_free_boundary")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--eq", default=os.path.join(TOK, "tokamak_L32M32_fb.h5"))
    ap.add_argument("--coils", default=os.path.join(TOK, "tokamak_L32M32_field.h5"))
    ap.add_argument("--factors", type=float, nargs=3, default=(2, 2, 1))
    ap.add_argument("--edge-power", type=int, default=2)
    ap.add_argument("--res", type=int, nargs="+", default=[64, 16, 128, 32, 256, 48, 384, 64],
                    help="pairs n_gamma n_s for the convergence test")
    a = ap.parse_args()
    t0 = time.time()
    inp = DESCInput.load(a.eq, field=a.coils)
    eq = inp.eq
    ep = {"edge_power": a.edge_power} if a.edge_power != 2 else {}
    cur = fit_cache.fit(BoundaryPotential, inp, factors=tuple(a.factors), **ep)
    pre = fit_cache.fit(Pressure, inp, factors=tuple(a.factors), edge_order=2)
    b = cur.basis
    res = tuple(int(round(2 * (u + v))) for u, v in ((b.L, eq.L), (b.M, eq.M), (b.N, eq.N)))
    fr = ForceResidual(inp, None, resolution=res)
    desc = {k: np.asarray(v) for k, v in fr.desc_fields().items()}
    u, _ = force_matched_u(fr, cur, eps=np.inf, desc=desc)
    rtz, phi = np.asarray(fr.rtz), np.asarray(fr.phi)
    w = fr.dV / fr.dV.sum()
    rho = rtz[:, 0]
    rmsB = np.sqrt(np.sum(w * np.sum(desc["B"] ** 2, 1)))
    B_coil = np.asarray(fr.B_ext)
    print(f"{inp}\ncollocation {fr.tgrid.shape} ({len(rho):,} points); setup {time.time() - t0:.0f} s",
          flush=True)

    # convergence in (n_gamma, n_s) on every 25th point
    sub = np.arange(0, len(rho), 25)
    prev = None
    for ng, ns in zip(a.res[::2], a.res[1::2]):
        t1 = time.time()
        B = plasma_field(cur, rtz[sub], u=u, n_gamma=ng, n_s=ns)
        msg = ""
        if prev is not None:
            msg = f"  change from the previous: {np.sqrt(np.mean(np.sum((B - prev) ** 2, 1))) / rmsB:.2e}"
        print(f"  n_gamma {ng:4d} n_s {ns:3d}: {time.time() - t1:6.1f} s for {len(sub)} points{msg}", flush=True)
        prev = B
    ng, ns = a.res[-2], a.res[-1]
    t1 = time.time()
    B_pl = cylindrical_to_cartesian(plasma_field(cur, rtz, u=u, n_gamma=ng, n_s=ns), phi)
    B_pl = np.asarray(B_pl)
    print(f"all points at n_gamma {ng}, n_s {ns}: {time.time() - t1:.0f} s", flush=True)
    e = B_coil + B_pl - desc["B"]
    ephi = np.stack([-np.sin(phi), np.cos(phi), 0 * phi], 1)
    et = np.sum(e * ephi, 1)
    ep_ = e - et[:, None] * ephi
    edges = (0.0, 0.5, 0.8, 0.9, 0.95, 0.99, 1.0)

    def bins(v):
        return [np.sqrt(np.sum(w * v) / np.sum(w))] + [
            np.sqrt(np.sum((w * v)[(rho > x) & (rho <= y)]) / np.sum(w[(rho > x) & (rho <= y)]))
            for x, y in zip(edges[:-1], edges[1:])]

    names = ["all"] + [f"{x:g}-{y:g}" for x, y in zip(edges[:-1], edges[1:])]
    print(f"\n|B_coil + B_plasma,accurate(J(x0)) - B_DESC| / rms B_DESC:\n{'':10s}"
          + " ".join(f"{n:>9s}" for n in names))
    for lab, v in (("total", np.sum(e**2, 1)), ("toroidal", et**2), ("poloidal", np.sum(ep_**2, 1))):
        print(f"{lab:10s}" + " ".join(f"{x / rmsB:9.2e}" for x in bins(v)))
    # the force at x0 with this B
    J = np.asarray(cylindrical_to_cartesian(fr._series(cur.with_u(u).point_evaluator(eq), rtz), phi))
    pg = np.asarray(fr._series(pre.point_evaluator(eq), rtz))
    grad_p = np.asarray(cylindrical_to_cartesian(pg[:, 1:], phi))
    F = np.cross(J, B_coil + B_pl) - grad_p
    FD = np.cross(desc["J"], desc["B"]) - desc["grad(p)"]
    print(f"\n<|F|>/norm by rho:\n{'':10s}" + " ".join(f"{n:>9s}" for n in names))
    for lab, FF in (("DESC", FD), ("NEST x0", F)):
        f = np.linalg.norm(FF, axis=1) / fr.norm
        v = [np.sum(w * f)] + [np.sum((w * f)[(rho > x) & (rho <= y)]) / np.sum(w[(rho > x) & (rho <= y)])
                               for x, y in zip(edges[:-1], edges[1:])]
        print(f"{lab:10s}" + " ".join(f"{x:9.2e}" for x in v))
    np.savez(os.path.join(ROOT, "results", "gs", "axisym_check.npz"), B_pl=B_pl, e=e, rtz=rtz, w=w)


if __name__ == "__main__":
    main()
