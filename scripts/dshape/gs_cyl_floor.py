"""Where does the cylindrical-annulus B_plasma floor (~5e-9 of rms B at a/320 and a/640) come from?

At the J-fitted start x0 of the L = M = 32 free-boundary tokamak (basis x2, edge power 3), at
h = a / frac, B_plasma at the collocation points against the exact axisymmetric field, changing
one ingredient at a time:
  reference   the exact field at (n_gamma, n_s) = (256, 48) against (512, 96);
  locate      the nodes' flux coordinates from map_to_flux at atol = 1e-8 m (the default) or
              1e-13 m; reported: |X(rho, theta) - node| over the inside nodes;
  J at nodes  interpolated from the flux grid (Fields' default) or the basis evaluated at
              every inside node directly (no flux grid).
Each row: rms |B - B_exact| / rms |B_DESC| overall and by rho bin.  Writes
results/gs/cyl_floor_a<frac>.json.

run (GPU): sbatch run.sbatch scripts/dshape/gs_cyl_floor.py --frac 320
"""

import argparse
import functools
import json
import os
import sys
import time

import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.join(ROOT, "nest"))

from desc import set_device  # noqa: E402

set_device(os.environ.get("NEST_DEVICE", "cpu"))

import jax  # noqa: E402
import jax.numpy as jnp  # noqa: E402

import fit_cache  # noqa: E402
from axisym_field import plasma_field_cached  # noqa: E402
from boundary_potential import BoundaryPotential, _current  # noqa: E402
from cartesian_grid import cylindrical_to_cartesian  # noqa: E402
from cylindrical_grid import CylindricalGrid  # noqa: E402
from dense import force_matched_u  # noqa: E402
from desc_input import DESCInput  # noqa: E402
from force import ForceResidual  # noqa: E402
from objective import Fields, _cartesian  # noqa: E402
from pressure import Pressure  # noqa: E402

jax.config.update("jax_enable_x64", True)
TOK = os.path.join(ROOT, "desc_input", "tokamak_free_boundary")
EDGES = (0.0, 0.5, 0.8, 0.9, 0.95, 1.0)


def node_J(cur, u, rtz, chunk=20000):
    """J = curl T of the basis at the nodes' flux coordinates, (R, phi, Z), no flux grid."""
    f = jax.jit(functools.partial(_current, cur.S, cur._RZ))
    out = []
    for i in range(0, len(rtz), chunk):
        p = rtz[i : i + chunk]
        n = len(p)
        if n < chunk:
            p = np.concatenate([p, np.repeat(p[-1:], chunk - n, 0)])
        out.append(np.asarray(f(jnp.asarray(u), cur.I, *(jnp.asarray(p[:, k]) for k in range(3))))[:n])
    return np.concatenate(out)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--eq", default=os.path.join(TOK, "tokamak_L32M32_fb.h5"))
    ap.add_argument("--coils", default=os.path.join(TOK, "tokamak_L32M32_field.h5"))
    ap.add_argument("--frac", type=int, default=320)
    ap.add_argument("--margin", type=float, default=2.0)
    ap.add_argument("--atols", type=float, nargs="+", default=[1e-8, 1e-13])
    a = ap.parse_args()
    t0 = time.time()
    inp = DESCInput.load(a.eq, field=a.coils)
    eq = inp.eq
    cur = fit_cache.fit(BoundaryPotential, inp, factors=(2, 2, 1), edge_power=3)
    pre = fit_cache.fit(Pressure, inp, factors=(2, 2, 1), edge_order=2)
    b = cur.basis
    res = tuple(int(round(2 * (p + q))) for p, q in ((b.L, eq.L), (b.M, eq.M), (b.N, eq.N)))
    fr0 = ForceResidual(inp, None, resolution=res)
    desc = {k: np.asarray(v) for k, v in fr0.desc_fields().items()}
    u0, _ = force_matched_u(fr0, cur, eps=np.inf, desc=desc)
    rtz, phi = np.asarray(fr0.rtz), np.asarray(fr0.phi)
    w = fr0.dV / fr0.dV.sum()
    rho = rtz[:, 0]
    rmsB = float(np.sqrt(np.sum(w * np.sum(desc["B"] ** 2, 1))))

    def err(e):
        m_all = float(np.sqrt(np.sum(w * np.sum(e**2, 1)))) / rmsB
        bins = []
        for lo, hi in zip(EDGES[:-1], EDGES[1:]):
            m = (rho > lo) & (rho <= hi)
            bins.append(float(np.sqrt(np.sum((w * np.sum(e**2, 1))[m]) / np.sum(w[m]))) / rmsB)
        return m_all, bins

    rows = []

    def report(name, e, **extra):
        m, bins = err(e)
        rows.append({"case": name, "err": m, "by rho": bins, **extra})
        print(f"{name:52s} {m:9.2e}   " + " ".join(f"{v:8.1e}" for v in bins)
              + "".join(f"  {k} {v}" for k, v in extra.items()), flush=True)
        json.dump({"args": vars(a), "rho bins": EDGES, "rows": rows},
                  open(os.path.join(ROOT, "results", "gs", f"cyl_floor_a{a.frac}.json"), "w"), indent=1)

    B1 = np.asarray(cylindrical_to_cartesian(plasma_field_cached(cur, rtz, u=u0), phi))
    B2 = np.asarray(cylindrical_to_cartesian(plasma_field_cached(cur, rtz, u=u0, n_gamma=512, n_s=96), phi))
    print(f"{inp}\nh = a/{a.frac}; collocation {fr0.tgrid.shape}; exact fields {time.time() - t0:.0f} s\n"
          f"{'case':52s} {'rms err':>9s}   " + " ".join(f"{lo:.2f}-{hi:.2f}"[:8].rjust(8)
                                                      for lo, hi in zip(EDGES[:-1], EDGES[1:])), flush=True)
    report("reference: exact (256, 48) - exact (512, 96)", B1 - B2)
    for atol in a.atols:
        t1 = time.time()
        g = CylindricalGrid.around(inp, inp.a / a.frac, a.margin)
        zs = g.z_axis
        sym = bool(eq.sym) and abs(zs.a + zs.b) <= 1e-12 * max(abs(zs.a), abs(zs.b))
        box = g.locate_plasma(inp, sym=sym, atol=atol, maxiter=60)
        t_loc = time.time() - t1
        # how well the nodes' flux coordinates reproduce the nodes
        nrtz = np.asarray(box.rtz)
        X = np.concatenate([np.asarray(cur._RZ.evaluate(*(jnp.asarray(nrtz[i:i + 50000, k]) for k in range(3))))
                            for i in range(0, len(nrtz), 50000)])
        k_, i_, l_ = np.unravel_index(np.asarray(box.inside), box.shape)
        node = np.column_stack([box.r_axis.s[i_], box.z_axis.s[l_]])
        miss = np.linalg.norm(X[:, :2] - node, axis=1)
        print(f"  locate atol {atol:g}: {len(nrtz):,} inside nodes, {t_loc:.0f} s; |X(rho, theta) - node| "
              f"max {miss.max():.1e} m, rms {np.sqrt(np.mean(miss**2)):.1e} m", flush=True)
        fr = ForceResidual(inp, box, resolution=res)
        fields = Fields(fr, cur, pre, kernel="cylindrical", dtype=jnp.float64, cyl_half=False)
        Bf = np.asarray(fields.B_plasma(jnp.asarray(u0)))
        report(f"atol {atol:g}, J from the flux grid (default)", Bf - B2, miss_max=f"{miss.max():.1e}")
        # J evaluated at the nodes directly
        Jn = node_J(cur, u0, nrtz)
        Jg = np.asarray(fields.J_box(jnp.asarray(u0))).reshape(-1, 3)[np.asarray(box.inside)]
        dJ = float(np.sqrt(np.mean(np.sum((Jg - Jn) ** 2, 1)) / np.mean(np.sum(Jn**2, 1))))
        Jpad = jnp.pad(jnp.asarray(Jn), ((0, 1), (0, 0)))
        Jgrid = Jpad[fields.src].reshape(fields.shape + (3,))
        Bp, Bz = fields.solver.field_modes(Jgrid)
        Bd = np.asarray(_cartesian(fields.to_points(Bp, Bz), fields.phi))
        report(f"atol {atol:g}, J evaluated at the nodes directly", Bd - B2, flux_vs_direct_J=f"{dJ:.1e}")
        del fields, fr, box
        jax.clear_caches()
    print(f"total {time.time() - t0:.0f} s")


if __name__ == "__main__":
    main()
