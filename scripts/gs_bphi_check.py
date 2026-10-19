"""Where is the box error of B_plasma, toroidal or poloidal?  And the exact toroidal part.

In axisymmetry the plasma's toroidal field comes from its poloidal current alone, and
Ampere's law on the toroidal circle through a point with Stokes on J = curl T gives it
exactly: the poloidal current through the flat annulus between the LCFS and the
circle is 2 pi (T_zeta(point) - T_zeta(LCFS)), and T_zeta = 0 on the LCFS
(BoundaryPotential: f(1) = 0, d_zeta chi = 0), so

    B_phi,plasma = mu0 T_zeta / R          (no quadrature, no box).

At the J-fitted start x0 on the run's collocation grid, with the coils' field:
  - e = B_coil + B_box - B_DESC split into its toroidal and poloidal parts, by rho;
  - the same with B_phi,plasma replaced by mu0 T_zeta / R (then e_phi is DESC's own
    free-boundary consistency plus the fit, the box is gone from it).

run (GPU): sbatch run.sbatch scripts/gs_bphi_check.py [--frac 20]
"""

import argparse
import os
import sys

import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, "nest"))

from desc import set_device  # noqa: E402

set_device(os.environ.get("NEST_DEVICE", "cpu"))

import jax  # noqa: E402
import jax.numpy as jnp  # noqa: E402

import fit_cache  # noqa: E402
from boundary_potential import BoundaryPotential  # noqa: E402
from current_potential import MU_0  # noqa: E402
from dense import force_matched_u  # noqa: E402
from desc_input import DESCInput  # noqa: E402
from force import ForceResidual  # noqa: E402
from grid import CartesianGrid  # noqa: E402
from objective import Fields  # noqa: E402
from pressure import Pressure  # noqa: E402

jax.config.update("jax_enable_x64", True)
TOK = os.path.join(ROOT, "desc_input", "tokamak_free_boundary")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--eq", default=os.path.join(TOK, "tokamak_L32M32_fb.h5"))
    ap.add_argument("--coils", default=os.path.join(TOK, "tokamak_L32M32_field.h5"))
    ap.add_argument("--factors", type=float, nargs=3, default=(2, 2, 1))
    ap.add_argument("--frac", type=int, default=20)
    a = ap.parse_args()
    inp = DESCInput.load(a.eq, field=a.coils)
    eq = inp.eq
    cur = fit_cache.fit(BoundaryPotential, inp, factors=tuple(a.factors))
    pre = fit_cache.fit(Pressure, inp, factors=tuple(a.factors), edge_order=2)
    b = cur.basis
    res = tuple(int(round(2 * (u + v))) for u, v in ((b.L, eq.L), (b.M, eq.M), (b.N, eq.N)))
    box = CartesianGrid.periodic_box(inp, inp.a / a.frac, odd=bool(eq.sym)).locate_plasma(inp)
    fr = ForceResidual(inp, box, resolution=res)
    fields = Fields(fr, cur, pre, kernel="truncated", dtype=jnp.float64)
    desc = {k: np.asarray(v) for k, v in fields.desc.items()}
    u, _ = force_matched_u(fr, cur, eps=np.inf, desc=desc)
    x0 = jnp.concatenate([jnp.asarray(u), fields.x0[fields.n_u:]])
    f = {k: np.asarray(v) for k, v in fields.compute(x0).items()}
    w = np.asarray(fields.sw) ** 2
    rho = np.asarray(fr.rtz)[:, 0]
    phi = np.asarray(fr.phi)
    ephi = np.stack([-np.sin(phi), np.cos(phi), 0 * phi], 1)
    R = np.hypot(np.asarray(fr.x)[:, 0], np.asarray(fr.x)[:, 1])
    rmsB = fields.rms["B"]
    edges = (0.0, 0.5, 0.8, 0.9, 0.95, 1.0)

    def split(e):
        ep = np.sum(e * ephi, 1)
        epol = e - ep[:, None] * ephi
        out = {}
        for name, v in (("toroidal", ep**2), ("poloidal", np.sum(epol**2, 1))):
            out[name] = [float(np.sqrt(np.sum(w * v) / np.sum(w))) / rmsB] + [
                float(np.sqrt(np.sum((w * v)[(rho > x) & (rho <= y)]) / np.sum(w[(rho > x) & (rho <= y)])))
                / rmsB for x, y in zip(edges[:-1], edges[1:])]
        return out

    B_coil = np.asarray(fields.B_ext)
    e_box = B_coil + f["B_plasma"] - desc["B"]
    # the exact toroidal plasma field from T_zeta
    cu = cur.with_u(np.asarray(u))
    T = np.asarray(cu.T(np.asarray(fr.rtz)))
    Bphi_exact = MU_0 * T[:, 1] / R
    Bphi_box = np.sum(f["B_plasma"] * ephi, 1)
    B_fix = f["B_plasma"] + (Bphi_exact - Bphi_box)[:, None] * ephi
    e_fix = B_coil + B_fix - desc["B"]
    names = ["all"] + [f"{x:g}-{y:g}" for x, y in zip(edges[:-1], edges[1:])]
    print(f"{inp}\nbox a/{a.frac} {box.shape}; collocation {fr.tgrid.shape}; J-fitted x0; "
          f"rms of each part / rms |B_DESC|")
    print(f"{'':34s}" + " ".join(f"{n:>9s}" for n in names))
    for lab, e in (("B_coil + B_box - B_DESC", e_box), ("same, B_phi,plasma = mu0 T_zeta/R", e_fix)):
        s = split(e)
        for part in ("toroidal", "poloidal"):
            print(f"{lab[:24] + ' ' + part:34s}" + " ".join(f"{v:9.2e}" for v in s[part]))
    d = np.sqrt(np.sum(w * (Bphi_exact - Bphi_box) ** 2) / np.sum(w)) / rmsB
    print(f"rms (mu0 T_zeta / R - B_phi,box) / rms B: {d:.2e}; plasma B_phi itself: "
          f"{np.sqrt(np.sum(w * Bphi_exact**2) / np.sum(w)) / rmsB:.2e}")


if __name__ == "__main__":
    main()
