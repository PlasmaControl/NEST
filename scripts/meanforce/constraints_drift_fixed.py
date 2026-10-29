"""How far have fixed-boundary states drifted in the quantities that --pin-current / --pin-flux
hold fixed? Per state (J_fit first): the toroidally averaged toroidal flux Psi_bar (direct and
affine, against J_fit's and DESC's Psi), and the enclosed toroidal current I(rho, zeta) =
2 pi <T_theta>_theta on DESC's surfaces (its zeta-average profile and the change from J_fit,
as fractions of the net current I), plus the m = 0 coefficients' relative change.
usage: python constraints_drift_fixed.py --LMN 48 32 18 --frac 40 --state name=path.npz ... --out x.npz
"""
import argparse
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

jax.config.update("jax_enable_x64", True)
import fit_cache  # noqa: E402
import meanforce_fixed as mff  # noqa: E402
from boundary_potential import BoundaryPotential  # noqa: E402
from desc_input import DESCInput  # noqa: E402
from force import ForceResidual  # noqa: E402
from free_boundary import _chunked, reference  # noqa: E402
from objective import Fields  # noqa: E402
from optimize_fb import annulus  # noqa: E402
from pressure import Pressure  # noqa: E402

D = os.path.join(ROOT, "desc_input", "vmec_free_boundary", "NCSX")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--eq", default=os.path.join(D, "NCSX_fb_L16M16N12.h5"))
    ap.add_argument("--coils", default=os.path.join(D, "NCSX_coils.h5"))
    ap.add_argument("--LMN", type=int, nargs=3, required=True)
    ap.add_argument("--colloc", type=float, default=1.5)
    ap.add_argument("--frac", type=float, default=40)
    ap.add_argument("--pad", type=float, default=0.02)
    ap.add_argument("--state", action="append", default=[])
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    t0 = time.time()
    inp = DESCInput.load(a.eq, field=a.coils)
    eq = inp.eq
    kw = {"L": a.LMN[0], "M": a.LMN[1], "N": a.LMN[2]}
    cur = fit_cache.fit(BoundaryPotential, inp, **kw)
    pre = fit_cache.fit(Pressure, inp, edge_order=2, **kw)
    b = cur.basis
    res = tuple(int(round(a.colloc * (u + v))) for u, v in ((b.L, eq.L), (b.M, eq.M), (b.N, eq.N)))
    ref = reference(inp, res, current=cur)
    box = annulus(inp, inp.a / a.frac, a.pad, None)
    box.locate_plasma_planes(inp, sym=bool(eq.sym), verbose=False)
    fr = ForceResidual(inp, box, resolution=res)
    fr.norm, fr.F_desc_vol = ref["norm"], ref["F_desc_vol"]
    if "B_ext" not in vars(fr):
        fr.B_ext = _chunked(inp.B_ext, fr.x)
    fields = Fields(fr, cur, pre, dtype=jnp.float64, cyl_stencil=True, mode_chunk=16, cyl_fused=True,
                    cyl_rblock=32, cyl_order=8)
    print(f"basis {b.L}/{b.M}/{b.N}; setup {time.time() - t0:.0f} s", flush=True)
    u0 = jnp.asarray(cur.u)
    LC = mff.LinearConstraints(cur, fields, fr, u0, pin_current=True, pin_flux=True)
    rho = np.linspace(0.05, 0.95, 19)
    zeta = np.linspace(0, np.pi / eq.NFP, 5)
    I0 = mff.enclosed_current(cur, u0, rho, zeta)
    states = {"J_fit": np.asarray(u0)}
    for spec in a.state:
        name, path = spec.split("=", 1)
        states[name] = np.load(path)["u"]
    out = {"rho": rho, "zeta": zeta, "I": cur.I, "Psi_desc": float(eq.Psi), "Psi_ref": LC.psi_ref}
    print(f"net current I {cur.I:.6e} A; Psi_bar(J_fit) {LC.psi_ref:.6e} Wb vs DESC Psi {float(eq.Psi):.6e} Wb "
          f"(ratio {LC.psi_ref / float(eq.Psi):.5f})")
    for name, u in states.items():
        psi_d = float(LC.psi(jnp.asarray(u)))
        Iu = mff.enclosed_current(cur, u, rho, zeta)
        dI = (Iu - I0) / abs(cur.I)
        v = LC.violation(u)
        out[f"{name}_I"], out[f"{name}_psi"] = Iu, psi_d
        prof = ", ".join(f"{r:.2f}:{x:+.4f}" for r, x in zip(rho[::3], dI.mean(axis=1)[::3]))
        print(f"{name}: dPsi/Psi {(psi_d - LC.psi_ref) / LC.psi_ref:+.4e} (affine {v['dPsi/Psi']:+.4e}); "
              f"enclosed current change / I: max {np.abs(dI).max():.4f}, zeta-avg profile [{prof}]; "
              f"m=0 coefficients max|du|/max|u_ref| {v['pin max|du|/max|u_ref|']:.3e}", flush=True)
    np.savez(a.out, **out)
    print(f"saved {a.out}; total {time.time() - t0:.0f} s")


if __name__ == "__main__":
    main()
