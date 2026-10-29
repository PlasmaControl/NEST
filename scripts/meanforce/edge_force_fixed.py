"""Force decomposition and edge B.n of fixed-boundary mean-force states, on DESC's coordinates.

On rho x theta x zeta (one field period; volume weights sqrt(g) drho dtheta dzeta), per state:
  F = J x B - grad p (p the state's pressure), F_par = F . b = -b . grad p (J x B has no
  component along B: the part of F that no J can cancel for this p).
  M_grid = sum dV |F| / (V norm) (the solver's M, as a check), the same for |F_par|, and the
  radial distribution of both (per rho shell, normalized so the shells sum to M_grid).
  On the LCFS rho = 1: B.n / |B| (n = grad rho / |grad rho|), area-weighted rms and max, for
  the total field and for the coils alone.
usage: python edge_force_fixed.py --LMN 32 32 24 --frac 40 --state name=path.npz ... --out x.npz
"""
import argparse
import os
import sys
import time
from types import SimpleNamespace

import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.join(ROOT, "nest"))
sys.path.insert(0, os.path.join(ROOT, "scripts"))
VFB = os.path.join(ROOT, "desc_input", "vmec_free_boundary")
sys.path.insert(0, VFB)

from desc import set_device  # noqa: E402

set_device(os.environ.get("NEST_DEVICE", "cpu"))

import jax  # noqa: E402
import jax.numpy as jnp  # noqa: E402

jax.config.update("jax_enable_x64", True)

import fit_cache  # noqa: E402
from boundary_potential import BoundaryPotential  # noqa: E402
from cylindrical_grid import CylindricalInterpolator  # noqa: E402
from desc_input import DESCInput  # noqa: E402
from flux_interp import TensorGeometry, TensorGrid  # noqa: E402
from force import ForceResidual  # noqa: E402
from free_boundary import _chunked, reference  # noqa: E402
from objective import Fields  # noqa: E402
from optimize_fb import annulus  # noqa: E402
from pressure import Pressure  # noqa: E402
from separable import Separable  # noqa: E402

D = os.path.join(VFB, "NCSX")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--eq", default=os.path.join(D, "NCSX_fb_L16M16N12.h5"))
    ap.add_argument("--coils", default=os.path.join(D, "NCSX_coils.h5"))
    ap.add_argument("--factors", type=float, nargs=3, default=(1, 1, 1))
    ap.add_argument("--LMN", type=int, nargs=3, default=None)
    ap.add_argument("--colloc", type=float, default=1.5)
    ap.add_argument("--frac", type=float, default=40)
    ap.add_argument("--pad", type=float, default=0.02)
    ap.add_argument("--state", action="append", default=[], help="name=path.npz (u, c_p); repeatable")
    ap.add_argument("--rho", type=float, nargs=3, default=(0.01, 0.995, 198), help="lo hi n")
    ap.add_argument("--nt", type=int, default=96)
    ap.add_argument("--nz", type=int, default=32, help="zeta nodes per field period")
    ap.add_argument("--no-desc", action="store_true")
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    t0 = time.time()
    inp = DESCInput.load(a.eq, field=a.coils)
    eq = inp.eq
    kw = {"L": a.LMN[0], "M": a.LMN[1], "N": a.LMN[2]} if a.LMN else {"factors": tuple(a.factors)}
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
    print(f"{inp}\nbasis {b.L}/{b.M}/{b.N}, annulus {box}; setup {time.time() - t0:.0f} s", flush=True)


    rho = np.linspace(a.rho[0], a.rho[1], int(a.rho[2]))
    th = 2 * np.pi * np.arange(a.nt) / a.nt
    ze = 2 * np.pi / eq.NFP * np.arange(a.nz) / a.nz
    RZ = Separable([eq.R_basis, eq.Z_basis], [eq.R_lmn, eq.Z_lmn], eq.NFP)

    def surf_grid(rr):
        g = TensorGrid(rr, th, ze, eq.NFP)
        geo = TensorGeometry(RZ, g)
        nodes = g.nodes()
        R, Z, phi = np.asarray(geo.R).ravel(), np.asarray(geo.Z).ravel(), nodes[:, 2]
        one, zero = jnp.ones(g.shape), jnp.zeros(g.shape)
        grho = np.asarray(geo.gradient(one, zero, zero)).reshape(-1, 3)
        X = np.column_stack([R * np.cos(phi), R * np.sin(phi), Z])
        Bx = np.concatenate([np.asarray(inp.B_ext(X[i : i + 200_000])) for i in range(0, len(X), 200_000)])
        c, s_ = np.cos(phi), np.sin(phi)
        Bext = np.column_stack([Bx[:, 0] * c + Bx[:, 1] * s_, -Bx[:, 0] * s_ + Bx[:, 1] * c, Bx[:, 2]])
        itp = CylindricalInterpolator(fields.cyl_grid, np.column_stack([R, phi, Z]), dtype=jnp.float64, order=8)
        return g, geo, nodes, grho, Bext, itp, np.abs(np.asarray(geo.sqrt_g)).ravel()

    g, geo, nodes, grho, Bext, itp, sg = surf_grid(rho)
    dV = sg * (rho[1] - rho[0]) * (th[1] - th[0]) * (ze[1] - ze[0])
    V = dV.sum()
    shp = g.shape
    ge, geoe, nodes_e, grho_e, Bext_e, itp_e, sg_e = surf_grid(np.array([1.0]))
    dA = sg_e * np.linalg.norm(grho_e, axis=1)  # area element sqrt(g) |grad rho| dtheta dzeta
    nE = grho_e / np.linalg.norm(grho_e, axis=1)[:, None]

    def bn(B):
        x = np.sum(B * nE, 1) / np.linalg.norm(B, axis=1)
        return float(np.sqrt(np.sum(dA * x**2) / dA.sum())), float(np.abs(x).max())

    out = dict(rho=rho, coils_Bn=np.array(bn(Bext_e)))
    print(f"LCFS B.n/|B| of the coils alone: rms {out['coils_Bn'][0]:.3e}, max {out['coils_Bn'][1]:.3e}; "
          f"grid volume {V * eq.NFP:.4f} m^3", flush=True)
    states = [("start", np.asarray(cur.u), np.asarray(pre.c))]
    for spec in a.state:
        name, path = spec.split("=", 1)
        d = np.load(path)
        states.append((name, d["u"], d["c_p"]))
    M_desc = ref["F_desc_vol"] / ref["norm"]
    for name, u, c_p in states:
        cu = cur.with_u(jnp.asarray(u))
        J = np.asarray(cu.evaluate(nodes))
        pg = np.asarray(pre.with_c(jnp.asarray(c_p)).evaluate(SimpleNamespace(nodes=nodes)))
        Bp, Bz = fields.cyl_field_modes(jnp.asarray(u))
        B = np.asarray(itp(Bp, Bz)) + Bext
        F = np.cross(J, B) - pg[:, 1:4]
        b = B / np.linalg.norm(B, axis=1)[:, None]
        Fpar = np.abs(np.sum(F * b, 1))
        aF = np.linalg.norm(F, axis=1)
        M = float(np.sum(dV * aF) / (V * ref["norm"]))
        Mp = float(np.sum(dV * Fpar) / (V * ref["norm"]))
        shellF = (dV * aF).reshape(shp).sum((1, 2)) / (V * ref["norm"])
        shellP = (dV * Fpar).reshape(shp).sum((1, 2)) / (V * ref["norm"])
        band = np.abs(rho - 0.802) < 0.05
        Be = np.asarray(itp_e(Bp, Bz)) + Bext_e
        out.update({f"{name}_M": M, f"{name}_Mpar": Mp, f"{name}_shellF": shellF, f"{name}_shellP": shellP,
                    f"{name}_Bn": np.array(bn(Be))})
        print(f"{name}: M_grid/M_desc {M / M_desc:.4f}, |F_par| part {Mp / M_desc:.4f} ({Mp / M:.0%} of M); "
              f"3/5 band rho 0.75-0.85: {shellF[band].sum() / M:.1%} of M, {shellP[band].sum() / Mp:.1%} of M_par "
              f"(volume share {(dV.reshape(shp).sum((1, 2))[band].sum() / V):.1%}); LCFS B.n/|B| rms "
              f"{out[name + '_Bn'][0]:.3e}, max {out[name + '_Bn'][1]:.3e}", flush=True)
    os.makedirs(os.path.dirname(os.path.abspath(a.out)), exist_ok=True)
    np.savez(a.out, **out)
    print(f"saved {a.out}; total {time.time() - t0:.0f} s")


if __name__ == "__main__":
    main()
