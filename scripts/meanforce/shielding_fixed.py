"""Shielding diagnostics of fixed-boundary mean-force states on DESC's flux surfaces.

For each state (the start J_fit and mf_fixed npz states), on rho x theta x zeta (one field
period) of DESC's coordinates, with theta* = theta + lambda (DESC's straight-field-line angle):

  sigma = J . B / |B|^2   (J_par = sigma B): radial profiles of its (m, n) harmonics in
          (theta*, zeta). Ideal MHD near iota = n/m: the Pfirsch-Schlueter part goes as
          1 / (iota - n/m) (sign change across rho_res), a shielding sheet is a
          single-signed peak.
  b^rho = B . grad rho / B . grad zeta: its (m, n) harmonics for the total field and for
          B_ext alone (the coils' resonant field); at rho_res the pendulum full width
          w = 4 sqrt(|b_mn| / (m |iota'|)).
  p:      the surface average <p>(rho) (flattening at a chain) and the surface rms of
          B . grad p / (|B| |grad p|).
DESC's own J, B, p give the reference (its b^rho ~ 0 by construction).

B_plasma is the run's annulus field (``Fields.cyl_field_modes``) evaluated at the points by
``CylindricalInterpolator`` (no table). Harmonic convention: f = sum a cos(m theta* - n zeta)
+ b sin(m theta* - n zeta), n a multiple of NFP; resonance at iota = n / m.

usage: python scripts/meanforce/shielding_fixed.py --LMN 32 32 24 --frac 40 \\
          --state a0.05=results/meanforce_fixed/states/fxD05b_a0.05.npz --out out/shield.npz
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
HARM0 = [(5, 3), (6, 3), (7, 3), (8, 3), (11, 6), (0, 0), (1, 0), (2, 0), (1, 3), (4, 3), (0, 3)]
N_RES = 5  # the first five are the resonant ones


def harmonics(f, ths, jac, zeta, harm):
    """(a_mn, b_mn) per rho of f (n_rho, n_t, n_z): f = sum a cos(m t* - n z) + b sin(...)."""
    out = np.empty((f.shape[0], len(harm), 2))
    for k, (m, n) in enumerate(harm):
        c = np.mean(f * np.exp(-1j * (m * ths - n * zeta)) * jac, axis=(1, 2))
        s = 1.0 if (m, n) == (0, 0) else 2.0
        out[:, k, 0], out[:, k, 1] = s * c.real, -s * c.imag
    return out


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
    ap.add_argument("--rho", type=float, nargs=3, default=(0.05, 0.97, 185), help="lo hi n")
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

    # the surface grid and its geometry
    rho = np.linspace(a.rho[0], a.rho[1], int(a.rho[2]))
    th = 2 * np.pi * np.arange(a.nt) / a.nt
    ze = 2 * np.pi / eq.NFP * np.arange(a.nz) / a.nz
    g = TensorGrid(rho, th, ze, eq.NFP)
    shp = g.shape
    RZ = Separable([eq.R_basis, eq.Z_basis], [eq.R_lmn, eq.Z_lmn], eq.NFP)
    geo = TensorGeometry(RZ, g)
    R, Z = np.asarray(geo.R).ravel(), np.asarray(geo.Z).ravel()
    nodes = g.nodes()
    phi = nodes[:, 2]
    one, zero = jnp.ones(shp), jnp.zeros(shp)
    grad_rho = np.asarray(geo.gradient(one, zero, zero)).reshape(-1, 3)
    lam = inp.compute_tensor(["lambda", "lambda_t"], rho, th, ze)
    ths = (nodes[:, 1] + np.asarray(lam["lambda"]).ravel()).reshape(shp)
    jac = (1 + np.asarray(lam["lambda_t"]).ravel()).reshape(shp)
    zeta = phi.reshape(shp)
    iota, iota_r = inp.iota(rho)
    sg = int(np.sign(np.mean(iota)))  # DESC's sign of iota: resonance at n = m iota
    HARM = [(m, sg * n) for m, n in HARM0]
    res_ = []
    for m, n in HARM[:N_RES]:
        s = np.sign(iota - n / m)
        for i in np.flatnonzero(np.diff(s)):
            r0 = rho[i] + (n / m - iota[i]) * (rho[i + 1] - rho[i]) / (iota[i + 1] - iota[i])
            res_.append((m, n, float(r0), float(np.interp(r0, rho, iota_r))))
    print("resonances (m, n, rho, iota'): " + ", ".join(f"{m}/{n} at {r:.3f} ({ir:+.3f})" for m, n, r, ir in res_))

    # B_ext (cylindrical components) at the points
    X = np.column_stack([R * np.cos(phi), R * np.sin(phi), Z])
    Bx = np.concatenate([np.asarray(inp.B_ext(X[i : i + 200_000])) for i in range(0, len(X), 200_000)])
    c, s = np.cos(phi), np.sin(phi)
    Bext = np.column_stack([Bx[:, 0] * c + Bx[:, 1] * s, -Bx[:, 0] * s + Bx[:, 1] * c, Bx[:, 2]])
    itp = CylindricalInterpolator(fields.cyl_grid, np.column_stack([R, phi, Z]), dtype=jnp.float64, order=8)

    def brho(B):
        return (np.sum(B * grad_rho, 1) / (B[:, 1] / R)).reshape(shp)

    def p_diag(p, gp, B):
        Bgp = np.sum(B * gp, 1) / (np.linalg.norm(B, axis=1) * np.linalg.norm(gp, axis=1) + 1e-300)
        return np.mean(p.reshape(shp), axis=(1, 2)), np.sqrt(np.mean(Bgp.reshape(shp) ** 2, axis=(1, 2)))

    out = dict(rho=rho, iota=iota, iota_r=iota_r, harm=np.array(HARM), res=np.array(res_),
               brho_ext=harmonics(brho(Bext), ths, jac, zeta, HARM))
    states = [("start", np.asarray(cur.u), np.asarray(pre.c))]
    for spec in a.state:
        name, path = spec.split("=", 1)
        d = np.load(path)
        states.append((name, d["u"], d["c_p"]))
    if not a.no_desc:
        t1 = time.time()
        dd = inp.compute_tensor(["J", "B", "p", "grad(p)"], rho, th, ze)
        Jd, Bd = np.asarray(dd["J"]).reshape(-1, 3), np.asarray(dd["B"]).reshape(-1, 3)
        sig = (np.sum(Jd * Bd, 1) / np.sum(Bd * Bd, 1)).reshape(shp)
        out["DESC_sigma"] = harmonics(sig, ths, jac, zeta, HARM)
        out["DESC_brho"] = harmonics(brho(Bd), ths, jac, zeta, HARM)
        out["DESC_p"], out["DESC_Bgradp"] = p_diag(np.asarray(dd["p"]).ravel(),
                                                   np.asarray(dd["grad(p)"]).reshape(-1, 3), Bd)
        print(f"DESC reference in {time.time() - t1:.0f} s", flush=True)
    for name, u, c_p in states:
        t1 = time.time()
        J = np.asarray(cur.with_u(jnp.asarray(u)).evaluate(nodes))
        pg = np.asarray(pre.with_c(jnp.asarray(c_p)).evaluate(SimpleNamespace(nodes=nodes)))
        Bp, Bz = fields.cyl_field_modes(jnp.asarray(u))
        B = np.asarray(itp(Bp, Bz)) + Bext
        sig = (np.sum(J * B, 1) / np.sum(B * B, 1)).reshape(shp)
        out[f"{name}_sigma"] = harmonics(sig, ths, jac, zeta, HARM)
        out[f"{name}_brho"] = harmonics(brho(B), ths, jac, zeta, HARM)
        out[f"{name}_p"], out[f"{name}_Bgradp"] = p_diag(pg[:, 0], pg[:, 1:4], B)
        line = []
        for m, n, r0, ir in res_:
            k = HARM.index((m, n))
            bb = out[f"{name}_brho"][:, k]
            amp = float(np.hypot(np.interp(r0, rho, bb[:, 0]), np.interp(r0, rho, bb[:, 1])))
            be = out["brho_ext"][:, k]
            amp_e = float(np.hypot(np.interp(r0, rho, be[:, 0]), np.interp(r0, rho, be[:, 1])))
            w = 4 * np.sqrt(amp / (m * abs(ir)))
            line.append(f"{m}/{n}@{r0:.2f}: |b| {amp:.2e} (coils {amp_e:.2e}, ratio {amp / amp_e:.3f}) w {w:.4f}")
        print(f"{name}: {time.time() - t1:.0f} s\n   " + "\n   ".join(line), flush=True)
    os.makedirs(os.path.dirname(os.path.abspath(a.out)), exist_ok=True)
    np.savez(a.out, **out)
    print(f"saved {a.out}; total {time.time() - t0:.0f} s")


if __name__ == "__main__":
    main()
