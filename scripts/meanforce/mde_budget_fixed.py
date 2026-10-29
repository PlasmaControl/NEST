"""Magnetic-differential-equation budget near the rational surfaces, on DESC's coordinates.

Exact identity (div J = 0): B.grad(sigma) = -div(J_perp), sigma = J.B/B^2. With perpendicular force
balance, J_perp = B x grad p / B^2, so B.grad(sigma) = S, S = -div(B x grad p / B^2) (the
Pfirsch-Schlueter source). Per state, on rho x theta x zeta (one period): sigma, S, B.grad(sigma) and
R = B.grad(sigma) - S (= -div(J_perp - B x grad p/B^2): nonzero where perpendicular balance fails),
the checks div J and div B (should be ~0: discretization), and the (m, n) harmonics in (theta*, zeta)
of sigma, S, B.grad sigma, R and p. Divergences: (1/sqrt g) d_i(sqrt g V^i), V^i = V.grad x^i;
d_rho by finite differences, d_theta and d_zeta spectrally.
usage: python mde_budget_fixed.py --LMN 48 32 18 --frac 40 --state name=path.npz ... --out x.npz
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
    ap.add_argument("--rho", type=float, nargs=3, default=(0.40, 0.95, 221), help="lo hi n")
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
    g = TensorGrid(rho, th, ze, eq.NFP)
    shp = g.shape
    RZ = Separable([eq.R_basis, eq.Z_basis], [eq.R_lmn, eq.Z_lmn], eq.NFP)
    geo = TensorGeometry(RZ, g)
    R, Z = np.asarray(geo.R).ravel(), np.asarray(geo.Z).ravel()
    nodes = g.nodes()
    phi = nodes[:, 2]
    one, zero = jnp.ones(shp), jnp.zeros(shp)
    e_r = np.asarray(geo.gradient(one, zero, zero)).reshape(-1, 3)
    e_t = np.asarray(geo.gradient(zero, one, zero)).reshape(-1, 3)
    e_z = np.asarray(geo.gradient(zero, zero, one)).reshape(-1, 3)
    sg = np.asarray(geo.sqrt_g).reshape(shp)
    lam = inp.compute_tensor(["lambda", "lambda_t"], rho, th, ze)
    ths = (nodes[:, 1] + np.asarray(lam["lambda"]).ravel()).reshape(shp)
    jac = (1 + np.asarray(lam["lambda_t"]).ravel()).reshape(shp)
    zeta = phi.reshape(shp)
    kt = np.fft.fftfreq(a.nt, 1.0 / a.nt)
    kz = np.fft.fftfreq(a.nz, 1.0 / a.nz) * eq.NFP
    dr = rho[1] - rho[0]

    def d_t(f):
        return np.real(np.fft.ifft(1j * kt[None, :, None] * np.fft.fft(f, axis=1), axis=1))

    def d_z(f):
        return np.real(np.fft.ifft(1j * kz[None, None, :] * np.fft.fft(f, axis=2), axis=2))

    def d_r(f):
        return np.gradient(f, dr, axis=0)

    def contra(V):
        return [np.sum(V * e, 1).reshape(shp) for e in (e_r, e_t, e_z)]

    def div(V):
        Vr, Vt, Vz = contra(V)
        return (d_r(sg * Vr) + d_t(sg * Vt) + d_z(sg * Vz)) / sg

    def bgrad(B, f):
        Br, Bt, Bz = contra(B)
        return Br * d_r(f) + Bt * d_t(f) + Bz * d_z(f)

    X = np.column_stack([R * np.cos(phi), R * np.sin(phi), Z])
    Bx = np.concatenate([np.asarray(inp.B_ext(X[i : i + 200_000])) for i in range(0, len(X), 200_000)])
    c, s_ = np.cos(phi), np.sin(phi)
    Bext = np.column_stack([Bx[:, 0] * c + Bx[:, 1] * s_, -Bx[:, 0] * s_ + Bx[:, 1] * c, Bx[:, 2]])
    itp = CylindricalInterpolator(fields.cyl_grid, np.column_stack([R, phi, Z]), dtype=jnp.float64, order=8)
    iota, iota_r = inp.iota(rho)
    sgn = int(np.sign(np.mean(iota)))
    HARM = [(5, 3 * sgn), (6, 3 * sgn), (11, 6 * sgn), (0, 0)]
    out = dict(rho=rho, iota=iota, harm=np.array(HARM))

    def budget(name, J, B, p, gp):
        B2 = np.sum(B * B, 1)
        sig = (np.sum(J * B, 1) / B2).reshape(shp)
        V = np.cross(B, gp) / B2[:, None]
        S = -div(V)
        Bs = bgrad(B, sig)
        Rm = Bs - S
        dJ, dB = div(J), div(B)
        nJ = np.sqrt(np.mean(np.linalg.norm(J, axis=1) ** 2)) / np.sqrt(np.mean(R**2)) * 0 + 1.0
        # scale for the checks: |J| / (a/L) ~ typical size of one divergence term
        scaleJ = np.sqrt(np.mean(np.sum(J * J, 1))) / inp.a
        scaleB = np.sqrt(np.mean(B2)) / inp.a
        Bmod = np.sqrt(B2).reshape(shp)
        Bz_av = np.mean(contra(B)[2], axis=(1, 2))
        for k, f in (("sigma", sig), ("S", S), ("Bgs", Bs), ("R", Rm), ("p", p.reshape(shp))):
            out[f"{name}_{k}"] = harmonics(f, ths, jac, zeta, HARM)
        out[f"{name}_rmsS"] = np.sqrt(np.mean(S**2, axis=(1, 2)))
        out[f"{name}_rmsR"] = np.sqrt(np.mean(Rm**2, axis=(1, 2)))
        out[f"{name}_Bzeta"] = Bz_av
        print(f"{name}: rms div J / (|J|/a) {np.sqrt(np.mean(dJ**2)) / scaleJ:.2e}, rms div B / (|B|/a) "
              f"{np.sqrt(np.mean(dB**2)) / scaleB:.2e}; rms |R| / rms |S| over the grid "
              f"{np.sqrt(np.mean(Rm**2)) / np.sqrt(np.mean(S**2)):.3f}", flush=True)

    if not a.no_desc:
        dd = inp.compute_tensor(["J", "B", "p", "grad(p)"], rho, th, ze)
        budget("DESC", np.asarray(dd["J"]).reshape(-1, 3), np.asarray(dd["B"]).reshape(-1, 3),
               np.asarray(dd["p"]).ravel(), np.asarray(dd["grad(p)"]).reshape(-1, 3))
    states = [("start", np.asarray(cur.u), np.asarray(pre.c))]
    for spec in a.state:
        name, path = spec.split("=", 1)
        d = np.load(path)
        states.append((name, d["u"], d["c_p"]))
    for name, u, c_p in states:
        J = np.asarray(cur.with_u(jnp.asarray(u)).evaluate(nodes))
        pg = np.asarray(pre.with_c(jnp.asarray(c_p)).evaluate(SimpleNamespace(nodes=nodes)))
        Bp, Bz = fields.cyl_field_modes(jnp.asarray(u))
        B = np.asarray(itp(Bp, Bz)) + Bext
        budget(name, J, B, pg[:, 0], pg[:, 1:4])
    os.makedirs(os.path.dirname(os.path.abspath(a.out)), exist_ok=True)
    np.savez(a.out, **out)
    print(f"saved {a.out}; total {time.time() - t0:.0f} s")


if __name__ == "__main__":
    main()
