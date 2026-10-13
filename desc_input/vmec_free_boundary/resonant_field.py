"""Resonant radial field of NEST solutions on DESC's flux surfaces, as a function of rho.

On each surface rho of the reference DESC equilibrium (straight-field-line angles
theta* = theta + lambda and zeta = phi), the radial excursion of a field line per
radian of zeta,

    b^rho = B . grad rho / B . grad zeta,

is evaluated for B = B_ext + B_plasma of each field and Fourier analysed:

    c_mn(rho) = (1 / (2 pi)^2) int int b^rho e^{-i (m theta* - n zeta)} dtheta* dzeta,

for m = 0..M_MAX and n = k NFP, |k| <= K_MAX. A field line on the surface iota = n/m
sees the (m, n) harmonic as a constant drive, so 2 |c_mn| there sets the island
width (``resonant_widths.py``). Away from the resonance the harmonics are
non-resonant distortions of DESC's surfaces. DESC's own field (b^rho = 0
identically) checks the machinery, and the coils alone (B_plasma = 0) are a reference
for the vacuum forcing.

The fields are the start (the fits of DESC's J and p) and each optimized x
(``optimize_qa.py`` npz). B_ext is evaluated once on all surface points, B_plasma
once per field on the box. Output: figures/<out>.npz with c[<run>] of shape
(n_rho, M_MAX + 1, 2 K_MAX + 1), the rms of b^rho per surface, and DESC's iota.

run (GPU): sbatch --mem=128G run_della.sbatch desc_input/vmec_free_boundary/resonant_field.py \
    --eq desc_input/vmec_free_boundary/NCSX/NCSX_L16M16N12.h5 \
    --coils desc_input/vmec_free_boundary/NCSX/NCSX_coils.h5 \
    --x wJ0=results/ncsx/opt_L16M16N12.npz wJ1e-3=results/ncsx/opt_L16M16N12_wJ1e-3.npz \
    --out ncsx_resonant_field
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

import fit_cache  # noqa: E402
from boundary_potential import BoundaryPotential  # noqa: E402
from desc_input import DESCInput  # noqa: E402
from force import ForceResidual  # noqa: E402
from fourier import fft_J  # noqa: E402
from grid import CartesianGrid  # noqa: E402
from magnetic_field import interpolate  # noqa: E402
from objective import Fields  # noqa: E402
from pressure import Pressure  # noqa: E402

jax.config.update("jax_enable_x64", True)
M_MAX, K_MAX = 12, 4


def surfaces(inp, rho, n_t, n_z, rho_chunk=4):
    """Points, grad rho, B_DESC (cylindrical), theta*, d theta*/d theta on rho x theta x
    zeta (one field period), each of shape (n_rho, n_t, n_z, ...)."""
    th = 2 * np.pi * np.arange(n_t) / n_t
    ze = 2 * np.pi / inp.NFP * np.arange(n_z) / n_z
    d = inp.compute_tensor(["R", "Z", "phi", "e^rho", "B", "lambda", "lambda_t"],
                           rho, th, ze, rho_chunk=rho_chunk)
    sh = (len(rho), n_t, n_z)
    out = {k: v.reshape(sh + v.shape[1:]) for k, v in d.items()}
    out["theta*"] = th[None, :, None] + out["lambda"]
    out["zeta"] = np.broadcast_to(ze[None, None, :], sh)
    return out


def b_rho(Bcyl, s):
    """B . grad rho / B . grad zeta (zeta = phi: B . grad zeta = B_phi / R)."""
    return np.sum(Bcyl * s["e^rho"], -1) / (Bcyl[..., 1] / s["R"])


def spectrum(b, s, NFP):
    """c_mn per surface, m = 0..M_MAX, n = k NFP (k = -K_MAX..K_MAX): the average of
    b e^{-i (m theta* - n zeta)} over theta* and zeta (d theta* = (1 + lambda_t) d theta)."""
    w = b * (1 + s["lambda_t"])
    m = np.arange(M_MAX + 1)
    n = NFP * np.arange(-K_MAX, K_MAX + 1)
    Em = np.exp(-1j * m[:, None, None, None] * s["theta*"][None])  # (m, rho, t, z)
    En = np.exp(1j * n[:, None] * s["zeta"][0, 0][None, :])  # (n, z)
    return np.einsum("rtz,mrtz,nz->rmn", w, Em, En) / (b.shape[1] * b.shape[2])


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--eq", required=True)
    ap.add_argument("--coils", required=True)
    ap.add_argument("--x", nargs="*", default=[], help="label=optimize_qa.py npz, ...")
    ap.add_argument("--frac", type=int, default=20)
    ap.add_argument("--kernel", default="truncated")
    ap.add_argument("--factors", type=float, nargs=3, default=(3, 3, 2.5))
    ap.add_argument("--rho", type=float, nargs=3, default=(0.20, 0.99, 80),
                    help="rho_min rho_max n_rho")
    ap.add_argument("--nt", type=int, default=128)
    ap.add_argument("--nz", type=int, default=48, help="zeta points per field period")
    ap.add_argument("--chunk", type=int, default=4, help="surfaces per B_ext call")
    ap.add_argument("--out", default="resonant_field")
    a = ap.parse_args()
    t0 = time.time()
    inp = DESCInput.load(a.eq, field=a.coils)
    eq = inp.eq
    cur = fit_cache.fit(BoundaryPotential, inp, factors=tuple(a.factors))
    pre = fit_cache.fit(Pressure, inp, factors=tuple(a.factors), edge_order=2)
    box = CartesianGrid.periodic_box(inp, inp.a / a.frac, odd=bool(eq.sym)).locate_plasma(inp)
    fields = Fields(ForceResidual(inp, box), cur, pre, kernel=a.kernel)
    us = {"start": cur.u}
    for item in a.x:
        label, path = item.split("=", 1)
        us[label] = np.load(path)["x"][: cur.n_free]
    print(f"{inp}\nbox h = a/{a.frac} {box.shape}; fields: {', '.join(us)}", flush=True)

    rho = np.linspace(a.rho[0], a.rho[1], int(a.rho[2]))
    s = surfaces(inp, rho, a.nt, a.nz)
    phi = s["phi"]
    X = np.stack([s["R"] * np.cos(phi), s["R"] * np.sin(phi), s["Z"]], -1).reshape(-1, 3)
    c, sn = np.cos(phi).ravel(), np.sin(phi).ravel()

    def to_cyl(B):
        return np.stack([B[:, 0] * c + B[:, 1] * sn, -B[:, 0] * sn + B[:, 1] * c, B[:, 2]],
                        -1).reshape(phi.shape + (3,))

    t1 = time.time()
    per = a.nt * a.nz * a.chunk
    B_ext = np.concatenate([inp.B_ext(X[i : i + per]) for i in range(0, len(X), per)])
    print(f"B_ext at {len(X)} points: {time.time() - t1:.0f} s", flush=True)

    spec, rms = {}, {}
    b = b_rho(s["B"], s)
    spec["DESC"], rms["DESC"] = spectrum(b, s, eq.NFP), np.sqrt(np.mean(b**2, (1, 2)))
    b = b_rho(to_cyl(B_ext), s)  # the coils alone (b^rho is linear in B only at fixed
    # B . grad zeta, so this is the coil field's own excursion, a reference)
    spec["coils"], rms["coils"] = spectrum(b, s, eq.NFP), np.sqrt(np.mean(b**2, (1, 2)))
    for name, u in us.items():
        t1 = time.time()
        B_box = fields._solve(fft_J(fields.J_box(jnp.asarray(u))), fields.axes, fields.K_hat)
        B_pl = np.concatenate([np.asarray(interpolate(np.asarray(B_box), fields.axes,
                                                      X[i : i + per]))
                               for i in range(0, len(X), per)])
        b = b_rho(to_cyl(B_ext + B_pl), s)
        spec[name], rms[name] = spectrum(b, s, eq.NFP), np.sqrt(np.mean(b**2, (1, 2)))
        print(f"{name}: {time.time() - t1:.0f} s; rms b^rho {rms[name].min():.2e} - "
              f"{rms[name].max():.2e}", flush=True)
    del B_box

    iota, iota_r = inp.iota(rho)
    print(f"DESC check: max |c_mn| {np.abs(spec['DESC']).max():.1e}")
    out = os.path.join(ROOT, "figures", a.out + ".npz")
    os.makedirs(os.path.dirname(out), exist_ok=True)
    np.savez(out, rho=rho, iota=iota, iota_r=iota_r, NFP=eq.NFP, m=np.arange(M_MAX + 1),
             n=eq.NFP * np.arange(-K_MAX, K_MAX + 1), runs=np.array(list(spec)),
             **{f"c_{k}": v for k, v in spec.items()},
             **{f"rms_{k}": v for k, v in rms.items()})
    print(f"saved {out}; total {time.time() - t0:.0f} s")


if __name__ == "__main__":
    main()
