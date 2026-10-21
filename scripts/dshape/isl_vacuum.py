"""The resonant error field on the tokamak without plasma response: b^rho_mn of
eps B_pert on DESC's rational surfaces and the vacuum island width
w = 4 sqrt(|b_mn| / (m |iota'|)) per unit eps, for both helicities of the 2/1 (and
its sidebands' resonances 3/1, 5/2, ...). Calibrates the amplitude scan.

run: python scripts/isl_vacuum.py [--m 2 --n -1]
"""

import argparse
import os
import sys
from fractions import Fraction

import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.join(ROOT, "nest"))

from desc import set_device  # noqa: E402

set_device(os.environ.get("NEST_DEVICE", "cpu"))

import jax  # noqa: E402

from desc_input import DESCInput  # noqa: E402
from flux_interp import TensorGeometry, TensorGrid  # noqa: E402
from separable import Separable  # noqa: E402
from perturbation import Perturbation  # noqa: E402

jax.config.update("jax_enable_x64", True)
TOK = os.path.join(ROOT, "desc_input", "tokamak_free_boundary")


def resonant_b(inp, rho0, B_pert, harmonics, n_t=128, n_z=16):
    """|b^rho_mn| per unit eps on DESC's surface rho0: b^rho = eps B_pert . grad rho /
    B_DESC . grad zeta (the linear radial excursion per radian of zeta), analysed in
    (theta*, zeta) as islands.resonant_spectrum."""
    eq = inp.eq
    th = 2 * np.pi * np.arange(n_t) / n_t
    ze = 2 * np.pi / eq.NFP * np.arange(n_z) / n_z
    g = TensorGrid(np.array([rho0]), th, ze, eq.NFP)
    geo = TensorGeometry(Separable([eq.R_basis, eq.Z_basis], [eq.R_lmn, eq.Z_lmn], eq.NFP), g)
    R, Z = np.asarray(geo.R).ravel(), np.asarray(geo.Z).ravel()
    nodes = g.nodes()
    phi = nodes[:, 2]
    X = np.column_stack([R * np.cos(phi), R * np.sin(phi), Z])
    B = B_pert(X)
    c, s = np.cos(phi), np.sin(phi)
    Bcyl = np.column_stack([B[:, 0] * c + B[:, 1] * s, -B[:, 0] * s + B[:, 1] * c, B[:, 2]])
    one, zero = jax.numpy.ones(g.shape), jax.numpy.zeros(g.shape)
    grad_rho = np.asarray(geo.gradient(one, zero, zero)).reshape(-1, 3)
    d = inp.compute_tensor(["B", "lambda", "lambda_t"], [rho0], th, ze)
    b = np.sum(Bcyl * grad_rho, 1) / (np.asarray(d["B"])[:, 1] / R)
    ths = nodes[:, 1] + d["lambda"]
    jac = 1 + d["lambda_t"]
    return {(m, n): 2 * abs(np.mean(b * np.exp(-1j * (m * ths - n * phi)) * jac)) for m, n in harmonics}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--eq", default=os.path.join(TOK, "tokamak_L32M32_fb.h5"))
    ap.add_argument("--coils", default=os.path.join(TOK, "tokamak_L32M32_field.h5"))
    ap.add_argument("--m", type=int, default=2)
    ap.add_argument("--n", type=int, nargs="+", default=[-1, 1])
    a = ap.parse_args()
    inp = DESCInput.load(a.eq, field=a.coils)
    rho = np.linspace(0, 1, 4001)
    iota, iota_r = inp.iota(rho)
    print(inp, f"\niota {iota[0]:.4f} -> {iota[-1]:.4f}")
    # resonances: iota = n' / m' (n' < 0 here), the lowest m' first
    res = {}
    for mm in range(1, 8):
        for nn in range(-mm, mm + 1):
            if nn == 0:
                continue
            q = Fraction(nn, mm)
            if q not in res and iota.min() <= float(q) <= iota.max():
                i = np.argmin(np.abs(iota - float(q)))
                res[q] = (mm, nn, rho[i], iota_r[i])
    B0 = None
    for n in a.n:
        pert = Perturbation(inp.field, a.m, n)
        print(pert)
        for q, (mm, nn, r, ir) in sorted(res.items(), key=lambda t: (t[1][0], -t[1][2])):
            if nn % n:  # only harmonics of the perturbation's toroidal mode reach the surface
                continue
            b = resonant_b(inp, r, pert, [(mm, nn)])[(mm, nn)]
            w1 = 4 * np.sqrt(b / (mm * abs(ir)))
            print(f"   iota {str(q):5s} rho {r:.3f} iota' {ir:+.3f}: |b_{mm},{nn}| {b:.3e} per A; w = {w1:.3e} sqrt(eps / A) -> w = 0.05 at eps = "
                  f"{(0.05 / w1) ** 2:.3g} A", flush=True)
        if B0 is None:
            B0 = True


if __name__ == "__main__":
    main()
