"""Step 2 driver: J on a Cartesian grid with spacing h = a / frac, with checks.

usage: python run_step2.py [EQ.h5] [--frac 30] [--out G2.npz]

Checks against DESC (flux-coordinate quadrature): plasma volume from the mask,
int |J| dV, and the current through the half-plane phi = 0 (y = 0, x > 0).
"""

import argparse
import os
import time

import numpy as np
from desc import set_device

# DESC uses the CPU unless set_device('gpu') runs before any other DESC import
set_device(os.environ.get("NEST_DEVICE", "cpu"))

import desc.io  # noqa: E402
from desc.grid import LinearGrid, QuadratureGrid  # noqa: E402

from cartesian_grid import cartesian_grid, current_on_cartesian  # noqa: E402
from current_fz import CurrentFZ  # noqa: E402

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def cross_section_current(eq, zeta=0.0, L=64, M=256):
    """Current through the zeta = const cross-section, int J^zeta sqrt(g) drho dtheta.

    Gauss-Legendre in rho. Equals I(rho=1) only if J.grad(rho) = 0 on the
    boundary; otherwise it varies with zeta.
    """
    x, w = np.polynomial.legendre.leggauss(L)
    g = LinearGrid(rho=(x + 1) / 2, theta=M, zeta=np.array([zeta]), NFP=1, sym=False)
    d = eq.compute(["J^zeta", "sqrt(g)"], grid=g)
    f = np.asarray(g.meshgrid_reshape(np.asarray(d["J^zeta"] * d["sqrt(g)"]), "rtz"))
    return np.sum(f[:, :, 0].mean(axis=1) * 2 * np.pi * w / 2)


def volume_integral(eq, name="|J|", factor=2):
    """int f dV over the plasma for the DESC quantity ``name`` (default |J|)."""
    g = QuadratureGrid(L=factor * eq.L, M=factor * eq.M, N=factor * eq.N, NFP=eq.NFP)
    d = eq.compute([name, "sqrt(g)"], grid=g)
    return np.sum(
        np.asarray(d[name]) * np.abs(np.asarray(d["sqrt(g)"])) * np.asarray(g.weights)
    )


p = argparse.ArgumentParser()
p.add_argument("eq", nargs="?",
               default=os.path.join(ROOT, "desc_input", "precise_QA_free_boundary_L16N14.h5"))
p.add_argument("--frac", type=float, default=30, help="h = a / frac")
p.add_argument("--out", default=None, help="save J, mask and axes (.npz)")
p.add_argument("--newton-chunk", type=int, default=200000,
               help="points per inverse-map call (each call recompiles)")
p.add_argument("--eval-chunk", type=int, default=20000,
               help="points per J series evaluation")
args = p.parse_args()

t0 = time.time()
log = lambda msg: print(f"[{time.time() - t0:7.1f}s] {msg}", flush=True)

eq = desc.io.load(args.eq)
eq = eq[-1] if hasattr(eq, "__len__") else eq
a = float(eq.compute("a")["a"])
h = a / args.frac
log(f"eq NFP={eq.NFP} (L,M,N)=({eq.L},{eq.M},{eq.N}), a = {a:.4f} m, h = a/{args.frac:g}"
    f" = {h * 1e3:.2f} mm")

current = CurrentFZ.fit(eq)
log(f"J series fitted: {current.basis.num_modes} modes")

axes = cartesian_grid(eq, h)
n = [len(ax) for ax in axes]
log(f"G3 (periodic box): {n[0]} x {n[1]} x {n[2]} = {np.prod(n):,} points")
d = current_on_cartesian(eq, current, axes, newton_chunk=args.newton_chunk,
                         eval_chunk=args.eval_chunk, verbose=True)
mask, J, dV = d["mask"], d["J"], np.prod(d["dx"])
log(f"J on G3 done: {mask.sum():,} points inside the plasma")

V, V_desc = mask.sum() * dV, float(eq.compute("V")["V"])
log(f"volume:       {V:.5f} m^3 vs DESC {V_desc:.5f} m^3 "
    f"(rel. diff {abs(V - V_desc) / V_desc:.1e})")

A, A_desc = np.linalg.norm(J, axis=-1).sum(dtype=np.float64) * dV, volume_integral(eq, "|J|")
log(f"int |J| dV:   {A:.3f} A m vs DESC {A_desc:.3f} A m "
    f"(rel. diff {abs(A - A_desc) / A_desc:.1e})")

j0 = n[1] // 2
assert abs(d["ys"][j0]) < 1e-12
Jy = J[:, j0, :, 1] * (d["xs"] > 0)[:, None]
dA = d["dx"][0] * d["dx"][2]
I, I_abs, I_desc = Jy.sum(dtype=np.float64) * dA, np.abs(Jy).sum(dtype=np.float64) * dA, cross_section_current(eq, 0.0)
log(f"I(phi=0):     {I:.3f} A vs DESC {I_desc:.3f} A "
    f"(|diff| / int|J_y| dA = {abs(I - I_desc) / I_abs:.1e})")

if args.out:
    np.savez_compressed(args.out, xs=d["xs"], ys=d["ys"], zs=d["zs"],
                        J=J, mask=mask)
    log(f"saved {args.out}")
