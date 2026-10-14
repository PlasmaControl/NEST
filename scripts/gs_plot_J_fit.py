"""J of the BoundaryPotential fit against DESC's J (axisymmetric tokamak).

On a (rho, theta) grid in the plane zeta = 0, dense near the LCFS:
  a. flux-surface averages of J_phi and |J_pol| (DESC solid, fit dashed);
  b. the mismatch |J_fit - J_DESC| by rho (sqrt of the flux-surface average of
     its square), split into the parts along and across B_DESC, over the
     volume rms of J_DESC; the dotted line is DESC's own force error by rho
     (|F| / norm), for scale;
  c. log10 |J_fit - J_DESC| / rms J_DESC in the cross-section;
  d. the same as b against 1 - rho (the edge layer), with <|J|> of both.

usage: python scripts/gs_plot_J_fit.py [--eq ...] [--factors 2 2 1] [--target T]
    [--edge-weight 1] [--out figures/gs_J_fit_L24.png]
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
import matplotlib  # noqa: E402

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
from desc.grid import QuadratureGrid  # noqa: E402

import fit_cache  # noqa: E402
from boundary_potential import BoundaryPotential  # noqa: E402
from cartesian_grid import evaluate_chunked  # noqa: E402
from desc_input import DESCInput  # noqa: E402

jax.config.update("jax_enable_x64", True)
TOK = os.path.join(ROOT, "desc_input", "tokamak_free_boundary", "tokamak_L24M24")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--eq", default=TOK + ".h5")
    ap.add_argument("--factors", type=float, nargs=3, default=(2, 2, 1))
    ap.add_argument("--target", default="T", choices=["T", "J"])
    ap.add_argument("--edge-weight", type=float, default=1.0)
    ap.add_argument("--out", default=os.path.join(ROOT, "figures", "gs_J_fit_L24.png"))
    a = ap.parse_args()
    inp = DESCInput.load(a.eq)
    eq = inp.eq
    kw = dict(factors=tuple(a.factors), target=a.target)
    if a.edge_weight != 1.0:
        kw["edge_weight"] = a.edge_weight
    cur = fit_cache.fit(BoundaryPotential, inp, **kw)
    rho = np.unique(np.concatenate([np.linspace(0.01, 0.9, 90), 1 - np.logspace(-1, -3.5, 70)]))
    theta = np.linspace(0, 2 * np.pi, 256, endpoint=False)
    zeta = np.array([0.0])
    d = inp.compute_tensor(["J", "B", "sqrt(g)", "R", "Z", "|F|"], rho, theta, zeta)
    shape = (len(rho), len(theta))
    rr, tt = np.meshgrid(rho, theta, indexing="ij")
    rtz = np.column_stack([rr.ravel(), tt.ravel(), np.zeros(rr.size)])
    J_fit = np.asarray(evaluate_chunked(cur.point_evaluator(eq), rtz, 50000))  # (R, phi, Z)
    J_D, B = d["J"], d["B"]
    dJ = J_fit - J_D
    bhat = B / np.linalg.norm(B, axis=1)[:, None]
    dpar = np.sum(dJ * bhat, 1)
    dperp = np.linalg.norm(dJ - dpar[:, None] * bhat, axis=1)
    q = QuadratureGrid(L=2 * eq.L, M=2 * eq.M, N=0, NFP=1)
    dq = inp.compute(["|J|", "sqrt(g)", "<|grad(|B|^2)|/2mu0>_vol"], q)
    w = np.abs(dq["sqrt(g)"]) * q.weights
    rmsJ = float(np.sqrt(np.sum(w * dq["|J|"] ** 2) / np.sum(w)))
    norm = float(np.ravel(dq["<|grad(|B|^2)|/2mu0>_vol"])[0])
    sg = np.abs(d["sqrt(g)"]).reshape(shape)
    fs = lambda v: np.sum(sg * v.reshape(shape), 1) / np.sum(sg, 1)  # noqa: E731
    e_all = np.sqrt(fs(np.sum(dJ**2, 1))) / rmsJ
    e_par = np.sqrt(fs(dpar**2)) / rmsJ
    e_perp = np.sqrt(fs(dperp**2)) / rmsJ
    F_D = fs(d["|F|"]) / norm
    Jphi_D, Jphi_f = fs(J_D[:, 1]), fs(J_fit[:, 1])
    Jp_D = fs(np.hypot(J_D[:, 0], J_D[:, 2]))
    Jp_f = fs(np.hypot(J_fit[:, 0], J_fit[:, 2]))
    absJ_D, absJ_f = fs(np.linalg.norm(J_D, axis=1)), fs(np.linalg.norm(J_fit, axis=1))
    b = cur.basis

    fig, ax = plt.subplots(2, 2, figsize=(13, 10))
    ax = ax.ravel()
    ax[0].plot(rho, Jphi_D / 1e3, "C0", label=r"$\langle J_\phi\rangle$ DESC")
    ax[0].plot(rho, Jphi_f / 1e3, "k--", lw=1, label="fit")
    ax[0].plot(rho, Jp_D / 1e3, "C1", label=r"$\langle |J_{pol}|\rangle$ DESC")
    ax[0].plot(rho, Jp_f / 1e3, "k:", lw=1, label="fit")
    ax[0].set_xlabel(r"$\rho$")
    ax[0].set_ylabel("kA/m²")
    ax[0].legend(fontsize=8)
    ax[0].set_title("a. current density, flux-surface averages")
    for x, ax_ in ((rho, ax[1]), (1 - rho, ax[3])):
        ax_.semilogy(x, e_all, "k", label=r"$|J_{fit}-J_{DESC}|$")
        ax_.semilogy(x, e_par, "C0", label="along B")
        ax_.semilogy(x, e_perp, "C3", label="across B")
        ax_.semilogy(x, F_D, "C2:", label="DESC |F| / norm (scale)")
        ax_.set_ylabel(r"rms on the surface / rms $J_{DESC}$")
    ax[3].semilogy(1 - rho, absJ_D / rmsJ, "C7", lw=2, alpha=0.5, label=r"$\langle|J|\rangle$ DESC")
    ax[3].semilogy(1 - rho, absJ_f / rmsJ, "C7--", lw=1, label=r"$\langle|J|\rangle$ fit")
    ax[1].set_xlabel(r"$\rho$")
    ax[1].legend(fontsize=8)
    ax[1].set_title("b. mismatch by flux surface")
    ax[3].set_xscale("log")
    ax[3].set_xlabel(r"$1-\rho$")
    ax[3].legend(fontsize=8)
    ax[3].set_title("d. the edge layer")
    R, Z = d["R"].reshape(shape), d["Z"].reshape(shape)
    E = np.log10(np.linalg.norm(dJ, axis=1).reshape(shape) / rmsJ + 1e-30)
    Rc, Zc, Ec = (np.concatenate([v, v[:, :1]], 1) for v in (R, Z, E))  # close in theta
    im = ax[2].contourf(Rc, Zc, Ec, np.linspace(-9, -2, 29), cmap="magma", extend="both")
    plt.colorbar(im, ax=ax[2], label=r"log$_{10}$ |J$_{fit}$ - J$_{DESC}$| / rms J$_{DESC}$")
    ax[2].set_aspect("equal")
    ax[2].set_xlabel("R [m]")
    ax[2].set_ylabel("Z [m]")
    ax[2].set_title(r"c. mismatch in the cross-section ($\phi$ = 0)")
    fig.suptitle(f"{inp.name}: fit of {a.target} (potential L, M = {b.L}, {b.M}; edge weight "
                 f"{a.edge_weight:g}) against DESC's J; rms J_DESC = {rmsJ / 1e3:.1f} kA/m²")
    fig.tight_layout()
    fig.savefig(a.out, dpi=140)
    print(f"saved {a.out}; mismatch / rms J: volume rms "
          f"{np.sqrt(np.sum(sg * np.sum(dJ**2, 1).reshape(shape)) / np.sum(sg)) / rmsJ:.2e}")


if __name__ == "__main__":
    main()
