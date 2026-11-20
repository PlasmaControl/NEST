"""Shielding (sheet) current at q = 2 in DESC's rippled candidate-C states.

For each state: sigma = J.B / B^2 (the parallel current function, J_par = sigma B), and its (2, 1) harmonic in
straight-field-line angles,
    sigma_21(rho) = 2 <sigma exp(-i (2 theta* + zeta)) d theta*/d theta>_{theta, zeta},   theta* = theta + lambda,
on a radial grid that is very fine (d rho = 2.5e-4) across q = 2. In ideal MHD with nested surfaces the shielding
current is sigma_21 = K delta(rho - rho_s) (plus a smooth part), so as L grows the peak should grow ~L, the
width shrink ~1/L, and the integral across the surface converge to K.

Writes <out>/<tag>_sheet.npz (profiles) and <tag>_sheet.json (scalars). The phase reference is the arg of
sigma_21 at the peak; 'signed' is Re(sigma_21 exp(-i phi_peak)).
"""
import argparse
import json
import os
import time

import numpy as np

ap = argparse.ArgumentParser()
ap.add_argument("files", nargs="+")
ap.add_argument("--out", default="diag")
ap.add_argument("--chunk", type=int, default=8, help="radial surfaces per compute call")
ap.add_argument("--half", type=float, default=0.08, help="fine window half width in rho")
ap.add_argument("--drho", type=float, default=2.5e-4)
ap.add_argument("--part", type=int, default=0, help="this job's share of the surfaces (interleaved)")
ap.add_argument("--nparts", type=int, default=1, help="split the surfaces over this many jobs; then --merge")
ap.add_argument("--merge", action="store_true", help="assemble <tag>_part*.npz and write the summary")
a = ap.parse_args()
os.makedirs(a.out, exist_ok=True)

from desc import set_device  # noqa: E402

if os.environ.get("NEST_DEVICE") == "gpu":
    set_device("gpu")
import desc.io  # noqa: E402
from desc.equilibrium import Equilibrium  # noqa: E402
from desc.grid import LinearGrid, QuadratureGrid  # noqa: E402
from scipy.interpolate import CubicSpline  # noqa: E402
from scipy.optimize import brentq  # noqa: E402

T0 = time.time()


def log(*s):
    print(f"[{time.time() - T0:8.1f} s]", *s, flush=True)


def resonance(eq):
    rho = np.linspace(0.02, 0.98, 961)
    g = LinearGrid(rho=rho, M=8, N=0, NFP=1)
    io = g.compress(np.asarray(eq.compute("iota", grid=g)["iota"]))
    f = CubicSpline(rho, np.abs(io) - 0.5)
    i = np.flatnonzero(np.sign(f(rho[:-1])) != np.sign(f(rho[1:])))
    r = float(brentq(f, rho[i[0]], rho[i[0] + 1], xtol=1e-13))
    return r, float(CubicSpline(rho, io)(r, 1))


def profiles(eq, rho):
    """sigma_21 (complex), sigma_00, <|F|>_s, <|grad B^2/2mu0|>_s on the given surfaces."""
    Mth, Nz = 2 * eq.M + 6, 2 * eq.N + 2  # 4M+13 theta, 4N+5 zeta points
    keys = ["J*B", "|B|^2", "theta_PEST", "theta_PEST_t", "|F|", "|grad(|B|^2)|/2mu0", "sqrt(g)"]
    s21, s00, Fs, Ns = (np.zeros(len(rho), complex), np.zeros(len(rho)), np.zeros(len(rho)), np.zeros(len(rho)))
    for i0 in range(0, len(rho), a.chunk):
        r = rho[i0:i0 + a.chunk]
        g = LinearGrid(rho=r, M=Mth, N=Nz, NFP=1, sym=False)
        d = {k: np.asarray(v) for k, v in eq.compute(keys, grid=g).items()}
        sig = d["J*B"] / d["|B|^2"]
        ph = np.exp(-1j * (2 * d["theta_PEST"] + g.nodes[:, 2])) * d["theta_PEST_t"]
        sg = np.abs(d["sqrt(g)"])
        idx = g.inverse_rho_idx
        for j in range(len(r)):
            s = idx == j
            s21[i0 + j] = 2 * np.mean(sig[s] * ph[s])
            s00[i0 + j] = np.mean(sig[s])
            Fs[i0 + j] = np.sum(d["|F|"][s] * sg[s]) / np.sum(sg[s])
            Ns[i0 + j] = np.sum(d["|grad(|B|^2)|/2mu0"][s] * sg[s]) / np.sum(sg[s])
    return s21, s00, Fs, Ns


def fwhm(x, y, i):
    """full width at half max of y around index i (linear interpolation)."""
    h = y[i] / 2
    lo = i
    while lo > 0 and y[lo] > h:
        lo -= 1
    hi = i
    while hi < len(y) - 1 and y[hi] > h:
        hi += 1
    xl = np.interp(h, [y[lo], y[lo + 1]], [x[lo], x[lo + 1]])
    xh = np.interp(h, [y[hi], y[hi - 1]], [x[hi], x[hi - 1]])
    return float(xh - xl)


for path in a.files:
    eq = desc.io.load(path)
    eq = eq if isinstance(eq, Equilibrium) else eq[-1]
    tag = os.path.splitext(os.path.basename(path))[0]
    rs, iop = resonance(eq)
    fine = rs + np.arange(-a.half, a.half + a.drho / 2, a.drho)
    coarse = np.arange(0.01, 0.995, 0.005)
    rho = np.unique(np.concatenate([coarse[np.abs(coarse - rs) > a.half], fine]))
    if a.merge:
        s21, s00, Fs, Ns = (np.zeros(len(rho), complex), np.zeros(len(rho)), np.zeros(len(rho)), np.zeros(len(rho)))
        for i in range(a.nparts):
            z = np.load(os.path.join(a.out, f"{tag}_part{i}of{a.nparts}.npz"))
            assert np.allclose(z["rho"], rho[i::a.nparts])
            s21[i::a.nparts], s00[i::a.nparts], Fs[i::a.nparts], Ns[i::a.nparts] = z["s21"], z["s00"], z["F"], z["Fn"]
        log(f"{tag}: merged {a.nparts} parts")
    else:
        sel = rho[a.part::a.nparts]
        log(f"{tag}: L {eq.L} M {eq.M} N {eq.N}; q=2 at rho {rs:.6f}; {len(sel)} of {len(rho)} surfaces")
        s21, s00, Fs, Ns = profiles(eq, sel)
        if a.nparts > 1:
            np.savez(os.path.join(a.out, f"{tag}_part{a.part}of{a.nparts}.npz"), rho=sel, s21=s21, s00=s00, F=Fs,
                     Fn=Ns)
            log(f"{tag}: part {a.part} of {a.nparts} saved")
            continue
    ab = np.abs(s21)
    win = np.abs(rho - rs) <= a.half
    ip = int(np.flatnonzero(win)[np.argmax(ab[win])])
    phi = float(np.angle(s21[ip]))
    signed = np.real(s21 * np.exp(-1j * phi))
    out = {"file": path, "L": eq.L, "M": eq.M, "N": eq.N, "rho_s": rs, "d iota/d rho": iop,
           "peak rho": float(rho[ip]), "peak offset": float(rho[ip] - rs), "peak |sigma_21|": float(ab[ip]),
           "peak phase": phi, "sigma_00 at rho_s": float(np.interp(rs, rho, s00)),
           "FWHM": fwhm(rho, ab, ip)}
    # integral of the signed profile across the surface, for several half-windows
    out["integral signed"] = {}
    for dlt in (0.005, 0.01, 0.02, 0.04, 0.08):
        s = np.abs(rho - rs) <= dlt
        out["integral signed"][f"{dlt:g}"] = float(np.trapezoid(signed[s], rho[s]))
    # same with the complex harmonic (phase-free check)
    out["|integral complex| 0.02"] = float(abs(np.trapezoid(s21[np.abs(rho - rs) <= 0.02],
                                                            rho[np.abs(rho - rs) <= 0.02])))
    # ringing away from the surface
    for lo_, hi_ in ((0.02, 0.05), (0.05, 0.08)):
        s = (np.abs(rho - rs) > lo_) & (np.abs(rho - rs) <= hi_)
        out[f"max |sigma_21| at {lo_}<|rho-rho_s|<={hi_}"] = float(ab[s].max())
    out["max |sigma_21| outside window"] = float(ab[~win].max())
    out["rho of max outside window"] = float(rho[~win][np.argmax(ab[~win])])
    fr = Fs / Ns
    out["force at rho_s"] = float(np.interp(rs, rho, fr))
    out["force max in window"] = float(fr[win].max())
    out["force rho of max"] = float(rho[np.argmax(fr)])
    gq = QuadratureGrid(L=eq.L_grid, M=eq.M_grid, N=eq.N_grid, NFP=1)
    dq = eq.compute(["<|F|>_vol", "<|grad(|B|^2)|/2mu0>_vol"], grid=gq)
    out["force vol"] = float(dq["<|F|>_vol"] / dq["<|grad(|B|^2)|/2mu0>_vol"])
    # toroidal spectrum of the coefficients: max |c| per n
    out["n spectrum"] = {nm: [float(np.abs(np.asarray(getattr(eq, nm + "_lmn"))[
        np.abs(getattr(eq, nm + "_basis").modes[:, 2]) == n]).max(initial=0.0)) for n in range(eq.N + 1)]
        for nm in ("R", "Z", "L")}
    np.savez(os.path.join(a.out, tag + "_sheet.npz"), rho=rho, s21=s21, s00=s00, F=Fs, Fn=Ns, rho_s=rs)
    with open(os.path.join(a.out, tag + "_sheet.json"), "w") as f:
        json.dump(out, f, indent=1)
    log(f"{tag}: peak |s21| {ab[ip]:.4g} at {rho[ip]:.5f} (offset {rho[ip] - rs:+.2e}), FWHM {out['FWHM']:.3e}, "
        f"int(0.02) {out['integral signed']['0.02']:.4g}, force vol {out['force vol']:.2e}")
