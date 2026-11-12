"""Fixed-boundary DESC tokamak with a resonant (m, n) boundary ripple (NFP = 1).

Base: an axisymmetric candidate-C equilibrium (tokamak_island_C_L{L}M{L}.h5: hollow p with dp/ds = 0 at q = 2,
prescribed current, NFP = 1, stellarator symmetric).

Ripple: a normal displacement of the base LCFS,
    x -> x + xi * cos(m theta + k zeta) * nhat(theta),     xi = amp * a,
with nhat the outward unit normal in the poloidal plane and k = -m * iota(rho_r) rounded (the helicity that is
constant along field lines at the resonance; k = +1 for m = 2 and iota = -1/2). It is fitted by least squares onto
DESC's boundary basis, so no sign convention is hand-placed. cos(m theta + k zeta) is even under
(theta, zeta) -> (-theta, -zeta), so dR is cos-type and dZ is sin-type: stellarator symmetry is kept.
--helicity flip uses k -> -k (a non-resonant control).

Each amplitude is reached by continuation (perturb to second order, then a full force solve) and saved with
diagnostics: volume force error (DESC's grid and a 2x grid), radial profile of the surface-averaged |F|, the iota
profile, q = m/n location and shear, and the (m, k) harmonic of lambda = J.B/B^2 against rho (in DESC's theta, zeta;
the shielding current shows up there).

Usage (local CPU, from this directory):
    python make_bdry_ripple.py --base tokamak_island_C_L16M16.h5 --N 5 --amps 1e-4 3e-4 1e-3 --out out/
"""
import argparse
import json
import os
import sys
import time

import numpy as np

ap = argparse.ArgumentParser()
ap.add_argument("--base", required=True, help="axisymmetric base equilibrium (.h5)")
ap.add_argument("--LM", type=int, nargs=2, default=None, help="change L, M (default: the base's)")
ap.add_argument("--N", type=int, default=5, help="toroidal resolution (NFP = 1)")
ap.add_argument("--m", type=int, default=2)
ap.add_argument("--n", type=int, default=1)
ap.add_argument("--amps", type=float, nargs="+", default=[1e-4, 3e-4, 1e-3, 3e-3],
                help="xi / a, the normal displacement amplitude over the minor radius (continuation order)")
ap.add_argument("--helicity", choices=["auto", "flip"], default="auto")
ap.add_argument("--maxiter", type=int, default=300)
ap.add_argument("--ftol", type=float, default=1e-4,
                help="stop when an accepted step lowers the cost by < ftol x cost (1e-12 crawls for hundreds of its)")
ap.add_argument("--name", default="tokamak_ripple_C")
ap.add_argument("--out", default="out", help="output directory")
ap.add_argument("--mirror", default=None, help="also copy outputs here (e.g. a Modal Volume path)")
a = ap.parse_args()
os.makedirs(a.out, exist_ok=True)
if a.mirror:
    os.makedirs(a.mirror, exist_ok=True)

import desc.io  # noqa: E402
from desc.equilibrium import Equilibrium  # noqa: E402
from desc.grid import LinearGrid, QuadratureGrid  # noqa: E402
from scipy.interpolate import CubicSpline  # noqa: E402
from scipy.optimize import brentq  # noqa: E402

T0 = time.time()


def log(*s):
    print(f"[{time.time() - T0:8.1f} s]", *s, flush=True)


def save_json(path, obj):
    txt = json.dumps(obj, indent=1, default=float)
    for d in [a.out] + ([a.mirror] if a.mirror else []):
        with open(os.path.join(d, path), "w") as f:
            f.write(txt)


# ------------------------------------------------------------------ base equilibrium
eq = desc.io.load(a.base)
if not isinstance(eq, Equilibrium):
    eq = eq[-1]
assert eq.NFP == 1 and eq.sym, "base must be NFP = 1 and stellarator symmetric"
assert eq.N == 0, "base must be axisymmetric"
L, M = a.LM if a.LM else (eq.L, eq.M)
log(f"base {a.base}: L {eq.L} M {eq.M} N {eq.N}, NFP {eq.NFP}, Psi {eq.Psi}")

iota_r_target = a.n / a.m


def resonance(eq):
    """rho, iota, d iota/d rho where |iota| = n/m (surface-averaged iota from a fine 1-D grid)."""
    rho = np.linspace(0.02, 0.98, 481)
    g = LinearGrid(rho=rho, M=2 * eq.M, N=2 * eq.N, NFP=eq.NFP)
    io = g.compress(eq.compute("iota", grid=g)["iota"])
    f = CubicSpline(rho, np.abs(io) - iota_r_target)
    i = np.flatnonzero(np.sign(f(rho[:-1])) != np.sign(f(rho[1:])))
    if len(i) != 1:
        raise RuntimeError(f"{len(i)} crossings of |iota| = {iota_r_target}")
    r = float(brentq(f, rho[i[0]], rho[i[0] + 1], xtol=1e-12))
    sp = CubicSpline(rho, io)
    return r, float(sp(r)), float(sp(r, 1))


rho_r, iota_r, iota_p = resonance(eq)
k = int(round(-a.m * iota_r))
if a.helicity == "flip":
    k = -k
log(f"resonance |iota| = {a.n}/{a.m}: rho {rho_r:.5f}, iota {iota_r:.6f}, d iota/d rho {iota_p:.5f}; "
    f"ripple helicity cos({a.m} theta {'+' if k >= 0 else '-'} {abs(k)} zeta) ({a.helicity})")

if (L, M) != (eq.L, eq.M) or a.N != eq.N:
    eq.change_resolution(L=L, M=M, N=a.N, L_grid=2 * L, M_grid=2 * M, N_grid=2 * a.N)
    log(f"resolution -> L {eq.L} M {eq.M} N {eq.N} (grid {eq.L_grid}/{eq.M_grid}/{eq.N_grid})")
# a truncation in L, M is not an equilibrium: re-solve the (axisymmetric) base then. Raising only N needs no
# re-solve (the n != 0 residuals of an axisymmetric equilibrium vanish by symmetry), and a re-solve there only
# walks along DESC's interior poloidal-angle gauge (seen at L24 N5: coefficients 2-6e-4 at flat cost).
if (L, M) != (base_LM := (int(desc.io.load(a.base).L), int(desc.io.load(a.base).M))):
    log(f"re-solving the base at L {L} M {M} (truncated from {base_LM})")
    eq.solve(objective="force", ftol=a.ftol, xtol=1e-14, gtol=1e-14, maxiter=a.maxiter, verbose=3)
minor_a = float(eq.compute("a")["a"])
log(f"minor radius a = {minor_a:.6f} m")

# ------------------------------------------------------------------ the ripple, fitted onto the boundary basis
srf = eq.surface
gs = LinearGrid(rho=np.array([1.0]), M=4 * srf.M + 8, N=4 * max(srf.N, 1) + 8, NFP=1, sym=False)
ds = srf.compute(["R", "Z", "R_t", "Z_t"], grid=gs)
R, Z, Rt, Zt = (np.asarray(ds[q]) for q in ("R", "Z", "R_t", "Z_t"))
nR, nZ = Zt, -Rt
nn = np.hypot(nR, nZ)
nR, nZ = nR / nn, nZ / nn
R0c, Z0c = R.mean(), Z.mean()
sgn = np.sign(np.mean(nR * (R - R0c) + nZ * (Z - Z0c)))
nR, nZ = sgn * nR, sgn * nZ  # outward
th, ze = gs.nodes[:, 1], gs.nodes[:, 2]
shape = np.cos(a.m * th + k * ze)
AR = np.asarray(srf.R_basis.evaluate(gs))
AZ = np.asarray(srf.Z_basis.evaluate(gs))
cR, *_ = np.linalg.lstsq(AR, shape * nR, rcond=None)  # per metre of xi
cZ, *_ = np.linalg.lstsq(AZ, shape * nZ, rcond=None)
fit_err = max(np.abs(AR @ cR - shape * nR).max(), np.abs(AZ @ cZ - shape * nZ).max())
log(f"ripple fit onto boundary basis (M {srf.M}, N {srf.N}): max error {fit_err:.2e} per unit xi")
if fit_err > 1e-6:
    log("WARNING: the boundary basis does not represent the ripple exactly (raise the surface M)")
Rb0, Zb0 = np.asarray(eq.Rb_lmn).copy(), np.asarray(eq.Zb_lmn).copy()
big = lambda c, b: [(tuple(int(x) for x in b.modes[i][1:]), float(c[i]))  # noqa: E731
                    for i in np.argsort(-np.abs(c))[:6] if abs(c[i]) > 1e-8]
log(f"largest dRb modes (m, n): {big(cR, srf.R_basis)}")
log(f"largest dZb modes (m, n): {big(cZ, srf.Z_basis)}")


# ------------------------------------------------------------------ diagnostics
def geom_spectrum(eq, rhos=(0.3, 0.5, 0.6375, 0.8, 0.95), nth=512, mmax=4):
    """Gauge-invariant shape of flux surfaces: for each zeta plane the cross-section (a closed curve, whatever
    its theta labelling) -> centroid (R_c, Z_c) and r(omega) about it (omega the geometric angle). Returns, per
    rho, 2|c_mn| / r_00 of r(omega, zeta) for m = 0..mmax, n = 0..N (sum of +-n), and the n-amplitudes of R_c, Z_c."""
    nze = 4 * eq.N + 4
    res = {}
    w = 2 * np.pi * np.arange(nth) / nth
    for r in rhos:
        g = LinearGrid(rho=np.array([r]), theta=nth, zeta=nze, NFP=1, sym=False)
        dd = eq.compute(["R", "Z"], grid=g)
        R, Z = np.asarray(dd["R"]), np.asarray(dd["Z"])
        th, ze = g.nodes[:, 1], g.nodes[:, 2]
        zs = np.unique(ze)
        rr, cxs, cys = np.zeros((len(zs), nth)), np.zeros(len(zs)), np.zeros(len(zs))
        for iz, z0 in enumerate(zs):
            s_ = np.isclose(ze, z0)
            o = np.argsort(th[s_])
            x, y = R[s_][o], Z[s_][o]
            x1, y1 = np.roll(x, -1), np.roll(y, -1)
            cr = x * y1 - x1 * y
            A = cr.sum() / 2
            cx, cy = ((x + x1) * cr).sum() / (6 * A), ((y + y1) * cr).sum() / (6 * A)
            om = np.arctan2(y - cy, x - cx)
            oo = np.argsort(om)
            rr[iz] = np.interp(w, om[oo], np.hypot(x - cx, y - cy)[oo], period=2 * np.pi)
            cxs[iz], cys[iz] = cx, cy
        F = np.fft.fft2(rr) / rr.size  # [n, m], basis exp(i (n zeta + m omega))
        r00 = abs(F[0, 0])
        tab = [[float((abs(F[nn, mm]) + (abs(F[-nn, mm]) if nn else 0.0)) * (2 if mm else 1) / r00)
                for nn in range(eq.N + 1)] for mm in range(mmax + 1)]
        fc = lambda v: [float(abs(np.fft.fft(v)[nn]) * (2 if nn else 1) / len(v)) for nn in range(eq.N + 1)]  # noqa
        res[f"{r:g}"] = {"r_mn/r00 [m][n]": tab, "R_c by n [m]": fc(cxs), "Z_c by n [m]": fc(cys)}
    return res


def diagnostics(eq, amp):
    out = {"amp xi/a": amp, "xi [m]": amp * minor_a, "L": eq.L, "M": eq.M, "N": eq.N, "k": k,
           "helicity": a.helicity}
    # toroidal spectrum of the solution: max |coefficient| per n (driven response should scale with amp)
    out["n spectrum"] = {}
    for nm in ("R", "Z", "L"):
        cc, md = np.asarray(getattr(eq, nm + "_lmn")), getattr(eq, nm + "_basis").modes
        out["n spectrum"][nm] = [float(np.abs(cc[np.abs(md[:, 2]) == nn]).max(initial=0.0))
                                 for nn in range(eq.N + 1)]
    out["n spectrum"]["note"] = "max |coefficient| per n; gauge-dependent (interior theta relabelling)"
    out["geometric spectrum"] = geom_spectrum(eq)
    for tag, fac in (("grid x1", 1), ("grid x2", 2)):
        g = QuadratureGrid(L=fac * eq.L_grid, M=fac * eq.M_grid, N=fac * eq.N_grid, NFP=1)
        d = eq.compute(["<|F|>_vol", "<|grad(|B|^2)|/2mu0>_vol"], grid=g)
        out[f"<|F|>_vol / <|grad B^2/2mu0|>_vol ({tag})"] = float(d["<|F|>_vol"] / d["<|grad(|B|^2)|/2mu0>_vol"])
        out[f"<|F|>_vol [N/m^3] ({tag})"] = float(d["<|F|>_vol"])
    r, io, iop = resonance(eq)
    out["resonance"] = {"rho": r, "iota": io, "d iota/d rho": iop, "shear rho iota'/iota": r * iop / io}
    # radial profiles: surface means of |F| and of the norm, the (m, k) harmonic of lambda = J.B/B^2
    rho = np.unique(np.concatenate([np.linspace(0.02, 0.99, 98), r + np.linspace(-0.08, 0.08, 81)]))
    rho = rho[(rho > 0) & (rho < 1)]
    g = LinearGrid(rho=rho, M=4 * eq.M + 4, N=4 * eq.N + 4, NFP=1, sym=False)
    d = eq.compute(["|F|", "|grad(|B|^2)|/2mu0", "sqrt(g)", "J*B", "|B|^2", "iota", "theta_PEST", "theta_PEST_t"],
                   grid=g)
    sg = np.abs(np.asarray(d["sqrt(g)"]))
    surf = lambda q: g.compress(g.expand(np.asarray(  # noqa: E731
        [np.sum((q * sg)[g.inverse_rho_idx == i]) / np.sum(sg[g.inverse_rho_idx == i]) for i in range(g.num_rho)])))
    F_s = surf(np.asarray(d["|F|"]))
    N_s = surf(np.asarray(d["|grad(|B|^2)|/2mu0"]))
    lam = np.asarray(d["J*B"]) / np.asarray(d["|B|^2"])
    th, ze = g.nodes[:, 1], g.nodes[:, 2]
    # (m, k) harmonic in straight-field-line angles (theta* = theta + lambda_DESC, weight d theta*/d theta):
    # independent of DESC's interior poloidal-angle gauge
    tp, tpt = np.asarray(d["theta_PEST"]), np.asarray(d["theta_PEST_t"])
    ph = np.exp(-1j * (a.m * tp + k * ze)) * tpt
    c = np.array([np.mean((lam * ph)[g.inverse_rho_idx == i]) * 2 for i in range(g.num_rho)])
    lam0 = np.array([np.mean(lam[g.inverse_rho_idx == i]) for i in range(g.num_rho)])
    out["profiles"] = {"rho": rho.tolist(), "<|F|>_s": F_s.tolist(), "<|grad B^2/2mu0|>_s": N_s.tolist(),
                       "lambda_mk |.|": np.abs(c).tolist(), "lambda_mk arg": np.angle(c).tolist(),
                       "lambda_00": lam0.tolist(), "iota": g.compress(np.asarray(d["iota"])).tolist()}
    i = int(np.argmax(np.abs(c)))
    out["lambda_mk peak"] = {"rho": float(rho[i]), "|lambda_mk|": float(np.abs(c[i])),
                             "|lambda_mk| / |lambda_00|": float(np.abs(c[i]) / abs(lam0[i]))}
    ir = int(np.argmin(np.abs(rho - r)))
    out["at resonance"] = {"rho": float(rho[ir]), "|lambda_mk|": float(np.abs(c[ir])),
                           "|lambda_mk| / |lambda_00|": float(np.abs(c[ir]) / abs(lam0[ir])),
                           "<|F|>_s / <|grad B^2/2mu0|>_s": float(F_s[ir] / N_s[ir])}
    j = int(np.argmax(F_s / N_s))
    out["force peak"] = {"rho": float(rho[j]), "<|F|>_s / <|grad B^2/2mu0|>_s": float(F_s[j] / N_s[j])}
    return out


summary = {"args": vars(a), "base": a.base, "a [m]": minor_a, "k": k,
           "base resonance": {"rho": rho_r, "iota": iota_r, "d iota/d rho": iota_p},
           "ripple fit error": float(fit_err), "states": []}
tag0 = f"{a.name}_L{eq.L}M{eq.M}N{eq.N}" + ("_flip" if a.helicity == "flip" else "")
d0 = diagnostics(eq, 0.0)
summary["states"].append(d0)
log(f"amp 0: force {d0['<|F|>_vol / <|grad B^2/2mu0|>_vol (grid x1)']:.3e}, geometric n!=0 max "
    f"{max(max(row[1:]) for v in d0['geometric spectrum'].values() for row in v['r_mn/r00 [m][n]']):.2e}, "
    f"lambda_mk peak {d0['lambda_mk peak']}")
save_json(tag0 + "_summary.json", summary)

for amp in a.amps:
    xi = amp * minor_a
    t1 = time.time()
    log(f"=== amp xi/a = {amp:g} (xi = {xi:.3e} m) ===")
    dRb = Rb0 + xi * cR - np.asarray(eq.Rb_lmn)
    dZb = Zb0 + xi * cZ - np.asarray(eq.Zb_lmn)
    eq.perturb(deltas={"Rb_lmn": dRb, "Zb_lmn": dZb}, order=2, verbose=2, copy=False)
    eq.solve(objective="force", ftol=a.ftol, xtol=1e-14, gtol=1e-14, maxiter=a.maxiter, verbose=3)
    tag = f"{tag0}_x{amp:g}"
    for d in [a.out] + ([a.mirror] if a.mirror else []):
        eq.save(os.path.join(d, tag + ".h5"))
    dg = diagnostics(eq, amp)
    dg["wall [s]"] = time.time() - t1
    summary["states"].append(dg)
    save_json(tag0 + "_summary.json", summary)
    log(f"amp {amp:g}: force {dg['<|F|>_vol / <|grad B^2/2mu0|>_vol (grid x1)']:.3e} "
        f"(x2 grid {dg['<|F|>_vol / <|grad B^2/2mu0|>_vol (grid x2)']:.3e}); resonance {dg['resonance']}; "
        f"at resonance {dg['at resonance']}; geometric r_2n/r00 at rho 0.6375 {['%.2e' % v for v in dg['geometric spectrum']['0.6375']['r_mn/r00 [m][n]'][2]]}; force peak {dg['force peak']}; lambda_mk peak {dg['lambda_mk peak']}; {dg['wall [s]']:.0f} s")
log("done")
