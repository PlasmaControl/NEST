"""Can the pressure basis (window (1-rho^2)^3 x Fourier-Zernike, ansi) represent the
flattened pressure of a pendulum island? Model island at rho_res, full width w:
Omega = 8 x^2 / w^2 - cos(chi), chi = m theta* - n zeta, x = rho - rho_res; p = p0(rho_res)
inside the separatrix (Omega < 1), p = p0(rho_res + sign(x) (w / sqrt 8) sqrt(Omega - 1))
outside (constant on the model's field lines). Decomposed into cos(k chi); harmonic k needs
poloidal m k <= M and the radial functions window * R_l^{mk}, l <= L, l = mk mod 2 (the
basis's own set). Approximation: theta* = theta (lambda's mixing of poloidal harmonics ignored).
Error on the band |x| < w, relative to the unflattened drop |p0'| w."""
import sys, os
import numpy as np
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..", "nest"))
from desc import set_device; set_device("cpu")
from desc.basis import FourierZernikeBasis, zernike_radial
from desc_input import DESCInput
from pressure import window

D = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..", "desc_input", "vmec_free_boundary", "NCSX")
inp = DESCInput.load(os.path.join(D, "NCSX_fb_L16M16N12.h5"))
eq = inp.eq
from desc.grid import LinearGrid
rho = np.linspace(1e-3, 0.999, 6000)
p0 = np.asarray(eq.compute("p", grid=LinearGrid(rho=rho, NFP=eq.NFP))["p"])
p0f = lambda r: np.interp(r, rho, p0)
chi = 2 * np.pi * np.arange(512) / 512
m0, rho_res = 5, 0.802
dp = np.interp(rho_res, rho, np.gradient(p0, rho))
wr = rho * np.gradient(rho)  # area weight rho drho


def target(w):
    x = rho[:, None] - rho_res
    Om = 8 * x**2 / w**2 - np.cos(chi)[None]
    xb = np.sign(x) * (w / np.sqrt(8)) * np.sqrt(np.clip(Om - 1, 0, None))
    return np.where(Om < 1, p0f(rho_res), p0f(rho_res + xb))


def harmonics(P, kmax):
    out = [P.mean(1)]
    for k in range(1, kmax + 1):
        out.append(2 * np.mean(P * np.cos(k * chi)[None], axis=1))
    return np.array(out)  # (k, rho)


def fit_profile(prof, m, ls):
    A = window(rho, 2)[:, None] * np.asarray(zernike_radial(rho[:, None], np.array(ls), np.full(len(ls), m)))
    sw = np.sqrt(wr)
    c, *_ = np.linalg.lstsq(A * sw[:, None], prof * sw, rcond=None)
    return A @ c


KMAX = 40
for w in (0.03, 0.059, 0.12):
    Pt = target(w)
    hk = harmonics(Pt, KMAX)
    band = np.abs(rho - rho_res) < w
    E = lambda Pfit: np.sqrt(np.sum(((Pfit - Pt) ** 2 * wr[:, None])[band]) / np.sum(np.broadcast_to(wr[:, None], Pt.shape)[band])) / (abs(dp) * w)
    E_unflat = E(np.broadcast_to(p0[:, None], Pt.shape))
    line = [f"w {w:.3f} (Delta p {abs(dp) * w / 1e3:.1f} kPa): unflattened p0(rho) error {E_unflat:.3f}"]
    for (L, M) in ((32, 32), (48, 32), (64, 48), (96, 64), (160, 96)):
        b = FourierZernikeBasis(L, M, 0, NFP=3, sym=False, spectral_indexing="ansi")
        modes = b.modes
        Pf = np.zeros_like(Pt)
        for k in range(KMAX + 1):
            m = m0 * k
            ls = sorted(set(modes[modes[:, 1] == m, 0]))
            if not ls:
                continue
            prof = fit_profile(hk[k], m, ls)
            Pf += prof[:, None] * (np.cos(k * chi)[None] if k else 1.0)
        kept = max([k for k in range(KMAX + 1) if (modes[:, 1] == m0 * k).any()])
        line.append(f"  L/M {L}/{M} (helical k <= {kept}): error {E(Pf):.3f}, flattening captured {1 - E(Pf) / E_unflat:.0%}")
    print("\n".join(line))
# truncation in k alone (no radial limit): the angular content of the flattened profile
Pt = target(0.059); hk = harmonics(Pt, KMAX); band = np.abs(rho - rho_res) < 0.059
for kc in (1, 2, 3, 6, 10, 20):
    Pf = hk[0][:, None] + sum(hk[k][:, None] * np.cos(k * chi)[None] for k in range(1, kc + 1))
    e = np.sqrt(np.mean(((Pf - Pt) ** 2)[band])) / (abs(dp) * 0.059)
    print(f"w 0.059, exact radial profiles, helical harmonics k <= {kc}: error {e:.3f}")
