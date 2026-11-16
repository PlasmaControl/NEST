"""Resonant 2/1 diagnostics on a tensor collocation grid (``force.ForceResidual``): b21 = the sin(chi) harmonic of
b^rho = B.grad rho / B.grad zeta and s21 = the cos(chi) harmonic of sigma = J.B / B^2 per radial node, chi = m theta*
- n zeta in DESC's straight-field-line angle, with the theta* Jacobian (shared by dshape_fixed_diag.py and the
per-trial log of mf_fixed.py)."""

import jax.numpy as jnp
import numpy as np


def resonant_setup(inp, fr, m=2, n=-1):
    """Per collocation point: grad rho (Cartesian), R, phi, and the resonant phase weights for the
    m/|n| harmonic in DESC's straight-field-line angle theta* = theta + lambda: chi = m theta* - n zeta
    (n = -1: chi = 2 theta* + zeta, the field-line phase at iota = -1/2), Jacobian 1 + lambda_theta."""
    tg = fr.tgrid
    d = inp.compute_tensor(["lambda", "lambda_t"], tg.rho, tg.theta, tg.zeta)
    lam, lam_t = np.asarray(d["lambda"]).ravel(), np.asarray(d["lambda_t"]).ravel()
    one, zero = jnp.ones(tg.shape), jnp.zeros(tg.shape)
    gr = np.asarray(fr.geo.gradient(one, zero, zero)).reshape(-1, 3)  # (R, phi, Z)
    rtz = np.asarray(fr.rtz)
    chi = m * (rtz[:, 1] + lam) - n * rtz[:, 2]
    return {"grad_rho": gr, "R": np.asarray(fr.geo.R).ravel(), "phi": rtz[:, 2], "chi": chi, "jac": 1 + lam_t,
            "shape": tuple(tg.shape), "rho": np.asarray(tg.rho)}


def resonant_profiles(rs, B, J=None):
    """Per radial node: b21 = the sin(chi) harmonic of b^rho = B.grad rho / B.grad zeta (odd under
    stellarator symmetry) and s21 = the cos(chi) harmonic of sigma = J.B / B^2 (even), both averaged over
    (theta, zeta) with the theta* Jacobian; the half-period symmetric grid gives the full-torus average
    of these even products. B, J Cartesian (n, 3)."""
    ph = rs["phi"]
    c, s = np.cos(ph), np.sin(ph)
    BR, Bp, BZ = B[:, 0] * c + B[:, 1] * s, -B[:, 0] * s + B[:, 1] * c, B[:, 2]
    br = (BR * rs["grad_rho"][:, 0] + Bp * rs["grad_rho"][:, 1] + BZ * rs["grad_rho"][:, 2]) / (Bp / rs["R"])
    sh = rs["shape"]
    wgt = rs["jac"]
    b21 = 2 * np.mean((br * np.sin(rs["chi"]) * wgt).reshape(sh[0], -1), 1)
    out = {"b21": b21}
    if J is not None:
        sig = np.sum(J * B, 1) / np.sum(B * B, 1)
        out["s21"] = 2 * np.mean((sig * np.cos(rs["chi"]) * wgt).reshape(sh[0], -1), 1)
        out["s00"] = np.mean((sig * wgt).reshape(sh[0], -1), 1)
    return out


def peak_width(rho, y, rq, band=0.1):
    """|y| peak near rq: location, height, full width at half maximum (linear interpolation)."""
    m = np.abs(rho - rq) < band
    r, a = rho[m], np.abs(y[m])
    if len(r) < 3:
        return {}
    k = int(np.argmax(a))
    h = a[k] / 2
    lo = k
    while lo > 0 and a[lo] > h:
        lo -= 1
    hi = k
    while hi < len(a) - 1 and a[hi] > h:
        hi += 1
    left = r[lo] if a[lo] > h else r[lo] + (h - a[lo]) / (a[lo + 1] - a[lo]) * (r[lo + 1] - r[lo])
    right = r[hi] if a[hi] > h else r[hi - 1] + (a[hi - 1] - h) / (a[hi - 1] - a[hi]) * (r[hi] - r[hi - 1])
    return {"rho": float(r[k]), "peak": float(a[k]), "fwhm": float(right - left),
            "edge-limited": bool(lo == 0 or hi == len(a) - 1)}
