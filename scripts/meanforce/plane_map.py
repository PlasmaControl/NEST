"""Exact force maps on a plane of current states (NEST fixed boundary, option A).

F(z) is exactly quadratic in z (J linear, B = B_ext + B_ann[J] affine, the least-squares pressure a linear
projection), so on the plane z(s, t) = z0 + s d1 + t d2

    F(s, t) = F0 + s Fs + t Ft + s^2 Fss + s t Fst + t^2 Ftt     (per collocation point, 3 components)

and six state evaluations give F everywhere on the plane: (0,0), (+-1,0), (0,+-1), (1,1). A seventh point checks
the fit. The misfit g (|J - J_fit| rows) is affine and b_mk linear, so they need no extra evaluations.
On a grid of (s, t) the map holds: the volume mean of |F| (/ DESC's), the band p-norm and band max (/ DESC's, on
the --p-band mask), the max over all points, |J - J_fit|/rms, and the projection of b_mk on the island phase.
"""
import json
import os
import time

import jax.numpy as jnp
import numpy as np


def _load(tok, u0):
    return jnp.asarray(u0) if tok == "J_fit" else jnp.asarray(np.load(tok)["u"])


def plane(A, spec, u0, rho_pts, w_pts, pmask, norm, M_desc, desc_p, bmk, b_vac, out, mirror, log, p=16.0):
    t0 = time.time()
    parts = spec.split("|")
    name, base, d1p, d1m, d2p, d2m = parts[:6]
    s0, s1, ns = (float(x) for x in parts[6].split(","))
    t0_, t1_, nt = (float(x) for x in parts[7].split(","))
    z0 = _load(base, u0)
    d1 = _load(d1p, u0) - _load(d1m, u0)
    d2 = _load(d2p, u0) - _load(d2m, u0)

    def ev(s, t):
        st = A.state(z0 + s * d1 + t * d2)
        return jnp.asarray(st["F"]), jnp.asarray(st["g"])
    F00, g00 = ev(0, 0)
    Fp0, gp0 = ev(1, 0)
    Fm0, _ = ev(-1, 0)
    F0p, g0p = ev(0, 1)
    F0m, _ = ev(0, -1)
    F11, _ = ev(1, 1)
    Fs, Fss = (Fp0 - Fm0) / 2, (Fp0 + Fm0) / 2 - F00
    Ft, Ftt = (F0p - F0m) / 2, (F0p + F0m) / 2 - F00
    Fst = F11 - F00 - Fs - Ft - Fss - Ftt
    gs, gt = gp0 - g00, g0p - g00

    def Fq(s, t):
        return F00 + s * Fs + t * Ft + s * s * Fss + s * t * Fst + t * t * Ftt
    sc, tc = 0.5, -0.7
    Fc, _ = ev(sc, tc)
    fit_err = float(jnp.linalg.norm(Fq(sc, tc) - Fc) / jnp.linalg.norm(Fc))
    w = jnp.asarray(w_pts)
    m = jnp.asarray(pmask, dtype=w.dtype)
    Wb = float(jnp.sum(w * m))
    dp = desc_p[p]
    S = np.linspace(s0, s1, int(ns))
    T = np.linspace(t0_, t1_, int(nt))
    keys = ["mean/M_desc", "band p16/DESC", "band max/DESC", "max/DESC band max", "|J-J_fit|/rms"]
    maps = {k: np.zeros((len(S), len(T))) for k in keys}
    for i, s in enumerate(S):
        for j, t in enumerate(T):
            aF = jnp.linalg.norm(Fq(s, t), axis=1)
            Fi = aF * m
            Fmax = jnp.max(Fi)
            Mp = Fmax * (jnp.sum(w * m * (Fi / Fmax) ** p) / Wb) ** (1.0 / p) / norm
            maps["mean/M_desc"][i, j] = float(jnp.sum(w * aF) / norm) / M_desc
            maps["band p16/DESC"][i, j] = float(Mp) / dp["M_p"]
            maps["band max/DESC"][i, j] = float(Fmax / norm) / dp["max_int"]
            maps["max/DESC band max"][i, j] = float(jnp.max(aF) / norm) / dp["max_int"]
            maps["|J-J_fit|/rms"][i, j] = float(jnp.linalg.norm(g00 + s * gs + t * gt))
    res = {"spec": spec, "fit check (s, t) = (0.5, -0.7)": fit_err, "s": S.tolist(), "t": T.tolist(),
           "|d1|/|z0|": float(jnp.linalg.norm(d1) / jnp.linalg.norm(z0)), "|d2|/|z0|": float(jnp.linalg.norm(d2) / jnp.linalg.norm(z0))}
    if bmk is not None:
        b0 = bmk(z0)
        b1 = bmk(z0 + d1) - b0
        b2 = bmk(z0 + d2) - b0
        res["b_mk"] = {"base": [b0.real, b0.imag], "per unit s": [b1.real, b1.imag], "per unit t": [b2.real, b2.imag],
                       "vacuum |b|": b_vac}
        bm = np.abs(b0 + S[:, None] * b1 + T[None, :] * b2) / b_vac
        maps["|b_mk|/vacuum"] = bm
    j0 = int(np.argmin(np.abs(T)))
    i0 = int(np.argmin(np.abs(S)))
    res["along s at t = 0"] = {k: v[:, j0].tolist() for k, v in maps.items()}
    res["along t at s = 0"] = {k: v[i0, :].tolist() for k, v in maps.items()}
    best = np.argmin(maps["mean/M_desc"], axis=1)
    res["min over t, per s (mean)"] = {"t": T[best].tolist(), "mean/M_desc": maps["mean/M_desc"][np.arange(len(S)), best].tolist()}
    tag = os.path.basename(out) + f"_plane_{name}.npz"
    for d_ in [os.path.dirname(out) or "."] + ([mirror] if mirror else []):
        np.savez(os.path.join(d_, tag), s=S, t=T, **{k.replace("/", "_per_").replace("|", ""): v for k, v in maps.items()})
    sm = res["along s at t = 0"]["mean/M_desc"]
    log(f"[plane] {name}: fit check {fit_err:.1e}; |d1|/|z0| {res['|d1|/|z0|']:.3f}, |d2|/|z0| {res['|d2|/|z0|']:.3f}; "
        f"mean/M_desc at (0,0) {maps['mean/M_desc'][i0, j0]:.4f}; along s (t=0): "
        + " ".join(f"s{S[k]:.2f}:{sm[k]:.3g}" for k in range(0, len(S), max(1, len(S) // 8)))
        + f"; min over the plane {maps['mean/M_desc'].min():.4f} at (s, t) = "
        + "({:.2f}, {:.2f})".format(*[(S[a], T[b]) for a, b in [np.unravel_index(np.argmin(maps['mean/M_desc']), maps['mean/M_desc'].shape)]][0])
        + f"; {time.time() - t0:.0f} s")
    if bmk is not None:
        log(f"[plane] {name}: b_mk/vacuum at (0,0) {abs(b0) / b_vac:.4f}, per unit s {abs(b1) / b_vac:.4f}, per unit t {abs(b2) / b_vac:.4f}")
    return res
