"""Force and misfit along straight lines in current space, exactly.

On the fixed geometry J is linear in (u, I), B = B_ext + B_plasma(u, I) is affine and the
eliminated pressure is a fixed linear projection (I - P G^-1 P^T) of a = sw (J x B). Along the
segment (u, I)(t) = (u_a, I_a) + t ((u_b, I_b) - (u_a, I_a)) therefore

    a(t) = a0 + t a1 + t^2 a2,  a0 = sw J_a x B_a,  a1 = sw (dJ x B_a + J_a x dB),  a2 = sw dJ x dB
    f(t) = f0 + t f1 + t^2 f2   (f_k = projection of a_k, with the state's refinement step)

so two field evaluations give M(t) = sum w |F| / norm and the misfit |J(t) - J_fit|^2 / 2 rms^2
for every t. The endpoints are checked against FixedAnchor.state where I = I_DESC.
Segments: name=A:B[:tmax], A and B one of  zero (u = 0, I = 0), JI (u = 0, I = I_DESC),
fit (J_fit), or a state name given by --state name=path.npz.
usage: python line_scan_fixed.py --LMN 48 32 18 --frac 40 --state a005=s.npz --seg fit:zero ... --out x.npz
"""
import argparse
import os
import sys
import time

import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.join(ROOT, "nest"))
sys.path.insert(0, os.path.join(ROOT, "scripts"))
from desc import set_device  # noqa: E402

set_device(os.environ.get("NEST_DEVICE", "cpu"))
import jax  # noqa: E402
import jax.numpy as jnp  # noqa: E402

jax.config.update("jax_enable_x64", True)
import fit_cache  # noqa: E402
import meanforce_fixed as mff  # noqa: E402
from boundary_potential import BoundaryPotential  # noqa: E402
from desc_input import DESCInput  # noqa: E402
from force import ForceResidual  # noqa: E402
from free_boundary import _chunked, reference  # noqa: E402
from objective import Fields  # noqa: E402
from optimize_fb import annulus  # noqa: E402
from pressure import Pressure  # noqa: E402

D = os.path.join(ROOT, "desc_input", "vmec_free_boundary", "NCSX")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--eq", default=os.path.join(D, "NCSX_fb_L16M16N12.h5"))
    ap.add_argument("--coils", default=os.path.join(D, "NCSX_coils.h5"))
    ap.add_argument("--LMN", type=int, nargs=3, required=True)
    ap.add_argument("--colloc", type=float, default=1.5)
    ap.add_argument("--frac", type=float, default=40)
    ap.add_argument("--pad", type=float, default=0.02)
    ap.add_argument("--state", action="append", default=[])
    ap.add_argument("--seg", action="append", default=[], help="A:B[:tmax]")
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    t0 = time.time()
    inp = DESCInput.load(a.eq, field=a.coils)
    eq = inp.eq
    kw = {"L": a.LMN[0], "M": a.LMN[1], "N": a.LMN[2]}
    cur = fit_cache.fit(BoundaryPotential, inp, **kw)
    pre = fit_cache.fit(Pressure, inp, edge_order=2, **kw)
    b = cur.basis
    res = tuple(int(round(a.colloc * (u + v))) for u, v in ((b.L, eq.L), (b.M, eq.M), (b.N, eq.N)))
    ref = reference(inp, res, current=cur)
    norm = ref["norm"]
    box = annulus(inp, inp.a / a.frac, a.pad, None)
    box.locate_plasma_planes(inp, sym=bool(eq.sym), verbose=False)
    fr = ForceResidual(inp, box, resolution=res)
    fr.norm, fr.F_desc_vol = norm, ref["F_desc_vol"]
    if "B_ext" not in vars(fr):
        fr.B_ext = _chunked(inp.B_ext, fr.x)
    fields = Fields(fr, cur, pre, dtype=jnp.float64, cyl_stencil=True, mode_chunk=16, cyl_fused=True,
                    cyl_rblock=32, cyl_order=8)
    M_desc = ref["F_desc_vol"] / norm
    sw = jnp.asarray(fields.sw)
    w = sw**2
    u0 = jnp.asarray(cur.u)
    I0 = float(fields.I)
    Jfit = mff._J(fields, u0, I0)
    rmsJ = float(jnp.sqrt(jnp.sum(sw[:, None] ** 2 * Jfit**2)))
    G = mff.gram_P(fields, cur, pre, chunk=1024, log=lambda *x: None)
    A = mff.FixedAnchor(fields, cur, pre, u0, rmsJ, norm, G, log=print)
    Lg, n_p = A.Lg, A.n_p
    print(f"basis {b.L}/{b.M}/{b.N}, n_u {A.n_u}, points {A.n_pts}; M_desc {M_desc:.4e}; setup {time.time() - t0:.0f} s",
          flush=True)
    ends = {"zero": (jnp.zeros_like(u0), 0.0), "JI": (jnp.zeros_like(u0), I0), "fit": (u0, I0)}
    for spec in a.state:
        name, path = spec.split("=", 1)
        ends[name] = (jnp.asarray(np.load(path)["u"]), I0)

    proj = jax.jit(lambda x: mff._project(fields, sw, x, Lg, n_p)[0])

    def P2(x):  # the state's projection with its refinement step
        f = proj(x)
        return proj(f)

    def JB(u, I):
        return mff._J(fields, u, I), fields.B_ext + fields.B_plasma(u, I)

    tf = np.unique(np.concatenate([[0.0], np.logspace(-6, 0, 61), np.linspace(0, 1, 101)]))
    out = {"M_desc": M_desc}
    for spec in a.seg:
        parts = spec.split(":")
        A_, B_ = parts[0], parts[1]
        tmax = float(parts[2]) if len(parts) > 2 else 1.0
        t1 = time.time()
        (ua, Ia), (ub, Ib) = ends[A_], ends[B_]
        Ja, Ba = JB(ua, Ia)
        Jb, Bb = JB(ub, Ib)
        dJ, dB = Jb - Ja, Bb - Ba
        a0 = (sw[:, None] * jnp.cross(Ja, Ba)).ravel()
        a1 = (sw[:, None] * (jnp.cross(dJ, Ba) + jnp.cross(Ja, dB))).ravel()
        a2 = (sw[:, None] * jnp.cross(dJ, dB)).ravel()
        f0, f1, f2 = P2(a0), P2(a1), P2(a2)
        n = Ja.shape[0]
        ts = tf * tmax if tmax <= 1 else np.unique(np.concatenate([tf, np.linspace(0, tmax, 6 * int(np.ceil(tmax)) * 10 + 1)]))
        Ms, mis, rms = [], [], []
        for t in ts:
            f = (f0 + t * f1 + t * t * f2).reshape(n, 3) / sw[:, None]
            absF = jnp.linalg.norm(f, axis=1)
            Ms.append(float(jnp.sum(w * absF) / norm) / M_desc)
            rms.append(float(jnp.sqrt(jnp.sum(w * absF**2)) / norm))
            g = sw[:, None] * (Ja + t * dJ - Jfit) / rmsJ
            mis.append(0.5 * float(jnp.sum(g * g)))
        Ms, mis = np.array(Ms), np.array(mis)
        chk = ""
        for tt, (uu, II) in ((0.0, (ua, Ia)), (1.0, (ub, Ib))):
            if II == I0:
                s = A.state(uu)
                k = int(np.argmin(np.abs(ts - tt)))
                chk += (f" t={tt:g}: scan M/M_desc {Ms[k]:.6e} vs state {float(s['M']) / M_desc:.6e}, misfit {mis[k]:.6e} "
                        f"vs {0.5 * float(jnp.sum(s['g'] ** 2)):.6e};")
        key = f"{A_}_{B_}"
        out[f"{key}_t"], out[f"{key}_M"], out[f"{key}_misfit"], out[f"{key}_rmsF"] = ts, Ms, mis, np.array(rms)
        sel = [0.0, 1e-4, 1e-3, 1e-2, 0.05, 0.1, 0.25, 0.5, 0.75, 1.0, 1.5, 2.0, 3.0]
        rows = "; ".join(f"t {x:g}: M {Ms[np.argmin(np.abs(ts - x))]:.4e}, mis {mis[np.argmin(np.abs(ts - x))]:.4e}"
                         for x in sel if x <= ts.max() + 1e-12)
        i_min = int(np.argmin(Ms))
        print(f"[seg {A_} -> {B_}] {time.time() - t1:.0f} s;{chk}\n    min M/M_desc {Ms[i_min]:.4e} at t {ts[i_min]:.4g} "
              f"(misfit {mis[i_min]:.4e})\n    {rows}", flush=True)
        np.savez(a.out, **out)
    print(f"saved {a.out}; total {time.time() - t0:.0f} s")


if __name__ == "__main__":
    main()
