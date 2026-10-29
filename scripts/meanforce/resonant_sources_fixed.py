"""Which currents set the resonant field at the rational surfaces? Source decomposition of the
resonant b^rho harmonics of fixed-boundary mean-force states.

b^rho = B . grad rho / B . grad zeta on DESC's surfaces (each piece's B . grad rho over the state's
total B . grad zeta, so the pieces add up exactly), harmonics (m, n) in (theta*, zeta) as in
shielding_fixed.py. B = B_ext (coils) + B_plasma, and B_plasma is linear in the current on the
flux grid: J_flux -> to_box -> annulus sources -> modal field (``Fields.cyl_field_modes``). So any
split of J_flux gives a split of b^rho that sums exactly to the total, computed with the solver's
own field operator. Per state the split is
  J = J_fit + dJ,  each masked into radial bands of the flux grid's rho (``--edges``), and
  each band split into its parallel part (J.b) b and perpendicular part, b = B / |B| of the
  state's own field (J_fit's split uses the start state's field).
Saved per state: the harmonic profiles (n_rho, harm, 2) of b^rho from the coils, from each
(J_fit | dJ) x band x (par | perp) piece, and from the total, plus the gates
|sum of pieces - total| / |total| (resonant harmonics). Band masks act before the box
interpolation, so a piece's current is not divergence-free; its field is still well defined
(the operator is linear) and the pieces sum to the total.
usage: python resonant_sources_fixed.py --LMN 48 32 18 --frac 40 --state a0.05=s.npz ... --out x.npz
"""
import argparse
import os
import sys
import time

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import shielding_fixed as sf  # noqa: E402  (setup, harmonics, resonant set)

jnp, jax = sf.jnp, sf.jax
from objective import VECTOR_PARITY  # noqa: E402

EDGES = (0.0, 0.3, 0.5, 0.6, 0.65, 0.7, 0.73, 0.76, 0.78, 0.79, 0.80, 0.81, 0.82, 0.83, 0.84,
         0.86, 0.88, 0.91, 0.95, 1.0001)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--eq", default=os.path.join(sf.D, "NCSX_fb_L16M16N12.h5"))
    ap.add_argument("--coils", default=os.path.join(sf.D, "NCSX_coils.h5"))
    ap.add_argument("--LMN", type=int, nargs=3, default=None)
    ap.add_argument("--factors", type=float, nargs=3, default=(1, 1, 1))
    ap.add_argument("--colloc", type=float, default=1.5)
    ap.add_argument("--frac", type=float, default=40)
    ap.add_argument("--pad", type=float, default=0.02)
    ap.add_argument("--state", action="append", default=[], help="name=path.npz (u); repeatable")
    ap.add_argument("--rho", type=float, nargs=3, default=(0.45, 0.95, 101), help="evaluation rho lo hi n")
    ap.add_argument("--nt", type=int, default=96)
    ap.add_argument("--nz", type=int, default=32)
    ap.add_argument("--edges", type=float, nargs="+", default=EDGES, help="band edges in rho")
    ap.add_argument("--chunk", type=int, default=200_000)
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    t0 = time.time()
    inp = sf.DESCInput.load(a.eq, field=a.coils)
    eq = inp.eq
    kw = {"L": a.LMN[0], "M": a.LMN[1], "N": a.LMN[2]} if a.LMN else {"factors": tuple(a.factors)}
    cur = sf.fit_cache.fit(sf.BoundaryPotential, inp, **kw)
    pre = sf.fit_cache.fit(sf.Pressure, inp, edge_order=2, **kw)
    b = cur.basis
    res = tuple(int(round(a.colloc * (u + v))) for u, v in ((b.L, eq.L), (b.M, eq.M), (b.N, eq.N)))
    ref = sf.reference(inp, res, current=cur)
    box = sf.annulus(inp, inp.a / a.frac, a.pad, None)
    box.locate_plasma_planes(inp, sym=bool(eq.sym), verbose=False)
    fr = sf.ForceResidual(inp, box, resolution=res)
    fr.norm, fr.F_desc_vol = ref["norm"], ref["F_desc_vol"]
    if "B_ext" not in vars(fr):
        fr.B_ext = sf._chunked(inp.B_ext, fr.x)
    fields = sf.Fields(fr, cur, pre, dtype=jnp.float64, cyl_stencil=True, mode_chunk=16, cyl_fused=True,
                       cyl_rblock=32, cyl_order=8)
    print(f"{inp}\nbasis {b.L}/{b.M}/{b.N}, annulus {box}; setup {time.time() - t0:.0f} s", flush=True)

    # evaluation surfaces (as shielding_fixed)
    rho = np.linspace(*a.rho[:2], int(a.rho[2]))
    th = 2 * np.pi * np.arange(a.nt) / a.nt
    ze = 2 * np.pi / eq.NFP * np.arange(a.nz) / a.nz
    g = sf.TensorGrid(rho, th, ze, eq.NFP)
    shp = g.shape
    RZ = sf.Separable([eq.R_basis, eq.Z_basis], [eq.R_lmn, eq.Z_lmn], eq.NFP)
    geo = sf.TensorGeometry(RZ, g)
    R, Z = np.asarray(geo.R).ravel(), np.asarray(geo.Z).ravel()
    nodes = g.nodes()
    phi = nodes[:, 2]
    one, zero = jnp.ones(shp), jnp.zeros(shp)
    grad_rho = np.asarray(geo.gradient(one, zero, zero)).reshape(-1, 3)
    lam = inp.compute_tensor(["lambda", "lambda_t"], rho, th, ze)
    ths = (nodes[:, 1] + np.asarray(lam["lambda"]).ravel()).reshape(shp)
    jac = (1 + np.asarray(lam["lambda_t"]).ravel()).reshape(shp)
    zeta = phi.reshape(shp)
    iota, iota_r = inp.iota(rho)
    sg = int(np.sign(np.mean(iota)))
    HARM = [(m, sg * n) for m, n in sf.HARM0[: sf.N_RES]]

    def ext_cyl(Rq, Pq, Zq):
        X = np.column_stack([Rq * np.cos(Pq), Rq * np.sin(Pq), Zq])
        Bx = np.concatenate([np.asarray(inp.B_ext(X[i : i + a.chunk])) for i in range(0, len(X), a.chunk)])
        c, s = np.cos(Pq), np.sin(Pq)
        return np.column_stack([Bx[:, 0] * c + Bx[:, 1] * s, -Bx[:, 0] * s + Bx[:, 1] * c, Bx[:, 2]])

    Bext = ext_cyl(R, phi, Z)
    itp = sf.CylindricalInterpolator(fields.cyl_grid, np.column_stack([R, phi, Z]), dtype=jnp.float64, order=8)

    def brho_h(B, Bphi):  # b^rho of a piece: its B.grad rho over the TOTAL field's B.grad zeta (additive)
        br = (np.sum(B * grad_rho, 1) / (Bphi / R)).reshape(shp)
        return sf.harmonics(br, ths, jac, zeta, HARM)

    def modes_of(Jf):  # B_plasma modal slabs of a current on the flux grid
        Jr = fields._cyl_rows(fields.to_box(Jf.astype(fields.cyl_dtype), VECTOR_PARITY))
        return fields.solver.field_modes_F(*fields._cyl_sources(Jr))

    def b_of(Jf):
        Bp, Bz = modes_of(jnp.asarray(Jf))
        return np.asarray(itp(Bp, Bz))

    # the flux grid's nodes and the field there (for the parallel / perpendicular split)
    TC = fields.J_flux
    fg = TC.grid
    Rf, Zf = np.asarray(TC.geo.R).ravel(), np.asarray(TC.geo.Z).ravel()
    Pf = np.broadcast_to(fg.zeta[None, None, :], fg.shape).ravel()
    Bext_f = ext_cyl(Rf, Pf, Zf)
    itp_f = [sf.CylindricalInterpolator(fields.cyl_grid, np.column_stack([Rf[i:i + a.chunk], Pf[i:i + a.chunk], Zf[i:i + a.chunk]]),
                                        dtype=jnp.float64, order=8) for i in range(0, len(Rf), a.chunk)]

    def bhat_f(u):
        Bp, Bz = fields.cyl_field_modes(jnp.asarray(u))
        B = np.concatenate([np.asarray(f(Bp, Bz)) for f in itp_f]) + Bext_f
        return (B / np.linalg.norm(B, axis=1, keepdims=True)).reshape(fg.shape + (3,))

    edges = np.asarray(a.edges)
    band = np.digitize(fg.rho, edges) - 1  # band index per radial node
    nb = len(edges) - 1
    print(f"flux grid {fg.shape} ({fg.size} nodes), {nb} bands; eval grid {shp}; "
          f"prep {time.time() - t0:.0f} s", flush=True)

    out = dict(rho=rho, iota=iota, iota_r=iota_r, harm=np.array(HARM), edges=edges)
    u_fit = np.asarray(cur.u)
    I = fields.I
    Jfit = np.asarray(TC(jnp.asarray(u_fit), I))
    bh_start = bhat_f(u_fit)
    states = [("start", u_fit)] + [(s.split("=", 1)[0], np.load(s.split("=", 1)[1])["u"]) for s in a.state]
    k53 = HARM.index((5, sg * 3))
    for name, u in states:
        t1 = time.time()
        Jtot = np.asarray(TC(jnp.asarray(u), I))
        Bp, Bz = fields.cyl_field_modes(jnp.asarray(u))
        Bpl = np.asarray(itp(Bp, Bz))
        Bphi = Bpl[:, 1] + Bext[:, 1]  # the state's total B_phi: the common denominator
        tot = brho_h(Bpl, Bphi)  # plasma part of b^rho, solver's own path
        out[f"{name}_plasma"] = tot
        out[f"{name}_coils"] = brho_h(Bext, Bphi)
        out[f"{name}_total"] = brho_h(Bpl + Bext, Bphi)
        parts = [("fit", Jfit, bh_start)] if name == "start" else [("fit", Jfit, bh_start), ("dJ", Jtot - Jfit, bhat_f(u))]
        acc = np.zeros_like(tot)
        for tag, Jp, bh in parts:
            par = np.sum(Jp * bh, -1, keepdims=True) * bh
            H = np.zeros((nb, 2) + tot.shape)
            for k in range(nb):
                msk = (band == k).astype(float)[:, None, None, None]
                if not msk.any():
                    continue
                H[k, 0] = brho_h(b_of(par * msk), Bphi)
                H[k, 1] = brho_h(b_of((Jp - par) * msk), Bphi)
            out[f"{name}_{tag}_bands"] = H  # (band, par|perp, rho, harm, cos|sin)
            acc += H.sum((0, 1))
        rel = np.linalg.norm(acc[:, :, :] - tot[:, :, :]) / np.linalg.norm(tot)
        r53 = np.linalg.norm(acc[:, k53] - tot[:, k53]) / np.linalg.norm(tot[:, k53])
        out[f"{name}_gate"] = np.array([rel, r53])
        print(f"{name}: {time.time() - t1:.0f} s; gate |sum pieces - total|/|total|: all resonant {rel:.2e}, "
              f"(5,{sg * 3}) {r53:.2e}", flush=True)
    os.makedirs(os.path.dirname(os.path.abspath(a.out)), exist_ok=True)
    np.savez(a.out, **out)
    print(f"saved {a.out}; total {time.time() - t0:.0f} s")


if __name__ == "__main__":
    main()
