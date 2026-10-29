"""Hamada / Newcomb solvability test along field lines of fixed-boundary states.

For nested surfaces with p' != 0 on a rational surface iota = n/m, the magnetic differential
equation B.grad(sigma) = -div(B x grad p / B^2) has a bounded solution only if
U = closed-line integral of dl/B is the same on every closed line of that surface. Its violation
is the resonant (Pfirsch-Schlueter) drive. This script traces lines with the same spline table as
poincare_fixed.py and accumulates V(phi) = int dphi / B^phi (= int dl / B) along each line. It
records V and (R, Z) at every crossing of phi = pi/NFP. The closed-orbit sum
W_k = V_{k+P} - V_k (P periods = one closed orbit, e.g. 15 for 3/5 with NFP 3) then varies with
the starting angle only through the resonant part.
usage: python hamada_lines_fixed.py --LMN 48 32 18 --frac 40 --state name=state.npz ... --out x.npz
"""
import os
import sys
import time

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import poincare_fixed as pf  # noqa: E402  (same setup, table, gate)

jnp, jax = pf.jnp, pf.jax


def trace_V(Rs, phis, Zs, B, R0, Z0, NFP, n_periods, steps):
    """RK4 in phi for (R, Z, V), dV/dphi = R / B_phi. Returns R, Z, V at phi = pi/NFP + k period."""
    from scipy.ndimage import map_coordinates, spline_filter
    Bp = np.concatenate([B[:, -3:], B, B[:, :3]], axis=1)
    coef = [spline_filter(Bp[..., k], order=3, mode="nearest") for k in range(3)]
    dR, dZ, dphi = Rs[1] - Rs[0], Zs[1] - Zs[0], phis[1] - phis[0]
    period = 2 * np.pi / NFP

    def rhs(R, Z, phi):
        q = np.stack([(R - Rs[0]) / dR, np.full_like(R, np.mod(phi, period) / dphi + 3), (Z - Zs[0]) / dZ])
        BR, Bphi, BZ = (map_coordinates(c, q, order=3, prefilter=False, mode="nearest") for c in coef)
        return R * BR / Bphi, R * BZ / Bphi, R / Bphi

    R, Z, V = np.array(R0, float), np.array(Z0, float), np.zeros(len(R0))
    h = period / steps
    oR, oZ, oV = [], [], []
    alive = np.ones(len(R), bool)
    for k in range(n_periods):
        for i in range(steps):
            phi = (k * steps + i) * h
            k1 = rhs(R, Z, phi)
            k2 = rhs(R + h / 2 * k1[0], Z + h / 2 * k1[1], phi + h / 2)
            k3 = rhs(R + h / 2 * k2[0], Z + h / 2 * k2[1], phi + h / 2)
            k4 = rhs(R + h * k3[0], Z + h * k3[1], phi + h)
            R = R + h / 6 * (k1[0] + 2 * k2[0] + 2 * k3[0] + k4[0])
            Z = Z + h / 6 * (k1[1] + 2 * k2[1] + 2 * k3[1] + k4[1])
            V = V + h / 6 * (k1[2] + 2 * k2[2] + 2 * k3[2] + k4[2])
            if i == steps // 2 - 1:
                alive &= (R > Rs[1]) & (R < Rs[-2]) & (Z > Zs[1]) & (Z < Zs[-2])
                oR.append(np.where(alive, R, np.nan))
                oZ.append(np.where(alive, Z, np.nan))
                oV.append(np.where(alive, V, np.nan))
        if k % 50 == 0:
            print(f"  period {k}/{n_periods}: {alive.sum()} alive", flush=True)
    return np.array(oR), np.array(oZ), np.array(oV)


def main():
    ap = pf.argparse.ArgumentParser()
    ap.add_argument("--eq", default=os.path.join(pf.D, "NCSX_fb_L16M16N12.h5"))
    ap.add_argument("--coils", default=os.path.join(pf.D, "NCSX_coils.h5"))
    ap.add_argument("--LMN", type=int, nargs=3, default=None)
    ap.add_argument("--colloc", type=float, default=1.5)
    ap.add_argument("--frac", type=float, default=40)
    ap.add_argument("--pad", type=float, default=0.02)
    ap.add_argument("--state", action="append", default=[], help="name=state.npz; 'start' is always included")
    ap.add_argument("--rho", type=float, nargs=3, default=(0.48, 0.90, 141), help="seed rho lo hi n (theta = 0)")
    ap.add_argument("--grid", type=int, nargs=3, default=(161, 64, 161))
    ap.add_argument("--periods", type=int, default=400)
    ap.add_argument("--steps", type=int, default=64)
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    t0 = time.time()
    inp = pf.DESCInput.load(a.eq, field=a.coils)
    eq = inp.eq
    kw = {"L": a.LMN[0], "M": a.LMN[1], "N": a.LMN[2]} if a.LMN else {"factors": (1, 1, 1)}
    cur = pf.fit_cache.fit(pf.BoundaryPotential, inp, **kw)
    pre = pf.fit_cache.fit(pf.Pressure, inp, edge_order=2, **kw)
    b = cur.basis
    res = tuple(int(round(a.colloc * (u + v))) for u, v in ((b.L, eq.L), (b.M, eq.M), (b.N, eq.N)))
    ref = pf.reference(inp, res, current=cur)
    box = pf.annulus(inp, inp.a / a.frac, a.pad, None)
    box.locate_plasma_planes(inp, sym=bool(eq.sym), verbose=False)
    fr = pf.ForceResidual(inp, box, resolution=res)
    fr.norm, fr.F_desc_vol = ref["norm"], ref["F_desc_vol"]
    if "B_ext" not in vars(fr):
        fr.B_ext = pf._chunked(inp.B_ext, fr.x)
    fields = pf.Fields(fr, cur, pre, dtype=jnp.float64, cyl_stencil=True, mode_chunk=16, cyl_fused=True,
                       cyl_rblock=32, cyl_order=8)
    print(f"{inp}\nbasis {b.L}/{b.M}/{b.N}; setup {time.time() - t0:.0f} s", flush=True)
    us = {"start": np.asarray(cur.u)}
    for spec in a.state:
        name, path = spec.split("=", 1)
        us[name] = np.load(path)["u"]
    ra, za = fields.cyl_grid.r_axis, fields.cyl_grid.z_axis
    Rs = np.linspace(ra.a + 1e-4, ra.b - 1e-4, a.grid[0])
    Zs = np.linspace(za.a + 1e-4, za.b - 1e-4, a.grid[2])
    phis = 2 * np.pi / eq.NFP * np.arange(a.grid[1]) / a.grid[1]
    RZ = pf.Separable([eq.R_basis, eq.Z_basis], [eq.R_lmn, eq.Z_lmn], eq.NFP)
    rho = np.concatenate([[0.005], np.linspace(*a.rho[:2], int(a.rho[2]))])  # first line ~ axis
    st = np.asarray(RZ.evaluate(jnp.asarray(rho), jnp.zeros(len(rho)), jnp.zeros(len(rho))))
    out = {"seed_rho": rho}
    for name, u in us.items():
        t1 = time.time()
        B = pf.table(fields, inp, u, Rs, phis, Zs)
        chk = pf.check(fields, Rs, phis, Zs, B, u, eq.NFP)
        print(f"{name}: table in {time.time() - t1:.0f} s; gate rel rms {chk[0]:.2e}, max {chk[1]:.2e}", flush=True)
        t1 = time.time()
        R, Z, V = trace_V(Rs, phis, Zs, B, st[:, 0], st[:, 1], eq.NFP, a.periods, a.steps)
        print(f"{name}: traced {len(rho)} lines x {a.periods} periods in {time.time() - t1:.0f} s; "
              f"{int(np.isnan(R[-1]).sum())} lost", flush=True)
        out.update({f"{name}_R": R, f"{name}_Z": Z, f"{name}_V": V, f"{name}_check": np.array(chk)})
    os.makedirs(os.path.dirname(os.path.abspath(a.out)), exist_ok=True)
    np.savez(a.out, **out)
    print(f"saved {a.out}; total {time.time() - t0:.0f} s")


if __name__ == "__main__":
    main()
