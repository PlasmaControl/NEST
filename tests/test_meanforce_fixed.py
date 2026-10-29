"""Gates of nest/meanforce_fixed.py on a tiny NCSX model (CPU, ~minutes).

G1 slice-built G = P^T P against the explicit columns;  G2 slice-built H_J against the
explicit J table;  G3 implicit projection f against the explicit thin-QR projection, and the
Galerkin identity;  G4 J v against central differences of (g, z);  G5 adjoint;  G6 the
majorizer gradient: grad c_F from J^T against differences of M.
"""

import os
import sys
import time

import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, "nest"))
sys.path.insert(0, os.path.join(ROOT, "scripts"))

from desc import set_device  # noqa: E402

set_device(os.environ.get("NEST_DEVICE", "cpu"))
import jax  # noqa: E402
import jax.numpy as jnp  # noqa: E402

jax.config.update("jax_enable_x64", True)

import fit_cache  # noqa: E402
import meanforce as mf  # noqa: E402
import meanforce_fixed as mff  # noqa: E402
from boundary_potential import BoundaryPotential  # noqa: E402
from desc_input import DESCInput  # noqa: E402
from force import ForceResidual  # noqa: E402
from free_boundary import _chunked, reference  # noqa: E402
from objective import Fields  # noqa: E402
from optimize_fb import annulus  # noqa: E402
from pressure import Pressure  # noqa: E402

D = os.path.join(ROOT, "desc_input", "vmec_free_boundary", "NCSX")
FAC = tuple(float(x) for x in os.environ.get("FAC", "0.5 0.5 0.5").split())
FRAC = float(os.environ.get("FRAC", "8"))


def rel(a, b):
    a, b = np.asarray(a), np.asarray(b)
    return float(np.linalg.norm(a - b) / max(np.linalg.norm(b), 1e-300))


def main():
    t0 = time.time()
    inp = DESCInput.load(os.path.join(D, "NCSX_fb_L16M16N12.h5"),
                         field=os.path.join(D, "NCSX_coils.h5"))
    eq = inp.eq
    cur = fit_cache.fit(BoundaryPotential, inp, factors=FAC)
    pre = fit_cache.fit(Pressure, inp, factors=FAC, edge_order=2)
    b = cur.basis
    res = tuple(int(round(1.5 * (u + v))) for u, v in ((b.L, eq.L), (b.M, eq.M), (b.N, eq.N)))
    ref = reference(inp, res, current=cur)
    norm = ref["norm"]
    M_desc = ref["F_desc_vol"] / norm
    box = annulus(inp, inp.a / FRAC, 0.02)
    box.locate_plasma_planes(inp, sym=bool(eq.sym), verbose=False)
    fr = ForceResidual(inp, box, resolution=res)
    fr.norm, fr.F_desc_vol = norm, ref["F_desc_vol"]
    if "B_ext" not in vars(fr):
        fr.B_ext = _chunked(inp.B_ext, fr.x)
    fields = Fields(fr, cur, pre, dtype=jnp.float64, cyl_stencil=True, mode_chunk=16,
                    cyl_fused=True, cyl_rblock=32, cyl_order=8)
    n_u, n_p, n = fields.n_u, fields.n_p, len(fields.phi)
    print(f"basis {b.L}/{b.M}/{b.N}: n_u {n_u}, n_p {n_p}, points {n}, {box}; "
          f"setup {time.time() - t0:.0f} s", flush=True)
    sw = jnp.asarray(fields.sw)
    ok = {}

    # G1: G by slices vs explicit
    G = mff.gram_P(fields, cur, pre, chunk=128)
    Pcols = jax.vmap(lambda e: mff._Pmul(fields, sw, e))(jnp.eye(n_p))  # (n_p, 3n)
    G_ex = Pcols @ Pcols.T
    ok["G1 G slices vs explicit"] = rel(G, G_ex)

    # G2: H_J by slices vs explicit
    u0 = jnp.asarray(cur.u)
    Jfit = mff._J(fields, u0, fields.I)
    rmsJ = float(jnp.sqrt(jnp.sum(sw[:, None] ** 2 * Jfit**2)))
    HJ = mff.gram_J(fields, cur, pre, rmsJ, chunk=128)
    Acols = jax.vmap(lambda e: (sw[:, None] * mff._J(fields, e, 0.0)).ravel() / rmsJ)(
        jnp.eye(n_u))
    ok["G2 H_J slices vs explicit"] = rel(HJ, Acols @ Acols.T)
    del Acols

    # G3: projection vs thin QR
    A = mff.FixedAnchor(fields, cur, pre, u0, rmsJ, norm, G)
    rng = np.random.default_rng(0)
    z = u0 * (1 + 0.01 * jnp.asarray(rng.normal(size=n_u)))
    s = A.state(z)
    a_vec = (sw[:, None] * jnp.cross(s["J"], s["B"])).ravel()
    Q, R = jnp.linalg.qr(Pcols.T)
    f_qr = a_vec - Q @ (Q.T @ a_vec)
    ok["G3 f implicit vs QR"] = rel(s["f"], f_qr)
    ok["G3 Galerkin (max rel |P^T f|)"] = float(s["galerkin"])
    M_qr = float(jnp.sum(sw**2 * jnp.linalg.norm(f_qr.reshape(n, 3) / sw[:, None], axis=1))) / norm
    ok["G3 M implicit vs QR"] = abs(float(s["M"]) / M_qr - 1)
    print(f"  M/M_desc at z: {float(s['M']) / M_desc:.4f}; cond(G) {A.p_cond:.2e}", flush=True)

    # G4: J v vs central differences
    W = mf.Weights(0.7 * M_desc, 1.0, 1.0)
    rows = mff.FixedRows(A, s, W, norm)
    v = jnp.asarray(rng.normal(size=n_u)) * float(jnp.linalg.norm(u0)) / np.sqrt(n_u) * 1e-3
    dg, dz = rows.lin(v)
    h = 1e-3
    sp, sm = A.state(z + h * v), A.state(z - h * v)
    dg_fd = (sp["g"] - sm["g"]) / (2 * h)
    dz_fd = (sp["z"].ravel() - sm["z"].ravel()) / (2 * h)
    ok["G4 dg vs FD"] = rel(dg, dg_fd)
    ok["G4 dz vs FD"] = rel(dz, dz_fd)

    # G5: adjoint
    wg = jnp.asarray(rng.normal(size=3 * n))
    wz = jnp.asarray(rng.normal(size=3 * n))
    lhs = float(dg @ wg + dz @ wz)
    rhs = float(v @ rows.linT(wg, wz))
    ok["G5 adjoint"] = abs(lhs - rhs) / max(abs(lhs), 1e-300)

    # G6: grad c_F vs differences of c_F
    c_fd = (W.constraints(sp)[0] - W.constraints(sm)[0]) / (2 * h)
    ok["G6 grad c_F . v vs FD"] = abs(float(rows.grad_cF @ v) / c_fd - 1)

    print("\n".join(f"  {k}: {val:.3e}" for k, val in ok.items()))
    tol = {"G1": 1e-12, "G2": 1e-12, "G3 f": 1e-9, "G3 G": 1e-10, "G3 M": 1e-9, "G4": 1e-5,
           "G5": 1e-10, "G6": 1e-4}
    bad = [k for k, val in ok.items() if val > next(t for p, t in tol.items() if k.startswith(p))]
    print("FAILED: " + ", ".join(bad) if bad else "ALL PASSED", f"({time.time() - t0:.0f} s)")


if __name__ == "__main__":
    main()
