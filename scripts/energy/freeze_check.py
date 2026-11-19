"""Check of meanforce_fixed.frozen_n0_block (CPU): every feasible n = 0 direction (C^T du = 0) has at most tau of its
current norm (volume-weighted |J|^2 on the collocation points) outside the band, measured by direct J evaluation; and
the pinned rows are well conditioned. usage: python scripts/energy/freeze_check.py --LMN 12 12 2 --band 0.3 0.9 --tau 0.05"""
import argparse, os, sys
import numpy as np
ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.join(ROOT, "nest")); sys.path.insert(0, os.path.join(ROOT, "scripts"))
from desc import set_device
set_device("cpu")
import jax, jax.numpy as jnp
jax.config.update("jax_enable_x64", True)
import fit_cache, meanforce_fixed as mff
from boundary_potential import BoundaryPotential
from desc_input import DESCInput
from force import ForceResidual
from free_boundary import reference
from objective import Fields
from optimize_fb import annulus
from pressure import Pressure
ap = argparse.ArgumentParser(); ap.add_argument("--eq", default=os.path.join(ROOT, "desc_input/tokamak_boundary_perturbed/bp2/L32N4/tokamak_ripple_C_L32M32N4_x0.001.h5"))
ap.add_argument("--LMN", type=int, nargs=3, default=(12, 12, 2)); ap.add_argument("--band", type=float, nargs=2, default=(0.3, 0.9))
ap.add_argument("--tau", type=float, default=0.05); ap.add_argument("--frac", type=float, default=20); ap.add_argument("--N-phi", type=int, default=13)
a = ap.parse_args()
inp0 = DESCInput.load(a.eq); eq0 = inp0.eq
kw = {"L": a.LMN[0], "M": a.LMN[1], "N": a.LMN[2]}
cur0 = fit_cache.fit(BoundaryPotential, inp0, **kw); pre0 = fit_cache.fit(Pressure, inp0, edge_order=2, **kw)
b = cur0.basis
res = tuple(int(round(1.5 * (u + v))) for u, v in ((b.L, eq0.L), (b.M, eq0.M), (b.N, eq0.N)))
ref = reference(inp0, res, current=cur0)
box = annulus(inp0, inp0.a / a.frac, 0.02, a.N_phi); box.locate_plasma_planes(inp0, sym=bool(eq0.sym), verbose=False, chunk=1 << 17)
fr = ForceResidual(inp0, box, resolution=res); fr.norm, fr.F_desc_vol = ref["norm"], ref["F_desc_vol"]; fr.B_ext = np.zeros((len(fr.x), 3))
fields = Fields(fr, cur0, pre0, dtype=jnp.float64, cyl_stencil=True, mode_chunk=16, cyl_fused=True, cyl_rblock=32, cyl_order=8)
Ct, info = mff.frozen_n0_block(fields, fr, cur0, a.band, a.tau)
idx = mff.n0_indices(cur0)
C = Ct[idx].T                                   # (k, m) on the n = 0 coefficients
_, sv, Vt = np.linalg.svd(C, full_matrices=True)
null = Vt[C.shape[0]:].T                        # feasible n = 0 directions (m - k)
print("pinned rows: singular values min/max %.3e / %.3e; feasible n=0 dimension %d (info free %d)" % (sv.min(), sv.max(), null.shape[1], info["free"]))
rho = np.asarray(fr.rtz[:, 0]); w = np.asarray(fields.sw) ** 2; out = (rho < a.band[0]) | (rho > a.band[1])
rng = np.random.default_rng(0); worst = 0.0
for t in range(20):
    du = np.zeros(cur0.n_free); du[idx] = null @ rng.standard_normal(null.shape[1])
    J = np.asarray(mff._J(fields, jnp.asarray(du), 0.0)); e = w * np.sum(J**2, 1)
    worst = max(worst, e[out].sum() / e.sum())
print("random feasible n=0 directions: worst outside-band norm fraction %.4f (tau %.3f)" % (worst, a.tau))
du = np.zeros(cur0.n_free); du[idx] = rng.standard_normal(len(idx)); J = np.asarray(mff._J(fields, jnp.asarray(du), 0.0)); e = w * np.sum(J**2, 1)
print("unconstrained random n=0 direction: outside fraction %.4f" % (e[out].sum() / e.sum()))
print("PASS" if worst <= a.tau * 1.0001 else "FAIL")
