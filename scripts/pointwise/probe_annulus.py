"""Cost of one annulus J -> B solve on NCSX: setup, B_plasma(u), its transpose, per h.

python scripts/pointwise/probe_annulus.py --fracs 20 40 80 [--factors 1.5 1.5 1.5]
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

import fit_cache  # noqa: E402
from boundary_potential import BoundaryPotential  # noqa: E402
from desc_input import DESCInput  # noqa: E402
from force import ForceResidual  # noqa: E402
from objective import Fields  # noqa: E402
from optimize_qa import make_box  # noqa: E402
from pressure import Pressure  # noqa: E402

jax.config.update("jax_enable_x64", True)
NC = os.path.join(ROOT, "desc_input", "vmec_free_boundary", "NCSX")

ap = argparse.ArgumentParser()
ap.add_argument("--fracs", type=int, nargs="+", default=[20, 40, 80])
ap.add_argument("--factors", type=float, nargs=3, default=(1.5, 1.5, 1.5))
ap.add_argument("--colloc", type=float, default=1.5)
a = ap.parse_args()
inp = DESCInput.load(os.path.join(NC, "NCSX_L12M12N8.h5"), field=os.path.join(NC, "NCSX_coils.h5"))
eq = inp.eq
t0 = time.time()
cur = fit_cache.fit(BoundaryPotential, inp, factors=tuple(a.factors))
pre = fit_cache.fit(Pressure, inp, factors=tuple(a.factors), edge_order=0)
b = cur.basis
res = tuple(int(round(a.colloc * (u + v))) for u, v in ((b.L, eq.L), (b.M, eq.M), (b.N, eq.N)))
print(f"fits {time.time() - t0:.0f} s; n_u {cur.n_free}, n_p {pre.basis.num_modes}, colloc {res}", flush=True)
for frac in a.fracs:
    t0 = time.time()
    box = make_box(inp, frac, "cylindrical", True)
    fr = ForceResidual(inp, box, resolution=res)
    fields = Fields(fr, cur, pre, dtype=jnp.float64, S_dtype=jnp.float64)
    t_set = time.time() - t0
    u = jnp.asarray(np.random.default_rng(0).standard_normal(cur.n_free))
    f = jax.jit(lambda u, fl: fl.B_plasma(u, 0.0))
    B = f(u, fields).block_until_ready()
    t1 = time.time(); B = f(u, fields).block_until_ready(); t_b = time.time() - t1
    fT = jax.jit(lambda w, fl: jax.linear_transpose(lambda v: fl.B_plasma(v, 0.0), u)(w)[0])
    g = fT(B, fields).block_until_ready()
    t1 = time.time(); g = fT(B, fields).block_until_ready(); t_t = time.time() - t1
    fv = jax.jit(jax.vmap(lambda u, fl: fl.B_plasma(u, 0.0), in_axes=(0, None)))
    U = jnp.asarray(np.random.default_rng(1).standard_normal((8, cur.n_free)))
    fv(U, fields).block_until_ready()
    t1 = time.time(); fv(U, fields).block_until_ready(); t_v = time.time() - t1
    mem = jax.devices()[0].memory_stats() or {}
    print(f"a/{frac}: N_phi {box.N_phi}, r x z {box.r_axis.n + 1} x {box.z_axis.n + 1}, setup "
          f"{t_set:.0f} s; B_plasma {t_b:.2f} s, transpose {t_t:.2f} s, vmap 8 {t_v:.2f} s "
          f"({t_v / 8:.2f} s/col -> G {t_v / 8 * cur.n_free / 60:.0f} min); peak "
          f"{mem.get('peak_bytes_in_use', 0) / 1e9:.1f} GB", flush=True)
    del fields, fr, box, f, fT, fv
    jax.clear_caches()
