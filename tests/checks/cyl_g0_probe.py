"""G0 probe: main's default Fields paths, run from a given nest source tree."""
import os, sys
src, out = sys.argv[1], sys.argv[2]
sys.path.insert(0, src)
import jax
jax.config.update("jax_enable_x64", True)
import jax.numpy as jnp
import numpy as np
from boundary_potential import BoundaryPotential
from desc_input import DESCInput
from force import ForceResidual
from grid import CartesianGrid
from objective import Fields
from pressure import Pressure
R = "/Users/singh/PycharmProjects/NEST/NEST-cyl/desc_input/"
inp = DESCInput.load(R + "precise_QA_free_boundary_smooth_J/precise_QA_free_boundary_smooth_J_edgeJ_r0.3.h5",
                     field=R + "precise_QA_free_boundary/precise_QA_helical_coils.h5")
cur = BoundaryPotential.fit(inp.eq, factors=(1, 1, 1))
pre = Pressure.fit(inp.eq, factors=(1, 1, 1), edge_order=2)
box = CartesianGrid.periodic_box(inp, inp.a / 6, odd=True).locate_plasma(inp)
fr = ForceResidual(inp, box)
res = {"Bpl_numpy": fr.B_plasma(cur)}
rng = np.random.default_rng(0)
for kernel in ("hockney", "truncated"):
    for dt in (jnp.float32, jnp.float64):
        f = Fields(fr, cur, pre, kernel=kernel, dtype=dt)
        x0 = jnp.asarray(f.x0)
        o = f.compute(x0)
        v = jnp.asarray(rng.standard_normal(f.n))
        lin = jax.jit(lambda fl, vv, J0, B0: fl._tangent(vv, J0, B0)["F"])
        jv = lin(f, v, o["J"], o["B"])
        w = jnp.asarray(rng.standard_normal(jv.shape))
        _, vjp = jax.vjp(lambda vv: f._tangent(vv, o["J"], o["B"])["F"], v)
        tag = f"{kernel}_{np.dtype(dt).name}"
        res[tag + "_F"], res[tag + "_Bpl"] = np.asarray(o["F"]), np.asarray(o["B_plasma"])
        res[tag + "_Jv"], res[tag + "_JTw"] = np.asarray(jv), np.asarray(vjp(w)[0])
np.savez(out, **res)
print("saved", out, len(res), "arrays")
