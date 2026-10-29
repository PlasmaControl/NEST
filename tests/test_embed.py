"""E1: J of a low-resolution BoundaryPotential state equals J of its zero-padded embedding
in a higher-resolution basis, pointwise (same functions, same I, same geometry)."""
import os, sys, time
import numpy as np
import jax
jax.config.update("jax_enable_x64", True)
import jax.numpy as jnp
HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, "..", "nest"))
from desc_input import DESCInput  # noqa: E402
from boundary_potential import BoundaryPotential  # noqa: E402
import fit_cache  # noqa: E402
import meanforce_fixed as mff  # noqa: E402
D = os.path.join(HERE, "..", "desc_input", "vmec_free_boundary", "NCSX")
t0 = time.time()
inp = DESCInput.load(os.path.join(D, "NCSX_fb_L16M16N12.h5"), field=os.path.join(D, "NCSX_coils.h5"))
lo = fit_cache.fit(BoundaryPotential, inp, factors=(0.5, 0.5, 0.5))
hi = fit_cache.fit(BoundaryPotential, inp, factors=(0.75, 0.75, 0.75))
print("lo", lo.basis.L, lo.basis.M, lo.basis.N, lo.n_free, "| hi", hi.basis.L, hi.basis.M, hi.basis.N, hi.n_free)
rng = np.random.default_rng(0)
rho, th, ze = (jnp.asarray(rng.uniform(0.01, 0.99, 400)), jnp.asarray(rng.uniform(0, 2 * np.pi, 400)),
               jnp.asarray(rng.uniform(0, 2 * np.pi, 400)))
fl, fh = lo.current_function(), hi.current_function()
ok = True
for name, u in [("fit", lo.u), ("random", rng.standard_normal(lo.n_free) * np.abs(lo.u).max())]:
    Jl = jnp.stack(fl(jnp.asarray(u), rho, th, ze)); Jh = jnp.stack(fh(jnp.asarray(mff.embed(lo, hi, u)), rho, th, ze))
    e = float(jnp.max(jnp.abs(Jl - Jh)) / jnp.max(jnp.abs(Jl)))
    print(f"E1 {name}: max rel |J_lo - J_hi(embed)| = {e:.2e}"); ok &= e < 1e-12
print("ALL PASSED" if ok else "FAILED", f"({time.time() - t0:.0f} s)")
