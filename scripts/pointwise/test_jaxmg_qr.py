"""jaxmg.qr on the visible GPUs against jnp.linalg.qr: single process (all devices) or
one process per GPU (srun, MULTI=1)."""
import os
import time

import jax

if os.environ.get("MULTI"):
    vis = [v for v in os.environ.get("CUDA_VISIBLE_DEVICES", "").split(",") if v]
    loc = int(os.environ.get("SLURM_LOCALID", 0))
    jax.distributed.initialize(local_device_ids=[loc] if len(vis) > 1 else None)
jax.config.update("jax_enable_x64", True)
import jax.numpy as jnp  # noqa: E402
import numpy as np  # noqa: E402
from jax.sharding import NamedSharding, PartitionSpec as P  # noqa: E402

import jaxmg  # noqa: E402

nd = jax.device_count()
mesh = jax.make_mesh((nd, 1), ("pr", "pc"))
m, n = 40000, 3000
nc = -(-(n + 1) // (2 * nd)) * (2 * nd)  # columns padded to the process grid
A_np = np.zeros((m, nc))
A_np[:, : n + 1] = np.random.default_rng(0).standard_normal((m, n + 1))
A = jax.make_array_from_callback((m, nc), NamedSharding(mesh, P("pr", "pc")),
                                 lambda idx: A_np[idx])
t0 = time.time()
Q, R = jaxmg.qr(A, 256, mesh=mesh, donate=False)
R = jax.block_until_ready(R)
t1 = time.time()
Q, R = jaxmg.qr(A, 256, mesh=mesh, donate=False)
R = np.asarray(jax.device_get(jax.experimental.multihost_utils.process_allgather(R, tiled=True)
                              if jax.process_count() > 1 else R))
t2 = time.time()
if jax.process_index() == 0:
    R0 = np.linalg.qr(A_np[:, : n + 1], mode="r")
    err = np.max(np.abs(np.abs(R[:n + 1, :n + 1]) - np.abs(R0))) / np.abs(R0).max()
    print(f"processes {jax.process_count()}, devices {nd}: R shape {R.shape}, |R| vs numpy {err:.1e}, "
          f"first {t1 - t0:.2f} s, second {t2 - t1:.2f} s", flush=True)
