"""Time the stacked-Jacobian QR of the dense trust-region step on 1, 2, 4 GPUs (jaxmg).

The matrix is the one ``al_run.py --method qr`` factors at a given state (rows: the n_u
compressed current rows and the constraint rows with dm > 0, columns: n + 1).

    sbatch --gres=gpu:nvidia_b200:G --ntasks=G run_mg.sbatch scripts/pointwise/bench_qr.py \
        --tables T.npz --state X_a0.3.npz
"""
import argparse
import json
import os
import sys
import time

import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.join(ROOT, "nest"))
import jax  # noqa: E402

vis = [v for v in os.environ.get("CUDA_VISIBLE_DEVICES", "").split(",") if v]
if os.environ.get("MULTI"):
    jax.distributed.initialize(local_device_ids=[int(os.environ.get("SLURM_LOCALID", 0))]
                               if len(vis) > 1 else None)
jax.config.update("jax_enable_x64", True)
import jax.numpy as jnp  # noqa: E402

import pointwise as pw  # noqa: E402

ap = argparse.ArgumentParser()
ap.add_argument("--tables", required=True)
ap.add_argument("--state", required=True)
ap.add_argument("--delta", type=float, default=1e-2, help="smoothing (rows with dm > 1e-14)")
ap.add_argument("--repeat", type=int, default=3)
a = ap.parse_args()


def load_tables(path):
    f = np.load(path, mmap_mode="r")
    return {k: (f[k].item() if f[k].ndim == 0 else np.asarray(f[k])) for k in f.files}


def make_qr(tile):
    """As al_run.make_qr("jaxmg"): each process builds its row block, jaxmg.qr, all-gather."""
    import jaxmg
    from jax.experimental import multihost_utils
    from jax.sharding import NamedSharding, PartitionSpec as P
    npr, pid = jax.process_count(), jax.process_index()
    mesh = jax.make_mesh((npr, 1), ("pr", "pc"))

    def qr_fn(rows_fn, n_rows, n_cols):
        nc = -(-n_cols // npr) * npr
        m = -(-max(n_rows, nc) // npr) * npr
        mb = m // npr
        sel = np.arange(pid * mb, min((pid + 1) * mb, n_rows))
        local = jnp.zeros((mb, nc))
        if len(sel):
            local = local.at[: len(sel), :n_cols].set(rows_fn(sel))
        A = jax.make_array_from_process_local_data(NamedSharding(mesh, P("pr", None)), local, (m, nc))
        _, R = jaxmg.qr(A, tile, mesh=mesh, donate=True)
        if npr > 1:
            R = multihost_utils.process_allgather(R, tiled=True)
        return jnp.asarray(np.asarray(R)[:n_cols, :n_cols])
    return qr_fn


T = load_tables(a.tables)
st = np.load(a.state)
model = pw.DenseModel(T)
prob = pw.Problem(model, T)
L, HJ = pw.metric(T)
s, V = np.linalg.eigh(HJ)
Rg = (V * np.sqrt(np.maximum(s, 0.0))).T
x0 = jnp.concatenate([jnp.asarray(T["u0"]), jnp.asarray(T["c0"])])
solver = pw.ALSolver(prob, pw.Metric(L), x0, step="qr", T=T, qr_fn=make_qr(256), Rg=Rg,
                     log=lambda *x: None)
x, lam, rho, eps = jnp.asarray(st["x"]), jnp.asarray(st["lam"]), float(st["rho"]), float(st["eps"])
r, m, c, base = prob.residual(x, lam, rho, eps, a.delta)
n_u = model.n_u
act = np.flatnonzero(np.asarray(m[1]) > 1e-14)
n_rows = n_u + len(act)


def rows_fn(sel):
    return pw.stacked_rows(solver.Td, base, x, x0, m, rho, eps, prob.norm, solver.RgLt,
                           solver.Linv, n_u, rows=sel)


times = []
for k in range(a.repeat + 1):
    t0 = time.time()
    R = solver.qr_fn(rows_fn, n_rows, model.dim + 1)
    jax.block_until_ready(R)
    times.append(time.time() - t0)
if jax.process_index() == 0:
    print(json.dumps({"processes": jax.process_count(), "rows": int(n_rows), "cols": int(model.dim + 1),
                      "first (compile)": times[0], "times": times[1:],
                      "best": min(times[1:])}), flush=True)
