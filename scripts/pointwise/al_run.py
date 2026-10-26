"""The pointwise-bound problem on NCSX by augmented Lagrangian, continued in alpha.

    min |J - J_fit|^2 / 2  s.t.  |F(x_i)| / norm <= eps = alpha F_ref  at every point

Two inner trust-region solvers (``nest/pointwise.py``), run independently:

    --method matfree   Lanczos trust-region steps (GLTR; --tr steihaug: truncated CG) with
                       J v and J^T w from the annulus Fields (one annulus solve per
                       product); main's environment:
        sbatch run_sai.sbatch scripts/pointwise/al_run.py --tables T.npz --method matfree --out O
    --method qr        the stacked Jacobian from the column tables, QR by jaxmg
                       (cuSOLVERMp, one process per GPU), DESC's exact trust-region step;
                       the nest_mg environment (JAX 0.11, no DESC):
        sbatch --gres=gpu:nvidia_b200:4 --ntasks=4 run_mg.sbatch scripts/pointwise/al_run.py \
            --tables T.npz --method qr --out O
    --method qr --qr jnp   the same on one GPU with jnp.linalg.qr

F_ref: DESC's own 99th percentile of |F_i| / norm at the points (``--f-ref p99`` / ``max``:
the start's). Each
alpha starts from the previous state and multipliers; the run stops at the first alpha
whose bound is not met (infeasible at this resolution). Saves O.json (per alpha: metrics,
the AL outer and trust-region histories) and O_a<alpha>.npz (x, lam, rho, per-point |F|,
c, lam, p).
"""

import argparse
import json
import os
import sys
import time

import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.join(ROOT, "nest"))
sys.path.insert(0, os.path.join(ROOT, "scripts"))

ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
ap.add_argument("--tables", required=True)
ap.add_argument("--method", choices=["matfree", "qr"], required=True)
ap.add_argument("--qr", choices=["jaxmg", "jnp"], default="jaxmg")
ap.add_argument("--tr", choices=["gltr", "steihaug"], default="gltr",
                help="matrix-free trust-region subproblem: Lanczos (exact in the Krylov space) "
                     "or Steihaug's truncated CG")
ap.add_argument("--tile", type=int, default=256, help="cuSOLVERMp tile width")
ap.add_argument("--alphas", type=float, nargs="+",
                default=[5.0, 3.0, 2.0, 1.5, 1.0, 0.7, 0.5, 0.3, 0.2, 0.15, 0.1, 0.07, 0.05, 0.03,
                         0.02, 0.01])
ap.add_argument("--f-ref", choices=["desc", "p99", "max"], default="desc",
                help="desc: DESC's own 99th percentile of |F| / norm at the points; p99, max: "
                     "the start's")
ap.add_argument("--rho0", type=float, default=1.0)
ap.add_argument("--max-outer", type=int, default=20)
ap.add_argument("--maxiter", type=int, default=30, help="trust-region steps per AL subproblem")
ap.add_argument("--viol-tol", type=float, default=2e-2, help="c_i <= this (|F| <= eps (1 + tol / 2))")
ap.add_argument("--delta0", type=float, default=1e-2,
                help="smoothed plus function (s + sqrt(s^2 + delta^2)) / 2, delta = delta0 10^-k at "
                     "outer iteration k down to --delta-min; 0: max(0, s)")
ap.add_argument("--delta-min", type=float, default=1e-6)
ap.add_argument("--cg-tol", type=float, default=1e-2)
ap.add_argument("--cg-maxiter", type=int, default=100)
ap.add_argument("--start", default=None, help="state npz of a previous run (x, lam, rho) to continue from")
ap.add_argument("--adaptive", type=float, default=1.05,
                help="when an alpha is not met: restore the last feasible state and try the geometric "
                     "midpoint, while alpha_prev / alpha > this (0: stop at the first failure)")
ap.add_argument("--time-limit", type=float, default=None, help="seconds; stops between alphas")
ap.add_argument("--out", required=True)
a = ap.parse_args()

if a.method == "qr" and a.qr == "jaxmg":
    import jax

    vis = [v for v in os.environ.get("CUDA_VISIBLE_DEVICES", "").split(",") if v]
    if os.environ.get("MULTI"):
        loc = int(os.environ.get("SLURM_LOCALID", 0))
        jax.distributed.initialize(local_device_ids=[loc] if len(vis) > 1 else None)
elif a.method == "matfree":
    from desc import set_device

    set_device(os.environ.get("NEST_DEVICE", "cpu"))
import jax  # noqa: E402

jax.config.update("jax_enable_x64", True)
import jax.numpy as jnp  # noqa: E402

import pointwise as pw  # noqa: E402

MAIN = jax.process_index() == 0


def log(*s):
    if MAIN:
        print(*s, flush=True)


def load_tables(path, with_G):
    keep = None if with_G else {"G"}
    f = np.load(path, mmap_mode="r")
    T = {}
    for k in f.files:
        if keep and k in keep:
            continue
        v = f[k]
        T[k] = v.item() if v.ndim == 0 else np.asarray(v)
    return T


def make_qr(kind, tile):
    """qr_fn(rows_fn, n_rows, n_cols) -> the triangular factor of the stacked matrix."""
    if kind == "jnp":
        def qr_fn(rows_fn, n_rows, n_cols):
            M = rows_fn(np.arange(n_rows))
            if n_rows < n_cols:
                M = jnp.vstack([M, jnp.zeros((n_cols - n_rows, n_cols))])
            return jnp.linalg.qr(M, mode="r")[:n_cols, :n_cols]
        return qr_fn
    import jaxmg
    from jax.experimental import multihost_utils
    from jax.sharding import NamedSharding, PartitionSpec as P

    npr = jax.process_count()
    mesh = jax.make_mesh((npr, 1), ("pr", "pc"))
    pid = jax.process_index()

    def qr_fn(rows_fn, n_rows, n_cols):
        nc = -(-n_cols // npr) * npr
        m = -(-max(n_rows, nc) // npr) * npr
        mb = m // npr
        sel = np.arange(pid * mb, min((pid + 1) * mb, n_rows))
        local = jnp.zeros((mb, nc))
        if len(sel):
            local = local.at[: len(sel), :n_cols].set(rows_fn(sel))
        A = jax.make_array_from_process_local_data(NamedSharding(mesh, P("pr", None)), local,
                                                   (m, nc))
        _, R = jaxmg.qr(A, tile, mesh=mesh, donate=True)
        if npr > 1:
            R = multihost_utils.process_allgather(R, tiled=True)
        return jnp.asarray(np.asarray(R)[:n_cols, :n_cols])
    return qr_fn


def main():
    t0 = time.time()
    T = load_tables(a.tables, with_G=(a.method == "qr"))
    meta = json.loads(str(T["meta"]))
    if a.method == "matfree":
        from build_tables import setup
        *_, fields = setup(meta["eq"], meta["coils"], meta["factors"], meta["p_factors"],
                           meta["frac"], meta["colloc"], verbose=MAIN)
        model = pw.FieldsModel(fields)
    else:
        model = pw.DenseModel(T)
    prob = pw.Problem(model, T)
    L, HJ = pw.metric(T)
    met = pw.Metric(L)
    x = jnp.concatenate([jnp.asarray(T["u0"]), jnp.asarray(T["c0"])])
    x0 = x
    Rg = None
    if a.method == "qr":  # g = R_g (u - u0) exactly: R_g^T R_g = H_J (symmetric square root)
        s, V = np.linalg.eigh(HJ)
        Rg = (V * np.sqrt(np.maximum(s, 0.0))).T
    solver = pw.ALSolver(prob, met, x0, step=a.tr if a.method == "matfree" else "qr",
                         T=T if a.method == "qr" else None,
                         qr_fn=make_qr(a.qr, a.tile) if a.method == "qr" else None, Rg=Rg,
                         log=log, cg_tol=a.cg_tol, cg_maxiter=a.cg_maxiter)
    base0 = model.base(x0)
    FD = np.linalg.norm(np.cross(T["J_desc"], T["B_desc"]) - T["grad_p_desc"], axis=1) / T["norm"]
    F_ref = {"desc": float(np.percentile(FD, 99)), "p99": float(T["F_ref_p99"]),
             "max": float(T["F_ref_max"])}[a.f_ref]
    m0, _ = pw.state_metrics(T, base0, x0, np.zeros(model.n), F_ref, base0)
    log(f"{a.method}{'/' + a.qr if a.method == 'qr' else ''} on {jax.process_count()} process(es), "
        f"{jax.device_count()} device(s); {meta}\nF_ref ({a.f_ref}) {F_ref:.4e}; start: "
        + ", ".join(f"{k} {v:.3e}" for k, v in m0.items() if isinstance(v, float))
        + f"; setup {time.time() - t0:.0f} s")
    lam = jnp.zeros(model.n)
    rho = a.rho0
    if a.start:
        st = np.load(a.start)
        x, lam, rho = jnp.asarray(st["x"]), jnp.asarray(st["lam"]), float(st["rho"])
        log(f"continuing from {a.start} (alpha {float(st['alpha']):g}, rho {rho:.1e})")
    out = {"args": vars(a), "meta": meta, "F_ref": F_ref, "start": m0, "alphas": []}
    os.makedirs(os.path.dirname(os.path.abspath(a.out)), exist_ok=True)
    queue = list(a.alphas)
    last = None  # (alpha, x, lam, rho) of the last bound met
    if a.start:
        last = (float(np.load(a.start)["alpha"]), x, lam, rho)
    while queue:
        alpha = queue.pop(0)
        if a.time_limit and time.time() - t0 > a.time_limit:
            log(f"time limit: stopping before alpha {alpha}")
            break
        eps = alpha * F_ref
        t1 = time.time()
        log(f"alpha {alpha:g}: eps {eps:.4e}")
        x1, lam1, rho1, outer = solver.solve_alpha(x, eps, lam, rho, max_outer=a.max_outer,
                                                   viol_tol=a.viol_tol, maxiter=a.maxiter,
                                                   delta0=a.delta0, delta_min=a.delta_min)
        base = model.base(x1)
        m, arrays = pw.state_metrics(T, base, x1, lam1, eps, base0)
        feasible = outer[-1]["viol"] <= a.viol_tol
        n_tr = sum(len(o["inner"]) for o in outer)
        m.update({"alpha": alpha, "feasible": feasible, "rho": rho1, "outer": len(outer),
                  "tr steps": n_tr, "time": time.time() - t1})
        out["alphas"].append({"metrics": m, "outer": outer})
        log(f"alpha {alpha:g} {'met' if feasible else 'NOT MET'}: " + ", ".join(
            f"{k} {v:.3e}" for k, v in m.items() if isinstance(v, float)) + f" ({n_tr} TR steps)")
        if MAIN:
            np.savez(f"{a.out}_a{alpha:g}.npz", x=np.asarray(x1), lam=np.asarray(lam1), rho=rho1,
                     eps=eps, alpha=alpha, feasible=feasible, **arrays)
            with open(a.out + ".json", "w") as fh:
                json.dump(out, fh, indent=1, default=float)
        if feasible:
            x, lam, rho = x1, lam1, rho1
            last = (alpha, x, lam, rho)
            continue
        if last is not None and a.adaptive and last[0] / alpha > a.adaptive:
            mid = float(np.sqrt(last[0] * alpha))
            log(f"alpha {alpha:g} not reached: back to alpha {last[0]:g}, trying {mid:.4g} first")
            _, x, lam, rho = last
            rho = min(rho, 1e6)  # the failed attempt escalated rho: start the retry moderate
            queue = [mid, alpha] + queue
            continue
        log(f"alpha {alpha:g} not reached: stopping the continuation")
        break
    log(f"done: {time.time() - t0:.0f} s")


if __name__ == "__main__":
    main()
