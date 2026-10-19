"""Levenberg-Marquardt with exact steps for a dense (axisymmetric) NEST problem.

optimize_qa.py takes each LM step by truncated PCG (cg <= 100 iterations).  On these
problems A (the scaled-residual Jacobian) has sigma_min / sigma_max ~ 1e-6..1e-7, so
CG barely resolves the soft directions: the damping sits at its floor, every step
uses all 100 CG iterations, and the runs creep (1-3 % per 50 iterations).  Here the
problem is small enough (n ~ 3e3 unknowns, m ~ 3e5 residuals) to take every step
exactly:

  each iteration, at x:
    A (m x n) from the run's own linearization (ObjectiveFunction.linearize,
      vmapped over the unit vectors, ``--batch`` columns at a time; exact: the model
      is bilinear), with the residual r appended as a last column;
    one QR of [A | r]  ->  R (n x n), c = Q^T r, and the part of r outside range(A);
    the SVD of R: R = U diag(s) V^T;
    the LM step for any damping mu without refactoring:
      d(mu) = -V diag(s / (s^2 + mu)) U^T c,   model f(d) = |c + R d|^2 / 2 + |r_perp|^2 / 2;
    mu scanned over decades (s_max^2 1e-2 .. 1e-26), the step with the lowest actual f
    kept (each trial is one objective evaluation).
  Reports <|F|>/norm, the step |d|/|x|, mu / s_max^2, |grad| and the share of f the
  full Gauss-Newton step would still remove (-> 0 when converged for the GN model).

The model is the run's: the same fits, grid, box, dense G, B_ext (the run's saved
B_ext, or B_DESC - B_plasma(x0) for --bext desc, or the coils' field) and the same
objectives.  Starts from the run's end (or --from-start), writes <out>_iter.npz/json
in optimize_qa.py's format (gs_check_end.py and gs_hessian.py read it).

run (GPU): sbatch --mem=96G run.sbatch scripts/dshape/gs_dense_solve.py --run results/gs/RUN
    [--iters 30] [--from-start] [--out results/gs/RUN_dense]
"""

import argparse
import json
import os
import sys
import time

import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.join(ROOT, "nest"))

from desc import set_device  # noqa: E402

set_device(os.environ.get("NEST_DEVICE", "cpu"))

import jax  # noqa: E402
import jax.numpy as jnp  # noqa: E402

import fit_cache  # noqa: E402
from boundary_potential import BoundaryPotential  # noqa: E402
from dense import DenseFields, j_target  # noqa: E402
from desc_input import DESCInput  # noqa: E402
from boxes import make_box  # noqa: E402
from force import ForceResidual  # noqa: E402
from grid import CartesianGrid  # noqa: E402
from objective import (CurrentError, FieldError, Fields, ForceBalance,  # noqa: E402
                       CurrentMoments, ObjectiveFunction, PressureEnergy, PressureError, PressurePositivity, _base,
                       _cartesian)
from pressure import Pressure  # noqa: E402

jax.config.update("jax_enable_x64", True)


def build(run, st, coils=False):
    """The run's fields and objective function."""
    inp = DESCInput.load(run["eq"], field=run["coils"])
    eq = inp.eq
    ep = run.get("edge_power", 2)
    cur = fit_cache.fit(BoundaryPotential, inp, factors=tuple(run["factors"]),
                        **({"edge_power": ep} if ep != 2 else {}))
    pre = fit_cache.fit(Pressure, inp, factors=tuple(run["factors"]), edge_order=2)
    b = cur.basis
    res = tuple(int(round(run["colloc"] * (u + v))) for u, v in ((b.L, eq.L), (b.M, eq.M), (b.N, eq.N)))
    box = make_box(inp, run["frac"], run.get("grid", "cartesian"), bool(eq.sym), run.get("margin", 2.0))
    fr = ForceResidual(inp, box, resolution=res)
    coils = coils or ("B_ext" not in st.files and run.get("bext") != "desc")
    if not coils:
        fr.B_ext = np.zeros((len(fr.rtz), 3))  # replaced below
    fl = Fields(fr, cur, pre, kernel=run["kernel"], dtype=getattr(jnp, run.get("dtype", "float64")),
                S_dtype=getattr(jnp, run.get("S_dtype", "float32")))
    if run.get("j_nodes") == "direct" and getattr(fl, "cyl", False):  # J at the nodes from the basis
        from node_current import NodeFields
        fl = NodeFields.from_fields(fl, cur, box)
    fields = DenseFields.from_fields(fl, cur, method=run.get("dense_g", "box"))
    x0 = jnp.asarray(st["x0"])
    fields.x0 = x0
    if coils:
        pass  # the coils' field (fr.B_ext)
    elif "B_ext" in st.files:
        fields.B_ext = jnp.asarray(st["B_ext"])
    elif run.get("bext") == "desc":
        fields.B_ext = fields.desc["B"] - fields.compute(x0)["B_plasma"]
    J_start = j_target(run, fields, x0, inp)  # w_J's target (None: J_DESC)
    terms = [ForceBalance()]
    for k, cls in (("J", CurrentError), ("B", FieldError), ("p", PressureError)):
        if (run.get(f"w_{k}") or 0) > 0:
            terms.append(cls(run[f"w_{k}"], target=J_start, rho_max=run.get("w_J_rho")) if k == "J"
                         else cls(run[f"w_{k}"]))
    if (run.get("w_M") or 0) > 0:
        terms.append(CurrentMoments(run["w_M"], which=tuple(run.get("moments", ("I", "R", "Z", "V")))))
    if (run.get("w_W") or 0) > 0:
        terms.append(PressureEnergy(run["w_W"], scale=run.get("p_scale", 1.0)))
    if (run.get("w_pos") or 0) > 0:
        terms.append(PressurePositivity(run["w_pos"]))
    return inp, fields, ObjectiveFunction(terms, fields).build()


def column_tables(fields, batch=64):
    """What the columns of the Jacobian are made of, at any x: J per unit u (dJ_k) and
    p, grad p per unit c_p, Cartesian at the collocation points, computed once into host
    memory (they reach the device one batch at a time: at x3 the J table alone is
    ~8 GB); B_plasma per unit u is the run's own G (on the device already)."""
    n_u, n_p = fields.n_u, fields.n - fields.n_u
    fJ = jax.jit(jax.vmap(lambda e: _cartesian(fields.J_col(e, 0.0).reshape(-1, 3), fields.phi)))
    fP = jax.jit(jax.vmap(lambda e: (lambda pg: (pg[0].ravel(), _cartesian(pg[1].reshape(-1, 3), fields.phi)))(
        fields.p_col(e))))
    n_pts = len(fields.phi)
    AJ = np.empty((n_u, n_pts, 3))
    for i in range(0, n_u, batch):
        k = min(batch, n_u - i)
        AJ[i : i + k] = np.asarray(fJ(jnp.eye(n_u)[i : i + k]))
    PP, GP = np.empty((n_p, n_pts)), np.empty((n_p, n_pts, 3))
    for i in range(0, n_p, batch):
        k = min(batch, n_p - i)
        pp, gp = fP(jnp.eye(n_p)[i : i + k])
        PP[i : i + k], GP[i : i + k] = np.asarray(pp), np.asarray(gp)
    return AJ, jnp.asarray(fields.G), PP, GP


def assembler(obj, fields, tables, batch=256):
    """x -> (r, A): the residual and the exact Jacobian (all columns), from the tables:
    u-column k: dJ = AJ_k, dB = G_k, dF = AJ_k x B(x) + J(x) x G_k;  c_p-column k:
    dp = P_k, dF = -grad P_k; each objective's own tangent and scaling then."""
    AJ, G, PP, GP = tables
    n_u, n_p = AJ.shape[0], PP.shape[0]

    def rows(base, dJ, dB, p, gp):
        d = {"J": dJ, "B": dB, "B_plasma": dB, "p": p, "grad_p": gp,
             "F": jnp.cross(dJ, base["B"]) + jnp.cross(base["J"], dB) - gp}
        return jnp.concatenate([o.compute_scaled_tangent(o.tangent(d, base)) for o in obj.objectives])

    zero3 = jnp.zeros(AJ.shape[1:])
    zero1 = jnp.zeros(AJ.shape[1])
    # dB_k = G_k reshaped to (points, 3) (G: (n_u, 3 n_points), the run's dense B_plasma)
    col_u = jax.jit(jax.vmap(lambda base, dJ, g: rows(base, dJ, g.reshape(-1, 3).astype(dJ.dtype), zero1, zero3),
                             in_axes=(None, 0, 0)))
    col_p = jax.jit(jax.vmap(lambda base, p, gp: rows(base, zero3, zero3, p, gp), in_axes=(None, 0, 0)))

    def assemble(x, host=False):
        """r and the column blocks of A (on the host with ``host``: each block leaves the
        device as soon as it is made, for Jacobians larger than the device)."""
        r, base = _base(obj, x, fields)
        out = np.asarray if host else (lambda v: v)
        blocks = [out(col_u(base, jnp.asarray(AJ[i : i + batch]), G[i : i + batch]).T) for i in range(0, n_u, batch)]
        blocks += [out(col_p(base, jnp.asarray(PP[i : i + batch]), jnp.asarray(GP[i : i + batch])).T)
                   for i in range(0, n_p, batch)]
        return r, blocks

    return assemble


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", required=True)
    ap.add_argument("--iters", type=int, default=30)
    ap.add_argument("--from-start", action="store_true", help="start from the run's x0, not its end")
    ap.add_argument("--batch", type=int, default=256, help="Jacobian columns at a time")
    ap.add_argument("--rtol", type=float, default=1e-9, help="stop when f decreases by less than this, relative")
    ap.add_argument("--gn-tol", type=float, default=1e-7,
                    help="stop when a full Gauss-Newton step would remove less than this share of f")
    ap.add_argument("--gpu-bytes", type=float, default=9e9,
                    help="above this size of [A | r], QR by row blocks with A on the host")
    ap.add_argument("--gpu-tsqr-bytes", type=float, default=22e9,
                    help="[A | r] up to this size stays on the device as its column blocks and is "
                         "QR-factorized there by row blocks (TSQR); larger goes through the host")
    ap.add_argument("--row-block", type=int, default=40000)
    ap.add_argument("--coils", action="store_true",
                    help="B_ext from the coils (box B_plasma, no correction), ignoring a saved B_ext")
    ap.add_argument("--out", default=None)
    a = ap.parse_args()
    t0 = time.time()
    info = json.load(open(a.run + "_iter.json"))
    run = info["args"]
    st = np.load(a.run + "_iter.npz")
    inp, fields, obj = build(run, st, coils=a.coils)
    n, m = obj.dim_x, obj.dim_f
    free = np.asarray(st["free"]) if "free" in st.files else np.ones(n, bool)
    cols = np.flatnonzero(free)
    x = jnp.asarray(st["x0"] if a.from_start else st["x"])
    out = a.out or a.run + ("_dense0" if a.from_start else "_dense")
    fval = lambda xx: float(obj.compute_scalar(xx))  # noqa: E731
    Fmet = lambda xx: float(obj.report(xx)["<|F|>/norm"])  # noqa: E731
    f, F = fval(x), Fmet(x)
    print(f"{inp}\nrun {a.run} ({'x0' if a.from_start else 'end'}): {n} unknowns ({len(cols)} free), {m} residuals; "
          f"objectives {[o.name for o in obj.objectives]}; B_ext "
          f"{'coils' if a.coils else 'saved' if 'B_ext' in st.files else 'B_DESC - B_plasma(x0)' if run.get('bext') == 'desc' else 'coils'}; "
          f"setup {time.time() - t0:.0f} s\nstart: f {f:.6e}, <|F|>/norm {F:.4e}", flush=True)
    t1 = time.time()
    assemble = assembler(obj, fields, column_tables(fields), batch=a.batch)
    print(f"column tables: {time.time() - t1:.0f} s", flush=True)
    hist = [{"it": 0, "f": f, "<|F|>/norm": F, "time": 0.0}]
    xs = [np.asarray(x)]
    mu = None

    def save():
        args = dict(run, out=out, solver="dense LM (gs_dense_solve.py)", start_from=a.run,
                    **({"bplasma": "box", "bext": "field"} if a.coils else {}))
        np.savez(out + "_iter.npz", x=np.asarray(x), x0=np.asarray(st["x0"]), free=free,
                 p_scale=run.get("p_scale", 1.0), xs=np.array(xs), B_ext=np.asarray(fields.B_ext))
        with open(out + "_iter.json", "w") as fo:
            json.dump({"args": args, "history": hist}, fo, indent=1)

    for it in range(1, a.iters + 1):
        t1 = time.time()
        size = m * (len(cols) + 1) * 8
        big, huge = size >= a.gpu_bytes, size >= a.gpu_tsqr_bytes
        try:
            r, blocks = assemble(x, host=huge)
        except Exception as e:  # noqa: BLE001  the device is full: the host path from now on
            if huge or "RESOURCE_EXHAUSTED" not in str(e):
                raise
            print("Jacobian does not fit on the device: through the host from now on", flush=True)
            a.gpu_tsqr_bytes, huge = 0.0, True
            r, blocks = assemble(x, host=True)
        if it == 1:  # the assembled columns against the run's own linearization
            _, Jv, _ = obj.linearize(x)
            v = jnp.asarray(np.random.default_rng(0).standard_normal(n))
            w_ = [b_.shape[1] for b_ in blocks]
            vn = np.asarray(v)
            Av = sum(jnp.asarray(np.asarray(b_) @ vn[i0 : i0 + k]) for b_, i0, k in zip(blocks, np.cumsum([0] + w_[:-1]), w_))
            print(f"assembled Jacobian against the linearization along a random v: "
                  f"{float(jnp.linalg.norm(Av - Jv(v)) / jnp.linalg.norm(Jv(v))):.1e}", flush=True)
        if len(cols) < n:
            Afull = (np if huge else jnp).concatenate(blocks, axis=1)
            blocks = [Afull[:, cols]]
            del Afull
        blocks.append(jnp.asarray(r)[:, None])
        if not big:  # [A | r] and its QR on the device
            Aaug = jnp.concatenate(blocks, axis=1).block_until_ready()
            t_jac = time.time() - t1
            Raug = jnp.linalg.qr(Aaug, mode="r").block_until_ready()
            del Aaug
        if big and not huge:  # [A | r] as its column blocks on the device, QR by row blocks there
            # (TSQR): no concatenated copy, no trip through the host
            jax.block_until_ready(blocks)
            t_jac = time.time() - t1
            try:
                Raug = None
                for i0 in range(0, m, a.row_block):
                    blk = jnp.concatenate([b_[i0 : i0 + a.row_block] for b_ in blocks], axis=1)
                    Raug = jnp.linalg.qr(blk if Raug is None else jnp.concatenate([Raug, blk]), mode="r")
                Raug = Raug.block_until_ready()
            except Exception as e:  # noqa: BLE001
                if "RESOURCE_EXHAUSTED" not in str(e):
                    raise
                print("device TSQR does not fit: through the host from now on", flush=True)
                Raug, blocks = None, [np.asarray(b_) for b_ in blocks]
                a.gpu_tsqr_bytes, huge = 0.0, True
        if big and huge:  # too large: [A | r] on the host, QR by row blocks (TSQR) on the device
            blocks[-1] = np.asarray(blocks[-1])
            Aaug = np.concatenate(blocks, axis=1)
            blocks = []
            t_jac = time.time() - t1
            Raug = None
            for i0 in range(0, m, a.row_block):
                blk = jnp.asarray(Aaug[i0 : i0 + a.row_block])
                Raug = jnp.linalg.qr(blk if Raug is None else jnp.concatenate([Raug, blk]), mode="r")
            Raug = Raug.block_until_ready()
            del Aaug
        del blocks
        t_qr = time.time() - t1
        R, cvec, rperp = Raug[:-1, :-1], Raug[:-1, -1], float(jnp.abs(Raug[-1, -1]))
        U, s, Vt = jnp.linalg.svd(R)
        ch = U.T @ cvec
        s2 = s**2
        g = float(jnp.linalg.norm(R.T @ cvec))
        if mu is None:
            mu = 1e-10 * float(s2[0])
        gn_share = (f - 0.5 * rperp**2) / f  # what the full GN step would remove

        def step(mu_):
            dd = -(Vt.T @ (s * ch / (s2 + mu_)))
            fpred = 0.5 * float(jnp.sum((ch * mu_ / (s2 + mu_)) ** 2)) + 0.5 * rperp**2
            return dd, fpred

        # every damping is a cheap re-solve with the SVD: scan mu over decades and keep
        # the step with the lowest actual f (mu ~ s_max^2 10^-k, k = 2 .. 26)
        best = None
        for k in np.arange(2, 27, 0.5):
            mu_ = float(s2[0]) * 10.0 ** (-k)
            d_, fpred_ = step(mu_)
            xt_ = x.at[jnp.asarray(cols)].add(d_)
            ft_ = fval(xt_)
            if best is None or ft_ < best[0]:
                best = (ft_, mu_, d_, fpred_, xt_)
        ft, mu, d, fpred, xt = best
        if not ft < f:
            print("no decrease found: stopping", flush=True)
            break
        rho = (f - ft) / max(f - fpred, 1e-300)
        rel = (f - ft) / f
        x, f = xt, ft
        F = Fmet(x)
        dx = float(jnp.linalg.norm(d) / jnp.linalg.norm(x))
        h = {"it": it, "f": f, "<|F|>/norm": F, "|d|/|x|": dx, "mu/s_max^2": mu / float(s2[0]),
             "gain ratio": rho, "|grad|": g, "GN share left": gn_share, "sigma_min/sigma_max":
             float(s[-1] / s[0]), "time": time.time() - t1}
        hist.append(h)
        xs.append(np.asarray(x))
        save()
        print(f"it {it:3d}  f {f:.6e}  <|F|>/norm {F:.4e}  |d|/|x| {dx:.2e}  mu/s_max^2 {mu / float(s2[0]):.1e}  "
              f"gain {rho:+.2f}  |g| {g:.2e}  GN share {gn_share:.2e}  s_min/s_max {float(s[-1] / s[0]):.1e}  "
              f"{time.time() - t1:.0f} s (Jacobian {t_jac:.0f} s, QR {t_qr - t_jac:.0f} s)", flush=True)
        if rel < a.rtol:
            print(f"converged: relative decrease {rel:.1e} < {a.rtol:g}", flush=True)
            break
        if gn_share < a.gn_tol:  # a full Gauss-Newton step would remove less than this of f
            print(f"converged: GN share left {gn_share:.1e} < {a.gn_tol:g}", flush=True)
            break
    save()
    print(f"saved {out}_iter.npz/.json; total {time.time() - t0:.0f} s")


if __name__ == "__main__":
    main()
