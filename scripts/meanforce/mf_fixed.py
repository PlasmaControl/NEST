"""Fixed-boundary, matrix-free (GLTR) mean-force study with the Poisson pressure
(``nest/meanforce_fixed.py``): closest state to J_fit under M <= alpha M_DESC on the
equilibrium's own boundary, coils' B_ext, annulus B_plasma.

Memory: no u tables, no n x k pressure matrix, no n x n_u metric table. Grams by slices.
Every step is logged and written to <out>.json (and --mirror, fsynced); per alpha the state
<out>_a<alpha>.npz (u, c_p, nu, rho, eps). --time-budget stops cleanly with the state saved.
"""

import argparse
import functools
import json
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

jax.config.update("jax_enable_x64", True)

import fit_cache  # noqa: E402
import meanforce as mf  # noqa: E402
import meanforce_fixed as mff  # noqa: E402
import pointwise as pw  # noqa: E402
from boundary_potential import BoundaryPotential  # noqa: E402
from desc_input import DESCInput  # noqa: E402
from force import ForceResidual  # noqa: E402
from free_boundary import _chunked, reference  # noqa: E402
from objective import Fields  # noqa: E402
from optimize_fb import annulus  # noqa: E402
from pressure import Pressure  # noqa: E402

D = os.path.join(ROOT, "desc_input", "vmec_free_boundary", "NCSX")
log = functools.partial(print, flush=True)


class Budget(Exception):
    pass


def mem():
    out = {}
    try:
        st = jax.devices()[0].memory_stats() or {}
        out = {"in use GB": st.get("bytes_in_use", 0) / 1e9,
               "peak GB": st.get("peak_bytes_in_use", 0) / 1e9}
    except Exception:
        pass
    try:
        import resource
        out["host maxrss GB"] = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1e6
    except Exception:
        pass
    return out


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--eq", default=os.path.join(D, "NCSX_fb_L16M16N12.h5"))
    ap.add_argument("--coils", default=os.path.join(D, "NCSX_coils.h5"))
    ap.add_argument("--factors", type=float, nargs=3, default=(1, 1, 1))
    ap.add_argument("--LMN", type=int, nargs=3, default=None,
                    help="NEST basis L M N directly (overrides --factors)")
    ap.add_argument("--colloc", type=float, default=1.5)
    ap.add_argument("--frac", type=float, default=40)
    ap.add_argument("--N-phi", type=int, default=None)
    ap.add_argument("--pad", type=float, default=0.02)
    ap.add_argument("--mode-chunk", type=int, default=16)
    ap.add_argument("--rblock", type=int, default=32)
    ap.add_argument("--interp-order", type=int, default=8)
    ap.add_argument("--alphas", type=float, nargs="+", default=[0.7])
    ap.add_argument("--rho0", type=float, default=1.0)
    ap.add_argument("--band", type=float, default=0.05)
    ap.add_argument("--maxiter", type=int, default=40)
    ap.add_argument("--max-outer", type=int, default=25)
    ap.add_argument("--gtol", type=float, default=1e-3)
    ap.add_argument("--cg", type=int, default=400, help="Lanczos cap per trust-region subproblem")
    ap.add_argument("--subproblem-test", type=int, nargs="+", default=None,
                    help="Lanczos caps: at the --start state solve the GN trust-region subproblem once per "
                         "cap (radius --test-delta), report step, predicted/actual decrease, residual, Ritz "
                         "extremes, then exit")
    ap.add_argument("--test-delta", type=float, default=1.6e-2)
    ap.add_argument("--test-model", choices=("gn", "hvp", "beta"), default="gn",
                    help="gn: base GN rows; hvp: exact Hessian of the merit; beta: the production beta-model rows (--beta)")
    ap.add_argument("--precond", choices=("hj", "force"), default="force",
                    help="TR metric: chol(H_J + H_F(local) + a a^T), the GN matrix without the J0 x dB part, "
                         "rebuilt on outer updates and Lanczos-cap / rent-or-buy triggers (default), or "
                         "chol(H_J) (misfit only, fixed; enough for loose bounds, fails at alpha 0.05)")
    ap.add_argument("--hvp-far", type=float, default=0.1,
                    help="hybrid: use the GN/majorizer model while c_F = M/eps - 1 exceeds this")
    ap.add_argument("--hvp-fallback", type=int, default=5,
                    help="hybrid: GN subproblems after an exact-model step that shrank the radius")
    ap.add_argument("--beta", type=float, default=0.5, help="model beta: radial curvature beta/|z| (1 = majorizer)")
    ap.add_argument("--no-mult-ls", action="store_true", help="disable the least-squares multiplier estimate")
    ap.add_argument("--refresh-lz", type=float, default=None,
                    help="force metric: also rebuild when Lanczos >= this x the count after the last rebuild")
    ap.add_argument("--refresh-dM", type=float, default=None,
                    help="force metric: also rebuild when M moved by more than this (relative) since the rebuild")
    ap.add_argument("--outer-rebuild-factor", type=float, default=2.0,
                    help="force metric: rebuild at an outer iteration only if psi' = nu + rho c changed by "
                         "more than this factor since the last build (always at the first)")
    ap.add_argument("--beta-adapt", action="store_true", help="self-tuning beta (model beta)")
    ap.add_argument("--beta-min", type=float, default=1.0 / 32)
    ap.add_argument("--model", choices=("hvp", "gn", "hybrid", "beta"), default="beta",
                    help="TR model: beta (default; the majorizer with the force rows' curvature along the "
                         "active direction scaled by --beta, the only model that converged alpha 0.05), "
                         "hvp (exact Hessian-vector products, indefinite Lanczos-TR; fastest at loose "
                         "bounds, collapses near small ones), gn (Gauss-Newton/majorizer of meanforce.Rows, "
                         "collaborator parity) or hybrid (gn far from the bound, hvp near it)")
    ap.add_argument("--hvp", dest="model", action="store_const", const="hvp", help="alias for --model hvp")
    ap.add_argument("--gn", dest="model", action="store_const", const="gn", help="alias for --model gn")
    ap.add_argument("--cg-tol", type=float, default=1e-2)
    ap.add_argument("--stat-tol", type=float, default=1e-3)
    ap.add_argument("--stat-window", type=int, default=10)
    ap.add_argument("--delta0", type=float, default=1e-3, help="first trust radius (and the radius at each outer reset)")
    ap.add_argument("--no-delta-reset", action="store_true",
                    help="keep the trust radius across outer iterations / bounds (old behaviour)")
    ap.add_argument("--gram-chunk", type=int, default=1024)
    ap.add_argument("--time-budget", type=float, default=None, help="seconds; stop cleanly")
    ap.add_argument("--setup-only", action="store_true", help="stop after the start state")
    ap.add_argument("--cross", default=None,
                    help="npz of a LOWER-resolution optimum: embed it (and that basis's J_fit) in this "
                         "basis, evaluate objective / M / decomposition here, and exit (no solve)")
    ap.add_argument("--cross-factors", type=float, nargs=3, default=None, help="basis factors of --cross")
    ap.add_argument("--cross-LMN", type=int, nargs=3, default=None, help="basis L M N of --cross")
    ap.add_argument("--cross-ref", default=None, help="npz of this resolution's own optimum (optional)")
    ap.add_argument("--start", default=None, help="npz with u (and nu, rho) to continue from")
    ap.add_argument("--start-LMN", type=int, nargs=3, default=None,
                    help="basis L M N of --start when it differs from this run's (embedded; modes "
                         "this basis lacks are dropped)")
    ap.add_argument("--lanczos-stop", choices=("both", "tr"), default="tr",
                    help="models gn/beta: Lanczos stops on the TR residual alone (tr, default) or only when "
                         "the TR step and the unconstrained GN minimizer have both converged (both: the "
                         "pointwise.lanczos_tr_step rule; 224-400 iterations per boundary step near the bound)")
    ap.add_argument("--no-lean-metric", action="store_true",
                    help="force metric: the reference build (default: the lean, low-memory build)")
    ap.add_argument("--force-norm", choices=("mean", "p"), default="mean",
                    help="bounded force measure: the volume mean of |F| (default) or the interior p-norm "
                         "(sum_int w |F|^p / sum_int w)^(1/p), eps = alpha x DESC's own interior p-norm on "
                         "the same points; the mean, interior max and edge mean are logged as monitors")
    ap.add_argument("--p", type=float, default=8.0, help="--force-norm p: the exponent")
    ap.add_argument("--p-schedule", type=float, nargs="+", default=None,
                    help="--force-norm p: a continuation in p (e.g. 2 4 8 16), each warm-started from the last")
    ap.add_argument("--rho-cut", type=float, default=0.95,
                    help="--force-norm p: the interior is DESC rho < rho-cut (the edge is monitored only)")
    ap.add_argument("--edge-alpha", type=float, default=None,
                    help="--force-norm p: also bound the edge mean (rho >= rho-cut) at edge-alpha x DESC's edge mean, "
                         "through one composite constraint (M_p^16 + (r M_edge)^16)^(1/16) <= eps, r = eps / eps_edge")
    ap.add_argument("--p-model-beta", action="store_true",
                    help="--force-norm p: use --beta along F_i instead of the exact p - 1")
    ap.add_argument("--eval-only", action="store_true",
                    help="--force-norm p: report the start (J_fit) and --start states' p-norms, interior max, "
                         "edge and mean against DESC's, for each p of the schedule, then exit (no metric build)")
    ap.add_argument("--eval-state", action="append", default=[],
                    help="--eval-only: extra states name=path.npz (this run's basis); repeatable")
    ap.add_argument("--out", default=os.path.join(ROOT, "results", "meanforce", "fixed"))
    ap.add_argument("--mirror", default=None, help="directory for fsynced copies (Volume)")
    ap.add_argument("--pin-current", action="store_true",
                    help="exact constraint: the m = 0 modes of a stay at J_fit's (the current enclosed by every "
                         "DESC surface, in every cross-section); a --start state is restored onto it")
    ap.add_argument("--pin-flux", action="store_true",
                    help="exact constraint: the toroidally averaged toroidal flux stays at J_fit's")
    a = ap.parse_args()
    a.hvp = True  # per-subproblem diagnostics (DIAG_H) are recorded for every model
    os.makedirs(os.path.dirname(a.out) or ".", exist_ok=True)
    if a.mirror:
        os.makedirs(a.mirror, exist_ok=True)
    t0 = time.time()
    run = {"args": vars(a), "setup": {}, "alphas": [], "memory": []}

    def write(path, text):
        with open(path, "w") as fh:
            fh.write(text)
            fh.flush()
            os.fsync(fh.fileno())

    def dump():
        if a.hvp:
            run["hvp diag"] = mff.DIAG_H
        txt = json.dumps(run, indent=1, default=float)
        write(a.out + ".json", txt)
        if a.mirror:
            write(os.path.join(a.mirror, os.path.basename(a.out) + ".json"), txt)

    def save_state(tag, z, s, W):
        for d in [os.path.dirname(a.out) or "."] + ([a.mirror] if a.mirror else []):
            p = os.path.join(d, os.path.basename(a.out) + f"_{tag}.npz")
            np.savez(p, u=np.asarray(z), c_p=np.asarray(s["c"]), nu=W.nu, rho=W.rho, eps=W.eps)

    def phase(name, t1):
        run["setup"][name] = time.time() - t1
        m = mem()
        run["memory"].append({"phase": name, "t": time.time() - t0, **m})
        log(f"[setup] {name}: {time.time() - t1:.1f} s; mem {m}")
        dump()

    t1 = time.time()
    inp0 = DESCInput.load(a.eq, field=a.coils)
    eq0 = inp0.eq
    kw = ({"L": a.LMN[0], "M": a.LMN[1], "N": a.LMN[2]} if a.LMN else
          {"factors": tuple(a.factors)})
    cur0 = fit_cache.fit(BoundaryPotential, inp0, **kw)
    pre0 = fit_cache.fit(Pressure, inp0, edge_order=2, **kw)
    b = cur0.basis
    res = tuple(int(round(a.colloc * (u + v))) for u, v in
                ((b.L, eq0.L), (b.M, eq0.M), (b.N, eq0.N)))
    ref = reference(inp0, res, current=cur0)
    norm = ref["norm"]
    M_desc = ref["F_desc_vol"] / norm
    run["basis"] = [b.L, b.M, b.N]
    log(f"{inp0}\nbasis {b.L}/{b.M}/{b.N}, collocation resolution {res} {tuple(ref['shape'])}; "
        f"DESC <|F|>_vol / norm {M_desc:.4e}")
    phase("load + fits + reference", t1)

    t1 = time.time()
    box = annulus(inp0, inp0.a / a.frac, a.pad, a.N_phi)
    box.locate_plasma_planes(inp0, sym=bool(eq0.sym), verbose=True)
    fr = ForceResidual(inp0, box, resolution=res)
    fr.norm, fr.F_desc_vol = norm, ref["F_desc_vol"]
    if "B_ext" not in vars(fr):
        fr.B_ext = _chunked(inp0.B_ext, fr.x)
    phase("annulus + locate + ForceResidual", t1)

    t1 = time.time()
    fields = Fields(fr, cur0, pre0, dtype=jnp.float64, cyl_stencil=True,
                    mode_chunk=a.mode_chunk, cyl_fused=True, cyl_rblock=a.rblock,
                    cyl_order=a.interp_order or None)
    run["sizes"] = {"n_u": int(fields.n_u), "n_p": int(fields.n_p), "points": len(fields.phi),
                    "annulus": str(box), "annulus nodes": int(np.prod(box.shape))}
    log(f"sizes: {run['sizes']}")
    phase("Fields", t1)

    t1 = time.time()
    G = mff.gram_P(fields, cur0, pre0, chunk=a.gram_chunk, log=log)
    phase("pressure Gram", t1)
    u0 = jnp.asarray(cur0.u)
    Jfit = mff._J(fields, u0, fields.I)
    sw = jnp.asarray(fields.sw)
    rmsJ = float(jnp.sqrt(jnp.sum(sw[:, None] ** 2 * Jfit**2)))
    A = mff.FixedAnchor(fields, cur0, pre0, u0, rmsJ, norm, G, log=log)
    del G
    run["pressure space"] = {"rank": A.p_rank, "cond(G)": A.p_cond}
    M_desc_mean = M_desc
    p_list = [None]
    def set_pn(p_, alpha_):
        r_ = None
        if a.edge_alpha is not None:
            r_ = (alpha_ * desc_p[p_]["M_p"]) / (a.edge_alpha * desc_p[p_]["edge"])
        A.set_pnorm(p_, pmask, exact_beta=not a.p_model_beta, edge_ratio=r_)

    def pn_report(ss, p_):
        d_ = desc_p[p_]
        return {"p": p_, "M_p/DESC": float(ss["M_p"]) / d_["M_p"], "max_int/DESC": float(ss["M_max_int"]) / d_["max_int"],
                "edge/DESC": float(ss["M_edge"]) / d_["edge"], "mean/M_desc": float(ss["M_mean"]) / M_desc_mean}

    if a.force_norm == "p":
        if a.model in ("hvp", "hybrid"):
            raise SystemExit("--force-norm p needs --model beta or gn (no exact HVP of the p-norm)")
        rho_pts = np.asarray(fr.rtz[:, 0])
        assert len(rho_pts) == A.n_pts, (len(rho_pts), A.n_pts)
        pmask = rho_pts < a.rho_cut
        tg = fr.tgrid
        absF_desc = np.asarray(inp0.compute_tensor(["|F|"], tg.rho, tg.theta, tg.zeta)["|F|"]).ravel()
        assert absF_desc.size == A.n_pts, (absF_desc.size, A.n_pts)
        w_pts = np.asarray(sw) ** 2
        p_list = list(a.p_schedule or [a.p])
        desc_p = {p_: mff.desc_pnorm(absF_desc, w_pts, pmask, p_, norm) for p_ in p_list}
        run["pnorm"] = {"rho_cut": a.rho_cut, "interior points": int(pmask.sum()), "points": int(A.n_pts),
                        "interior volume": float(w_pts[pmask].sum()), "DESC": {str(k): v for k, v in desc_p.items()},
                        "M_desc (mean)": M_desc_mean}
        for p_, d_ in desc_p.items():
            log(f"[p-norm] DESC on the same points, p {p_:g}: interior p-norm {d_['M_p']:.4e}, interior max "
                f"{d_['max_int']:.4e}, interior mean {d_['mean_int']:.4e}, edge mean {d_['edge']:.4e} (/ norm); "
                f"interior rho < {a.rho_cut}: {int(pmask.sum())} of {A.n_pts} points, volume {w_pts[pmask].sum():.3f}")
        A.set_pnorm(p_list[0], pmask, exact_beta=not a.p_model_beta)
        M_desc = desc_p[p_list[0]]["M_p"]
        if a.edge_alpha is not None:
            set_pn(p_list[0], a.alphas[0])
            log(f"[p-norm] edge bound: mean over rho >= {a.rho_cut} <= {a.edge_alpha:g} x DESC edge mean "
                f"{desc_p[p_list[0]]['edge']:.4e}, composite q = 16")

    t1 = time.time()
    z = u0
    s0 = A.state(z)
    run["start"] = {"M": float(s0["M"]), "M/M_desc": float(s0["M"]) / M_desc,
                    "rms|F|": float(s0["rmsF"]), "galerkin": float(s0["galerkin"]),
                    "rms J_fit": rmsJ, "M_desc": M_desc}
    log(f"start: M/M_desc {run['start']['M/M_desc']:.4f}, rms|F| {run['start']['rms|F|']:.3e}, "
        f"Galerkin {run['start']['galerkin']:.1e}")
    phase("start state (first compile)", t1)
    t1 = time.time()
    s0 = A.state(z)
    jax.block_until_ready(s0["M"])
    run["setup"]["state eval [s]"] = time.time() - t1

    if a.cross:
        t1 = time.time()
        kwl = ({"L": a.cross_LMN[0], "M": a.cross_LMN[1], "N": a.cross_LMN[2]} if a.cross_LMN else
               {"factors": tuple(a.cross_factors)})
        lo = fit_cache.fit(BoundaryPotential, inp0, **kwl)
        idx = mff.embed_index(lo, cur0)
        u_lo = np.load(a.cross)["u"]
        assert u_lo.shape == lo.u.shape, (u_lo.shape, lo.u.shape)
        eps = a.alphas[0] * M_desc
        cases = {"J_fit (this basis)": np.asarray(u0), "J_fit (low basis, embedded)": mff.embed(lo, cur0, lo.u, idx),
                 "low-res optimum (embedded)": mff.embed(lo, cur0, u_lo, idx)}
        if a.cross_ref:
            cases["this-res optimum"] = np.load(a.cross_ref)["u"]
        G, out = {}, {}
        for name, zz in cases.items():
            sc = A.state(jnp.asarray(zz))
            G[name] = sc["g"]
            out[name] = {"|J-J_fit|/rms": float(jnp.sqrt(jnp.sum(sc["g"] ** 2))), "M/M_desc": float(sc["M"]) / M_desc,
                         "M/eps": float(sc["M"]) / eps, "rms|F|": float(sc["rmsF"]), "galerkin": float(sc["galerkin"])}
            log(f"[cross] {name}: {out[name]}")
        d_sol = G["low-res optimum (embedded)"] - G["J_fit (low basis, embedded)"]
        d_tgt = G["J_fit (low basis, embedded)"]
        dec = {"|J_lo - J_fit_lo| (solution deviation) ^2": float(jnp.sum(d_sol**2)),
               "|J_fit_lo - J_fit_hi| (target change) ^2": float(jnp.sum(d_tgt**2)),
               "2 <cross> ": float(2 * jnp.sum(d_sol * d_tgt)),
               "|J_lo - J_fit_hi| ^2 (total)": float(jnp.sum(G["low-res optimum (embedded)"] ** 2))}
        dec["sqrt: solution, target, total"] = [float(np.sqrt(dec[k])) for k in list(dec)[:2]] + \
            [float(np.sqrt(dec["|J_lo - J_fit_hi| ^2 (total)"]))]
        log(f"[cross] decomposition: {dec}")
        run["cross"] = {"low basis": [lo.basis.L, lo.basis.M, lo.basis.N], "low n_u": int(lo.n_free),
                        "file": a.cross, "eps/M_desc": a.alphas[0], "cases": out, "decomposition": dec,
                        "time [s]": time.time() - t1}
        run["stage"] = "cross evaluation"
        dump()
        log(f"cross evaluation done in {time.time() - t1:.0f} s")
        return

    if a.eval_only:
        if a.force_norm != "p":
            raise SystemExit("--eval-only needs --force-norm p")
        cases = [("start (J_fit)", u0)]
        if a.start:
            ue = np.load(a.start)["u"]
            if a.start_LMN:
                lo = fit_cache.fit(BoundaryPotential, inp0, L=a.start_LMN[0], M=a.start_LMN[1], N=a.start_LMN[2])
                ue = mff.embed_truncate(lo, cur0, ue)[0]
            cases.append((os.path.basename(a.start), jnp.asarray(ue)))
        for spec in a.eval_state:
            nm_, pth = spec.split("=", 1)
            cases.append((nm_, jnp.asarray(np.load(pth)["u"])))
        run["eval"] = {}
        for nm_, zz in cases:
            run["eval"][nm_] = []
            bins = np.array([x for x in (0.0, 0.5, 0.6, 0.7, 0.75, 0.78, 0.81, 0.84, 0.87, 0.9, 0.93) if x < a.rho_cut] + [a.rho_cut])
            for p_ in p_list:
                A.set_pnorm(p_, pmask, exact_beta=not a.p_model_beta)
                ss = A.state(zz)
                rep = pn_report(ss, p_)
                aF_ = np.asarray(ss["absF"])
                iF = np.where(pmask, aF_, -1.0)
                imax = int(np.argmax(iF))
                q_ = np.where(pmask, w_pts * (aF_ / aF_[imax]) ** p_, 0.0)
                share = np.histogram(rho_pts, bins=bins, weights=q_)[0] / q_.sum()
                top = np.argsort(iF)[-20:]
                rep.update({"rho of max": float(rho_pts[imax]), "theta, zeta of max": [float(fr.rtz[imax, 1]), float(fr.rtz[imax, 2])],
                            "rho of top-20 points": [float(x) for x in np.sort(rho_pts[top])],
                            "share of |F|^p by rho bin": dict(zip([f"{bins[i]:.2f}-{bins[i + 1]:.2f}" for i in range(len(bins) - 1)],
                                                                 [float(x) for x in share]))})
                run["eval"][nm_].append(rep)
                if p_ == p_list[-1] or p_ == 16:
                    log(f"[eval] {nm_}, p {p_:g}: max at rho {rep['rho of max']:.3f}; top-20 rho "
                        f"{np.round(np.sort(rho_pts[top]), 3).tolist()}; share of |F|^p by rho bin "
                        + ", ".join(f"{k}: {v:.2f}" for k, v in rep["share of |F|^p by rho bin"].items() if v >= 0.01))
                log(f"[eval] {nm_}, p {p_:g}: M_p / DESC {rep['M_p/DESC']:.4f}, interior max / DESC "
                    f"{rep['max_int/DESC']:.4f}, edge mean / DESC {rep['edge/DESC']:.4f}, mean / M_desc "
                    f"{rep['mean/M_desc']:.4f}")
        run["stage"] = "eval only"
        dump()
        log(f"eval only: done in {time.time() - t0:.0f} s")
        return

    t1 = time.time()
    lean = a.precond == "force" and not a.no_lean_metric
    if lean:  # H_J to host memory; the device copy becomes the first metric factor in place
        HJ = mff.gram_J_lean(fields, cur0, pre0, rmsJ, chunk=a.gram_chunk, log=log)
        HJ_host = np.asarray(HJ)
        L = mff.metric_L_lean(HJ)
        del HJ
        HJ = None
    else:
        HJ = mff.gram_J(fields, cur0, pre0, rmsJ, chunk=a.gram_chunk, log=log)
        L = mff.metric_L(HJ)
        if a.precond != "force":
            del HJ
    LC = None
    if a.pin_current or a.pin_flux:
        t1 = time.time()
        LC = mff.LinearConstraints(cur0, fields, fr, u0, pin_current=a.pin_current, pin_flux=a.pin_flux, log=log)
        phase("linear constraints", t1)
    mk_metric = (lambda LL: LC.metric(LL)) if LC is not None else pw.Metric
    met = mk_metric(L)
    if lean and not a.subproblem_test:
        del L  # the metric object holds the only reference (released before each rebuild)
    phase("metric (H_J by slices, Cholesky)", t1)

    W = mf.Weights(a.alphas[0] * M_desc, 1.0, 1.0, rho=a.rho0, band=a.band)
    if a.start:
        st = np.load(a.start)
        z = jnp.asarray(st["u"])
        if a.start_LMN:
            lo = fit_cache.fit(BoundaryPotential, inp0, L=a.start_LMN[0], M=a.start_LMN[1], N=a.start_LMN[2])
            zh, frac, nd = mff.embed_truncate(lo, cur0, st["u"])
            z = jnp.asarray(zh)
            sz = A.state(z)
            log(f"start embedded from {tuple(a.start_LMN)} (n_u {lo.n_free}) into this basis (n_u {A.n_u}): "
                f"{nd} coefficients dropped (|u| fraction {frac:.2e}); M/M_desc {float(sz['M']) / M_desc:.4f}, "
                f"M/eps {float(sz['M']) / (a.alphas[0] * M_desc):.4f}, |J - J_fit|/rms "
                f"{float(jnp.linalg.norm(sz['g'])):.5f}")
            run["start embed"] = {"from": list(a.start_LMN), "dropped": nd, "|u| fraction": frac,
                                  "M/M_desc": float(sz["M"]) / M_desc}
            del sz
        if "nu" in st:
            W.nu, W.rho = np.asarray(st["nu"], float), max(float(st["rho"]), a.rho0)
        log(f"start from {a.start}: nu {W.nu.tolist()}, rho {W.rho}")
    if LC is not None:
        v0 = LC.violation(z)
        s_b = A.state(z)
        z = LC.restore(z, met.L)
        s_a = A.state(z)
        v1 = LC.violation(z)
        cons_log = lambda tag, sx, vx: log(  # noqa: E731
            f"[constraints] {tag}: " + ", ".join(f"{k} {v:.3e}" for k, v in vx.items())
            + f"; M/M_desc {float(sx['M']) / M_desc:.5f}, |J - J_fit|/rms {float(jnp.linalg.norm(sx['g'])):.5f}")
        cons_log("start before restore", s_b, v0)
        cons_log("start after restore", s_a, v1)
        if "dPsi/Psi" in v1 and LC.psi is not None:
            log(f"[constraints] Psi_bar check at the restored start: direct {float(LC.psi(z)):.9e}, "
                f"affine {LC.psi_affine(z):.9e}, target {LC.psi_ref:.9e}")
        run["constraints"] = {"pin_current": bool(a.pin_current), "pin_flux": bool(a.pin_flux),
                              "n_pinned": int(len(LC.idx)), "before": v0, "after": v1,
                              "Psi_ref": getattr(LC, "psi_ref", None), "Psi_desc": float(inp0.eq.Psi)}
        del s_b, s_a
    # one product of each kind, timed (after compile)
    rows = mff.FixedRows(A, s0, W, norm)
    v = jnp.ones(A.n_u) * 1e-6
    for k in range(2):
        t1 = time.time()
        jax.block_until_ready(rows.J(v))
        tJ = time.time() - t1
        t1 = time.time()
        jax.block_until_ready(rows.JT(rows.r))
        tJT = time.time() - t1
    run["setup"]["J v [s]"], run["setup"]["J^T w [s]"] = tJ, tJT
    log(f"products: J v {tJ:.3f} s, J^T w {tJT:.3f} s")
    phase("products", time.time())
    del rows
    if a.setup_only:
        run["stage"] = "setup only"
        dump()
        log(f"setup only: done in {time.time() - t0:.0f} s")
        return

    mff.use_fixed_rows()  # placeholder; the step model is installed after the solver exists
    name = os.path.basename(a.out)
    solver = mff.FixedSolver(A, met, W, norm, step="gltr", dmax=lambda xi: 0.0, max_disp=1e9,
                       reanchor=None, cg_tol=a.cg_tol, cg_maxiter=a.cg,
                       stall_dir=os.path.dirname(a.out) or ".", run_name=name)
    solver.diagnose = lambda *args, **kw: {"skipped": "fixed boundary"}
    solver.delta0 = a.delta0
    solver.delta_reset = not a.no_delta_reset
    best = {"z": z, "s": s0, "n": 0}

    build_state = {"psi1": None}

    def refresh(zz):
        t1 = time.time()
        ss, _, _, c_ = solver.evaluate(zz)
        psi1 = W.d1(c_[0], W.nu[0])
        p0 = build_state["psi1"]
        if p0 is not None and p0 > 0 and psi1 > 0 and max(psi1 / p0, p0 / psi1) < a.outer_rebuild_factor:
            log(f"  [metric] kept at the outer update (psi' {p0:.3e} -> {psi1:.3e}, < x{a.outer_rebuild_factor:g})")
            return None
        build_state["psi1"] = psi1
        rr = mff.FixedRows(A, ss, W, norm)
        if lean:
            solver.met.L = None  # one factor on the device at a time
            Lf, tr = mff.force_metric_lean(A, cur0, pre0, HJ_host, ss, W, norm, rows=rr, chunk=a.gram_chunk, log=log)
        else:
            Lf, tr = mff.force_metric(A, cur0, pre0, HJ, ss, W, norm, rows=rr, chunk=a.gram_chunk, log=log)
        mt = mk_metric(Lf)
        Dn = float(jnp.linalg.norm(mt.back_T(rr.JT(rr.r))))
        run.setdefault("metric refreshes", []).append({"t [s]": time.time() - t1, "trace H_J": tr[0],
                                                       "trace H_F": tr[1], "Delta": Dn, **mem()})
        log(f"[metric] force metric built in {time.time() - t1:.0f} s; mem {mem()}")
        dump()
        return mt, Dn

    def refresh_rows(rows):
        ss = rows.s
        build_state["psi1"] = rows.W.d1(float(rows.c[0]), rows.W.nu[0])
        rr = mff.FixedRows(A, ss, rows.W, norm)
        t1 = time.time()
        if lean:
            solver.met.L = None  # one factor on the device at a time (|L_old^-1 g| already taken)
            Lf, tr = mff.force_metric_lean(A, cur0, pre0, HJ_host, ss, rows.W, norm, rows=rr, chunk=a.gram_chunk,
                                           log=log)
        else:
            Lf, tr = mff.force_metric(A, cur0, pre0, HJ, ss, rows.W, norm, rows=rr, chunk=a.gram_chunk, log=log)
        mt = mk_metric(Lf)
        Dn = float(jnp.linalg.norm(mt.back_T(rows.JT(rows.r))))
        del mt
        run.setdefault("metric refreshes", []).append({"t [s]": time.time() - t1, "trace H_J": tr[0],
                                                       "trace H_F": tr[1], "Delta": Dn, "inner": True, **mem()})
        dump()
        return Lf, Dn

    if a.precond == "force":
        solver.metric_refresh = refresh
    solver.mult_ls = not a.no_mult_ls
    mff.use_models(a.model, far=a.hvp_far, fallback=a.hvp_fallback, beta=a.beta, solver=solver,
                   refresh=refresh_rows if a.precond == "force" else None,
                   refresh_lz=a.refresh_lz, refresh_dM=a.refresh_dM, beta_adapt=a.beta_adapt,
                   beta_min=a.beta_min, log=log, lz_stop=a.lanczos_stop)
    log(f"[model] {a.model}" + (f" (beta {a.beta})" if a.model == "beta" else "")
        + (f" (GN while |c_F| > {a.hvp_far}, GN x{a.hvp_fallback} after a radius cut)" if a.model == "hybrid" else "")
        + (" adaptive" if a.beta_adapt else "")
        + f"; metric {a.precond}" + (f" (rebuild: Lanczos cap or rent-or-buy; outer if psi' changed > x{a.outer_rebuild_factor:g})"
                                     if a.precond == "force" else "")
        + f"; LS multiplier {'on' if solver.mult_ls else 'off'}; Lanczos stop {a.lanczos_stop}")
    if a.subproblem_test:
        W.eps = a.alphas[0] * M_desc
        s, f, phi, c = solver.evaluate(z)
        if a.test_model == "beta":
            rows = type("RowsBeta", (mff.FixedRowsB,), {"beta": a.beta})(A, s, W, norm)
        else:
            rows = mff.FixedRowsH(A, s, W, norm)
        r = rows.r
        f0 = 0.5 * float(r @ r)
        grad = rows.JT(r)
        Hv = rows.hvp if a.test_model == "hvp" else (lambda v: rows.JT(rows.J(v)))
        L_hj = L  # the H_J metric factor: |L_hj^T d| measures every step in the same units
        tdelta = a.test_delta
        if a.precond == "force":
            met, D_f = refresh(z)
            if tdelta <= 0:
                tdelta = 10.0 * D_f
            log(f"[subproblem test] force metric: |L^-1 grad| = {D_f:.3e}; test Delta {tdelta:.3e} (force-metric units)")
        log(f"[subproblem test] model {a.test_model}, Delta {tdelta:.2e}, f {f:.6e}, M/eps {float(s['M']) / W.eps:.4f}, "
            f"nu {W.nu.tolist()}, rho {W.rho}")
        out = []
        for cap in a.subproblem_test:
            t1 = time.time()
            y, m_step, hit, k, share, gn, mu = mff.lanczos_tr_step_H(Hv, grad, f0, met, tdelta, a.cg_tol, cap)
            tl = time.time() - t1
            d = dict(mff.LAST_H)
            s_new, f_new, phi_new, c_new = solver.evaluate(z + met.back(y))
            pred, act = f0 - m_step, f - f_new
            e = {"cap": cap, "lanczos": int(k), "converged": d["converged"], "res/|g|": d["res"],
                 "theta_min": d["theta_min"], "theta_max": d["theta_max"], "kappa_eff": d["theta_max"] / max(d["theta_min"], 1e-300),
                 "|y|": float(jnp.linalg.norm(y)), "|d|_HJ": float(jnp.linalg.norm(L_hj.T @ met.back(y))),
                 "hit": bool(hit), "pred": pred, "act": act,
                 "ratio": act / pred if pred > 0 else float("nan"),
                 "M/eps new": float(s_new["M"]) / W.eps, "phi new": float(phi_new), "t [s]": tl}
            out.append(e)
            log(f"[subproblem test] cap {cap}: lanczos {e['lanczos']} conv {e['converged']} res/|g| {e['res/|g|']:.2e} "
                f"theta [{e['theta_min']:.2e}, {e['theta_max']:.2e}] |y| {e['|y|']:.3e} |d|_HJ {e['|d|_HJ']:.3e} hit {e['hit']} "
                f"pred {pred:.4e} act {act:.4e} ratio {e['ratio']:.3f} M/eps {e['M/eps new']:.4f} t {tl:.0f}s")
            run["subproblem test"] = {"model": a.test_model, "precond": a.precond, "Delta": tdelta, "f": f,
                                      "M/eps": float(s["M"]) / W.eps, "results": out}
            dump()
        log("subproblem test done")
        return
    try:
        stop = False
        for p_cur in p_list:
            ptag = "" if p_cur is None else f"p{p_cur:g}_"
            if p_cur is not None:
                A.set_pnorm(p_cur, pmask, exact_beta=not a.p_model_beta)
                M_desc = desc_p[p_cur]["M_p"]
                if a.edge_alpha is not None:
                    set_pn(p_cur, a.alphas[0])
                build_state["psi1"] = None  # new weights: rebuild the force metric at the first outer
                log(f"[p-norm] p {p_cur:g}: eps = alpha x DESC interior p-norm {M_desc:.4e}")
            for alpha in a.alphas:
                t1 = time.time()
                W.eps = alpha * M_desc
                if p_cur is not None and a.edge_alpha is not None:
                    set_pn(p_cur, alpha)
                log(f"alpha {alpha}: eps {W.eps:.4e}")
                entry = {"alpha": alpha, "eps": W.eps, "p": p_cur}
                run["alphas"].append(entry)

                def cb(zz, s, outer, entry=entry):
                    entry["outer"] = outer
                    entry["time"] = time.time() - t1
                    entry.pop("inner (current outer)", None)
                    run["memory"].append({"phase": f"alpha {alpha} outer {len(outer)}",
                                          "t": time.time() - t0, **mem()})
                    save_state(f"{ptag}a{alpha}_latest", zz, s, W)
                    dump()

                def on_step(hist, entry=entry):
                    entry["inner (current outer)"] = hist
                    if hist and hist[-1]["accepted"] and A.pnorm is not None:
                        rep = pn_report(A.last_s, A.pnorm["p"])
                        hist[-1]["pnorm"] = rep
                        log(f"      [p-norm] p {rep['p']:g}: M_p/DESC {rep['M_p/DESC']:.4f}, interior max/DESC "
                            f"{rep['max_int/DESC']:.4f}, edge mean/DESC {rep['edge/DESC']:.4f}, mean/M_desc "
                            f"{rep['mean/M_desc']:.4f}")
                    if hist and hist[-1]["accepted"] and LC is not None:
                        vx = LC.violation(A.last_z)
                        hist[-1]["constraints"] = vx
                        log("      [constraints] " + ", ".join(f"{k} {v:.2e}" for k, v in vx.items()))
                    if hist and hist[-1]["accepted"]:
                        best["z"], best["s"] = A.last_z, A.last_s
                        best["n"] += 1
                        if best["n"] % 5 == 0:
                            save_state(f"{ptag}a{alpha}_latest", best["z"], best["s"], W)
                    dump()
                    if a.time_budget and time.time() - t0 > a.time_budget:
                        raise Budget()

                solver.on_step = on_step
                z, outer = solver.solve_alpha(z, max_outer=a.max_outer, maxiter=a.maxiter,
                                              gtol=a.gtol, callback=cb, stat_tol=a.stat_tol,
                                              stat_window=a.stat_window)
                s = A.state(z)
                c = W.constraints(s)
                entry.update({"M/M_desc": float(s["M"]) / M_desc, "rms|F|": float(s["rmsF"]),
                              "|J-J_fit|/rms": float(jnp.sqrt(jnp.sum(s["g"] ** 2))),
                              "galerkin": float(s["galerkin"]), "feasible": bool(c[0] <= 1e-2),
                              "nu": W.nu.tolist(), "rho": W.rho, "time": time.time() - t1})
                if A.pnorm is not None:
                    entry["pnorm"] = pn_report(s, p_cur)
                if LC is not None:
                    entry["constraints"] = LC.violation(z)
                    if LC.psi is not None:
                        entry["constraints"]["dPsi/Psi direct"] = (float(LC.psi(z)) - LC.psi_ref) / LC.psi_ref
                save_state(f"{ptag}a{alpha}", z, s, W)
                log(f"alpha {alpha} done in {time.time() - t1:.0f} s: |J - J_fit|/rms "
                    f"{entry['|J-J_fit|/rms']:.4e}, M/M_desc {entry['M/M_desc']:.4f}")
                dump()
                if not entry["feasible"]:
                    log(f"alpha {alpha} not met: stopping the sweep")
                    stop = True
                    break
            if stop:
                break
        run["stage"] = "sweep ended"
    except Budget:
        save_state("budget", best["z"], best["s"], W)
        run["stage"] = "time budget reached"
        log("time budget reached: state saved")
    run["total [s]"] = time.time() - t0
    run["memory"].append({"phase": "end", "t": time.time() - t0, **mem()})
    dump()
    log(f"saved {a.out}.json; total {time.time() - t0:.0f} s; mem {mem()}")


if __name__ == "__main__":
    main()
