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
from objective import Fields, _cartesian  # noqa: E402
from optimize_fb import annulus  # noqa: E402
from pressure import Pressure  # noqa: E402

D = os.path.join(ROOT, "desc_input", "vmec_free_boundary", "NCSX")
log = functools.partial(print, flush=True)


class NextStage(Exception):
    """End the current alpha stage (control file NEXT, or --stage-flat) and continue with the next alpha."""


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
    ap.add_argument("--bext", choices=("coils", "desc-fit", "desc-J"), default="coils",
                    help="external field: the coils (default); desc-fit: B_DESC - B_ann[J_fit] (the start is B_DESC "
                         "at the points); desc-J: B_DESC - B_ann[J_DESC] (vacuum up to the annulus error). Both are "
                         "built with this run's own Fields (annulus, N_phi, interpolation) at the collocation points")
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
    ap.add_argument("--anatomy", type=int, default=None,
                    help="with the subproblem test's model at the --start state: one Lanczos run to this many "
                         "iterations, then every tolerance / radius from it (scripts/meanforce/subproblem_anatomy.py)")
    ap.add_argument("--stiff-modes", type=int, nargs=3, default=None, metavar=("K_FULL", "K_LOC", "NVEC"),
                    help="test 2 at the --start state: Lanczos on the full and the local (no J0 x dB) step model in "
                         "the same metric, and the anatomy of the top/bottom NVEC Ritz vectors (scripts/meanforce/stiff_modes.py)")
    ap.add_argument("--line-scan", type=float, nargs="+", default=None,
                    help="with --anatomy: scan M exactly along the converged steps at these radius multiples "
                         "(scripts/meanforce/line_anatomy.py), t in [0, --line-tmax]")
    ap.add_argument("--line-tmax", type=float, default=2.0)
    ap.add_argument("--chord-study", type=int, default=None,
                    help="at the --start state (nu, rho from it): K production inner steps keeping every iterate, then the "
                         "merit gradient along each chord and a steepest-descent line probe at each iterate (line_anatomy.chord_study)")
    ap.add_argument("--accel", action="store_true", help="with --line-scan: geodesic acceleration along each step and an "
                    "exact scan of M along the curved path (line_anatomy.accel_scan)")
    ap.add_argument("--anatomy-deltas", type=float, nargs="+", default=[3.0, 30.0, 1e4],
                    help="radii as multiples of |L^-1 grad| (force metric)")
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
    ap.add_argument("--geo-accel", action="store_true",
                    help="geodesic acceleration on trust-region boundary steps (trial z + d + a, a = -(J^T J)^-1 J^T q)")
    ap.add_argument("--accel-amax", type=float, default=0.75, help="--geo-accel: skip when |a|/|y| (metric) exceeds this")
    ap.add_argument("--accel-tol", type=float, default=1e-4, help="--geo-accel: Lanczos tolerance of the a solve")
    ap.add_argument("--accel-maxiter", type=int, default=300, help="--geo-accel: Lanczos cap of the a solve")
    ap.add_argument("--radius-by-delta", action="store_true",
                    help="after an accepted interior step with ratio < 0.25, shrink the radius to max(0.25|y|, 0.25 Delta)")
    ap.add_argument("--f-noise", default=None,
                    help="noise-tolerant trust-region ratio: an absolute sigma_f, or 'auto' (measured at the start)")
    ap.add_argument("--no-share-stop", action="store_true",
                    help="do not end the inner solve on a small model decrease of the Krylov minimizer (share < 1e-8)")
    ap.add_argument("--keep-start-mult", action="store_true",
                    help="with --ref-start: keep the --start file's nu and rho instead of resetting them")
    ap.add_argument("--stat-tol", type=float, default=1e-3)
    ap.add_argument("--stat-window", type=int, default=10)
    ap.add_argument("--delta0", type=float, default=1e-3, help="first trust radius (and the radius at each outer reset)")
    ap.add_argument("--no-delta-reset", action="store_true",
                    help="keep the trust radius across outer iterations / bounds (old behaviour)")
    ap.add_argument("--gram-chunk", type=int, default=1024)
    ap.add_argument("--time-budget", type=float, default=None, help="seconds; stop cleanly")
    ap.add_argument("--stall-stop", type=float, default=None,
                    help="end the sweep when the bound is unmet and M/eps changed by less than this fraction over an outer")
    ap.add_argument("--stage-flat", type=float, default=None,
                    help="continuation: end a stage that is not the last alpha once the bound is met (M/eps <= 1.01) and "
                         "phi fell by less than this fraction over the last outer iteration")
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
    ap.add_argument("--p-band", type=float, nargs=2, default=None, metavar=("LO", "HI"),
                    help="--force-norm p: the p-norm region is the DESC-rho band LO <= rho <= HI instead of rho < rho-cut; "
                         "with --edge-alpha the composite then also bounds the mean over everything outside the band")
    ap.add_argument("--mean-whole", action="store_true",
                    help="--force-norm p with --edge-alpha: the composite's second term is the whole-volume mean (not the "
                         "mean outside the band/interior), so the volume mean is bounded too (strictly tighter than a mean bound)")
    ap.add_argument("--ref-start", action="store_true",
                    help="--force-norm p: normalize alpha and --edge-alpha by the --start state's own p-norm and "
                         "outside mean (after embedding) instead of DESC's; logged ratios '/DESC' then mean '/start'")
    ap.add_argument("--ref-state", default=None,
                    help="--ref-start: take the reference band p-norm and mean from this npz (this run's basis) instead "
                         "of the --start state (e.g. warm-start a tighter run while keeping the original reference)")
    ap.add_argument("--colloc-rho", type=float, default=None,
                    help="radial collocation factor (Gauss-Legendre radii = colloc-rho (L + L_eq) / 2 + 1); "
                         "default --colloc. Angles keep --colloc")
    ap.add_argument("--snap-outer", action="store_true", help="also save the state after every outer iteration (_o<k>)")
    ap.add_argument("--shell-max", type=float, nargs=2, default=None, metavar=("LO", "HI"),
                    help="--eval-only: per collocation radius in [LO, HI], the max and mean of |F| over the shell "
                         "(/ norm) for DESC and each state")
    ap.add_argument("--eval-save", action="store_true",
                    help="--eval-only: save each evaluated state (u and its least-squares pressure c_p) as <out>_eval_<name>.npz")
    ap.add_argument("--p-model-beta", action="store_true",
                    help="--force-norm p: use --beta along F_i instead of the exact p - 1")
    ap.add_argument("--eval-dump", action="store_true",
                    help="--eval-only: save J, B, F (cylindrical-point arrays) of every evaluated state, with the points' "
                         "rtz, weights and norm, as <out>_dump.npz (same collocation on every --frac: diff across grids)")
    ap.add_argument("--subspace", default=None,
                    help="exact merit search on each trust-region boundary step over span{d, ...}: 'prev' (previous "
                         "accepted step), 'accel' (geodesic-acceleration direction), or 'prev,accel'")
    ap.add_argument("--subspace-smax", type=float, default=4.0, help="--subspace: grid of the d coefficient, [0, smax]")
    ap.add_argument("--subspace-tmax", type=float, default=2.0,
                    help="--subspace: grid of the other coefficients, [-tmax, tmax] (directions scaled to |d|)")
    ap.add_argument("--subspace-all", action="store_true", help="--subspace: also on interior steps")
    ap.add_argument("--no-rebuild-on-hit", action="store_true",
                    help="force metric: no inner rebuild after a subproblem whose step hit the trust-region boundary "
                         "(no cap trigger, no rent-or-buy time); interior cap hits still rebuild")
    ap.add_argument("--locate-chunk", type=int, default=1 << 20,
                    help="points per Newton batch in locate_plasma_planes (its transient is chunk x DESC modes x 8 B)")
    ap.add_argument("--mem-every", type=int, default=0,
                    help="append JAX in-use/peak memory to every K-th inner-step log line (0: off)")
    ap.add_argument("--eval-only", action="store_true",
                    help="--force-norm p: report the start (J_fit) and --start states' p-norms, interior max, "
                         "edge and mean against DESC's, for each p of the schedule, then exit (no metric build)")
    ap.add_argument("--rho-bins", type=float, nargs="+", default=None,
                    help="--eval-only: also print the force density per rho bin relative to the volume mean, for each state and DESC")
    ap.add_argument("--eval-blend", type=float, nargs="+", default=[],
                    help="--eval-only: also evaluate J_fit + s (J - J_fit) for each state and each s (grid error is linear in J - J_fit)")
    ap.add_argument("--eval-state", action="append", default=[],
                    help="--eval-only: extra states name=path.npz (this run's basis); repeatable")
    ap.add_argument("--out", default=os.path.join(ROOT, "results", "meanforce", "fixed"))
    ap.add_argument("--mirror", default=None, help="directory for fsynced copies (Volume)")
    ap.add_argument("--pin-b21", type=float, default=None,
                    help="hold the resonant field b_mk at q res at BETA x the vacuum amplitude, along the island phase "
                         "(an affine row: b_mk is linear in u under option A)")
    ap.add_argument("--b21-mk", type=int, nargs=2, default=(2, 1), help="(m, k) of the resonant harmonic (start_check)")
    ap.add_argument("--b21-vac", type=float, default=3.778e-4, help="vacuum |b_mk| at the resonance (start_check)")
    ap.add_argument("--diag-every", type=int, default=0,
                    help="log a [diag] line every K accepted steps (also at the start, at every outer end and at the "
                         "end): M/M_desc, |J - J_fit|/rms, b_mk (/ vacuum, phase), and |F| in the --diag-band "
                         "(p16, max) and outside it (mean), each / DESC's on the same points. 0 = off")
    ap.add_argument("--diag-band", type=float, nargs=2, default=(0.5675, 0.7075), metavar=("LO", "HI"),
                    help="--diag-every: the DESC-rho band of the resonant surface (default: q = 2 of the ripple case)")
    ap.add_argument("--save-every", type=int, default=5,
                    help="write <out>_a<alpha>_latest.npz every K accepted steps (default 5)")
    ap.add_argument("--b21-phase-state", default=None,
                    help="npz whose b_mk sets the island phase (an island seed); default: the phase of --start")
    ap.add_argument("--b21-report", action="store_true", help="--eval-only: print b_mk at q res for each state")
    ap.add_argument("--plane", action="append", default=[],
                    help="--eval-only: exact map of the force on a plane, 'name|base|d1+|d1-|d2+|d2-|s0,s1,ns|t0,t1,nt' "
                         "(npz paths or J_fit; d = u(d+) - u(d-)); repeatable")
    ap.add_argument("--pin-current", action="store_true",
                    help="exact constraint: the m = 0 modes of a stay at J_fit's (the current enclosed by every "
                         "DESC surface, in every cross-section); a --start state is restored onto it")
    ap.add_argument("--freeze-n0-band", type=float, nargs=2, default=None, metavar=("RHO1", "RHO2"),
                    help="exact constraint: freeze the n = 0 (toroidally symmetric) part of J outside rho in [RHO1, RHO2] "
                         "(its current-norm directions with more than --freeze-tau of their norm outside the band stay at "
                         "the --freeze-ref state's values); n != 0 and the band's n = 0 content stay free")
    ap.add_argument("--freeze-ref", default=None, help="state (npz) whose n = 0 J outside the band is frozen (default: "
                    "--start, else J_fit)")
    ap.add_argument("--freeze-tau", type=float, default=1e-2, help="outside-band norm fraction above which an n = 0 "
                    "direction is frozen (--freeze-n0-band)")
    ap.add_argument("--pin-flux", action="store_true",
                    help="exact constraint: the toroidally averaged toroidal flux stays at J_fit's")
    ap.add_argument("--energy-weight", type=float, default=0.0,
                    help="lambda: add lambda (W_Omega - W_Omega(J_fit)) / W_Omega(J_fit) to the objective (plasma-volume "
                         "magnetic energy on the collocation points). The energy is the free energy only with "
                         "--bn-modes and --pin-flux (B.n = 0 on the LCFS, Phi_tor and I_p fixed). Use with --no-share-stop")
    ap.add_argument("--bn-modes", type=int, nargs=2, default=None, metavar=("MC", "NC"),
                    help="exact constraint: the Fourier modes |m| <= MC, |n| <= NC of sqrt(g) dB.e^rho on the LCFS "
                         "vanish (B.n = 0 there; dB = B_ann[J - J_fit], option A's start field is DESC's)")
    a = ap.parse_args()
    a.hvp = True  # per-subproblem diagnostics (DIAG_H) are recorded for every model
    os.makedirs(os.path.dirname(a.out) or ".", exist_ok=True)
    if a.mirror:
        os.makedirs(a.mirror, exist_ok=True)
    if a.mem_every:  # memory on every K-th inner-step line ("    it  K ...")
        import re as _re
        _plain = globals()["log"]

        def _log_mem(msg="", *args_, **kw_):
            m_ = _re.match(r"\s+it\s+(\d+) ", str(msg))
            if m_ and int(m_.group(1)) % a.mem_every == 0:
                mm_ = mem()
                msg = (f"{msg} | mem in-use {mm_.get('in use GB', 0):.2f} peak {mm_.get('peak GB', 0):.2f} GB "
                       f"rss {mm_.get('host maxrss GB', 0):.1f}")
            _plain(msg, *args_, **kw_)
        globals()["log"] = _log_mem
    t0 = time.time()
    run = {"args": vars(a), "setup": {}, "alphas": [], "memory": []}

    def write(path, text):  # atomic: a kill mid-write leaves the previous file intact
        tmp = path + ".tmp"
        with open(tmp, "w") as fh:
            fh.write(text)
            fh.flush()
            os.fsync(fh.fileno())
        os.replace(tmp, path)

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
            tmp = p + ".tmp"  # atomic, so a kill mid-write never leaves a truncated checkpoint
            with open(tmp, "wb") as fh:
                np.savez(fh, u=np.asarray(z), c_p=np.asarray(s["c"]), nu=W.nu, rho=W.rho, eps=W.eps)
                fh.flush()
                os.fsync(fh.fileno())
            os.replace(tmp, p)

    def phase(name, t1):
        run["setup"][name] = time.time() - t1
        m = mem()
        run["memory"].append({"phase": name, "t": time.time() - t0, **m})
        log(f"[setup] {name}: {time.time() - t1:.1f} s; mem {m}")
        dump()

    t1 = time.time()
    inp0 = DESCInput.load(a.eq, field=a.coils if a.bext == "coils" else None)
    eq0 = inp0.eq
    kw = ({"L": a.LMN[0], "M": a.LMN[1], "N": a.LMN[2]} if a.LMN else
          {"factors": tuple(a.factors)})
    cur0 = fit_cache.fit(BoundaryPotential, inp0, **kw)
    pre0 = fit_cache.fit(Pressure, inp0, edge_order=2, **kw)
    b = cur0.basis
    res = tuple(int(round(a.colloc * (u + v))) for u, v in
                ((b.L, eq0.L), (b.M, eq0.M), (b.N, eq0.N)))
    if a.colloc_rho:
        res = (int(round(a.colloc_rho * (b.L + eq0.L))),) + res[1:]
    ref = reference(inp0, res, current=cur0)
    norm = ref["norm"]
    M_desc = ref["F_desc_vol"] / norm
    run["basis"] = [b.L, b.M, b.N]
    log(f"{inp0}\nbasis {b.L}/{b.M}/{b.N}, collocation resolution {res} {tuple(ref['shape'])}; "
        f"DESC <|F|>_vol / norm {M_desc:.4e}")
    phase("load + fits + reference", t1)

    t1 = time.time()
    box = annulus(inp0, inp0.a / a.frac, a.pad, a.N_phi)
    box.locate_plasma_planes(inp0, sym=bool(eq0.sym), verbose=True, chunk=a.locate_chunk)
    fr = ForceResidual(inp0, box, resolution=res)
    fr.norm, fr.F_desc_vol = norm, ref["F_desc_vol"]
    if a.bext != "coils":
        fr.B_ext = np.zeros((len(fr.x), 3))  # set from DESC's field once Fields exists
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
    if a.bext != "coils":
        t1 = time.time()
        B_D = np.asarray(fields.desc["B"])
        if a.bext == "desc-fit":
            B_src = np.asarray(fields.B_plasma(u0))
        else:  # DESC's J at the annulus' inside nodes, through the run's own operator
            J_rows = np.asarray(inp0.compute_points(["J"], np.asarray(fields.rows_rtz))["J"])
            Jr = fields._cyl_rows(jnp.asarray(J_rows, fields.cyl_dtype))
            if getattr(fields, "cyl_fused", False):
                itp_ = getattr(fields.to_points, "interp", fields.to_points)
                Bc = itp_.from_planes(*fields._cyl_planes(Jr, itp_.Ep))[: len(fields.phi)]
            else:
                Bc = fields.to_points(*fields.solver.field_modes_F(*fields._cyl_sources(Jr)))
            B_src = np.asarray(_cartesian(Bc, fields.phi))
        B_ext = B_D - B_src
        fields.B_ext = jnp.asarray(B_ext)
        fr.B_ext = B_ext
        wv = fr.dV / fr.dV.sum()
        rmsB = float(np.sqrt(np.sum(wv * np.sum(B_D**2, 1))))
        dB0 = np.asarray(fields.B_ext + fields.B_plasma(u0)) - B_D
        run["bext"] = {"kind": a.bext, "rms |B_ext| / rms |B_DESC|": float(np.sqrt(np.sum(wv * np.sum(B_ext**2, 1))) / rmsB),
                       "start rms |B - B_DESC| / rms |B_DESC|": float(np.sqrt(np.sum(wv * np.sum(dB0**2, 1))) / rmsB)}
        log(f"[bext] {a.bext}: {run['bext']}")
        for d_ in [os.path.dirname(a.out) or "."] + ([a.mirror] if a.mirror else []):
            np.savez(os.path.join(d_, os.path.basename(a.out) + "_bext.npz"), B_ext=B_ext, B_DESC=B_D,
                     x=np.asarray(fr.x), rtz=np.asarray(fr.rtz))
        phase(f"B_ext ({a.bext})", t1)
    Jfit = mff._J(fields, u0, fields.I)
    sw = jnp.asarray(fields.sw)
    rmsJ = float(jnp.sqrt(jnp.sum(sw[:, None] ** 2 * Jfit**2)))
    A = mff.FixedAnchor(fields, cur0, pre0, u0, rmsJ, norm, G, log=log)
    A.edge_all = bool(a.mean_whole)
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
        if a.p_band:
            pmask = (rho_pts >= a.p_band[0]) & (rho_pts <= a.p_band[1])
            log(f"[p-norm] band {a.p_band[0]:.4f} <= rho <= {a.p_band[1]:.4f}: {int(pmask.sum())} points on "
                f"{len(np.unique(rho_pts[pmask]))} radii; the 'edge' terms are the mean outside the band")
        tg = fr.tgrid
        absF_desc = np.asarray(inp0.compute_tensor(["|F|"], tg.rho, tg.theta, tg.zeta)["|F|"]).ravel()
        assert absF_desc.size == A.n_pts, (absF_desc.size, A.n_pts)
        w_pts = np.asarray(sw) ** 2
        p_list = list(a.p_schedule or [a.p])
        desc_p = {p_: mff.desc_pnorm(absF_desc, w_pts, pmask, p_, norm, edge_all=a.mean_whole) for p_ in p_list}
        if a.mean_whole:
            log("[p-norm] composite second term = the WHOLE-volume mean (--mean-whole)")
        run["pnorm"] = {"rho_cut": a.rho_cut, "interior points": int(pmask.sum()), "points": int(A.n_pts),
                        "interior volume": float(w_pts[pmask].sum()), "DESC": {str(k): v for k, v in desc_p.items()},
                        "M_desc (mean)": M_desc_mean}
        for p_, d_ in desc_p.items():
            log(f"[p-norm] DESC on the same points, p {p_:g}: interior p-norm {d_['M_p']:.4e}, interior max "
                f"{d_['max_int']:.4e}, interior mean {d_['mean_int']:.4e}, edge mean {d_['edge']:.4e} (/ norm); "
                + (f"band {a.p_band[0]:.4f}-{a.p_band[1]:.4f}" if a.p_band else f"interior rho < {a.rho_cut}")
                + f": {int(pmask.sum())} of {A.n_pts} points, volume {w_pts[pmask].sum():.3f}")
        A.set_pnorm(p_list[0], pmask, exact_beta=not a.p_model_beta)
        M_desc = desc_p[p_list[0]]["M_p"]
        if a.edge_alpha is not None:
            set_pn(p_list[0], a.alphas[0])
            log(f"[p-norm] edge bound: mean over " + ("the points outside the band" if a.p_band else f"rho >= {a.rho_cut}")
                + f" <= {a.edge_alpha:g} x DESC's mean there "
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
            G[name] = sc["gJ"]
            out[name] = {"|J-J_fit|/rms": float(jnp.sqrt(jnp.sum(sc["gJ"] ** 2))), "M/M_desc": float(sc["M"]) / M_desc,
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

    bfn = None
    if a.pin_b21 is not None or a.b21_report or a.plane or a.diag_every:
        t1 = time.time()
        bfn, binfo = mff.resonant_b_fn(inp0, fields, fr, u0, a.b21_mk[0], a.b21_mk[1], log=log)
        # not jax.jit(bfn): the closure would capture the field tables as constants; bfn calls a jitted kernel
        run["b_mk"] = binfo
        phase(f"b{a.b21_mk[0]}{a.b21_mk[1]} functional", t1)

    def bmk(uu):
        v = np.asarray(bfn(jnp.asarray(uu)))
        return complex(v[0], v[1])

    # energy reference (W_Omega of J_fit), its n-split, and the LCFS B.n functional: needed by --eval-only too
    per_ = 2 * eq0.NFP if fr.sym else eq0.NFP
    vol = float(per_ * np.sum(fr.dV))
    A.enable_energy(0.0, mf.Weights(1.0, 1.0, 1.0), vol, log=log)
    modes_ = np.asarray(cur0.basis.modes)
    n_of_u = np.concatenate([np.asarray(cur0.chi_modes)[:, 1], modes_[:, 2], modes_[:, 2]])
    assert n_of_u.size == A.n_u, (n_of_u.size, A.n_u)
    m_n0 = jnp.asarray((n_of_u == 0).astype(float))
    bnf = None
    if a.bn_modes:
        t1 = time.time()
        bnf = mff.LCFSField(inp0, fields, fr, a.bn_modes[0], a.bn_modes[1], log=log)
        phase("B.n functional", t1)

    def energy_report(zz, ss):
        """dW / W_ref and its split by the toroidal mode number of du = u - u0 (n = 0 block, n != 0 block, cross
        term), E = lambda dW / W_ref, and the LCFS B.n modes from a real field evaluation (with --bn-modes)."""
        ep = A.energy_parts(ss)
        out_ = {"dW/W": ep["dW/W"], "dW [J]": ep["dW/W"] * A.W_ref_J, "E": a.energy_weight * ep["dW/W"]}
        du = jnp.asarray(zz)[: A.n_u] - u0
        w_ = sw**2
        Bn0 = A.state(u0 + du * m_n0)["B"] - A.B_ref
        Bn1 = ss["B"] - A.B_ref - Bn0
        S_ = A.S_ref
        out_["dW/W n=0"] = float(jnp.sum(w_[:, None] * (2.0 * A.B_ref * Bn0 + Bn0**2)) / S_)
        out_["dW/W n!=0"] = float(jnp.sum(w_[:, None] * (2.0 * A.B_ref * Bn1 + Bn1**2)) / S_)
        out_["dW/W cross"] = float(jnp.sum(w_[:, None] * 2.0 * Bn0 * Bn1) / S_)
        if bnf is not None:
            bm = np.asarray(bnf.modes_of(fields, du))
            out_["Bn max|b| (field)"] = float(np.max(np.abs(bm)))
            out_["Bn rms (field)"] = float(np.sqrt(np.mean(np.asarray(bnf.f(fields, du)) ** 2))) / bnf.scale
        return out_

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
        for s_ in a.eval_blend:
            cases += [(f"{nm_}@s{s_:g}", u0 + s_ * (zz - u0)) for nm_, zz in list(cases[1:]) if "@s" not in nm_]
        run["eval"] = {}
        if a.b21_report:
            run["b_mk report"] = {}
            for nm_, zz in cases:
                bv = bmk(zz)
                run["b_mk report"][nm_] = [bv.real, bv.imag]
                log(f"[b{a.b21_mk[0]}{a.b21_mk[1]}] {nm_}: |b| {abs(bv):.4e} ({abs(bv) / a.b21_vac:.2%} of vacuum), "
                    f"phase {np.degrees(np.angle(bv)):+.1f} deg")
        for spec in a.plane:
            sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
            import plane_map as PM
            run.setdefault("planes", {})
            run["planes"][spec.split("|")[0]] = PM.plane(A, spec, u0, rho_pts, w_pts, pmask, norm, M_desc_mean,
                                                         desc_p, bmk if bfn is not None else None, a.b21_vac,
                                                         a.out, a.mirror, log)
            dump()
        if a.eval_save:
            for nm_, zz in cases:
                se_ = A.state(zz)
                tag_ = "".join(ch if ch.isalnum() or ch in "._-" else "_" for ch in nm_)
                for d_ in [os.path.dirname(a.out) or "."] + ([a.mirror] if a.mirror else []):
                    np.savez(os.path.join(d_, os.path.basename(a.out) + f"_eval_{tag_}.npz"), u=np.asarray(zz),
                             c_p=np.asarray(se_["c"]), nu=np.zeros(3), rho=1.0, eps=0.0)
                log(f"[eval] saved {nm_}: M/M_desc {float(se_['M_mean']) / M_desc_mean:.4f}, |J - J_fit|/rms "
                    f"{float(jnp.sqrt(jnp.sum(se_['g'] ** 2))):.5f}")
        if a.eval_dump:
            dd_ = {"rtz": np.asarray(fr.rtz), "w": np.asarray(w_pts), "norm": float(norm),
                   "M_desc": float(M_desc_mean), "frac": float(a.frac), "names": np.array([c_[0] for c_ in cases])}
            for k_, (nm_, zz) in enumerate(cases):
                se_ = A.state(zz)
                for key_ in ("J", "B", "F"):
                    dd_[f"{key_}_{k_}"] = np.asarray(se_[key_])
                log(f"[dump] {nm_}: M/M_desc {float(se_['M_mean']) / M_desc_mean:.4f}")
            for d_ in [os.path.dirname(a.out) or "."] + ([a.mirror] if a.mirror else []):
                np.savez(os.path.join(d_, os.path.basename(a.out) + "_dump.npz"), **dd_)
        if a.shell_max:
            ur = np.unique(rho_pts)
            ur = ur[(ur >= a.shell_max[0]) & (ur <= a.shell_max[1])]
            sel_ = [rho_pts == r_ for r_ in ur]

            def shells(absF):
                return ([float(absF[k_].max()) / norm for k_ in sel_],
                        [float(np.sum(w_pts[k_] * absF[k_]) / np.sum(w_pts[k_])) / norm for k_ in sel_])
            run["shell max"] = {"rho": ur.tolist(), "DESC": shells(absF_desc)}
            log("[shell] rho          " + " ".join(f"{x:8.4f}" for x in ur))
            mx, mn = run["shell max"]["DESC"]
            log("[shell] DESC max     " + " ".join(f"{x:8.2e}" for x in mx))
            log("[shell] DESC mean    " + " ".join(f"{x:8.2e}" for x in mn))
            for nm_, zz in cases:
                mx, mn = shells(np.asarray(A.state(zz)["absF"]))
                run["shell max"][nm_] = (mx, mn)
                log(f"[shell] {nm_[:9]:9s} max  " + " ".join(f"{x:8.2e}" for x in mx))
                log(f"[shell] {nm_[:9]:9s} mean " + " ".join(f"{x:8.2e}" for x in mn))
            dump()
        if a.rho_bins:
            rb = np.array(a.rho_bins)
            vol_b = np.histogram(rho_pts, bins=rb, weights=w_pts)[0]

            def dens(absF):
                tot = float(np.sum(w_pts * absF) / np.sum(w_pts))
                return (np.histogram(rho_pts, bins=rb, weights=w_pts * absF)[0] / np.maximum(vol_b, 1e-300)) / tot
            run["rho bins"] = {"edges": rb.tolist(), "volume": vol_b.tolist(), "DESC": dens(absF_desc).tolist()}
            log("[bins] edges " + " ".join(f"{x:.3f}" for x in rb))
            log("[bins] DESC        " + " ".join(f"{x:5.2f}" for x in dens(absF_desc)))
            for nm_, zz in cases:
                dd = dens(np.asarray(A.state(zz)["absF"]))
                run["rho bins"][nm_] = dd.tolist()
                log(f"[bins] {nm_[:11]:11s} " + " ".join(f"{x:5.2f}" for x in dd))
        run["energy eval"] = {"W_ref [J]": A.W_ref_J}
        for nm_, zz in cases:
            er_ = energy_report(zz, A.state(zz))
            run["energy eval"][nm_] = er_
            log(f"[energy] {nm_}: " + ", ".join(f"{k} {v:+.5e}" for k, v in er_.items()))
        dump()
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
    if bnf is not None:
        t1 = time.time()
        b0_ = np.asarray(bnf.modes_of(fields, jnp.zeros(A.n_u)))
        assert np.max(np.abs(b0_)) == 0.0, b0_
        Ct_bn = bnf.rows(fields, A.n_u, log=log)
        run["bn"] = {"modes": list(a.bn_modes), "k": int(bnf.k), "points": int(bnf.n_b), "scale": bnf.scale}
        phase("B.n constraint rows", t1)
    blocks_ = []
    if bnf is not None:
        blocks_.append(("Bn", Ct_bn, np.zeros(bnf.k)))
    if a.freeze_n0_band is not None:
        t1 = time.time()
        Ct_fz, fz_info = mff.frozen_n0_block(fields, fr, cur0, a.freeze_n0_band, a.freeze_tau, log=log)
        fz_ref = a.freeze_ref or a.start
        if fz_ref:
            u_fz = np.load(fz_ref)["u"]
            assert u_fz.size == A.n_u, "--freeze-ref must be in the run basis"
            tgt_fz = Ct_fz.T @ (u_fz - np.asarray(u0))
        else:
            tgt_fz = np.zeros(Ct_fz.shape[1])
        fz_info["reference"] = fz_ref or "J_fit"
        log(f"[freeze n0] reference {fz_info['reference']}: |target| max {np.max(np.abs(tgt_fz)):.3e}")
        blocks_.append(("freeze_n0", Ct_fz, tgt_fz))
        run["freeze n0"] = fz_info
        phase("freeze-n0 rows", t1)
    if a.pin_current or a.pin_flux or a.pin_b21 is not None or blocks_:
        t1 = time.time()
        aff = []
        if a.pin_b21 is not None:
            ph_src = a.b21_phase_state or a.start
            u_ph = np.load(ph_src)["u"]
            if a.start_LMN and u_ph.size != A.n_u:  # phase state from the --start basis: embed it like the start
                lo_ph = fit_cache.fit(BoundaryPotential, inp0, L=a.start_LMN[0], M=a.start_LMN[1], N=a.start_LMN[2])
                u_ph = mff.embed_truncate(lo_ph, cur0, u_ph)[0]
            bph = bmk(u_ph)
            ph0 = float(np.angle(bph))
            cph, sph = float(np.cos(ph0)), float(np.sin(ph0))
            tgt = a.pin_b21 * a.b21_vac
            log(f"[b21 pin] island phase {np.degrees(ph0):+.1f} deg from {ph_src} (|b| {abs(bph):.3e}); target "
                f"Re(b e^-i phase) = {a.pin_b21:g} x {a.b21_vac:.3e} = {tgt:.4e}")
            run["b21 pin"] = {"beta": a.pin_b21, "phase deg": float(np.degrees(ph0)), "phase from": ph_src, "target": tgt}
            aff.append(("b21_proj", lambda uu: bfn(uu)[0] * cph + bfn(uu)[1] * sph, tgt))
        LC = mff.LinearConstraints(cur0, fields, fr, u0, pin_current=a.pin_current, pin_flux=a.pin_flux, log=log,
                                   affine=aff, blocks=blocks_)
        phase("linear constraints", t1)
    mk_metric = (lambda LL: LC.metric(LL)) if LC is not None else pw.Metric
    met = mk_metric(L)
    if lean and not (a.subproblem_test or a.anatomy or a.stiff_modes):
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
                f"{float(jnp.linalg.norm(sz['gJ'])):.5f}")
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
            + f"; M/M_desc {float(sx['M']) / M_desc:.5f}, |J - J_fit|/rms {float(jnp.linalg.norm(sx['gJ'])):.5f}")
        cons_log("start before restore", s_b, v0)
        cons_log("start after restore", s_a, v1)
        if "dPsi/Psi" in v1 and LC.psi is not None:
            log(f"[constraints] Psi_bar check at the restored start: direct {float(LC.psi(z)):.9e}, "
                f"affine {LC.psi_affine(z):.9e}, target {LC.psi_ref:.9e}")
        run["constraints"] = {"pin_current": bool(a.pin_current), "pin_flux": bool(a.pin_flux),
                              "n_pinned": int(len(LC.idx)), "before": v0, "after": v1,
                              "Psi_ref": getattr(LC, "psi_ref", None), "Psi_desc": float(inp0.eq.Psi)}
        del s_b, s_a
    # energy: W_Omega relative to J_fit's is always reported; lambda > 0 adds it to the objective
    A.enable_energy(a.energy_weight, W, vol, log=log)
    W.f_floor = 0.0
    run["energy"] = {"lambda": a.energy_weight, "W_ref [J]": A.W_ref_J, "volume [m^3]": vol, "e_c": A.ec,
                     "DESC volume [m^3]": float(np.asarray(eq0.compute("V")["V"]))}
    if a.energy_weight > 0:
        if not (bnf is not None and a.pin_flux):
            log("[energy] WARNING: without --bn-modes and --pin-flux the plasma-volume energy is not the free energy")
        if not a.no_share_stop:
            log("[energy] WARNING: the energy rows carry a constant lambda in |r|^2; use --no-share-stop")
    e0_ = energy_report(z, A.state(z))
    run["energy"]["start"] = e0_
    log("[energy] start: " + ", ".join(f"{k} {v:.4e}" for k, v in e0_.items()))
    s0 = A.state(z)
    if a.ref_start and a.force_norm == "p":
        # the bounds are fractions of the START state's own band p-norm and outside mean (a new constraint:
        # the start's multiplier and penalty, from a different bound, are reset)
        z_ref = jnp.asarray(np.load(a.ref_state)["u"]) if a.ref_state else z
        if a.ref_state:
            log(f"[p-norm] reference state: {a.ref_state}")
        for p_ in p_list:
            A.set_pnorm(p_, pmask, exact_beta=not a.p_model_beta)
            sr = A.state(z_ref)
            aF_ = np.asarray(sr["absF"])
            desc_p[p_] = {"M_p": float(sr["M_p"]), "max_int": float(sr["M_max_int"]), "edge": float(sr["M_edge"]),
                          "mean_int": float(np.sum(w_pts[pmask] * aF_[pmask]) / w_pts[pmask].sum() / norm)}
            im = int(np.argmax(np.where(pmask, aF_, -1.0)))
            log(f"[p-norm] reference = start state, p {p_:g}: band p-norm {desc_p[p_]['M_p']:.4e}, band max "
                f"{desc_p[p_]['max_int']:.4e} at rho {rho_pts[im]:.4f}, band mean {desc_p[p_]['mean_int']:.4e}, "
                f"outside mean {desc_p[p_]['edge']:.4e}, volume mean {float(sr['M_mean']):.4e} (/ norm)")
        run["pnorm"]["reference"] = {"kind": "start", **{str(k): v for k, v in desc_p.items()}}
        A.set_pnorm(p_list[0], pmask, exact_beta=not a.p_model_beta)
        if a.edge_alpha is not None:
            set_pn(p_list[0], a.alphas[0])
        M_desc = desc_p[p_list[0]]["M_p"]
        W.eps = a.alphas[0] * M_desc
        if a.keep_start_mult and a.start and "nu" in np.load(a.start).files:
            st_ = np.load(a.start)
            W.nu, W.rho = np.asarray(st_["nu"], float), float(st_["rho"])
        else:
            W.nu, W.rho = np.zeros_like(np.asarray(W.nu, float)), a.rho0
        log(f"[p-norm] eps = {a.alphas[0]:g} x start band p-norm = {W.eps:.4e}"
            + (f"; outside mean <= {a.edge_alpha:g} x start outside mean" if a.edge_alpha is not None else "")
            + f"; multiplier nu {np.asarray(W.nu).tolist()}, rho {W.rho}")
        z = jnp.asarray(z)
        s0 = A.state(z)
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
    solver.radius_by_delta = a.radius_by_delta
    solver.no_share_stop = a.no_share_stop
    solver.f_sigma = None if a.f_noise is None else ("auto" if a.f_noise == "auto" else float(a.f_noise))
    solver.geo_accel = ({"amax": a.accel_amax, "tol": a.accel_tol, "maxiter": a.accel_maxiter}
                        if a.geo_accel else None)
    solver.subspace = ({"dirs": [x.strip() for x in a.subspace.split(",") if x.strip()], "smax": a.subspace_smax,
                        "tmax": a.subspace_tmax, "all": a.subspace_all,
                        "accel": {"amax": a.accel_amax, "tol": a.accel_tol, "maxiter": a.accel_maxiter}}
                       if a.subspace else None)
    if solver.subspace is not None:
        log(f"[solver] subspace search {solver.subspace}")
    if a.geo_accel or a.radius_by_delta or a.no_share_stop or a.f_noise:
        log(f"[solver] geodesic acceleration {solver.geo_accel}; radius-by-delta {a.radius_by_delta}; f-noise {a.f_noise}; "
            f"share stop {'off' if a.no_share_stop else 'on'}")
    best = {"z": z, "s": s0, "n": 0}

    diag = None
    if a.diag_every:
        # cheap periodic diagnostics: one extra B_ann solve (b_mk) and host reductions over the points
        t1 = time.time()
        rho_d = np.asarray(fr.rtz[:, 0])
        w_d = np.asarray(sw) ** 2
        band_d = (rho_d >= a.diag_band[0]) & (rho_d <= a.diag_band[1])
        tg_ = fr.tgrid
        absF_D = np.asarray(inp0.compute_tensor(["|F|"], tg_.rho, tg_.theta, tg_.zeta)["|F|"]).ravel()
        assert absF_D.size == len(rho_d), (absF_D.size, len(rho_d))
        ref_d = mff.desc_pnorm(absF_D, w_d, band_d, 16.0, norm)
        run["diag"] = {"band": list(a.diag_band), "band points": int(band_d.sum()), "DESC": ref_d,
                       "b_vac": a.b21_vac, "M_desc": M_desc_mean}
        log(f"[diag] band {a.diag_band[0]:.4f} <= rho <= {a.diag_band[1]:.4f}: {int(band_d.sum())} points; DESC there: "
            f"p16 {ref_d['M_p']:.4e}, max {ref_d['max_int']:.4e}, outside mean {ref_d['edge']:.4e} (/ norm); "
            f"set up in {time.time() - t1:.1f} s")

        def diag(zz, ss, where):
            nd = mff.desc_pnorm(np.asarray(ss["absF"]), w_d, band_d, 16.0, norm)
            bv = bmk(zz)
            out_ = {"where": where, "t [s]": time.time() - t0,
                    "M/M_desc": float(ss.get("M_mean", ss["M"])) / M_desc_mean,
                    "|J-J_fit|/rms": float(jnp.sqrt(jnp.sum(ss["gJ"] ** 2))), **energy_report(zz, ss),
                    "b/vac": abs(bv) / a.b21_vac, "b phase deg": float(np.degrees(np.angle(bv))),
                    "band p16/DESC": nd["M_p"] / ref_d["M_p"], "band max/DESC": nd["max_int"] / ref_d["max_int"],
                    "outside mean/DESC": nd["edge"] / ref_d["edge"]}
            run.setdefault("diag log", []).append(out_)
            log(f"[diag] {where}: M/M_desc {out_['M/M_desc']:.5f} |J-J_fit|/rms {out_['|J-J_fit|/rms']:.4e} "
                f"b{a.b21_mk[0]}{a.b21_mk[1]}/vac {out_['b/vac']:.4f} phase {out_['b phase deg']:+.1f} deg | band p16/DESC "
                f"{out_['band p16/DESC']:.4f} max/DESC {out_['band max/DESC']:.4f} | outside mean/DESC "
                f"{out_['outside mean/DESC']:.4f} | dW/W {out_['dW/W']:+.4e} (n=0 {out_['dW/W n=0']:+.3e}, n!=0 "
                f"{out_['dW/W n!=0']:+.3e}, cross {out_['dW/W cross']:+.2e}) E {out_['E']:+.4e}"
                + (f" | B.n max|b| {out_['Bn max|b| (field)']:.2e} rms {out_['Bn rms (field)']:.2e}"
                   if "Bn rms (field)" in out_ else ""))
            return out_

        diag(z, A.state(z), "start")
        dump()

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
                   beta_min=a.beta_min, log=log, lz_stop=a.lanczos_stop,
                   skip_hit=a.no_rebuild_on_hit)
    log(f"[model] {a.model}" + (f" (beta {a.beta})" if a.model == "beta" else "")
        + (f" (GN while |c_F| > {a.hvp_far}, GN x{a.hvp_fallback} after a radius cut)" if a.model == "hybrid" else "")
        + (" adaptive" if a.beta_adapt else "")
        + f"; metric {a.precond}" + (f" (rebuild: Lanczos cap or rent-or-buy; outer if psi' changed > x{a.outer_rebuild_factor:g})"
                                     if a.precond == "force" else "")
        + f"; LS multiplier {'on' if solver.mult_ls else 'off'}; Lanczos stop {a.lanczos_stop}")
    if a.chord_study:
        import json as _json
        sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
        import line_anatomy as LA
        W.eps = a.alphas[0] * M_desc
        if a.precond == "force":
            met_c, D_c = refresh(z)
            solver.met, solver.Delta = met_c, D_c
        log(f"[chord] start: M/eps {float(A.state(z)['M']) / W.eps:.4f}, nu {W.nu.tolist()}, rho {W.rho}, Delta {solver.Delta}")
        cres = LA.chord_study(A, solver, W, norm, solver.met, z, a.chord_study, log)
        with open(a.out + "_chord.json", "w") as fh:
            _json.dump(cres, fh, indent=1, default=float)
        if a.mirror:
            import shutil
            shutil.copy(a.out + "_chord.json", a.mirror)
        log("chord study done")
        return
    if a.subproblem_test or a.anatomy or a.stiff_modes:
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
        if a.stiff_modes:
            sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
            import stiff_modes as SM

            SM.run(rows, A, met, s, *a.stiff_modes, log, out_path=a.out + "_stiff.npz")
            if a.mirror:
                import shutil
                shutil.copy(a.out + "_stiff.npz", a.mirror)
            log("stiff modes done")
            return
        if a.anatomy:
            sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
            import subproblem_anatomy as SA

            def ev(d):
                s_n, f_n, phi_n, c_n = solver.evaluate(z + d)
                return float(f_n), float(s_n["M"]) / W.eps, float(phi_n)
            rec = SA.run(Hv, grad, f0, met, a.anatomy, a.anatomy_deltas, ev, (float(f), float(s["M"]) / W.eps, float(phi)),
                         log, out_path=a.out + "_anatomy.npz", keep=tuple(a.line_scan or ()))
            if a.line_scan:
                import json as _json
                import line_anatomy as LA
                ts = np.linspace(0.0, a.line_tmax, 41)
                lres = {}
                for m in a.line_scan:
                    if a.accel:
                        lres[str(m)] = LA.accel_scan(A, z, rec["steps"][m], met, rows, norm, W.eps, log, ts,
                                                     tag=f"Delta {m:g}|g|")
                    else:
                        lres[str(m)] = LA.run(A, z, met.back(rec["steps"][m]), norm, W.eps, log, ts, tag=f"Delta {m:g}|g|")
                with open(a.out + "_line.json", "w") as fh:
                    _json.dump(lres, fh, indent=1)
                if a.mirror:
                    import shutil
                    shutil.copy(a.out + "_line.json", a.mirror)
            if a.mirror:
                import shutil
                shutil.copy(a.out + "_anatomy.npz", a.mirror)
            log("anatomy done")
            return
        out = []
        for cap in a.subproblem_test or []:
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
                log(f"[p-norm] p {p_cur:g}: eps = alpha x " + ("start" if a.ref_start else "DESC")
                    + f" {'band' if a.p_band else 'interior'} p-norm {M_desc:.4e}")
            for alpha in a.alphas:
                t1 = time.time()
                W.eps = alpha * M_desc
                if p_cur is not None and a.edge_alpha is not None:
                    set_pn(p_cur, alpha)
                log(f"alpha {alpha}: eps {W.eps:.4e}")
                entry = {"alpha": alpha, "eps": W.eps, "p": p_cur}
                run["alphas"].append(entry)

                is_last = alpha == a.alphas[-1]

                def cb(zz, s, outer, entry=entry, is_last=is_last):
                    entry["outer"] = outer
                    entry["time"] = time.time() - t1
                    entry.pop("inner (current outer)", None)
                    run["memory"].append({"phase": f"alpha {alpha} outer {len(outer)}",
                                          "t": time.time() - t0, **mem()})
                    save_state(f"{ptag}a{alpha}_latest", zz, s, W)
                    if diag is not None:
                        diag(zz, s, f"alpha {alpha} outer {len(outer) - 1} end")
                    if a.snap_outer:
                        save_state(f"{ptag}a{alpha}_o{len(outer)}", zz, s, W)
                    dump()
                    best["z"], best["s"] = zz, s
                    if a.stall_stop and len(outer) >= 2:
                        cA, cB = outer[-2]["c"][0], outer[-1]["c"][0]
                        if cB > 1e-2 and abs(cA - cB) <= a.stall_stop * (1 + cB):
                            log(f"[stage] alpha {alpha}: infeasible stall (M/eps {1 + cA:.4f} -> {1 + cB:.4f}): ending the sweep")
                            raise NextStage("stall")
                    if a.stage_flat and not is_last and len(outer) >= 2:
                        c0, ph0, ph1 = outer[-1]["c"][0], outer[-2]["phi"], outer[-1]["phi"]
                        if c0 <= 1e-2 and (ph0 - ph1) <= a.stage_flat * abs(ph0):
                            log(f"[stage] alpha {alpha}: bound met (M/eps {1 + c0:.4f}) and phi fell "
                                f"{(ph0 - ph1) / abs(ph0):.2%} over the last outer (< {a.stage_flat:.0%}): next stage")
                            raise NextStage("flat")

                def on_step(hist, entry=entry):
                    entry["inner (current outer)"] = hist
                    if hist and hist[-1]["accepted"] and A.pnorm is not None:
                        rep = pn_report(A.last_s, A.pnorm["p"])
                        im_ = int(jnp.argmax(jnp.where(A.pnorm["mask"] > 0, A.last_s["absF"], -1.0)))
                        rep["rho of max"] = float(rho_pts[im_])
                        hist[-1]["pnorm"] = rep
                        log(f"      [p-norm] p {rep['p']:g}: M_p/ref {rep['M_p/DESC']:.4f}, max/ref "
                            f"{rep['max_int/DESC']:.4f} at rho {rep['rho of max']:.4f}, outside mean/ref {rep['edge/DESC']:.4f}, "
                            f"mean/M_desc {rep['mean/M_desc']:.4f}")
                    if hist and hist[-1]["accepted"] and LC is not None:
                        vx = LC.violation(A.last_z)
                        hist[-1]["constraints"] = vx
                        log("      [constraints] " + ", ".join(f"{k} {v:.2e}" for k, v in vx.items()))
                    if hist and hist[-1]["accepted"]:
                        best["z"], best["s"] = A.last_z, A.last_s
                        best["n"] += 1
                        if best["n"] % max(a.save_every, 1) == 0:
                            save_state(f"{ptag}a{alpha}_latest", best["z"], best["s"], W)
                        if diag is not None and best["n"] % a.diag_every == 0:
                            hist[-1]["diag"] = diag(best["z"], best["s"], f"alpha {alpha} accepted step {best['n']}")
                    dump()
                    if a.time_budget and time.time() - t0 > a.time_budget:
                        raise Budget()
                    ctl = os.path.join(a.mirror or os.path.dirname(os.path.abspath(a.out)), "CONTROL")
                    if os.path.exists(ctl):
                        cmd_ = open(ctl).read().strip().upper()
                        os.remove(ctl)
                        log(f"[control] {cmd_} at alpha {alpha}, inner step {len(hist)}")
                        if cmd_ == "STOP":
                            raise Budget()
                        if cmd_ == "NEXT":
                            raise NextStage("manual")

                solver.on_step = on_step
                forced = False
                try:
                    z, outer = solver.solve_alpha(z, max_outer=a.max_outer, maxiter=a.maxiter,
                                                  gtol=a.gtol, callback=cb, stat_tol=a.stat_tol,
                                                  stat_window=a.stat_window)
                except NextStage as e:
                    z = best["z"] if best.get("z") is not None else z
                    forced = str(e) == "manual"
                    entry["stage end"] = str(e)
                s = A.state(z)
                c = W.constraints(s)
                entry.update({"M/M_desc": float(s["M"]) / M_desc, "rms|F|": float(s["rmsF"]),
                              "|J-J_fit|/rms": float(jnp.sqrt(jnp.sum(s["gJ"] ** 2))), **energy_report(z, s),
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
                if diag is not None:
                    entry["diag"] = diag(z, s, f"alpha {alpha} end")
                dump()
                if not entry["feasible"] and not forced:
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
        if diag is not None:
            diag(best["z"], best["s"], "end (budget)")
    run["total [s]"] = time.time() - t0
    run["memory"].append({"phase": "end", "t": time.time() - t0, **mem()})
    dump()
    log(f"saved {a.out}.json; total {time.time() - t0:.0f} s; mem {mem()}")


if __name__ == "__main__":
    main()
