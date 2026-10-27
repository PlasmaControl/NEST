# Handoff: the mean-force / Poisson-pressure free-boundary study (NCSX_fb_L16M16N12)

2026-10-27. Written so another person or model can continue the work from this branch
(`closest-poisson`) without the earlier conversation. The results are in `docs/meanforce_results.md`;
the plan the work follows is `docs/mean_force_poisson_plan.md`; the health review of the first runs is
`docs/meanforce_gltr_a40.md`.

## 1. What the study does

Closest state to DESC's current under a bound on DESC's mean force, on the **NEST free boundary**
(annulus), with the **pressure eliminated by the weak Poisson equation**:

    min over (u, xi)   |J - J_fit|^2 / 2          (volume rms, relative to rms J_fit)
    s.t.   M = <|F|>_vol / <|grad|B|^2|/2mu0>_vol  <=  alpha * M_DESC      (DESC's own definition)
           B.n/|B| rms on the boundary  <=  beta = 4.53e-3                 (DESC's free-boundary value)
           Psi = Psi_start                                                  (toroidal flux held)

- u: current-potential coefficients (J = curl T), xi: boundary displacement (413 modes, M = 16, N = 12).
- p = p*(u, xi): weighted least-squares (weak Poisson) projection of J x B onto the gradients of the
  pressure space. Its xi-derivative (variable projection) is included exactly.
- Sweep alpha = 0.7, 0.5, 0.3, 0.2, 0.1, 0.05 (M_DESC = 8.7788e-4 normalized, DESC quadrature).
- Two solvers: matrix-free GLTR (`--method gltr`) and dense QR trust region (`--method qr`, u columns
  from per-anchor tables, xi columns by JVPs). Augmented Lagrangian over the three constraints.
- Study 1: annulus spacing h = a/20 ... a/120. Study 2: NEST basis L = M = 16, 20, 24, 28 at N = 12, and 16/16/16.

## 2. Code map (branch `closest-poisson`)

| file | what |
|---|---|
| `nest/meanforce.py` | `Anchor` (pressure projection, state, rows, linearization), `Weights`, `Rows` (sum-of-squares model of the AL; dense rows), `Solver` (inner trust region, AL outer loop, stall detector and diagnostic, forced re-anchor for QR), `metric`, `hessian_lanczos`, `iota_profile` |
| `scripts/meanforce/mf_run.py` | the driver: gates, start, sweep over alphas with retries, re-anchoring, saves `<out>.json`, `<out>_a<alpha>.npz` (u, xi, c_p, nu, rho, eps) and `<out>_a<alpha>_eq.h5` (geometry); `--start <out>_a<alpha>` continues from a saved state; `--hessian-only` |
| `scripts/meanforce/eval_common.py` | saved states evaluated on one grid / annulus / pressure space against J_DESC |
| `scripts/meanforce/collect.py` | tables and plots (`figures/meanforce/study_*`) |
| `scripts/meanforce/health.py`, `radial_force.py`, `plot_radial.py`, `bn_check.py`, `submit_studies.sh` | health plots of a run; |F| per coordinate surface; annulus vs direct Biot-Savart B.n; the submission script of the studies |
| `tests/test_meanforce.py` | gates of the formulation (Poisson, Galerkin, linearization incl. d(projector)/d(xi), adjoint, dense vs matrix-free rows, GLTR vs QR); runs on a GPU in ~40 s |
| `nest/free_boundary.py`, `scripts/optimize_fb.py`, `tests/test_free_boundary_cyl.py` | the annulus free boundary ported from the (uncommitted) `NEST_cylfb` worktree |
| `run_sai_cyl.sbatch` | the Stellar-AI job script used for everything (1 B200, 200 GB host, excludes node sai-n01g4) |
| `UPDATES.md` | a dated entry per commit (the repository convention; add one with every commit) |
| `results/meanforce/` | every run's JSON, saved states (npz, _eq.h5), stall diagnostics, Hessian checks, condensed logs in `log_summaries/` |

## 3. How to reproduce / continue (Stellar-AI)

- Login: `ssh sai` (alias for stellarai-vis1; needs the user's key). Work copy: `/scratch/gpfs/EKOLEMEN/lk1158/NEST_mf`
  (this branch rsynced, minus results/figures/logs). Env: `/scratch/gpfs/EKOLEMEN/lk1158/envs/nest13`
  (JAX with CUDA 13). `.pylib` there holds threadpoolctl (needed by the annulus setup).
  Do NOT use Della; Stellar-AI only. Do not run compute on the user's laptop.
- Account kolemen, partition gpu, QOS gpu-short (4 h), gpu-medium (24 h), gpu-test (30 min). Per-user limit
  of 10 submitted jobs per QOS (QOSMaxSubmitJobPerUserLimit when exceeded).
- One run: `sbatch --qos=gpu-medium --time=24:00:00 run_sai_cyl.sbatch scripts/meanforce/mf_run.py --method gltr
  --alphas 0.7 0.5 0.3 0.2 0.1 0.05 --frac 40 --chunk 32 --out results/meanforce/sweep_gltr_a40`
  (QR: `--method qr`; spacing `--frac 20|60|80`; basis `--factors 1.25 1.25 1` (L = M = 20), `1.5 1.5 1`, `1.75 1.75 1`,
  `1 1 1.3333333333333333` (16/16/16); at a/60 and finer pass `--tan-chunk` (default scales as 32 (40/frac)^3, at most 32)).
- Continue from a state: `--start results/meanforce/<out>_a<alpha> --alphas <new alpha>`; for a converged start add
  `--delta0 1e-4 --stat-window 5`.
- Hessian at a saved state: `--hessian-only --start ... --alphas <alpha> --hessian-h 1e-9 --hessian-k 40`.
- Gates: `sbatch --qos=gpu-test --time=00:30:00 run_sai_cyl.sbatch tests/test_meanforce.py` (expect `ALL PASSED`).
- Evaluate states on one apparatus: `scripts/meanforce/eval_common.py` (see its docstring and the `common_L28_a*.json` runs).
- Tables and plots: rsync `results/meanforce/*.json` locally, then
  `python scripts/meanforce/collect.py --dir results/meanforce --out figures/meanforce/study`.

## 4. Jobs that were still running when this was written (2026-10-27 ~15:30)

They write into `/scratch/gpfs/EKOLEMEN/lk1158/NEST_mf/results/meanforce/` (the JSON is updated at every step).

| job | run | state |
|---|---|---|
| 6279 | `s1_gltr_a60`, a/60 GLTR sweep | last bound alpha 0.05 |
| 6281 | `s1_gltr_a80`, a/80 GLTR sweep | on alpha 0.2 (will hit the 24 h limit about alpha 0.1: continue the rest with `--start`) |
| 6298 | `s1_qr_a80`, a/80 QR sweep | on alpha 0.2 (same: table rebuilds take ~20 min per re-anchor) |
| 6540 | `s2_gltr_L16N16_r05` | continuation of the unconverged 16/16/16 alpha 0.05 point (120 steps, tighter stop) |
| 6543, 6544 | `s1c_gltr_a100_a0.05`, `s1c_gltr_a120_a0.05` | a/100 and a/120 at alpha 0.05 from the a/40 GLTR state |

When they end: rsync the JSONs, rerun `collect.py`, and read the (!) marks in the table (see 5).

## 5. Things to know before trusting a number

1. **A bound is only converged if the mean-force bound is active.** The `(!)` mark in `figures/meanforce/study_table.md`
   flags M < 0.99 alpha (inactive bound: the objective could still fall). The 16/16/16 alpha 0.05 point of the first run
   ended this way (M = 0.0464); it is being redone (6540). The "flat" stop now requires an active bound.
2. **a/20 is not trustworthy below alpha ~0.2**: its mean force is inflated by the coarse annulus (the a/20 alpha 0.05 states
   have M = 0.0447 on a/40 and a/60 when solved to 0.0500 on a/20). a/40 re-evaluated on a/60 agrees to 0.8%.
3. **Per-basis J_fit**: |J - J_fit| of different bases is measured against different fits and is NOT monotone in L = M at
   the loose bounds. Use the common-apparatus numbers (`study_common_table.md`, `common_L28_a*.json`: fixed target J_DESC).
4. **Hessian step**: Hessian-vector products are central differences of the exact gradient. h = 1e-6 gave spurious negative
   Ritz values; use h = 1e-9 (default now). The Hessians stored in the sweep JSONs from before 2026-10-27 used 1e-6 and are
   unreliable below alpha 0.7; recompute with `--hessian-only` (done for a/40 GLTR, L = M 20/24/28 at alpha 0.1 and 0.05,
   a/40 QR alpha 0.1 and 0.05: all positive).
5. **QR vs GLTR** agree to 0.01-0.3% down to alpha 0.1. At alpha 0.05 they reach different local minima (a/40: GLTR 0.1128 with B.n
   at 0.52 beta; QR 0.1141/0.1161 with B.n at its bound).
6. QR's u columns come from the anchor's tables (exact at xi = 0): near a hard optimum with the boundary moved ~2 mm they
   mislead the model; the solver now re-anchors when the radius collapses (forced re-anchor).
7. The penalty rho now rises only while a constraint is outside its tolerance (jobs started before 2026-10-27 15:00 ran with
   the old rule and show rho up to 1e6 at alpha 0.05).

## 6. Open work (from the plan, section 12)

- Finish the runs of section 4; continue a/80 for the remaining alphas.
- Study 1 plot with each state's force re-evaluated on the finest annulus (a/120): `eval_common.py --frac 120`
  needs `--factors 1 1 1` and states from the 16/16/12 runs.
- QR at L = M >= 24 is not run (the dense table is 62-96 GB per copy: needs the multi-GPU jaxmg split of
  `scripts/pointwise/al_run.py`; GLTR covers study 2).
- Poincare sections of the free-boundary end states (`<out>_a<alpha>_eq.h5` is the moved geometry; the tracer
  `scripts/pointwise/poincare_state.py` shows how the annulus field is traced; it needs an update for moved geometries).
- The written report in the style of `docs/pointwise_al_ncsx.md` (and a shareable page if wanted).
- Optional solver improvements found in the health review: exact Hessian-vector products of M inside GLTR (the
  majorizer's conservative curvature makes late steps tiny), Lanczos preconditioning.

## 7. Working conventions of the user (carried over)

- Edit locally, run everything on Stellar-AI (gates and smoke runs too); not Della, not the laptop.
- A dated entry in `UPDATES.md` with every commit.
- Check long runs while they run (a watcher on the logs for errors, 8 rejected steps in a row, radius collapse, job end,
  hourly heartbeat), diagnose and fix bugs early instead of finishing a bad run.
- Report numbers faithfully, including failures and corrections (several earlier statements were corrected: a/20 at alpha 0.05,
  the negative Hessian values, the 16/16/16 point).
