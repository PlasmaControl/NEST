# Updates

One entry per commit, newest first.

## 2026-10-26: Hessian step, penalty rule, host-built metric (branch closest-poisson)
- Hessian-vector products: the old step 1e-6 gave spurious negative Ritz values (a/40 QR alpha 0.1 state: -1e4, -175, +169 at h = 1e-4, 1e-6, 1e-8). Default now 1e-8; `mf_run.py --hessian-only --start <state> --hessian-h ...` recomputes at saved states. a/40 QR alpha 0.05: +9.29e3 (1e-8), +9.33e3 (1e-9): a local minimum; a/20 QR alpha 0.05: -6e5 .. +8e4 depending on h (not stationary; not used).
- Penalty rho rises only while a constraint is outside its tolerance (the alpha 0.05 redo went 1 -> 1e3 on a Psi drift of 6e-4).
- `metric()` builds H_J from a host copy of the current-column table in blocks of points (the L = M = 28 table is 46 GB; the device Gram matrix needed >180 GB); `Anchor.A_table(host=True)`.
- a/40 QR alpha 0.05 redone with the fixed solver (6374): |J - J_fit|/rms 0.1161 (first run 0.1141), M = 0.0500 M_DESC (first 0.0504).

## 2026-10-26: QR forced re-anchor on a collapsed radius; flat test ignores vanishing steps (branch closest-poisson)
- a/20 QR (6276) at alpha 0.05 stalled: |g| 0.19 with the radius at 2e-13 and ratios 0.05-0.18 at tiny steps, i.e. a wrong model gradient: QR's u columns are the anchor's tables (exact at xi = 0), off by O(xi) once the boundary has moved (1.85 mm), and near a hard optimum that error dominates. Its "flat" test took the stall for convergence (|J - J_fit|/rms 0.205 against 6.5e-2 at alpha 0.1; not a valid point of the front).
- Now: QR re-anchors (exact tables) when the radius is below 1e-8 with accepted ratios < 0.25 and the boundary moved; "flat" (outer and inner) does not count when the radius has collapsed. a/20 QR alpha 0.05 redone from its 0.1 state (6370); gates 6369. L = M = 28 GLTR submitted.

## 2026-10-26: inner flat stop, --start, field-solve batch scaled with the annulus (branch closest-poisson)
- `Solver.inner(flat_tol)`: stops once L_A is flat to 0.1 stat_tol over 10 accepted steps (the QR sweep spent ~25 steps at a converged alpha 0.3 shrinking its radius to 4e-9).
- `mf_run.py --start <out>_a<alpha>`: continue from a saved state and its geometry (targets, rms J_fit and metric from DESC's start), for a/100, a/120 from the a/40 states.
- `Anchor(tan_chunk)`, `--tan-chunk` (default 32 (40/frac)^3, at most 32): the a/80 QR job (6280) needed 186 GB for 32 xi columns at once; resubmitted as 6298. a/60 QR peaks at 137 GB.
- Probes: a/100 301 M nodes, 53 GB GPU, 74 GB host, 3.5-5.3 min per GLTR step; a/120 518 M nodes, 76 GB, 124 GB, 5.8-8.8 min (vs ~40 s at a/40).
- Gates pass on Stellar-AI (6295); start-from-state smoke (6296) reproduces the saved state's M, B.n, Psi.

## 2026-10-26: alpha 0.7 re-run after the fixes; the convergence studies submitted (branch closest-poisson)
- Sweep jobs 6232 (QR), 6233 (GLTR) at a/40: both stop alpha 0.7 on the stationarity test; |J - J_fit|/rms 1.4882e-2 (QR), 1.4868e-2 (GLTR), within 0.1% (0.35% before); M = 0.700 M_DESC, B.n = beta, Psi within 3e-5. GLTR still loses ~3 steps at the start of an outer iteration (B.n at its bound with nu_B = 0: the penalty grows like |step|^4 there, which no quadratic model sees); the trust region recovers.
- `scripts/meanforce/submit_studies.sh`: study 1 (a/20, a/60, a/80, QR and GLTR) and study 2 (GLTR, L = M 20, 24) submitted as jobs 6276-6283; L = M 28 and 16/16/16 wait for gpu-medium slots (10 submitted jobs per user); probes a/100, a/120 (6285, 6286, gpu-short).

## 2026-10-26: the step model keeps a bound's penalty curvature near the bound (branch closest-poisson)
- First sweep (jobs 6203, 6204, cancelled): GLTR's radius collapsed at the start of outer 2, ratios down to -3e5. Stall diagnostic: Jacobian vs differences 4e-8, adjoint 5e-15; d'Hd = 4270 against d'(J^T J)d = 0.38 along the step. The B.n bound sat at c_B = 0 with nu_B = 0, so the model had neither B.n rows nor the penalty's curvature, and the steps pushed B.n above beta. Now the model's penalty-curvature rows use rho within `--band` 0.05 of a bound (`Weights.d2_model`); gradient unchanged. Gate added. Also: zero residual (at J_fit with inactive bounds) ends the inner solve instead of dividing by zero.
- Resubmitted: gates 6230, retry smoke 6231, sweeps 6232 (QR), 6233 (GLTR).

## 2026-10-26: the three fixes of the health review, alpha sweep (branch closest-poisson)
- QR trust region: an undamped Gauss-Newton step (mu = 0) whose model does not decrease is replaced by a damped one (mu = 1e-8 |R|_F^2, cut to the radius); these were all 20 rejected QR steps at alpha 0.7. Logged as "guard".
- Outer loop: converged only when the constraints are met and the inner solve was stationary (its own stop, or |J - J_fit|^2 flat to `--stat-tol` 1e-3 over its last 10 accepted steps). The inner solve reports why it stopped.
- GLTR: Lanczos cap 200 (`--cg`). Sweep: a bound not met goes to the geometric mean of it and the last one met, then retries it (`--retries` 3 in all), then stops; per-alpha `feasible`, `stationary`; Hessian only for met bounds.
- Sweep 0.7, 0.5, 0.3, 0.2, 0.1, 0.05 at a/40, both methods (Stellar-AI jobs 6203 QR, 6204 GLTR), after the gates (6201) and a retry smoke run (6202).

## 2026-10-26: radial distribution of the force error (branch closest-poisson)
- `scripts/meanforce/radial_force.py`, `plot_radial.py`: |F|/norm per coordinate surface and per rho range, and each range's share of M, for DESC, NEST's start and the alpha 0.7 end states (`results/meanforce/radial_a40.json`, `figures/meanforce/radial_a40.png`). End state vs DESC: 2.0-2.4x lower for rho < 0.6, equal at 0.6-0.95, 2-5x lower for rho > 0.95; 86% of the gain over DESC's M comes from rho > 0.95. Section added to `docs/meanforce_gltr_a40.md`.

## 2026-10-26: mean-force runs at alpha 0.7, GLTR and QR (branch closest-poisson)
- NCSX_fb_L16M16N12, NEST 16/16/12, annulus a/40, free boundary (Stellar-AI jobs 6149, 6150): both reach M = 0.70 M_DESC (from 5.74) with |J - J_fit|/rms 1.50e-2 (GLTR) / 1.49e-2 (QR), B.n/|B| 4.53e-3 = beta, Psi within 4e-4; Hessians positive on the tangent space. Annulus B.n vs direct Biot-Savart on the same points: 7e-5 rms (`scripts/meanforce/bn_check.py`). Health review `docs/meanforce_gltr_a40.md`, `scripts/meanforce/health.py`, figures `figures/meanforce/`. Fixed: the inner step budget now holds across re-anchorings; per-step progress in the JSON; `run_sai_cyl.sbatch` excludes sai-n01g4.

## 2026-10-26: mean-force bound with the Poisson pressure on the annulus free boundary (branch closest-poisson)
- Port of the annulus free boundary from the uncommitted `NEST_cylfb` work (that worktree untouched): `nest/free_boundary.py`, `scripts/optimize_fb.py`, `nest/label_pressure.py`, `cylindrical_bs.py` (mesh split + main's mapped-node pairs), `cylindrical_grid.py` (`locate_plasma_planes`), `flux_interp.py` (`StencilInterpolator`), `objective.py` (union of both `Fields` signatures; `factored` passed to `Pressure.tensor`), tests `test_free_boundary_cyl.py`, `cyl_memory.py`, `run_sai_cyl.sbatch`. `test_free_boundary_cyl.py` passes (xi = 0 vs Fields 4e-15, J v vs differences 6e-10, adjoint 5e-14, |xi|^2 scaling). `test_pressure_elim.py` T7 fails (2.7e-10 vs 1e-10) identically on a clean `origin/pressure-poisson`.
- `nest/meanforce.py`: unknowns (u, xi); p = p*(u, xi) from the weighted least-squares (weak Poisson) projection, the QR of sqrt(w) grad phi_j at the moved geometry; DESC's mean force M = sum w|F|/norm; augmented Lagrangian with M <= eps, B.n/|B| rms <= beta, Psi held; sum-of-squares model (majorizer rows w F / sqrt(t) for M, exact gradient), GLTR or QR (u columns from per-anchor tables, xi columns by JVPs) trust regions; the projector's xi derivative (Golub-Pereyra); stall detector and diagnostic; Lanczos Hessian on the tangent space; iota profile.
- `tests/test_meanforce.py` (CPU, all pass): manufactured Poisson 1e-14, trailing block = projection 9e-16, Galerkin 4e-15, J v vs differences <= 8e-10 (u, xi, all), adjoint 6e-14, envelope 9e-8, dense rows vs J v 3e-14, grad L_A vs differences 3e-8, GLTR and QR steps agree.
- `scripts/meanforce/mf_run.py`: the driver (gates: DESC's |F| on NEST's points, B.n start by direct Biot-Savart; per alpha: state, geometry, B.n check, Hessian).

## 2026-10-26: NCSX pointwise force bound: augmented Lagrangian, matrix-free GLTR and multi-GPU QR trust regions
- `nest/pointwise.py`: min |J - J_fit|^2 s.t. |F(x_i)|/norm <= eps at every collocation point, p free; augmented Lagrangian (smoothed plus function), trust region in the physical metric of J and p, solved either matrix-free (Lanczos/GLTR on the annulus Fields) or dense (QR of the stacked Jacobian by jaxmg/cuSOLVERMp, DESC's exact-QR rule); exact Hessians and diagnostics. `nest/pointwise_tables.py`: column tables (J, B_plasma, p per coefficient).
- `scripts/pointwise/`: tables, runs, Hessians, Poincare on the annulus field, cross-resolution check, plots, summary tables, QR benchmark; `run_mg.sbatch` (jaxmg, one process per GPU, env `nest_mg`), `run_sai_bash.sbatch`; `run_sai.sbatch` excludes `sai-n01g4` (GPU 2 fails cuInit).
- `tests/test_pointwise.py`: adjoint, exact Hessian vs differences (1e-10), GLTR and QR reach the same optimum.
- Report `docs/pointwise_al_ncsx.md` (figures in `figures/pointwise/`); plan of the next study `docs/mean_force_poisson_plan.md`.
## 2026-10-27: speed elim on larger laptop models (L-ladder), production hand-off (branch pressure-poisson)
- `lm_ab.py --eq --factors --frac` (default file names unchanged), `pressure_elim_conditioning.py` likewise (its --p-factors now relative to the current factors), `scripts/diagnostics/pressure_elim_conditioning_isl.py` (isl_scan model + a scan state, device SVDs; for production), `docs/pressure_poisson_handoff.md` (sbatch lines). `_elim_cols` is a numpy index (valid with gpus=G).
- Ladder:

| model (current basis, a/h, unknowns) | kappa phys w_J 1e-3: joint / reduced | kappa phys w_J 1e-7: joint / reduced | f/f0 after 30 its, w_J 1e-7: phys,tr / elim,phys,tr / elim + p 1.5x | s/it (same three) | its (time) to converge w_J 1e-3: phys,tr / elim,phys,tr |
|---|---|---|---|---|---|
| L16M16 f1, a/20, 750 | 7.2e3 / 275 | 2.4e5 / 1.4e4 | 0.527 / 0.424 / 0.378 | 3.5 / 3.6 / 5.6 | 21 (65 s) / 11 (48 s) |
| L16M16 f1.5 (L24), a/30, 1,554 | 1.45e4 / 277 | 1.0e6 / 3.0e4 | 0.464 / 0.409 / 0.330 | 18 / 18 / 33 | 24 (390 s) / 9 (186 s) |
| L32M32 f1 (L32, production eq), a/40, 2,646 | 2.3e4 / 283 | 2.5e6 / 4.7e4 | 0.471 / 0.288 / 0.280 | 109 / 108 / 200 | 18 (1926 s) / 16 (1685 s) |
- f0: the joint start (the w_J 1e-3 stage state); both variants reach the same converged f at w_J 1e-3 (2.534e-10 at L32).
The reduced kappa at w_J 1e-3 is flat in L (275, 277, 283) while the joint one grows (~L^1.6). At w_J 1e-7 both grow
(the w_J floor), the reduced one ~50x lower at L32. `results/lm/lad_*`, `lad_cond_*`.

## 2026-10-26: `--p-factors` (pressure space decoupled from the current basis), T8, the pressure ladder E2 (branch pressure-poisson)
- `isl_scan.py --p-factors`, `lm_ab.py --p-factors` (states of the default space embedded), `lm_ab.py` saves each variant's final state; `HarmonicModel.p_range` and `p min/max` in the elim log line.
- T8 (weak Poisson, manufactured q outside every V_h, pole at x = 1.15): error 2.3e-3, 2.4e-4, 2.8e-5, 3.9e-7, 5.6e-9 at L 8, 12, 16, 24, 32 (0.11-0.12 per +4 in L).
- E2 (laptop, current factors 1): f converged at w_J 1e-3: 3.0971e-10, 2.8957e-10, 2.8914e-10, 2.8821e-10 at p factors 1, 1.5, 2, 3 (<|F|>/norm 1.485e-5 -> 1.386e-5): saturates at ~1.5x. Reduced kappa (phys) saturates at 2.8e4; joint phys kappa 2.4e5 -> 5.0e6. w_J 1e-7, 30 its: elim,phys,tr 5.67e-11 vs phys,tr 6.69e-11 (p 1.5), 5.66e-11 vs 7.78e-11 (p 2). At w_J 1e-3, p 1.5 the joint converged in 8 its vs elim 16. Converged p has min/max -1e-4..-3e-4 in both. Table in docs/pressure_poisson_plan.md.

## 2026-10-26: `speed elim`: p eliminated by the weak Poisson (least-squares) solve of force balance (branch pressure-poisson)
- `HarmonicModel(speed={"elim": 1, ...})`, `isl_scan.py --speed elim,phys,tr`: the force rows are affine in p with constant coefficients, so min_p |r(u, p)| is the weak (force-row-weighted) Poisson equation sum w grad(phi_i).grad p = sum w grad(phi_i).(J x B) on the pressure space (p(1) = 0), plus the constant <p> row. The LM / trust-region step is taken on u alone with the exact reduced Jacobian (I - P) A_u = the trailing block of the streamed QR of [A_p | A_u | r] (`_tsqr(cperm=...)`); every trial re-solves p exactly (`_elim_trials`: f* = f(u', p) - |R_pp^-T A_p^T r|^2 / 2 from one pass), so p = p*(u) at every accepted iterate (`|c_p|/|r|` and `f* chk` logged per iteration). Same minimiser as the joint problem; same TSQR cost; the TR subproblem has n_u instead of n_u + n_p unknowns. Requires par = 1, no transport, w_pos = 0 (asserted); default paths unchanged.
- Gates `tests/test_pressure_elim.py` T1-T7 pass (Schur identity 1.3e-14, c_p <= 7e-12, f* vs direct 1.7e-15, reduced GN = u part of joint GN 2.9e-11, FD 3.2e-10, fixed point of a converged joint state, an added grad q removed to 5e-11).
- Conditioning (`scripts/diagnostics/pressure_elim_conditioning.py`, laptop model, eps 1000 stage): kappa Jacobi joint 1.9e5 -> reduced 1.0e4 (w_J 1e-3), 3.5e5 -> 2.7e4 (w_J 1e-7); with the physical metric 7.2e3 -> 275 (w_J 1e-3), 2.4e5 -> 1.4e4 (w_J 1e-7; floor from the u directions whose force change is a pure gradient, held only by w_J).
- Laptop A/B (`lm_ab.py --tag e1`, 30 its; `--start base` new): w_J 1e-3 from the eps 0 base, all variants reach f = 3.0970982e-10: iterations to the stop elim,phys,tr 11, elim,tr 20, phys,tr 21, tr 30 (not converged, GN share 2.6e-9); 48 / 71 / 65 / 200 s. w_J 1e-7 from the stage, f after 30 its (f0 1.500e-10): elim,phys,tr 6.36e-11, elim,tr 7.34e-11, phys,tr 7.91e-11, tr 1.06e-10; GN share 0.011 / 0.056 / 0.069 / 0.24; 3-3.5 s/it for all. `results/lm/e1_*.json`.

## 2026-10-23: `speed tr` without the mu floor: DESC's exact-QR rule (branch lm-trust)
- `_tr_step`: the undamped Gauss-Newton step y_N = -(R D)^-1 c (one triangular solve, as DESC's `solve_triangular_regularized`; R D is upper triangular for both scalings) is taken whenever |y_N| <= Delta (mu = 0, as `desc.optimize.tr_subproblems.trust_region_step_exact_qr`); otherwise mu on the boundary: the Cholesky secular Newton for mu >= 1e-10 lambda_max, and below it a log bisection with the damped QR (Hs = (RD)^T RD squares kappa, so Cholesky is not used there). The radius rule is unchanged.
- Why: the former floor `mu_min = 1e-13 lambda_max` held every step of the production Tier 0 closure runs (tokamak_island_C L32, factors 2 2 1, a/120, N = 1, 9,894 unknowns, w_J 1e-5, 94 A, q = 2 shielded, q = 3, 4 shielded; branch physics-closure): ratio 1.00, steps inside the radius, f -0.1% per iteration, a full GN step could still remove 65% of f after 40 iterations (the closure quantity K lives in modes with s/s_max < 1e-5). With the fix, same start: f 5.75e-13 -> 2.98e-13 in one step (mu/lambda 2e-17, ratio 0.85), vs 5.39e-13 after 40 floored iterations; then accepted steps shorten (2-4 tries), i.e. the remaining 15% is limited by the nonlinearity of the soft directions, not by the floor.
- Laptop A/B (`scripts/lm/lm_ab.py`, as the entry below, 20 its): phys,tr f/f0 0.5914 (unchanged: the radius binds there, mu/lambda ~2e-9), phys 0.7192 (as before); `results/lm/abfix_L16a20_full_eps1000_wJ1e-07.json`. Closure tests (physics-closure `tests/test_closure.py` D8, 6 its at 100 A on the L16 island tokamak): shield <|F|>/norm 1.34e-4 -> 8.3e-5, reconnect 1.43e-4 -> 8.1e-5 (force-only 7.8e-5).

## 2026-10-23: LM step options `phys` (physical-metric damping) and `tr` (trust region) (branch lm-trust)
- `HarmonicModel(speed=...)`, `isl_scan.py --speed ...,phys,tr`. `phys`: the damped step is taken in the variables y with z-step d = L^-T y, L L^T = M = sum_points w (J_i.J_j/rms_J^2 + p_i p_j/rms_p^2) (Gram matrix of the columns' J and p; built once from the tables, no field solves; `_Scale`, `_metric_chunk`, `_phys_L`), instead of the Jacobi column scaling; works with `chol`, `narrow`, `reuse`, `soc`; not with `newton` or `--steps lsqr` (asserted). `tr`: after a first scan iteration, mu from |y(mu)| = Delta (safeguarded Newton on the secular equation, Cholesky per mu; damped QR for the step when mu/lambda_max < 1e-10), exact trial, DESC radius rule, rejected trials shrink Delta and retry (`_tr_step`; log shows ratio/tries).
- Laptop A/B (`scripts/lm/lm_ab.py`; tokamak_L16M16_fb, factors 1, a/20, N = 1, 750 unknowns; full-J objective as `a120Cn2d1000`, w_J 1e-7, 1000 A, from the converged w_J 1e-3 stage; 20 its): f/f0 0.790 (current), 0.743 (`tr`), 0.719 (`phys`), 0.591 (`phys,tr`; identical with `chol,narrow`); removable f (f x GN share) left 0.27 vs 0.09 of f0; same time per iteration. Results `results/lm/ab*_L16a20_full_eps1000_wJ1e-07.json`, figure `figures/lm/lm_ab_production_objective_L16a20.png`. Production (16k unknowns) not yet tested: commands in `docs/lm_solver_handoff.md`.
- Note: `isl_scan.py` without `--base` has no j_target, so `--j-mode full` there silently penalizes only the toroidal average for rho < 0.9; `lm_ab.py` builds j_target as `model_kwargs` does for a base.

## 2026-10-22: DSHAPE perturbed, w_J = 1e-7: results so far (runs in progress)
- N = 1, 3000 A, w_J 1e-7 direct (`a120Cd3000k7`, cancelled at iteration 237): <|F|>/norm 4.01e-5, <|B.grad p|/|B|>/norm 1.80e-5, GN share 0.51. N = 2 from that state (`a120Cd3000w2`): 4.75e-5 -> 1.3e-5 in 30 steps, ~1.0e-5 after 250 (GN share 0.74). N = 3 warm (`a120Cd3000w3`, cancelled at iteration ~12): 1.29e-5 at iteration 10 against N = 2's 1.41e-5, 2x the cost per step.
- N = 2 direct from the unperturbed base at w_J 1e-7 (`a120Cn2d300`, `a120Cn2d1000`, `a120Cd3000w2`), states at iterations 200-250 of 400 when copied: <|F|>/norm 2.4e-7 (300 A), 2.2e-6 (1000 A), 9.9e-6 (3000 A), GN share 0.74-0.92: still creeping, not converged. At N = 2 and 300 A <|B.grad p|/|B|>/norm is 75% of <|F|>/norm.
- w_J continuation at N = 1 (`a120Ce<eps>s<k>`, w_J = 1e-k; stages 3 and 4 only, cancelled): leftover resonant field |b_21(NEST)|/|b_21(vac)| at w_J 1e-3 -> 1e-4: 0.204 -> 0.056 (300 A), 0.212 -> 0.136 (1000 A), 0.251 -> 0.451 (3000 A, not converged).
- Copied: states `results/isl/<name>_eps<eps>.npz/.json`, island summaries `<name>_islands.json`, filtered Slurm logs `results/isl/logs/wJ1e-7/`, figures `figures/dshape/islands/isl_a120Ce*`. Not copied (on Stellar AI `/scratch/gpfs/EKOLEMEN/lk1158/NEST_isl`): Poincare sections and pressure maps (`*_sections.npz`, `*_pmap.npz`), and the final states of the N = 2 runs and the benchmark jobs.

## 2026-10-22: DSHAPE perturbed, w_J = 1e-7: run and benchmark scripts
- `isl_stage.sbatch` (N = 1) and `isl_stage2.sbatch` (N = 2, 4 GPUs, fast solver): one job per w_J stage of the continuation, scan + island and pressure diagnostics, then the next stage submits itself (one queued job per chain: the queue has a per-user submit limit). `isl_variants.sbatch`: solver variants from one snapshot state. `bench_qr.py`, `bench_qr2.py`: QR rate against block size and one thread per device (23-25 TFLOP/s for 32k-256k rows, 4 blocks on 4 devices 0.28 s with 4 threads against 1.09 s from one). `isl_par_summary.py`, `isl_poincare_cmp.py`: the 300 A summary of the force-along-B runs and the side-by-side Poincare sections (figures `isl_par_*.png`).

## 2026-10-22: DSHAPE perturbed, w_J = 1e-7: solver speed options (branch dshape-perturbed-wJ1e-7)
- Objective of these runs: force balance + w_J (J - J_fit) on the full J at every collocation point, nothing else (`--j-mode full --w-W 0 --w-pos 0`, no transport, roughness or force-along-B weight); direct starts at w_J = 1e-7 against a continuation in w_J.
- `HarmonicModel(speed=...)`, `isl_scan.py --speed a,b,k=v`: `resident` (tables stay on the device: step 18 s -> 13 s at N = 1), `chol` + `narrow` (Cholesky damping scan around the last best mu instead of ~10 dense QRs; the final damped QR only for LSQR), `merge=M` (rows per QR block: the QR runs at ~24 TFLOP/s whatever the block), `gpus=G` (one Python thread per device: the cuSOLVER calls block the host thread; N = 1 step 18 s -> 4 s on 4 B200, N = 2 57 s -> 13 s; trajectories identical to the unchanged solver), `fp32`, `profile`. Tested ideas: `reuse=k` (Cholesky factor of a QR step kept for k - 1 steps with the exact gradient: 1 s against 12 s per step at N = 2, ~2.5x the progress per second at 300 A), `soc`, `accel=m`, `pproj`, `newton` (residual-curvature term: no effect at N = 1). All off by default.
- `isl_scan.py`: with `gpus=` in the arguments `desc.set_device` no longer hides all but one GPU (it sets CUDA_VISIBLE_DEVICES to one); prints `<|B.grad p|/|B|>/norm` per amplitude.

## 2026-10-21: DSHAPE islands: the force-only runs' states, sections and logs from Stellar
- `results/isl/<name>_eps<eps>.npz/.json` and `<name>_sections.npz` of `a120n1`, `a120n2` (eps 0-1000 A), `a120w2`, `a120w4`; the scans' logs of Stellar (`results/isl/logs/stellar/`, XLA warnings removed). The base `results/gs/hN_a120D_iter.{json,npz}` is now Stellar's original (the copy of the previous commit was regenerated on Stellar AI).

## 2026-10-21: DSHAPE islands: kappa = 1e8 and the force-along-B weight; converged states for the runs
- `--par w` (`isl_scan.py`, `HarmonicModel(par=...)`): the force component along B (B.grad p) weighted w times against the rest of F, no transport term. At 300 A from the unperturbed base (J on n = 0, roughness 1e-1 / 1e-3 / 1e-5): w = 1e2 gives net b_21 of 0.16-0.3% of vacuum (width 0.003) whatever the roughness; w = 1e4 gives 1e-5 of vacuum, B.grad p/(|B||grad p|) 5e-9 and no island (its force error stalls at 5e-5 in 250 steps). The reversed sign of b_21 of the earlier runs goes away as the enforcement grows. So B.grad p = 0 with a smooth current selects the nested (ideal, shielded) state; an island needs a reconnection branch (finite kappa_perp, a start with an island).
- kappa 1e8 (w_T 1e-4), N = 1, from the kappa 1e6 states: leftover b_21 2.7-3.2% of vacuum (4.5e-6, 1.6e-5, 5.3e-5), widths none / 0.0097 / 0.023, p spread at fixed s 0.21-0.26 of the drop of p0 (1.0 at kappa 1e6), B.grad p median 1e-6 to 6e-6.
- Committed: the base `results/gs/hN_a120D_iter.{json,npz}` (regenerated on Stellar AI, f and <|F|> equal to the earlier ones), the converged states `results/isl/<name>_eps<eps>.npz/.json` and the Poincare sections `<name>_sections.npz` of `a120T6wq`, `a120T8e100/300/1000`, `a120P2r1/P2r3/P2r5/P4r3`, the scans' logs (XLA warnings removed) in `results/isl/logs/`. The states of `a120n1`, `a120w2`, `a120w4` (Stellar) are not here, only their `.json`.

## 2026-10-21: DSHAPE with the 2/1 helical perturbation: amplitude scan, islands and pressure
- `scripts/dshape/isl_scan.py` (exact LM for Phi = eps sin(2 theta + zeta), eps 0-1000 A, N = 1 and 2, from the axisymmetric a/120 state; `nest/harmonic_dense.py`, `nest/perturbation.py`), `isl_islands.py` (resonant b^rho_mn, Poincare sections, island widths, NEST against the vacuum superposition), `isl_pmap.py`, `isl_pzoom.py`, `isl_vacuum.py`, `isl_response.py`, `isl_diag.py`; `run_sai.sbatch` for Stellar AI (B200, CUDA 13 env).
- Location-free formulation (`--j-mode n0 --transport kappa --w-T --rough`): J penalty on n = 0 only, pressure transport along the computed field, smoothness on the n >= 1 current. Widths (a) at 100/300/1000 A: vacuum 0.036/0.061/0.116; kappa 1e6, N = 1: 0.009/0.019/0.048; N = 2: 0.009/0.019/0.040. Net b_21 is 8-17% of vacuum at kappa 1e6 with the sign reversed (overshielding), and w follows Landreman's w = 4 sqrt(b/(m iota')) to ~10%.
- The pressure does not flatten at the island at any kappa or N tried (spread at fixed s as in the unperturbed profile, B.grad p in the island 3-4x outside); kappa 1e8 runs were still going at the time of this commit.
- Figures in `figures/dshape/islands/`; results of the runs: `results/isl/<name>.json` (scan summary) and `<name>_islands.json` (the larger npz of states and sections are not tracked).

## 2026-10-19: Remove the legacy step-2 driver and CurrentFZ
- `nest/run_step2.py` (the first J-on-the-box driver, unused) and `nest/current_fz.py` (the first Fourier-Zernike J, replaced by `boundary_potential.BoundaryPotential`) removed.
- `tests/test_desc_grid.py` now checks `desc_grid.flux_grid` and `evaluate_on_flux_grid` (full-torus tiling, Cartesian rotation) with DESC's own J as the current: equal to DESC's J on precise_QA and W7-X (error 0, tolerance 1e-12). `desc_grid.py` docstring and README updated.

## 2026-10-19: Repository cleanup: scripts, tests and figures in subfolders
- `scripts/`: the drivers stay at the top (`optimize_qa.py`, `optimize_fb.py`); `dshape/` (the axisymmetric tokamak study), `stellarator/` (precise_QA, NCSX, QH, CTH-like), `diagnostics/`, `modal_cyl/` (unchanged). Moved scripts find the repository one level higher; sibling imports from their own folder.
- Removed the DSHAPE scripts of the Cartesian-box and corrected-B_plasma runs, or superseded: gs_box_convergence, gs_bphi_check, gs_axisym_check, gs_check_norms, gs_force_fit, gs_x0_force, gs_plots, gs_plot_drift, gs_plot_null64, gs_nullspace, gs_plot_J_fit, gs_poincare, gs_edge_J (-> gs_edge_zoom), gs_cyl_convergence (-> gs_ladder), gs_long_run, gs_newton, gs_dense_plot (read gs_long_run's output).
- `tests/checks/`: the accuracy and convergence studies (cyl_*, convergence_b_plasma, axisym_tables, exact_G, floor_fixes); unit tests stay in `tests/`.
- `figures/`: `dshape/` (the h ladder, comparison with DESC, Poincare, edge J and force), `precise_QA/`, `ncsx/`, `qh/`, `cth/`; removed the figures of the L24 corrected-B_plasma runs (gs_fdrift*, gs_drift, gs_null*, gs_J_fit_L24, gs_weights) and the uncut edge plot. The DSHAPE scripts write to `figures/dshape/`.
- References in docstrings, README, docs and the modal_cyl scripts follow the new paths (earlier UPDATES entries keep the old ones). README: a "Repository layout" section.

## 2026-10-19: DSHAPE (axisymmetric, no perturbation) on the cylindrical annulus: h ladder to the exact limit
- Free-boundary DSHAPE (DESC L = M = 32, coils' field), basis x2 (L = M = 64, edge power 3), J fitted to J_DESC as the start, collocation 2 (L_basis + L_DESC) = 97 x 385, w_J 1e-3 toward the start J over the whole volume; each point converged with the exact LM; checked with the exact field on a common 81 x 321 grid.
- Exact axisymmetric B_plasma (`nest/axisym_field.py`): ring-current quadrature in polar coordinates about each target, with c/rho, R, Z tabulated exactly as Chebyshev(rho) x Fourier(theta) (one matrix product per node block; ~20-30x faster at the same accuracy). n_s = 96 by default: n_s = 48 was off by ~7e-9 of rms B near the edge, (256, 96) is at round-off against (1024, 192).
- Exact dense G (`axisym_field.plasma_field_matrix`, `DenseFields.from_fields(method="exact")`, `optimize_qa.py --dense-g exact`): B_plasma per coefficient from per-target quadrature moments, x3 in ~5 min instead of ~2 h on the a/640 annulus; equal to plasma_field to 1e-15.
- The annulus B_plasma floor (~5e-9 of rms B from a/320 on) had three causes: the reference quadrature (above), J interpolated to the nodes from the 4L x 6M flux grid (1.5e-6 in J) and S_m stored in float32 (1.2e-9, uniform). `nest/node_current.py` (`NodeFields`, `--j-nodes direct`: J from the basis at every node) and `--S-dtype float64` remove them: 5.2e-9 (a/160), 5.4e-10 (a/320), 7.1e-11 (a/640), ~h^3 (`scripts/gs_cyl_floor.py`, `gs_cyl_floor2.py`, `tests/floor_fixes.py`).
- Faster setup: |m| <= 1 single-layer matrices on the GPU (`nest/single_layer_fast.py`, same S to 5e-14, ~1 min at a/320 instead of ~30); the plasma location cached on disk (`cylindrical_grid.py`); `nest/boxes.py`.
- Exact LM (`scripts/gs_dense_solve.py`): dense Jacobian from column tables, QR of [A | r], SVD, damping scan, device TSQR up to 22 GB, stop on the Gauss-Newton share. Hessian in the physical norm (`gs_hessian.py`). Also `--j-target-from`, `CurrentError(rho_max)`, `CurrentMoments`, factored tensor tables (`flux_interp.FactoredFactors`).
- h ladder a/20 ... a/240 and the exact G (`figures/gs_ladder_h_fixed.png`, `scripts/gs_ladder.py --no-ref-panel`): the change between steps falls by ~5 decades, a/240 -> exact G is the smallest (B 3e-11, J along B 3e-9); <|F|>/norm 1.8e-6 (a/20) ... 8.63e-10 (a/240) against 8.65e-10 (exact G) and DESC's 5.88e-9.
- Against DESC L32 (`scripts/gs_vs_desc.py`, `figures/gs_vs_desc_h.png`): B 2.4e-7, J along B 6.0e-6, across 1.2e-6, p 2.5e-6, the same from a/160 to the exact G. Poincare (`scripts/gs_cyl_poincare.py`, `figures/gs_poincare_hN_a240.png`): surfaces within 2.3-5.8e-6 of DESC's in rho, iota within 3e-6; DESC's own field traced the same way stays within 2-40e-9 (`--field desc`, `figures/gs_poincare_desc_check.png`). J falls as (1 - rho)^2 to the LCFS, the force as (1 - rho)^3, DESC's J and force stay finite (`scripts/gs_edge_zoom.py`, `gs_edge_zoom_plot.py`, `figures/gs_edge_zoom_h_cut.png`).

## 2026-10-19: Merge branch `cylindrical-grid` into main
- The cylindrical-annulus J -> B solver (Chebyshev in r, z, Fourier in phi, James' free-space correction with S_m), its axisymmetric path, GPU S_m setup, mapped nodes and compact J sampling, next to main's free boundary (2026-10-16). Only UPDATES.md conflicted (entries of both sides kept, newest first).

## 2026-10-18: Compact J sampling (branch `cylindrical-grid`)
- `PointInterpolator(compact=True)`: the 4x4x4 Lagrange stencil formed per chunk from the points' stencil coordinates (25 bytes per point against 1280); values to rounding (<= 7e-16). `Fields(j_compact=...)`, `optimize_qa --j-sampling compact|cached`; default compact on the cylindrical grid, main's box unchanged.
- GPU (A100, production precise_QA, 5 iterations): no time cost (a/20 22.8 vs 23.0 s/it, a/30 51.1 vs 52.4), peak memory -0.48 GiB at a/20 and -1.63 GiB at a/30 (the cache size); the J -> B model drops 1.26x at a/40. Docs: `docs/cylindrical_grid.md`, "Compact J sampling".

## 2026-10-17: Cylindrical grid: mapped (KTE) nodes supported; not a memory lever (branch `cylindrical-grid`)
- `annulus_poisson`: radial eigendecompositions in real block form (complex pairs of mapped operators, `_block_eig`), complex z eigenbasis when the mapped z operator has pairs; `CylindricalSolver` divides per pair. Unmapped grids run the previous path (bitwise). Gate A2c (block reconstruction 4.5e-15, Dirichlet vs dense <= 1.5e-13, NumPy and JAX).
- `tests/cyl_b4.py cyl --alpha`. B4 on the smooth-current precise_QA: at equal h the map halves the nodes but raises the error (a/20 2.25e-4 -> 2.8e-4); at equal accuracy it needs 1.24-1.39x fewer total nodes but 1.2-1.3x more inside nodes (J sampling), converging more slowly. No net memory saving; default stays alpha = 0. Docs: `docs/cylindrical_grid.md`, "Mapped nodes".

## 2026-10-17: Cylindrical grid: axisymmetric path (branch `cylindrical-grid`)
- `CylindricalGrid.around` uses one plane (N_phi = 1) for N = 0 equilibria (`axisym=False` to override): the solve is two 2-D problems; `optimize_qa --grid cylindrical`, `--check-grid cylindrical`, `--dense` and `--bext desc --fit force` work unchanged. The axisymmetric cylindrical check defaults to 2x finer than the run.
- Gates D3a–D3e (`tests/test_cylindrical_axisym.py`): N_phi = 1 equals N_phi = 5 to ~1e-14 (B, F, J v, dense G); B order 3.3 in h.
- Tokamak L24M24, factors (2, 2, 1), 30 dense float64 iterations on the laptop: cylindrical a/80 reaches 3.9e-8 (checked 4.0e-8 on a/160), below DESC's 1.14e-7; at a/20 and on the box a/10 the optimizer fits the grid's B error (checks rise to 2.2e-6 and 9.8e-5). Start residual at equal h 2-3x below the box's; the box a/20 start takes 22 min, the cylindrical a/80 2 min.
- S_m CPU build: the block budget counts the Lagrange matrices (axisymmetric a/80 196 s -> 16 s on the laptop). Docs: `docs/cylindrical_grid.md`, "Axisymmetric path".

## 2026-10-17: Cylindrical grid: S_m setup on the GPU (branch `cylindrical-grid`)
- `nest/single_layer_jax.py`: the single-layer matrices S_m built on the device (Miller start sweeps per start-length bucket, one recurrence pass per chunk of targets, Lagrange maps and face products on the device). Used automatically on a GPU with x64 (`NEST_S_BACKEND=auto|jax|numpy`).
- Against the float64 CPU references for precise_QA a/20–a/60: ≤ 3.9e-14 row-relative. S_m at a/60 (16.3 GB) in 30 s on an A100-80GB (41 s on 40 GB), against 141 s on 32 CPU processes (~10 min on the 8 CPUs of a GPU node).
- `AnnulusPoisson`: per-mode eigendecompositions on threads (`NEST_EIG_THREADS`; BLAS pinned per worker; bitwise equal), a/60 96 s → 14 s on 8 CPUs; `cond_Vr` computed on demand.
- CPU build: process pool by default on Linux (`NEST_S_JOBS`); S_m cache key by `single_layer.S_NUMERICS` instead of the source of the building code; `tests/cyl_s_build.py grids --eq/--coils`.
- The precise_QA references on the Modal Volume were re-keyed to the new scheme. Docs: `docs/cylindrical_grid.md`, "Before a high-resolution run"; `tests/cyl_s_gpu.py`, `scripts/modal_cyl/run_s_gpu.sh`.

## 2026-10-17: Cylindrical-annulus J → B (branch `cylindrical-grid`)
- New free-space J → B solve on one field period of an annulus: Fourier in φ, Chebyshev in r and z, James' correction per toroidal mode (`nest/cylindrical_bs.py`, `cylindrical_grid.py`, `annulus_freespace.py`, `single_layer.py`, `toroidal_kernel.py`, `annulus_poisson.py`, `chebyshev.py`).
- `Fields` with a `CylindricalGrid` box, `optimize_qa.py --grid cylindrical`; `DenseFields` works unchanged; main's paths bitwise unchanged (C4).
- Gates A1–C6 pass (`tests/test_cylindrical_bs.py`, `tests/test_cylindrical_integration.py`).
- precise_QA, cylindrical path in float64 (the default, `cyl_dtype`; float32 is not accurate at production sizes and stalled the GPU optimizations): at equal h 3.4–4.6× less memory than the float32 VG box for the JᵀW product + J sampling (model), ~0.6× its exterior B error (converging ~O(h²) to 3.3e-5 at a/40), and 1.8× less time per optimizer iteration on an A100 (a/20: 23.6 vs 43.2 s). The box cannot reach a/40 on an 80 GB card; cylindrical float64 is estimated at a/60–65.
- GPU (Modal A100) results, the float32 correction and open items: `docs/cylindrical_grid.md`; job scripts `scripts/modal_cyl/`.

## 2026-10-16: NEST with a free boundary: B.n = 0 on a moving boundary, solved with force balance
- `nest/free_boundary.py`: the boundary's normal displacement xi is a new set of unknowns, x = (u, c_p, xi), extended inside as rho^|m|. J and p are carried with the coordinates.
  - New objectives: `BoundaryNormalField` (B.n/|B| on the boundary), `ToroidalFlux` (Psi held), `StoredEnergy` (int p dV in the moving volume) and `TransportedCurrentError`.
  - Pressure balance [[B^2/2mu0 + p]] = 0 holds identically: no sheet current, and p = 0 on the boundary.
  - The geometry is exact in xi, and so are the points at which B_plasma is interpolated (`MovingInterpolator`). B_ext and the box current are first order in the displacement (`BoxShift`). Re-anchoring (`Displacement.apply`) makes them exact.
  - `DirectField`: B_plasma on the boundary by direct Biot-Savart over the volume current (offset source grid, symmetry copies), for B.n. The box's B.n on NCSX's boundary is off by ~1.5e-3 |B| at a/10 (5e-4 at a/20) and differs between field periods. A run with the box's B.n fitted that error on its half-period grid.
  - Derivative of B on the boundary: the box's by default; `direct_tangent` uses a (coarser) direct field.
  - `FreeBoundaryPreconditioner`: the (u, c_p) blocks plus one exact xi block. xi is scaled so its columns match the current's (otherwise the damping floor kills the steps of u). A coupled Schur form (`coupled=True`) converged worse on NCSX and is off.
  - `release`: deletes an anchor's device arrays. JAX's caches otherwise kept about 4 GB per anchor, and the run ran out of memory after four anchors.
- `scripts/optimize_fb.py`: anchored Levenberg-Marquardt, with one box per stage and lam carried across anchors.
  - Options: `--fix-I` (I(rho) held), `--targets` (continue a run with its W, Psi and B.n normalization), `--start-eq`, `--direct N_RHO Q` (default 32 4), `--direct-tangent`.
  - Step control: `--max-disp` per anchor and `--max-step` per iteration (default max-disp / 3).
  - `--grad-every`: the gradient of B_ext is reused for a few anchors (B_ext itself is always exact).
  - It saves x and the moved geometry as a DESC file (`<out>_eq.h5`) for `poincare.py --eq ... --skip-start`.
- `scripts/fb_compare.py`: B.n/|B| maps, the normal displacement of the boundary and the boundaries of several runs.
- `nest/optimize.py`: `quadratic=False` checks each step against the true residual; `step_limit(d)` bounds the step length (used to bound the boundary's motion per iteration); a callback returning True stops the iterations.
- `tests/test_free_boundary.py` (NCSX, CPU):
  - xi = 0 reproduces `Fields` to 5e-15;
  - J v matches central differences to 5e-10 in xi, and the adjoint identity holds to 6e-15;
  - the model agrees with the displaced rebuild to second order.
- `free_boundary_desc.py`: chunked Jacobian in the final solve (40 GB A100). NCSX result `NCSX_fb_L16M16N12.h5`: the pressure jump drops from 8.3e-3 to 2.6e-3 rms, but B.n/|B| does not (4.4e-3 -> 4.5e-3 rms).
- NEST on it, fixed boundary (a/20 check; DESC 8.8e-4):
  - `--w-J 1e-3`: 6.8e-5, but the core iota rises from 0.35 to 0.44;
  - `--fix-I`: 8.9e-5 with DESC's iota. Its field has B.n/|B| = 2.9e-3 on DESC's LCFS.
- NEST free boundary, from the `--fix-I` optimum (xi up to M = 16, N = 12, direct B.n, a/10 then a/20):
  - force error 1.2e-4 on the a/20 check; B.n/|B| 3.0e-4 rms (2.7e-3 max), against 2.8e-3 (2.8e-2) with the boundary fixed;
  - the boundary moves up to 9 mm (3.3 mm rms).
  - Islands (`figures/ncsx_fixed_vs_free_poincare_periodized.png`, 800 periods):
    - rho 0.62-0.78 has nested surfaces between thin chains in both fields (Chirikov sigma < 0.72).
    - The 3/6 chain is about 3x wider with the free boundary: the free field does not shield the coils' 3/6 drive (2|c_mn| 9.8e-4 against 1.2e-4 fixed).
    - 3/5 is about the same width in both.
    - The a/10 stage of the free run is chaotic over rho 0.66-0.88 even periodized: its high-order b_mn are 3-10x those at a/20, so that chaos is resolution error.
  - Details and the lessons of the runs in `desc_input/vmec_free_boundary/README.md`.
- `poincare.py`: B_plasma is averaged over the NFP rotations before tracing (default; `--no-periodize`).
  - The box's field differs between periods by 1e-4 |B| rms at a/20. Its jump at the seam of the repeated period pushed every line outward: rho +0.07 over 800 periods, free boundary.
  - That drift was also the "stochastic band" at rho 0.62-0.78 and the narrow 3/5 chain of the first free-boundary sections.
  - New options: `--grid` (the traced field's cylindrical grid) and `--steps` (RK4 steps per period).
- `poincare.py --skip-start`; `poincare_flux.py` takes an optional DESC equilibrium for its iota curve; `poincare_compare.py` draws several runs side by side.
- `resonant_field.py`: options `--m-max`, `--k-max`, `--skip-start`, `--iota-eq`.
- `resonant_overlap.py`: chains in a band of rho, with b_mn, pendulum width and Chirikov sigma, against a long trace.
  - Per line, it reports the drift of rho and the thickness (rms of rho about a fit rho(theta)): surfaces < 6e-3, the 3/5 separatrix layer 2e-2.
  - The weighted Birkhoff rotation number stops converging at 1e-4-1e-5 on surfaces with this tracer, so it is not used to flag chaos.

## 2026-10-14: Free-boundary DESC inputs for NCSX, QH and CTH-like in their coils
- `desc_input/vmec_free_boundary/free_boundary_desc.py CASE`: from the fixed-boundary re-solve on the VMEC LCFS (B.n/|B| 2-3.5e-3 rms), the LCFS is solved for in the fixed coils. It minimizes `BoundaryError` with p, I and Psi fixed, `proximal-lsq-exact` at 12/12/8-10, and releases LCFS modes for k = 2, 4, 6. A tight solve then runs at 16/16/12, and the result is saved as `<case>_fb_L..M..N...h5`. Jobs can be split with `--resume` and `--no-final`.

## 2026-10-14: Axisymmetric tokamak: DESC-matched start, dense model, rank and drift of force balance
- `desc_input/tokamak_free_boundary/`: a free-boundary tokamak (DSHAPE boundary, p ∝ (1-ρ²)³, I ∝ 1-(1-ρ²)³, ⟨β⟩ 2.4 %, q 1.18 → 7.5) at L = M = 12, 16, 24, 32.
  - The external field is a REGCOIL `FourierCurrentPotentialField`, with B.n/|B| 5e-7.
  - Releasing the LCFS in that field moves it by at most 2.5 mm (`free_boundary_tokamak.py`).
- DESC-matched start, so that F_NEST(x0) = F_DESC on the collocation points:
  - `optimize_qa.py --bext desc` sets B_ext = B_DESC - B_plasma(x0);
  - `--fit force` (`nest/dense.py`) minimizes |(J - J_DESC) x B_DESC|² + eps²|J - J_DESC|². The difference drops to 2e-8 at L = 24, against DESC's 1.1e-7 (the fit of T gives 9e-7).
- `nest/dense.py`, `--dense`: B_plasma(u) = G u, with G formed once (one box solve per unknown) and cached on disk. A force-only run of 100 iterations takes 2-5 min on one A100. Also `--dtype float64` (float32 stops at ~1e-8), `--no-check`, `--track`, `--keep-iterates`, `--dense-batch`.
- `BoundaryPotential.fit`:
  - with stellarator symmetry the b modes with m > 0, n = 0 were never fitted; they are now (precise_QA start 1.7x better);
  - new options: `target="J"` and `edge_weight`.
- `objective.PressureSquare` (`--w-p2`).
- `ForceResidual` works without a box, and computes B_ext on first use. `DESCInput.B_ext` uses a fine, chunked source grid for surface currents.
- Scripts:
  - `gs_nullspace.py` (`--run`: exact Jacobian, QR + SVD);
  - `gs_force_fit.py`, `gs_x0_force.py`, `gs_check_norms.py`, `gs_box_convergence.py`;
  - `gs_check_end.py`: independent grid, a/30 box, DESC L32 as reference;
  - `gs_poincare.py`, `gs_plot_drift.py`, `gs_plot_null64.py`, `gs_plot_J_fit.py`, `gs_plots.py`.
- Findings (L = M = 24, float64):
  - The force Jacobian has full rank with no gap; 10-12 % of its singular values lie below 1e-4. These directions are J along B and Grad-Shafranov neighbours δp(ψ), δF(ψ).
  - Force-only optimization from the DESC-matched start reaches 2-6e-9 on its own grid. It does not deflate: ⟨p⟩, rms J and I are unchanged to ~1e-6, and the surfaces are within ~2e-5 of DESC's.
  - On an independent grid and box, F is 2-3x below DESC, and the fields are no closer to DESC L32 than DESC L24 is.
- Figures: `figures/gs_*.png` and `desc_input/tokamak_free_boundary/*.png`.

## 2026-10-13: NCSX resonant b^rho_mn against island widths
- `desc_input/vmec_free_boundary/resonant_field.py`: b^rho = B.grad rho / B.grad zeta on DESC's surfaces, Fourier analysed in (theta*, zeta) for m <= 12, |n| <= 4 NFP. It covers DESC (check: 7e-18), the coils alone, the start and any optimized x.
- `resonant_widths.py`: for each chain, the field's own resonance and shear from the traced lines, b_mn there, the pendulum width, and the width measured from locked lines.
- NCSX, low-order chains: predicted and measured widths agree to 20-30 %.
  - 3/5 does not move (rho 0.796). Optimization halves b_5,-3 (1.1e-3 -> 5.5-6.5e-4 for every w_J), and the island narrows from about 0.09 to 0.05.
  - 3/7 at w_J = 1e-3 grows: b is 7x the start's at the same rho, and the chain moves inward (0.365 -> 0.31) to where b is larger still.
  - 3/6 at w_J = 0 follows iota: the chain moves to rho 0.43, where b is 4e-4.
  - 6/9 ("2/3") sits at the iota maximum; it has no pendulum width.
  - Core b_mn of 1e-4 differ by up to 4x between a/10 and a/20; 3/5 does not.
- Figure: `figures/ncsx_resonant_field_widths.png`.

## 2026-10-11: QH and CTH-like through NEST; island interiors in the Poincare sections
- DESC re-solves: QH_5beta at 16/16/12 (force error 6.5e-4) and CTH_like at 12/12/8 (7.6e-4).
- NEST with `--w-J`:
  - QH reaches 4.3e-5 (15x below DESC) with iota held within 0.01.
  - CTH-like reaches 1.5e-4 at 3e-3. Its core iota sits just above 5/4, so a second 5/4 chain still opens.
- `poincare.py`:
  - also records the half-period plane;
  - `--islands` seeds each field's own chains (measured iota) across the chain at theta* = 0, pi/m, pi and pi + pi/m, so island interiors are traced.
- `poincare_flux.py` draws both planes. New `poincare_rz.py`: the sections in (R, Z) over DESC's surfaces.
- Figures: `figures/{ncsx,qh,cth}_*_islands_poincare_{flux,rz}.png`.

## 2026-10-11: precise_QA islands at iota = 1/2 with the current held (--w-J 1e-3)
- `desc_input/vmec_free_boundary/poincare.py`: `--island-lines k` starts k more lines across each low-order chain on theta = pi / m. This fills chains whose X-points lie on both theta = 0 and pi.
- precise_QA (`smooth_J_edgeJ_r0.3`), with `--w-J 1e-3` and refined to a/20: force error 2.13e-5, against DESC's 4.24e-5. iota stays within 0.015 of DESC's, and the 4-island iota = 1/2 chain is kept at rho 0.58-0.74.
- Pressure scans (`--p-scale` 1.5-2.5, with or without `--fix-I` or `--w-J`) push iota above 1/2 at fixed coils.
- Figures: `precise_QA_wJ1_a20_poincare_flux.png`, `wJ1_islands.png`, and the scans `pscale*_islands.png`. See `OPTIMIZATION_STATUS.md`.

## 2026-10-11: run_della.sbatch: compilation cache and platform allocator, as on Stellar
- `run_della.sbatch` now matches `run.sbatch`: a persistent JAX compilation cache in `.cache/jax`, and the platform allocator (override with `--export=ALL,XLA_PYTHON_CLIENT_ALLOCATOR=default`).

## 2026-10-11: Default workflow: free pressure with p >= 0, --refine, --eq/--coils; validation and island scripts
- `scripts/optimize_qa.py` defaults: pressure free, stored energy held (`--w-W 10`), p >= 0 (`--w-pos 10`), up to 100 iterations per box.
  - `--refine 20` continues on finer boxes.
  - `--eq` and `--coils` take any equilibrium.
  - The check is evaluated in radial pieces.
- `nest/objective.py`: `PressurePositivity`, a one-sided penalty on min(p, 0). `nest/precondition.py`: Gram matrices accumulated over radial slabs, and table columns gathered outside the Gram program. `nest/force.py`: `rho_slice`.
- `run.sbatch`: persistent JAX compilation cache and the platform allocator.
- New scripts:
  - `validate_opt.py`: pressure sign, net current, exact J, force split, Poincare;
  - `islands.py`: resonant b^rho and island widths;
  - `time_tests.py`: the test suite with timings.
- `OPTIMIZATION_STATUS.md`: precise_QA continued on a/20 reaches 8.3e-6, against DESC's 4.2e-5. Four equilibria in the same coils all end below DESC's own force error.

## 2026-10-11: Self-contained Poincare scripts
- `desc_input/vmec_free_boundary/poincare.py` has its own copy of the field-line tracer, so it no longer imports the uncommitted `scripts/validate_opt.py`. It saves only the crossings; `poincare_flux.py` draws the figure, and `--x start` traces the start alone.
- The R-Z Poincare figures are removed (unreadable for NCSX's thin crescent). The figures kept are the flux-coordinate ones (`figures/ncsx_*_poincare_flux.png`).

## 2026-10-11: Finite-beta free-boundary benchmarks with net current and their coils
- `desc_input/vmec_free_boundary/`: three published VMEC free-boundary runs with their coils, re-solved in DESC on the VMEC LCFS:
  - NCSX c09r00: <beta> 4.1 %, I = -179 kA;
  - QH: 5 %, bootstrap current -2.3 MA, coils from Wiedman et al. 2024;
  - CTH-like: 0.19 %, 43 kA.
- B.n/|B| in the coils is 2-4e-3 rms. The DESC force error is 2-7e-3.
- `vmec_profiles.py` rebuilds p and I from the VMEC input definitions as even series in rho. The `VMECIO.load` spline of the current makes iota spike at the axis.
- NCSX at L = M = 16, N = 12: the DESC force error is 5.8e-4. NEST (`optimize_qa.py --refine 20`) brings it to 5.2e-5, 11x below DESC.
- Without a profile constraint NEST moves the current to the axis (iota0 0.35 -> 0.56). With `--w-J 1e-3`, iota stays within 0.02 of DESC and F reaches 6.2e-5. The 3/5, 3/6, 6/11, 3/7 and 2/3 island chains appear.
- `poincare.py` and `poincare_flux.py`: Poincare sections of a NEST solution in DESC flux coordinates, with the iota of each traced line. Figures are in `figures/ncsx_*`.
- `run_della.sbatch`: XLA_PYTHON_CLIENT_MEM_FRACTION is 0.90, as on Stellar. At the default 75 %, the NCSX coil field ran out of memory on a 40 GB A100.

## 2026-10-10: Free pressure with the stored energy held: force balance below DESC's
- `nest/objective.py`: `PressureEnergy`, <p>_vol held at its start (one residual), so that the pressure can be freed without the plasma deflating; `SpectralInterpolator` one field component at a time (a third of the fine-grid memory; a/20 peak 12.9 -> 9.5 GB).
- `nest/precondition.py`: block Gram matrices accumulated over radial slabs (bounded memory), blocks padded to a few sizes; fast path also with pressure-only objectives.
- `scripts/optimize_qa.py`: truncated kernel by default; independent check on a finer box (2 x, at most a/20) and grid; `--w-W`, `--start`; device memory freed before the check. `scripts/decompose_opt.py`: F split along b (F_par = -b.grad p, which no current can change) and across, and the change of p.
- Result (smooth-current precise_QA, a/10, truncated kernel): with p fixed to DESC's profile the force error stalls at 1.09e-4, mostly F_par (DESC's surfaces are not exactly flux surfaces of the coil + plasma field). With p free and the stored energy held: <|F|>/norm 1.69e-5, and 2.75e-5 on the a/20 check, against DESC's 4.24e-5; p changes by 2.7 % rms. See `OPTIMIZATION_STATUS.md`.

## 2026-10-10: Objective, Jacobian, optimizer and preconditioner; speed and memory
- `nest/objective.py`: DESC-like `ObjectiveFunction` with ForceBalance, FieldError, CurrentError and PressureError (weights, targets, normalizations). `Fields` evaluates J, B, p and F at a tensor collocation grid: J and p by tensor evaluation, B_plasma from J on the box (cached flux-grid to box interpolation, as one gather), a Hockney-Eastwood or truncated kernel, and `SpectralInterpolator` (type-2 non-uniform FFT split so that the point-dependent part is cached; one component at a time). The derivative is explicit (dF = J_lin x B + J x B_lin - grad p): J v is one evaluation, J^T w its transpose; the linearization keeps only J and B at the points.
- `nest/flux_interp.py` (tensor grids, factors, geometry, current, pressure, Lagrange interpolation), `nest/batching.py`, `nest/jacobian.py`.
- Stellarator symmetry: half collocation grid, half flux grid, half box (`boundary_potential.py`, `pressure.py`, `force.py`, `grid.py`).
- `nest/optimize.py`: Levenberg-Marquardt, PCG on the normal equations, exact quartic line search, stagnation stop, time limit, callback; `nest/precondition.py`: block preconditioner over the unknowns sharing (|m|, |n|), built from the local columns with tables restricted to each block's modes, accumulated over radial slabs.
- `nest/fit_cache.py`: on-disk cache of the fits (keyed by the equilibrium file, the arguments and the fitting code). `force.py`: positions, sqrt(g) and DESC's targets on the tensor grid (checked against DESC's pointwise evaluation on a sample). `boundary_potential.py`: optimized einsum in the fit (140 s -> 0.02 s).
- Scripts: `optimize_qa.py` (optimization with an independent check on a finer grid and a finer box), `audit.py` (time and memory), `spectrum.py`, `precond_test.py`, `local_blocks.py`, `diagnose_opt.py`. `run.sbatch`: JAX may use 90% of the GPU. Tests: `test_objective.py`, `test_force.py` updated.
- Smooth-current precise_QA, a/10, 90k unknowns, truncated kernel: <|F|>/norm 4.68e-4 -> 1.09e-4 (DESC 4.24e-5) in about 16 min on one A100 (setup 1.5 min, 0.11 s per CG iteration, GPU peak 5.8 GB); before, 7.5 h and 1.8e-4. `OPTIMIZATION_STATUS.md` summarizes the methods, tests and results.

## 2026-10-09: Truncated Green's function for B_plasma, convergence study
- `nest/magnetic_field.py`: `truncated_green_hat` and `b_plasma_truncated`, free-space B = mu0 curl (G_L * J) with the Vico-Greengard-Ferrando truncated kernel G_L_hat = (1 - cos(L k)) / k^2, L = half the box diagonal (same no-image region as Hockney-Eastwood). The kernel is built once from s_x s_y s_z residue-class FFTs of the box size (no oversampled grid): under 2 s up to 144M points on one A100. `b_plasma_periodic` and the new method share `_curl_solve`.
- `tests/test_magnetic_field.py`: Gaussian-blob check of the kernel against the exact potential and B (3.9e-10 and 9.1e-8 in float64; Hockney-Eastwood 2e-2 on the same grid); the truncated method printed in checks 1-3; `continuous_biot_savart` takes a quadrature resolution and works in blocks.
- `tests/convergence_b_plasma.py`: B_plasma vs h for the three methods at fixed points (nested boxes): exterior vs continuous Biot-Savart, interior self-convergence, rho bands. Smooth-current precise_QA (BoundaryPotential L, M, N = 48, 48, 35, s = 2), interior rms vs the truncated B at a/30: truncated 1.4e-3, 5.2e-4, 3.0e-4 at a/10, a/15, a/20, against Hockney-Eastwood 1.2e-2, 5.2e-3, 2.9e-3 (exactly second order); exterior, both about 1e-4 rms from a/15 on. W7-X, BoundaryPotential at the equilibrium resolution: inside the plasma the truncated kernel is about 2 to 2.5 times more accurate than Hockney-Eastwood at the same h (rms 1.5e-3 vs 3.7e-3 at a/15 for s = 4); outside, Hockney-Eastwood is better (a trapezoidal rule of a smooth integrand, rms 6.7e-5 at a/20 for s = 4, against 8.1e-4). With the fit at 3 times the equilibrium resolution (L, M, N = 36, 36, 30), J is not resolved on the box up to a/20 and no method converges.

## 2026-10-09: Plot scripts
- `scripts/plot_J_vs_desc.py`: makes `figures/J_vs_desc.png` (DESC J vs the BoundaryPotential fit).
- `scripts/plot_edge_layer.py`: makes `figures/edge_layer_J.png`; needs the removed edge taper (run at commit a5cbd41).

## 2026-10-09: Job scripts, step 2 driver and figures
- `run.sbatch`: sets NEST_DEVICE=gpu and stops if JAX does not get the GPU; `run_cpu.sbatch` (CPU jobs on Stellar); `run_della.sbatch` (one A100 on Della, env `nest-gpu` in /scratch/gpfs/EKOLEMEN/lk1158).
- `nest/run_step2.py`: driver for J on a Cartesian grid (h = a / frac) with checks against DESC (uses `CurrentFZ`).
- `figures/J_vs_desc.png` (DESC J vs the BoundaryPotential fit) and `figures/edge_layer_J.png` (the same with the edge taper, since removed).

## 2026-10-09: Force residual at iteration 0, spectral interpolation of B_plasma, reconstruction tests
- `nest/force.py`: `ForceResidual`, F = J x (B_ext + B_plasma) - grad p on a flux-coordinate collocation grid (default DESC's QuadratureGrid(2 L, 2 M, 2 N), volume weights sqrt(g) w), measured as <|F|>_vol / <|grad |B|^2 / 2 mu0|>_vol like DESC; `decompose` splits F exactly into DESC's own imbalance, the current fit, the field mismatch and the pressure fit. Smooth-current precise_QA at iteration 0: 4.6e-4 (DESC 4.24e-5), dominated by DESC's free-boundary field mismatch (3.9e-4), the same at h = a/10 and a/20.
- `nest/magnetic_field.py`: `interpolate`, spectral interpolation of a box field at arbitrary points with a type-2 non-uniform FFT (`nufftax`, new dependency), one component and chunk of points at a time, single precision for float32 fields; the Biot-Savart kernel keeps the requested precision.
- `nest/desc_input.py`: `iota(rho)` and `compute_points`, DESC quantities at scattered flux coordinates without surface integrals.
- Tests: `test_force.py`; `test_reconstruction.py` (manufactured field: FFT + interpolation second order; J of the box vs the fitted series; B_plasma convergence a/10, a/20, a/30, order ~2.2). Large boxes need XLA_PYTHON_CLIENT_ALLOCATOR=platform.

## 2026-10-09: Remove the edge taper
- `nest/boundary_potential.py`, `nest/current_potential.py`: the `edge_width` option (DESC's current tapered to zero over the outer layer before the initial fit, and the boundary-compatibility step of `target_potential`) and `edge_taper` are removed; the initial fits use DESC's current as it is. The structural edge conditions are unchanged.
- Tests updated (`test_boundary_potential.py`, `test_smooth_edge.py`); both pass on the smooth-current precise_QA and W7-X.

## 2026-10-09: Boundary-adapted current potential (J_potential_notes.pdf)
- `nest/boundary_potential.py`: `BoundaryPotential`, J = curl T with T_theta = I/2pi h_0 + d_theta chi + rho^2 f a, T_zeta = d_zeta chi + f b, f = (1 - rho^2)^s. For every coefficient vector: div J = 0, J = 0 on the LCFS, J regular at the axis (exact rho = 0 limit), net toroidal current I through every cross-section. Free coefficients u = (chi, a, b), J linear in u, differentiable with JAX.
- Changes from the notes: rho^2 factor on a (axis regularity), chi includes the sin(|m| theta), n = 0 modes, alpha(zeta) kept out as a pure gauge but fitted jointly when matching the target T.
- Initial fit of T itself (not J), mode by mode; DESC's current tapered to zero over the outer 0.1 in rho, with the boundary values made compatible with the representation, so the initial J goes to zero smoothly across that layer. Smooth-current precise_QA: J matches DESC to 4.9e-4 for rho <= 0.9.
- `nest/separable.py`: coefficients can be passed as JAX arrays (`scatter`, `evaluate(..., C=)`).
- `tests/test_boundary_potential.py`: divergence, LCFS, axis, net current, DESC reconstruction, derivatives, save/load.

## 2026-10-08: Grid classes and a common DESC input class
- `nest/desc_input.py`: `DESCInput`, the starting point for any DESC input.
  - Loads an equilibrium file (an Equilibrium, or the last member of an EquilibriaFamily), or wraps an Equilibrium, with an optional external field (a coil or field file, or a field object).
  - Gives NFP, resolution, prescribed profile, a, R0, volume, <beta>, the boundary extent, the R, Z geometry series, and `B_ext(x)` at Cartesian points.
  - Checks phi = zeta and the sign of sqrt(g).
- `nest/grid.py`:
  - `FluxGrid` (G1): one-period DESC grid with its torus copies, positions and weights.
  - `CartesianGrid` (G3): `periodic_box`, spacing, cell volume and wavevectors; `locate_plasma` (mask, inside points, flux coordinates); `scatter` / `gather` between the box and the inside points.
  - The inverse map `map_to_flux` and `plasma_extent` moved here from `cartesian_grid.py`, which re-exports them.
- `cartesian_grid.py`, `desc_grid.py`: `cartesian_grid`, `current_on_cartesian`, `flux_grid`, `evaluate_on_flux_grid` keep their signatures and use the classes. They take a `DESCInput` or an Equilibrium.
- `tests/test_grid.py`: DESCInput against DESC (B_ext vs the coils to 5e-16), FluxGrid positions, the box, plasma volume on the box vs DESC (4e-4 to 5e-4), and the inverse map. All existing tests pass with the change.

## 2026-10-08: Smooth edge for J and p, J.n = 0 removed, tests on the smooth-current precise_QA
- `nest/current_potential.py`: smooth-edge option (`edge_order` k, e.g. 2): d_rho^j T(1) = 0 for j = 1..k, so sqrt(g) J^theta and sqrt(g) J^zeta vanish like (1 - rho)^k at the LCFS; fit of J under these constraints, with DESC's J tapered to zero over the outer 0.1 in rho (net currents kept); `impose_smooth_edge` for any coefficients. The J.n = 0 projection (`impose_no_normal_current`) is removed for now.
- `nest/pressure.py`: smooth-edge option p = (1 - rho^2)^(k+1) Q.
- Tests use `precise_QA_free_boundary_smooth_J_edgeJ_r0.3.h5` instead of the iota-prescribed precise_QA: the Zernike fit of J is exact to 2e-8 and B_plasma converges like h^2. New `test_smooth_edge.py`; the current through phi = 0 is now compared with DESC's J through the same cross-section; check 3 of `test_magnetic_field.py` handles current-prescribed equilibria.

## 2026-10-07: precise_QA free boundary with the current density going to zero at the LCFS
- `desc_input/precise_QA_free_boundary_smooth_J/`: the current is prescribed with dI/ds = 0 and p ~ (1-rho^2)^3 at the edge, and the free boundary is redone with LCFS modes <= 5. The current is rescaled so iota = 1/2 stays at rho = 0.55, and the edge J_parallel enters the free-boundary objective.
- Recommended initial configuration: `precise_QA_free_boundary_smooth_J_edgeJ_r0.3.h5`. B.n/|B| is 2.45e-3; the net edge current is ~0, with a residual rms of 6 % of the peak.
- Diagnostics and tests in the same folder: edge-current tables, J_parallel maps on the LCFS, the (m, n) spectrum, and the edge force-balance and resolution tests.
- `desc_input/README.md`: overview of the free-boundary inputs and how they were made.

## 2026-10-07: B_plasma from J (step 5), pressure as a Fourier-Zernike field
- `nest/magnetic_field.py`: B_plasma on the box from the FFT of J, periodic (i mu0 k x J_hat / |k|^2) and free space (Hockney-Eastwood convolution with the Biot-Savart kernel, no periodic images).
- `nest/pressure.py`: p(rho, theta, zeta) as a Fourier-Zernike series, fitted from DESC's p(rho), with p = 0 on the boundary and grad p from the fixed geometry.
- `nest/cartesian_grid.py`: shared chunked evaluator; p and grad p on the box (`pressure_on_cartesian`).
- Tests: `test_magnetic_field.py` (free space = direct grid sum to 1e-7; 0.4 % vs exact B at a/20; periodic images 3-10 %; B_coil + B_plasma vs DESC B), `test_pressure.py` (fit, p = 0 on the edge, box).

## 2026-10-07: Current potential J = curl T, FFT of J
- `nest/current_potential.py`: J = curl T with T a Fourier-Zernike potential (gauge T_rho = 0), fitted from DESC; J.n = 0 imposed through the boundary values of T, keeping the net currents.
- `nest/separable.py`: shared separable (l, m) x n evaluation of Fourier-Zernike series.
- `nest/fourier.py`: rfft of J on the periodic box, wavevectors, inverse.
- `nest/current_fz.py`, `nest/cartesian_grid.py`: point evaluators, so the Cartesian mapping works with either J representation; evaluation chunk 50000.
- Tests: `test_current_potential.py` (curl T vs DESC J, J.n = 0), `test_fourier.py` (wavevectors, inverse, Parseval, k = 0 mode), `test_cartesian_grid.py` now uses the potential.

## 2026-10-06: Scripts that built precise_QA_free_boundary_L16N14.h5
- `desc_input/precise_QA_free_boundary/`: free-boundary run and refinement scripts.

## 2026-10-06: Periodic box, faster inverse map, finite-beta test equilibria
- Periodic box G3 (factor 2, FFT-friendly sizes), J in float32.
- Inverse map by Newton in (rho cos theta, rho sin theta), separable series evaluation.
- Tests on the free-boundary precise_QA and W7-X; cheb1 grids; higher fit resolution.

## 2026-10-06: Cartesian grid mapping of J
- `nest/cartesian_grid.py`: uniform grid around the plasma, inverse map to flux coordinates, J on the grid (zero outside).
- `tests/test_cartesian_grid.py`: current through phi = 0 and int |J| dV vs DESC.

## 2026-10-05: Fourier-Zernike current fit
- `nest/current_fz.py`, `nest/desc_grid.py`: J fitted as a Fourier-Zernike series, evaluated on a ConcentricGrid; test vs DESC J.
