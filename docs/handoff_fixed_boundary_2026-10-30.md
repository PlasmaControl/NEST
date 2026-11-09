# NEST fixed boundary: orientation, how-to, pitfalls, and hand-off to a DSHAPE island study

Written at the end of the 2026-10-28 to 10-30 session for an agent that has not seen the conversation.
Everything quoted as a number was measured in this project; items marked **hypothesis** were not tested.
Read this together with `docs/fixed_boundary_report.md` (the full fixed-boundary record and run commands) and
`UPDATES.md` (dated log, newest first).

---

## 1. Where things are

| what | where |
|---|---|
| repository | `PlasmaControl/NEST`, branch `fixed-boundary`, worktree `/Users/singh/PycharmProjects/NEST/NEST-cp` |
| other worktrees | `NEST` (main), `NEST-lm` (`lm-trust`), `NEST-closure`, `NEST-cyl`, `NEST-pois` (all under `/Users/singh/PycharmProjects/NEST/`) |
| DESC checkout | `/Users/singh/PycharmProjects/DESC` (put it on `PYTHONPATH`) |
| local conda env | `nest` (`/Users/singh/.claude-science/conda/envs/nest`, Python 3.12, JAX CPU) |
| Modal image / volume | `im-85tKPZB30SmMyYst74flDg` (JAX 0.7.2 CUDA) / Volume `nest-container-mask` mounted at `/results` |
| Modal job scripts | `scripts/modal_fixed/` (`fx_run.sh`, `py_run.sh`, `test_run.sh`, `gpu_entry.py`, `README.md`) |
| states (warm starts) | `results/meanforce_fixed/states/*.npz` (`results/` is gitignored: `git add -f`) |

**Git.** Commits are authored with
`git -c user.name="Jay Singh" -c user.email="jaydeepsingh15@gmail.com" commit ...`. The sandbox cannot resolve
github.com, so the user pushes (`cd ~/PycharmProjects/NEST/NEST-cp && git push origin fixed-boundary`). At the end
of this session the branch was 4 commits ahead of origin: `b5f8812` line scans, `d065e38` J_I scans + a/80 grid
check, `49c7dd3` subproblem anatomy, `404d9f9` test-2 driver (`stiff_modes.py`).

**The sandbox workspace is wiped when idle.** Uncommitted work in the workspace (staged source trees, tarballs,
copied states) disappears. Commit code to the repo as soon as it compiles, and save results with `save_artifacts`.
This happened in this session: the test-2 driver had to be rebuilt from the transcript.

---

## 2. What the fixed-boundary code solves

- **Domain and unknowns.** The plasma volume is the LCFS of a DESC equilibrium (NCSX: `NCSX_fb_L16M16N12.h5`).
  J = curl T from a `BoundaryPotential` (Fourier–Zernike potentials a, b plus chi modes) in the DESC flux
  coordinates. By construction div J = 0, J -> 0 at the LCFS, and the net toroidal current I is fixed at DESC's
  value. Unknowns u: n_u = 2K + n_chi (48/32/18: 41206 = 2 x 20011 + 1184).
- **Field.** B = B_ext + B_plasma. B_ext comes from the coils passed with `--coils` (NCSX: `NCSX_coils.h5`, a
  MixedCoilSet). B_plasma is Biot–Savart of J on a cylindrical annulus (FFT); `--frac` sets the grid spacing a/frac.
  B.n on the LCFS and the toroidal flux are **not** constrained.
- **Pressure is eliminated.** p = (1 - rho^2)^3 Q, with Q a Fourier–Zernike series at the **same (L, M, N) as the
  current potentials** (n_p = 20011 at 48/32/18). p is the weighted least-squares fit of grad p to J x B, so the
  force is f = Pi(J x B) with Pi = I - P G^-1 P^T. The pressure Gram G is factored by Cholesky (cond ~1.6e6).
  Collocation is at 1.5 x (basis + equilibrium) resolution, so the products are dealiased. p, however, cannot
  absorb gradient content of J x B above (M, N).
- **Objective.** Misfit phi = 1/2 |sw (J - J_fit)/rms|^2 to the DESC current J_fit, subject to a mean-force bound
  M = <|F|>_vol / norm <= alpha M_DESC (NCSX: M_DESC = 8.7788e-4 at a/40). The bound is enforced with an augmented
  Lagrangian, psi = rho/2 [max(0, c + nu/rho)^2 - (nu/rho)^2], with c = M/eps - 1.
- **Solver.** Matrix-free trust region (GLTR/Lanczos) on a Gauss–Newton-type model:
  - beta model (`FixedRowsB`): the curvature of |z_i| along z_i is scaled by beta = 0.5;
  - force-metric preconditioner L L^T = H_J + H_F,loc + a_F a_F^T. H_F,loc includes the pressure projection
    exactly but **drops J0 x dB** (the Biot–Savart response);
  - least-squares multiplier update, clipped to [1/4, 4] of the estimate.

**Code map.**

| file | contents |
|---|---|
| `nest/meanforce_fixed.py` | `FixedAnchor` (state, implicit p projection), `FixedRows` / `FixedRowsB` / `FixedRowsH` (GN, beta, exact-HVP rows), `_lin_impl`/`_linT_impl` (exact linearization), `gram_J*`, `gram_F_local*`, `force_metric*` (preconditioner), `lanczos_tr_step_H`, `_tridiag_tr_indef`, `use_models`, `LinearConstraints`/`ProjectedMetric` (pins, untested), `embed`/`embed_truncate`, p-norm bound |
| `nest/meanforce.py` | `Weights` (psi, d1, `d2_model` with the band rule), `Solver` (outer/inner loop rules) |
| `nest/pointwise.py` | `Metric` (`back`, `back_T`), `lanczos_tr_step` |
| `nest/pressure.py`, `nest/desc_input.py` | pressure basis; DESC loading and B_ext |
| `scripts/meanforce/mf_fixed.py` | the driver (all runs) |
| `scripts/meanforce/*.py` | diagnostics: `poincare_fixed.py`, `pressure_sections.py`, `resonant_sources_fixed.py`, `radial_force.py`, `shielding_fixed.py`, `sigma_lines_fixed.py`, `hamada_lines_fixed.py`, `fieldline_iota.py`, `line_scan_fixed.py`, `subproblem_anatomy.py`, `stiff_modes.py`, `constraints_drift_fixed.py`, `mde_budget_fixed.py`, `edge_force_fixed.py` |
| `tests/test_*_fixed.py` | gates: `test_meanforce_fixed` (G1–G6), `test_hvp_fixed`, `test_pnorm_fixed` (pass); `test_constraints_fixed` (C1–C7) **not yet run** |

**Driver defaults are the working configuration:**
`--model beta --beta 0.5 --precond force --lanczos-stop tr --cg 400 --delta0 1e-3`, LS multiplier on, lean metric
builds. Other useful flags:
- `--start <npz>` (warm start) and `--alphas` (bound schedule);
- `--force-norm mean|p` (p-norm/peak bound) and `--eval-only` / `--eval-state` (evaluate states, no solve);
- `--pin-current` / `--pin-flux` (untested constraints);
- diagnostic modes: `--subproblem-test` (one step), `--anatomy K --anatomy-deltas ...` (Lanczos/tolerance
  anatomy) and `--stiff-modes K_FULL K_LOC NVEC` (test 2). These use `--test-model beta|gn|hvp` and exit after the
  diagnostic.

---

## 3. How to run experiments

### Local (CPU), smoke tests only
**Ask the user before any local CPU job**: other agents may share the 8 GB laptop. Run one job at a time. A
smoke test at 8/8/6 takes ~6 min and ~3 GB:
```
cd /Users/singh/PycharmProjects/NEST/NEST-cp
PYTHONPATH=/Users/singh/PycharmProjects/DESC:nest:scripts/meanforce JAX_PLATFORMS=cpu \
  python scripts/meanforce/mf_fixed.py --LMN 8 8 6 --frac 20 --alphas 0.7 --stiff-modes 40 30 3 \
  --test-model beta --out /tmp/smk
```
macOS has no `timeout`. Run long local commands with `background: true`.

### Modal (GPU)
1. **Stage the inputs** in the workspace (they are not persistent):
   - source: `git -C NEST-cp archive HEAD nest scripts tests | tar -x -C stage/src`, plus `desc_input/...` for the
     case (NCSX: `desc_input/vmec_free_boundary/NCSX`);
   - `tar --exclude=__pycache__ --exclude='*.pyc' -czf modal_fx/nest_src.tar.gz -C stage/src nest scripts tests desc_input`;
   - `desc_src.tar.gz`: `tar -czf ... -C /Users/singh/PycharmProjects/DESC desc`;
   - `deps_src.tar.gz`: `nufftax adv_jax_math interpax_fft optax absl` from the `nest` env's site-packages;
   - copy `scripts/modal_fixed/{fx_run.sh,py_run.sh,gpu_entry.py}` and the start state.
2. **Submit one job per `repl` cell, with `background: true`**, and tell the user an approval card is coming:
   ```python
   H200 = {'image': 'im-85tKPZB30SmMyYst74flDg', 'gpu': 'H200', 'cpu': 16, 'memory': 262144,
           'volumes': {'/results': 'nest-container-mask'}, 'timeout': 3600}
   c = host.compute.create('byoc:modal', provider_params=H200)
   j = c.submit_job(intent='...', command='bash fx_run.sh <TAG> --LMN 48 32 18 --frac 60 --alphas 0.001 --start /work/<state>.npz ...',
                    inputs=['modal_fx/fx_run.sh', 'modal_fx/nest_src.tar.gz', 'modal_fx/desc_src.tar.gz',
                            'modal_fx/deps_src.tar.gz', 'modal_fx/gpu_entry.py', 'rsin/<state>.npz'],
                    outputs=['out/<TAG>.log', ...], run_timeout_s=3300)
   ```
   `fx_run.sh <tag> <mf_fixed args>` adds `--out ../out/<tag> --mirror /results/fx/<tag>`. `py_run.sh <tag>
   <script> <args>` runs any diagnostic with `--out ../out/<tag>.npz`.
3. **Monitor.** The per-step JSON (`/results/fx/<tag>/<tag>.json`) is mirrored as the run goes. The `.log` is
   copied only at the end. For a live log, exec into the job's sandbox from `compute_provider`:
   `modal.Sandbox.list()` → `sb.exec("bash", "-c", "tail /work/out/<tag>.log")`. Volume reads go through a
   CPU `modal.Sandbox.create(..., volumes={"/results": vol})`.
4. **Harvest.** Outputs arrive under `hpc/<job_id>/out/`. If the harvest fails, the volume mirror has them.
5. **Sizing (48/32/18).**
   - H200 for solves: a/40 uses ~20 GB in use with a 53–56 GB peak; a/60 peaks at ~70 GB at the metric build.
   - Setup takes ~6 min at a/60; Lanczos costs ~0.53 s per iteration at a/60.
   - A100-80GB is enough for diagnostics.

---

## 4. Pitfalls (each one cost time in this project)

### Objectives and diagnostics
- **Pointwise targets pin DESC's surfaces.** The misfit to J_fit pulls toward DESC's nested current, shielding
  sheet included. Calibration sources and coefficients do not pin surfaces; pointwise targets on J or p do. Decide
  deliberately what the target is.
- **Normalize against the right quantity.** In `subproblem_anatomy.py` the logged `dec/f0` divides by
  1/2 |r|^2 of the beta-model rows (9.95), not by the merit (1.93). The npz stores absolute decreases. Ratios,
  cosines and shortfalls are unaffected. This was misreported once in this session.
- **Check the units of a saved field before plotting.** Example: `gc` in the anatomy npz is the signed gradient
  component, so the share is gc^2/|g|^2.
- **Grid resolution matters at small force.** The alpha 0.001 checkpoint has mean M/M_DESC 0.0037 / 0.0016 /
  0.0015 at a/40 / a/60 / a/80. a/40 is inadequate below about alpha 0.005.
- **The pressure space has the same resolution as the current potentials.** On another code path, equal resolution
  left f and <|F|> ~7% high, and ~1.5x saturated them. This has not been checked on this branch.
- **DESC sign conventions.** iota is negative for NCSX, so write resonant harmonics as (m, -n). The IASR map
  flipped Z for m < 0 sine modes; check signs when adding boundary modes.

### Solver
- **The augmented-Lagrangian penalty dominates at small alpha.** In fxA001 the penalty was >97% of the merit, with
  rho·c >> nu* (~0.0075–0.02). Steps are then effectively "minimize M".
- **The LS multiplier clip needs stationary inner solves.** With capped inner loops it holds nu near 0 and rho runs
  away (a capped 8/8/6 test stalled at M/eps 1.97). Capped local tests of multiplier rules are invalid.
- **Keep `--lanczos-stop tr`.** The old `both` rule needed 224–400 Lanczos iterations per boundary step versus
  79–102.
- **tol 1e-2 is adequate; tol 1e-1 is not.** 1e-1 produced a false interior stop at k = 29 (|y| 0.014, cos 0.2 to
  the converged step).
- **Production Lanczos has no stall stop.** It stops only on the residual tolerance, Lanczos breakdown, or the cap
  (400).
- **Exact HVP fails near the bound.** `--model hvp` was 2.8x faster at alpha 0.7. At alpha 0.05 it collapsed the
  trust radius and met negative curvature.
- **Known bug:** a "flat" step counts as stationary when the trust radius has collapsed.
- **Use FP64 for the metric.** FP32/TF32 builds lose definiteness once trace H_F/H_J > ~1e8 (it was 6.9e11 at
  snapC).
- **Use Cholesky, not eigh, on large matrices.** `jnp.linalg.eigh` fails above ~25k on the H200 (cusolver);
  Cholesky works at 29k.

### Modal and operations
- **Long jobs must stream per-iteration progress** (never silent). Checkpoint every accepted iteration. Stop a job
  as soon as it can no longer produce new information.
- **Approval cards.**
  - A submit that blocks with no sandbox is waiting on an approval card: tell the user promptly.
  - After a missed card the queue can wedge, with no card and no sandbox; restarting the app fixed it.
  - A force-cleared submit can still land later, so check the ledger before resubmitting.
  - If the user declines a card, do not resubmit unless they ask.
- **Workarounds for failures.**
  - Harvests can fail on jobs that completed; recover from `/results/fx/<tag>/`, base64 + sha256.
  - `run.sh` is a reserved input name, so use `fx_run.sh`.
  - B200 and CPU-only submits have hung in staging; use H200 or A100, and run CPU tests in an A100 container via
    `test_run.sh`.
  - `compute_provider` cannot create GPU sandboxes. A diagnostic can run inside an already-running H200 job's
    sandbox (`sb.exec` + nohup, `XLA_PYTHON_CLIENT_MEM_FRACTION=0.40`).
- **Smoke-test new code** (CPU at 8/8/6) before paying ~6 min of GPU setup.

### Working style the user expects
- For conditioning or scaling problems, first look at how DESC handles them, then discuss.
- Argue a method change from data before adopting it.
- Explain the mechanism, not only the numbers.
- Commit and push only when asked, with a dated `UPDATES.md` entry.
- Write-ups go slowly, deriving each step, with worked examples.

---

## 5. What was run in this session (NCSX, fixed boundary, 48/32/18)

- **Alpha ladder with the mean bound.**
  - alpha 0.05 → 0.005 (a/40) lowered |F| in all 193 radial shells (peak x0.125, mean x0.10).
  - alpha 0.001 (fxA001, a/60) fell from M/eps 3.67 to 1.59 in ~2 h and was cancelled. The snapC checkpoint has
    mean M ~0.0016 M_DESC.
  - Islands close as alpha falls: the 3/5 chain is flagged in 0.049 / 0.0075 / <=0.004 of seeds at alpha 0.05 /
    0.005 / 0.001.
  - p(axis) drifts down: 6.855e4 → 6.13e4 → 5.97e4.
- **Peak (p-norm) bound fxPA3:** 0.02508 x DESC interior plus edge, from alpha 0.005.
  - The extra force went to the core and mid-radius, mostly parallel. No island is p-flattened.
  - Islands store error; the max-type bound is the weaker constraint. The user chose to keep the mean bound.
- **Delta-sheet question.** The shielding-sheet structure is already in J_fit, which over-cancels the coils by 15%.
  ΔJ only removes that excess, smoothly. The pressure-driven Pfirsch–Schlüter resonance is suppressed separately
  (Hamada x0.24–0.54).
  - The free B.n is not used to dodge the drive: (5,-3) b^rho at rho 0.95 goes from -0.41 (J_fit) to ~0 in every
    optimized state.
- **Line scans** (F is exactly quadratic along lines in (u, I)):
  - J = 0 is outside the solver's space. J_I (u = 0, I fixed) has M = 21.7 M_DESC.
  - Chords between alpha states rise in the middle; rays from J_I fall to a sharp minimum exactly at each state.
  - So the feasible set is a narrow curved valley.
- **Subproblem anatomy** (fxANAT at snapC, full-reorthogonalized Lanczos K = 3000):
  - At tol 1e-2 the step is already the converged one: -0.10% (Δ 0.88) and -0.81% (Δ 8.8) of the merit, ratio
    0.94–1.04.
  - The full Newton step predicts -51%, but the merit goes from 1.93 to 420.
  - Production sat at Δ = 19.2, with ratio 0.55–0.69 and ~1% per boundary step. The usable radius is therefore
    between 9 and 19: progress is limited by model validity, not by the inner tolerance.
  - Ritz values span [3.5e-8, 7.5e5] (κ ≈ 2e13). 8 modes hold 94% of |g|^2, yet all step norm lies at θ < 0.75.
    Momentum methods are ruled out (√κ ~ 5e6).
- **Deflation simulation** (offline, from the anatomy spectrum; reproduces the GPU iteration counts):
  - ideal deflation saves ~1 Lanczos iteration per mode up to m ≈ 32;
  - 99% of the decrease takes 112 / 104 / 80 / 52 / 31 iterations at m = 0 / 8 / 32 / 64 / 128 (Δ 0.88);
  - deflation also needs a decrease-based stop, because the relative tolerance becomes stricter.
- **Test 2 (stiff-mode origin):** only the CPU smoke test ran (8/8/6, alpha 0.7, from J_fit).
  - Local model (no J0 x dB): spectrum [0.50, 0.998]. Full model: κ ≈ 1.1e4.
  - Top modes: R_nl/R_loc 14–20. Bottom modes: local and nonlocal parts cancel.
  - The snapC H200 run was declined and has **not been run**:
    `--stiff-modes 600 300 16 --test-model beta`, ~15 min.

Artifacts (read with `read_file(version_id=...)`):
- `anatomy_snapC.png` 37f801da-9953-47ee-b246-924b13d7463c;
- `fxANAT_anatomy.npz` 51c557ac-ba4b-4675-971d-df496b28f4ed;
- `fxANAT.log` 3627dfcf-295b-4aa0-9d99-a8fb7c7841eb;
- `deflation_sim.py` 5b3227d1-3bf0-47cf-a8c6-a525be4cc770 / table 04dc6b50-f038-4602-9978-e8d9c46f6b41;
- `line_scan_force.png` ddeea0fc-4279-4c67-b2e3-2dcdcbcf7111;
- `poincare_alpha_trend.png` e2313096-3a21-453b-a2ec-a9455693957c.

Open items on NCSX:
- run test 2 at snapC;
- check whether the production "interior" steps (|y| 0.03–0.2, ~40 Lanczos, at states with |grad| 1.2–3.7) are
  false stops caused by a stiff-mode-dominated |g| (**hypothesis**);
- a radius sweep between 9 and 19;
- a richer-p re-evaluation of existing states (`--eval-only`);
- constraint tests C1–C7.

---

## 6. Next session: DSHAPE with an explicit boundary perturbation

**Goal (user).** A nearly axisymmetric case that should have islands, set up so that the **initial** NEST current
gives small or no islands. Then we can see whether the optimizer opens them by itself. This is cleaner than NCSX,
where the start already had coil/boundary-mismatch islands and the solver closed them.

### 6.1 What exists from earlier sessions (find it first)
Use `search_memory` (queries: "DSHAPE", "tokamak island", "B_ext consistent", "virtual casing") and
`host.artifacts(search='dshape')`, `host.artifacts(search='tokamak_island')`.

- **Collaborator's 2/1 study (coil-side perturbation, not boundary):** hand-off artifact
  `dshape_perturbed_handoff.md` (version db3b16cd-0740-4695-acc5-d3792849a290); branch
  `origin/dshape-perturbed-wJ1e-7`.
  - DESC free-boundary tokamak `desc_input/tokamak_free_boundary/tokamak_L32M32_fb.h5` on the DSHAPE boundary,
    with modified profiles: p = 1.6 kPa (1 - rho^2)^3, I0 = -120 kA, q from 1.18 to 7.3, q = 2 at rho 0.5797,
    iota' 0.988.
  - Perturbation B_ext = B_coils + eps·B_pert with Phi = sin(2θ + ζ) on the winding surface; DESC is never
    re-solved. Vacuum b_21 is 1.65e-4 per 100 A.
  - Findings:
    - leftover b_21 (NEST relative to vacuum) is not monotonic in eps;
    - pressure is not constant inside the islands;
    - enforcing B·grad p = 0 harder always drives the state to the nested, fully shielded solution.
  - Scripts: `scripts/dshape/isl_*.py` (scan, islands/b_mn/Poincaré, pressure maps); model in `HarmonicModel`
    (`nest/harmonic_dense.py`, on that branch/main).
- **Smooth-edge DSHAPE** (commit b463dbe on another branch; not in this worktree's `desc_input/`):
  - files `DSHAPE_free_boundary_smooth{,_M20}.h5` with TF and PF coils. Artifacts:
    `dshape_smooth_final_M20.h5` e2b1cc78-f518-4d84-b623-fd279f8e76f5 and `dshape_smooth_coils.h5`
    862182d2-2022-41fc-a9a4-7a496b6dd64c;
  - |iota| 1.0026 → 0.308, iota = 1/2 at rho 0.866 with strong shear;
  - exactly axisymmetric external field; force error 9.2e-10 at M20.
- **Tearing-stable candidate C:** `tokamak_island_C_L32M32.h5` 6fcfbcc1-a716-4f43-86ec-f3b5171fd888 (+ `_field.h5`
  7e495f5e-fbdc-4fe8-b989-281d0002643f, diagnostics 4525a342-f666-4624-81b6-5c190564d8f3),
  `tearing_stable_candidates.json` 1a9197d0-ffb8-4feb-9eeb-41d7a10ffbb6.
  - It was chosen for moderate shear at q = 2 (0.41).
  - A tearing-stable equilibrium (Δ' < 0) matters: islands should come from the boundary drive, not from a
    spontaneous instability.
- **Validation plan:** `NEST_validation_test_plan.md` a9677d01-2b12-424d-a108-53a782cf36b1. Test (4) "resonant
  boundary perturbations vs the pendulum estimate" is exactly this study. The island-width estimate is
  w = 4 sqrt(|b|/(m|iota'|)).
- **No 3-D boundary-perturbed DESC DSHAPE seems to exist yet.** The next agent probably has to build one (§6.3).

### 6.2 Choice of B_ext so that the start has (almost) no islands
Three options:
1. **Consistent, NEST-discretized:** B_ext = B_DESC - B_plasma[J_fit] at NEST's own points. With u = 0
   (J = J_fit), B = B_DESC at the collocation points by construction, so the start has DESC's nested surfaces up
   to round-off.
   - Implemented in the DSHAPE scripts as `--bext desc` (`scripts/dshape/gs_dense_solve.py`; "B_DESC -
     B_plasma(x0)").
   - **Not available in `mf_fixed.py`, which only takes `--coils`.** Port it: set `fr.B_ext` before l.238, where
     the driver already skips `inp0.B_ext` if `fr.B_ext` is set.
   - Tracing (Poincaré) needs B_ext off the collocation points: evaluate B_DESC(x) - B_plasma[J_fit](x) there, or
     interpolate.
   - Caveat: circular by design. The start is exactly NEST's representation of DESC, which is what this test
     wants.
2. **Virtual casing (independent):** B_ext = BS[K = -n x B_DESC / mu0] on the LCFS. This needs B.n = 0, i.e. a
   fixed-boundary DESC equilibrium, which a boundary-perturbed DSHAPE is. Start islands are then set by NEST's
   discretization error (on NCSX the start islands were expected to drop to w <= 0.015 at a/20 this way;
   **hypothesis**, not run).
3. **Real coils plus a free-boundary DESC solve:** coil/boundary mismatch produces start islands. This is what
   NCSX had; avoid it here.

**Recommendation:** option 1 for the first test, then option 2 as the independent check. If the two disagree, the
discretization is not converged (the user's standing rule: results that change between consistent-field
formulations are not trustworthy).

### 6.3 Suggested plan
1. **Build the equilibrium in DESC (fixed boundary).**
   - Start from the DSHAPE-boundary tokamak with the candidate-C (tearing-stable) profiles, or the
     `tokamak_L32M32` profiles. Keep q = 2 in the mid-radius at moderate shear.
   - Add a resonant boundary mode (m, n) = (2, 1) with NFP = 1. Keep stellarator symmetry: R cos-type, Z sin-type
     with DESC's (m, n) sign conventions.
   - Solve at a few amplitudes (e.g. boundary displacement 1e-3 to 1e-2 a) and resolutions; N = 2–3 is cheap with
     NFP = 1.
   - DESC's nested answer must carry a shielding current at q = 2. Record DESC's force error there as a function of
     resolution: it is the drive NEST could relieve by reconnecting.
2. **Vacuum reference.** Compute the vacuum resonant b_21 and the island width for each amplitude.
3. **Start check in NEST** with option-1 B_ext: Poincaré at u = 0 should show no islands; b_21 on q = 2 should be at
   round-off.
4. **Choose the objective deliberately (§4, first pitfall).**
   - The fixed-boundary misfit to J_fit pulls toward DESC's shielded current and keeps islands closed. On NCSX the
     islands closed as alpha fell.
   - Alternatives to discuss with the user:
     - misfit only to the axisymmetric part of J_fit, or to the profiles (I(s), p(s)) rather than pointwise J;
     - a force-only solve at fixed I (pure feasibility);
     - an alpha scan to see whether islands open as the bound loosens or tightens.
5. **Measure island opening.**
   - Resonant b_21 on the measured (not nominal) q = 2 surface: q = 2 moves with the response in the collaborator's
     runs.
   - Traced island width against the vacuum and pendulum estimates.
   - Pressure along field lines inside the island.
   - Force maps by radius.
   - Run the same diagnostics on DESC's own field as a control.
6. **Size before running.**
   - NFP = 1 with small N should make each run far cheaper than NCSX 48/32/18.
   - Do the 8/8/N smoke test locally (ask first), then an A100/H200 run.
   - Stream progress and checkpoint every step.
