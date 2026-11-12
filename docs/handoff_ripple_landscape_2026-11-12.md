# Hand-off: NEST ripple landscape study (shielding vs islands at q = 2), 2026-11-09 to 11-12

Written at the end of the session for an agent that has not seen the conversation. Every number here was measured
in this project and can be traced to a file named next to it. Items marked **hypothesis** or **not tested** were not
measured.

Read in this order:

1. this file;
2. `docs/ripple_landscape_report.md`: self-contained report of the first half (setup, band-max runs, solver noise,
   plane maps, β scan, island diagnostics, release tests), with a glossary;
3. `results/ripple_landscape/README.md` (state manifest) and `results/ripple_landscape/resolution/README.md`
   (resolution rungs and the grid-convergence test);
4. `UPDATES.md`: dated log, newest first;
5. `docs/handoff_fixed_boundary_2026-10-30.md`: the earlier fixed-boundary hand-off. Its code map, solver pitfalls and
   Modal notes still apply.

---

## 0. Status in one paragraph

NEST optimizations started from DESC's shielded equilibrium never open the q = 2 island. To find out why, we built a
constraint that pins the resonant field b21 at β × its vacuum value and lets everything else relax, then used it to
map the landscape between shielded (β = 0) and vacuum-island (β = 1) states:

- **Islands are real but cost force at the first resolution.** At 32/32/6 they are genuine (traced width = pendulum
  width, pressure flattened along field lines). The force rises monotonically with β, there is no barrier, and
  released islands close.
- **The force gap shrinks with resolution.** From 32/32/6 to 40/40/6 it falls about 4× (0.081 → 0.015–0.019 M_DESC on
  a common reference grid). At 40/40/6 it is stable under field-grid and collocation refinement. Model C (one valley
  in which shielded and reconnected states merge as resolution grows) is the working hypothesis; it has not been
  tested beyond two rungs.
- **The island's extra force sits at q = 3, not q = 2.** About 70% of it is around q = 3. The island state also
  changes ι(s) by up to 2% globally and displaces the core helically.
- **Force and energy disagree.** Island states have *lower* magnetic energy (−126 J of 1.93 MJ) but higher force
  error and higher least-squares thermal energy. Helicity changes by a fraction of 3e-3, at the level of a gauge
  ambiguity (B·n ≠ 0 on the LCFS).
- **Next.** The natural continuation is a comparison with MRX, an ideal (topology-preserving) energy minimizer with
  island seeding (Section 6).

---

## 1. Where things are

| what | where |
|---|---|
| repository / branch | `PlasmaControl/NEST`, branch **`ripple-landscape`** (from `fixed-boundary` at 8e09c03), worktree `/Users/singh/PycharmProjects/NEST/NEST-cp` |
| DESC checkout | `/Users/singh/PycharmProjects/DESC` |
| test equilibrium (DESC) | `desc_input/tokamak_boundary_perturbed/bp2/L32N4/tokamak_ripple_C_L32M32N4_x0.001.h5`. Every NEST state in this study uses this file. L24, L32 and L48 variants and ξ/a = 1e-4 to 3e-3 are alongside it; L48 is in `bp2/L48N4/` |
| states, logs, diagnostics | `results/ripple_landscape/{states,runs,diagnostics}` (first half), `resolution/` (R0, R1, R1b/c/f), `compare/` (β = 0 vs 1 traces, energy and helicity). `results/` is gitignored: use `git add -f` |
| report and figures | `docs/ripple_landscape_report.md`, `figures/ripple_landscape/` (and `compare/`) |
| Modal job scripts | `scripts/modal_fixed/ripple/` (`job_*.sh`, `rung_run.sh`, `gpu_entry.py`) |
| Modal image / volume | `im-85tKPZB30SmMyYst74flDg` / Volume `nest-container-mask` at `/results`. Staged inputs are in `/results/stage` (`desc_src.tar.gz`, `deps_src.tar.gz`, `ripple_L32N4_x1e-3.h5`, `gpu_entry.py`); run outputs are in `/results/fxr/...` |
| big tables (not in git) | `/results/fxr/res/tables` on the Volume (275 MB float32 flux-grid tables for R1f β = 0 / β = 1, DESC and vacuum) |

**Git.** Commit with `git -c user.name="Jay Singh" -c user.email="jaydeepsingh15@gmail.com" commit ...`. The sandbox
cannot reach github; the user pushes. The sandbox also cannot create worktrees (`.git` writes are blocked except
refs).

**The agent workspace is wiped when idle.** Anything in `modal_rung/`, `hpc/` or `cmp/` there is temporary. The repo
copies above are authoritative. Rebuild the source tarball for a job with:

```
tar --exclude=__pycache__ --exclude='*.pyc' --exclude='results' -czf modal_rung/nest_src.tar.gz \
    -C /Users/singh/PycharmProjects/NEST/NEST-cp nest scripts tests
```

Copy the job script and `scripts/modal_fixed/ripple/rung_run.sh` next to it.

---

## 2. The question

**Physics.** Candidate C is a DSHAPE-like tokamak:

- a = 1.198 m, R0 = 3.509 m, ⟨β⟩ 0.021, NFP 1;
- hollow p with dp/ds = 0 at q = 2;
- q = 2 at ρ = 0.6375 (ι = −0.5, |ι′| = 0.234), q = 3 at ρ = 0.913. q = 1.5, 2.5 and 4 are at 0.225, 0.845 and 0.9875;
- (2,1) tearing-stable; (3,1) has Δ′ > 0; Mercier-unstable for ρ ≈ 0.45–0.62.

It carries a resonant (2,1) boundary ripple with ξ/a = 1e-3. DESC assumes nested surfaces, so its answer is perfect
shielding (a current sheet at q = 2). The vacuum resonant field is |b21|_vac = 3.778e-4, which implies a pendulum
width of 0.1136 in ρ. **Does NEST, which does not assume nested surfaces, open the island and flatten p, or keep
DESC's shielding? And why?**

**NEST setup** (details in report Section 1):

- Unknowns: current-potential coefficients u, with J linear in u at fixed net current I.
- Field: B = B_ext + B_ann[J], where B_ann is the FFT Biot–Savart on a cylindrical grid of spacing a/frac.
- Pressure: eliminated by least squares.
- Force: F = J×B − ∇p*.
- Objective: minimize |J − J_fit|²/rms² subject to the volume-mean force M = ⟨|F|⟩ / norm ≤ α M_DESC, with
  M_DESC = 4.9995e-7 at every rung (same DESC file).
- Option A: B_ext = B_DESC − B_ann[J_fit], so the start is exactly DESC's shielded field.

**Landscape models** (the user's framing):

- **A:** a flat valley connects shielded and reconnected states, but they are far apart.
- **B:** separate basins with no low-force path between them.
- **C:** shielded and reconnected states are close in a flat valley; finite resolution makes one win, and they merge
  at infinite resolution.

The user started with A, moved to B mid-study, and the data now support a weak C (Section 4).

---

## 3. What was done (chronological; numbers in M / M_DESC unless stated)

1. **Shielded start** (`states/shielded_start_A60_a0.1.npz`; 32/32/6, a/60, N_φ 25). This is an α continuation
   0.3 → 0.1 that stalled at M = 0.1243 (bound not met). It has |b21| 1.24% of vacuum in the *anti*-island phase
   (−90°), i.e. it is over-shielded. Its residual force sits near q = 3 and the edge.
2. **Band-max experiment Q2b.** We bounded the q = 2 band p16 at α × the start's, with the whole-volume mean ≤ the
   start's. The band reached 0.103 × the start's (0.032 × DESC's band p16), but the mean rose to 1.104 ×. The force
   moved to ρ 0.40–0.57 and ρ > 0.95. **No island formed.**
3. **Stall anatomy.** The stall was solver-limited, not a floor. The merit has deterministic roughness of 2–4e-8
   relative (J×B − ∇p cancellation), so trust-region ratios at small predicted decrease are noise, the radius
   collapses, and the solver locks. Fixes are in `nest/meanforce.py`: `--f-noise auto`, `--radius-by-delta`,
   `--no-share-stop`, and (tested, no effect) `--geo-accel`. A push-through from the stall then descended steadily
   (M/ε 1.1242 → 1.1194 in 25 min), along the valley, still with no island.
4. **Exact plane maps** (force is quadratic in u). The shielded start is the minimum on every plane. A *straight* move
   to 18% of vacuum b21 costs 119 M_DESC.
5. **b21 pin and β scan** (`--pin-b21 β`; two affine equality rows; the functional agrees with an independent tracer
   to 1.2%). With 30 min per β, M rose from 0.124 (start) to 0.126 / 0.132 / 0.150 / 0.180 / 0.211 / 0.214 / 0.244 /
   0.247 at β 0.02 / 0.05 / 0.1 / 0.2 / 0.35 / 0.5 / 0.75 / 1. Relaxation removed 97–99.8% of the straight-move cost.
6. **Are they islands?** Yes. At β = 1 the traced width is 0.107 against a pendulum width of 0.113. The along-line p
   spread is 0.01–0.07 of DESC's drop, i.e. p is flattened. The resonant current moves off q = 2 to *both* flanks
   (outer peak at ρ ≈ 0.72; larger inner peak at ρ ≈ 0.48).
7. **Release tests** (pin removed). β = 0.05 → b21 0.83% at M 0.1228, below the start. β = 0.5 → 9.6% at 0.151,
   still closing. Conclusion at 32/32/6: one valley, islands on a monotone wall.
8. **Resolution rungs** (`resolution/`). Fresh multipliers, ≤ 400 steps, same starts:

   | state | R0: 32/32/6 a/60 (eval a/80) | R1: 40/40/6 a/80 |
   |---|---|---|
   | shielded | 0.1235 (0.1225) | **0.100** (α = 0.1 bound met) |
   | β = 1 | 0.215 (0.203) | **0.100** (α = 0.1 bound met) |

9. **Push toward α = 0.05 at R1, and the grid-convergence test.** Reference evaluation on a/120 with 120 radii:

   | variant (solve grid) | shielded | β = 1 | gap |
   |---|---|---|---|
   | a/80, 55 radii | 0.0755 | 0.0909 | 0.0154 |
   | a/80, 98 radii | 0.0745 | 0.0900 | 0.0155 |
   | a/100, 98 radii | 0.0637 | 0.0822 | 0.0185 |

   Radial collocation does not matter (about 1%). The field grid lowers both floors but leaves the gap unchanged.
10. **β = 0 vs β = 1 characterization** (R1f states; `state_tables.py` on GPU, then `compare_states.py` on CPU: 190
    field lines × 400 turns, field-line averages, ρ_eff = √⟨ρ²⟩ as the flux label). Figures are in
    `figures/ripple_landscape/compare/`.
    - Extra force of β = 1 by band: +0.0116 at ρ 0.86–0.94 (q = 3), +0.0000 in the q = 2 band, +0.0018 in the core.
    - q = 3 stays shielded in both states: b31 3.9e-7 against 5.8e-7, vacuum 1.64e-4.
    - The β = 1 state has a global ι shift (up to ±1.6e-2), a helical core displacement of about 0.006 in ρ, and
      ⟨p⟩ that is lower in the core and higher at s 0.5–0.8.
    - In *both* states the parallel force |b·∇p| is the larger part of the residual (0.044 of 0.066 for shielded).
11. **Energy and helicity** (`energy_helicity.py`; plasma volume; DESC's W_B = 1.927469 MJ, matching DESC's own
    compute to 1e-6). For R1f, β = 1 minus shielded:
    - ΔW_B = −126.5 J (−6.6e-5 relative);
    - ΔW_th = +161 J, which is NEST's least-squares p and not a physical constraint;
    - ΔK/K = −3.1e-3, tracking ΔΦ_pol/Φ_pol = −1.2e-3; Φ_tor is fixed to 2e-5.

    On the R0 β scan, W_B falls monotonically from β ≈ 0.2 to 1. B·n at the LCFS is 1–6e-4, so K is gauge-ambiguous
    at about that level.
12. **MRX discussion.** MRX (Blickhan, Stratton and Kaptanoglu, arXiv 2510.26986) relaxes ideally, i.e. it preserves
    topology, and seeds islands with an energy criterion (amplitude a*). Our NEST island states have nearly DESC's
    helicity but changed ι(s) and topology. So helicity is not what blocks islands; frozen-in flux is.

---

## 4. Conclusions and how sure we are

| claim | confidence | evidence / caveat |
|---|---|---|
| Model B (separate basins) is false at 32/32/6 and 40/40/6 | high | continuous pinned path, no barrier; released islands close at both rungs |
| Island states with flattened p are legitimate NEST states | high | traced width = pendulum width; along-line p ratio 0.005–0.07 |
| The island–shielding force gap shrinks with resolution | medium | 0.081 → 0.015–0.019 on reference grids; only **two rungs**; R0 measured at α = 0.1, R1 at the α → 0.05 push (different force levels) |
| The R1 gap is real (not grid exploitation) | medium-high | stable across a/80 / a/100 and 55 / 98 radii on one reference grid; β = 1 was still falling faster at 80 steps |
| The extra force of islands sits at q = 3 | medium | one state pair at R1, field-line averages; not yet checked at R0 or R2 |
| Islands lower magnetic energy | medium | consistent across rungs and grids, but likely dominated by the global current-profile (n = 0) change, not by the island; not decomposed |
| Helicity is roughly conserved between states | low | changes of 3e-3 are at the B·n gauge-ambiguity level |
| At infinite resolution the choice is made by the objective, which favours shielding by construction | **hypothesis** | J_fit is DESC's shielded current; untested |

---

## 5. Open questions and recommended next experiments (priority order)

1. **Decompose ΔW_B into n = 0 and helical parts.** This tells whether the island itself lowers the energy or the
   global ι relaxation does. It is cheap: extend `energy_helicity.py` to also integrate |B̄|² (toroidal average at
   fixed DESC (ρ, θ)) and |B − B̄|².
2. **MRX comparison** (user's proposal):
   1. DESC L32 ripple → MRX with no seed. MRX should hold DESC's ideal state; any b21 that appears is grid
      reconnection.
   2. Seed scan a/a* ∈ {0.1, 0.3, 1, 2, 3} plus an anti-phase control.
   3. For each state record b21 → β, traced width, W_B, W_total, K, ι(s), b31 at q = 3.
   4. Compare W(β) from MRX (ideal, fixed topology) with NEST's W_B(β) (fixed I, ι free).
   5. Cross-evaluate: NEST force of MRX states, MRX energy of NEST states.
   6. MRX's Section 7 (prescribed-current resistive relaxation) is the closest analogue to NEST's objective.

   Settle first how MRX imports DESC and how it evolves p at finite β.
3. **R2** (48/48/6) to test the gap trend. Solve at a/120 (h·L/a = 0.4) or finer, with default collocation and a
   reference evaluation one grid step finer (a/150). Use matched step counts above 80. This needs an 80 GB GPU (H200
   for a/150).
4. **Objective-bias test.** Anchor the objective only to the axisymmetric (or flux-surface-averaged) part of J_fit, so
   the shielding current is not rewarded. Then repeat release tests from β = 0.5.
5. **Clean helicity.** Enforce B·n = 0 on the LCFS (NEST has `--pin-flux`, `--pin-current`; a B·n row would be new),
   or use a relative helicity with matched boundary flux.
6. **Solver.** In mean-bound runs the penalty κ escalates to 1e7–1e8 with erratic least-squares multiplier swings.
   Fresh multipliers (ν = 0, κ = 1) behaved much better at R1. Consider capping κ or resetting per stage.

---

## 6. How to run experiments

### Local CPU
Only for small things (tracing and plotting from tables, parsing logs). The laptop has 8 GB and is shared; ask
before long local jobs. `compare_states.py` runs locally in about 40 s per field (190 lines × 400 turns).

### Modal GPU (what worked in this study)
- **Job script pattern.** Copy `scripts/modal_fixed/ripple/job_res.sh`. It unpacks `nest_src.tar.gz`, takes the DESC
  and deps tarballs from `/results/stage`, runs each step through `rung_run.sh TAG script args`, and streams the log
  to `/work/out/TAG.log` and `/results/fxr/TAG/`. `mf_fixed.py --mirror DIR` fsyncs states and JSON there, and
  `--snap-outer` saves a state after every outer iteration. Write a `progress.txt` line per run.
- **Submit** (repl, `background: true`):

  ```python
  pp = {'image': 'im-85tKPZB30SmMyYst74flDg', 'gpu': 'A100-40GB', 'cpu': 8, 'memory': 65536,
        'volumes': {'/results': 'nest-container-mask'}, 'timeout': 11400}     # hard cutoff in s
  c = host.compute.create('byoc:modal', provider_params=pp)
  j = c.submit_job(intent='...', command='bash job_res.sh R0 32 32 6 60 80 1980',
                   inputs=['modal_rung/job_res.sh', 'modal_rung/rung_run.sh', 'modal_rung/nest_src.tar.gz'],
                   run_timeout_s=11300)
  ```

- **Monitor.** From `compute_provider` (modal), `modal.Sandbox.list()` then `sb.exec("bash", "-c", "tail ...")` on
  `/work/out/*.log` and `/results/fxr/res/<rung>/progress.txt`. `vol.read_file` fails through the proxy; read Volume
  files with a small CPU sandbox (`modal.Sandbox.create(app=modal.App.lookup('nest-probe', create_if_missing=True),
  image=modal.Image.from_id(IMAGE), volumes={'/results': vol}, cpu=1, memory=2048, timeout=600)`), and terminate it.
- **Harvest.** Outputs land in the workspace under `hpc/<job_id>/out/`. A job can be "orphaned" (sandbox gone, nothing
  harvested), as d0de187d was. Its mirrored files on the Volume survive.
- **Sizes and timings measured here:**

  | case | n_u | annulus nodes | GPU memory | setup | per inner step |
  |---|---|---|---|---|---|
  | 32/32/6 a/60 | 7,726 | 1.04 M | ~10 GB JAX | ~2.5 min | 0.7 s median, 3.4 s mean (boundary steps ~10–40 s) |
  | 40/40/6 a/80 | 11,734 | 1.81 M | 21 GB JAX (A100-40GB fits) | ~7.5 min | 3–9 s; ~14–19 s on average at the α 0.05 push |
  | 40/40/6 a/100 | 11,734 | ~2.8 M | needs 80 GB (plasma-locate alone allocates 39.5 GB) | ~8 min | ~14 s on average |
  | 48/48/6 a/120 (eval) | 16,574 | 4.0 M | 33 GB peak | ~3.5 min | — |

  An a/100 *evaluation* at 40/40/6 runs out of memory on an A100-40GB, because `locate_plasma_planes` allocates 39.5
  GB. Use 80 GB.
- **Budget rules the user set:**
  - Default to A100 (40 GB where it fits); H100 is more expensive.
  - Use iteration-based budgets with hard container cutoffs and as few containers as possible.
  - State the GPU-hour total before dispatch.
  - Do not cancel a job merely for sitting in "scheduling" for 10–25 minutes (A100s placed after 12–19 min here); fall
    back to H100 only if it persists.
  - Every long run must stream per-iteration progress.
- **Warm starts.** `--start file.npz` loads u, ν and κ. Strip ν and κ for a fresh augmented Lagrangian
  (`np.savez(out, u=d['u'], c_p=d['c_p'])`). `--start-LMN 32 32 6` embeds a lower-basis state, and
  `--b21-phase-state` is embedded too (patched in 769e63c).
- **Evaluation.** `mf_fixed.py --eval-only --force-norm p --p-schedule 16 --p-band 0.5675 0.7075 --b21-report
  --eval-state NAME=path` (needs `--force-norm p`). Use `--colloc-rho X` to change the radial collocation: radii =
  X (L + L_eq)/2 + 1.

---

## 7. Standards to hold to (healthy skepticism, with the cases that taught it)

**When a result looks too good, assume the solver found a loophole until shown otherwise.**

1. **Grid exploitation.**
   - The 48/48/6 states "reached α = 0.1 cleanly" at a/60 (M 0.102), but evaluate to 0.142 / 0.171 / 0.165 at a/80 /
     a/100 / a/120. The optimizer had pushed into the field-grid error.
   - The R1 smoke test fell from 0.59 to 0.113 in 6 steps, which looked like a breakthrough. It needed an R0 control
     with identical (fresh) multipliers and a finer-grid evaluation before it meant anything.

   Rule: evaluate every end state on a finer field grid **and** compare states on one common reference grid (here
   a/120, 120 radii). h·L/a ≲ 0.5 was calibrated at α = 0.1; at tighter bounds the solver exploits more (the shielded
   state rose 18% from a/80 to a/100 at the α → 0.05 push).
2. **Single-state grid shifts are not trends.** One state evaluated on a/80 / a/100 / a/120 gave 0.0705 / 0.0831 /
   0.0755, which is non-monotone. Compare *differences between states* on the same grid.
3. **Stalls are not floors.** Q2b's "floor" was trust-region collapse on merit noise; a push-through kept descending.
   Before claiming a floor, test whether the solver is healthy (ratio statistics, radius history, σ_f).
4. **Straight paths overestimate barriers.** A straight move cost 119 M_DESC; the relaxed island cost 0.12. Never
   infer a barrier from lines or planes; use constrained relaxation (the pin).
5. **Equal effort, or upper bounds.**
   - Budget-limited runs give upper bounds, so compare at matched step counts.
   - Last night's follow-up gave shielded 56 steps against β = 1's 177 steps; the comparison had to be redone at step
     56.
   - The "converged" shielded start was beaten by a release run (0.1228 < 0.1243).
6. **Hold settings identical across a comparison.** Fresh against carried multipliers (κ = 1 vs 1e8) changes results
   as much as resolution does. Re-run the baseline rung with exactly the new settings.
7. **Check classifications with pictures.** "Island-trapped" lines flagged near q = 1.5 turned out to be a helical
   core displacement (uniform radial spread 0.0115, ι passing smoothly through −2/3). Look at the Poincaré section
   before claiming a new island.
8. **Verify a new diagnostic against an independent one.**
   - The b21 pin functional was checked against the trace-based diagnostic (1.2%) before use.
   - The energy quadrature was checked against DESC's own `compute("W_B")` (1e-6).
   - The ∇·B residual of the reconstructed A was reported (1e-7).
9. **Know what a quantity can mean.**
   - NEST's pressure is a least-squares fit, so its thermal energy is not a physical constraint.
   - Helicity is gauge-ambiguous when B·n ≠ 0 on the boundary; report the ambiguity next to the number.
   - The objective anchors to DESC's shielded current, so it is biased toward shielding.
   - Option A puts the unfitted current into B_ext, so the start cannot show islands by construction.
10. **Units and labels.**
    - Quote forces as M / M_DESC. M/ε (ε = α M_DESC) caused a misreading.
    - State the solve grid and the evaluation grid with every number.
    - Report max-type targets as explicit tables of fractions of DESC's values on the same points.
11. **Never type numbers from memory into a report.** Two errors in this session's deliverables were caught that way:
    - a traced spread of 0.005–0.007 that should have been 0.0045;
    - a README saying the 48/48/6 state used the L48 DESC file when it used L32.

    Read values back from the JSON or logs.
12. **Separate measured from inferred.** Say "upper bound", "two rungs only", "hypothesis" in the text where the claim
    is made, not only in a caveats section.

**What the user expects in working style:**
- Answer the question asked, and never refer to a table that was not shown.
- Report key values and links.
- Reports are self-contained for a collaborator unfamiliar with the branch, at early-graduate level, with every
  quantity defined, the grids stated, a data manifest and a reproduction table.
- For conditioning or scaling problems, look at how DESC does it first.
- Ask before large or open-ended compute.
- Commit with a dated `UPDATES.md` entry; the user pushes.

---

## 8. Inventory (where to find each state)

| family | files | basis / grid | load with |
|---|---|---|---|
| shielded start, β scan, release, Q2b, seeds | `results/ripple_landscape/states/` (manifest in its README) | 32/32/6, solved a/60 | `--LMN 32 32 6 --frac 60` |
| R0 re-run (S, β 0.2 / 0.5 / 1, release) | `results/ripple_landscape/resolution/states/R0_*` | 32/32/6 a/60 | same |
| R1 (α 0.1) | `resolution/states/R1_*` | 40/40/6 a/80 | `--LMN 40 40 6 --frac 80` |
| R1b / R1c / R1f (α → 0.05) | `resolution/states/R1b_*`, `R1c_*`, `R1f_*` | 40/40/6; a/80 / a/80 (98 radii) / a/100 (98 radii) | `--frac 80` or `100`, `--colloc-rho 2.7` |
| 48/48/6 | `states/A48_cont_a0.1.npz` | 48/48/6 a/60, **not grid converged** | `--LMN 48 48 6` |

New scripts in this study: `scripts/meanforce/{plane_map.py, island_seed.py, start_check_bext.py, state_tables.py,
compare_states.py, energy_helicity.py}` and the new `mf_fixed.py` flags (`--pin-b21`, `--b21-*`, `--plane`,
`--f-noise`, `--radius-by-delta`, `--no-share-stop`, `--geo-accel`). The test suite was not re-run; the new code
paths are opt-in flags.
