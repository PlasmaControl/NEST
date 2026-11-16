# Ripple landscape with B-splines: findings

*2026-11-12/13. Branch `ripple-spline` (worktree `/Users/lee/codes/NEST_rspl`, from `origin/ripple-landscape`), not
committed. Runs on Stellar-AI in `/scratch/gpfs/EKOLEMEN/lk1158/NEST_rspl/out/`. The report with figures is
`docs/ripple_spline_report.md`; the run log and the decisions taken along the way are in `docs/ripple_spline_plan.md`.
Data are in `results/ripple_spline/` and figures in `figures/ripple_spline/`.*

M is the volume-mean force |F| in units of M_DESC (DESC's own value on the same grid). "Parallel" is the volume mean
of |b·F| = |b·∇p|; "perpendicular" is the volume mean of the rest. β is the pinned resonant field b21 as a fraction of
its vacuum value, in the island phase.

## 1. The question

`docs/ripple_landscape_report.md` (Fourier-Zernike, FZ 32/32/6, a/60) found three things:

- shielding at q = 2 is the single valley floor;
- islands with flattened pressure sit on a monotone wall above it;
- about half the force error is b·∇p.

We asked whether a B-spline radial representation of the current potential T and the pressure p, refined at the 2/1
island, changes this landscape and lowers b·∇p. The experiments are the report's: the shielded start, the β scan with
b21 pinned, the release tests, and the island and pressure diagnostics.

## 2. Findings in brief

1. **Pure cubic B-splines cannot reach FZ's force level at practical sizes.** They stall at 0.38–0.50 M_DESC against
   0.098 for FZ, with the excess in the smooth core.
2. **A hybrid basis works.** It is FZ 32/32/6 plus B-splines confined to the q = 2 and q = 3 bands. At equal effort it
   roughly halves M and halves b·∇p at every island size, compared with an FZ control on the same grid.
3. **b·∇p is set by J, not by p.** Extra pressure freedom at fixed J leaves the parallel force unchanged. It falls
   only when T and p are refined together.
4. **The bottom of the landscape changes with the hybrid.** It is flat from the over-shielded state to small islands
   in the island phase (b21 up to about 5–10% of vacuum). Large islands still cost force and still close when released.
5. **The field grid alone matters.** The same FZ basis optimized at a/100 instead of a/60 is 20–35% lower at every β.
6. **Two solver and setup problems were found and fixed.** The Lanczos cap caused costly metric rebuilds at large
   n_u, and spline runs had M_DESC computed on a different radial grid from the run.

## 3. Setup and verification

- **Branch setup.** The `ripple-landscape` branch was cloned on Stellar-AI (`NEST_rl`). All 17 DESC files in
  `desc_input/tokamak_boundary_perturbed/` load, with a = 1.198 m, R0 = 3.509 m, ι = −0.5 at ρ = 0.6375 and −1/3 at
  0.9131. An eval-only run reproduced the collaborator's numbers for the shielded start and the β = 0.05, 0.5, 1.0 and
  release states: M 0.1243, 0.1314, 0.2129, 0.2456 and 0.1228, with b21 1.25%, 5%, 50%, 100% and 0.83%.
- **Ported spline code.** The spline code from `dshape-fixed` (uncommitted there; that branch was left untouched) was
  ported by patch, and the three overlapping files were merged by hand.
- **Same boundary conditions.** Only the radial functions change. The `BoundaryPotential` structure (window
  (1 − ρ²)², χ terms, h₀, net current I) and p = (1 − ρ²)³ Q are kept, so div J = 0, J = O(1 − ρ), I fixed and p = 0
  on the edge hold for any coefficients. The gates check this with random coefficients.

## 4. Pure B-splines

### 4.1 Axis regularity

The existing rule dropped the three splines that touch the axis for m ≠ 0. It cannot represent the m = 1 ~ ρ term of a
shifted axis (fit error 5e-2), and odd m are not regular.

The new option `--spl-reg rhom` uses ρ^{q_m} B_j(ρ²), scaled to unit maximum, with q_m = |m| capped at 6 (C⁴ at the
axis). The four splines that touch the axis are kept for |m| ≤ 6. The m = 1 axis shift is then exact (1e-16), and the
θ-spread of J and ∇p falls linearly with ρ (regular).

With a Fourier basis in θ, regularity decouples per m. A cos/sin compensation between harmonics is only needed when θ
is also represented by splines (polar splines).

### 4.2 Knots and truncation

- **Knots.** Outside the bands the spacing follows FZ's zero spacing (`--knots-fz 32 C`, dense at the edge like FZ).
  Two bands are added with `--spl-band LO HI H`: q = 2 (ρ 0.45–0.76, which covers the β = 1 island and both displaced
  λ21 peaks) and q = 3 (ρ 0.86–0.95).
- **Truncation.** Dropping splines with `m_tol 1e-2` and `m_max 16` costs 10–100× in the fit of DESC's J at
  ρ 0.4–0.72, so high-m content matters there. The FZ-reach rule `--spl-mfz 32` costs nothing.
- **Unknown counts.** Matching FZ's accuracy needs about 3× FZ's unknowns: a + b of 19–31k against 7.3k for FZ
  32/32/6. A count-matched comparison is therefore not possible at FZ accuracy.

### 4.3 Why pure splines fail

The shielded-start continuations stalled:

| config | knots | n_u | M at the stall |
|---|---|---|---|
| s1 | 0.6× FZ spacing, bands 0.015 / 0.02 | 23.7k | 0.50 |
| s2 | 0.45× FZ spacing, bands 0.01 / 0.015 | 31.2k | 0.38 |
| FZ control (same grid) | — | 7.7k | 0.098 |

In the stalled s1 state, the perpendicular force per volume at ρ < 0.5 is 0.6–1.1 M_DESC against 0.04 for FZ. Inside
the island band it is only about 4× FZ.

The core error is already present at J_fit: the perpendicular force there is 10–12 M_DESC with s1's knots, and 75–88
with knots at FZ's spacing. Yet the rms J fit in the core is better than FZ's (7e-6 against 2.5e-5).

The force at M ~ 0.1 M_DESC resolves J to about 1e-7 of |J|. A cubic-spline fit leaves an error that oscillates on
the knot scale, and its curl, which no pressure can balance, is large. FZ's spectral error is smooth. Neither the
truncation nor the axis cap was the cause: no truncation and q_cap = 10 give the same J_fit force.

## 5. The hybrid basis (the solution)

`--current fz+spline --pressure fz+spline` (`nest/spline_basis.CompositeBasis`):

- **Modes.** FZ 32/32/6 plus cubic B-splines in s = ρ² whose support lies inside a band: q = 2 at 0.45–0.76 with
  spacing 0.01, and q = 3 at 0.86–0.95 with spacing 0.015. The splines have |m| ≤ 16 and all n ≤ 6.
- **What FZ keeps.** Each spline and its first two derivatives vanish at the band edges. The FZ part therefore keeps
  its spectral accuracy, axis regularity and edge refinement, and the boundary conditions are FZ's.
- **Fits.** J_fit and the p fit are FZ's exactly (spline coefficients start at 0), so FZ states embed with no change.
  Any difference therefore comes from the band refinement alone.
- **Sizes.** n_u 21056 (FZ 7726); n_p 10320 (FZ 3655).
- **Gates** (`tests/test_hybrid_gates.py`). J_fit and p equal FZ's (4e-16). The spline part of J and p is exactly 0
  outside the bands. Tensor and point evaluators agree to 1e-16. Save/load and embedding are exact.

## 6. Results

The runs were the FZ control (`fzc`: FZ 32/32/6 through the same pipeline on the same grid) and the hybrid (`h1`).
Both used field grid a/100 and started from the report's FZ shielded start. The β runs had 12 outer iterations each,
about 480 trials for both bases. Values below are on the common grid (field a/120, the same radial collocation).

### 6.1 β scan

| β | report FZ (a/60) | FZ control | **hybrid** | FZ control par / perp | **hybrid par / perp** |
|---|---|---|---|---|---|
| start | 0.128 | 0.098 | **0.054** | 0.070 / 0.054 | **0.036 / 0.031** |
| 0.02 | 0.129 | 0.099 | **0.051** | 0.071 / 0.054 | **0.031 / 0.033** |
| 0.05 | 0.133 | 0.107 | **0.050** | 0.080 / 0.055 | **0.029 / 0.034** |
| 0.1 | 0.150 | 0.122 | **0.051** | 0.093 / 0.059 | **0.029 / 0.036** |
| 0.2 | 0.175 | 0.148 | **0.058** | 0.104 / 0.082 | **0.032 / 0.040** |
| 0.35 | 0.205 | 0.138 | **0.074** | 0.076 / 0.101 | **0.039 / 0.053** |
| 0.5 | 0.207 | 0.135 | **0.081** | 0.077 / 0.095 | **0.041 / 0.059** |
| 0.75 | 0.235 | 0.155 | **0.095** | 0.098 / 0.100 | **0.053 / 0.066** |
| 1.0 | 0.236 | 0.184 | **0.110** | 0.124 / 0.111 | **0.064 / 0.074** |

The FZ control's dip at β 0.35–0.5 is effort-limited; the report's curve flattened there too. The hybrid's curve is
monotone from β = 0.05 on.

### 6.2 Releases (pin removed, α = 0.02)

| released from | FZ control: M, b21 | hybrid: M, b21 |
|---|---|---|
| β = 0.05 | 0.097, 0.65% (closes) | 0.049, 4.4% (stays open, island phase) |
| β = 0.5 | 0.107, 20.9% (closing) | 0.054, 19.1% (closing) |

### 6.3 Islands and pressure

All pinned states are real (2,1) islands in both bases: the traced width equals the pendulum width (0.025 / 0.051 /
0.080 / 0.114 at β = 0.05 / 0.2 / 0.5 / 1). The pressure flattens inside them. The hybrid keeps a slightly larger p′
at the resonance (0.04–0.12 of DESC's, against 0.001–0.02 for FZ), because its band splines carry a sharper p across
the island.

### 6.4 Where the remaining force sits

In the hybrid, the parallel force is now small in the core and inside the island band. It peaks at q = 3 (ρ 0.9–0.95:
0.16), at the edge (0.12), and at q = 2 itself (ρ 0.6–0.7: 0.04).

## 7. b·∇p: what controls it (fixed-J tests)

- **Baseline.** In every FZ state of the report, b·∇p is about half of M (0.079 of 0.124 at the shielded start). It
  peaks at q = 3, at the edge, and in the q = 2 band for island states.
- **More pressure freedom at fixed J changes only the perpendicular part.** In the hybrid p space, which contains
  FZ's, M fell 6–9% (start 0.128 → 0.120), all of it perpendicular; the parallel part stayed 0.0803 → 0.0803. In a pure
  spline p space the perpendicular force rose to 4–40 M_DESC, because the FZ-optimized J relies on an FZ p to 1e-7. The
  least-squares solve itself is exact (Galerkin residual 1e-16).
- **Lowering b·∇p therefore requires J to change.** With T and p refined together in the hybrid, the parallel force
  halved: 0.070 → 0.036 at the start and 0.124 → 0.064 at β = 1.

## 8. Interpretation

- **Most of the remaining force was a representation limit near the resonant surfaces**, not a property of the
  equilibrium. Refining the radial representation in the q = 2 and q = 3 bands lowers the whole landscape by about 45%
  at equal effort.
- **The report's picture holds for FZ and for large islands:** one valley, islands on its wall, released large islands
  close.
- **With the hybrid, the floor flattens between shielding and small islands.** The β = 0.02–0.1 states (island widths
  up to about 0.03 in ρ) sit at or below the over-shielded start, and a released small island stays open. This
  matches the hand-off's model C: shielded and reconnected states approach each other as the representation improves,
  and the hybrid acts as a resolution step concentrated at the resonance.

## 9. Solver and infrastructure findings

- **Lanczos cap.** With the default 400, every cap hit at n_u > 20k triggered a force-metric rebuild of 10–16 min.
  Rebuilds took 75–90% of the run time (about 1000 s per trial for s2). Use `--cg 1500`.
- **M_DESC on the wrong grid.** Spline runs computed M_DESC on the uniform Gauss grid while the run used composite
  nodes. Fixed: `free_boundary.reference(rho_nodes=…)`.
- **Bound met too easily.** At a/100 both bases met α = 0.1 within minutes (the report never did at a/60). A met bound
  hides the landscape, since every β state then sits at M = 0.1, so the shielded start used α 0.1 → 0.05 and the β
  and release runs used α = 0.02.
- **Unequal cost per trial.** Hybrid trials are slower (n_u 21k), so its β runs were extended from their end states
  to the same 12 outer iterations as FZ (`p2_beta_ext.sh`).
- **Stellar-AI scheduling.** The QOS is set by the time limit (> 4 h gives gpu-medium). Each QOS allows 10 submitted
  jobs per user, including jobs from other sessions.
- **Live convergence log.** Every run writes two files:
  - `<out>_live.tsv`, one line per trust-region trial: η, predicted and actual decrease, Δ → Δ_next, step kind
    (boundary, interior or noise), σ_f, |g| and |g|/|g₀|, GN share, Lanczos count, cap hits and residual, Ritz values,
    M/ε, M/M_DESC, b21, objective, penalty and multiplier, Galerkin residual, time.
  - `<out>_status.json`: window statistics and warnings (radius collapse, runs of rejections, noise-level steps,
    M/ε not falling while the bound is unmet).

  `python scripts/meanforce/tr_live.py out/rs/` shows one line per run; `tr_live.py <run>_live.tsv -f` follows one run.

## 10. Caveats

- **Upper bounds.** Every value comes from a fixed effort, not a converged minimum.
- **Effort at the start.** The hybrid's shielded start had fewer trials (325) than its β runs (about 480). Part of
  "β 0.02–0.1 below the start" may be effort, as in the report's own release test. The open release from β = 0.05
  (121 trials) is suggestive, not conclusive.
- **Limited scope.** One equilibrium, one FZ resolution, one band layout, |m| ≤ 16 in the bands. The α targets differ
  from the report's.
- **Grid.** The new runs were made at a/100 and evaluated at a/120; the FZ control changed by 0–1% between the two.

## 11. Next steps

1. A longer release from the hybrid's β = 0.05 and 0.1 states, plus a longer shielded start, to settle whether the
   floor's minimum is shielded or a small island.
2. Finer bands at q = 3 and the edge, where the remaining parallel force sits.
3. A hybrid resolution check: FZ 40/40/6 with the same bands, and band spacing 0.005.
4. The hybrid with a max-type bound (band p16, as in the report's Q2b).

## 12. Code added on `ripple-spline`

- **Spline bases.** `nest/spline_basis.py` and `nest/spline_pressure.py` (ported from dshape-fixed, then extended):
  - ρ^q axis regularity (`reg="rhom"`) and explicit knot bands;
  - band-only splines and the hybrid `CompositeBasis`;
  - `separable_for`, and the `radial_table` hook in `flux_interp.FactoredFactors`.
- **Hybrid fits and storage.** `nest/boundary_potential.py` and `nest/pressure.py`: hybrid fits
  (`radial="fz+spline"`, `spline=…`), exact FZ embedding, and save/load.
- **`scripts/meanforce/mf_fixed.py`.**
  - New flags: `--current/--pressure fz+spline`, `--spl-band`, `--spl-reg`, `--spl-qcap`, `--colloc-knots`,
    `--force-split`.
  - `--eval-state` now embeds states from the `--start-LMN` basis.
  - New `LiveLog` and status file.
- **Helper scripts and tests.** `scripts/meanforce/basis_args.py` (spline and hybrid bases for `island_seed.py` and
  `start_check_bext.py`) and `scripts/meanforce/tr_live.py`. Gates in `tests/test_ripple_spline_gates.py` and
  `tests/test_hybrid_gates.py`.
- **Job scripts.** `scripts/ripple_spline/`: env, cfg (KIND = spline, hybrid or fz), gates, phase0/0b, knot scan,
  p2_* chain, `p2_beta_ext.sh`, `p3_common.sh`, `collect.py`.

## 13. Is the spline J_fit error "less gradient-like"? (fit-split test, 2026-11-13)

At J_fit, option A gives B = B_DESC exactly (2.5e-17), so the initial force is DESC's own force F_D plus
dG = (J_fit − J_DESC) × B_DESC, minus what the pressure absorbs. `mf_fixed.py --fit-split` splits dG = ∇δp + R by the
runs' own weighted least squares (`scripts/ripple_spline/fitsplit.sh`, a/100, data `results/ripple_spline/fitsplit/`).
FZ, s1 and the two crosses share one radial grid; s2 and the hybrid use their own. Values are in M_DESC units.

| T basis / p basis | rms δJ / rms J_DESC | dG mean / rms | ∇δp rms | R mean / rms | rms R / rms dG |
|---|---|---|---|---|---|
| FZ / FZ | 3.42e-4 | 6.5 / 19.7 | 11.9 | 6.8 / 15.7 | 0.80 |
| spline s1 / spline s1 | 1.95e-4 | 11.2 / 150 | 128 | 7.7 / 77.3 | 0.52 |
| spline s2 / spline s2 | 2.38e-4 | 9.0 / 210 | 180 | 5.7 / 108 | 0.51 |
| spline s1 / **FZ p** | 1.95e-4 | 11.2 / 150 | 0.2 | 11.2 / 150 | 1.00 |
| FZ / **spline s1 p** | 3.42e-4 | 6.5 / 19.7 | 19.3 | 2.2 / 3.7 | 0.19 |
| hybrid / hybrid | 2.55e-4 | 6.4 / 19.1 | 11.9 | 4.1 / 14.9 | 0.78 |

- **‖δJ_spl‖ < ‖δJ_FZ‖: true** (1.9–2.4e-4 against 3.4e-4).
- **‖R_spl‖ ≫ ‖R_FZ‖: true in rms (5–7×), not in the mean** (1.1× for s1, 0.84× for s2). The spline remainder is
  concentrated at ρ 0.3–0.5 (R per bin 12–18 against 4 for FZ) and in the last edge bin.
- **"The FZ error is mostly a gradient that p absorbs": false.** FZ's own pressure absorbs only ~20% of its dG (rms
  R / rms dG = 0.80); the spline pressure absorbs ~48% of the spline dG.
- **What makes the difference is dG itself.** The spline dG is 7–11× larger in rms than FZ's although δJ is smaller:
  rms dG / rms δJ is 13–15× larger for splines.
- **Direct split of δJ along b = B_DESC/|B_DESC|** (`dj_*.json`; rms as a fraction of rms J_DESC):

  | basis | rms δJ | parallel | perpendicular | perpendicular share of abs(δJ)² |
  |---|---|---|---|---|
  | FZ 32/32/6 | 3.42e-4 | 3.42e-4 | 1.19e-5 | 0.1% |
  | pure spline s1 | 1.95e-4 | 1.73e-4 | 8.86e-5 | 21% |
  | pure spline s2 | 2.38e-4 | 2.03e-4 | 1.24e-4 | 27% |
  | hybrid | 2.54e-4 | 2.54e-4 | 1.16e-5 | 0.2% |

  FZ's fit error is 99.9% field-aligned, i.e. force-free. The pure-spline error has 7–10× more perpendicular current
  (8.9e-5 and 1.24e-4 against 1.19e-5), which matches the 7.6× and 10.7× larger rms dG. By ρ bin, the spline δJ⊥ is
  5× FZ's at ρ 0.3–0.5 (s1) and 8–11× FZ's in the last edge bin; the hybrid's δJ⊥ equals FZ's everywhere.
- **The crosses confirm that T is the problem, not p.** FZ's pressure cannot absorb any of the spline dG (R/dG = 1.00).
  The spline pressure absorbs FZ's dG better than FZ's own pressure does (0.19 against 0.80), so the spline pressure
  space is the richer one; the defect is in the spline current.
- **So the boxed inequality holds, but for this reason:** the pure-spline fit puts a fifth to a quarter of its error
  across B, and that part both makes force and is only half absorbable by p.
- **At J_fit the remainder is almost entirely perpendicular** (R parallel 0.07–0.13 of 4–11 in the mean). The large
  b·∇p of the optimized states develops during the optimization.

## 14. Why is the spline fit error perpendicular? Weighted current fits (2026-11-13)

Test (`mf_fixed.py --fit-split --wfit λ… --wfit-direct`, `scripts/ripple_spline/wfit_direct.sh`; data
`results/ripple_spline/fitsplit/wfitd_*.json`). J is re-fitted at the collocation points with the force's volume
weights, min ‖δJ∥‖² + λ‖δJ⊥‖² (b = B_DESC/|B_DESC|), starting from the current fit. It is solved exactly: the dense
normal matrices N∥ and N⊥ are built column by column and N∥ + λN⊥ is factored by Cholesky; the normal-equation
residual is 1e-8 to 1e-12 relative. A first attempt by CG (3000 iterations, `wfit_*.json`) had not converged and
showed almost no change; disregard it. "Start force" is the force of an option-A start on that fit (B = B_DESC, the
run's own least-squares p), volume mean and rms in M_DESC.

| basis / fit | rms δJ | rms δJ⊥ | perpendicular share of abs(δJ)² | start force mean / rms |
|---|---|---|---|---|
| FZ, standard fit (T in the radial gauge) | 3.42e-4 | 1.19e-5 | 0.1% | 6.89 / 15.7 |
| FZ, pointwise J fit, λ = 1 | 3.25e-4 | 8.6e-6 | 0.1% | 6.16 / 9.27 |
| FZ, λ = 100 | 3.27e-4 | 4.7e-6 | 0.02% | 4.24 / 5.55 |
| spline s1, standard fit | 1.95e-4 | 8.86e-5 | 21% | 7.88 / 77.3 |
| spline s1, λ = 1 | 1.10e-4 | 5.3e-6 | 0.2% | 6.42 / 9.17 |
| spline s1, λ = 100 | 1.11e-4 | 5.1e-6 | 0.2% | 5.91 / 8.79 |
| spline s1, λ = 1e4 | 1.73e-4 | 4.5e-6 | 0.07% | 4.97 / 7.68 |
| spline s2, standard fit | 2.38e-4 | 1.24e-4 | 27% | 5.85 / 108 |
| spline s2, λ = 1 | 8.2e-5 | 3.2e-6 | 0.15% | 4.34 / 5.57 |
| spline s2, λ = 100 | 8.2e-5 | 2.7e-6 | 0.1% | 3.69 / 4.88 |
| spline s2, λ = 1e4 | 1.36e-4 | 2.2e-6 | 0.03% | 2.80 / 3.89 |

**The perpendicular error came from the standard fitting procedure, not from the spline basis, and not mainly from
the objective treating both components alike.**

- **Already at λ = 1 the spline fit is field-aligned.** The plain least-squares fit of J at the collocation points
  leaves 99.8% of the pure-spline error along B. Its δJ⊥ is 17× (s1) and 39× (s2) smaller than the standard fit's.
- **The start force then matches or beats FZ's.** For s2 it is 4.34 / 5.57 against FZ's 6.89 / 15.7, and the 5–7×
  rms excess disappears. The standard fit (`BoundaryPotential.fit`) fits the potential T in the radial gauge, mode by
  mode, on its own radial nodes. Its error in T reaches J through ∂T/∂ρ, which amplifies the knot-scale oscillations
  of a spline fit into perpendicular current. For FZ that derivative error is smooth.
- **Weighting the perpendicular part (λ > 1) helps further in both bases.** At λ = 1e4 the start force falls 23–36%
  for splines against λ = 1; at λ = 100 FZ falls 31%. The total δJ grows slightly, as expected.
- **The core is the exception.** At ρ 0.2–0.5, R is unchanged by any λ (s1: 6–18 per bin; s2: 3–7, against FZ's 3–4
  at λ = 100). There the spline space itself cannot represent the balanced current; the core knots are too coarse.
  Away from the core, and especially in the edge bin (s1: R 31 → 4–7), the large error was the fit's.
- **The amplitude-error explanation fails for FZ.** DESC's own current is 17% perpendicular in rms, but FZ's error is
  only 3.5% perpendicular even in the standard fit. FZ's error is field-aligned beyond what an error proportional to J
  would give.

Implication for the pure-spline runs: their J_fit carried large, avoidable perpendicular errors, and |J − J_fit| anchored
the optimizer to that target. A rerun of the pure-spline continuation from the pointwise J fit (λ = 100) shows how
much of the 0.38–0.50 stall came from the start and how much from the core basis limit.

**Rerun (running, 2026-11-13).** `mf_fixed.py --jfit-wfit 100` (`meanforce_fixed.weighted_jfit`) replaces J_fit by the
exact weighted fit before option A is set up, so the start, the |J − J_fit| target and B_ext all use it. Runs `s1w` and
`s2w` are otherwise identical to s1 and s2 (knots, a/100, α 0.3 → 0.01, `--cg 1500`): three chained 4 h gpu-short
jobs each (`CONT_EXTRA=--jfit-wfit_100`), outputs `out/rs/s1w`, `out/rs/s2w`. To compare: s1 0.50 and s2 0.38 after
~11 h, FZ control 0.098, hybrid 0.054 M_DESC.

## 15. The standard spline fit at much higher resolution (2026-11-13)

Test: the same standard fit (`BoundaryPotential.fit`, T potential in the radial gauge) with s1's layout at finer knots,
and enough radial fit nodes (`--jfit-nrho`; the default caps them at 300). Script `scripts/ripple_spline/fitres.sh`,
data `results/ripple_spline/fitsplit/res_*.json`. J_fit diagnostics only (a/100, own spline p).

| fit | knots (outer × FZ spacing / q2 / q3) | n_u | min spacing | rms δJ | rms δJ⊥ | rms R | start force mean |
|---|---|---|---|---|---|---|---|
| s1, 300 fit nodes | 0.6 / 0.015 / 0.02 | 23.7k | 1.6e-3 | 1.95e-4 | 8.9e-5 | 77 | 7.9 |
| s1n, 900 fit nodes | 0.6 / 0.015 / 0.02 | 23.7k | 1.6e-3 | 1.95e-4 | 8.8e-5 | 77 | 7.9 |
| f1, 1200 fit nodes | 0.3 / 0.0075 / 0.01 | 42k | 3.8e-4 | 4.4e-4 | 1.9e-4 | 165 | 6.6 |
| f2, 2000 fit nodes | 0.15 / 0.004 / 0.005 | 79k | 9.7e-5 | 9.7e-3 | 1.2e-3 | 963 | 57 |
| weighted J fit (λ = 100), s1 knots | 0.6 / 0.015 / 0.02 | 23.7k | 1.6e-3 | 1.1e-4 | 5.1e-6 | 8.8 | 5.9 |

δJ⊥ by ρ bin (1e-4 of rms J_DESC), innermost / ρ 0.3–0.5 / outermost: s1 0.01 / 0.10 / 3.1; f1 0.35 / 0.03 / 6.6;
f2 8.6 / 0.07 / 42.

- **The standard fit was not undersampled.** Tripling its radial nodes changes nothing.
- **Refining the knots makes the standard fit worse, not better.** Only the core (ρ 0.3–0.5) improves, by 3–5×. Near
  the axis and at the edge the error grows: there FZ's zero spacing puts the knots very close together (down to 1e-4
  at the edge for f2), the splines become nearly collinear, and the fitted T's knot-scale error is amplified by
  J ∝ ∂T/∂ρ. At 4× resolution the start force is 7× worse than at the original resolution.
- **The fix is the fitting procedure, not resolution.** The pointwise weighted J fit at the original knots beats every
  standard fit: δJ⊥ 17× smaller than s1's, rms R 9× smaller.

## 16. Follow-up of the report's next steps (2026-11-13 to 11-16)

Job list `scripts/ripple_spline/jobs_followup.txt` (queue runner `queue.sh`); data in `results/ripple_spline/runs/` and
`results/ripple_spline/common/`. M in M_DESC on the common grid (field a/120) unless marked "run grid".

### 16.1 Convergence and stationarity (item 1)

Long runs (up to 60 outer iterations, α = 0.02) for the hybrid (h1) and the FZ control (fzc). The modes are S (free
from the shielded start), R05 (free from the β = 0.05 state), P0 (island-phase b21 pinned to 0) and P05 (pinned
β = 0.05). Then an uncapped continuation (x*); `converge_check.py` reads the live logs.

**All 16 runs ended at the merit's noise floor, not at a converged minimum.** The augmented-Lagrangian penalty κ
climbs to its 1e8 cap when the bound is never met, and κc² then swamps the merit. Every step in the last 50 was
noise-level, and the trust radius fell to ~1e-13. The one exception is the FZ release from β = 0.05, which converged
(stationary stops).

The `--rho-max` cap added for the continuations had no effect: the fixed-boundary runs use
`FixedSolver.solve_alpha`, which overrides `Solver.solve_alpha`. That override is now fixed (it clamps a resumed κ),
and capped continuations (κ ≤ 1e4, `cy_*`) are running.

Lowest M reached (run grid):

| | S (free, start) | P0 (b21 = 0) | R05 (free, from β 0.05) | P05 (pinned β 0.05) |
|---|---|---|---|---|
| hybrid | 0.0522 (b21 1.3%, anti) | 0.0520 | 0.0448 (b21 4.35 → 5.2%, stays open) | 0.0454 |
| FZ control | 0.0969 (b21 0.3%) | 0.0948 | 0.0948 (b21 0.9%, closes; converged) | 0.0937 |

- **Hybrid:** the small-island states sit ~14% below its shielded and zero-island states, and a released island grows
  slightly instead of closing.
- **FZ:** the four states agree within 3%, and a released island closes.
- These are stuck states; the capped continuations will tell whether the gap holds.
- Caveat: the merit trades M against |J − J_fit|; the free runs (S, R05) are not pure minimizations of M.

### 16.2 Resolution (item 2) and q = 3 / edge refinement (item 3)

Common-grid M (parallel / perpendicular) and band p16 / DESC's at the q = 2 band. β runs had 12 outer iterations (h3
and h4 only 6–8, being extended).

| config | n_u | start | β = 0.05 | β = 0.2 | β = 1 |
|---|---|---|---|---|---|
| FZ 32/32/6 (fzc) | 7.7k | 0.098 (0.070/0.054) | 0.107 | 0.148 | 0.184 |
| hybrid 32 (h1) | 21k | 0.054 (0.036/0.031) | 0.050 | 0.058 | 0.110 |
| FZ 40/40/6 (fz40) | 11.7k | 0.057 (0.039/0.033) | 0.059 | 0.0545 | 0.086 |
| hybrid 40, bands 0.01 (h2) | 25k | 0.038 (0.020/0.027) | 0.031 (0.014/0.024) | 0.031 (0.012/0.026) | 0.057 (0.032/0.039) |
| hybrid 32, bands 0.005 (h3)* | 37k | 0.050 | 0.031 | 0.048 | 0.153 |
| hybrid 32 + q = 3 / edge bands (h4)* | 29k | 0.051 | 0.037 | 0.055 | 0.104 |

\* h3 and h4 had 241–323 trials against 480 (their trials are slower). Their β = 1 runs were still falling 20–25%
over the last 20% of trials, so their large-β values are not comparable yet. Extensions to 12 outer iterations
(`x_h3_*`, `x_h4_*`) and their common-grid evaluations are queued.

- **The hybrid's advantage persists at FZ 40/40/6.** h2 is 34–47% below fz40 at every β, at equal effort (480 trials
  each). It also lowers the parallel force: start 0.020 against 0.039.
- **The landscape flattens with resolution, in both bases.** At 40/40/6 the β = 0.05 and β = 0.2 states sit at or
  below the shielded start (fz40: 0.0545 < 0.057; h2: 0.031 < 0.038), while β = 1 still costs ~1.5× the start.
  Several of these runs were still falling (5–15% over their last 20% of trials), so the small-β ordering is
  suggestive. This supports the hand-off's model C.
- **Finer bands (h3) and q = 3 / edge refinement (h4) at 32/32/6** improve the start and the small islands
  (0.050–0.051 against h1's 0.054; β = 0.05: 0.031–0.037 against 0.050). h4 did not lower the start's parallel force
  (0.039 against 0.036), so the q = 3 / edge residual is not removed by these bands alone. Final numbers await the
  extensions.

### 16.3 Localized force bound (item 4)

The band p16 of |F| over ρ 0.5675–0.7075 (a smooth maximum at q = 2) was bounded at α × DESC's band value, with α going
0.3 → 0.02, and the whole-volume mean ≤ 0.25 M_DESC in the same composite constraint (`loc.sh`).

- Both bases met α = 0.02 (band p16 = 2% of DESC's) for the shielded state (S) and the pinned β = 0.5 island (P5);
  the hybrid also met it for the pinned β = 0.05 island (P05).
- FZ P05 stalled at 1.38× its α = 0.03 target when the budget ran out.
- The hybrid met the bound with half the departure from J_fit (|J − J_fit|/rms 0.013 against 0.026 for S).
- So the globally low mean force is not hiding a large force concentration at q = 2: the q = 2 band can be made
  nearly force-free in both bases, including with islands. The remaining force sits away from q = 2 (q = 3, edge,
  core). The whole-volume means of these states come with the common-grid evaluation (`e_h1`, `e_fzc`).

### 16.4 Pure splines from the weighted J fit

`--jfit-wfit 100` reruns of s1 and s2 (three 4 h jobs each, same settings):

| run | start force mean | floor (run grid) |
|---|---|---|
| s1, standard fit | 7.88 | 0.501 |
| s1w, weighted fit | 5.91 | 0.500 |
| s2, standard fit | 5.85 | 0.381 |
| s2w, weighted fit | 3.69 | 0.386 |

The much better start (17–46× less δJ⊥) leaves the floor unchanged within 1.5%. **The pure-spline stall is a basis
limit** (the coarse smooth core, §4.3), not an artifact of the start. The standard fit's perpendicular error made the
start worse but is not what the optimizer gets stuck on.

### 16.5 Pending

- Capped-penalty continuations `cy_h1_*`, `cy_fzc_*` and their common-grid evaluations `e_h1`, `e_fzc` (convergence
  table, localized states' mean force).
- h3 / h4 extensions to 12 outer iterations and their common-grid evaluations.
