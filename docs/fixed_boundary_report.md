# Fixed-boundary, matrix-free mean-force NEST (branch `fixed-boundary`)

Branch `fixed-boundary` sits on `closest-poisson` (e3fbe0f). It adds new files only; no existing module is changed, so free-boundary users of `meanforce`, `pointwise`, `objective` are unaffected.

What it gives you:

1. A fixed-boundary version of the mean-force problem with the pressure eliminated by the weak Poisson projection and everything matrix-free (Lanczos/GLTR trust region), `nest/meanforce_fixed.py` + driver `scripts/meanforce/mf_fixed.py`.
2. The solver changes that were needed to converge small force bounds (alpha = 0.05 and below) on NCSX at 32/32/24 and 48/32/18, a/40, on one H200.
3. Diagnostics that read the saved states: Poincare sections, field-line iota, shielding of resonant fields, force decomposition, pressure along field lines, the magnetic differential equation (MDE) budget, and two coordinate-free Pfirsch-Schlueter tests along field lines.
4. Run records, the converged states (for warm starts) and the diagnostic outputs, under `results/meanforce_fixed/` and `figures/meanforce_fixed/`.

Section 7 lists what is not finished. In short: alpha = 0.05 is converged at 32/32/24; alpha <= 0.01 is not.

---

## 1. The problem being solved

Everything lives on the fixed geometry of the input DESC equilibrium (`NCSX_fb_L16M16N12.h5`, NFP 3, a = 0.3226 m): DESC's (rho, theta, zeta) are only a mesh and a basis domain. The unknown is the vector u of coefficients of the current potential (`BoundaryPotential` basis, size n_u). The magnetic field is B = B_ext (coils, c09r00) + B_plasma(u), with B_plasma from the cylindrical annulus solve at spacing a/frac (frac = 40 here, 20.4 M nodes).

On n collocation points with volume weights w_i = dV_i / V and sw = sqrt(w):

| symbol | definition | size |
|---|---|---|
| a | sw (J x B) | 3n |
| P c | sw grad p(c), linear in the pressure coefficients c; constant because the geometry is fixed | 3n x n_p (never stored) |
| G | P^T P, built once by radial slices, Cholesky-factored | n_p x n_p |
| p* | G^-1 P^T a, the weak (Galerkin) Poisson pressure: the best p for this J | n_p |
| f | a - P p* = sw F, the force residual left after the best pressure | 3n |
| M | sum_i w_i \|F_i\| / norm, DESC-normalized mean force | scalar |
| g | sw (J - J_fit) / rms J_fit, distance from the fitted DESC current | 3n |

The problem: minimize phi(u) = 1/2 \|g\|^2 subject to c_F = M / eps - 1 <= 0, with eps = alpha M_DESC (M_DESC = 8.7788e-4 for this input). It is solved with `meanforce`'s augmented-Lagrangian outer loop (multiplier nu_F, penalty rho) and a trust-region inner loop.

Why eliminate p: with p as an unknown, the joint Jacobian is at least as ill-conditioned as the pressure-gradient columns alone, which grows with spectral resolution and rules out iterative solvers. The reduced problem depends on p only through range(P); its conditioning is set by cond(G) = 4.4e4 (16/16/12), 2.8e6 (32/32/24), which the Cholesky route handles safely. The G3 gate checks the implicit projection against an explicit thin-QR projection to 1e-12.

There is no B.n constraint and no flux constraint (fixed boundary). The LCFS B.n/|B| of the solutions is monitored instead (Section 6).

---

## 2. Code map

| file | role |
|---|---|
| `nest/meanforce_fixed.py` | `FixedAnchor` (state: J, a, p*, f, M by the implicit projection), `FixedRows` (Gauss-Newton/majorizer rows with `J v`, `J^T w`), `FixedRowsB` (beta model), `FixedRowsH` + `lanczos_tr_step_H` (exact HVP), `FixedSolver` (outer-loop rules), `use_models` (selects model, metric and rebuild rules), `gram_P`, `gram_J`, `gram_F_local`, `force_metric` and their `*_lean` versions, `embed` / `embed_truncate` (warm starts across bases) |
| `scripts/meanforce/mf_fixed.py` | driver; logs every step to `<out>.json` (and `--mirror`, fsynced), writes `<out>_a<alpha>.npz` and `<out>_a<alpha>_latest.npz` |
| `tests/test_meanforce_fixed.py` | G1-G6: slice G and H_J against explicit builds, implicit projection against QR, J v against finite differences, adjoint, grad c_F |
| `tests/test_hvp_fixed.py`, `test_beta_rows.py`, `test_force_precond.py`, `test_lean_metric.py`, `test_embed.py` | gates of the exact HVP, the beta rows, the force preconditioner, the lean builds, the basis embedding |
| `tests/prec_force_precond.py` | measurement: FP32/TF32 against FP64 force-metric builds |
| `scripts/meanforce/*_fixed.py`, `fieldline_iota.py`, `pressure_sections.py`, `island_p_representability.py`, `poincare_multi.py` | diagnostics (Section 6) |

All gates run on CPU on NCSX 8/8/6 (n_u 694, n_p 295, 38,836 points, 0.23 M annulus nodes) in 8-160 s each and print `ALL PASSED`:

```
export PYTHONPATH=$PWD/nest:<DESC checkout> JAX_PLATFORMS=cpu NEST_DEVICE=cpu NEST_CACHE=<cache dir>
for t in test_meanforce_fixed test_hvp_fixed test_beta_rows test_force_precond test_lean_metric test_embed; do
  python tests/$t.py; done
```

Gate values on this branch: G 1.1e-15, H_J 5.7e-16, implicit vs QR f 9.5e-13, J v vs FD 2.2e-10, adjoint 9.3e-15, grad c_F vs FD 2.4e-5; HVP vs FD 4.4e-9 at h = 1e-8 (O(h^2)), symmetry 5.4e-15; beta rows <= 6.9e-15; force-metric Gram 2.9e-15; lean vs reference H_F 1.5e-15, Cholesky 2.0e-13.

---

## 3. Solver components and why each is there

Each item is opt-in or default as stated. The evidence is from NCSX a/40 on one H200 unless noted.

1. **Pressure elimination and slice-built Grams (default).** No n x n_p matrix and no n x n_u current table are stored. G and H_J are accumulated over radial slices of the collocation grid, then Cholesky-factored on the device (cuSOLVER `eigh` fails above ~25k; `potrf` works). At 32/32/24: setup 257 s, J v 0.042 s, J^T w 0.132 s.
2. **Least-squares multiplier with a [1/4, 4]x clip (default; `--no-mult-ls` disables).** At the start of each bound nu_F is set to the least-squares estimate, i.e. the multiplier that best cancels grad phi + nu grad c_F in the metric norm, clipped >= 0. After each outer the usual first-order update nu_F + rho c_F is clipped to [1/4, 4] times the current least-squares estimate. This assumes near-stationary inner solves; with very small inner budgets it can hold nu_F near zero (seen in capped local tests).
3. **Complementarity stop and trust-radius reset per outer (default; `--no-delta-reset`).** The bound loop stops when the inner solve is stationary, c_F <= 1e-2, and the bound is consistent with its multiplier: either nu_F = 0, or c_F >= -1e-2 (active). A bound that is slack while nu_F > 0 counts as a violation. The trust radius is reset to `--delta0` at each outer.
4. **Local force preconditioner (`--precond force`).** With the H_J-only metric chol(H_J), the alpha = 0.05 Gauss-Newton subproblem was far from converged at the 400-iteration Lanczos cap (residual/\|g\| 1.21; Ritz values [1.8e5, 9e10]). The force-bound curvature dominates at small alpha. The metric chol(H_J + H_F_loc + a a^T) adds the weighted, pressure-projected Gram of sw (dJ x B0) (the J0 x dB part is dropped). It is rebuilt per outer and on triggers (Lanczos count >= `--outer-rebuild-factor` x the post-build count, `--refresh-lz`, `--refresh-dM`); the trust radius is mapped across rebuilds. Gate F2: 41 -> 13 Lanczos iterations. FP64 is required: FP32/TF32 builds lose positive definiteness once trace H_F / trace H_J exceeds ~1e8 (TF32 already at 2.7e4).
5. **Lean builds (default with `--precond force`; `--no-lean-metric` for the reference).** H_J is kept in host RAM and the force metric is assembled by blocks with donated Cholesky. At 48/32/18 (n_u 41,206, n_p 20,011): force-metric peak 53.1 GB instead of ~101 GB, 20.3 GB in use afterwards, 289 s per build.
6. **Beta model rows (`--model beta --beta 0.5`).** The Gauss-Newton majorizer of c_F is conservative near the bound; the beta model keeps its gradient but scales the curvature of the force rows along the active direction by beta. This is the only model that converged alpha = 0.05. The exact HVP (`--model hvp`, `FixedRowsH`) is exact and cheap (1.0x a GN product) and converged alpha = 0.7 in 2.8x less solve time than GN (fxL32h, same optimum to 1e-7), but at alpha = 0.05 it collapsed the trust radius far from the bound and met negative curvature near it. `--beta-adapt` (self-tuning beta) was on in the 48/32/18 runs; on its own it showed no gain in local tests.
7. **`--lanczos-stop tr`.** `pointwise.lanczos_tr_step` stops only when both the TR step and the unconstrained Gauss-Newton minimizer have converged. Near the bound that made boundary steps take 224-400 Lanczos iterations (interior 34-48) and fired needless metric rebuilds. `tr` (`lanczos_tr_step_H` on J^T J) stops on the TR residual only; identical steps in a local A/B, and the alpha = 0.005 restart state went from the 400 cap to 79-102 iterations.
8. **Warm starts across bases (`--start <npz> --start-LMN L M N`).** `embed` zero-pads exactly (`test_embed.py`). `embed_truncate` drops the coefficients missing from the target basis (48/32/18 from 32/32/24 drops the N > 18 part: 7,116 coefficients, 0.33% of \|u\|).

The driver defaults are still `--model hvp --precond hj --lanczos-stop both`. They are kept for parity with earlier runs, but **use the recipe in Section 4 for alpha <= 0.1.**

---

## 4. Running it

Environment (as used on Modal; Stellar needs the same variables):

```
export PYTHONPATH=$PWD/nest:$PWD:<DESC checkout>
export NEST_DEVICE=gpu JAX_PLATFORMS=cuda,cpu XLA_PYTHON_CLIENT_PREALLOCATE=false XLA_PYTHON_CLIENT_MEM_FRACTION=0.95
export NEST_CACHE=<shared cache dir>     # basis fits are cached here
```

From scratch at alpha = 0.7 (exact HVP is fastest here):

```
python scripts/meanforce/mf_fixed.py --LMN 32 32 24 --frac 40 --alphas 0.7 --cg 400 --delta0 1e-3 --out out/a07
```

alpha = 0.05 at 32/32/24, with the flags of the converged run fxD05b. fxD05b itself was continued from partial alpha = 0.05 states (fxD05 -> fxD05h, not committed) at M/eps 1.69. A single run from the alpha = 0.7 state with these flags has not been tried:

```
python scripts/meanforce/mf_fixed.py --LMN 32 32 24 --frac 40 --alphas 0.05 --cg 400 --delta0 1e-3 \
  --model beta --beta 0.5 --precond force --refresh-lz 2 --refresh-dM 0.3 \
  --start results/meanforce_fixed/states/fxL32h_a0.7.npz --out out/a005_32
```

Warm start into 48/32/18 (fxR48):

```
python scripts/meanforce/mf_fixed.py --LMN 48 32 18 --frac 40 --alphas 0.05 --cg 400 --delta0 1e-3 \
  --model beta --beta 0.5 --beta-adapt --precond force \
  --start results/meanforce_fixed/states/fxD05b_a0.05.npz --start-LMN 32 32 24 --out out/a005_48
```

Small alpha (fxR48b005; a smaller inner budget so the outer multiplier updates happen):

```
python scripts/meanforce/mf_fixed.py --LMN 48 32 18 --frac 40 --alphas 0.005 --cg 400 --delta0 1e-3 \
  --maxiter 25 --gtol 3e-3 --model beta --beta 0.5 --beta-adapt --precond force --lanczos-stop tr \
  --start results/meanforce_fixed/states/fxR48_a0.05_latest.npz --out out/a0005_48
```

`--time-budget <s>` stops cleanly with the state saved; `_latest.npz` is rewritten after every accepted step. Every step prints f, ratio, \|grad\|, Delta, GN share, M, c, nu, Lanczos count and time.

Memory and time on one H200 (JAX peak from the run JSON; nvidia-smi shows ~1.7-2.1x the JAX peak with preallocation off, so budget the device by that):

| basis | n_u / n_p | JAX peak GB | metric build | s per TR step, median (p10-p90) |
|---|---|---|---|---|
| 16/16/12 | 4,234 / 1,917 | 26.1 (setup transient) | 17 s (H_J) | 15 (7-29), fx16g |
| 32/32/24 | 29,074 / 13,753 | 31.0 (hj), 58.3 (force, reference build) | 66 s (H_J), ~132 s (force) | 9 (1-51) fxL32h; 21 (14-49) fxD05b |
| 48/32/18 | 41,206 / 20,011 | 53.3 (force, lean) | 270-290 s (force) | 50 (36-61) fxR48; 10 (1-40) fxR48b005 |

The ~26 GB peak at setup (annulus location, ForceResidual, Fields) is shared with the free-boundary path and is not yet attributed.

---

## 5. Results (NCSX, a/40)

Records: `results/meanforce_fixed/runs/<run>.json`; states: `results/meanforce_fixed/states/`.

| run | basis | alpha | model / metric | M / M_DESC | \|J - J_fit\| / rms | outers (inner stop) | solve s | status |
|---|---|---|---|---|---|---|---|---|
| fx16g | 16/16/12 | 0.7 | GN / H_J | 0.697 | 0.01795 | 2 (stationary, stationary) | 595 | converged |
| fxL32h | 32/32/24 | 0.7 | exact HVP / H_J | 0.700 | 0.02230 | 2 (stationary, stationary) | 617 | converged |
| fxD05b | 32/32/24 | 0.05 | beta 0.5 / force | 0.0500 | 0.09109 | 2 (stationary, stationary) | 1466 | converged; nu_F 2.44e-3 |
| fxR48 | 48/32/18 | 0.05 | beta 0.5, adapt / force | 0.0501 (M/eps 1.0023) | 0.0947 | stopped in outer 1 | - | checkpoint; nearly identical to 32/32/24 |
| fxA01 | 32/32/24 | 0.01 | beta 0.5, adapt / force | 0.0102 (M/eps 1.019) | 0.178 | stopped in outer 1 | - | not converged (killed: no progress) |
| fxR48b005 | 48/32/18 | 0.005 | beta 0.5, adapt / force, `--lanczos-stop tr` | 0.0050 | 0.2089 | 3 (budget x3) | 2104 | not stationary (see 7.2) |

For the two killed runs, \|J - J_fit\|/rms = sqrt(2 f) at the checkpoint (this identity reproduces the recorded value for fxR48b005: 0.2088 vs 0.2089).

`fxT1.json` is a cross-evaluation: the 16/16/12 alpha = 0.7 optimum embedded into 32/32/24 is feasible there (M/eps 0.954) with \|J - J_fit32\|/rms 0.0409. The difference from the 32/32/24 optimum (0.0223) is entirely the change of target J_fit between bases, not a weaker constraint.

For comparison, the collaborator's free-boundary 16/16/12 a/40 alpha = 0.7 run reaches 0.0149 against 0.01795 here at identical setup: boundary freedom lowers the J deviation by ~17% at that resolution.

---

## 6. What the states look like (diagnostics)

All diagnostics evaluate B = B_ext + the run's own annulus field. DESC's iota is negative, so resonant harmonics are written (m, -n); rationals are labelled n/m by \|iota\|. alpha = 0.005 diagnostics mostly use `fxR48b005_snap1.npz` (after outer 1; M/eps 0.993, f 2.31e-2); the MDE budget uses the final state. Outputs are in `results/meanforce_fixed/diagnostics/`. Figures named below are in `figures/meanforce_fixed/`.

1. **Islands shrink as alpha falls** (`poincare_fixed.py`, `fieldline_iota.py`, `shielding_fixed.py`; figures `fxP_48_05v005_rz.png`, `fxS48_vs_32.png`, `island_thetastar.png`). At each state's own resonant radius, for alpha 0.7 / 0.05 / 0.01 / 0.005 (the last two not converged): the 3/5 pendulum width is 0.109 / 0.059 / 0.032 / 0.023 in rho, and the 3/5 resonant b^rho relative to the coils alone is 0.12 / 0.038 / 0.013 / 0.007.
2. **The current profile drifts** (`iota_fieldlines.png`). The fixed-boundary problem does not pin <J.B>. alpha = 0.05 raises iota by +0.026 in the core and ~+0.005 at the edge; alpha 0.01-0.005 by ~+0.06 in the core, so 3/8 and 3/7 leave the plasma. The 3/5 surface moves from rho 0.793 (alpha 0.05) to 0.817 (0.005). Shielding numbers must therefore be taken at each state's own rational radius, not DESC's.
3. **No pressure flattening** (`p_contours_35chain.png`, `pressure_contours.png`, `island_p_representability.py`). p is constant along traced field lines to 1e-5 to 2e-4 of p(axis) for rho < 0.7, i.e. it follows the field's own (displaced) surfaces. Flattening fraction inside the islands is ~0 (3/6: -0.03 at alpha 0.05, +0.02 at 0.005; 3/5: -0.04 at 0.005; 3/5 at 0.05 not measurable). The pressure basis is not the obstacle: it can carry 53% (32/32) or 70% (48/32) of a pendulum island's flattening at 3/5.
4. **Where the force sits** (`edge_force_fixed.py`, `force_radial.png`). The grid M reproduces the solver's to ~1%. The parallel share \|b.grad p\| of M is 26% (alpha 0.7), ~52% (0.05), 60% (0.01), 53% (0.005). The 3/5 band (rho 0.75-0.85, 15.8% of the volume) holds 17.7% of M at 0.05, 29% at 0.01, 25% at 0.005. LCFS B.n/|B| rms stays ~2.8e-3 for all states (J_fit 2.7e-3, coils alone 2.9e-2).
5. **The MDE is satisfied increasingly well** (`mde_budget_fixed.py`). R = B.grad(sigma) + div(B x grad p/B^2), with sigma = J.B/B^2, measures where perpendicular force balance with the Pfirsch-Schlueter response fails. rms R / rms S: DESC 0.35, start 0.42, alpha 0.05 0.105, alpha 0.005 0.008; within 0.02 of 3/5: 0.13, 0.028, 0.006. Checks: div J <= 7e-3 \|J\|/a, div B <= 2e-3 \|B\|/a. In DESC coordinates the parallel operator is not diagonal for these displaced states, so DESC-coordinate sigma_mn cannot show or refute a 1/x tail. Hence the two tests below.
6. **No 1/x Pfirsch-Schlueter tail along field lines** (`sigma_lines_fixed.py`, `sigma_lines_res.png`). sigma at each traced line's crossings, Fourier-fitted in the line's own poloidal angle (0.1% residual). The m = 5 harmonic is 10-30 kA m^-2 T^-1 and smooth across 3/5 at alpha 0.05 and 0.005. DESC's own 3/5 drive would give a tail A/\|x\| with A = 528, i.e. 176 at \|x\| = 0.003, where \|x\| = \|iota - 3/5\|. Bound on the resonant drive relative to DESC's: 0.36 (alpha 0.05) and 0.48 (0.005) over \|x\| < 0.01; 0.07 and 0.11 over \|x\| < 0.003 (2-3 lines only). 3/6: 0.21 / 0.12 over \|x\| < 0.01.
7. **The resonant drive itself falls** (`hamada_lines_fixed.py`, `hamada_lines.png`). For nested surfaces with p' != 0 on a rational surface, a bounded sigma exists only if the closed-line integral of dl/B is the same on every closed line of that surface (Hamada/Newcomb). Its violation is the resonant harmonic (1/B^2)_mn in Boozer coordinates, the Pfirsch-Schlueter drive. The script integrates dl/B along traced lines over one closed orbit (15 field periods for 3/5, 6 for 3/6) and takes the resonant harmonic of W/<W>. The detuning nodes appear where predicted (\|x\| ~ 0.04 for 3/5), which confirms the quantity is resonant. Median over 0.003 < \|x\| < 0.03, relative to the start state (J_fit ~ DESC): 3/5 x0.54 at alpha 0.05 and x0.24 at 0.005; 3/6 x0.49 and x0.50 (but ~x0.1 near x -> 0 at 0.005; the 3/6 drive is ~20x smaller to begin with). With p' at 3/5 up by ~1.14, the expected tail at alpha 0.005 is ~0.3 of DESC's, consistent with item 6.

Reading (provisional): as alpha falls, the solver reshapes |B| on the low-order rational surfaces towards the solvability condition. This shrinks the pressure-driven resonant drive, the islands and the singular current together, while p stays unflattened. The remaining tail is at or below the background and the basis scale. Whether the drive keeps falling or stops at a basis-set floor needs a resolution ladder at fixed alpha.

---

## 7. Caveats and known issues

1. **Defaults are not the recipe.** `mf_fixed.py` defaults to `--model hvp --precond hj --lanczos-stop both`, which did not converge alpha = 0.05. Use Section 4.
2. **The stationarity flag can be fooled by a collapsed trust radius.** `FixedSolver` counts an inner solve as stationary if it is "flat" with the bound active (`stationary = status == "stationary" or (flat and c_F >= -0.01)`). fxR48b005's last outer is flat at Delta ~2e-7 with \|grad\| 0.21 (GN share 0.99, ratios ~1), so it reports `stationary: True` but is not a stationary point. Treat alpha = 0.005 as unconverged. A fix: do not let "flat" count when the radius has collapsed, as `meanforce` already does for its own flat test.
3. **alpha <= 0.01 is not converged.** fxA01 stalled near M/eps 1.02. fxR48b005 reached the bound but is not stationary. Trends quoted at these alphas are indicative.
4. **Fixed boundary does not fix the current profile.** iota drifts (Section 6.2). Separating resolution effects from profile drift needs a run with <J.B> or iota pinned.
5. **`--beta-adapt` was on in the 48/32/18 and fxA01 runs**, but not in the converged 32/32/24 alpha = 0.05 run.
6. **One target, one boundary.** All results use J_fit of the input DESC equilibrium and its LCFS (not the coil-consistent free-boundary LCFS).
7. Diagnostics at alpha = 0.005 mostly use the after-outer-1 snapshot, not the final state.

## 8. Open work

1. Resolution ladder at fixed alpha (radial L first) to see whether the resonant drive and the tail keep falling or reach a floor, and whether flattening appears.
2. Pin the current profile (<J.B> or iota) to separate drift from shielding.
3. Fix the stationarity test (7.2); then finish alpha 0.01 and 0.005.
4. Optional solver work: Ritz-pair recycling across Lanczos solves; beta-projection in the force metric.
5. Port the solver rules (beta model, force metric, LS multiplier, complementarity stop) to the free-boundary path.
