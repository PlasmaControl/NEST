# Energy objective with a flux-surface boundary in NEST: build, validation and run plan

Status: approved 2026-11-17 with alpha = 0.05, no unconstrained-energy contrast arm, first runs on Modal. Build done on branch `energy-objective`; G0 passed; its results table is section 2 of the status note energy_objective_status_2026-11-18.md (also docs/ in the repo), with raw JSON in results/energy_objective/g0/.
Code references are to branch `ripple-landscape`, commit 6249d4e (worktree NEST-cp).

## 1. Goal and problem statement

We want to test how well-behaved the optimization is, and what it selects, when magnetic energy enters the objective. Energy should prefer reconnected (island) states over shielded ones, so the question is whether the optimizer actually moves from the shielded branch to the island branch, and whether it does so smoothly.

The problem solved at each rung is

    minimize    f(u) = (1/2) |g(u)|^2  +  lambda * [W_Omega(u) - W_Omega(u_fit)] / W_DESC
    subject to  M(u) <= alpha * M_DESC                      (mean force bound, PHR augmented Lagrangian, as now)
                (B . n)(u) = 0 on the LCFS                   (NEW: flux-surface boundary, exact linear constraint)
                Phi_tor(u) = Phi_tor(u_fit)                  (existing --pin-flux)
                I_p fixed                                    (built into the J = curl T representation)

Definitions:
- u: current-potential coefficients; J(u) = curl T(u).
- g = sw (J - J_fit) / rms(J_fit), with sw_i = sqrt(dV_i / V). So (1/2)|g|^2 is the existing J - J_fit term (called phi in the logs).
- W_Omega = int_Omega B^2 / (2 mu0) dV, evaluated on the collocation grid as (V / 2 mu0) sum_i w_i |B_i|^2. W_DESC = W_Omega(u_fit) is the normalization.
- B = B_ext + B_plasma(u). In option A this equals B_DESC + dB, with dB = B_ann[J(u) - J_fit] = `fields.B_plasma(u - u_fit, 0)`.

Why these constraints: with B . n = 0 on the LCFS, I_p fixed and Phi_tor fixed, the field outside the plasma is the same for every admissible u. So no external circuit does work between admissible states, and the free energy W_total - sum_c I_c Phi_c equals W_Omega + const. That is the derivation agreed in the previous message. Without the B . n constraint, W_Omega is not the free energy; the missing exterior term is estimated at about 1e-4 W, comparable to the island energy difference, so the constraint matters.

The target is homogeneous. In option A the start field is DESC's, so B_DESC . n = 0 to DESC accuracy. The constraint is therefore dB . n = 0, i.e. C (u - u_fit) = 0, and u_fit satisfies it exactly.

Pressure is unchanged: it is still eliminated by least squares at every state, and thermal energy is not in the objective, because NEST's least-squares p carries no mass or entropy constraint.

## 2. Build

| ID | Item | Where | Notes |
|---|---|---|---|
| B1 | Field on the LCFS | `objective.Fields` / new helper | A second `CylindricalInterpolator` for a rho = 1 shell on the collocation zeta planes (n_theta_b x n_zeta points), sharing `cyl_planes(u, I)` with the collocation points. Restructure `_state_impl` so the plane field is computed once per evaluation and interpolated to both point sets. Geometry at rho = 1 (sqrt(g), e^rho) from the DESC equilibrium. |
| B2 | B . n constraint rows | `LinearConstraints` (new block) | Fourier projection of sqrt(g) dB^rho at rho = 1 onto stellarator-symmetric modes with abs(m) <= M_c, abs(n) <= N_c (flag `--bn-modes M_c N_c`, default the basis M, N). k is about M_c (2 N_c + 1): about 520 at 40/40/6. Rows built by batched vjp of u -> b_mn (one transpose per batch). The (0,0) mode is identically zero (div B); the existing QR rank cut in `ProjectedMetric` drops it. `restore()` maps a start onto the constraint with the smallest metric-norm change. `violation()` reports the max constrained mode and the rms of the unconstrained remainder, both as a fraction of rms abs(B). |
| B3 | Energy term | `_state_impl`, `Weights.merit`, rows | Flag `--energy-weight lambda`. State returns dW = (V / 2 mu0) sum w (2 B_fit . dB + abs(dB)^2), computed in this cross-term form so there is no cancellation between two totals of about 1.9 MJ. Merit f = phi + E with E = lambda dW / W_DESC; phi and E are logged separately. Rows r_E = c sw B with c = sqrt(lambda V / (mu0 W_DESC)), so that (1/2)abs(r_E)^2 = lambda W_Omega / W_DESC: exact Gauss-Newton rows (linear residual). J: c sw dB(v); J^T by the existing linear transpose. |
| B4 | Thread energy through every row and merit site | `meanforce_fixed.py`: FixedRows, FixedRowsB, FixedRowsH (exact Hessian gets + c^2 K^T W K), `ls_multiplier` (gradient of phi must include E); `meanforce.py`: merit (~352), rows (~387), subspace/accel (~610-634), ~973, ~1066 | Geodesic acceleration: energy rows are linear, so their second-order term is 0. The inner "flat" stop test reads phi; switch it to phi + E. |
| B5 | Diagnostics | `mf_fixed.py` `[diag]` line and run JSON | E, dW / W_DESC, a split of dW into the n = 0 and n != 0 coefficient blocks of u - u_fit (each block's energy plus their cross term), B . n (constrained max, remainder rms), dPhi_tor / Phi_tor, b21 / b31 / b41 (amplitude and phase), phi, M / M_DESC, nu_F, Lanczos iterations per step. Per-step streaming as now (`--diag-every`). |
| B6 | Eval-only checks | `scripts/meanforce/` | (a) energy and B . n for any saved state; (b) `exterior_check.py`: direct Biot-Savart of dJ on shells outside the LCFS (rho_ext = 1.02, 1.1, 1.5 in a DESC-extended or offset-surface parameterization, plus far points), and the exterior energy change on a shell grid. Extends `bn_check.py`'s direct Biot-Savart. |
| B7 | Metric (contingency only) | force metric | Not changed at first. The energy Hessian adds a few large eigenvalues in low-order modes: an order-of-magnitude estimate (lambda V mu0 a^2 rms(J)^2 / W_DESC, with rms(J) about 4e4 A/m^2 assumed) gives about 0.15 lambda x H_J for the lowest modes, falling like 1/(k a)^2. Trigger: Lanczos iterations per step more than 2x the lambda = 0 rung. Remedy: a rank-r (r about 50-200) randomized eigen-approximation of H_E added to the metric. |

## 3. Validation gates

Every gate states what it tests, its pass band, and what could make it pass or fail for the wrong reason.

### G0: code correctness (CPU, laptop; tiny basis, e.g. 12/12/2 on the L32 ripple file, coarse field grid)

| Check | Pass band | Wrong-reason risks and cover |
|---|---|---|
| G0.1 Adjoints: <C v, w> = <v, C^T w> for the B . n map; same for the energy rows' J / J^T | rel. err <= 1e-12 | A transpose that is right only for symmetric inputs: use random v, w with no symmetry. |
| G0.2 Gradient: full merit (phi + E + AL) vs central differences along 5 random directions | rel. err <= 1e-7 | Step-size noise hiding errors: the merit is quadratic in E and phi, so also check the exact 3-point quadratic identity. |
| G0.3 Exact quadratic: E(u + t d) from 3 points predicts a 4th | <= 1e-10 relative to the E change | None significant. |
| G0.4 dW cross-term form vs difference of totals | agree to the totals' rounding (about 1e-16 x 1.9 MJ x sqrt(n)); cross form stays accurate at abs(du) 1e-8 | Confirms the cancellation fix rather than assuming it. |
| G0.5 Constraint keeping: 50 random projected steps keep abs(C (u - u_fit)) at rounding; `restore()` lands on C u = d | <= 1e-12 relative to abs(C) abs(u) | Rank cut dropping a real row: log the rank and the dropped singular values. |
| G0.6 Regression: lambda = 0, no new constraint, reproduces the current code's trajectory (default flags and `--subspace prev,accel`) | identical to rounding over 20 steps | This passes even if B4 misses a site, because energy rows are zero at lambda = 0. G0.7 covers that. |
| G0.7 Closed form: force bound off, lambda > 0, B . n and flux constraints on. The problem is then an equality-constrained quadratic; solve it with dense linear algebra and compare with the solver's end state, once with default flags and once with `--subspace prev,accel` | abs(u - u*) / abs(u*) <= 1e-8 for both | This is the test that catches a missing energy row in the subspace/accel path, which G0.6 cannot. |

### G1: physics of the boundary condition (GPU, 40/40/6 a/100, the run setting)

| Check | Pass band | Wrong-reason risks and cover |
|---|---|---|
| G1.1 Quadrature: W_Omega(u_fit) vs DESC's W_B (1.927469 MJ, from the energy_helicity job) | rel. err <= 1e-5 | Fails falsely if the collocation grid under-resolves B^2 radially: if so, compare at `--colloc-rho` 3. |
| G1.2 (also run with the cutoff doubled, `--bn-modes 80 12`: at the 8/8/2 test basis the constrained modes sat at 2e-17 while the full B . n on the LCFS points had rms 7e-5 of scale, i.e. geometry couples basis modes into boundary harmonics above the basis M, N) Constraint vs true B . n: for 10 random constrained perturbations, B . n from direct Biot-Savart on the LCFS | true rms(B . n) / rms abs(B) at least 10x below the unconstrained value of the same abs(dJ), and below 1e-5 | The annulus says 0 by construction, so only direct Biot-Savart is a test. Residual = field-grid error plus truncated modes: report both separately (higher M_c, N_c vs finer field grid). |
| G1.3 Exterior silence, the key check: for constrained vs unconstrained dJ of equal norm, exterior field change on shells 1.02, 1.1, 1.5 and far points, and dW_total vs dW_Omega | constrained abs(dB_out) at least 10x below unconstrained at every shell; abs(dW_total - dW_Omega) <= 0.1 x abs(dW_island) (taken from G2) | False pass from checking only far shells, where every field has decayed: the 1.02 shell is the decisive one. False pass from low-order dJ only: include the end states of G2 as test perturbations. TF-linked flux: confirm dPhi_tor = 0 in every cross-section, not only on average. |
| G1.4 Constraint cost: restore the R1f shielded state onto the constraint and evaluate M | report only, no band | The restore may raise M by a lot. That is not a failure, but it sets how much solving G2.1 has to do. |

### G2: reference states under the new constraint (GPU, 40/40/6 a/100, 98 radii, alpha 0.05, lambda = 0)

| Run | Purpose |
|---|---|
| G2.1 S0: the R1f shielded state, restored, solved at lambda = 0 | Shielded baseline under the consistent boundary condition: W_Omega, phi, M, b21. |
| G2.2 I(beta): `--pin-b21 beta` for beta = 0.25, 0.5, 1.0 (beta = 1 is a vacuum-size island), lambda = 0 | Energy and phi along the island path, under the consistent energy. |
| G2.3 Fine-grid check: evaluate S0 and I(beta) at a/120, 120 radii | dW(beta) must agree with the solve grid to within 20%; otherwise the ladder needs a finer grid. Compare states only on this common grid. |

**Decision point after G2.** The reduced 1-D model along the island path is

    F(beta; lambda) = phi(beta) + lambda * dW(beta) / W_DESC.

It predicts the lambda at which the minimum leaves beta = 0 (lambda* = dphi / (-dW / W_DESC) at the relevant beta), and whether the move is continuous (F convex in beta) or a jump with possible hysteresis (F with two minima).
- The old estimate (lambda about 75) used the inconsistent energy and mixed resolutions, so G2 replaces it.
- If dW(beta) >= 0 under the consistent energy, energy does not favour the island at this setting. That is itself the main result; we stop and discuss before running G3.

### G3: lambda ladder (main experiment; GPU, 40/40/6 a/100, alpha 0.05)

- **Main arm:** B . n = 0 + `--pin-flux`. lambda in lambda* x {0, 0.1, 0.3, 1, 3, 10}, warm-started up from S0; then back down from the top state to 0.
- **Control arm:** add `--pin-current` (enclosed toroidal current on every DESC surface fixed). Up-ladder only. This separates helical changes from axisymmetric profile changes; it also forbids part of the island's own flattening current, so it is a control, not the main result.
- (The optional unconstrained-energy contrast arm was dropped by decision, 2026-11-17.)

"Well behaved" is defined in advance as:
1. Each rung meets the outer stop (feasible, stationary, complementarity) within its budget.
2. Across rungs, phi is nondecreasing and dW nonincreasing in lambda. This holds for global minimizers; a violation means an unconverged rung or a branch jump, and is flagged, not averaged away.
3. Lanczos iterations per step are within 2x the lambda = 0 rung (otherwise B7).
4. Constraint residuals stay at rounding; true B . n (direct Biot-Savart) stays within the G1.2 band at the end of each rung.
5. Up and down ladders agree, or the hysteresis is measured: the lambda range and both states' phi, dW, M.

Outcome observables: b21 amplitude and phase (shielded -90 deg to island +90 deg), traced island width (Poincare, `compare_states.py`) at rungs where b21 flips, the dW split n = 0 vs n != 0, the iota shift, q = 3 / q = 4 harmonics (b31, b41), and how the force is distributed (q = 2 band, q = 3 band, edge).

Wrong-reason risks for G3:
- An "island" appearing at large lambda because the force bound went slack: check complementarity and that M is still at the bound.
- Energy lowered by axisymmetric reshaping only (dW dominated by the n = 0 block) with b21 unchanged: the control arm and the dW split expose this.
- A flip caused by field-grid error: re-evaluate flipped states on the G2.3 fine grid.
- Comparing states from different grids: use the common reference grid only.

### G4: 64/64/6 (only if G3 is clean)

Repeat the main arm at the rungs around the transition, from the 40/40/6 states carried up (`--start-LMN`). This asks whether the transition lambda and the island width converge with resolution.

## 4. Resources (estimates, to be revised after timing the first run)

| Stage | Where | Estimate |
|---|---|---|
| G0 | laptop CPU | no GPU |
| G1 | 1 x A100-80GB | about 1-2 GPU-h (setup about 10 min plus constraint rows; direct Biot-Savart shells) |
| G2 | A100-80GB | 4 solves of about 1 h plus fine evals: about 5 GPU-h |
| G3 | A100-80GB | about 15 rungs (6 up, 5 down, 5 control) of about 0.5-1 h: about 8-15 GPU-h |
| G4 | H200 or A100-80GB | to be sized from G3 (64/64/6 runs were about 3-20 s per step) |

Basis: R1 (40/40/6) runs took about 7.5 min setup and 3-9 s per step; a/100 evaluation needs 80 GB unless `--locate-chunk 131072` is used. All runs stream per-step logs with `--diag-every`.

## 5. Decisions (2026-11-17)

1. alpha = 0.05 for G2 and G3.
2. B . n truncation: G1 runs both (40, 6) and (80, 12); the G2/G3 runs use whichever G1.2 shows is needed.
3. No unconstrained-energy contrast arm for now.
4. First runs on Modal (scripts/modal_fixed/energy/job_g12.sh).
5. J - J_fit stays a weighted objective for now; the constrained form can be added later as a second AL constraint.
