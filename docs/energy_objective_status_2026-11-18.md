# Energy objective: build, G0, G1 and the first G2 run (status, 2026-11-18)

Branch `energy-objective` in NEST-cp: `d7948ed` (build), `84d8ffa` (flag fix), `03ad6e9` (direct-row option).
Plan: `docs/energy_objective_plan.md`. Raw outputs: `results/energy_objective/` (g0/ on CPU, gpu/ from Modal).
GPU used: 463 + 473 + 3325 s (+ two 8 s failed starts) on A100-80GB, about 1.2 GPU-h.

## 1. What was built

- `--energy-weight lambda` adds E = lambda [W_Omega(u) - W_Omega(J_fit)] / W_Omega(J_fit) to the objective.
  - W_Omega = (V / 2 mu0) sum_i w_i |B_i|^2 on the collocation points.
  - It enters as exact Gauss-Newton rows e_c sqrt(w) B, appended to the misfit vector g. So every path that uses g (merit, trust-region model, subspace search, acceleration, multiplier estimate) picks it up with no separate code.
  - The merit evaluates the energy part offset-free: sum g_ref (g_E - g_ref) + |g_E - g_ref|^2 / 2.
- `--bn-modes MC NC` adds an exact linear constraint: the Fourier modes |m| <= MC, |n| <= NC of sqrt(g) dB . e^rho on the LCFS vanish.
  - dB = B_ann[J - J_fit]; option A's start field is DESC's, so the constraint is homogeneous.
  - The LCFS points lie on the collocation planes, so the expensive plane field is shared.
- Diagnostics (in `[diag]`, the run JSON and `--eval-only`):
  - dW / W, split by the toroidal mode number of du into n = 0, n != 0 and cross terms;
  - E;
  - B.n on the LCFS from a real field evaluation.

## 2. G0: code correctness (CPU, 8/8/2 basis on the L32N4 ripple file)

| check | result | band |
|---|---|---|
| W_Omega(J_fit) vs DESC's W_B (1.927471 MJ) | -1.3e-10 relative | <= 1e-5 |
| adjoints: rows with energy (FixedRows, FixedRowsB), B.n modes | 1.5e-14, 3.0e-15, 3.9e-16 | <= 1e-12 |
| LCFS interpolator rebuilt on the collocation points vs `B_plasma` | 0.0 (exact) | - |
| energy gradient vs central differences, force bound off (exactly quadratic) | <= 1.6e-10 | <= 1e-7 |
| full-merit gradient (force bound active), best over h | 2.5e-9 to 5.6e-7 | O(h^2) to a ~1e-7 floor from the abs(F) terms |
| E(u + t d) exactly quadratic (4th point from 3) | 4.5e-13 | <= 1e-10 |
| cross-term energy vs lambda dW/W | 2.1e-14 (du ~ 1e-2), 1.8e-10 (1e-6) | - |
| constraint kept by 50 projected steps; restore | 3.5e-18; 8.7e-19 | rounding |
| lambda = 0, no new constraint: 20 steps vs base commit 6249d4e | identical merits and end state (default and --subspace prev,accel) | identical |
| closed-form constrained optimum (force bound off, lambda 30) vs solver end state | 1.7e-9 relative in u, for default flags and --subspace prev (22 subspace steps) | <= 1e-8 |

Two notes:
- The first closed-form comparison disagreed (2.6e-3 relative). The cause was the check script building `Fields` with a different interpolation order than the driver, not the solver.
- With the force bound off, `--subspace accel` raises inside the acceleration Lanczos, because its force-curvature input is exactly zero. This is an artifact of the force-off test only. A force-active energy run with `--subspace prev,accel` had no failures.

## 3. G1: does the B.n constraint silence the field outside? (GPU, 40/40/6, a/100, 98 radii)

The field change is computed by direct Biot-Savart quadrature of dJ, not by the annulus solver.
- Field values are rms|dB| / rms|B_LCFS|.
- "Random" means a random du with |J - J_fit| / rms = 1e-2.
- The last column is an estimate (no sign cancellation), relative to W.

| perturbation | true B.n | 0.02a | 0.1a | 0.5a | far | exterior linear-term estimate / W |
|---|---|---|---|---|---|---|
| random, unconstrained | 4.3e-5 | 4.2e-5 | 1.6e-5 | 1.6e-6 | 5.1e-7 | 2.5e-5 |
| random, constrained, cutoff 40/6 | 7.6e-6 | 7.4e-6 | 4.8e-7 | 4.9e-8 | 8.6e-9 | 2.4e-6 |
| random, constrained, cutoff 80/12 | 7.0e-6 | 9.2e-6 | 1.0e-6 | 2.7e-7 | 1.7e-7 | 3.5e-6 |
| old R1f island - shielded (unconstrained) | 1.9e-4 | 2.3e-4 | 1.7e-4 | 6.9e-5 | 4.8e-5 | 2.5e-4 |

What the table says:
1. **The old states:** the island-vs-shielded difference changes B.n on the LCFS by 1.9e-4 of |B|. (The last column ignores sign cancellation and overstates the energy error; section 6 replaces it with a per-mode calculation: about 10% of the island energy gap, not the 4x first quoted.)
2. **The constraint at 40/6** removes the mid and far field 30-60x (0.1a, 0.5a, far), but the boundary and 0.02a values only about 6x.
3. **Doubling the cutoff to 80/12 does not help:**
   - it empties the annulus B.n above the cutoff (1.3e-5 down to 3.7e-7), but the true B.n stays at 7e-6, equal to the annulus-vs-direct difference (field-grid error);
   - it makes the far field 20x worse, because constraining harmonics the annulus gets wrong forces J to fit grid error.
4. **The near-field values are measurement-limited.** The direct quadrature's toroidal source spacing (about 17 cm at q = 4, 16 planes) is much larger than the 0.02a offset (2.4 cm). The mid and far-field values are the reliable ones.
5. **Direct-Biot-Savart rows** (code in `03ad6e9`; transpose identity, rows match the forward quadrature to 1e-10) do not help on the 8/8/2 test. They zero the sampled B.n (to 4e-10) but leave a larger far field than annulus rows (3.9e-7 vs 2.7e-8).

Decision taken: keep the annulus rows at cutoff 40/6 (the basis M, N).

## 4. G2.1: S0, the shielded state under the constraint (lambda = 0, alpha 0.05)

| state | M / M_DESC | abs(J - J_fit) / rms | b21 / vac (phase) |
|---|---|---|---|
| J_fit (satisfies the constraint exactly) | 7.18 | 0 | - |
| R1f_S, unconstrained (start) | 0.0616 | 0.0127 | 1.8% (-90) |
| R1f_S restored onto B.n + flux | 34.0 | 0.0126 | 18% (-90) |
| S0 after 5 accepted steps | 0.636 | 0.0048 | 2.2% (-90) |
| S0 at the 3300 s budget (91 steps) | **0.5766** | 0.0072 | 2.4% (-90) |

- **The constraint raises the force about 10x, at least within this budget.** After the first few steps M/eps crept from 11.56 to 11.53 over about 80 accepted steps. Steps were still accepted with good model agreement (ratios 1.2-1.7), but each gained very little. One hour cannot separate a higher floor from a slower valley.
- **Where the force sits:** spread over the volume (outside the q = 2 band: 0.58x DESC's mean), with the q = 2 band p16 at 1.9x DESC and the band max at 2.3x DESC.
- **The state is still shielded** (b21 -90 deg, 2.4% of vacuum). dW / W is -9.4e-8 relative to J_fit (about -0.2 J), almost all in the n = 0 block.
- **Reading:** the unconstrained R1f state has true B.n of 2.6e-5 |B| on the LCFS. The old fixed-boundary states lowered their force partly by moving magnetic flux through the LCFS, which a fixed-boundary (flux-conserving) problem does not allow.

## 5. Open questions (for the user)

1. **Is the S0 floor real?** Two cheap tests, each about 1 GPU-h:
   - (a) continue S0 for 2-3 h;
   - (b) a B.n cutoff scan at lambda = 0 (e.g. 4/2, 8/4, 16/6, 40/6) from R1f_S, mapping force against cutoff. The physical argument for a low cutoff: the linear exterior term int B_coil . dB_out couples only to the smooth, large-scale coil field, so high-m B.n contributes little to the free energy, while low-m B.n reaches far and couples strongly. The G1 table agrees: the constraint removes the far field well.
2. Only after that: the island references I(beta) at the chosen cutoff, then the lambda ladder.


## 6. B.n-leak study (2026-11-18; job_leak.sh, commit 2584655)

### Test 3: leak spectrum and energy error (bn_spectrum.py, Modal b016d3a5)

To first order, using W_Omega with a leak misstates the free energy by -(1/mu0) oint phi_B (dB . n) dS, where phi_B is the surface potential of B on the LCFS. phi_B is concentrated at m = 1-3 (|a| 5-8e-3), 5e-4 at m = 5-8, 1e-5 at m = 9-16.

Leak in the table below is the rms of sqrt(g) dB . e^rho over rms(sqrt(g) B^zeta); all other columns are relative to W.

| state | leak (direct) | dW_Omega / W | energy error / W | error left above 4/2 | above 8/4 |
|---|---|---|---|---|---|
| R1f_S (unconstrained) | 7.8e-5 | +1.0e-6 | -4.7e-7 | -4.4e-9 | +4e-11 |
| R1f_b1 island (unconstrained) | 4.6e-4 | -6.5e-5 | +5.8e-6 | -2.7e-7 | -8.5e-10 |
| S0 (constrained 40/6) | 6.3e-7 | -9.4e-8 | +3.9e-9 | -2.6e-11 | -1e-12 |

- The energy-relevant leak sits in (m, n) = (1,0), (2,0), (3,0): an axisymmetric shift, elongation and triangularity of the boundary flux, amounting to an effective displacement of about 0.1 mm.
- The leak moves the old island-vs-shielded energy comparison by about 10%.
- A 4/2 cutoff removes all but 0.5% of that error.

### Tests 1 and 2: force floor against constraint

Setup: floor pushes from J_fit (M = 7.18 M_DESC), alpha 0.001, one outer, same flags. Every arm ended on the solver's stationarity test.

| arm | constraint | M / M_DESC | abs(J - J_fit) / rms | b21 / vac | band p16 / DESC | outside mean / DESC |
|---|---|---|---|---|---|---|
| A0 | none | **0.0623** | 0.0165 | 2.0% | 0.40 | 0.056 |
| Aflux | toroidal flux | **0.0633** | 0.0150 | 2.3% | 0.45 | 0.056 |
| A4 | flux + B.n 4/2 (22 modes) | **0.3475** | 0.0327 | 4.6% | 0.96 | 0.34 |
| A8 | flux + B.n 8/4 | **0.4771** | 0.0270 | 2.2% | 1.19 | 0.48 |
| A16 | flux + B.n 16/6 | **0.5448** | 0.0057 | 2.4% | 2.08 | 0.54 |
| A40 | flux + B.n 40/6 | **0.5766** | 0.0057 | 2.4% | 1.90 | 0.58 |

All states are shielded (b21 phase -90 deg). A40 equals S0 (0.5766), reached from a different start, so this is a real floor of the constrained problem at 40/40/6, not a slow valley. Notes:
- A4's container was lost after the run completed. Its outputs are on the Modal volume (/results/fxr/energy/leak/A4/, /results/fxr/lk_A4/).
- The other arms' outputs are in results/energy_objective/gpu/leak/.

### Reading

- **The flux pin costs nothing** (1.6%).
- **Forbidding boundary flux leak raises the floor 5.6x with only the 22 lowest modes, and 9.3x with all modes to 40/6.**
- **Where the extra force goes:** it spreads over the whole volume (outside-band mean 0.34-0.58 of DESC's, against 0.056 unconstrained), and the q = 2 band rises to 1-2x DESC.
- **Conclusion:** B.n leak through the LCFS is a major channel by which the fixed-boundary NEST runs lowered their force. The earlier NCSX check could not see it, because it compared total B.n levels dominated by J_fit's own error rather than the counterfactual cost.
- **Hypothesis, not tested:** NEST's basis forces J -> 0 at the LCFS, which removes the edge currents that would otherwise cancel the exterior field of interior current changes at low force cost.

### Decision (user, 2026-11-18)

- Take the leak channel seriously. The higher-resolution reruns with the B.n constraint go to a separate session.
- Here: keep the constraint and do the energy-objective work at higher force-balance levels.
- Proposed, awaiting confirmation: cutoff 40/6 (S0 is the shielded reference) and alpha 0.8, about 1.4x the floor.
