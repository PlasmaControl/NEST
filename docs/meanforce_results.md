# Results so far: mean-force bound with the Poisson pressure on the NEST free boundary (NCSX_fb_L16M16N12)

2026-10-27. Snapshot while six jobs are still running (see `docs/meanforce_handoff.md` for what is pending and how to continue).
Problem, code and commands: `docs/meanforce_handoff.md`; plan: `docs/mean_force_poisson_plan.md`; first-run health review:
`docs/meanforce_gltr_a40.md`. Raw data: `results/meanforce/` (JSON per run, saved states, condensed logs).
Tables and figures regenerate with `scripts/meanforce/collect.py`.

All distances below are |J − J_fit| / rms J_fit (volume rms) unless stated; alpha is the mean-force bound in units of DESC's
M_DESC = <|F|>_vol / norm = 8.7788e-4. Every finished bound meets M = alpha M_DESC, B·n/|B| rms ≤ 4.53e-3 (the annulus value
agrees with direct Biot-Savart to 7e-5 rms at the start and the end states) and Ψ within its 1e-2 tolerance (below 1e-3 for most points), except the points marked
(!) (see "Caveats").

## Start

NEST at 16/16/12 with the Poisson pressure starts at M = 5.74 M_DESC (rms |F| 8.7e-3), B·n/|B| 4.58e-3 (DESC 4.53e-3),
Ψ 0.497005 (DESC 0.497070). Gates: DESC's own |F| on NEST's points reproduces DESC's quadrature to 0.19%; all unit gates pass
(`tests/test_meanforce.py`).

## The front at a/40, both solvers (study 1 and 2 baseline)

| alpha | 0.7 | 0.5 | 0.3 | 0.2 | 0.1 | 0.05 |
|---|---|---|---|---|---|---|
| GLTR | 1.4868e-2 | 1.8907e-2 | 2.7739e-2 | 3.6434e-2 | 5.3919e-2 | 1.1279e-1 |
| QR | 1.4882e-2 | 1.8917e-2 | 2.7737e-2 | 3.6501e-2 | 5.4068e-2 | 1.1414e-1 / 1.1605e-1 (redo) |

The solvers agree to 0.01–0.3% down to alpha 0.1. At alpha 0.05 the two end in different local minima: GLTR's has B·n at 0.52 β
(2.36e-3), QR's at its bound. Hessians of the Lagrangian on the tangent space of the active constraints, h = 1e-9: all positive
for a/40 GLTR at all six bounds (smallest eigenvalue 1.08, 3.6, 7.9, 15.6, 154, 7.4e3), QR alpha 0.1: +169, alpha 0.05: +9.3e3.
Earlier negative values were finite-difference error (h = 1e-6).

## Study 1: annulus spacing (NEST 16/16/12)

| run | 0.7 | 0.5 | 0.3 | 0.2 | 0.1 | 0.05 |
|---|---|---|---|---|---|---|
| a/20 GLTR | 1.4945e-2 | 1.9127e-2 | 2.8746e-2 | 3.8719e-2 | 6.4190e-2 | 2.0743e-1 (!) |
| a/40 GLTR | 1.4868e-2 | 1.8907e-2 | 2.7739e-2 | 3.6434e-2 | 5.3919e-2 | 1.1279e-1 |
| a/60 GLTR | 1.4863e-2 | 1.8895e-2 | 2.7683e-2 | 3.6314e-2 | 5.3502e-2 | running |
| a/60 QR | 1.4875e-2 | 1.8905e-2 | 2.7693e-2 | 3.6416e-2 | 5.3613e-2 | 1.1225e-1 |
| a/80 GLTR | 1.4862e-2 | 1.8893e-2 | 2.7681e-2 | 3.6305e-2 | running | |
| a/80 QR | 1.4876e-2 | 1.8912e-2 | 2.7688e-2 | running | | |
| a/100 GLTR (from a/40 state) | | 1.8895e-2 | | | 5.3523e-2 | running |
| a/120 GLTR (from a/40 state) | | 1.8895e-2 | | | 5.3523e-2 | running |

- The annulus is converged by a/60: a/60, a/100 and a/120 agree to 0.05% (α = 0.1) and four digits (α = 0.5). a/40 is 0.2–0.8% high at α ≤ 0.3 and
  ≤ 0.1% at α ≥ 0.5. a/20 is 0.5% high at α = 0.7 and 20% high at α = 0.1.
- a/20 at alpha 0.05 (0.196–0.207, both solvers) is a discretization artifact: those states have M = 0.0447 (a/40) and 0.0446 (a/60)
  when solved to 0.0500 on a/20, the a/20 mean force being inflated by about 0.005 M_DESC. The a/40 alpha 0.05 state re-evaluated on
  a/60 gives M = 0.0496, B·n 4.458e-3, the same J.
- Cost per bound at a/40: GLTR 17–50 min, QR 10–30 min (13 s per step); a/80 QR needs ~20 min per re-anchor for the tables.

## Study 2: NEST basis (a/40, GLTR)

Own fit (each basis fits DESC's J itself, so the numbers are NOT directly comparable at the loose bounds):

| alpha | 0.7 | 0.5 | 0.3 | 0.2 | 0.1 | 0.05 |
|---|---|---|---|---|---|---|
| 16/16/12 | 1.4868e-2 | 1.8907e-2 | 2.7739e-2 | 3.6434e-2 | 5.3919e-2 | 1.1279e-1 |
| 20/20/12 | 1.3829e-2 | 1.8150e-2 | 2.6376e-2 | 3.3618e-2 | 4.6431e-2 | 6.3669e-2 |
| 24/24/12 | 1.4974e-2 | 1.9157e-2 | 2.7183e-2 | 3.4097e-2 | 4.6013e-2 | 5.9462e-2 |
| 28/28/12 | 1.6667e-2 | 2.0623e-2 | 2.8363e-2 | 3.5100e-2 | 4.6733e-2 | 5.9082e-2 |
| 16/16/16 | 1.5123e-2 | 1.9121e-2 | 2.7832e-2 | 3.6365e-2 | 5.2987e-2 | 1.0401e-1 (!) |

On one apparatus (`eval_common.py`: L = M = 28's collocation grid, a/40 annulus, 28 pressure space, fixed target J_DESC),
|J − J_DESC| / rms J_DESC (M/M_DESC of the state there in parentheses):

| alpha | 16/16/12 | L = M = 20 | L = M = 24 | L = M = 28 |
|---|---|---|---|---|
| 0.7 | 4.163e-2 (0.666) | 3.666e-2 (0.690) | 3.455e-2 (0.700) | 3.401e-2 (0.700) |
| 0.5 | 4.346e-2 (0.474) | 3.872e-2 (0.495) | 3.648e-2 (0.502) | 3.582e-2 (0.500) |
| 0.3 | 4.833e-2 (0.281) | 4.346e-2 (0.298) | 4.126e-2 (0.303) | 4.043e-2 (0.300) |
| 0.2 | 5.399e-2 (0.185) | 4.827e-2 (0.199) | 4.599e-2 (0.203) | 4.508e-2 (0.200) |
| 0.1 | 6.700e-2 (0.091) | 5.771e-2 (0.099) | 5.491e-2 (0.102) | 5.375e-2 (0.100) |
| 0.05 | 1.199e-1 (0.047) | 7.19e-2 (0.049) | 6.59e-2 (0.050) | 6.36e-2 (0.050) |

- Monotone in the basis at every bound once the target is fixed; converging: the step 24 → 28 is 1.6% (alpha 0.7) to 3.4% (alpha 0.05), against 6–8% for 20 → 24.
  The radial and poloidal resolution (L, M) limits the tight end of the front; L = M = 24 is nearly enough.
- N = 12 → 16 changes the own-fit distance by ≤ 1.7% for alpha ≥ 0.1 (NFP = 3, so N = 12 already reaches toroidal mode 36). The 16/16/16
  alpha 0.05 point is unconverged (below) and is being redone (job 6540).
- Hessians at h = 1e-9 are positive for L = M 20, 24, 28 at alpha 0.1 (smallest 293, 327, 331) and 0.05 (5.4e3, 1.9e3, 2.1e3).

## Where the force error is (a/40, alpha 0.7; `docs/meanforce_gltr_a40.md`, `figures/meanforce/radial_a40.png`)

Against NEST's start the force dropped 8–23× in every ρ range. Against DESC: 2.0–2.4× lower for ρ < 0.6, equal at ρ = 0.6–0.95
(where the iota = 1/2 and 3/5 rational surfaces sit), 2–5× lower for ρ > 0.95; 86% of the gain over DESC's M comes from ρ > 0.95 (8% of the
volume). Part of the edge advantage is structural (J = 0 and p = 0 on the LCFS).

## Optimization health (details in `docs/meanforce_gltr_a40.md`, UPDATES.md)

Trust radius, ratios, Gauss-Newton share, multipliers, Galerkin identity (≤ 5e-13), re-anchoring (continuous to 3e-4 in M at ~2 mm) were
reviewed with the user at alpha 0.7. Bugs found and fixed along the way: QR's undamped Gauss-Newton step on a near-singular matrix
(guard), a bound sitting at c = 0 with a zero multiplier hiding its curvature from the step model (curvature kept within 5% of a bound),
QR's table error after the boundary moves (forced re-anchor), the "flat" stop mistaking a stall for convergence (now needs an active
mean-force bound), penalty escalation within tolerance, device-memory limits of the metric at L = M = 28 and the xi columns at a/80.

## Caveats and corrections

- (!) 16/16/16 alpha 0.05: ended at M = 0.0464 (bound 0.05), B·n 0.5 β, objective still falling: not converged.
- (!) a/20 alpha 0.05 and (a/20 QR alpha 0.1, M = 0.0988): see Study 1.
- At alpha 0.05 the front is steep and has several local minima (GLTR vs QR at a/40; 3% apart).
- Hessians stored in the JSONs of runs started before 2026-10-27 used h = 1e-6 and show spurious negative Ritz values; the verified ones are listed above.
- Study 2 at L = M ≥ 24 has no QR comparison (memory).

## Figures and tables

`figures/meanforce/`: `study_front.png` (all runs), `study_spacing.png`, `study_basis.png`, `study_common.png`, `study_table.md` (every run, every bound),
`study_common_table.md`, `health_gltr.png`, `health_qr.png`, `radial_a40.png`.
