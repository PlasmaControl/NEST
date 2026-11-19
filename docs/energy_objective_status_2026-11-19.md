# Energy objective, status 2026-11-19 (overnight)

Branch `energy-objective` (commits up to 585a085; the user pushes). Continues
`docs/energy_objective_status_2026-11-18.md`. Test case: DESC ripple candidate C, 40/40/6 basis, field grid a/100,
98 GL radii, option A (`--bext desc-fit`), exact B.n = 0 on the LCFS (annulus rows, cutoff 40/6) + `--pin-flux`,
I_p fixed. All energies are the plasma-volume magnetic energy W_Omega relative to J_fit's (W = 1.927471e6 J), which is
the free energy under these constraints.

## 1. What the B.n leak was (job 86a5fa14; `scripts/energy/bn_char.py`)

The ripple drive is a pure (m, n) = (2, 1) normal boundary displacement of amplitude 1.198 mm (cos(2 theta + zeta)
in DESC angles). The unconstrained floor state (A0, M = 0.062 M_DESC) leaks B.n equivalent to a boundary shift of
0.29 mm rms, almost all n = 1: (2,1) -0.212 mm, i.e. it cancels 18 % of the drive, plus (3,1) -0.23, (1,1) -0.07,
(5,1)/(6,1) +-0.14 mm. Force floor vs. leak: cutoff 4/2 still leaks 0.26 mm (moved to m 5-9), M 0.35; 8/4 leaks
0.03 mm, M 0.48; 16/6 and 40/6 leak ~0.004 mm (grid level), M 0.54-0.58. Conclusion: exact B.n = 0 is not an
over-correction; a tolerance loose enough to restore the low floor lets the solver weaken the drive.
Note: a cutoff on f = sqrt(g) dB^rho modes is not a cutoff on displacement modes (sqrt(g)B^theta couples m to m+-1,2).

## 2. Island references under exact B.n = 0 (alpha 0.8)

Shielded S0 meets alpha 0.8 (|J-J_fit|/rms 1.7e-4). b21-pinned islands plateau above it: beta 0.25 / 0.5 / 1 at
M = 1.40 / 2.47 / 4.69 M_DESC (starts were force-blind restores, so these may be path-dependent, not true floors).

## 3. Energy ladder without a profile constraint (lambda 1, 10, 100; alpha 1.5; from S0 at alpha 0.8)

All converged. dW/W -1.64e-6 / -2.47e-6 / -3.38e-6, > 99.5 % from n = 0 coefficient changes, |J-J_fit|/rms 8e-4 /
2.7e-3 / 8.7e-3. The energy is released from the poloidal field (table energies: dW_pol -3.2 / -4.9 / -6.6 J, dW_tor
+0.05 / +0.12 / +0.35 J): current moves out of the core (|iota| on axis -0.018 at lambda 100, q0 1.37 -> 1.41).
The J change is 93-99 % axisymmetric and only 7-13 % inside the q = 2 band (19 % of the volume). b21 grows in the
OPPOSITE phase (-90 deg, 0.95 / 2.7 / 2.95 % of vacuum); Poincare sections show a real opposite-phase (2,1) island
(O-points at theta* = pi/2, 3pi/2, width ~0.02; the vacuum island has O-points at 0, pi and width 0.11). Even the
shielded start has ~0.003 islets at q = 2. Interpretation: W has a first-order slope along axisymmetric profile
changes, the island gain is second order in the ripple and needs |J-J_fit| ~0.06-0.13; an L2 J - J_fit bound cannot
keep the change local. Artifacts: energy_ladder_analysis.png, energy_ladder_summary.csv,
energy_ladder_dJ_decomposition.csv.

## 4. Frozen axisymmetric profile outside the q = 2 band (new constraint)

`--freeze-n0-band RHO1 RHO2 --freeze-tau TAU --freeze-ref STATE` (nest/meanforce_fixed.frozen_n0_block). The n = 0
coefficient directions are split by a generalized eigenproblem G_out v = mu G_all v (G = volume-weighted J Gram on
the collocation points; mu = fraction of the direction's current norm outside the band); every direction with mu >
tau is held at the reference state's value (exact linear rows). Checks: scripts/energy/freeze_check.py (random
feasible directions put <= 0.7 % of their norm outside the band at tau 0.05; unconstrained 55 %); constraints hold to
1e-17 (B.n) / 1e-14 relative (freeze) in the runs. At 40/40/6, band [0.5675, 0.7075], tau 0.05: 922 n = 0
coefficients -> 899 pinned, 23 free. Two fixes were needed: the reference must be the shielded start (J_fit's n = 0
profile outside the band costs M 0.8 -> 2.6), and LinearConstraints.restore now normalizes rows to unit metric norm
and refines iteratively (dependent B.n n = 0 rows vs freeze rows); G0 checks unchanged.

Runs (Modal; all from S0 at alpha 0.8; freeze reference = that state):

| run | lambda | alpha | b21 / vac (+90 deg) | dW/W (n=0 block / helical block / total) | |J-J_fit|/rms |
|---|---|---|---|---|---|
| start (shielded) | - | 0.8 | 0.05 % | -2.45e-8 / -1.3e-9 / -2.59e-8 | 1.7e-4 |
| FZ0 (control) | 0 | 1.5 | 0.04 % | -2.73e-8 / +1.1e-9 / -2.61e-8 | 6.6e-5 |
| FZ1000 | 1e3 | 1.5 | 4.05 % | -4.88e-8 / +1.86e-8 / -3.02e-8 | 7.1e-4 |
| FZ10000 | 1e4 | 1.5 | 3.59 % | -4.92e-8 / +1.89e-8 / -3.02e-8 | 8.4e-4 |
| FZ1000 | 1e3 | 3 | 9.85 % | -7.61e-8 / +4.09e-8 / -3.51e-8 | 1.37e-3 |
| FI0.25 (pinned island) | 0 | 3 | 25 % (pinned) | -1.57e-8 / -1.38e-8 / -2.95e-8 | 1.6e-3 |

(blocks: toroidal mode of the coefficient change from J_fit; cross term ~1e-13.) Leak energy error E/W 9.0-9.8e-10 in
every state (differences ~3e-11, well below the 4-9e-9 energy differences).

Findings:
1. With the profile frozen outside the band, the energy runs open a VACUUM-PHASE (2,1) island (Poincare: O-points at
   theta* = 0, pi, as in vacuum; full width ~0.02 at alpha 1.5); the force-only control opens none. Lambda 1e3 and 1e4
   end in the same state at alpha 1.5, and alpha 3 gives a larger island (9.85 %): the island size is set by the force
   budget, not by lambda.
2. The net energy gain is tiny (-4e-9 to -9e-9 W, i.e. 8-18 mJ) and comes from the in-band n = 0 coefficients
   (|du| ~3-8 A of |u| 8.3e4; surface-averaged profile changes are negligible: |d<J.B/B^2>| <= 0.2, dp ~1e-3 Pa);
   the helical block RAISES W (+1.9e-8 at alpha 1.5). The J change is essentially all helical, ~40 % inside the band.
3. The pinned 25 % island at lambda 0 has a LOWER helical-block energy than the shielded state.

Open question (not settled): is the vacuum-phase island energy-preferred, or a side effect of a local n = 0 relaxation
in the band (as the opposite-phase island was in section 3)? The in-band n = 0 change could be the island's own
helical-flux flattening (the reconnection energy release) or independent profile relaxation.

## 5. Pending / next

- Follow-up job night2 (post-analysis of FI0.25_a3 and FZ1000_a3, then FR1000: release the pinned island under
  lambda 1e3 at alpha 3) was submitted at 03:22 and is waiting on a Modal approval card. If it runs: does the
  released island shrink toward the FZ1000_a3 state (9.85 %) or grow? Same end state from both sides = a single
  minimum at a partial island.
- Discriminating control (proposed): freeze n = 0 everywhere (no band) + energy: only helical freedom. If b21 still
  grows in the vacuum phase, the helical energy prefers the island; if not, the island in section 4 is driven by the
  in-band n = 0 relaxation.
- Resolution: the vacuum island spans ~2 radial node spacings at L = 40; energy differences of 1e-8 W are
  qualitative. A 64/64/6 repeat of the decisive pair is the natural next step.
- Bug fixed during this work: `--subspace accel` raised on q = 0 (slack force bound) and discarded the whole subspace
  search; now the accel direction is skipped (3e01af8).

Results: `results/energy_objective/gpu/{a08,ladder,freeze,leak/char}/`; artifacts frozen_n0_results.png,
frozen_n0_summary.csv, bn_leak_characterization.png/.csv.
