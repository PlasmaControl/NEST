# Radial spline bases for p and T, and how to compare them fairly with Fourier-Zernike

Branch `dshape-fixed` (2026-11-11). Code: `nest/spline_pressure.py`, `nest/spline_basis.py`, the `radial="spline"`
path of `nest/boundary_potential.py`, the `rho_nodes` option of `nest/force.py`. Gates: `tests/test_spline_pressure.py`,
`tests/test_spline_current.py`. Driver options: `scripts/meanforce/mf_fixed.py --pressure spline`, `--current spline`.

## 1. Why splines

A Fourier-Zernike (FZ) series spends its radial resolution evenly over 0 <= rho <= 1 and ties it to the poloidal
resolution: for a harmonic m it offers (L - |m|)/2 + 1 radial polynomials of degree |m| .. L, so the resonant m = 2
gets 24 radial functions at L = 48, resolving features ~1/L to 2/L wide anywhere in the plasma, and refining rho at
q = 2 means raising L for every m and n (cost ~L^2 N). Radial B-splines give every poloidal harmonic the same radial
functions, local in rho, with knots placed where the physics is (here around q = 2, the 2/1 island), so radial
resolution can be added in one band only.

## 2. The radial functions (shared by p and T)

- Cubic B-splines in s = rho^2 (functions of rho^2 are even in rho: smooth at the axis for m = 0).
- Breakpoints in rho, spacing `coarse` everywhere and `dense` within `half` of each centre; the centres are the
  rational surfaces of DESC's iota for the chosen helicities (default q = 2/1; iota from full flux surfaces, it is a
  surface average for a current-prescribed equilibrium). Default (coarse, dense, half) = (0.05, 0.01, 0.08): 34
  breakpoints, n_s = 36 splines, q = 2 at rho = 0.580 for the DSHAPE eps = 1000 A state.
- Knots s_i = rho_i^2, clamped (end breakpoints repeated 4 times).
- Axis regularity: m = 0 uses all n_s splines; m != 0 drops the first three (the only ones with value, slope or
  curvature at s = 0). Of the rest, B_3 behaves like s^3 = rho^6 on [0, s_1] (its own knots have 0 once) and
  B_4, B_5, ... vanish identically there. So every m != 0 term is O(rho^6) at the axis: smooth for even |m| <= 6,
  C^5 (bounded, finite smoothness) for odd |m| or |m| > 6; its size there is tiny (B_3 <= (s_1 / s_4)^3 ~ 2e-4 at
  rho = s_1^(1/2) = 0.05).
- Evaluation in JAX: Cox-de Boor recursion on the (static) knot vector, rho-derivatives through derivative matrices of
  the B-spline coefficients and the chain rule (d/drho = 2 rho d/ds; d2/drho2 = 2 d/ds + 4 rho^2 d2/ds2).
  `spline_basis.spline_radial(t, rho, dr)`, dr = 0, 1, 2.

## 3. Spline pressure (`SplinePressure`)

    p = (1 - rho^2)^(k+1) Q,   Q = sum_{(m,n), j} c_jmn B_j(rho^2) F_mn(theta, zeta),   k = 2 (as the FZ pressure)

- F_mn = cos(m theta - n NFP zeta) (and sin(...) without stellarator symmetry); with symmetry the modes are m = 0,
  n >= 0 and m >= 1, all n.
- p = 0 on the LCFS for any coefficients (the window); grad p = O((1 - rho^2)^k).
- Unknowns: n_p = n_s (N + 1) + (n_s - 3) M (2N + 1).
- Start: DESC's p(rho) fitted with the m = n = 0 splines (least squares), the rest zero.
- In the solver p is still eliminated by the weak (Galerkin) Poisson projection; only the space changes.
- Interface of `Pressure` (n, c, with_c, labels, tensor, point_evaluator, evaluate, fit, save/load), so
  `objective.Fields` and `meanforce_fixed` use it unchanged.

## 4. Spline current potential (`BoundaryPotential(radial="spline")`)

The potential keeps its structure (J = curl T, radial gauge):

    T_theta = I / 2pi h_0(rho) + d_theta chi + rho^2 f(rho) a,    T_zeta = d_zeta chi + f(rho) b,    f = (1 - rho^2)^s

Only a and b change: FZ series -> `SplineBasis` (modes (j, m, n), DESC's real Fourier functions F_m F_n, cos-type with
symmetry), evaluated by `SplineSeparable` (the `Separable` interface with B_j(rho^2) in place of Z_l^|m|(rho)). chi
keeps its single fixed radial function h_|m| per (m, n). Hence, for any coefficients, exactly as with FZ: div J = 0,
J = O((1 - rho)^(s-1)) on the LCFS (zero for s = 2), net toroidal current I through every cross-section, J regular at
the axis.

- Fit of DESC's potential: the same mode-by-mode least squares (`_fit_blocks`), with the radial columns taken from the
  basis (`SplineBasis.radial_columns`); fit radii 3 x the effective L.
- Effective L = round(1 / dense) = 100: grids sized by L (the uniform flux grid of `Fields`, 4 L radii) put several
  points in every fine knot interval.
- Pinned enclosed current / flux, Jacobians, dense tables and diagnostics work unchanged (they use mode lists and the
  `Separable` interface only). No warm start across FZ <-> spline (the modes differ).

## 5. Collocation

The FZ runs use Gauss-Legendre radii (L_res/2 + 1 of them), ~0.02 apart: some splines near q = 2 then had no points
under them and G = P^T P was singular. With any spline basis the radial collocation is composite Gauss-Legendre,
`--p-colloc` (default 4) nodes per knot interval of the union of the breakpoints (132 radii for n_s = 36), at least as
many as the FZ rule. theta and zeta are unchanged.

## 6. Gates (DSHAPE eps = 1000 A, Stellar-AI)

| gate | result |
|---|---|
| p: fit of DESC p(rho) | 5.3e-15 |
| p: tensor vs point evaluator (random c), p / grad p | 4e-16 / 1e-15 |
| p: axis regularity (spread over theta at rho = 1e-3), p on the LCFS | 8e-12, 0 |
| p: grad p vs central differences | 1e-8 |
| T: B_j(rho^2), d/drho, d2/drho2 vs SciPy | 3e-16, 4e-16, 5e-16 |
| T: rms \|J - J_DESC\| / rms J_DESC, spline M/N 16/6 (n_u 14,440) vs FZ 24/24/6 (n_u 4,550) | 1.2e-2 vs 3.9e-3 |
| T: max \|J\| on the LCFS | 0 |
| T: TensorCurrent vs point evaluator (random u) | 2e-15 |
| T: save / load | exact |
| solver smoke (16/6, a/40, alpha 0.7) | met in 4 steps; pinned current/flux to 1e-14 |

## 7. Unknowns (stellarator symmetric, NFP 1, n_s = 36)

| M / N | FZ p (L = M) | spline p | FZ a + b (L = M) | spline a + b |
|---|---|---|---|---|
| 16 / 6 | 994 | 7,116 | 1,998 | 14,232 |
| 24 / 8 | 2,762 | 13,788 | 5,538 | 27,576 |
| 32 / 8 | ~4.8k | 18,276 | 9,554 | 36,552 |

The spline count is linear in n_s (finer knots near q = 2: n_s 57 or 76 give 1.6x or 2.1x). chi adds ~2MN in both.

## 8. Results so far (indicative, NOT a fair comparison: see 9)

DSHAPE eps = 1000 A, pinned I and Psi; floors in units of DESC's own force M_DESC = 4.13e-5 (the force the alpha
ladder stopped at).

| current / pressure | annulus | n_u / n_p | floor (M_DESC) | wall to floor | island path (2/1 width) |
|---|---|---|---|---|---|
| FZ 24/24/4 / FZ | a/60 | 3,154 / 1,469 | 1.24 | 14 min | 0.015 -> 0.032 |
| FZ 24/24/4 / spline 24/4 | a/60 | 3,154 / 7,308 | 1.21 | 27 min | 0.015 -> 0.007 -> 0.029 |
| FZ 24/24/8 / FZ | a/40 | 5,946 / 2,769 | 0.38 | 23 min | 0.017 -> 0.043 (alpha 0.1) -> 0.025 |
| spline 24/8 / FZ | a/40 | ~28k / 2,769 | 0.29 | 3.2 h | 0.018 -> 0.034 -> 0.021 |
| spline 24/8 / spline 24/8 | a/60 | ~28k / 13,788 | 0.26 | 2.8 h | 0.016 -> 0.042 (earlier, alpha 0.3) -> 0.020 |
| FZ 48/48/8 / FZ | a/60 | ~22k / ~10k (est.) | ~0.10 | ~2.6 h to 0.13 | 0.012 -> 0.043 (alpha 0.05) -> 0.025 (0.02) |

- Every basis shows the same sequence (shielded -> island reopens with reversed phase -> partial re-shielding).
- Spline T alone or with spline p lowers the 24/8 floor a little (0.38 -> 0.26-0.29) at ~5x the unknowns and ~8x the
  time; FZ 48/48/8 gets further per unknown.
- Conditioning: cond(G = P^T P) 1.6e9 for spline p (n_s 36, 16/6) vs 7.9e4 for FZ 24/24/8.
- Spline p does not flatten p inside the island (O/X drop ratio 0.97-1.01).
- Physics reason (from the diagnostics): the 2/1 shielding current is a broad feature ~0.1 wide in rho centred at
  rho ~ 0.62, resolved on a 7x finer radial grid; FZ needs only ~2-3 radial functions across it (L ~ 48), and 0.01
  spline spacing at q = 2 over-resolves it.

Why these numbers are not comparable: different collocation grids (composite vs Gauss radii), different annuli (a/40
vs a/60), alpha relative to each run's own start force (2.25-6.6 M_DESC, so the same alpha is a different force),
different targets (each basis' own J_fit), and different unknown counts.

## 9. A fair comparison

1. **Refinement ladders, not equal counts.** For each basis a sequence of increasing resolution (FZ: L = M = 24, 32,
   48, 64 at N = 8; splines: n_s ~ 20, 36, 57 at M/N 24/8 and one larger M), and plot each quantity of interest against
   the number of unknowns n_u + n_p and against wall time: the floor, |J - J_ref| at a fixed force bound, the 2/1
   width, b21 and the s21 width at fixed alpha. Two answers: efficiency (which curve reaches a given accuracy with fewer
   unknowns or less time) and agreement (do both converge to the same limit; if not, one basis has a structural limit).
2. **One target.** DESC's J is not suitable as a common target for NCSX (nonzero on the LCFS, which neither basis can
   represent); for DSHAPE it does vanish at the edge, but the cleanest choice is one reference fit for all runs:
   fit DESC's potential once in the most accurate basis (e.g. FZ 64/64/8) and give every run that J_ref as its target
   (a small `mf_fixed` option: evaluate the reference potential at the run's collocation points in place of its own
   J_fit). Each run still starts from its own best fit; the fit misfit to J_ref is reported separately (it is the
   representation error of the basis).
3. **One force bound.** `--alpha-ref desc`: eps = alpha x M_DESC, DESC's own volume-mean force of that equilibrium,
   the same number for every basis and grid at a given eps (M_DESC differs between eps: 1.33e-5, 4.13e-5, 9.13e-5,
   9.51e-5 at 300 / 1000 / 2000 / 3000 A, so for an eps scan either compare at equal alpha x M_DESC or use one common
   number).
4. **One measuring apparatus.** Re-evaluate every final state on the same grid (fixed radial nodes dense across q = 2,
   independent of the basis; fixed theta, zeta) and the same annulus (a/80): M in M_DESC units, |J - J_ref|, the force
   by rho band, the off-node force (`dshape_fixed_diag.py --fine`), the island width, b21 and s21.
5. **Pareto fronts.** For each basis plot |J - J_ref| against M (common apparatus): the lower front gets better force
   balance for less change of current; overlay the 2/1 width against M to see whether the basis changes the island
   physics at equal force balance.
6. **Two pairings.** (a) Equal Fourier resolution (spline 24/8 vs FZ 24/24/8): isolates the radial freedom.
   (b) Matched local radial resolution of m = 2 at q = 2 (spline knot spacing there ~ the FZ spacing ~2/L), so the
   remaining difference is only how each basis spends its unknowns away from q = 2.
7. **Cost and robustness.** Wall time and memory per alpha, Lanczos iterations per step, cond(G), the floor reached
   and whether the ladder stalls.

Cost: the common-apparatus re-evaluation is ~1 h of code plus short GPU jobs on the existing states; the reference
target is a small `mf_fixed` option; the ladders are ~6 runs of a few hours each (gpu-short / gpu-medium on
Stellar-AI).

## 10. Which splines are used (coefficient spectrum, 2026-11-11)

`scripts/meanforce/spline_spectrum.py` on the two spline 24/8 runs (splTP: spline T + p, deepest state alpha 0.1;
splT: spline T + FZ p, alpha 0.05). For each (spline j, poloidal m) group (all n together): rms |J| of the group's
coefficients for J_fit and for the optimizer's change u - u_fit; rms |grad p| for the spline p*. Figures
`figures/dshape_fixed/spl{TP,T}24n8_spectrum.png`.

- 1,620 (j, m) groups per series (a and b), 13,788 coefficients each.
- Splines whose support ends where rho^|m| < 1e-3 (near-axis, high m): 39 % of the groups, carry <= 4e-5 of the
  rms^2 of J_fit or of the change (<= 1e-8 of grad p).
- By |m|: J_fit a 94.4 % in |m| <= 2, 5.5 % in 3-6, 0.1 % in 7-12; b 99.9 % in |m| <= 2. Optimizer change in a:
  98.6 % (splTP) / 90.9 % (splT) in |m| <= 2, 1.3 / 7.7 % in 3-6, 0.1 / 1.3 % in 7-12, <= 7e-4 above. grad p:
  99.999 % in |m| <= 2.
- In rho the content sits in the dense q = 2 band (0.5-0.7) and toward the edge; the near-axis high-m corner is empty.

Truncation (share of coefficients kept; worst share of rms^2 lost over J_fit, change in a and b, both runs):

| truncation | kept | lost (worst) |
|---|---|---|
| drop rho_end^\|m\| < 1e-3 | 61 % | 4e-5 |
| M <= 12 and drop rho_end^\|m\| < 1e-3 | 45 % | 7e-4 (splT change) |
| M <= 8 | 35 % | 5e-3 (splT change) |
| M <= 6 | 27 % | 1.4e-2 (splT change) |

The spectrum suggested ~40 % of the unknowns were free to drop. **The runs say otherwise** (`scripts/meanforce/dfx_trunc.sh`,
eps = 1000 A, same arguments as the full runs, alpha relative to the start force; floors in M_DESC):

| run | truncation | start force | floor | full-basis floor |
|---|---|---|---|---|
| 37098, spline T + p, a/60 | rho rule + M <= 12 | 6.47 | ~1.65 (stalled at alpha 0.1, cancelled) | 0.26 |
| 37099, spline T + FZ p, a/40 | rho rule + M <= 12 | 7.25 | ~1.51 (stalled at alpha 0.1, cancelled) | 0.29 |
| 37100, spline T + p, a/60 | rho rule only | 2.94 | 0.52 (alpha 0.1 not met, \|J - J_fit\| 0.25) | 0.26 |

Even the rho^|m| < 1e-3 rule, which drops splines carrying 4e-5 of J's rms^2, doubles the floor, and the M cap makes it
6x worse. The coefficient spectrum of J does not identify unused splines: force balance (J x B - grad p, with the
high-m geometry of the D shape) needs freedom where J itself has almost no content. Do not truncate the spline basis
by J content.

## 11. Spline knots matched to Fourier-Zernike, and the FZ vs spline runs (2026-11-11)

FZ's local radial resolution at L: the zeros of the top radial function R_L^m are spaced
dr_FZ(rho) = pi sqrt(1 - rho^2) / (L + 2) (within 1 % for m <= 2, 10 % up to m = 12 at L = 48), beyond the first
zero rho_1(m) (L = 48: 0.08, 0.105, 0.20, 0.34, 0.59 for m = 1, 2, 6, 12, 24). Matching options:

- `--knots-fz L C`: outer breakpoints rho_i = sin(i C pi / (L + 2)) (spacing C x dr_FZ), the q = 2 band at 0.01 kept.
  C = 1 matches the number of radial unknowns; C = 1/2 matches the accuracy at FZ's shortest scale (cubic spline
  error ~0.2 C^4 there). `--knots-fz L C RHO`: coarse below RHO, FZ spacing beyond (edge refinement).
- `--spl-mfz L`: for m != 0 drop splines whose support ends before rho_1(m) at L (FZ's radial reach of m).
- The current knots (0.05) are finer than FZ-48 for rho < 0.75 and up to 2.5x coarser near the edge, where about
  half of the floor force sits.

Runs (`scripts/meanforce/dfx_cmp.sh`, eps = 1000 A, a/60, absolute bound alpha x M_DESC with the eps48 ladder, FZ
pressure at the run's L M N everywhere):

| run | current | a + b | compared with |
|---|---|---|---|
| A | spline 48/8, knots FZ-48 x 1/2 + band, FZ-48 reach | 60,532 | FZ 48/48/8 (20,850; eps48 e1000) |
| B | FZ 24/24/8 | 5,538 | C |
| C | spline 24/8, knots FZ-24 x 1/2 + band, FZ-24 reach | 19,150 | B |
| D | spline 24/8, current knots | 27,576 | E |
| E | spline 24/8, current knots + FZ-48 x 1/2 beyond rho 0.66 | 44,256 | D |
