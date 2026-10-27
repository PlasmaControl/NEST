# The closest free-boundary state to DESC's current under a mean-force bound, with the Poisson pressure (NCSX)

2026-10-27. Report of the study planned in `docs/mean_force_poisson_plan.md`: methods, all results obtained so far, caveats, and
what remains. Branch `closest-poisson` (pushed to `PlasmaControl/NEST`). Code map and commands to continue:
`docs/meanforce_handoff.md`. Raw data: `results/meanforce/`; figures and tables regenerate with `scripts/meanforce/collect.py`.
Six jobs were still running when this was written; their values are marked "running" and appear in the JSONs when they end.

## 1. Summary

- **Question.** How far from DESC's fitted current J_fit must the current move for NEST's force balance error, measured exactly as DESC measures it,
  to fall to a fraction α of DESC's own, on the NEST free boundary (the boundary moves, B·n = 0 on it, toroidal flux held), with the pressure
  from the weak Poisson equation instead of free unknowns?
- **Method.** Augmented Lagrangian on (current potential u, boundary displacement ξ) with three scalar constraints (mean force, B·n, flux),
  the pressure eliminated by a weighted least-squares projection, two trust-region solvers (matrix-free GLTR and dense QR) on one B200,
  re-anchoring of the moving geometry every ~2 mm.
- **Result at a/40, 16/16/12.** The front ‖J − J_fit‖/rms J_fit is 1.49e-2 (α = 0.7), 1.89e-2 (0.5), 2.77e-2 (0.3), 3.64e-2 (0.2), 5.39e-2 (0.1)
  and 1.13e-1 (0.05). Both solvers agree to 0.01–0.3% down to α = 0.1; Hessians on the active tangent space are positive.
- **Convergence.** The annulus is converged by a/60 (a/60, a/100, a/120 agree to 0.05%); a/20 is unreliable below α ≈ 0.2. A larger NEST basis (L = M) lowers
  the front strongly at the tight end and converges by L = M ≈ 24 (against a fixed target); N = 12 → 16 matters little.
- **Remains:** finish six jobs, a few redos, QR at large L = M, Poincaré sections, the study-1 re-evaluation on the finest annulus.

## 2. Setting

- **Equilibrium:** `desc_input/vmec_free_boundary/NCSX/NCSX_fb_L16M16N12.h5` solved in the NCSX coils (`NCSX_coils.h5`), NFP = 3. On DESC's quadrature
  grid: ⟨|F|⟩_vol/norm = 8.78e-4, B·n/|B| rms 4.53e-3 (max 2.8e-2), pressure jump rms 2.6e-3.
- **Field:** B = B_ext (coils) + B_plasma(J), with B_plasma from the cylindrical annulus free-space solver (Chebyshev in r and z, planes in φ,
  `nest/cylindrical_bs.py`). The annulus spacing is h = a/frac with the planes following h (a = 0.3226 m).
- **Current:** J = ∇×T with T a Fourier–Zernike potential (`BoundaryPotential`; J = 0 on the LCFS, edge factor (1 − ρ²)², the enclosed current I(ρ) is left free, not held, and monitored through iota). NEST basis 16/16/12 (L/M/N) unless stated, n_u = 4,234. J_fit is DESC's J fitted in that basis; it is carried with the moving coordinates.
- **Pressure space:** p = (1 − ρ²) Σ c_j φ_j in the same basis as T (n_p = 1,917 at 16/16/12), p = 0 on the LCFS.
- **Collocation:** 1.5 × (basis + equilibrium) per direction: 25 × 97 × 37 = 89,725 points (half a field period, stellarator symmetry),
  Gauss–Legendre in ρ, uniform in θ and ζ, weights |√g|. The force is evaluated at these points.

## 3. Methods

### 3.1 Unknowns and the moving boundary
The unknowns are z = (u, ξ). ξ is the boundary's normal displacement (413 stellarator-symmetric modes, M = 16, N = 12, the equilibrium's own boundary
resolution: a lower ξ resolution leaves part of the coils' B·n unreachable). It is extended inside as ρ^|m| per mode, so the flux coordinates move with it
(`Displacement`, `MovingGeometry`). Exact in ξ: the geometry at all points, hence J, ∇p, the volume, the normal. First order in the displacement: B_ext at the
moved points and the samples of J fed to the field solver (`StencilShift`). Because of the first-order parts, the geometry is **re-anchored** (`Displacement.apply`,
ξ → 0) when the boundary has moved 2 mm (`--max-disp`); everything is rebuilt on the moved geometry and the objective is continuous across it
(M changes by ≤ 3e-4 relative, B·n and Ψ by ≤ 1e-6). Pressure balance across the boundary holds identically (no surface current: J = 0 and p = 0 there); what
remains of the free-boundary problem is B·n = 0 and the flux. B·n is taken from the annulus; it was checked against direct Biot–Savart (`DirectField`, 32 × 4 sources)
on the same points with DESC's plain-mean definition: 4.582e-3 against 4.585e-3 at the start, 4.528e-3 against 4.533e-3 at a/40 α = 0.7, pointwise difference 7e-5 rms
(5.5e-4 max), far below the switch threshold β/3.

### 3.2 The pressure from the weak Poisson equation (variable projection)
With w_i = dV_i/V on the moved geometry and sw = √w, let a = sw (J×B) (3n) and P the columns sw ∇φ_j of the pressure space (3n × k). The pressure p* minimizes
‖a − P c‖, i.e. it solves the weak Poisson equation ∫ w ∇q·∇p = ∫ w ∇q·(J×B) in the space (p = 0 on the boundary), and the remaining force is
F = (I − QQᵀ) a with P = QR (thin QR; columns below 1e-12 of the largest |R_jj| are dropped, none were). So p is the physical Poisson pressure of the current state,
eliminated exactly at every evaluation (the Galerkin identity Pᵀf = 0 holds to ≤ 5e-13 at every step). The derivative of F with respect to (u, ξ) includes the derivative of the
projector with respect to ξ (Golub–Pereyra): df = (I − QQᵀ) t − Q R⁻ᵀ(dPᵀ f), with t = da − dP c; dP involves only the pressure basis on the moved geometry (no field solve). This is the formulation of the
`pressure-poisson` branch (`speed elim`), re-implemented for the 3-D annulus free boundary in `nest/meanforce.py`.

### 3.3 The mean force as DESC computes it
M = Σ_i w_i |F_i| / norm with norm = DESC's ⟨|∇|B|²|/2μ₀⟩_vol of the starting equilibrium (held fixed), a plain volume integral: no 1/|F| weights, no point counting, no edge
exclusion. Gate: DESC's own |F| on NEST's collocation points gives 8.762e-4 against DESC's quadrature 8.779e-4 (−0.19%). The constraint is M ≤ ε = α M_DESC,
M_DESC = 8.7788e-4. The rms force ⟨|F|²⟩^{1/2}/norm is reported alongside.

### 3.4 The optimization problem and the augmented Lagrangian
    min_{u,ξ}  ½‖g‖²,   g = sw (J − J_fit)/rms J_fit
    s.t.  c_F = M/ε − 1 ≤ 0,   c_B = (B·n/|B| rms / β)² − 1 ≤ 0,   c_Ψ = Ψ/Ψ_t − 1 = 0
β = 4.53e-3 is DESC's own free-boundary B·n/|B| (plain mean over its boundary grid); Ψ_t is NEST's start flux through φ = 0 (0.497005; DESC 0.497070).
The augmented Lagrangian has three multipliers ν and one penalty ρ:
    L_A = ½‖g‖² + ψ(c_F, ν_F) + ψ(c_B, ν_B) + ν_Ψ c_Ψ + ½ρ c_Ψ²,   ψ(c, ν) = (ρ/2)(max(0, c + ν/ρ)² − (ν/ρ)²)
Outer loop: minimize L_A at fixed (ν, ρ), update ν_F = max(0, ν_F + ρ c_F) (same for B), ν_Ψ += ρ c_Ψ; ρ ×10 only while a constraint is outside its tolerance and the violation did
not fall by 4. Tolerances: M to 1%, B·n/β to 2% (on the squared ratio), Ψ to 1%. An outer iteration is converged when the constraints are met **and** the inner solve was stationary (its own
tests, or ‖g‖²/2 flat to 1e-4 relative over the last 10 accepted steps with the mean-force bound active within 1%). A missed bound is retried at the geometric mean with the last bound met.
The multipliers report the front's slope: ν_F = −dΦ*/dε.

### 3.5 The inner step model (sum of squares)
The inner problem at an iterate z_k is a sum of squares, so both trust-region solvers apply. Its rows: the J rows g; B·n rows with weight ψ'_B/(β² n_b)^{1/2}; a Ψ row; and for the mean
force the rows of the concave majorizer of √(|z|² + τ²) (z_i = w_i F_i, τ = 10⁻³ ε norm): s_i z_i with s_i² = ψ'_F/(ε norm t_i), t_i = max(|z_i|, w_i τ). Every decrease of the surrogate is a
true decrease and its gradient is exactly ψ'_F ∇c_F; the weights are refreshed at every accepted step. Rows for the penalty's own curvature, √ψ'' ∇c, are added; ψ'' is the AL's value, kept (as ρ) also
within 5% of an inactive bound (`Weights.d2_model`): without it the model cannot see a step that crosses a bound at c = 0 with ν = 0 (the cause of a radius collapse found in the first sweep). Every acceptance test,
every actual/predicted ratio and every reported number uses the plain L_A and the plain M.

### 3.6 Trust-region metric and solvers
The trust region is measured in the physical metric L Lᵀ = blockdiag(H_J, κ G_ξ): H_J the Gram matrix of g (so |y| is the rms change of J / rms J_fit), G_ξ the rms of the boundary
displacement, κ = 174 probed so the ξ and u columns of the Jacobian have the same mean size (the scaling that fixed the damping floor in the earlier free-boundary runs). First radius 1e-2;
radius ×0.25 when the ratio is < 0.25, ×2 when > 0.75 on the boundary. Two solvers act on the same rows, so states, Hessians and costs compare directly:
- **GLTR** (matrix-free): generalized Lanczos with full reorthogonalization, the trust-region subproblem solved exactly in the Krylov space (undamped Gauss–Newton step when it fits, else the
  secular equation); up to 200 Lanczos iterations, tolerance 1e-2. One annulus solve (and one transpose) per product.
- **QR** (dense, one B200): the stacked Jacobian [J L⁻ᵀ | r] (≈ 540,000 × 4,648 at 16/16/12) by `jnp.linalg.qr`, DESC's exact trust-region rule (undamped step when it fits, secular Newton, damped-QR
  bisection). The u columns come from per-anchor tables (4,234 field solves, 2 min at a/40; B is linear in u at fixed geometry), the 413 ξ columns by Jacobian–vector products at the state. Two safeguards found necessary:
  an undamped step whose model does not decrease (near-singular matrix) is replaced by a damped one; and when the radius has collapsed with the boundary moved, the geometry is re-anchored (the tables are exact only at ξ = 0).

### 3.7 Diagnostics
Per step: radius, step, ratio, Gauss–Newton share (|c|²/(|c|² + r⊥²) for QR, the μ = 0 Krylov model for GLTR), μ, constraints, multipliers, Galerkin identity, time. **Stall detector**
(radius < 1e-8 with GN share > 0.9 over 5 steps, or 3 rejections): saves the Jacobian against central differences (u, ξ, random directions, per block), the adjoint test, the model along the step (Gauss–Newton against second order against truth), the
smoothing weights and ξ against the re-anchoring limit. **Hessian**: the Lagrangian's Hessian in the metric variables, reduced to the tangent space of the active constraints, by Lanczos (40 steps) with Hessian–vector
products from central differences of the exact gradient; the step h = 1e-9 (1e-6 gave spurious negative Ritz values, 1e-8 still did at two bounds). Also: the iota profile (surface average of √g Bᶿ/√g Bᶻ), |F| averaged over coordinate surfaces
(`radial_force.py`), the B·n check against direct Biot–Savart (`bn_check.py`), and `eval_common.py` (states evaluated on one grid, annulus and pressure space against J_DESC).

### 3.8 Gates (`tests/test_meanforce.py`, GPU, ~40 s; all pass)
Manufactured Poisson (3e-15), trailing block of the QR = explicit projection (4e-16), Galerkin identity (1e-15), J v against central differences in u, ξ and both for g, f, B·n, Ψ (≤ 8e-10,
including the projector's ξ derivative), adjoint (2e-14), envelope identity (9e-8), dense rows against the matrix-free ones (3e-14), ∇L_A against differences of the plain merit (3e-8), GLTR and QR steps agree (3e-3 after 6 steps),
a bound at c = 0 with ν = 0 keeps its curvature row, iota against DESC's. The annulus free boundary itself passes `tests/test_free_boundary_cyl.py` (ξ = 0 against the fixed-boundary fields 4e-15, J v against differences 6e-10, adjoint 5e-14,
|ξ|² scaling of the first-order treatment).

## 4. Runs

Stellar-AI, one B200 (180 GB) per job, `run_sai_cyl.sbatch`. Each run is a sweep α = 0.7, 0.5, 0.3, 0.2, 0.1, 0.05 from DESC's state (or from a saved state, `--start`). Costs at a/40: GLTR 29–163 min per bound (steps ≈ 40 s,
the last bound longest), QR 14–37 min (≈ 13 s per step, 40 s when it rebuilds the Jacobian at a re-anchor), 5–21 re-anchorings per run. GPU peak 27 GB at a/40, 137 GB for a/60 QR; a/100 and a/120 (300 M and 518 M nodes)
need 53 and 76 GB and 74–124 GB of host memory, with 4–14 min per GLTR step.

## 5. Results

### 5.1 The start
NEST at 16/16/12 with the Poisson pressure starts at M = 5.74 M_DESC (5.04e-3; rms |F| 8.73e-3), B·n/|B| 4.58e-3, Ψ 0.497005. NEST's B·n at the start agrees with DESC's (4.53e-3); the force is higher than DESC's because
NEST's current vanishes on the LCFS and the basis is the fit's; this is what the study then reduces.

### 5.2 The front at a/40, both solvers
‖J − J_fit‖/rms J_fit (M/M_DESC = α, B·n/|B| = 4.53e-3 and Ψ within its 1e-2 tolerance, mostly below 1e-3, unless noted):

| α | 0.7 | 0.5 | 0.3 | 0.2 | 0.1 | 0.05 |
|---|---|---|---|---|---|---|
| GLTR | 1.4868e-2 | 1.8907e-2 | 2.7739e-2 | 3.6434e-2 | 5.3919e-2 | 1.1279e-1 |
| QR | 1.4882e-2 | 1.8917e-2 | 2.7737e-2 | 3.6501e-2 | 5.4068e-2 | 1.1414e-1; 1.1605e-1 (redo) |

- The solvers agree to 0.01–0.3% down to α = 0.1; the mean-force multiplier is the slope of the front (−dΦ/dε = 0.30 at α = 0.7). Hessians (h = 1e-9) on the tangent space are positive for a/40 GLTR at all six bounds (smallest eigenvalue
  1.08, 3.6, 7.9, 15.6, 154, 7.4e3) and for QR at α = 0.1 (+169) and 0.05 (+9.3e3), with the largest 1e3–4e7.
- **At α = 0.05 the solvers end in different local minima:** GLTR's has B·n at 0.52 β (2.36e-3), QR's at its bound (4.46–4.47e-3); the objectives differ by 1–3%.
- The iota profile (I(ρ) left free) moved by at most 3e-3 (at the axis); 6e-4 or less from ρ = 0.5 out.
- First health review at α = 0.7 (`docs/meanforce_gltr_a40.md`, `figures/meanforce/health_gltr.png`, `health_qr.png`): no radius collapse, Gauss–Newton share falling from 0.47 to ≈ 0 (the remaining
  residual is not reducible at first order, the opposite of the pointwise study's pattern), penalty ρ = 1 throughout, re-anchorings continuous, stall detector silent. The model under-predicts the decrease (ratio median ≈ 1.25): the majorizer's curvature is conservative.

### 5.3 Where the force error is (a/40, α = 0.7)
|F|/norm averaged over each coordinate surface and over ρ ranges (`figures/meanforce/radial_a40.png`): against NEST's start the force fell 8–23× in every range. Against DESC: 2.0–2.4× lower for ρ < 0.6, equal at ρ = 0.6–0.95 (the iota = 1/2 and 3/5 rational surfaces
sit here), 2–5× lower for ρ > 0.95; 86% of the gain over DESC's mean comes from ρ > 0.95 (8% of the volume). Part of that edge advantage is structural (J = p = 0 on the LCFS).

### 5.4 Study 1: annulus spacing (NEST 16/16/12, own fit)

| run | 0.7 | 0.5 | 0.3 | 0.2 | 0.1 | 0.05 |
|---|---|---|---|---|---|---|
| a/20 GLTR | 1.4945e-2 | 1.9127e-2 | 2.8746e-2 | 3.8719e-2 | 6.4190e-2 | 2.0743e-1 (!) |
| a/20 QR | 1.4958e-2 | 1.9138e-2 | 2.8758e-2 | 3.8684e-2 | 6.5191e-2 (!) | 2.0519e-1 (!); 1.9611e-1 (redo) |
| a/40 GLTR | 1.4868e-2 | 1.8907e-2 | 2.7739e-2 | 3.6434e-2 | 5.3919e-2 | 1.1279e-1 |
| a/60 GLTR | 1.4863e-2 | 1.8895e-2 | 2.7683e-2 | 3.6314e-2 | 5.3502e-2 | running |
| a/60 QR | 1.4875e-2 | 1.8905e-2 | 2.7693e-2 | 3.6416e-2 | 5.3613e-2 | 1.1225e-1 |
| a/80 GLTR | 1.4862e-2 | 1.8893e-2 | 2.7681e-2 | 3.6305e-2 | running | |
| a/80 QR | 1.4876e-2 | 1.8912e-2 | 2.7688e-2 | running | | |
| a/100 GLTR (from the a/40 state) | | 1.8895e-2 | | | 5.3523e-2 | running |
| a/120 GLTR (from the a/40 state) | | 1.8895e-2 | | | 5.3523e-2 | running |

- Converged by a/60: a/60, a/100, a/120 agree to 0.05% at α = 0.1 and four digits at α = 0.5 (the a/40 state evaluated on a/100 and a/120 has M/M_DESC = 0.4997 against the bound 0.5). a/40 is ≤ 0.1% high at α ≥ 0.5 and 0.2–0.8% high at α ≤ 0.3.
  a/20 is 0.5% high at α = 0.7 and 20% high at α = 0.1.
- **a/20 at α = 0.05 is a discretization artifact.** Its states (0.196–0.207 with two different solvers, B·n 0.45 β) have M = 0.0447 (a/40) and 0.0446 (a/60) when solved to 0.0500 on a/20: the coarse annulus inflates the mean force by about 0.005 M_DESC, negligible
  at α = 0.7 and 11% of the bound at 0.05. The a/40 α = 0.05 state re-evaluated on a/60 gives M = 0.0496, B·n 4.458e-3 and the same J: a/40 is converged there to 0.8%.

### 5.5 Study 2: NEST basis (a/40, GLTR)
Own fit (each basis fits J_DESC itself and has its own collocation grid, so the loose-bound values are not directly comparable):

| α | 0.7 | 0.5 | 0.3 | 0.2 | 0.1 | 0.05 |
|---|---|---|---|---|---|---|
| 16/16/12 | 1.4868e-2 | 1.8907e-2 | 2.7739e-2 | 3.6434e-2 | 5.3919e-2 | 1.1279e-1 |
| 20/20/12 | 1.3829e-2 | 1.8150e-2 | 2.6376e-2 | 3.3618e-2 | 4.6431e-2 | 6.3669e-2 |
| 24/24/12 | 1.4974e-2 | 1.9157e-2 | 2.7183e-2 | 3.4097e-2 | 4.6013e-2 | 5.9462e-2 |
| 28/28/12 | 1.6667e-2 | 2.0623e-2 | 2.8363e-2 | 3.5100e-2 | 4.6733e-2 | 5.9082e-2 |
| 16/16/16 | 1.5123e-2 | 1.9121e-2 | 2.7832e-2 | 3.6365e-2 | 5.2987e-2 | 1.0401e-1 (!) |

On one apparatus (`eval_common.py`: L = M = 28's collocation grid, the a/40 annulus, the 28 pressure space, fixed target J_DESC), |J − J_DESC|/rms J_DESC (M/M_DESC of the state there):

| α | 16/16/12 | L = M = 20 | L = M = 24 | L = M = 28 |
|---|---|---|---|---|
| 0.7 | 4.163e-2 (0.666) | 3.666e-2 (0.690) | 3.455e-2 (0.700) | 3.401e-2 (0.700) |
| 0.5 | 4.346e-2 (0.474) | 3.872e-2 (0.495) | 3.648e-2 (0.502) | 3.582e-2 (0.500) |
| 0.3 | 4.833e-2 (0.281) | 4.346e-2 (0.298) | 4.126e-2 (0.303) | 4.043e-2 (0.300) |
| 0.2 | 5.399e-2 (0.185) | 4.827e-2 (0.199) | 4.599e-2 (0.203) | 4.508e-2 (0.200) |
| 0.1 | 6.700e-2 (0.091) | 5.771e-2 (0.099) | 5.491e-2 (0.102) | 5.375e-2 (0.100) |
| 0.05 | 1.199e-1 (0.047) | 7.19e-2 (0.049) | 6.59e-2 (0.050) | 6.36e-2 (0.050) |

- The L = M = 28 state reproduces M/M_DESC = 0.1000 and 0.0500 under this apparatus (consistency check). Against the fixed target the distance falls monotonically with L = M at every bound and converges: the step 24 → 28 is 1.6% (α = 0.7) to
  3.4% (α = 0.05), against 6–8% for 20 → 24 and 44% for 16 → 20 at α = 0.05. The radial and poloidal resolution limits the tight end of the front; L = M = 24 is nearly enough. (The own-fit non-monotonicity at loose bounds is the per-basis J_fit;
  the common values are about twice the own-fit ones at α = 0.7 because the fixed target J_DESC is itself ≈ 3e-2 away from every basis's J_fit: an inference from adding the two in quadrature, not a separate measurement.)
- 16/16/16 against 16/16/12 (own fit): +1.7%, +1.1%, +0.3%, −0.2%, −1.7% for α = 0.7 … 0.1, so N is not the limit with NFP = 3 (N = 12 reaches toroidal mode 36). The α = 0.05 point (0.104) is unconverged and being redone (§7).
- Hessians (h = 1e-9) are positive for L = M = 20, 24, 28 at α = 0.1 (smallest 293, 327, 331) and 0.05 (5.4e3, 1.9e3, 2.1e3).

### 5.6 The front, all runs
`figures/meanforce/study_front.png` (‖J − J_fit‖ against M/M_DESC for every run), `study_spacing.png` (against h/a), `study_basis.png`, `study_common.png`; every number in `study_table.md` and `study_common_table.md`.

## 6. What went wrong on the way (and was fixed)

| symptom | cause | fix |
|---|---|---|
| a/80 QR out of memory at its first step | 32 ξ columns (field solves) at once need 186 GB at 156 M nodes | batch of field-solve products scales as 32 (40/frac)³ (`--tan-chunk`) |
| L = M = 28 out of memory at setup | the metric's Gram matrix built from the 46 GB table on the device | built on the host in blocks of 20,000 points |
| GLTR radius collapsed (ratios −3e5) at the start of an outer iteration | B·n bound at c = 0 with ν = 0: no B·n rows and no penalty curvature in the model, the penalty grows like |step|⁴ | penalty curvature kept within 5% of a bound; a gate |
| 20 of 80 QR steps rejected | undamped Gauss–Newton step of a near-singular matrix, model without decrease | damped step instead (guard) |
| a/20 QR, α = 0.05: radius 2e-13, |g| 0.19 | QR's tables are exact only at ξ = 0, wrong by O(ξ) after the boundary moved 2 mm | forced re-anchor when the radius collapses |
| "flat" stop took stalls for convergence (a/20 α = 0.05, 16/16/16 α = 0.05) | the flat test ignored a vanishing radius or an inactive bound | no flat when the radius is < 1e-8; requires M within 1% of the bound |
| negative Hessian Ritz values at α ≤ 0.5 | finite-difference error of the Hessian–vector products | h = 1e-9 (stable between 1e-8 and 1e-9 at the checked points) |
| penalty ρ 1 → 1e6 at α = 0.05 | escalation although all constraints were within tolerance | ρ rises only while a constraint is outside its tolerance (jobs started earlier ran with the old rule) |
| a/100, a/120 from a converged state: first 5 steps rejected, 8–14 min per step | first radius 1e-2, flat window of 10 accepted steps | `--delta0 1e-4 --stat-window 5` |

Two statements made earlier in the work were corrected after the evidence arrived: the a/20 α = 0.05 value was first attributed to a QR convergence failure (it is a discretization artifact, reproduced by GLTR and shown by re-evaluation),
and the 16/16/16 α = 0.05 value (−8%) was first reported as a result (it is unconverged: M = 0.0464 with the bound inactive).

## 7. Caveats

1. **Marked points (!)** in `study_table.md`: M below 0.99 α (inactive bound: not at the optimum of that bound). Currently: 16/16/16 α = 0.05 (M = 0.0464, objective still falling; continuation job 6540), a/20 QR α = 0.1 (M = 0.0988) and the a/20 α = 0.05 points (discretization artifact).
2. **α = 0.05 is a hard bound** with several local minima (GLTR and QR differ by 1–3%; the B·n branch differs); values there carry that spread. Down to α = 0.1 the solvers agree to < 0.3%.
3. **Per-basis J_fit:** compare bases with the common-apparatus table, not the own-fit one, at the loose bounds. The common apparatus uses the 28 pressure space, which lowers the force of the coarser-basis states relative to their own M.
4. **Hessians stored in runs started before 2026-10-27** used h = 1e-6 (spurious negatives below α = 0.7); the verified values are those listed here.
5. **The annulus converges slower in the force than in J:** a/20's mean force is off by ~0.005 M_DESC; this limits what a/20 can say about tight bounds (α ≲ 0.2).
6. **a/40 QR α = 0.05 redo vs first run** differ by 1.7% (M slack of the first run: 0.0504 against 0.0500).
7. **B·n from the annulus** was verified against direct Biot–Savart at the start and end states of a/40; it was not rechecked at every state of every run.
8. Only **one equilibrium** (NCSX_fb_L16M16N12) and **one seed** per run; no Poincaré sections of the free-boundary states yet (island or chaos content of the end states is unchecked).
9. The time to converge a bound is dominated by the creeping of late steps (model conservative; ratios > 1): a cost issue, not an accuracy one, except where a stop criterion fired too early (§6).

## 8. What remains

- **Running jobs** (Stellar-AI `/scratch/gpfs/EKOLEMEN/lk1158/NEST_mf`): 6279 (a/60 GLTR, α = 0.05), 6281 (a/80 GLTR, α ≤ 0.2; will hit the 24 h limit, continue with `--start`), 6298 (a/80 QR, same), 6540 (16/16/16 α = 0.05 continuation), 6543 and 6544 (a/100, a/120 at α = 0.05).
- **Redo** the points flagged (!) with the current code (active-bound stop); rerun a/20 QR α = 0.1 if needed.
- **Study 1 plot with the force re-evaluated on the finest annulus** (`eval_common.py --frac 120`), for every state, to show the discretization error directly.
- **QR at L = M ≥ 24** needs the multi-GPU jaxmg split (the dense table is 62–96 GB per copy); GLTR covers study 2.
- **Hessians at h = 1e-9** for the remaining states (QR α = 0.5–0.2, a/60, the finished a/80 and a/100–a/120 runs, 16/16/16).
- **Poincaré sections** of selected end states (the saved `<out>_a<alpha>_eq.h5` is the moved geometry; `scripts/pointwise/poincare_state.py` traces on the annulus field and needs an update for moved geometries), plus p constant along field lines.
- **Solver improvements** found in the health review: exact Hessian–vector products of M inside GLTR (the majorizer's conservative curvature makes late steps tiny), a Lanczos preconditioner, an adaptive τ.
- **A study of the α < 0.05 end** (where the front turns steep and B·n branches) and of the multipliers as slopes of the front, if the physics of the tight limit is wanted.
- **The shareable page** (an artifact like the pointwise report's) once the numbers are final.

## 9. Files

| | |
|---|---|
| code | `nest/meanforce.py`, `scripts/meanforce/{mf_run,eval_common,collect,health,radial_force,plot_radial,bn_check}.py`, `scripts/meanforce/submit_studies.sh`, `tests/test_meanforce.py`, `nest/free_boundary.py`, `scripts/optimize_fb.py`, `run_sai_cyl.sbatch` |
| data | `results/meanforce/*.json` (runs, evaluations, Hessian checks), `*_a<alpha>.npz` / `_eq.h5` (saved states), `stall_*.json` (diagnostics), `log_summaries/` |
| figures, tables | `figures/meanforce/` |
| docs | this report, `docs/meanforce_results.md` (snapshot), `docs/meanforce_handoff.md`, `docs/meanforce_gltr_a40.md` (health review), `docs/mean_force_poisson_plan.md` (plan), `docs/pointwise_al_ncsx.md` (the pointwise study this builds on) |
