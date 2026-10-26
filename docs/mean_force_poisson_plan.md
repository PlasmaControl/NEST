# Plan: the closest free-boundary state to DESC's current under a mean-force bound, with the Poisson pressure (NCSX)

2026-10-26 (revised). Plan only; nothing has been run for it. It builds on the pointwise study
(`docs/pointwise_al_ncsx.md`, branch `closest-al`), the branch `origin/pressure-poisson`, and NEST's free
boundary. Each requested point is addressed in its own section (§2–§10). §11 records the decisions, §12 the
order of work and the stop points.

Changes from the first version of this plan:
- **The solve is a NEST free-boundary solve** (§5), not fixed boundary.
- **The mean force is computed exactly as DESC computes ⟨|F|⟩_vol** (§3), with no other weighting.
- **The second convergence study raises L = M at N = 12** (§10). N counts modes per field period, and NFP = 3.

## 1. What changes, at a glance

| | pointwise study (done) | this study |
|---|---|---|
| equilibrium | NCSX_L12M12N8 (fixed-boundary re-solve) | **NCSX_fb_L16M16N12** (LCFS solved in the coils) |
| NEST basis (T and p) | 18/18/12 (factor 1.5) | **16/16/12** (factor 1) |
| boundary | fixed (DESC's LCFS) | **free**: the LCFS moves; B·n = 0 on it, Ψ held |
| constraint | \|F(x_i)\|/norm ≤ ε at every point | **mean force, as DESC computes ⟨\|F\|⟩_vol**, ≤ ε |
| pressure | free unknowns c | **eliminated**: p = p*(u, ξ) from the weak Poisson equation |
| unknowns | u (T), c (p) | **u (T) and ξ (boundary displacement)** |
| solvers | GLTR (matrix-free), jaxmg QR (dense) | the same two |
| studies | annulus a/20–a/80, basis N 12–20 | **annulus a/20–a/120**; **NEST L = M = 16, 20, 24, 28 at N = 12** |

B = B_ext (coils) + B_plasma(J), with B_plasma from the cylindrical annulus (free-space field).

## 2. Equilibrium and starting resolution

- **Equilibrium:** `desc_input/vmec_free_boundary/NCSX/NCSX_fb_L16M16N12.h5` in `NCSX_coils.h5`. On DESC's quadrature grid:

  | ⟨\|F\|⟩_vol / norm (DESC) | 99th pct (points / volume) | max (ρ ≤ 0.95 / all) | B·n/\|B\| rms (max) | pressure jump rms |
  |---|---|---|---|---|
  | 8.8e-4 | 1.7e-2 / 7.7e-3 | 2.4e-2 / 2.9e-1 (ρ = 1) | 4.5e-3 (2.8e-2) | 2.6e-3 |

- **NEST basis 16/16/12** for T and p (`--factors 1 1 1`).
- **Boundary displacement ξ:** M = 16, N = 12, the equilibrium's full boundary resolution (412 stellarator-symmetric modes). A lower ξ resolution leaves part of the coils' B·n unreachable, and the solver then cancels it with edge currents (lesson of the NCSX free-boundary runs).
- **Collocation:** 1.5 × (basis + equilibrium) in each direction. At 16/16/12 that is 25 × 97 × 37 = 89,725 points (half a period).
- **J_fit:** DESC's J fitted in the basis (`BoundaryPotential.fit`). It is carried with the moving coordinates, as `TransportedCurrentError` does.

## 3. The constraint: the mean force, computed as in DESC

**The quantity** is DESC's force error, exactly as `desc` defines it:

$$
M = \frac{\langle |F| \rangle_\text{vol}}{\langle |\nabla |B|^2| / 2\mu_0 \rangle_\text{vol}},
\qquad \langle q \rangle_\text{vol} = \frac{\int q\, dV}{\int dV},
\qquad F = J\times B - \nabla p .
$$

- **Normalization:** DESC's own `<|grad(|B|^2)|/2mu0>_vol` of the starting equilibrium. It is the same number NEST already uses (`ForceResidual.norm`), held fixed during the solve.
- **Integral:** the volume integral in the current (moved) geometry. NEST's collocation is DESC's own kind of rule (Gauss–Legendre in ρ, uniform in θ and ζ, weights |√g| dρ dθ dζ) at NEST's resolution. **No other weighting** enters the value that is constrained or reported: no 1/|F| weights, no point counting, no edge exclusion.
- **Gate:**
  - Put DESC's own J, B and ∇p on NEST's collocation points and integrate them this way. The result must reproduce DESC's `<|F|>_vol` from its QuadratureGrid to < 1% (8.8e-4).
  - The same comparison on DESC's QuadratureGrid itself is reported.
  - The reference M_DESC is DESC's own value.

**The bound:** M(u, ξ) ≤ ε = α M_DESC, swept over **α < 1**: α ∈ {0.7, 0.5, 0.3, 0.2, 0.1, …}, continued from one α to the next, with adaptive retries at smaller steps as in the pointwise study, until a bound can't be met.
- The continuation may pass through α = 1 on the way down if the start is above DESC's mean. That state is not one of the targets.
- The sweep gives the Pareto front, ‖J − J_fit‖ against M.

**Inside the solver only.** |F| is smoothed, √(|F|² + τ²) − τ with τ = 10⁻³ ε norm, which changes M by less than 10⁻³ ε. It is turned into a least-squares surrogate by iteratively reweighted least squares (§4). These are devices for the step computation. Every acceptance test, every ratio and every reported number uses the plain M above. The rms force ⟨|F|²⟩^{1/2}/norm is reported alongside.

## 4. Constraints, multipliers and the inner least squares

**The problem:**

$$
\min_{u,\,\xi}\ \tfrac12\|g\|^2 \quad\text{s.t.}\quad M \le \varepsilon,\quad \frac{\|B\cdot n\|_\text{rms}}{|B|} \le \beta,\quad \Psi = \Psi_\text{DESC},
$$

with g = √w (J − J_fit)/rms J_fit (the volume rms of the change in J).

- **β** is the B·n target on the moving boundary: **DESC's own free-boundary value at 16/16/12, B·n/|B| = 4.53·10⁻³ rms** (`NCSX_fb_L16M16N12_diagnostics.json`, final). The constraint is ‖B·n/|B|‖_rms ≤ β, with B·n/|B| computed on the same boundary grid as DESC's diagnostic (a step-1 gate checks that NEST's B·n/|B| reproduces 4.53·10⁻³ on DESC's own state and field).
- **Ψ** is the toroidal flux inside the boundary. It is held so the boundary cannot slide to a neighbouring surface.

**Augmented Lagrangian with three scalar multipliers.**
- Unlike the pointwise study, there are only three constraints, so ρ stays moderate. Stopping: each constraint met to 1%.
- **The multipliers report the front's slopes:** ν_F = −dΦ*/dε, and ν_B is the price of the B·n target.

**Inner problem: a sum of squares, so both trust-region solvers apply.** Its rows:
- the J rows g;
- the force rows √(ν̃_F w_i / (norm √(|F_i^k|² + τ²))) F_i, from the concave majorizer of √(|F|² + τ²) at the current iterate. Every surrogate decrease is a true decrease, and the weights are refreshed at every accepted step;
- the B·n rows (already a sum of squares), with weight ν̃_B;
- the Ψ row.

ν̃ are the AL's effective weights, max(0, ν + ρ c)-type, held fixed within an inner solve.

## 5. NEST free-boundary solve

**What NEST's free boundary does** (`nest/free_boundary.py`, `scripts/optimize_fb.py`):
- **Unknowns.** The boundary's normal displacement ξ(θ, ζ) is a new set of unknowns. It is extended inside as ρ^|m| per mode, so the flux coordinates move with it.
- **What moves with it.** √g J^i and p move with the coordinates. So div J = 0, J = 0 and p = 0 on the moving boundary, and the net current, hold for any ξ.
- **Pressure balance holds identically.** There is no surface current, and B is continuous across the boundary. What remains of the free-boundary problem is B·n = 0 (`BoundaryNormalField`) and Ψ held (`ToroidalFlux`).
- **Exact in ξ:** the geometry at all points (`MovingGeometry`), hence J, ∇p, the volume, the normal and the interpolation positions.
- **First order in the displacement:** B_ext at the moved points and the field solver's samples of J (`BoxShift`).
- **Re-anchoring** (`Displacement.apply`) makes everything exact. The geometry is rebuilt on the moved boundary and ξ is reset to 0, when |D| exceeds a step limit (1–3 mm, the setting that worked on NCSX).

**Annulus or box.**
- `main`'s free boundary uses the Cartesian box.
- The annulus version (`CylindricalGrid`, cylindrical `BoxShift`, `MovingInterpolator` on the annulus, multi-GPU mode split) exists **only as uncommitted work in `/Users/lee/codes/NEST_cylfb`** (branch `cyl-free-boundary`, in the middle of a merge: `UPDATES.md` unmerged).
- The plan ports its `nest/free_boundary.py` annulus parts and their hooks in `cylindrical_bs.py`, `cylindrical_grid.py`, `flux_interp.py` and `objective.py` into the new branch, after diffing them against `main`. `NEST_cylfb` itself is not touched. Confirmed by you as the version to use.
- **Gates** (from that worktree's tests):
  - ξ = 0 reproduces the fixed-boundary `Fields`;
  - the ξ derivative against central finite differences;
  - the adjoint identity;
  - second-order agreement with a re-anchored rebuild.

**B·n.**
- The box's B·n was too inaccurate (1.5·10⁻³ |B| at a/10, uneven between field periods), so it had to come from a direct Biot–Savart field (`DirectField`).
- The annulus is far more accurate near the boundary, and earlier annulus runs took B·n from it.
- The plan uses the annulus B·n in the solve, and checks it with `DirectField` at every final state. If the two disagree by more than β/3, B·n switches to `DirectField`.

**With the Poisson pressure.** The pressure space moves with the geometry, so the projector Π = Π(ξ) changes with ξ (§6).

**Both solvers on the moving geometry:**
- **Matrix-free (GLTR).** `FreeBoundaryFields` gives J v and Jᵀw for (u, ξ). The reduced residual (I − Π(ξ)) r(u, ξ) is differentiated by JAX through the least-squares solve, so the ∂Π/∂ξ term of the variable projection is included exactly. Each product costs one annulus solve.
- **Dense (QR).** Within an anchor, the u columns come from the tables (exact: quadratic in u at fixed geometry). The 412 ξ columns are recomputed by Jacobian–vector products at the current iterate, refreshed at every accepted step. The trust-region ratio uses the true residual, evaluated matrix-free. At each re-anchoring the tables (A, G, pressure columns) are rebuilt on the moved geometry: n_u annulus solves, about 1.5 min at a/40 for 16/16/12.
- **Device memory:** every anchor's arrays are freed (`release`). Earlier runs leaked about 4 GB per anchor through JAX's caches otherwise.

**Current profile.** Earlier free-boundary runs needed I(ρ) held (`--fix-I`), or the core iota drifted (0.35 → 0.44), but they had no J-distance objective. Here ‖J − J_fit‖ is the objective, which should hold the profile. iota(ρ) is reported at every anchor, and I(ρ) is left free and monitored (your choice).

## 6. The pressure from the Poisson equation (branch `pressure-poisson`)

**What the branch does.** It solves p from the weak Poisson equation of force balance, with the force rows' weights, on the plasma volume with p = 0 on its boundary:

$$
\int w\,\nabla q\cdot\nabla p = \int w\,\nabla q\cdot (J\times B)\quad\forall q\in V_h .
$$

- This is the normal equation of min_p ‖√w (J×B − ∇p)‖. So p is the weighted least-squares projection of J×B onto the gradients of the pressure space, and the remaining force is F = (I − Π)(J×B).
- The minimizer is the same as with p free in the same space, but the reduced problem is 10–20× better conditioned (measured on the branch).
- The branch does this in `HarmonicModel` (the dense DSHAPE path, `--speed elim`, tests T1–T8). It has no matrix-free version (deferred) and no 3-D `Fields` or free-boundary version.

**How it is brought in.**
- **Branch.** A new branch `closest-poisson` from `closest-al`, with `origin/pressure-poisson` merged in (its plan, tests and code present) and the annulus free boundary ported (§5).
- **The formulation is reimplemented in `nest/pointwise.py`**, for both models:
  - **Π from a thin QR** of the weighted pressure columns P_w = √w ∇φ_j at the current geometry. A rank check drops columns below 10⁻¹² of the largest singular value. Recomputed when ξ changes (cheap: pressure columns need no field solve; a QR of 3n × n_p, about 1 s at 16/16/12).
  - **The weights in Π** are the volume weights w of the force: the plain L2 Poisson equation. M's IRLS weights are not used in Π, so the pressure is the physical Poisson pressure of the current state.
  - **Dense steps.** As the branch, at fixed geometry: QR of the stacked [A_p | A_u | ξ-cols | r]. Its trailing block is the projected system.
  - **Matrix-free steps.** Projected products as in §5.
  - **After each accepted step** the pressure is p = p*(u, ξ) exactly.
- **Gates** (CPU, random tables as in `tests/test_pointwise.py`, then one NCSX state):
  - **T1.** Trailing block = explicit projection (10⁻¹⁰).
  - **T2.** Galerkin identity P_wᵀ r = 0 after every accepted step.
  - **T3.** The reduced gradient (including ∂Π/∂ξ) against finite differences.
  - **T4.** Manufactured Poisson: J×B = ∇q with q ∈ V_h gives p* = q, F = 0.
  - **T5.** GLTR and QR reach the same optimum.
  - **T6.** A joint (u, c) optimum from the old code is a fixed point of the reduced problem.

## 7. The two solvers

Both act on (u, ξ), in the physical metric: J for u (H_J, as before), and the boundary displacement in metres for ξ, scaled so its columns match the current's (the scaling that fixed the damping floor in the NCSX runs). Same radius rule, same starts, targets and stopping, so states, Hessians and costs compare directly.

- **Matrix-free GLTR.**
  - Lanczos trust region; one annulus solve (and transpose) per product.
  - It needs no tables, so every re-anchoring is cheap.
  - It is run where a product costs a few seconds or less (§9).
- **Dense multi-GPU QR (jaxmg).**
  - The stacked QR over 2 GPUs (`run_mg.sbatch`, excluding `sai-n01g4`); DESC's exact-QR trust-region step.
  - New: each process keeps only the table rows of its own points, which the L = M = 28 tables and a/120 need. Tables are about 48 GB per array at 28/28/12.

## 8. Hessian eigenvalues, Gauss–Newton share, trust-region radius, stall diagnosis

**Hessian at every converged state** (each α, each resolution, each method):
- The exact Hessian of the Lagrangian ½‖g‖² + ν_F M + ν_B B + ν_Ψ Ψ in (u, ξ), with p = p*(u, ξ) inside F.
- Assembled by forward-mode differentiation of the reduced gradient (n_u + 412 ≤ 12,342 columns), in the metric, and checked against finite differences of the gradient.
- Reduced to the tangent space of the active constraints (the null space of their gradients).
- Reported: negative and near-zero counts, the smallest/largest ratio, the Gauss–Newton part, and the composition of the softest directions (current along or across B, radial location, boundary modes).
- Expectation: with p eliminated, the pointwise study's 2,000-odd flat pressure directions disappear.

**Logged at every step:**
- Δ, |y|, μ, actual/predicted ratio, accepted or not;
- the Gauss–Newton share of the reduced residual (QR: |c|²/(|c|² + r⊥²); GLTR: the μ = 0 Krylov model);
- multipliers, M, B·n, Ψ, ‖g‖, the rms force, the largest ξ;
- the time and Lanczos iterations.

**Stall detector:** Δ < 10⁻⁸ in the metric with GN share > 0.9, over 5 steps or 3 rejected trials in a row. It saves the state and `stall_<run>_<k>.json` with:
1. **Jacobian correctness.**
   - J d against central finite differences at h = 10⁻²…10⁻⁸ |x|, for the step direction and 3 random directions. Separately for the u, ξ and projector parts, since the ξ part is first-order within an anchor and expected to differ at second order.
   - The adjoint test.
   - The Galerkin identity.
2. **The model along the step direction d.** For t = Δ·10^k, k = −4…4, compare:
   - the true objective;
   - the Gauss–Newton model;
   - the second-order model (with Hessian–vector products).

   Where each leaves the truth tells the cause:
   - the second-order model holds well beyond Δ: Gauss–Newton's neglected curvature limits the radius;
   - neither holds: nonlinearity, the IRLS reweighting, or the first-order ξ treatment limits it;
   - the models disagree with finite differences at small t: a Jacobian error.
3. **Smoothing and weights:** points with |F_i| ≲ τ, and the spread of the IRLS weights.
4. **Free boundary:** the size of ξ against the re-anchoring limit (whether a re-anchor would restore exactness).

Findings are reported. A change of method (for example an exact-Newton trust region with Hessian–vector products in GLTR) is made only with your agreement.

## 9. Convergence study 1: annulus spacing a/20 → a/120

**Runs.**
- **Only the cylindrical annulus**, no Cartesian box anywhere: spacing h = a/20, a/40, a/60, a/80, a/100, a/120 at NEST 16/16/12, with the planes following h (arc spacing h at the outer wall).
- Each spacing is a full free-boundary solve. The a/40 solve starts from DESC's state; the others start from the a/40 solution, so fewer re-anchorings are needed.

**Cost.**
- One annulus solve takes 0.02 s at a/40 and 0.18 s at a/80, extrapolated to about 1 s at a/120.
- Per anchor, the tables take 1.5 min (a/40) to about 1.2 h (a/120).
- GLTR products: about 1 s per solve at a/120, i.e. minutes per trust-region step.
- A probe of a/100 and a/120 (solve time, memory: about 4.8·10⁸ nodes at a/120) comes first.

**Methods.** QR at every spacing; GLTR at a/20, a/40, a/60 (a/80 if time allows).

**Plots** (x = 1/h, one curve per α; α = 0.5 and the smallest α met at every spacing):
1. ‖J − J_fit‖₂ (volume rms / rms J_fit).
2. The mean force M (as in DESC).

Each plot shows each state's own value and its value **re-evaluated with the a/120 field** (one solve per state), with the successive differences on a log scale. Also reported per spacing: B·n rms on the boundary (`DirectField` check) and the boundary displacement.

## 10. Convergence study 2: NEST spectral resolution, L = M = 16 → 28 at N = 12

**Why L and M.** N is the number of toroidal modes per field period. With NFP = 3, N = 12 already reaches toroidal mode number 36, and it matches the equilibrium's N = 12. In the pointwise study, raising N from 12 to 20 moved the front by less than 10%. The radial (L) and poloidal (M) structure, such as the edge current and the island regions, is the more likely limit.

**Runs.** NEST basis 16/16/12, 20/20/12, 24/24/12, 28/28/12 for T and p together, at the annulus spacing study 1 finds converged (a/40 or a/60):

| NEST basis | n_u | n_p | points | A or G, each |
|---|---|---|---|---|
| 16/16/12 | ≈ 4,200 | 1,917 | 89,725 | 9 GB |
| 20/20/12 | ≈ 6,340 | 2,893 | 112,924 | 17 GB |
| 24/24/12 | ≈ 8,910 | 4,069 | 138,787 | 30 GB |
| 28/28/12 | ≈ 11,930 | 5,445 | 167,314 | 48 GB |

Plus 412 ξ unknowns throughout. One single check, 16/16/16, confirms the insensitivity to N at NFP = 3.

**Comparing across bases.** J_fit and the collocation grid change with L and M. So:
- every state is evaluated on the collocation points of 28/28/12, with that study's annulus;
- both ‖J − J_fit‖ (with each basis's own fit) and ‖J − J_DESC‖ (one fixed target) are plotted;
- the mean force M is plotted, as in DESC.

**Plots:** x = L = M (16, 20, 24, 28), same y quantities as study 1, both methods.

## 11. Decisions (2026-10-26)

- **Grid:** only the cylindrical annulus, no Cartesian box anywhere. Spacing a/20 → a/120 with the planes following h.
- **α:** a sweep with α < 1 (§3).
- **Branch:** `closest-poisson`, commits local only.
- **Annulus free boundary:** port the uncommitted code of `/Users/lee/codes/NEST_cylfb` into the new branch (that worktree untouched).
- **B·n target:** β = DESC's own, 4.53·10⁻³ rms.
- **I(ρ):** free, monitored (iota per anchor).
- **Pressure space:** the same resolution as T, varied with it in study 2.

## 12. Order of work and stop points

1. Branch; merge `pressure-poisson`; port the annulus free boundary; the reduced Poisson formulation and the DESC-mean gate. Gates T1–T6 and the free-boundary gates on the CPU. **Stop: test results.**
2. Tables for NCSX_fb at 16/16/12, a/40. Start metrics: M (as in DESC) with the Poisson p, B·n, Ψ. One free-boundary run with each method, at the first target of the sweep only (α = 0.7), with the stall detector on.

   **Check-in with you: optimization health review, before any sweep or study.** I stop here and we look at the two runs together. For each method, a figure and a short table with:
   - **Trust region:** the radius Δ and step |y| per step; the actual/predicted ratio, and the fraction of steps with ratio > 0.75 and < 0.25; accepted vs rejected steps; where steps end on the boundary; μ (how often the undamped Gauss–Newton step fits).
   - **Gauss–Newton share** per step, and whether it falls as the inner solves converge or stays near 1 while the radius collapses (the pointwise study's pattern).
   - **Stall detector:** whether it fired, and if so the saved diagnostic (Jacobian against finite differences, the model along the step direction, smoothing and weights, ξ against the re-anchoring limit).
   - **Constraints and multipliers:** M, B·n, Ψ and their multipliers per outer iteration; the penalty ρ, which should stay moderate (it reached 10¹² in the pointwise failures).
   - **Free boundary:** the re-anchorings (count, displacement per anchor, residual drop at each), iota(ρ) per anchor.
   - **Pressure:** the Galerkin identity P_wᵀ r = 0 at accepted steps.
   - **Cost:** time per step (Lanczos iterations for GLTR), table rebuild time per anchor.
   - **The two methods against each other:** the final ‖J − J_fit‖, M and B·n of the two states.
   - **Hessian at the final state:** the sign test on the tangent space.

   **We decide together** whether the optimization is healthy enough to run the sweep and the two studies as planned, or what to change first (settings, the step model, the re-anchoring limit, the smoothing).
3. The a/100 and a/120 probe.
4. Study 1: QR at six spacings, GLTR at a/20–a/60, as parallel jobs.
5. Study 2: both methods at L = M = 16, 20, 24, 28 (plus the 16/16/16 check).
6. Hessians at every converged state; the four convergence plots; B·n checks with `DirectField`; Poincaré sections of the free-boundary states; a report in the style of `docs/pointwise_al_ncsx.md`, and an updated web page if wanted.

**Rough budget.** The free boundary multiplies the dense method's cost by the number of re-anchorings, since the tables are rebuilt at each one (about 4–8 per run when started from a converged neighbour). That is about 1–2 GPU-hours per QR run at a/40, and up to about 10 at a/120. GLTR at a/20–a/60 is a few hours per run. Jobs run in parallel within the queue limits (10 per QOS).
