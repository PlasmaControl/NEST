# NEST cylindrical-annulus J → B: implementation and test plan

Branch `cylindrical-grid`, from `origin/main` 263bfad. Working copy: a git worktree at
`~/PycharmProjects/NEST/NEST-cyl`. The `container-mask` checkout in `NEST/NEST` is not touched.

Source proposal: `radial_grid.md` (Fourier in φ over one field period, Chebyshev in r and z,
James free-space correction per toroidal mode). This plan changes its default domain to an
**annulus** R_in ≤ r ≤ R_out, |z| ≤ z0, as agreed on 2026-10-16.

Conventions for the whole campaign:
- Commits are local only. The user pushes.
- Long runs stream per-iteration or per-stage progress (no silent `verbose=1` runs). When piping, use only `grep --line-buffered`.
- No Modal or GPU job without the user's approval. Each job is capped at 2 h and checked periodically.
- For conditioning or quadrature questions, look at how DESC does it first (e.g. `desc/integrals/singularities.py` for near-singular surface quadrature), then ask.
- Bitwise comparisons use `XLA_FLAGS='--xla_cpu_multi_thread_eigen=false intra_op_parallelism_threads=1'`, `OMP/OPENBLAS/VECLIB/MKL_NUM_THREADS=1` and `NEST_CACHE=off`.
- Main's default code paths must stay bit-identical. The new solver is opt-in only.

Status legend: `[ ]` open, `[x]` done, `[~]` done with a caveat recorded in the gate's notes.

---

## 0. Design decisions (fixed unless a gate says otherwise)

| # | Decision | Reason |
|---|---|---|
| D1 | Annulus Ω = [R_in, R_out] × [−z0, z0] × one field period. R_in = R_min − δ, R_out = R_max + δ, z0 = max\|Z\| + δ, measured on the support surface (ρ = 1, or ρ_c for container runs). | Removes the empty hole and the axis. It costs a fourth face in S_m. |
| D2 | Margin δ is a parameter, default 2 h. Gate A6 measures sensitivity for δ ∈ {1, 2, 4} h. | J must vanish near ∂Ω. That keeps σ and the corner expansion smooth. |
| D3 | r and z use Chebyshev–Gauss–Lobatto nodes, optionally Kosloff–Tal-Ezer mapped (parameter α, default 0 = unmapped). Gate A3/A6 decides α. | Mapping recovers part of the ~2.5× wall-clustering cost. The dense-operator code is identical either way. |
| D4 | φ uses N_φ **odd** uniform planes per period, φ_k = 2πk/(NFP N_φ). | No Nyquist ambiguity for the complex J_c spectrum. The φ = 0 plane and its mirror pairs exist for stellarator symmetry. |
| D5 | Components: A_z with m = j·NFP (rfft, j = 0..(N_φ−1)/2). A₊ with m = j·NFP + 1 via the FFT of J_c = J_r + iJ_φ (j = −(N_φ−1)/2..(N_φ−1)/2). | Section "NFP mode structure" of the proposal, checked by hand. |
| D6 | Per-mode Dirichlet core by matrix diagonalisation: interior radial operator L_r,m = V_r,m Λ V_r,m⁻¹ per distinct \|m\|, interior D_zz = V_z Λ_z V_z⁻¹ shared. | Haidvogel–Zang fast Chebyshev Poisson solver. Steps 3 and 7 of the proposal. |
| D7 | Free space by the James correction: σ = ∂_n u_D on 4 faces, g = S_m σ, harmonic lift by rank-structured outer products. | u_free = u_D + S[σ] in Ω (sign checked by hand). |
| D8 | Toroidal kernel k_m(r,z;r',z') = (1/2π)∫cos(mt)/\|x−y(t)\| dt = Q_{\|m\|−½}(χ)/(π√(rr')). Use elliptic integrals for \|m\| = 0, 1, forward recurrence where m_max·arccosh χ ≲ 1, and Miller backward recurrence otherwise. S_m[σ](x) = −½∫ r′ k_m σ dℓ′. | A t-FFT cannot resolve near-coincident pairs. Miller recurrence is the stable route for the minimal solution. |
| D9 | S_m entries are computed against the Chebyshev Lagrange basis of each face, by composite Gauss–Legendre on the inter-node panels. Panels are graded geometrically toward the target (same face) or the shared corner (adjacent face). | The log singularity is integrable, and graded panels reach near machine precision without analytic subtraction. |
| D10 | First version forms u on the full (N_r+1)×(N_z+1) grid and takes the curl with dense D_r, D_z and m/r. Fused back-transforms come later (E-stage). | Correctness first. The extra memory is small at these sizes. |
| D11 | Output: B̂ modes → Fourier synthesis on each collocation ζ-plane (exact) → 2-D barycentric Chebyshev interpolation in (r, z) as GEMMs. Cylindrical → Cartesian at the points. | Collocation points lie on ζ = φ planes. |
| D12 | float64 throughout. A float32 trial happens only in stage E. | Chebyshev eigenvector matrices are only moderately conditioned. |
| D13 | Precompute in NumPy/SciPy, cached to disk (npz) keyed by a hash of (R_in, R_out, z0, NFP, N_r, N_z, N_φ, α, code version), honouring `NEST_CACHE`. The solve is a JAX pytree like `flux_interp._Pytree`. | Linear in J, so JAX reverse mode and its transpose are exact. |
| D14 | J on the solver nodes comes from main's machinery: `map_to_flux` (2-D Newton per φ-plane), a cached `PointInterpolator` from the uniform flux grid, and one gather. Cylindrical components are used directly, with no Cartesian conversion. | It reuses tested code. The nodes are fixed, so the Newton solve runs once. |

New files (all opt-in):
- `nest/chebyshev.py`: nodes, mapped nodes, D, D², barycentric weights and interpolation matrices, Clenshaw–Curtis weights.
- `nest/toroidal_kernel.py`: k_m for all m at many (r, z, r′, z′).
- `nest/cylindrical_grid.py`: `CylindricalGrid` (axes, `around(inp, h, margin, alpha)`, `locate_plasma`, scatter/gather, weights).
- `nest/cylindrical_bs.py`: `CylindricalSolver` (precompute and cache, `__call__(J_cyl) → B̂ modes or B on grid`) and `CylindricalInterpolator` (to fixed points).
- `tests/test_cylindrical_bs.py` (A and B gates as test functions), `tests/cyl_convergence.py` (B4 / D studies).
- `docs/cylindrical_grid.md` (design and results), plus entries in `UPDATES.md`.

Edits to existing files (opt-in branches only):
- `nest/objective.py`: `Fields(..., kernel="cylindrical", grid=CylindricalGrid)` path. The existing paths are untouched.
- `nest/force.py`: `ForceResidual` accepts a `CylindricalGrid` (no `K_hat`).
- `scripts/optimize_qa.py`: `--grid cylindrical`, `--cyl-h`, `--cyl-margin`, `--cyl-alpha`.

---

## Stage A: scalar per-mode solver (CPU, NumPy)

Goal: a free-space solve of Δu = f for an NFP-periodic scalar f in Ω, exact to discretisation error, before any vector or NEST code.

### A0. Setup
- [ ] Worktree exists (the user runs `git worktree add ../NEST-cyl cylindrical-grid`). The sandbox cannot create `.git/worktrees/`.
- [ ] Env `nest` plus `mpmath` (reference toroidal harmonics). `PYTHONPATH=<worktree>/nest:/Users/singh/PycharmProjects/DESC`.
- [ ] Commit this plan as `docs/cylindrical_grid_plan.md`.

### A1. Chebyshev operators
- Test: D and D² on polynomials of degree ≤ N (exact) and on e^{x}, sin(5x) (spectral decay). Mapped nodes: the derivative of a mapped function is correct and convergence is spectral. Barycentric interpolation matrices reproduce the nodes, and off-node values are spectrally accurate.
- Pass: polynomial error ≤ 1e-12·N² relative. Mapped and unmapped errors fall to ≤ 1e-12 by N = 64 for sin(5x).
- False pass: tests on polynomials only, which are exact for any correct-looking D. Covered by also testing non-polynomial functions and the mapped case.
- [ ] done

### A2. Radial eigendecomposition per |m|
- Test: interior L_r,m = D_rr + (1/r)D_r − m²/r² on [R_in, R_out] for |m| ∈ {0, 1, NFP, …, m_max}. Check that eigenvalues are real and negative, that ‖V Λ V⁻¹ − L‖/‖L‖ is small, and record κ(V_r,m). Same for D_zz.
- Pass: |Im λ| ≤ 1e-10|λ|, λ < 0, reconstruction ≤ 1e-11. κ(V) reported against m. If κ > 1e8 at any m: stop and discuss (Legendre–Galerkin fallback in the proposal).
- False pass: only checking small m. Covered by sweeping to m_max for the a/20 precise_QA size (m ≈ 500).
- [ ] done

### A3. Dirichlet core
- Test (a): the eigen path against a dense direct solve of the same Kronecker collocation system, at small N (N_r = N_z ≈ 24), for several m. This isolates implementation errors from discretisation error.
- Test (b): manufactured u = sin(π(r − R_in)/(R_out − R_in))·(z0² − z²)·e^{z}·(non-polynomial in r) for several m, including m_max. Check spectral convergence in N.
- Test (c): a compact bump f (zero near ∂Ω). u_D converges spectrally, and σ on each face converges and → 0 linearly at the corners.
- Pass: (a) ≤ 1e-11 relative. (b) error ≤ 1e-10 by N ≈ 48. (c) σ self-converges to ≤ 1e-10.
- False pass: a manufactured u that is a low-degree polynomial is reproduced exactly by any consistent scheme, and an m-independent u hides m²/r² bugs. Both covered by the choice in (b).
- False fail: a corner singularity from an f that does not vanish at the corners. Covered by using a compact f in (c); (b) has u = 0 on ∂Ω and smooth f.
- [ ] done

### A4. Toroidal kernel k_m
- Test: (i) |m| = 0, 1 against the closed forms with K, E (`ellipkm1` near χ → 1). (ii) m ∈ {0, 1, 2, 5, 20, 100, 500, 1000} at χ − 1 ∈ {1e-12, 1e-8, 1e-4, 1e-2, 1, 10, 100} against `mpmath.legenq(m − ½, 0, χ, type=3)`. (iii) the continuity of the hybrid forward/Miller switch. (iv) the log-singular asymptotics Q ≈ −½ln((χ−1)/2) − γ − ψ(m + ½).
- Pass: relative error ≤ 1e-12 everywhere tested (≤ 1e-10 for χ − 1 ≤ 1e-8, where the argument itself loses digits). Compute χ − 1 directly as d²/(2rr′), never as χ − 1.
- False pass: testing only moderate χ, where both recurrences work. Covered by the χ and m grid above.
- [ ] done

### A5. Single-layer matrices S_m
- Test: (i) S_m σ against an independent 3-D brute-force surface quadrature of −(1/4π)∫∫ σ(ℓ′)e^{imφ′}/|x − y| dS, using adaptive `scipy.integrate` in (φ′, ℓ′) with breakpoints at the singular point, for ~10 targets (face interiors, near corners, at corners) and m ∈ {0, 1, NFP, NFP + 1, 4NFP + 1, large}. (ii) Edge consistency: a corner node's row agrees when built from either face. (iii) Convergence of the grading depth.
- Pass: (i) ≤ 1e-10 relative to max|S σ|. (ii) ≤ 1e-13. (iii) entries stable to 1e-13 with one more grading level.
- False pass: a σ that vanishes near the singular point hides a wrong singular quadrature. Covered by testing σ = 1 and σ = a Chebyshev polynomial nonzero at the target. A factor-2 or sign error in the normalisation would pass a relative or self-consistency test. Covered by the absolute 3-D comparison in (i).
- [ ] done

### A6. Scalar free-space solve (James complete)
- Test: f = Σ over NFP copies and two off-axis, z-asymmetric centres of narrow Gaussians (width s; tail ≤ 1e-16 at ∂Ω), so f is NFP-periodic but has no z or stellarator symmetry. The exact u is Σ −q erf(ρ/(√2 s))/(4πρ). Run FFT in φ, the per-mode solve and the inverse FFT.
- Checks:
  - (a) g = S σ against u_exact on ∂Ω. This isolates S.
  - (b) u inside Ω against u_exact at all nodes, and at random off-node points through barycentric interpolation.
  - (c) convergence in (N_r, N_z, N_φ).
  - (d) margin δ ∈ {1, 2, 4} h and mapping α ∈ {0, 0.5, 0.9, 0.99}: error against total points.
- Pass: (a) and (b) ≤ 1e-9 relative to max|u| at the resolved size, with spectral convergence in (c).
- False pass: with a symmetric source, z-parity or φ-sign errors cancel. Covered by the asymmetric centres. Too-wide Gaussians leak to the walls and give a false floor (false fail). Pre-flight: print max f on ∂Ω.
- **Stop point S1**: report A1–A6 (tables and the convergence figure), the recommended α and δ, and the precompute time and storage. The user confirms before stage B.
- [ ] done

---

## Stage B: vector J → B (CPU, JAX)

### B1. Vector assembly
- Implement J_c = J_r + iJ_φ and J_z → modes. A₊ and A_z by the stage-A solver (JAX port of the apply path, NumPy precompute). Curl as in proposal step 7, implemented per D10. Inverse FFT gives (B_r + iB_φ, B_z).
- Test: the JAX apply path against the NumPy stage-A path on the same scalar data, ≤ 1e-13.
- [ ] done

### B2. Manufactured exact field (interior included)
- Test: pick a compactly supported smooth vector ψ in Ω (bump in (r, z) × NFP-periodic φ dependence, all three cylindrical components, no symmetry). Set A = ∇×ψ, so ∇·A = 0, and J = −ΔA/μ0 (by JAX autodiff, compact). The free-space solution is exactly A, so B = ∇×A everywhere, including inside the current. Also run an axisymmetric Gaussian ring (J_φ only) against an elliptic-integral loop-quadrature reference.
- Pass: rms and max |B − B_exact|/max|B_exact| ≤ 1e-9 at the resolved size, with spectral convergence. The ring agrees to ≤ 1e-9.
- False pass: a purely toroidal J only exercises the m = 1 A₊ path with J_r = 0. Covered by ψ with all components and φ dependence. Comparing only exterior points hides interior errors. Covered by the exact interior B.
- [ ] done

### B3. Identities and linear algebra
- Test: div B (spectral, on the grid) ≈ 0. curl B − μ0 J against the discrete div J (J from B2 with a deliberately non-solenoidal perturbation). Linearity. Adjoint ⟨L J, b⟩ = ⟨J, Lᵀ b⟩ with `jax.vjp`. `jax.test_util.check_grads` on a small grid.
- Pass: div B ≤ 1e-10 relative. The curl residual tracks ∇(∇·A) as predicted. Adjoint ≤ 1e-12. `check_grads` passes.
- [ ] done

### B4. NEST's J (precise_QA, kinked edge) against main's box
- Setup: `precise_QA_free_boundary_L16N14` with its helical coils, the `BoundaryPotential` fit at production factors (3, 3, 2.5) via `fit_cache`, and B_plasma at the production collocation points.
- Comparisons:
  - (a) Cylindrical self-convergence over h_cyl ∈ {a/10, a/15, a/20, a/30}-equivalent sizes; a/30 is local if memory allows (~4 GB estimate).
  - (b) Main's VG box at a/10 and a/15 locally. The difference to the finest cylindrical B should shrink at the box's measured O(h²) rate (1.4e-3 at a/10).
  - (c) Exterior points (0.3–0.6 a from the LCFS) against `continuous_biot_savart`.
  - (d) J at the nodes exact (point evaluator) against the cached Lagrange path, to bound the shared flux-grid interpolation bias.
- Deliverable: B error against points, memory and wall time for both methods, interior by ρ band, and the cylindrical size that matches the box's a/20 error.
- Pass (go criterion for stage C): the cylindrical method reaches the box's a/20 interior error with ≥ 3× less peak memory, or the user decides otherwise after seeing the curves.
- False pass: both methods share the flux-grid interpolation of J and the same J kink, so their agreement does not prove either is right. Covered by (c) and (d) and by B2's exact test. A cylindrical "reference" that is not converged would make the box look converged. Covered by requiring cylindrical self-convergence at ≥ 2 sizes beyond the comparison size.
- False fail: an h-matching rule that under-resolves φ at R_out. Pre-flight: print node spacings over the plasma in all three directions.
- **Stop point S2**: present the B4 curves; go/no-go for integration.
- [ ] done

---

## Stage C: integration into NEST (CPU)

### C1. `CylindricalGrid`
- `around(inp, h, margin, alpha, support_rho=1)`, then `locate_plasma` (inside nodes, rtz), scatter/gather and quadrature weights (Clenshaw–Curtis × uniform × r).
- Test: the node-weight volume of the inside set against DESC's volume (staircase, O(h)). The toroidal current through a φ-plane from the gathered J against the net current I (section current, as on container-mask).
- [ ] done

### C2. `Fields(kernel="cylindrical")`
- J at the solver nodes by gather (D14), the solver, and the interpolator to collocation points. `ForceResidual` takes the grid.
- Test: `Fields` B_plasma against the standalone B4 path at the same points, ≤ 1e-13. F(x0) against main's VG `Fields` at a/10 and a/15; the difference must be consistent with B4's box error. The `ForceResidual.decompose` field term is reported.
- [ ] done

### C3. Tangent and adjoint
- Test: `_tangent` (Jv) against `jax.jvp` of `_compute`. Jᵀw by adjoint identity. Mirror main's `tests/test_objective.py` checks.
- Pass: ≤ 1e-12 relative (float64).
- [ ] done

### C4. G0: main paths unchanged
- Test: on origin/main 263bfad against this branch, deterministic flags, `NEST_CACHE=off`: `Fields` with hockney and truncated, F, Jv and Jᵀw for 2 configurations. Bitwise identical. Run the main test-suite (`scripts/time_tests.py`, local subset).
- False pass: comparing with caches on can reuse the old result. Covered by `NEST_CACHE=off`.
- [ ] done

### C5. Short local optimisation
- `optimize_qa.py --grid cylindrical` at small factors (1, 1, 1), 20 iterations, streaming per-iteration output. Run the same settings with the VG box. Compare trajectories and check the final state on a finer cylindrical grid and on the a/15 box.
- [ ] done

### C6. `DenseFields` compatibility (axisymmetric use)
- G (one solve per column) built through the cylindrical solver. Check against the box-built G on a small case to the B4 level.
- [ ] done

---

## Stage D: production-size validation (GPU, needs approval)

- **Stop point S3**: no Modal job before the user approves the job list, sizes and time caps.
- [ ] D1. Error against memory and time on A100: cylindrical against VG box at a/10, a/20, a/30 (box a/30 if it fits; else a/25). Instrumented per-stage XLA peak memory.
- [ ] D2. precise_QA production `optimize_qa` with the cylindrical backend (a/10-equivalent then refine). Check on the VG a/20 box with 3× collocation and on a finer cylindrical grid. Compare against main's anchor (8.2e-6 on a/20).
- [ ] D3. DSHAPE axisymmetric (files from branch `optimization`): N_φ = 1, so the solve is two 2-D problems (m = 0 for A_z, m = 1 for A₊). Compare the force floor against the G6-axi box results, including the E2 effect (16 against 32 ζ planes), which the cylindrical solve should not show.

---

## Stage E: optimisations (each only if a measurement calls for it)

- [ ] E1. Banded or low-rank storage of S_m at high |m| (required before a/30 precise_QA: dense S_m is 4.3 GB).
- [ ] E2. z-reflection split of S_m (×½ storage and work).
- [ ] E3. Stellarator symmetry in the solve (cos and sin with fixed z-parity, ×½).
- [ ] E4. Fused curl back-transforms (proposal step 7). Large GEMMs over all modes.
- [ ] E5. float32 trial against float64 at B4 sizes.
- [ ] E6. Mode truncation K < N_φ if J's φ-spectrum allows (measure it first).

Each E item: before/after error at fixed size (must be unchanged to 1e-12 except E5 and E6), memory, time.

---

## Stop points (the user decides)

- S1, after A6: α, δ, and precompute cost.
- S2, after B4: go/no-go for integration.
- S3, before any GPU/Modal job.
- Any change to a default of main's code paths.
- Any gate failing for a reason not anticipated here.

## Risks to watch

1. Sign or normalisation of S_m. Guarded by A5(i) and A6(a).
2. Eigenvector conditioning at large |m|. Guarded by A2.
3. Corner smoothness: holds only if f vanishes near ∂Ω. Guarded by D2's margin and A6(d).
4. Matched resolution: Chebyshev clustering and φ spacing at R_out. Guarded by B4's pre-flight spacing printout.
5. Precompute storage and time ∝ h⁻³ for dense S_m. Guarded by the E1 trigger.
6. Shared biases with the box (J interpolation, J kink). Guarded by B2, B4(c) and B4(d).
