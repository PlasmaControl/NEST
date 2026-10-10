# Updates

One entry per commit, newest first.

## 2026-10-09: Solve driver, memory in the progress lines
- `nest/run_solve.py`: the combined objective FieldDistance + lambda^-1 ForceBalance by lambda continuation (lambda_ref = ||r_F(x_0)||^2 / dtol^2, nlam values over the given decades, from field-distance dominated down, warm-started) from the fitted current and the force-optimal pressure; every end state is checked on an independent grid QuadratureGrid(f L, f M, f N) by `ForceResidual.evaluate` of its series (one Biot-Savart application, no D_B). Writes run_solve.json and x_<k>.npy after each lambda. Runs with the existing job scripts: `sbatch run.sbatch nest/run_solve.py ARGS`.
- `nest/grid.py`: `device_memory()` (XLA peak / in use on the GPU, the host's peak resident size); in the D_B progress lines and, through the driver's callback, in every GN line.
- `nest/force.py`: `ForceResidual.average` of an empty radial bin is NaN without a warning (low resolutions).
- Local check (bases 4, 4, 3, h = a/4, 2 lambdas, evaluation grid 3x): 55 s on the laptop; lambda/ref 10 and 0.1 stop at the rounding level, <|F|>/F_0 1.54e-3 and 5.84e-4 on the solve grid, 1.57e-3 and 6.17e-4 on the evaluation grid; field distance 1.65e-3 and 1.97e-3.

## 2026-10-09: Pressure: one edge grouping, force-optimal fit
- `nest/pressure.py`: `_edge_groups` (the (m, n) groups and their lowest-l modes) shared by `impose_zero_edge` and `free_coefficients`; `Pressure.fit_force(rtz, dV, F, G=None)`, the force-optimal pressure for F = J x B (linear least squares on `p_columns`), next to `fit` (DESC's p(rho)).
- `nest/fields.py`: `fit_pressure` calls `Pressure.fit_force`.
- Equivalence with the previous commit: `impose_zero_edge` and `free_coefficients` identical (random coefficients, basis 6, 6, 4); `fit_pressure` 8.1e-15.
- Tests: `test_pressure`, `test_fields`, `test_optimize` pass.

## 2026-10-09: Fields built on ForceResidual; structured term Jacobians; one copy of the D's
- `nest/fields.py`: `Fields` is now a `ForceResidual` (its points, weights dV, B_ext, normalization, `metrics`, `decompose`, `desc_fields` and Biot-Savart map), plus the affine maps. The D's are stored once, as JAX arrays (`data()` returns them, no copies); D_B is filled in place (donated `dynamic_update_slice`). `state(x, data)` (was `evaluate`, which is `ForceResidual`'s), `objects(x)` (the `BoundaryPotential` and `Pressure` of a state), `report(x)` (`metrics` at a state).
- `nest/objectives.py`: the Jacobian of a term is d r_p / d f_p (one jvp per field component, all points at once) contracted with the D's: O(m n), as fftq's structured force Jacobian, instead of forward-mode products through D_B u (O(m n^2)); terms declare the fields they use (`uses`). `ForceBalance` is normalized by `ForceResidual.norm` (DESC's <|grad(|B|^2)| / 2 mu0>_vol; before, its rms, 1.10x larger on the test case); targets from `desc_fields`; `summary` adds DESC's measure <|F|>_vol / F_0.
- `nest/desc_input.py`: `B_ext(x, rtz=)` (a field defined through flux coordinates, `DESCExternalField.at_flux`, needs no inverse map); `nest/force.py`: `ForceResidual` uses it, and `desc_fields(names)` takes any DESC quantities.
- Equivalence with the previous commit (4, 4, 3 precise_QA, terms field, force, pressure 0.5, current 0.3 with the previous normalizations): residual 5.1e-14, Jacobian 9.5e-16; points, dV, B_ext 7e-16 to 1.3e-15; `fit_pressure` 8.1e-15. Synthetic D's, float64: structured vs forward-mode Jacobian 2.5e-16 to 2.9e-16, 1.47 s -> 0.06 s at 30,000 x 1,800 and 16.5 s -> 0.48 s at 60,000 x 3,500.
- Tests: `test_fields`, `test_optimize`, `test_force` pass (`test_force` iteration 0 still 4.66e-4).

## 2026-10-09: One geometry for every series; evaluate_chunked with per-point arrays
- `nest/desc_input.py`: `covariant_basis` (e_rho, e_theta, e_zeta and sqrt(g) of the geometry series, JAX), `dual_basis`, `current_from_sqrtg` (J = sum_i sqrt(g) J^i e_i / sqrt(g)); `DESCInput.basis_vectors(rtz)` (tangent and dual basis, sqrt(g), positions, in Cartesian components).
- `boundary_potential._current`, `current_potential._J_from_T` and `pressure._p_and_grad` use them instead of their own copies of the geometry; `fields.py` uses `basis_vectors` and `current_from_sqrtg` instead of its own geometry and assembly.
- `nest/cartesian_grid.py`: `evaluate_chunked(..., extra=)` (arrays along the points, sliced and padded with the chunk; e.g. precomputed geometry), used by `fields.py` instead of its own chunk loops; `cartesian_to_cylindrical` (used by `desc_external_field.py`); `objectives.py` uses `cylindrical_to_cartesian`.
- Equivalence with the previous commit (5000 random points, 50 on the axis, fitted 4, 4, 3 series of the smooth-current precise_QA): `_current` 4.4e-16 (axis points identical); `_p_and_grad` p identical, grad p 2.5e-16 (rho = 0 non-finite before and after, as sqrt(g) = 0 there); `Fields` arrays D_J, J_0, D_p, x, dV, E, sqrt_g identical, D_gp 5.7e-16, G 2.1e-16, B_0 1.1e-16, D_B 9.1e-14, J_box 4.6e-18.
- Tests: `test_current_potential`, `test_boundary_potential`, `test_pressure`, `test_smooth_edge`, `test_cartesian_grid`, `test_fields`, `test_optimize`, `test_force`, `test_desc_grid`, `test_grid` pass (26).

## 2026-10-09: One spectral chain in magnetic_field.py; Vico-Greengard the default kernel; scatter and gather in JAX
- `nest/magnetic_field.py` now holds the whole chain, built from NEST's own pieces: `vico_greengard_kernel_hat` (G on the rfft modes, zero on the Nyquist planes of k) beside `biot_savart_kernel_hat` (unchanged), `kernel_hat`; `b_hat_free` (the B_hat formula, factored out of `b_plasma_free`); `b_plasma_free(..., kernel=)`, Vico-Greengard by default; `interpolate_hat`, pure JAX, from the rfft half spectrum (the full spectrum rebuilt by Hermitian reflection, `full_spectrum`, or with `half=True` the NUFFT on the half spectrum itself); `interpolate` = rfftn + `interpolate_hat`; `BiotSavart` = scatter -> fft_J -> b_hat_free -> interpolate_hat, moved here from `biot_savart.py` (removed).
- `nest/grid.py`: `scatter`, `gather` in JAX (`CartesianGrid.scatter/gather` call them; module functions take the indices as an argument, for jitted code). `nest/fourier.py`: `wavevectors(full=)` removed again.
- `nest/force.py`: `ForceResidual.B_plasma` through `BiotSavart` (`kernel`, default vico_greengard; `dtype`, default float32 as before; `bs_kw`).
- Equivalence with the previous code (commit 384336e, even 48x40x36 and odd 49x41x35 boxes): `scatter`, `gather`, `biot_savart_kernel_hat`, `b_plasma_free(kernel="hockney")`, `b_plasma_periodic` identical; `interpolate` of the box B 1.6e-15 to 3.0e-15, of a random field (Nyquist content included) 5e-13 to 3e-12; `BiotSavart` hockney 9e-16; vico_greengard 9e-16 (odd) and 4e-10 (even, the Nyquist modes, now zero).
- `half=True` halves the spectrum and the NUFFT's grid, but the k_z = 0 modes sit at the edge of the NUFFT band: at upsampfac 1.25 it needs eps = 1e-14 (Gaussian blob 1e-10 to 4e-10 vs exact, about twice the interpolation time); off by default.
- Tests: all pass (`test_biot_savart` (+ half spectrum), `test_fourier`, `test_fields`, `test_optimize`, `test_force`, `test_cartesian_grid`, `test_reconstruction::test_manufactured`, `test_magnetic_field`, `test_grid`). With the Vico-Greengard default the manufactured bump converges at rate 5.0-5.3 (2.2e-6 at h = w/20); `test_force` iteration 0 is 4.66e-4.

## 2026-10-09: Objectives and the Gauss-Newton trust region
- `nest/objectives.py`: `ObjectiveFunction`, sum_i w_i ||r_i(x)||^2 over terms built from the fields, with volume weights sqrt(dV / V): `ForceBalance` ((J x B - grad p) / rms |grad(|B|^2)| / 2 mu0 of DESC), `FieldDistance`, `CurrentDistance`, `PressureDistance` (targets default to DESC's fields at the points); new terms subclass `Term`. Jacobian by forward-mode products through the affine fields in column chunks (no Biot-Savart call); `blocks` gives the rows by blocks of points.
- `nest/optimize.py`: `gauss_newton`, DESC's trust region (exact Cholesky or QR subproblem, column scaling, radius update) with step acceptance at rho > 0.1 and a stop at the rounding level; `points_per_block` reduces the rows to J^T J (cho) or R, z (qr, TSQR) without forming J; one `GN` JSON line per trial step. `continuation` over term weights, e.g. lambda in ||r_D||^2 + lambda^-1 ||r_F||^2.
- `tests/test_optimize.py` (precise_QA, bases 4, 4, 3, h = a/4): Jacobian vs central difference 1.2e-14, row blocks vs dense 1.0e-15; lambda continuation over 4 decades, each solve stops at the rounding level and the force term falls 3.5e-3 -> 9.9e-4 -> 6.2e-4; the QR step with row blocks reaches the Cholesky end state to 1.6e-8. With no pressure term in the objective, p departs from DESC's by up to 20 % as lambda falls.

## 2026-10-09: J, B, p, grad p as affine maps of the coefficients; DESC external field
- `nest/fields.py`: `Fields`, at the collocation points (default QuadratureGrid(2 L, 2 M, 2 N) of the current basis) J = J_0 + D_J u, B = B_0 + D_B u, p = D_p x_p, grad p = D_gp x_p, exact and constant; D_B = Biot-Savart of each coefficient's current on the box (one application per column, progress by elapsed time). Opt-in D_B cache (`cache_dir`): column blocks keyed by the geometry, box, kernel, points, basis and edge power; an interrupted build resumes. `fit_pressure`: the force-optimal pressure at a given current (linear least squares). `B_direct`: B without D_B (independent evaluation grids).
- `nest/boundary_potential.py`: `_sqrtg_J_columns` / `sqrtg_J_columns`, sqrt(g) J^i of single coefficients in closed form (cost independent of the number of modes; equal to the jvp of `_current` to 6.5e-15), `coefficient_modes`, `sqrtg_J`.
- `nest/pressure.py`: `pressure_function` (p linear in coefficients passed as an argument), `free_coefficients` / `to_coefficients` (without `edge_order`, the lowest-l coefficient of each (m, n) is fixed by p = 0 on the edge), `p_columns` in closed form.
- `nest/desc_external_field.py`: `DESCExternalField`, B_DESC - BS_h(J_DESC) as a field object to pass as the external field of a `DESCInput` (verification: B = B_DESC exactly for J = J_DESC at every h).
- `nest/grid.py`: `progress(..., unit=)`.
- `tests/test_fields.py` (precise_QA, bases 4, 4, 3, h = a/4): affine maps vs direct evaluation J 4.7e-16, p 6.0e-17, grad p 4.1e-16, B 8.6e-16; D_B cache identical (25 s build, 1 s restore); force-optimal p lowers rms |F| / rms |grad p| from 7.40e-3 to 6.78e-3; DESCExternalField reproduces DESC's B to 7.6e-13.

## 2026-10-09: B_plasma as a linear, differentiable map of J; Vico-Greengard kernel
- `nest/biot_savart.py`: `BiotSavart`, J at the box points inside the plasma -> B_plasma at fixed target points, one jitted JAX function (scatter, FFT, kernel, type-2 NUFFT straight from B_hat, with no inverse FFT to the box and back), linear in J: jvp, vjp and linear_transpose work, and the Jacobian columns of B are `apply` of one coefficient's current. Constants (inside indices, kernel, target angles) are a pytree passed as a jit argument (`consts()`). float64 by default (the solve path needs it), float32 optional.
- Two kernels: `hockney` (the discrete sum of `b_plasma_free`, second order in h) and `vico_greengard` (truncated free-space Green's function, Vico, Greengard and Ferrando 2016: spectrally accurate for the sampled J), R_cut = diagonal of the half-period box, precompute grid max(2 L_i, 2 R_cut) per axis, Nyquist planes removed; boxes of odd or even sizes.
- `nest/fourier.py`: `wavevectors(..., full=True)` for the full spectrum.
- `tests/test_biot_savart.py`: Gaussian blob vs its exact field (vico_greengard 5e-10 to 7e-10 already at s = 2 h; hockney 4.1e-2 -> 1.1e-2 when h halves, order 1.98); hockney equals `b_plasma_free` + `interpolate` to 4e-15; linearity 2e-13, jvp, vjp, linear_transpose and vmap to round-off, adjoint identity 1e-12 (upsampfac 1.25); float32 3e-5. Smooth-current precise_QA at h = a/10 (119,678 inside points, 62,985 targets): hockney equals the existing path to 2.6e-14 and differs from vico_greengard by 4.1e-2 of max |B_plasma|, the second-order error of the discrete sum at that h.

## 2026-10-09: Plot scripts
- `scripts/plot_J_vs_desc.py`: makes `figures/J_vs_desc.png` (DESC J vs the BoundaryPotential fit).
- `scripts/plot_edge_layer.py`: makes `figures/edge_layer_J.png`; needs the removed edge taper (run at commit a5cbd41).

## 2026-10-09: Job scripts, step 2 driver and figures
- `run.sbatch`: sets NEST_DEVICE=gpu and stops if JAX does not get the GPU; `run_cpu.sbatch` (CPU jobs on Stellar); `run_della.sbatch` (one A100 on Della, env `nest-gpu` in /scratch/gpfs/EKOLEMEN/lk1158).
- `nest/run_step2.py`: driver for J on a Cartesian grid (h = a / frac) with checks against DESC (uses `CurrentFZ`).
- `figures/J_vs_desc.png` (DESC J vs the BoundaryPotential fit) and `figures/edge_layer_J.png` (the same with the edge taper, since removed).

## 2026-10-09: Force residual at iteration 0, spectral interpolation of B_plasma, reconstruction tests
- `nest/force.py`: `ForceResidual`, F = J x (B_ext + B_plasma) - grad p on a flux-coordinate collocation grid (default DESC's QuadratureGrid(2 L, 2 M, 2 N), volume weights sqrt(g) w), measured as <|F|>_vol / <|grad |B|^2 / 2 mu0|>_vol like DESC; `decompose` splits F exactly into DESC's own imbalance, the current fit, the field mismatch and the pressure fit. Smooth-current precise_QA at iteration 0: 4.6e-4 (DESC 4.24e-5), dominated by DESC's free-boundary field mismatch (3.9e-4), the same at h = a/10 and a/20.
- `nest/magnetic_field.py`: `interpolate`, spectral interpolation of a box field at arbitrary points with a type-2 non-uniform FFT (`nufftax`, new dependency), one component and chunk of points at a time, single precision for float32 fields; the Biot-Savart kernel keeps the requested precision.
- `nest/desc_input.py`: `iota(rho)` and `compute_points`, DESC quantities at scattered flux coordinates without surface integrals.
- Tests: `test_force.py`; `test_reconstruction.py` (manufactured field: FFT + interpolation second order; J of the box vs the fitted series; B_plasma convergence a/10, a/20, a/30, order ~2.2). Large boxes need XLA_PYTHON_CLIENT_ALLOCATOR=platform.

## 2026-10-09: Remove the edge taper
- `nest/boundary_potential.py`, `nest/current_potential.py`: the `edge_width` option (DESC's current tapered to zero over the outer layer before the initial fit, and the boundary-compatibility step of `target_potential`) and `edge_taper` are removed; the initial fits use DESC's current as it is. The structural edge conditions are unchanged.
- Tests updated (`test_boundary_potential.py`, `test_smooth_edge.py`); both pass on the smooth-current precise_QA and W7-X.

## 2026-10-09: Boundary-adapted current potential (J_potential_notes.pdf)
- `nest/boundary_potential.py`: `BoundaryPotential`, J = curl T with T_theta = I/2pi h_0 + d_theta chi + rho^2 f a, T_zeta = d_zeta chi + f b, f = (1 - rho^2)^s. For every coefficient vector: div J = 0, J = 0 on the LCFS, J regular at the axis (exact rho = 0 limit), net toroidal current I through every cross-section. Free coefficients u = (chi, a, b), J linear in u, differentiable with JAX.
- Changes from the notes: rho^2 factor on a (axis regularity), chi includes the sin(|m| theta), n = 0 modes, alpha(zeta) kept out as a pure gauge but fitted jointly when matching the target T.
- Initial fit of T itself (not J), mode by mode; DESC's current tapered to zero over the outer 0.1 in rho, with the boundary values made compatible with the representation, so the initial J goes to zero smoothly across that layer. Smooth-current precise_QA: J matches DESC to 4.9e-4 for rho <= 0.9.
- `nest/separable.py`: coefficients can be passed as JAX arrays (`scatter`, `evaluate(..., C=)`).
- `tests/test_boundary_potential.py`: divergence, LCFS, axis, net current, DESC reconstruction, derivatives, save/load.

## 2026-10-08: Grid classes and a common DESC input class
- `nest/desc_input.py`: `DESCInput`, the starting point for any DESC input.
  - Loads an equilibrium file (an Equilibrium, or the last member of an EquilibriaFamily), or wraps an Equilibrium, with an optional external field (a coil or field file, or a field object).
  - Gives NFP, resolution, prescribed profile, a, R0, volume, <beta>, the boundary extent, the R, Z geometry series, and `B_ext(x)` at Cartesian points.
  - Checks phi = zeta and the sign of sqrt(g).
- `nest/grid.py`:
  - `FluxGrid` (G1): one-period DESC grid with its torus copies, positions and weights.
  - `CartesianGrid` (G3): `periodic_box`, spacing, cell volume and wavevectors; `locate_plasma` (mask, inside points, flux coordinates); `scatter` / `gather` between the box and the inside points.
  - The inverse map `map_to_flux` and `plasma_extent` moved here from `cartesian_grid.py`, which re-exports them.
- `cartesian_grid.py`, `desc_grid.py`: `cartesian_grid`, `current_on_cartesian`, `flux_grid`, `evaluate_on_flux_grid` keep their signatures and use the classes. They take a `DESCInput` or an Equilibrium.
- `tests/test_grid.py`: DESCInput against DESC (B_ext vs the coils to 5e-16), FluxGrid positions, the box, plasma volume on the box vs DESC (4e-4 to 5e-4), and the inverse map. All existing tests pass with the change.

## 2026-10-08: Smooth edge for J and p, J.n = 0 removed, tests on the smooth-current precise_QA
- `nest/current_potential.py`: smooth-edge option (`edge_order` k, e.g. 2): d_rho^j T(1) = 0 for j = 1..k, so sqrt(g) J^theta and sqrt(g) J^zeta vanish like (1 - rho)^k at the LCFS; fit of J under these constraints, with DESC's J tapered to zero over the outer 0.1 in rho (net currents kept); `impose_smooth_edge` for any coefficients. The J.n = 0 projection (`impose_no_normal_current`) is removed for now.
- `nest/pressure.py`: smooth-edge option p = (1 - rho^2)^(k+1) Q.
- Tests use `precise_QA_free_boundary_smooth_J_edgeJ_r0.3.h5` instead of the iota-prescribed precise_QA: the Zernike fit of J is exact to 2e-8 and B_plasma converges like h^2. New `test_smooth_edge.py`; the current through phi = 0 is now compared with DESC's J through the same cross-section; check 3 of `test_magnetic_field.py` handles current-prescribed equilibria.

## 2026-10-07: precise_QA free boundary with the current density going to zero at the LCFS
- `desc_input/precise_QA_free_boundary_smooth_J/`: the current is prescribed with dI/ds = 0 and p ~ (1-rho^2)^3 at the edge, and the free boundary is redone with LCFS modes <= 5. The current is rescaled so iota = 1/2 stays at rho = 0.55, and the edge J_parallel enters the free-boundary objective.
- Recommended initial configuration: `precise_QA_free_boundary_smooth_J_edgeJ_r0.3.h5`. B.n/|B| is 2.45e-3; the net edge current is ~0, with a residual rms of 6 % of the peak.
- Diagnostics and tests in the same folder: edge-current tables, J_parallel maps on the LCFS, the (m, n) spectrum, and the edge force-balance and resolution tests.
- `desc_input/README.md`: overview of the free-boundary inputs and how they were made.

## 2026-10-07: B_plasma from J (step 5), pressure as a Fourier-Zernike field
- `nest/magnetic_field.py`: B_plasma on the box from the FFT of J, periodic (i mu0 k x J_hat / |k|^2) and free space (Hockney-Eastwood convolution with the Biot-Savart kernel, no periodic images).
- `nest/pressure.py`: p(rho, theta, zeta) as a Fourier-Zernike series, fitted from DESC's p(rho), with p = 0 on the boundary and grad p from the fixed geometry.
- `nest/cartesian_grid.py`: shared chunked evaluator; p and grad p on the box (`pressure_on_cartesian`).
- Tests: `test_magnetic_field.py` (free space = direct grid sum to 1e-7; 0.4 % vs exact B at a/20; periodic images 3-10 %; B_coil + B_plasma vs DESC B), `test_pressure.py` (fit, p = 0 on the edge, box).

## 2026-10-07: Current potential J = curl T, FFT of J
- `nest/current_potential.py`: J = curl T with T a Fourier-Zernike potential (gauge T_rho = 0), fitted from DESC; J.n = 0 imposed through the boundary values of T, keeping the net currents.
- `nest/separable.py`: shared separable (l, m) x n evaluation of Fourier-Zernike series.
- `nest/fourier.py`: rfft of J on the periodic box, wavevectors, inverse.
- `nest/current_fz.py`, `nest/cartesian_grid.py`: point evaluators, so the Cartesian mapping works with either J representation; evaluation chunk 50000.
- Tests: `test_current_potential.py` (curl T vs DESC J, J.n = 0), `test_fourier.py` (wavevectors, inverse, Parseval, k = 0 mode), `test_cartesian_grid.py` now uses the potential.

## 2026-10-06: Scripts that built precise_QA_free_boundary_L16N14.h5
- `desc_input/precise_QA_free_boundary/`: free-boundary run and refinement scripts.

## 2026-10-06: Periodic box, faster inverse map, finite-beta test equilibria
- Periodic box G3 (factor 2, FFT-friendly sizes), J in float32.
- Inverse map by Newton in (rho cos theta, rho sin theta), separable series evaluation.
- Tests on the free-boundary precise_QA and W7-X; cheb1 grids; higher fit resolution.

## 2026-10-06: Cartesian grid mapping of J
- `nest/cartesian_grid.py`: uniform grid around the plasma, inverse map to flux coordinates, J on the grid (zero outside).
- `tests/test_cartesian_grid.py`: current through phi = 0 and int |J| dV vs DESC.

## 2026-10-05: Fourier-Zernike current fit
- `nest/current_fz.py`, `nest/desc_grid.py`: J fitted as a Fourier-Zernike series, evaluated on a ConcentricGrid; test vs DESC J.
