# Updates

One entry per commit, newest first.

## 2026-10-10: Free pressure with the stored energy held: force balance below DESC's
- `nest/objective.py`: `PressureEnergy`, <p>_vol held at its start (one residual), so that the pressure can be freed without the plasma deflating; `SpectralInterpolator` one field component at a time (a third of the fine-grid memory; a/20 peak 12.9 -> 9.5 GB).
- `nest/precondition.py`: block Gram matrices accumulated over radial slabs (bounded memory), blocks padded to a few sizes; fast path also with pressure-only objectives.
- `scripts/optimize_qa.py`: truncated kernel by default; independent check on a finer box (2 x, at most a/20) and grid; `--w-W`, `--start`; device memory freed before the check. `scripts/decompose_opt.py`: F split along b (F_par = -b.grad p, which no current can change) and across, and the change of p.
- Result (smooth-current precise_QA, a/10, truncated kernel): with p fixed to DESC's profile the force error stalls at 1.09e-4, mostly F_par (DESC's surfaces are not exactly flux surfaces of the coil + plasma field). With p free and the stored energy held: <|F|>/norm 1.69e-5, and 2.75e-5 on the a/20 check, against DESC's 4.24e-5; p changes by 2.7 % rms. See `OPTIMIZATION_STATUS.md`.

## 2026-10-10: Objective, Jacobian, optimizer and preconditioner; speed and memory
- `nest/objective.py`: DESC-like `ObjectiveFunction` with ForceBalance, FieldError, CurrentError and PressureError (weights, targets, normalizations). `Fields` evaluates J, B, p and F at a tensor collocation grid: J and p by tensor evaluation, B_plasma from J on the box (cached flux-grid to box interpolation, as one gather), a Hockney-Eastwood or truncated kernel, and `SpectralInterpolator` (type-2 non-uniform FFT split so that the point-dependent part is cached; one component at a time). The derivative is explicit (dF = J_lin x B + J x B_lin - grad p): J v is one evaluation, J^T w its transpose; the linearization keeps only J and B at the points.
- `nest/flux_interp.py` (tensor grids, factors, geometry, current, pressure, Lagrange interpolation), `nest/batching.py`, `nest/jacobian.py`.
- Stellarator symmetry: half collocation grid, half flux grid, half box (`boundary_potential.py`, `pressure.py`, `force.py`, `grid.py`).
- `nest/optimize.py`: Levenberg-Marquardt, PCG on the normal equations, exact quartic line search, stagnation stop, time limit, callback; `nest/precondition.py`: block preconditioner over the unknowns sharing (|m|, |n|), built from the local columns with tables restricted to each block's modes, accumulated over radial slabs.
- `nest/fit_cache.py`: on-disk cache of the fits (keyed by the equilibrium file, the arguments and the fitting code). `force.py`: positions, sqrt(g) and DESC's targets on the tensor grid (checked against DESC's pointwise evaluation on a sample). `boundary_potential.py`: optimized einsum in the fit (140 s -> 0.02 s).
- Scripts: `optimize_qa.py` (optimization with an independent check on a finer grid and a finer box), `audit.py` (time and memory), `spectrum.py`, `precond_test.py`, `local_blocks.py`, `diagnose_opt.py`. `run.sbatch`: JAX may use 90% of the GPU. Tests: `test_objective.py`, `test_force.py` updated.
- Smooth-current precise_QA, a/10, 90k unknowns, truncated kernel: <|F|>/norm 4.68e-4 -> 1.09e-4 (DESC 4.24e-5) in about 16 min on one A100 (setup 1.5 min, 0.11 s per CG iteration, GPU peak 5.8 GB); before, 7.5 h and 1.8e-4. `OPTIMIZATION_STATUS.md` summarizes the methods, tests and results.

## 2026-10-09: Truncated Green's function for B_plasma, convergence study
- `nest/magnetic_field.py`: `truncated_green_hat` and `b_plasma_truncated`, free-space B = mu0 curl (G_L * J) with the Vico-Greengard-Ferrando truncated kernel G_L_hat = (1 - cos(L k)) / k^2, L = half the box diagonal (same no-image region as Hockney-Eastwood). The kernel is built once from s_x s_y s_z residue-class FFTs of the box size (no oversampled grid): under 2 s up to 144M points on one A100. `b_plasma_periodic` and the new method share `_curl_solve`.
- `tests/test_magnetic_field.py`: Gaussian-blob check of the kernel against the exact potential and B (3.9e-10 and 9.1e-8 in float64; Hockney-Eastwood 2e-2 on the same grid); the truncated method printed in checks 1-3; `continuous_biot_savart` takes a quadrature resolution and works in blocks.
- `tests/convergence_b_plasma.py`: B_plasma vs h for the three methods at fixed points (nested boxes): exterior vs continuous Biot-Savart, interior self-convergence, rho bands. Smooth-current precise_QA (BoundaryPotential L, M, N = 48, 48, 35, s = 2), interior rms vs the truncated B at a/30: truncated 1.4e-3, 5.2e-4, 3.0e-4 at a/10, a/15, a/20, against Hockney-Eastwood 1.2e-2, 5.2e-3, 2.9e-3 (exactly second order); exterior, both about 1e-4 rms from a/15 on. W7-X, BoundaryPotential at the equilibrium resolution: inside the plasma the truncated kernel is about 2 to 2.5 times more accurate than Hockney-Eastwood at the same h (rms 1.5e-3 vs 3.7e-3 at a/15 for s = 4); outside, Hockney-Eastwood is better (a trapezoidal rule of a smooth integrand, rms 6.7e-5 at a/20 for s = 4, against 8.1e-4). With the fit at 3 times the equilibrium resolution (L, M, N = 36, 36, 30), J is not resolved on the box up to a/20 and no method converges.

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
