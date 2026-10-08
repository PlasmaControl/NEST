# Updates

One entry per commit, newest first.

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
