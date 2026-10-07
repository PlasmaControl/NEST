# Updates

One entry per commit, newest first.

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
