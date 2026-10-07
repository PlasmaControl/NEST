# Updates

One entry per commit, newest first.

## 2026-10-07: Current potential J = curl T, FFT of J
- `nest/current_potential.py`: J = curl T with T a Fourier-Zernike potential (gauge T_rho = 0), fitted from DESC; J.n = 0 imposed through the boundary values of T, keeping the net currents.
- `nest/separable.py`: shared separable (l, m) x n evaluation of Fourier-Zernike series.
- `nest/fourier.py`: rfft of J on the periodic box, wavevectors, inverse.
- `nest/current_fz.py`, `nest/cartesian_grid.py`: point evaluators, so the Cartesian mapping works with either J representation; evaluation chunk 50000.
- Tests: `test_current_potential.py` (curl T vs DESC J, J.n = 0), `test_fourier.py` (wavevectors, inverse, Parseval, k = 0 mode), `test_cartesian_grid.py` now uses the potential.

## 2026-10-06: Scripts that built precise_QA_free_boundary_L16N14.h5
- `desc_input/precise_QA_free_boundary/`: free-boundary run and refinement scripts, helical coils, diagnostics, README.

## 2026-10-06: Remove the precise_QA vacuum example
- Deleted `desc_input/precise_QA_output.h5`.

## 2026-10-06: Periodic box, faster inverse map, finite-beta test equilibria
- Periodic box G3 (factor 2, FFT-friendly sizes), J in float32.
- Inverse map by Newton in (rho cos theta, rho sin theta), separable series evaluation.
- Tests on the free-boundary precise_QA and W7-X; cheb1 grids; higher fit resolution.

## 2026-10-06: Cartesian grid mapping of J
- `nest/cartesian_grid.py`: uniform grid around the plasma, inverse map to flux coordinates, J on the grid (zero outside).
- `tests/test_cartesian_grid.py`: current through phi = 0 and int |J| dV vs DESC; to-do list in the README.

## 2026-10-05: Fourier-Zernike current fit
- `nest/current_fz.py`, `nest/desc_grid.py`: J fitted as a Fourier-Zernike series, evaluated on a ConcentricGrid; test vs DESC J.
