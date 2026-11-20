# DESC shielding (sheet) current at q = 2 vs resolution

Does DESC resolve the Dirac sheet current that ideal MHD with nested surfaces puts on the q = 2 surface? Test case:
the ripple-landscape equilibrium (candidate C, DSHAPE-like tokamak, NFP = 1) with a resonant (2,1) boundary ripple,
xi/a = 1e-3. `../tokamak_island_C_L32M32.h5` is the axisymmetric base (from the `ripple-landscape` branch).

## Scripts
- `sheet_solve.py`: `--stage base` re-solves the axisymmetric base at L = M; `--stage ripple` puts the ripple on the
  boundary and solves force balance, starting from `--init` (a converged state at lower L or N; the direct and
  first-order-perturbation starts both blow up at the resonance). `--seg/--snap/--budget` checkpoint and snapshot
  long runs. `--jblock` with `XLA_FLAGS=--xla_gpu_autotune_level=0` keeps one Jacobian-sized array on the GPU
  (blocked projected Jacobian, jitted Gram, blocked f@J and column norms; the shared DESC is not modified). Use
  `--tr qr`: `cho` stalls on the sheet (it squares the condition number).
- `sheet_diag.py` / `sheet_diag2.py` (several snapshots of one run sharing the transforms): sigma_21(rho), the (2,1)
  harmonic of sigma = J.B/B^2 in PEST angles, on a radial grid with d rho = 2.5e-4 across q = 2; peak, FWHM,
  integrals across q = 2, ringing, and the surface-averaged force error. `--part/--nparts/--merge` split the
  surfaces over jobs.
- `sheet_plot.py`, `plot_ladder.py`, `plot_force.py`: figures. `run_sai.sbatch`, `status.sh`, `auto_diag.sh`:
  Stellar-AI job script and helpers.

## States in `states/`
- `base_L64.h5`: axisymmetric base at L = M = 64 (force 1.9e-11).
- `ripple_L64N2_it850.h5`: L = M = 64, N = 2, 850 iterations (sigma_21 converged in iterations).
- `ripple_L64N5_x.h5` (+ `.json`): L = M = 64, N = 5 from that state, 890 iterations (QR, force 1.47e-7).

## Results so far (N = 2, sigma_21 converged in iterations)

| L = M | peak \|sigma_21\| | FWHM | int over +-0.02 | volume force |
|---|---|---|---|---|
| 32 | 313 | 0.124 | 12.0 | 4.5e-7 |
| 48 | 736 | 0.085 | 28.1 | 2.2e-7 |
| 64 | 1213 | 0.070 | 45.2 | 1.6e-7 |

- sigma_21 sharpens with iterations at fixed L long after the force has flattened (L = 64: peak 601 at 150
  iterations, 1213 at 1450): states have to be converged in sigma_21, not in force.
- sigma_21 ~ L^2 f(L (rho - rho_s)): the peak grows like L^2, the width shrinks like 1/L, and the integral across
  q = 2 grows ~ L. Up to L = 64 DESC does not converge to a sheet of fixed strength.
- The force error stalls at about 6.2e-7 on q = 2 for every L and N (the bump narrows ~ 1/L); away from the
  rational surfaces it falls with L (3e-7 -> 1e-7 -> 7e-8). A second bump at q = 3 ((3,1) content of the ripple).
- N: N = 2 vs N = 4 at L = 32 within 4 %; L = 64, N = 5 from the converged N = 2 state (`diag/`): peak 1214 -> 1253
  and int(+-0.02) 45.2 -> 46.6 over its first 100 iterations (N = 2 at the same count: ~1200, 45.0), force unchanged.
  N = 1 is wrong; N = 6 at L = 64 does not fit on one B200.
- Memory limit on one B200 (180 GB) with QR: L = M = 104 at N = 2 (L = 112 runs out). L = 96 and 104 were running
  from the converged L = 64 state when this was written.
