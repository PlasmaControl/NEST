# How `../precise_QA_free_boundary_L16N14.h5` was built

A finite-beta free-boundary version of DESC's `precise_QA` (NFP = 2), in the field
of the 16 helical coils `precise_QA_helical_coils.h5`:

- p = p0 (1 - rho^2)^2, <beta> = 1.04 %
- prescribed iota = 0.52 - 0.066 rho^2, which puts iota = 1/2 (m/n = 4/2) at rho = 0.55
- L = M = 16, N = 14; force error 6.1e-5, B.n/|B| rms 2.2e-3 (see the diagnostics)

Copied from `NEMO/desc_input` on Stellar, where it was run on one A100.

## Files

| file | role |
|---|---|
| `free_boundary_qa.py` | steps 1-2: finite beta at fixed boundary, then free boundary |
| `refine_resolution.py` | step 3: re-solve at L = M = 16, N = 14 with the boundary fixed |
| `fb_tools.py` | shared diagnostics (force error, boundary error, rational surfaces) |
| `run_free_boundary.sbatch` | Stellar job script (`SCRIPT=... sbatch run_free_boundary.sbatch ARGS`) |
| `precise_QA_helical_coils.h5` | the coil set (also in DESC's `tests/inputs/`, which the script loads) |
| `free_boundary_diagnostics.json` | diagnostics of every stage of steps 1-2 |
| `refine_L16N14_diagnostics.json` | before / after step 3 |

## Steps

1. Finite beta, fixed boundary: from the vacuum `precise_QA`, ramp p and iota to
   the targets by continuation.
2. Free boundary at L = M = N = 12: minimize `BoundaryError` (B.n = 0 and
   [[B^2/2 mu0 + p]] = 0, plasma field by virtual casing) with p, iota, Psi fixed
   and force balance kept by `proximal-lsq-exact`, releasing boundary modes up to
   |m|, |n| <= 2, then 8.
3. Re-solve at L = M = 16, N = 14 with the free-boundary LCFS fixed: force error
   5.6e-4 -> 6.1e-5, boundary unchanged.

```bash
sbatch --time=03:00:00 run_free_boundary.sbatch --outdir output_L12_C --k-list 2
sbatch --time=03:00:00 run_free_boundary.sbatch --outdir output_L12_C --resume output_L12_C/free_boundary_k2.h5 --k-list 8 --inner-ftol 1e-9
SCRIPT=refine_resolution.py sbatch --export=ALL run_free_boundary.sbatch output_L12_C/precise_QA_free_boundary.h5 output_L12_C/precise_QA_helical_coils.h5 16 16 14 output_L12_C/precise_QA_free_boundary_L16N14.h5 500
```
