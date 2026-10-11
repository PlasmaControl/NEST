# Free-boundary DESC equilibria for NEST

NEST keeps the LCFS of a DESC **free-boundary** equilibrium fixed and starts from its
current density. This folder holds those equilibria and the scripts that made them.
Each subfolder has its own README with the details.

## Equilibria

All are the finite-β (⟨β⟩ ≈ 1 %) `precise_QA` stellarator (NFP = 2) in the field of
the 16 helical coils `precise_QA_free_boundary/precise_QA_helical_coils.h5`, with
ι = 1/2 (m/n = 4/2), the surface where NEST should open islands, near ρ = 0.55.

| file | current density at the LCFS | B·n/\|B\| rms in the coils | force error | ι = 1/2 at ρ |
|---|---|---|---|---|
| `precise_QA_free_boundary_L16N14.h5` | finite: ι prescribed, so dI/dρ ≠ 0 at the edge | 2.2e-3 | 6.1e-5 | 0.550 |
| `precise_QA_free_boundary_smooth_J/precise_QA_free_boundary_smooth_J_edgeJ_r0.3.h5` | **≈ 0 net; residual rms 6 % of the peak** | 2.45e-3 | 4.2e-5 | 0.550 |
| `W7-X_output.h5` | DESC example (fixed boundary), used in tests | n/a | n/a | n/a |

The force error is ⟨|J×B − ∇p|⟩ / ⟨|∇B²/2μ₀|⟩. The resolution is L = M = 16, N = 14.

**Use `..._smooth_J_edgeJ_r0.3.h5` as the initial configuration** when J should go
to zero at the boundary.
- Its toroidal current density and the pressure (∝ (1−ρ²)³) vanish there with zero
  slope.
- Its LCFS is a free-boundary solution restricted to |m|, |n| ≤ 5.
- The current is rescaled so that ι = 1/2 stays at ρ = 0.55.
- What is left on the LCFS is a parallel current that averages to zero on the
  surface. Its rms is 23 kA/m² against a peak of 3.9e5 A/m²; locally it reaches
  ≈ 20 % of the peak at ρ = 1 and 10 % at ρ = 0.98.

## Subfolders

| folder | what |
|---|---|
| `precise_QA_free_boundary/` | how `precise_QA_free_boundary_L16N14.h5` was built: finite β by continuation, free boundary at L = M = N = 12, re-solve at 16/16/14 |
| `precise_QA_free_boundary_smooth_J/` | the smooth-current versions and every test made on the way, with numbers and figures |
| `vmec_free_boundary/` | published VMEC free-boundary runs at finite β with net current and their coils (NCSX 4.1 %, QH 5 % with bootstrap current, CTH-like), re-solved in DESC on the VMEC LCFS |

## How the free-boundary solves work

1. **Fixed-boundary start.** Pressure and ι (or the current) are ramped by
   continuation.
2. **Free boundary.** `BoundaryError` is minimized: B·n = 0 and
   [[B²/2μ₀ + p]] = 0 on the LCFS, with the plasma field from virtual casing. The
   profiles and Ψ are held fixed, force balance is kept by `proximal-lsq-exact`, and
   LCFS modes are released in stages.
3. **Re-solve at higher resolution** with the LCFS fixed.

Lessons from these runs:
- **Never release LCFS modes up to the equilibrium resolution.** The force error rises
  10–100×.
- **Use a tight inner equilibrium solve** (ftol 1e-9).
- **Run the proximal free-boundary solve at L = M = N = 12.** At 16/16/14 it does not
  fit in a 40 GB A100, and every Stellar GPU is 40 GB.
- **With ι prescribed, the edge current density is finite.** For J → 0 at the LCFS,
  prescribe a current with dI/ds = 0 and a pressure with p′ = p″ = 0 at the edge.
  Fit that current only *after* changing the pressure, because ι at finite β depends
  on it.
- **The residual edge current comes from short-wavelength LCFS shaping.** The free
  boundary picks it up from the discrete-coil field. More L, M, N does not reduce it,
  and extra edge force balance does not either. Limiting the LCFS to |m|, |n| ≤ 5
  does.

## Running

The scripts run on Stellar (one A100) through the `run_free_boundary.sbatch` in each
subfolder:

```bash
SCRIPT=<script.py> sbatch --export=ALL run_free_boundary.sbatch ARGS
```

Each script's docstring gives its arguments.
