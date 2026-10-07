# `precise_QA_free_boundary_smooth_J.h5`: current density smoothly to zero at the LCFS

An initial configuration for DESC, built from `../precise_QA_free_boundary_L16N14.h5`
(see `../precise_QA_free_boundary/`). It has the same free-boundary LCFS in the field
of the 16 helical coils, the same ⟨β⟩ and the same ι = 1/2 surface, but the current
density goes to zero smoothly at the boundary.

## What changed

The source equilibrium prescribes ι with p ∝ (1−ρ²)². There the pressure-driven
current (∝ p′) vanishes at ρ = 1, but the parallel current does not: the toroidal
current density dI/dρ needed for that ι is finite at the edge.

Here, with s = ρ²:

- **Current prescribed, not ι:** I(s) is a fit of the source's current with
  dI/ds = (1 − s)² Q(s), Q a degree-4 polynomial. The toroidal current density and
  its radial slope vanish at the edge.
  - The fit is tight for ρ < 0.77 (deviation 0.25 % of max I). This keeps ι near the
    resonant surface.
  - Outside that, the current flattens instead of turning down, so I_edge goes from
    4.9 kA to 5.3 kA.
- **Pressure:** p = p₀ (1 − ρ²)³, p₀ = 15.6 kPa, so p′ and p″ vanish at the edge too.
  ⟨β⟩ is unchanged.
- **Boundary and resolution:** same LCFS and resolution (L = M = 16, N = 14).

The coefficients are in `diagnostics.json["params"]`:

```python
eq.current.params[j]  # coefficient of rho^(2j)
```

## Result

| | source (ι prescribed) | **this equilibrium** |
|---|---|---|
| force error ⟨\|F\|⟩/⟨\|∇B²/2μ₀\|⟩ | 6.1e-5 | 8.3e-5 |
| ⟨β⟩ | 1.04 % | 1.04 % |
| ι on axis / edge | 0.520 / 0.454 | 0.518 / 0.457 |
| ι = 1/2 (m/n = 4/2) at ρ, shear | 0.550, 0.080 | **0.550, 0.079** |
| free boundary in the coils: B·n/\|B\| rms | 2.2e-3 | 2.3e-3 |
| net parallel current at ρ = 1, \|⟨J·B⟩\|/√⟨B²⟩ | 0.21 J_ref | **0.0014 J_ref** |
| ⟨\|J\|⟩ at ρ = 1 | 0.25 J_ref | 0.13 J_ref |

J_ref = 3.9e5 A/m², the peak ⟨|J|⟩ of this equilibrium.

### Remaining edge current

The net current at the edge is gone, reduced 150×, and the pressure-driven current
vanishes with p′ and p″. What remains at ρ = 1 (≈ 13 % of the peak) is a parallel
current that **averages to zero on each flux surface**.

With p′ = 0 it should vanish exactly, since J_∥/B is then constant on the surface. It
is a numerical residual. It decreases neither with radial resolution (at L = 20 the
force error is 5.4e-5 but the edge ⟨|J|⟩ is 0.15 J_ref) nor with M, N (see below).
It is tied to the short-wavelength shape of this free-boundary LCFS.

Check it with:

```bash
python check_edge_current.py ../precise_QA_free_boundary_L16N14.h5 precise_QA_free_boundary_smooth_J.h5
```

### Two attempts at removing it (2026-10-07)

`Jpar_boundary.png` shows where the residual sits. Most of it is one patch on the
inboard side at mid-period, where the free-boundary LCFS has short-wavelength
wiggles: the helical coils come closest to the plasma there, and the free-boundary
fit followed their field. Two fixes were tested.

| variant | ⟨\|J\|⟩ at ρ = 1 | at ρ = 0.98 | net \|⟨J·B⟩\|/√⟨B²⟩ at ρ = 1 | force error | B·n/\|B\| rms (max) | ι = 1/2 at ρ |
|---|---|---|---|---|---|---|
| this equilibrium (LCFS modes ≤ 8, L/M/N = 16/16/14) | 0.132 | 0.097 | 0.0014 | 8.3e-5 | 2.3e-3 (9.3e-3) | 0.550 |
| **LCFS trimmed to \|m\|,\|n\| ≤ 5** (`*_bdry_k5.h5`) | **0.070** | **0.034** | 0.0007 | **3.7e-5** | 3.3e-3 (1.5e-2) | 0.545 |
| LCFS trimmed to \|m\|,\|n\| ≤ 6 | 0.169 | 0.135 | 0.0023 | 9.0e-5 | 3.7e-3 (2.0e-2) | 0.561 |
| more resolution, L/M/N = 18/18/16 | 0.146 | 0.111 | 0.0037 | 4.8e-5 | 2.3e-3 (9.5e-3) | 0.550 |
| more resolution, L/M/N = 20/20/18 | out of memory on a 40 GB A100 (all Stellar GPUs are 40 GB) | | | | | |

Current densities are in units of J_ref. B·n is in the coil field, with the plasma
field from virtual casing.

- **More Fourier resolution does not help.** The interior force error improves but the
  edge current does not, so the residual is not M/N truncation.
- **Trimming the LCFS to |m|, |n| ≤ 5 removes the inboard patch.**
  `Jpar_boundary_trim_k5.png` compares it with the untrimmed case. The edge current
  halves, the force error is the lowest of all variants, and ι = 1/2 moves only to
  ρ = 0.545.
- **The cost:** the LCFS is no longer exactly the free-boundary surface. B·n/|B| in
  the coils goes from 2.3e-3 to 3.3e-3 rms.
- **Trimming is not monotonic:** cutting at 6 instead of 5 was worse than not
  trimming.
- **Across all variants the edge current follows the local edge force error.** It is
  the part of the DESC solution that does not converge at this boundary.

### What the residual is (2026-10-07)

- **Not under-weighted edge force balance.** `edge_force_test.py` adds a second
  `ForceBalance` term on ρ = 0.95–1.0. At edge weight 1 and 10, the pointwise
  max |J_∥| at ρ = 1 stays at 0.70 and 0.64 J_ref (it was 0.71). The edge force
  error stays near 8e-4–1e-3, and the volume force error rises to 2.7e-4. DESC
  cannot satisfy force balance at the edge for this LCFS. Results are in
  `edge_force_test.json`.
- **It is the near-resonant (4, 2) harmonic.** `jpar_spectrum.py` (straight-field-line
  angles) shows the edge J_∥ is dominated by (m, n) = (4, −2), which grows toward the
  boundary and does not peak at the ι = 1/2 surface:

  | ρ | 0.55 | 0.7 | 0.9 | 0.96 | 1.0 |
  |---|---|---|---|---|---|
  | smooth J, LCFS modes ≤ 8 | 15 | 19 | 17 | 42 | 62 |
  | smooth J, LCFS trimmed to ≤ 5 | 14 | 10 | 19 | 16 | 11 |

  (4, −2) amplitude of J_∥ in kA/m².
- **Likely mechanism** (consistent with the data, not proven):
  - At the edge ι ≈ 0.457 is close to 1/2, so (4, 2) is nearly field-aligned there
    (4ι − 2 ≈ −0.17).
  - J_∥ in a nearly field-aligned harmonic is the residual of ∇·J divided by that
    small k_∥, so small edge errors from the short-wavelength LCFS shaping are
    amplified there.
  - Trimming the LCFS removes the source. Pointwise max |J_∥| at ρ = 1 / 0.98 goes
    from 0.71 / 0.48 J_ref to 0.20 / 0.08 J_ref.

### Step 1: free boundary again, with the smooth profiles and LCFS modes ≤ 5

`free_boundary_smooth_J.py` starts from the trimmed variant and minimizes
`BoundaryError` in the coil field again. The smooth current, the pressure and Ψ are
fixed, and only LCFS modes with max(|m|, |n|) ≤ 5 are released.

- **Resolution.** The proximal free-boundary solve runs at L = M = N = 12; at
  16/16/14 its Jacobians do not fit in a 40 GB A100. The equilibrium is then
  re-solved at 16/16/14 with that LCFS fixed. The result is
  `precise_QA_free_boundary_smooth_J_fb_k5.h5`.
- **Diagnostics** are in `fb_k5_diagnostics.json`; the J_∥ map is
  `Jpar_boundary_fb_k5.png`.

| | LCFS modes ≤ 8 (`*_smooth_J.h5`) | trimmed ≤ 5 (`*_bdry_k5.h5`) | **free boundary, modes ≤ 5 (`*_fb_k5.h5`)** |
|---|---|---|---|
| B·n/\|B\| rms (max) | 2.3e-3 (9.3e-3) | 3.3e-3 (1.5e-2) | **2.5e-3** (9.8e-3) |
| [[B²+2μ₀p]]/B² rms | 2.8e-3 | 3.0e-3 | 4.2e-3 |
| force error | 8.3e-5 | 3.7e-5 | 4.9e-5 |
| ⟨\|J\|⟩ at ρ = 1 | 0.132 | 0.070 | **0.052** |
| max \|J_∥\| at ρ = 1 / 0.98 | 0.71 / 0.48 | 0.20 / 0.084 | 0.22 / 0.098 |
| rms J_∥ on the LCFS [kA/m²] | 64 | 34 | **27** |
| ι = 1/2 at ρ (shear) | 0.550 (0.079) | 0.545 | 0.496 (0.080) |

Current densities are in units of J_ref.

- **The coil fit is recovered** (B·n/|B| 2.5e-3) with the edge current of the trimmed
  case or lower. The inboard patch does not come back.
- **What remains** on the LCFS is weak field-aligned stripes, with pointwise max
  |J_∥| ≈ 0.2 J_ref.
- **ι moved.** The current is prescribed, so the new LCFS shifts the ι = 1/2 surface
  inward, to ρ = 0.50. Scaling the current would move it back to 0.55.

### Steps 2 and 3: ι back to 0.55, and the edge current as an objective

`fb_iota_edgeJ.py` starts from `*_fb_k5.h5` and does the following twice at
L = M = N = 12:

1. Scale the prescribed current (same shape, so dI/ds still vanishes at the edge) by
   a secant search, at fixed boundary, until ι = 1/2 is at ρ = 0.55.
2. Free-boundary solve with LCFS modes ≤ 5, minimizing `BoundaryError` plus
   w · J_∥/J_ref on ρ = 0.98 and 1. The weight w gives the edge term RATIO times the
   boundary cost at the start.

It then re-solves at 16/16/14.

| | step 1 (`*_fb_k5.h5`) | **ratio 0.3 (`*_edgeJ_r0.3.h5`)** | ratio 3 |
|---|---|---|---|
| current scale (net) | 1 | 1.038 | 1.04 |
| ι = 1/2 at ρ (shear) | 0.496 (0.080) | **0.550** (0.094) | 0.565 (0.090) |
| B·n/\|B\| rms (max) | 2.5e-3 (9.8e-3) | **2.45e-3** (9.8e-3) | 2.8e-3 (1.3e-2) |
| [[B²+2μ₀p]]/B² rms | 4.2e-3 | 4.2e-3 | 4.4e-3 |
| force error | 4.9e-5 | 4.2e-5 | 3.9e-5 |
| ⟨\|J\|⟩ at ρ = 1 | 0.052 | **0.041** | 0.053 |
| max \|J_∥\| at ρ = 1 / 0.98 | 0.22 / 0.098 | 0.21 / 0.105 | 0.24 / 0.15 |
| rms J_∥ on the LCFS [kA/m²] | 27 | **23** | not computed |
| (4, 2) J_∥ at ρ = 1 [kA/m²] | 17 | **12** | 22 |

Current densities are in units of J_ref.

- **ι is back at ρ = 0.55** and the coil fit is the best of the smooth-boundary
  variants.
- **The explicit edge-current term helps only marginally.** Both free-boundary passes
  stop after 6–8 iterations ("bad approximation"): the linearized proximal model does
  not predict how J_∥ at the edge responds to the boundary shape.
- **A larger weight (ratio 3) is worse on everything.**
- **Pointwise, J_∥ on the LCFS is still ≈ 0.2 J_ref at most.** It is 0.1 J_ref at
  ρ = 0.98, with rms 23 kA/m², 6 % of the peak.

Diagnostics are in `edgeJ_r0.3_diagnostics.json` and `edgeJ_r3_diagnostics.json`;
the map is `Jpar_boundary_edgeJ.png`.

`trim_boundary.py` makes the trimmed variants. `trim_diagnostics.json` has their full
diagnostics.

## Files

| file | role |
|---|---|
| `precise_QA_free_boundary_smooth_J.h5` | the equilibrium (exact free-boundary LCFS) |
| `precise_QA_free_boundary_smooth_J_edgeJ_r0.3.h5` | **recommended starting point: ι = 1/2 at ρ = 0.55, free-boundary LCFS with modes ≤ 5, B·n/\|B\| 2.45e-3, smallest edge current** |
| `precise_QA_free_boundary_smooth_J_fb_k5.h5` | free-boundary LCFS limited to \|m\|,\|n\| ≤ 5, smooth profiles (ι = 1/2 at ρ = 0.50) |
| `precise_QA_free_boundary_smooth_J_bdry_k5.h5` | variant with the LCFS trimmed to \|m\|,\|n\| ≤ 5: half the edge current, slightly worse coil fit |
| `smooth_current.py` | builds it: stages A–D below |
| `check_edge_current.py` | edge current-density table |
| `diagnostics.json` | source, stage C and final diagnostics; the current and pressure coefficients |
| `profiles_before_after.png` | ⟨\|J\|⟩, ⟨\|J_∥\|⟩, I, ι, p versus ρ: source (blue) and this equilibrium |
| `J_section.png`, `force_error_section.png` | \|J\| and the normalized force error in cross-sections |
| `Jpar_boundary.png`, `plot_jpar_boundary.py` | J_∥ over (θ, ζ) on ρ = 1 (source and smooth J) and ρ = 0.9, and the LCFS mean curvature |
| `fb_iota_edgeJ.py`, `edgeJ_r*_diagnostics.json`, `Jpar_boundary_edgeJ.png` | steps 2-3: current rescaled for ι = 1/2 at 0.55, edge J_∥ in the free-boundary objective |
| `edge_tools.py` | edge current-density diagnostics shared by the scripts |
| `free_boundary_smooth_J.py`, `fb_k5_diagnostics.json`, `Jpar_boundary_fb_k5.png` | step 1: free boundary with the smooth profiles and LCFS modes ≤ 5 |
| `edge_force_test.py`, `edge_force_test.json` | test: extra ForceBalance on ρ = 0.95–1.0 (negative result) |
| `jpar_spectrum.py` | (m, n) spectrum of J_∥ on surfaces, in PEST angles, with the matching rational surfaces |
| `fb_tools.py`, `run_free_boundary.sbatch` | shared diagnostics, Stellar job script |

## Stages (`smooth_current.py`)

- **A.** Pressure (1−ρ²)² → (1−ρ²)³ with ι still prescribed. At finite β, ι depends on
  the pressure profile, so the current is fitted only after this.
- **B.** Constrained fit of the stage-A current.
- **C.** Current prescribed, ramped from the stage-A current to the fit. The result is
  saved as `*_L16.h5`; if that file exists, the script resumes from it.
- **D.** Optional radial refinement (`L_FINAL`). L = 16 means none, as here.

Run on one A100 (≈ 45 min for A–C):

```bash
SCRIPT=smooth_current.py sbatch --export=ALL --time=01:30:00 run_free_boundary.sbatch \
  ../precise_QA_free_boundary_L16N14.h5 ../precise_QA_free_boundary/precise_QA_helical_coils.h5 \
  OUTDIR 500 16
```

The arguments after the coils are the output directory, the Jacobian chunk size (for
GPU memory) and L_FINAL.
