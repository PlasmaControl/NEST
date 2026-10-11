# `DSHAPE_free_boundary_smooth.h5`: an axisymmetric free-boundary equilibrium, J → 0 at the LCFS

This is a smooth-edge, current-prescribed version of DESC's `DSHAPE_CURRENT` example: a D-shaped tokamak
(NFP = 1, N = 0) at the same ⟨β⟩. It is solved at free boundary in a fitted set of coils. The current
density and the pressure gradient go to zero at the LCFS with zero slope. That makes it an axisymmetric
companion to `../precise_QA_free_boundary_smooth_J/`: it is cheap to run, it has an exact up-down symmetry,
and its edge is far better resolved.

It was built on 2026-10-07 with fftq's `tools/make_dshape_smooth.py` (DESC 0.16.0, about 11 min on a laptop
CPU). That script is not copied here; its steps are summarized below. `check_dshape_smooth.py` re-checks
both files with NEST's DESC and writes `diagnostics.json`.

## Files

| file | what |
|---|---|
| `DSHAPE_free_boundary_smooth.h5` | the free-boundary equilibrium, L, M, N = 26, 13, 0 (DSHAPE's resolution) |
| `DSHAPE_free_boundary_smooth_M20.h5` | the same LCFS re-solved at fixed boundary at L, M, N = 40, 20, 0: better-resolved edge |
| `DSHAPE_smooth_TF.h5`, `DSHAPE_smooth_PF_coils.h5` | the external field: a 1/R toroidal field and 24 circular PF rings (`CoilSet`) |
| `coil_parameters.json` | the fitted coil parameters (TF B0 R0, ring R, Z, currents) |
| `make_coils.py` | builds the two field files from `coil_parameters.json` |
| `check_dshape_smooth.py` | diagnostics → `diagnostics.json` (`--nest`: also NEST's checks and the iteration-0 force residual) |

**Loading:**

```python
inp = DESCInput.load("DSHAPE_free_boundary_smooth.h5",
                     field=["DSHAPE_smooth_TF.h5", "DSHAPE_smooth_PF_coils.h5"])   # summed
```

The field is in two files because DESC cannot save a `SumMagneticField`. In DESC 0.16 and 0.17 its I/O
attribute list is misnamed (`_io_attrs` instead of `_io_attrs_`), so the saved file holds no fields. The
file fftq wrote for this case is empty for that reason, and the coils here were rebuilt from the fitted
parameters. The rebuilt field reproduces the final B·n of the build exactly: 4.838e-6 (table below).

## Profiles (s = ρ², current prescribed)

- **Pressure:** p(s) = p₀ (1 − s)³ with p₀ = 2086.9 Pa, so dp/ds ∝ (1 − s)². p₀ was rescaled once so that
  ⟨β⟩ matches DSHAPE (2.91 %).
- **Current:** I(s) = c (s − s³ + s⁴/2), so dI/ds = c (1 − s)² (1 + 2s). The coefficient c = −438,473.78 A
  is `DSHAPE_CURRENT`'s ρ² coefficient, so the on-axis current density is the same as DSHAPE_CURRENT's.
  The total current is I(1) = c/2 = −219.2 kA.

## How it was built

1. **Fixed-boundary start:** a solve in the DSHAPE boundary, then again with p₀ rescaled to match β.
2. **Coil fit:**
   - Toroidal field B_φ = B₀R₀/R, with B₀ = 0.2175 T at R₀ = 3.5085 m and B₀R₀ = ⟨R B_φ⟩ on the LCFS.
   - 24 up-down-symmetric circular PF rings, placed on the LCFS offset outward by 1.0 m and equally
     spaced in arc length.
   - Ring currents by weighted least squares, so that B_coils·n = −B_plasma·n on the LCFS (virtual
     casing). Fit residual: B·n/|B| rms 5.1e-6.
3. **Free boundary:**
   - DESC's `BoundaryError` objective with `proximal-lsq-exact`; force balance, current, pressure and Ψ
     held fixed.
   - LCFS modes released in stages: |m| ≤ 2, then 4, then all.
   - The boundary moved at most 1.1 mm.
4. **Final:** a fixed-boundary re-solve at that LCFS (the M13 file). Re-solves at M = 10, 16 and 20 make a
   radial-resolution ladder; M = 20 is the M20 file.

## Result (`diagnostics.json`, NEST's DESC 0.17.3)

| | `DSHAPE_free_boundary_smooth.h5` | `..._M20.h5` |
|---|---|---|
| L, M, N | 26, 13, 0 | 40, 20, 0 |
| force error ⟨\|F\|⟩/⟨\|∇B²/2μ₀\|⟩ | 4.1e-7 | 9.2e-10 |
| ⟨β⟩_vol | 2.912 % | 2.912 % |
| a, R₀ | 1.198 m, 3.509 m | same LCFS |
| I(ρ = 1) | −219.2 kA | −219.2 kA |
| ι on axis / at ρ = 1 | −1.003 / −0.308 | −1.003 / −0.308 |
| rms \|J\| at ρ = 0.5 / 0.98 / 0.995 / 1 [A/m²] | 81,409 / 556 / 43.7 / **29.2** | 81,409 / 555 / 36.7 / **0.51** |
| B·n/\|B\| rms on the LCFS in the coils (coils + virtual casing) | 4.84e-6 | 5.03e-6 |
| NEST iteration-0 force residual ⟨\|F\|⟩/norm, h = a/4 (63 × 63 × 21 box) | 1.05e-3 | 1.05e-3 |

For comparison, DESC's `DSHAPE_CURRENT` has rms |J| = 1374 A/m² at ρ = 1, and `DSHAPE` (ι prescribed) has
3454 A/m².

The edge current density falls with radial resolution as J ∝ (1 − s)² requires: about 4× per halving of
1 − s, at a fixed interior. The M = 10 / 13 / 16 / 20 ladder gives an edge rms |J| of 933 / 29 / 17 /
0.5 A/m². The NEST number above is only a smoke test of the pipeline: the box at h = a/4 is very coarse,
with 3620 inside points.

The equilibrium is stellarator (here up-down) symmetric, so `BoundaryPotential.fit(..., sym=True)` and
`Pressure.fit(..., sym=True)` apply. The coils are up-down symmetric: in mirrored ring pairs the positions agree to 1e-14 m and the currents to 4e-12 (relative).
