# Finite-β free-boundary stellarators with net current, from published VMEC runs

Benchmark inputs for NEST beyond precise_QA. Each case is a converged **VMEC
free-boundary** run (`LFREEB = T`) at finite β with a net toroidal current. The
coils that produced it are public, and the case is re-solved here in DESC.

| case | source | NFP | R0, a [m] | ⟨β⟩ | I [A] | coils |
|---|---|---|---|---|---|---|
| `NCSX/` | NCSX c09r00, [landreman/vmec_equilibria](https://github.com/landreman/vmec_equilibria/tree/master/NCSX/free_boundary_from_vmec_wiki) (VMEC 8.52, ns = 99, fsqr 1e-12) | 3 | 1.44, 0.32 | 4.09 % | −1.786e5 | 10 groups: modular A–C, PF1–6, TF (`coils.c09r00`) |
| `QH_5beta/` | Landreman–Buller–Drevlak QH with bootstrap current, coils of Wiedman, Buller & Landreman, JPP 2024, [Zenodo 10211349](https://doi.org/10.5281/zenodo.10211349) (CC-BY 4.0), `LBDQH5_coils/` | 4 | 11.1, 1.70 | 5.08 % | −2.308e6 | 40 modular coils (5 per half period) |
| `CTH_like/` | CTH-like test of [VMEC++](https://github.com/proximafusion/vmecpp/tree/main/src/vmecpp/cpp/vmecpp/test_data) (MIT), low resolution (ns = 15, mpol = 5) | 5 | 0.77, 0.14 | 0.19 % | +4.32e4 | helical coil (96 turns × 4700 A) + 4 circular vertical-field coils |

The QH coils reproduce the 5 % target only approximately: the free-boundary surface
is the one the coils give, not the target. The paper reports a QS error of 0.009
against 0.0005. `input.LBDQH5_target` is the fixed-boundary target.

## Files (per case)

| file | what |
|---|---|
| `coils.*`, `extcur.*`, `input.*`, `wout_*.nc`, `makegrid*` | the original VMEC run |
| `<case>_coils.h5` | the coils as a DESC `MixedCoilSet`, currents in A |
| `<case>_vmec.h5` | the wout fitted by DESC (`VMECIO.load`); **not for NEST**, see below |
| `<case>_L{L}M{M}N{N}.h5` | **the NEST input**: DESC equilibrium on the VMEC LCFS |
| `*_diagnostics.json`, `*_check.json` | the numbers below, plus ι(ρ) and rational surfaces |

## Checks in the coils

The force error is ⟨|F|⟩/⟨|∇B²/2μ0|⟩. B·n/|B| and the pressure jump
[[B² + 2μ0 p]]/B² are measured on the LCFS with DESC's `BoundaryError`: the total
field is the coils plus the plasma field from virtual casing. These are the same
metrics as for precise_QA, where B·n/|B| rms = 2.45e-3.

| case | DESC L M N | force error | B·n/\|B\| rms (max) | pressure jump rms (max) | ι axis → edge | islands (n/m at ρ) |
|---|---|---|---|---|---|---|
| NCSX | 12 12 8 | 3.1e-3 | 4.4e-3 (1.7e-2) | 8.3e-3 (5.4e-2) | −0.370 → −0.667 | 3/6 at 0.59, 3/5 at 0.80, 3/7 at 0.35, 6/11 at 0.70 |
| **NCSX** | **16 16 12** | **5.8e-4** | 3.5e-3 (2.1e-2) | 4.9e-3 (2.0e-2) | −0.352 → −0.654 | 3/6 at 0.59, 3/5 at 0.80, 3/7 at 0.39, 6/11 at 0.69 |
| QH_5beta | 12 12 10 | 2.0e-3 | 2.1e-3 (6.1e-3) | 3.2e-3 (1.9e-2) | 1.282 → 1.048 | 8/7 at 0.56, 12/11 at 0.71 |
| CTH_like | 8 8 6 | 7.0e-3 | 3.3e-3 (8.8e-3) | 1.7e-3 (1.1e-2) | −1.252 → −0.863 | 5/4 at 0.71, 5/5 at 0.91 |

The wout fitted directly by DESC (`<case>_vmec.h5`) gives force errors of 0.4–0.9
and pressure jumps of 6–12 %. The fit puts spikes in |B| on the LCFS: 3.7 T against
VMEC's 1.9 T for NCSX. The re-solve fixes this.

## How the NEST inputs were made

1. `check_vmec_free_boundary.py`:
   - reads the MAKEGRID coils into DESC;
   - fits the wout (`VMECIO.load`, current profile);
   - computes the checks above for that fit.

   NCSX and QH coil files hold the EXTCUR currents (MAKEGRID scaled mode). CTH-like
   holds turns, which are multiplied by EXTCUR = [4700, 1000] (raw mode). A
   one-line filament there is a circle about the z axis (VMEC++ convention).
2. `solve_desc.py CASE --LMN L M N`:
   - keeps the VMEC LCFS;
   - replaces the profiles with those of `vmec_profiles.py`;
   - solves force balance (fixed boundary) at higher resolution.
3. `vmec_profiles.py` rebuilds p and I from the VMEC input definitions (AM, AC,
   AC_AUX) as even power series in ρ. The spline that `VMECIO.load` fits to the
   wout current is not smooth at the axis: I/ρ² changes sign and ι spikes for
   ρ < 0.01. The NCSX and CTH profiles are exact. The QH bootstrap current
   (`cubic_spline_ip`) is fitted with ρ²…ρ²⁴, to a residual of 1.7e-4 of I(1).

## What differs from precise_QA

- **The current density does not vanish at the edge.** NCSX has a steep pressure
  gradient near ρ ≈ 0.85, and the Pfirsch–Schlüter current gives |J| ≈ 0.67 MA/m²
  on the LCFS, about half the peak. NEST's fits force J and p to zero smoothly
  there (`edge_power`, `edge_order`), so the starting fit error is larger at the
  edge.
- **DESC's force error is larger:** 2–7e-3 at 12/12/8, against 4.2e-5 for
  precise_QA. In NCSX it grows toward the edge, from 1.6e-3 in the core to 2e-2 at
  ρ = 1.
  - At 16/16/12 it is 5.8e-4, solved on an A100 with
    `solve_desc.py NCSX --LMN 16 16 12 --device gpu --jac-chunk-size 200`.
  - Without the chunk size, the dense Jacobian needs 27 GB.
  - **Use `NCSX_L16M16N12.h5`.**

## NEST on NCSX

First run on `NCSX_L12M12N8.h5`, with the defaults of `scripts/optimize_qa.py` (free
pressure, stored energy held, h = a/10, 100 iterations, 11 min on an A100):

| ρ | start | end, checked on h = a/20 |
|---|---|---|
| 0–0.5 | 2.5e-3 | 1.3e-4 |
| 0.5–0.8 | 3.3e-3 | 2.6e-4 |
| 0.8–0.9 | 1.1e-2 | 5.0e-4 |
| 0.9–0.95 | 1.7e-2 | 4.2e-4 |
| 0.95–1 | 8.7e-2 | 3.3e-4 |
| **all** | 1.28e-2 | **2.82e-4**, against DESC's 3.11e-3 |

The start is worse than DESC near the edge because the smooth-edge fits of J and p
miss NCSX's edge current; the optimizer removes this. It was still decreasing at
iteration 100.

On `NCSX_L16M16N12.h5` with `--refine 20` (100 iterations at h = a/10, then 100 at
h = a/20; 62 min on an A100):

| ρ | start | end, checked on h = a/20 with the finer collocation grid (97, 385, 127) |
|---|---|---|
| 0–0.5 | 5.5e-4 | 1.6e-5 |
| 0.5–0.8 | 1.2e-3 | 3.5e-5 |
| 0.8–0.9 | 2.9e-3 | 9.2e-5 |
| 0.9–0.95 | 5.3e-3 | 9.8e-5 |
| 0.95–1 | 1.7e-2 | 1.1e-4 |
| **all** | 3.16e-3 | **5.23e-5**, against DESC's 5.76e-4 |

The check uses the same box as the second stage, because a/30 does not fit on one
A100. The residual is largest at ρ > 0.8, where NCSX's edge current sits.

### Holding the current profile (`--w-J`)

Without a constraint on the current profile the optimizer moves toroidal current to
the axis: the iota of the traced field lines rises from DESC's 0.35 to 0.56 there
(`figures/ncsx_opt_L16M16N12_poincare_flux.png`). Total current and stored energy
are held, but force balance does not fix I(rho). CurrentError (J - J_DESC, pointwise)
fixes this:

| | <\|F\|>/norm (check) | iota at rho = 0.05 / 0.3 / 0.6 / 0.9 | islands seen |
|---|---|---|---|
| DESC | 5.76e-4 | 0.354 / 0.404 / 0.504 / 0.655 | |
| no penalty | 5.23e-5 | 0.560 / 0.457 / 0.520 / 0.668 | 3/5; 3/6 and 3/7 moved or gone; 2/3 crossed |
| `--w-J 1e-3` | **6.17e-5** | 0.372 / 0.426 / 0.510 / 0.658 | 3/5 (rho 0.80), 3/6 (0.57), 6/11, 3/7 (0.31), 9-island 2/3 chain at rho 0.94 |
| `--w-J 1e-2` | 2.34e-4 | 0.340 / 0.396 / 0.502 / 0.655 | 3/5 and 3/6 wider, 3/7 (0.40), 3/8 (0.23), 9/14 edge chain |

**Use `--w-J 1e-3`.** At 1e-2 the pointwise penalty also damps the resonant
currents and pins the edge, where NEST's fit cannot match DESC's J.

Poincare sections, with only `nest/` needed:
- `poincare.py` traces the field lines of the start and of the optimized x and saves
  the crossings of phi = 0 (GPU, a few minutes; `--x start` traces the start only).
- `poincare_flux.py` maps them to DESC (rho, theta), measures iota per line and
  draws the figure (CPU, two minutes).

```bash
sbatch --mem=128G run.sbatch desc_input/vmec_free_boundary/poincare.py $E --x results/ncsx/wJ.npz --frac 20
python desc_input/vmec_free_boundary/poincare_flux.py desc_input/vmec_free_boundary/NCSX/NCSX_L16M16N12.h5 figures/ncsx_wJ_poincare.npz
```

Figures:
- `figures/ncsx_opt_L16M16N12_poincare_flux.png`: no penalty.
- `figures/ncsx_opt_L16M16N12_wJ1e-3_poincare_flux.png`
- `figures/ncsx_opt_L16M16N12_wJ1e-2_poincare_flux.png`

The NEST runs above used `optimize_qa.py` options that are not committed yet:
`--eq`, `--coils`, `--refine`, and the free-pressure defaults (`--w-W 10`,
`--w-pos 10`).

Run on Della, in jobs of 59 min or less for the gpu-test QOS:

```bash
D=desc_input/vmec_free_boundary/NCSX; E="--eq $D/NCSX_L16M16N12.h5 --coils $D/NCSX_coils.h5"
a=$(sbatch --parsable --constraint=gpu80 --time=00:59:00 run_della.sbatch scripts/optimize_qa.py $E --frac 10 --w-J 1e-3 --out results/ncsx/wJ_a10)
sbatch --dependency=afterok:$a --constraint=gpu80 --time=00:59:00 run_della.sbatch scripts/optimize_qa.py $E --frac 20 --start results/ncsx/wJ_a10.npz --time-limit 2400 --w-J 1e-3 --out results/ncsx/wJ
```

Run on Stellar:

```bash
sbatch --qos=gpu-stellar --time=03:30:00 run.sbatch scripts/optimize_qa.py --eq desc_input/vmec_free_boundary/NCSX/NCSX_L16M16N12.h5 --coils desc_input/vmec_free_boundary/NCSX/NCSX_coils.h5 --refine 20 --out results/ncsx/opt_L16M16N12
```
