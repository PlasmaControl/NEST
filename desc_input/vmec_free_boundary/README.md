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
| **QH_5beta** | **16 16 12** | **6.5e-4** | 1.9e-3 (7.6e-3) | 1.9e-3 (7.6e-3) | 1.284 → 1.057 | 8/7 at 0.56, 12/10 at 0.33, 12/11 at 0.72 |
| CTH_like | 8 8 6 | 7.0e-3 | 3.3e-3 (8.8e-3) | 1.7e-3 (1.1e-2) | −1.252 → −0.863 | 5/4 at 0.71, 5/5 at 0.91 |
| **CTH_like** | **12 12 8** | **7.6e-4** | 2.5e-3 (8.3e-3) | 2.2e-3 (1.1e-2) | −1.255 → −0.875 | 5/4 at 0.72, 5/5 at 0.90 |

The bold rows are the NEST inputs. They were re-solved on an A100 with
`solve_desc.py CASE --LMN L M N --device gpu --jac-chunk-size 200`.

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

## Free boundary in the coils (`free_boundary_desc.py`)

The inputs above keep the VMEC LCFS fixed. `free_boundary_desc.py CASE` solves for
the LCFS in the fixed coils, as was done for precise_QA. It starts from the
fixed-boundary input (bold row) and runs three steps:

1. Drop to the free-boundary resolution (NCSX 12/12/8, QH 12/12/10, CTH 12/12/8)
   and re-solve at fixed boundary.
2. Minimize `BoundaryError` (B·n = 0 and [[B²/2μ0 + p]] = 0, virtual casing) with
   `proximal-lsq-exact`. p, I(ρ) and Ψ are held fixed. LCFS modes
   max(|m|, |n|) ≤ k are released for k = 2, 4, 6. Higher modes keep their VMEC
   values, and k must stay below the resolution.
3. Run a tight fixed-boundary solve on the new LCFS at the start's resolution, and
   save it as `<case>_fb_L{L}M{M}N{N}.h5` with its diagnostics.

To run it on Della, in jobs of under an hour:

```bash
F=desc_input/vmec_free_boundary/free_boundary_desc.py
a=$(sbatch --parsable --time=00:59:00 run_della.sbatch $F NCSX --k-list 2 4 --no-final)
b=$(sbatch --parsable --dependency=afterok:$a --time=00:59:00 run_della.sbatch $F NCSX --resume desc_input/vmec_free_boundary/NCSX/fb/free_boundary_last.h5 --k-list 6 --no-final)
sbatch --dependency=afterok:$b --time=00:59:00 run_della.sbatch $F NCSX --resume desc_input/vmec_free_boundary/NCSX/fb/free_boundary_last.h5 --k-list
```

Then pass `--eq <case>/<case>_fb_L16M16N12.h5` to `optimize_qa.py` in place of
the fixed-boundary file. The final solve chunks its Jacobian (`--final-jac-chunk-size`,
default 200) so that 16/16/12 fits a 40 GB A100.

NCSX (`NCSX/NCSX_fb_L16M16N12.h5`, one Stellar job, 26 min):

| stage | force error | B·n/\|B\| rms | pressure jump rms |
|---|---|---|---|
| start, 12/12/8 | 3.5e-3 | 4.4e-3 | 8.3e-3 |
| k = 2, 4, 6 | 3.5e-3, 3.3e-3, 7.6e-3 | 4.4e-3, 4.6e-3, 3.6e-3 | 7.6e-3, 5.4e-3, 2.8e-3 |
| final, 16/16/12 | 8.8e-4 | 4.5e-3 | 2.6e-3 |

The cost (B·n² + jump²) fell 10×, nearly all from the pressure jump: B·n stays near
4e-3. NEST's own free boundary (below) brings B·n/|B| to 3e-4.

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

## NEST with a free boundary (`scripts/optimize_fb.py`)

In the runs above the boundary is DESC's LCFS, held fixed. In the coils, NEST's
own field crosses it. On the LCFS of `NCSX_fb_L16M16N12.h5`, B·n/|B| is 2.8e-3
rms, with peaks of 2.8e-2 on the flanks of the bean section; this is the same for
the fits and for the fixed-boundary optima. Field lines that leave through the
boundary carry p = 0 inward, and the optimizer can build edge currents against
B·n. Both can open edge islands that the coils alone would not.
`nest/free_boundary.py` makes the boundary an unknown, solved together with J and
p.

- **Conditions.** B is the total field everywhere (coils plus the Biot–Savart
  field of the volume current), and J = 0, p = 0 on the boundary. So there is no
  sheet current, B is continuous, and [[B²/2μ0 + p]] = 0 holds identically. The
  condition left is **B·n = 0** on the moving boundary (`BoundaryNormalField`,
  `--w-Bn`). The toroidal flux through φ = 0 is held (`ToroidalFlux`), as DESC
  holds Ψ.
- **Unknowns.** x = (u, c_p, ξ). ξ(θ, ζ) is the normal displacement of the
  boundary in each φ plane (`--xi-M`, `--xi-N`, cos-type), extended inside as
  ρ^|m|; the geometry stays a series in the equilibrium's bases. u and c_p move
  with the coordinates (√g J^i and p(ρ, θ, ζ) carried along), so div J = 0, J = 0
  at the edge and the net current hold for any ξ. `--w-J` compares J with DESC's
  current carried the same way (`TransportedCurrentError`). `--fix-I` holds I(ρ).
  The stored energy ∫p dV is held in the moving volume (`StoredEnergy`).
- **B on the boundary by direct Biot–Savart** (`DirectField`, `--direct 32 4`).
  The box's B_plasma there is off by ~1.5e-3 |B| in B·n at a/10 and ~5e-4 at a/20,
  and it changes from one field period to the next: a Cartesian box is not
  3-fold symmetric. The box converges to the direct value as it is refined (a/6,
  a/10, a/14: 2.2e-3, 8.7e-4, 4.8e-4). A run on the box's B·n reached 1.3e-4 on
  its half-period grid, but 1.4e-3 over the whole boundary: it had fitted the
  box's error. The direct sum:
  - Gauss points in ρ;
  - sources q = 4 times the boundary grid in θ and ζ, offset so that none meets a
    target, and copied by the symmetries;
  - converged to ~4e-5 in B·n/|B|, exact in ξ.

  Its derivative is the box's (`--direct-tangent` would be exact, but runs out of
  memory at full size; see the docstring).
- **Anchors.** Two parts are first order in the displacement:
  - B_ext at the moved points (its gradient);
  - J on the box: the box points' coordinates shift by −(∂X/∂s)⁻¹D, through
    the derivative of the Lagrange interpolation (`BoxShift`).

  After `--inner` iterations, or a motion of `--max-disp`, ξ is absorbed into the
  geometry (`Displacement.apply`) and everything is rebuilt on the same box. This
  repeats until an anchor moves the boundary by less than `--disp-tol` (a/3000).
  The moved geometry is saved as a DESC file, `<out>_eq.h5`.
- **Checks** (`tests/test_free_boundary.py`, NCSX at a/6 on a CPU):
  - ξ = 0 reproduces `Fields` to 5e-15.
  - J v matches central differences to 2e-13 (u), 2e-12 (c_p) and 5e-10 (ξ); the
    adjoint identity holds to 6e-15; the direct field's derivative matches to
    1.5e-12.
  - Against the fields rebuilt on the displaced geometry, the box current
    differs at second order in ξ.

### NCSX: fixed against free boundary

The DESC input is `NCSX/NCSX_fb_L16M16N12.h5` (`free_boundary_desc.py NCSX`,
Stellar A100, 26 min). DESC's free-boundary stage cut the pressure jump from 8.3e-3
to 2.6e-3 rms, but not B·n/|B| (DESC's metric 4.5e-3 against 4.4e-3). Its force
error is 8.8e-4. The NEST fits miss its edge current (J ≠ 0 on its LCFS), as for
the VMEC-LCFS input; the optimization takes the difference up.

| | <\|F\|>/norm, check (a/20, grid (97, 385, 127)) | B·n/\|B\| rms (max), direct | ι |
|---|---|---|---|
| DESC | 8.78e-4 | | |
| fixed, `--w-J 1e-3` | 6.80e-5 | 2.9e-3 (2.8e-2) | core 0.35 → 0.44: current moved to the axis |
| fixed, `--w-J 1e-3 --fix-I` | 8.90e-5 | 2.8e-3 (2.8e-2) | DESC's |
| **free**, `--w-J 1e-3 --fix-I` | **1.20e-4** | **3.0e-4 (2.7e-3)** | DESC's |

- The free-boundary run continues the fixed `--fix-I` optimum and keeps W and Ψ
  to 1e-8. B·n maps and the boundary's motion: `figures/ncsx_free_boundary_Bn.png`
  (`scripts/fb_compare.py`).
- **Moved boundary:** up to 9.0 mm, 3.3 mm rms, mostly (m, n) = (5, 1), (4, 1),
  (5, 0), (2, 1) at 1.3–1.7 mm. The axis moved 1 mm.
- **Force by ρ:** 5.4e-5, 9.6e-5, 1.7e-4, 2.1e-4, 2.3e-4 on the bins
  0–0.5 … 0.95–1, against 2.4e-5 … 1.6e-4 with the fixed boundary.
- **Current:** it differs from DESC's by 0.14 rms, against 0.12 fixed.

Islands (`figures/ncsx_fixed_vs_free_poincare_periodized.png`, 800 periods with
B_plasma periodized, see [Poincare sections](#poincare-sections); each run in its own
coordinates). Resonant b_mn and the lines' thickness: `figures/ncsx_fb_resonant_overlap.png`
(`resonant_field.py`, `resonant_overlap.py`).
- **ρ 0.62–0.78:** nested surfaces in both fields, between thin 6/11, 9/16 and
  higher-order chains; the free 6/11 chain is a little wider. Chirikov σ of
  neighbouring chains is 0.26–0.71 (free) and 0.35–0.48 (fixed). The lines' thickness
  (rms of ρ about a smooth ρ(θ)) is 6e-4 (free) and 4e-4 (fixed) in the median, none
  above 6e-3, against 2e-2 in the chaotic layer of the 3/5 separatrix.
- **3/6, ρ 0.58:** about three times wider with the free boundary (w from b_mn 0.083
  against 0.029; ~0.10 against ~0.035 in the sections). The coils drive it alike
  (2|c_mn| 8.6e-4 and 6.5e-4 on the two geometries); the fixed field shields that ~5×,
  the free one not at all (9.8e-4 against 1.2e-4).
- **3/5, ρ 0.80:** about the same width (~0.09) in both, with a thin chaotic layer
  through the X-points. The plasma shields the coils' 1.5e-2 to 4.3e-4 (fixed) and
  1.0e-3 (free).
- **Edge:** surfaces by the thickness test (< 1e-2) in both; the free field has 9/14
  and 6/9 chains at ρ 0.93 and 0.97.
- **The a/10 stage** of the same run (`d3_fbD_L16`) is chaotic over the band even
  periodized: 15 of 27 circulating lines are thicker than 1e-2 (median 1.3e-2). Its
  high-order b_mn are 3–10× those at a/20 (σ 1–1.4; the islands would cover 0.89 of
  the band). The a/20 stage removes them, so they are resolution error, of the size of
  the a/10 box's error (1.5e-3 |B|). Judge islands at a/20, which has not itself been
  checked against a finer box.
- The sections drawn before the periodizing (`figures/ncsx_fixed_vs_free_poincare.png`)
  showed a narrow 3/5 chain inside a stochastic band at ρ 0.62–0.78 with the free
  boundary. Both came from the period seam: every line drifted outward (ρ 0.37 → 0.44
  over 800 periods) and lines away from islands smeared over 2e-2.
- Without `--fix-I`, ι drifts and moves the chains (3/6 to ρ 0.45; 3/7 and 3/8
  gone), so islands can be compared only with I(ρ) held.

Run (Stellar, after the fixed `--fix-I` run `fixedI_fbL16`). It was run in stages
(a/10 then a/20, w_Bn 10 then 1); the settings that worked are:

```bash
E="--eq desc_input/vmec_free_boundary/NCSX/NCSX_fb_L16M16N12.h5 --coils desc_input/vmec_free_boundary/NCSX/NCSX_coils.h5"
sbatch --qos=gpu-stellar --time=10:00:00 run.sbatch scripts/optimize_fb.py $E \
    --start results/ncsx/fixedI_fbL16.npz --w-J 1e-3 --fix-I --w-Bn 1 --xi-M 16 --xi-N 12 \
    --frac 10 --refine 20 --inner 12 --max-disp 3e-3 --max-step 1.5e-3 --direct 32 4 \
    --out results/ncsx/fb
sbatch --qos=gpu-stellar --mem=128G run.sbatch desc_input/vmec_free_boundary/poincare.py $E \
    --eq results/ncsx/fb_eq.h5 --x results/ncsx/fb.npz --skip-start --frac 20 --island-lines 12
python desc_input/vmec_free_boundary/poincare_flux.py results/ncsx/fb_eq.h5 \
    figures/ncsx_fb_poincare.npz desc_input/vmec_free_boundary/NCSX/NCSX_fb_L16M16N12.h5
```

(The later `--eq` wins over the one in `$E` for `poincare.py`.)

Lessons from the runs that led there:
- **ξ needs the equilibrium's full boundary resolution (16, 12).** With (12, 8),
  half of B·n (coil ripple at n = 9–11 per period) is out of reach. The optimizer
  then cancels it with edge currents and loses force balance.
- **Small anchors.** After a 3.3 mm move, re-anchoring shifts B·n by ~1.5e-4, and
  the next anchor opens with a ~1.3 mm correction: the anchors cycle.
- **w_Bn 1, not 10.** With the box derivative, the steps stop decreasing the true
  residual once B·n/|B| is ~2e-4 at a/10 (inexact Newton). w_Bn = 1 lets force
  balance lead, and a/20 lowers the derivative's error.
- **Memory.**
  - `release` frees each anchor's arrays; JAX's caches kept ~4 GB per anchor.
  - The platform allocator (the `run.sbatch` default) is needed at a/20.
- Neither 300 CG iterations nor the coupled preconditioner sped the convergence
  up.

## Poincare sections

Only `nest/` is needed.
- `poincare.py` traces the field lines of the start and of the optimized x. It saves
  the crossings of phi = 0 and of the half period phi = pi / NFP (GPU, a few minutes;
  `--x start` traces the start only).
  - `--islands PREV_poincare_flux.npz` seeds each field's own chains, at the resonances
    of its measured iota. The seeds go across each chain (rho_res +- 0.03) at the
    straight-field-line angles theta* = 0, pi/m, pi and pi + pi/m, so the lines inside
    the islands are traced too.
  - `--island-lines` adds seeds on theta = pi / m around DESC's resonances.
  - B_plasma is averaged over the NFP rotations before tracing (`--no-periodize`
    skips it). The box's field is not exactly periodic, because a Cartesian box is not
    NFP-symmetric. Between periods it differs by 1e-4 |B| rms at a/20 (NCSX; 1.7e-3 at
    most). Tabulated over one period and repeated, it jumps at the seam, and the jump
    acts as a source of B. Every traced line then moves outward (NCSX free boundary:
    ρ +0.07 over 800 periods, +0.013 fixed), and lines between the chains look
    chaotic. Periodized, the drift is < 1e-3. `scripts/validate_opt.py` and
    `scripts/islands.py` still trace the field unperiodized.
- `resonant_overlap.py` checks a long trace, per line:
  - the secular drift of ρ, which a divergence-free field cannot give all lines alike;
  - the thickness: the rms of ρ about a Fourier fit ρ(θ). A surface stays at the
    tracer's error (1e-4 in the core, < 6e-3 near small chains), while a chaotic line
    reaches 1e-2 or more.

  It also gives the Chirikov σ of the chains from `resonant_field.py`'s b_mn. With
  this tracer, the weighted Birkhoff rotation number also stops converging on
  surfaces (at 1e-4–1e-5), so it cannot flag chaos.
- `poincare_flux.py` maps the crossings to DESC (rho, theta) at both planes and
  measures iota per line (CPU, a few minutes).
- `poincare_rz.py` draws them in (R, Z) over DESC's flux surfaces (CPU, seconds).

```bash
sbatch --mem=128G run.sbatch desc_input/vmec_free_boundary/poincare.py $E --x results/ncsx/wJ.npz --frac 20
python desc_input/vmec_free_boundary/poincare_flux.py desc_input/vmec_free_boundary/NCSX/NCSX_L16M16N12.h5 figures/ncsx_wJ_poincare.npz
sbatch --mem=128G run.sbatch desc_input/vmec_free_boundary/poincare.py $E --x results/ncsx/wJ.npz --frac 20 --islands figures/ncsx_wJ_poincare_flux.npz --out figures/ncsx_wJ_islands_poincare.npz
python desc_input/vmec_free_boundary/poincare_rz.py desc_input/vmec_free_boundary/NCSX/NCSX_L16M16N12.h5 figures/ncsx_wJ_islands_poincare.npz
```

NCSX figures:
- `figures/ncsx_opt_L16M16N12_poincare_flux.png`: no penalty.
- `figures/ncsx_opt_L16M16N12_wJ1e-2_poincare_flux.png`
- `figures/ncsx_opt_L16M16N12_wJ1e-3_islands_poincare_{flux,rz}.png`: `--w-J 1e-3`, with
  the island interiors.

## NEST on QH and CTH-like

Same workflow as NCSX:
1. the DESC re-solve (bold rows above);
2. `optimize_qa.py`, h = a/10 then a/20 in two Della jobs;
3. the Poincare sections with `--islands`.

| | DESC | NEST (check) | `--w-J` | iota vs DESC |
|---|---|---|---|---|
| QH_5beta 16/16/12 | 6.48e-4 | **4.30e-5** (15x) | 1e-3 | within 0.01 for rho > 0.15; 1.32 against 1.28 at the axis |
| CTH_like 12/12/8 | 7.60e-4 | 1.27e-4 | 1e-3 | core 1.16 against 1.25 |
| CTH_like 12/12/8 | 7.60e-4 | **1.46e-4** (5x) | 3e-3 | core 1.22 against 1.25 |

- **QH:** the optimized field lies on DESC's surfaces out to the LCFS, where the start
  does not (rho swings 0.75–0.95 beyond rho 0.75). The 7-island 8/7 chain at rho 0.55
  shrinks to a thin chain; 12/10 at rho 0.33 and 12/11 at 0.72 are thin in both
  (`figures/qh_opt_L16M16N12_wJ1e-3_islands_poincare_{flux,rz}.png`).
- **CTH-like is a stress test.** DESC's iota sits just above 5/4 = 1.25 across the
  whole core with almost no shear, so a drop of 0.03 crosses the resonance. Even at
  `--w-J 3e-3` a second, wide 4-island 5/4 chain opens at rho 0.39, next to the one at
  0.71 that DESC's own iota has. The edge beyond rho 0.8 is chaotic in both fields,
  with 5/5 and 10/9 chains (`figures/cth_opt_L12M12N8_wJ3e-3_islands_poincare_{flux,rz}.png`).
  The pointwise J penalty cannot hold iota this close to a rational. A constraint on
  the flux-surface averaged current I(rho) would.


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

## NCSX: resonant forcing against iota

`resonant_field.py` (GPU) Fourier-analyses b^rho = B.grad rho / B.grad zeta on DESC's
surfaces in (theta*, zeta). `resonant_widths.py` (CPU) compares b_mn of each chain, at
the field's own resonance, with the island width measured in the Poincare sections.
Since iota < 0, the resonant harmonic is (m, m iota), e.g. (5, -3) for 3/5. The
"2/3" chain is 6/9, at the iota maximum, where there is no pendulum width.

| field | 3/7: rho, b, w (pred / meas) | 3/6 | 3/5 |
|---|---|---|---|
| start | 0.365, 2.0e-5, 0.013 / 0.007 | 0.582, 1.1e-4, 0.028 / 0.024 | 0.796, 1.1e-3, 0.080 / 0.086-0.098 |
| w_J = 0 | (not crossed) | 0.428, 3.9e-4, 0.043 / 0.051-0.064 | 0.785, 6.5e-4, 0.054 / 0.045-0.053 |
| w_J = 1e-3 | 0.310, 2.6e-4, 0.046 / 0.044-0.052 | 0.569, 5.5e-5, 0.022 / 0.025-0.028 | 0.796, 5.9e-4, 0.056 / 0.047-0.073 |
| w_J = 1e-2 | 0.402, 6.7e-5, 0.022 / 0.022-0.026 | 0.597, 2.9e-4, 0.043 / 0.044-0.046 | 0.796, 5.5e-4, 0.055 / 0.046-0.057 |

The measured range is the raw extent of the locked lines, and that extent after
removing the non-resonant wobble of the neighbouring surfaces. The coils alone give
b_5,-3 = 2.0e-2 at rho 0.795: the plasma shields the vacuum 3/5 drive by about 20x.
Figure: `figures/ncsx_resonant_field_widths.png`.

```bash
sbatch --mem=128G --time=00:40:00 run_della.sbatch desc_input/vmec_free_boundary/resonant_field.py $E --x wJ0=results/ncsx/opt_L16M16N12.npz wJ1e-3=results/ncsx/opt_L16M16N12_wJ1e-3.npz --out ncsx_resonant_field
python desc_input/vmec_free_boundary/resonant_widths.py figures/ncsx_resonant_field.npz wJ1e-3=figures/ncsx_opt_L16M16N12_wJ1e-3_islands_poincare_flux.npz
```
