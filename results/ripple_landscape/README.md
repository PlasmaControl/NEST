# results/ripple_landscape: states, logs and diagnostics of the (2,1) ripple landscape study

This folder holds the inputs and outputs behind `docs/ripple_landscape_report.md`. Read the report first; this
file is the manifest and the "how do I use these states" note.

All states are NEST fixed-boundary states of the candidate-C ripple equilibrium, option A external field (see the
report, Section 1.4). Unless noted otherwise they were solved with:

| setting | value | flag |
|---|---|---|
| DESC equilibrium | `desc_input/tokamak_boundary_perturbed/bp2/L32N4/tokamak_ripple_C_L32M32N4_x0.001.h5` | `--eq` |
| external field | option A: B_ext = B_DESC − B_ann[J_fit] | `--bext desc-fit` |
| current basis | L, M, N = 32, 32, 6 (n_u = 7726 coefficients) | `--LMN 32 32 6` |
| annulus field grid | spacing a/60 (168 × 247 × 25 = 1.04 M nodes) | `--frac 60` |
| toroidal planes of the annulus | 25 | `--N-phi 25` |
| collocation grid | 49 radii × 193 θ × 16 ζ (half period) = 151,312 points | default `--colloc 1.5` |
| pressure basis | n_p = 3655 coefficients, eliminated by least squares | (automatic) |
| force bound | volume mean of \|F\| ≤ α M_DESC, α = 0.1 | `--alphas 0.1` |

M_DESC (DESC's own volume-mean |F| on the same collocation grid, in the code's force units) is 4.9995e-7.

## Files in a state `.npz`

| key | meaning |
|---|---|
| `u` | current-potential coefficients (length 7726 at 32/32/6, 16574 at 48/48/6). This is the state. |
| `c_p` | least-squares pressure coefficients for that `u` (recomputed on load; kept for convenience) |
| `nu` | augmented-Lagrangian multipliers at the end of the run (3 entries) |
| `rho` | augmented-Lagrangian penalty at the end of the run |
| `eps` | the force bound ε = α M_DESC in force units |

## How to use a state

Warm start a new run (any objective or constraint) from a state:

```bash
python scripts/meanforce/mf_fixed.py --eq desc_input/tokamak_boundary_perturbed/bp2/L32N4/tokamak_ripple_C_L32M32N4_x0.001.h5 \
    --bext desc-fit --LMN 32 32 6 --frac 60 --N-phi 25 --alphas 0.1 \
    --start results/ripple_landscape/states/beta_1.0.npz \
    --f-noise auto --radius-by-delta --no-share-stop --time-budget 1800 --out out/myrun
```

`--start` also loads `nu` and `rho`. The β-scan states end with ρ = 1e8, so for a fresh optimization strip them first
(this is what the release tests did):

```python
import numpy as np
d = np.load("results/ripple_landscape/states/beta_0.5.npz")
np.savez("fresh.npz", u=d["u"], c_p=d["c_p"])
```

Evaluate states without solving (force measures, band p-norms, resonant b21):

```bash
python scripts/meanforce/mf_fixed.py ... --eval-only --b21-report \
    --eval-state b1=results/ripple_landscape/states/beta_1.0.npz --eval-state start=results/ripple_landscape/states/shielded_start_A60_a0.1.npz
```

Island, Poincaré, pressure and J∥ diagnostics (what produced `diagnostics/bd_m2*.{json,npz}`):

```bash
python scripts/meanforce/start_check_bext.py --eq <EQ> --LMN 32 32 6 --frac 60 --N-phi 25 --m 2 --k 1 \
    --rho-grid 0.30 0.98 341 --band 0.04 0.08 0.16 --state <npz> [--state <npz> ...] \
    --perts "vacuum start:A b1:A" --turns 300 --lines 41 --seed-band 0.50 0.78 --seed-theta 0.0 --pressure --out out/diag
```

The coefficient vector `u` does not depend on the annulus grid, so a state can be evaluated at another `--frac`
(the grid only changes how accurately B_plasma is computed). It does depend on the basis (`--LMN`) and on the DESC
file (through J_fit and B_ext). A state from a different basis can be loaded with `--start-LMN`.

All runs shown here used a GPU (Modal H100; about 10 GB of JAX device memory, so an A100-40GB is enough).
The job scripts are in `scripts/modal_fixed/ripple/`.

## states/

| file | what it is | how it was made | mean \|F\| / M_DESC |
|---|---|---|---|
| `A40_a0.3.npz` | α = 0.3 end state at a/40 | α continuation from J_fit at a/40 (`job_A32_a03`, earlier work) | 0.300 (a/40) |
| `A60_a0.3.npz` | α = 0.3 end state re-seated at a/60 | `job_A60_cont.sh`, stage α 0.3 | 0.300 |
| `shielded_start_A60_a0.1.npz` | **the shielded start** used by every experiment below | `job_A60_cont.sh`, stage α 0.1 (stalled before reaching 0.1) | 0.1243 (0.1274 on the 97-radius grid) |
| `A48_cont_a0.1.npz` | 48/48/6 state at a/60 (same L32 DESC file as all other states; load with `--LMN 48 48 6`) | `job_A48.sh` | 0.102 at a/60, but 0.142 / 0.171 / 0.165 at a/80 / a/100 / a/120 (121 radii): **not grid converged at its solve grid**; re-solve at a/100 or finer before using it for physics |
| `seed_q2_w0.07.npz` | island seed: J_fit with its helical current removed for \|ρ − 0.6375\| ≤ 0.07 (cos² taper 0.03) | `scripts/meanforce/island_seed.py` | 120.6 (b21 = 18% of vacuum) |
| `seed_q3_w0.03.npz` | the same at q = 3 (ρ 0.9131, half width 0.03) | `island_seed.py` | 164.1 |
| `seed_all_helical.npz` | J_fit with all helical current removed (the vacuum drive; defines the island phase) | `island_seed.py --window -1` | 267.3 (b21 = 88% of vacuum) |
| `Q2_a0.5.npz` | first band-max run (q = 2 band p16 only), α 0.5 stage | `job_MQ2.sh` | not re-evaluated |
| `Q2b_a0.25.npz` | tight band-max run, α 0.25 stage | `job_MQ2b.sh` | 0.128 |
| `Q2b_stall_a0.1.npz` | tight band-max run, α 0.1 stall | `job_MQ2b.sh` | 0.141 (band p16 = 0.103 × start's) |
| `pushthrough_v3_end.npz` | push-through from the Q2b stall with noise-aware trust region | `job_PTA.sh` (v3) | 0.140 (band p16 = 0.103 × start's) |
| `beta_<β>.npz`, β ∈ {0.02, 0.05, 0.1, 0.2, 0.35, 0.5, 0.75, 1.0} | b21 pinned at β × vacuum in the island phase, mean bound α 0.1, 30 min each | `job_bs1.sh`, `job_bs2.sh` | 0.126 → 0.247 (report Table 3) |
| `release_from_beta0.05.npz` | β = 0.05 state with the pin removed, 20 min | `job_bd.sh` | 0.1228 (b21 0.83%) |
| `release_from_beta0.5.npz` | β = 0.5 state with the pin removed, 20 min | `job_bd.sh` | 0.151 (b21 9.6%, still closing) |

Values for the band-max states are on the 97-radius collocation grid (`--colloc-rho 3.0`) used by those runs.

## runs/

Full per-iteration logs (every inner step: merit f, ratio, |y|, trust radius Δ, M/ε, boundary hits) of the runs
above: `A60_cont.log` (+ `.json`), `A48_cont.log`, `Q2b.log`, `pushthrough_v3.log`, `bs<β>.log`, `rel0.05.log`,
`rel0.5.log`; and of the evaluation jobs: `dqb_*` (Q2b end states), `pv_*` (profile diagnostics of stall vs pushed),
`pm_*` (plane maps and the b21-pin smoke test), `bsdiag*` and `bd_*` (β-state diagnostics and b21 reports).

## diagnostics/

| file | content |
|---|---|
| `Q2b_alpha0.1_inner_steps.txt` | one row per inner step of the Q2b α 0.1 stage (stall anatomy) |
| `pv_ev60.json`, `pv_m2.json` | force by shell, λ_mk(ρ), ⟨p⟩(ρ), b21(ρ) for start, stall and pushed states |
| `pm_ev60.json`, `pm_ev60_plane_*.npz` | exact quadratic force maps on three planes through the shielded start |
| `bd_m2.json` | per-state island diagnostics for the β states: b_mk(ρ), pendulum width, Poincaré spread, along-line p, λ_mk(ρ), ⟨p⟩(ρ) |
| `bd_m2_poincare_*.npz` | Poincaré puncture points (`_p_` files carry p along each line) |
| `bd_b21.json` | b21 and force of the release-test snapshots |
| `bd_dist.txt` | distances and direction cosines between the β states |
