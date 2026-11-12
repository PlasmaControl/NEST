# Resolution rungs R0 and R1 of the ripple landscape study (2026-11-12)

Context: `docs/ripple_landscape_report.md` (Sections 6–11). Question: does the island–shielding force gap close with
resolution (model C) or stay (a tilted single valley)?

Fixed across rungs: the L32 DESC file `bp2/L32N4/tokamak_ripple_C_L32M32N4_x0.001.h5`, option A, N_phi 25, N = 6;
forces are volume-mean |F| / M_DESC, with M_DESC = 4.9995e-7 at both rungs (same DESC file). Every run started from
the 32/32/6 end states of the original study (embedded with `--start-LMN 32 32 6` at R1), with fresh multipliers
(nu, rho stripped), the α = 0.1 mean bound, at most 400 inner steps and a per-run time cap
(`scripts/modal_fixed/ripple/job_res.sh`). The follow-up (`job_res2.sh`) continued the R1 shielded and β = 1 states
toward α = 0.05 (neither met it; 25 / 50 min caps).

| rung | basis | field grid | h L / a | n_u | per-run cap | job |
|---|---|---|---|---|---|---|
| R0 | 32/32/6 | a/60 | 0.53 | 7,726 | 1,980 s | 5cb85e32 (A100-40GB, 3.04 h) |
| R1 | 40/40/6 | a/80 | 0.50 | 11,734 | 4,380 s | 0d615475 (A100-40GB, 4.66 h) |
| R1 follow-up | 40/40/6 | a/80 | 0.50 | 11,734 | 1,500 / 3,000 s | a985b25a (A100-80GB, 1.39 h) |

## Results: mean |F| / M_DESC (solve grid / one grid finer)

| state | R0 (a/60 / a/80) | R1, α 0.1 (a/80 / a/100) | R1, toward α 0.05 (a/80 / a/100) |
|---|---|---|---|
| shielded (no pin) | 0.1235 / 0.1225 | 0.1000 / 0.1105 | 0.0705 / 0.0831 (56 steps) |
| β = 0.2 | 0.1649 / 0.1584 | 0.1003 / 0.1040 | — |
| β = 0.5 | 0.1697 / 0.1607 | 0.1000 / 0.1030 | — |
| β = 1.0 | 0.2153 / 0.2031 | 0.1000 / 0.1014 | 0.0916 / 0.0976 (177 steps; 0.0938 at step 56) |
| released from β = 0.5 | 0.1280 / 0.1240 (b21 11.8%) | 0.0998 / 0.1059 (b21 6.3%) | — |

Gap G = M(β = 1) − M(shielded), on the solve grid / one grid finer:

- R0: 0.092 / 0.081.
- R1 at α 0.1: 0.000 / −0.009, because both states meet the bound.
- R1 toward α 0.05: 0.021 / 0.0145. At matched steps (56) on the solve grid it is 0.023.

R1's gap is therefore about 0.25 × R0's (a/80 against a/80 evaluations: 0.021 against 0.081). It is comparable in size
to R1's own grid error (the a/80 → a/100 shift of single states is 0.001–0.013).

|J − J_fit|/rms at R1 (α 0.1): shielded 0.0070, β 0.5 0.063, β 1 0.102, release 0.030.

Island diagnostics (`diagnostics/R*_diag.json`), β = 1:

| quantity | R0 | R1 |
|---|---|---|
| traced radial spread | 0.101 | 0.101 |
| pendulum width | 0.113 | 0.113 |
| along-line p ratio | 0.010 | 0.005 |
| λ21 inner / outer peak | 2506 / 1074 | 2565 / 1221 |

The a/100 evaluation of the R1 α = 0.1 end states ran in the first follow-up job (d0de187d), which was orphaned after
it finished. Its JSON is on the Modal volume at `/results/fxr/res/R1b/R1b_ev100.json`; the numbers above were read
from it.

## Files

- `states/`: end states (`R0_*`, `R1_*` at α 0.1, `R1b_*` toward α 0.05).
  - `R0_*` load with `--LMN 32 32 6 --frac 60`; `R1*` load with `--LMN 40 40 6 --frac 80`.
- `runs/`: per-step logs of every run and evaluation.
- `diagnostics/`: island, Poincaré, pressure and λ21 diagnostics (`R0_diag.json`, `R1_diag.json`, Poincaré npz) and
  the evaluation JSONs.

## R1 grid-convergence test (job 39c8d0a0, A100-80GB, 2.13 h; `job_res3.sh`)

This test reran the α → 0.05 continuation of the R1 shielded and β = 1 states. Both started from the same R1 α = 0.1
end states and ran 80 inner steps each. Neither met 0.05. Two grid variants were used:

- R1c: field grid a/80, 98 radii.
- R1f: field grid a/100 (h L / a = 0.40), 98 radii.

All end states were then evaluated on one reference grid, a/120 with 120 radii. Values are M / M_DESC.

| variant | shielded, solve grid | β = 1, solve grid | gap, solve grid | shielded, reference | β = 1, reference | gap, reference |
|---|---|---|---|---|---|---|
| R1b (a/80, 55 radii) | 0.0706 (step 56) | 0.0938 (step 56) | 0.0232 | 0.0755 | 0.0909 (177 steps) | 0.0154 |
| R1c (a/80, 98 radii) | 0.0711 | 0.0932 | 0.0221 | 0.0745 | 0.0900 | 0.0155 |
| R1f (a/100, 98 radii) | 0.0616 | 0.0815 | 0.0199 | 0.0637 | 0.0822 | 0.0185 |

- **Radial collocation:** going from 55 to 98 radii changes nothing beyond about 1%.
- **Field grid:** a/100 finds lower states for both shielded and β = 1, about −0.010 on the reference grid. At α ≈ 0.05
  force levels the floor is therefore set by the field grid.
- **The gap is stable:** it is about 0.015–0.019 on the reference grid across all three variants, about 0.2 × R0's
  0.081. It does not shrink as the grids are refined.
- **Effort caveat:** at step 80, β = 1 was still falling faster than shielded (−0.0035 against −0.001 over steps
  25 → 80). The gap could close somewhat with more steps.
- **Comparison caveat:** R0's gap is from α = 0.1 runs (R0 cannot get below 0.12), R1's from the α → 0.05 push.
- **Non-monotone evaluations:** single-state evaluations on different grids do not move monotonically. For R1b_S:
  a/80 (55 radii) 0.0705, a/100 (55 radii) 0.0831, a/120 (120 radii) 0.0755.
