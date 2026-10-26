# Hand-off: `speed elim` (p from the weak Poisson equation of force balance) at production size

Branch `pressure-poisson` (= `lm-trust` + this work). Plan and all laptop numbers: `docs/pressure_poisson_plan.md`.

## 1. What it is

The force rows are affine in p with constant coefficients, so the optimal p for a given current u solves the
weak (force-row-weighted) Poisson equation of force balance on the pressure space (p(1) = 0):

    sum_points w grad(phi_i) . grad p = sum_points w grad(phi_i) . (J x B)    (+ the constant <p> row)

`--speed ...,elim` takes the LM / trust-region step on u alone with the exact reduced Jacobian (I - P) A_u: the
trailing block of the same streamed TSQR with the columns ordered [p | u | r]. Every trial re-solves p exactly (one
pass), so every accepted iterate has p = p*(u). Same minimiser as the joint problem; same TSQR cost per iteration.

- Log line additions: `|c_p|/|r|` (the leftover p part of the residual; should be <= 1e-8), `f* chk` (the trial
  pass's f against a direct evaluation; <= 1e-8), `p min/max` (p >= 0 is not enforced).
- Asserted: par = 1, no transport, w_pos = 0, `--steps qr`, not with newton / soc / reuse / pproj / accel.
  The production island stages already run `--w-W 0 --w-pos 0`.
- `--p-factors L M N`: the pressure space's resolution, independent of the current's (laptop: saturates at
  ~1.5x the current factors; the default costs ~7% in f and <|F|>).

## 2. Laptop evidence (lm_ab.py, eps 1000 A, w_J 1e-7 from the w_J 1e-3 stage, 30 its; w_J 1e-3 from the eps 0 base)

| model (current basis, a/h, unknowns) | kappa phys w_J 1e-3: joint / reduced | kappa phys w_J 1e-7: joint / reduced | f/f0 after 30 its, w_J 1e-7: phys,tr / elim,phys,tr / elim + p 1.5x | s/it (same three) | its (time) to converge w_J 1e-3: phys,tr / elim,phys,tr |
|---|---|---|---|---|---|
| L16M16 f1, a/20, 750 | 7.2e3 / 275 | 2.4e5 / 1.4e4 | 0.527 / 0.424 / 0.378 | 3.5 / 3.6 / 5.6 | 21 (65 s) / 11 (48 s) |
| L16M16 f1.5 (L24), a/30, 1,554 | 1.45e4 / 277 | 1.0e6 / 3.0e4 | 0.464 / 0.409 / 0.330 | 18 / 18 / 33 | 24 (390 s) / 9 (186 s) |
| L32M32 f1 (L32, production eq), a/40, 2,646 | 2.3e4 / 283 | 2.5e6 / 4.7e4 | 0.471 / 0.288 / 0.280 | 109 / 108 / 200 | 18 (1926 s) / 16 (1685 s) |

f0: the joint start (the w_J 1e-3 stage state); both variants reach the same converged f at w_J 1e-3 (2.534e-10 at L32).
The reduced kappa at w_J 1e-3 is flat in L (275, 277, 283) while the joint one grows (~L^1.6). At w_J 1e-7 both grow
(the w_J floor), the reduced one ~50x lower at L32. `results/lm/lad_*`, `lad_cond_*`.

## 3. Production runs (Stellar AI; same flags as `a120Cn2d1000`, rsync the code first)

N = 2 at 1000 A has 16,457 unknowns (u 11,078, p 5,379; collocation 97 x 385 x 5). With `--p-factors 3 3 1`
(1.5x the base's 2 2 1) p grows to ~12.1k (estimate), the TSQR width to ~23.2k: ~2x the time per iteration, R ~4.3 GB.

```bash
COMMON="scripts/dshape/isl_scan.py --base results/gs/hN_a120D --N 2 --eps 1000 --w-J 1e-7 --j-mode full \
  --w-W 0 --w-pos 0 --iters 30 --rtol 1e-12 --gn-tol 0 --batch 16 8 --start-from a120Cn2d1000"
FAST="resident,chol,narrow,merge=100000,gpus=4"
# reference (lm-trust's best): already abN2B1000 if it was run
sbatch --gres=gpu:nvidia_b200:4 --mem=500G --job-name=abB run_sai.sbatch $COMMON --speed $FAST,phys,tr       --name abN2B1000
sbatch --gres=gpu:nvidia_b200:4 --mem=500G --job-name=abD run_sai.sbatch $COMMON --speed $FAST,elim,phys,tr  --name abN2D1000
sbatch --gres=gpu:nvidia_b200:4 --mem=500G --job-name=abE run_sai.sbatch $COMMON --speed $FAST,elim,phys,tr \
  --p-factors 3 3 1 --name abN2E1000
# conditioning at the start state (one TSQR + QR of R + 5 SVDs of ~16k; with --p-factors 3 3 1 ~23k)
sbatch --gres=gpu:nvidia_b200:1 --mem=300G --job-name=cond run_sai.sbatch scripts/diagnostics/pressure_elim_conditioning_isl.py \
  --base results/gs/hN_a120D --N 2 --eps 1000 --w-J 1e-7 --j-mode full --w-W 0 --w-pos 0 --batch 16 8 \
  --state results/isl/a120Cn2d1000_eps1000.npz --speed resident --out results/isl/cond_a120Cn2d1000.json
```

What to send back: the `it` lines (with `|c_p|/|r|`, `f* chk`, `p min/max`), and the cond json. The decisive
comparison is abD vs abB at equal iterations (and wall time), abE vs abD for the pressure space.

## 4. Caveats

- Laptop models only so far (up to L32, 2,646 unknowns). At w_J 1e-7 elim's f after 30 its is 1.1-1.6x lower (1.64x
  at L32); at w_J 1e-3 it needs 1.1-2.7x fewer iterations (only 1.1x at L32). Once (L16, p 1.5x, w_J 1e-3) the joint
  phys,tr converged faster (8 vs 16 its). The 1.5x pressure space costs 1.6-1.9x per iteration on the laptop; its
  gain in f shrinks at L32 (0.288 -> 0.280).
- The production --p-factors 3 3 1 is 1.5x the base's 2 2 1 (estimate ~2x per iteration); run abD first.
- At small w_J the reduced kappa is floored by the equilibrium-family directions held only by w_J: the w_J plateau
  test (docs/lm_solver_handoff.md section 5) still applies.
- The converged p has small negative regions (min/max -1e-4 .. -3e-3) in the joint and elim solutions alike.
