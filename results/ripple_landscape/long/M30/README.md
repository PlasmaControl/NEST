# M30: 30/30/8 α sweep and α = 0.001 floor (long-run chain, Stellar AI)

The M30 run of `docs/ripple_long_runs_runbook.md`, run with `scripts/cluster/ripple_long/` at commit dfda188 on
Stellar AI (one NVIDIA B200 per job; JAX 0.9.2, CUDA 13), 2026-11-16/17. Preflight passed every check first.

Grid and flags: `--LMN 30 30 8 --N-phi 33 --frac 100` (n_u = 8958), `--bext desc-fit`, and the recommended flags
(`--subspace prev,accel --cg 1500 --no-rebuild-on-hit --f-noise auto --radius-by-delta --no-share-stop
--locate-chunk 131072`). The exact command of every segment is in `segments/segNN/segment.txt`.

## Result

| quantity | J_fit (start) | end of phase A (α = 0.1) | floor (end of seg40) |
|---|---|---|---|
| M / M_DESC (solve grid) | 2.269 | 0.1195 | **0.10335** |
| \|J − J_fit\| / rms | 0 | — | 0.077 |
| b21 / vacuum, phase | 0 | 0.0075, −90° | **0.0154, −90°** |
| q = 2 band p16 \|F\| / DESC | 4.93 | 0.304 | 0.0795 |
| mean \|F\| outside the band / DESC | 6.41 | — | 0.1016 |

- **Shielding survives to the 30/30/8 floor.** The b21 harmonic stays on the anti-island (over-shielded) side at
  −90° throughout; it rose to ≈ 0.019 of vacuum early in the floor phase and settled at 0.0154 (figure, right). No
  phase flip, never near the ≈ 0.05 warning level; islands in earlier states had 6–12 %.
- **Phase A ended at the first α.** α = 0.1 is not reachable at this resolution: the augmented-Lagrangian sweep
  stopped (`--stall-stop`) at M = 0.1195 M_DESC after 0.29 h, so α = 0.05/0.02/0.01 were not run and phase B started
  from the α = 0.1 end state.
- **Floor.** M/ε fell 119.5 → 103.35 over 4.5 h of solve time (1612 accepted steps), most of it in the first 2 h;
  the last 23 segments together gained 0.04 % (103.40 → 103.354). The residual force is no longer concentrated at
  q = 2: the band p16 fell below the mean outside it.

## How the chain ended

`FINISHED`: *reached 40 segments*. The runbook's default cap is 10; it was raised to 40 for M30 (`MAX_SEG=40` in
`site.sh`) because the floor segments were short. Each segment's single inner solve stops on its flat test (merit
change < 1e-4 relative over 10 accepted steps) after 0.02–1 h, and the next segment restarts with a fresh metric,
which usually gained 0.1–2 % again early on. From segment 18 on most segments stopped after exactly 11 steps with
almost no gain. The chain's stall rule did not end the run because it only counts segments with more than 0.5 h of
solve time, so these short segments were never judged slow; only the segment cap ended it. Replaying the history with
a rule that pools short segments into ≥ 0.5 h windows would have stopped it at segment 26 (M/ε 103.392).

## Files

| file | contents |
|---|---|
| `report.md`, `M30.png` | `chainlib.py report` output: chain table, first→last diagnostics, figure (M/M_DESC vs solve time; b21/vac per [diag] record) |
| `steps_M30.csv` | every accepted inner step (segment, phase, α, cumulative solve h, M/ε, M/M_DESC, …) |
| `diag_M30.csv` | every `[diag]` record (M/M_DESC, \|J − J_fit\|/rms, b21/vac and phase, band p16/max, outside mean) |
| `chain.tsv`, `jobs.txt`, `FINISHED` | per-segment summary table; Slurm job ids and resubmissions; end reason |
| `segments/segNN/` | each segment's `summary.json` and `segment.txt` (command, start state, decision) |
| `logs_and_json.tar.gz` | each segment's `run.log`, full `M30.json` (every step, diag log, setup timings) and `gpumem.txt` |
| `M30_floor_final.npz` | final floor state (end of seg40): `u` (8958), `c_p`, `nu`, `rho`, `eps` |
| `M30_sweep_a0.1_end.npz` | end of phase A (α = 0.1 stage) |

Re-evaluate a state with the same `--LMN 30 30 8` (see runbook §7, "want to inspect a state").
