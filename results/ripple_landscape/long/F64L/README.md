# F64L: 64/64/6 α = 0.001 floor (long-run chain, Stellar AI) — interim, run in progress

The F64L run of `docs/ripple_long_runs_runbook.md`, run with `scripts/cluster/ripple_long/` at commit dfda188 on
Stellar AI (one NVIDIA B200 per job; JAX 0.9.2, CUDA 13). Started 2026-11-16 15:15 from
`results/ripple_landscape/valley/states/V_SA_budget.npz` (M/ε 16.1438). **Snapshot taken 2026-11-17 15:38, during
segment 7; the chain is still running**, so these files will be superseded.

Grid and flags: `--LMN 64 64 6 --N-phi 25 --frac 200` (n_u = 28,750), `--bext desc-fit`, `--alphas 0.001
--max-outer 1 --maxiter 100000`, and the recommended flags (`--subspace prev,accel --cg 1500 --no-rebuild-on-hit
--f-noise auto --radius-by-delta --no-share-stop --locate-chunk 131072`). Exact commands: `segments/segNN/segment.txt`.

## Result so far

| quantity | start (V_SA_budget) | end of seg01 (15.7 h) | snapshot (seg07, 4.1 h in) |
|---|---|---|---|
| M/ε | 16.143 | 13.962 | **13.44** |
| M / M_DESC (solve grid) | 0.01614 | 0.01396 | **0.01344** |
| \|J − J_fit\| / rms | 0.0127 | — | 0.0606 |
| b21 / vacuum, phase | 0.0208, −90° | 0.0129, −90° | **0.0106, −90°** |
| q = 2 band p16 \|F\| / DESC | 0.177 | 0.137 | 0.100 |
| mean \|F\| outside the band / DESC | 0.0132 | — | 0.0123 |

- **Shielding survives so far, and the change at q = 2 shrinks.** b21 (of the change from J_fit) stays at −90°
  (anti-island side) and falls steadily as M falls, 0.021 → 0.011 of vacuum (figure, right). No sign of a phase flip.
- **The force concentrated in the q = 2 band falls with M** (band p16 0.177 → 0.100 of DESC's), unlike the earlier
  64/64/6 history where it rose toward the floor (0.159 → 0.180); the rest of the volume barely changes.
- **Not at a floor yet.** 1603 accepted steps in 23.1 h of solve time; M/ε is falling ≈ 0.8 % per hour in segment 7
  (figure, left), faster than at the end of segment 1.

## Segments

Segment 1 ran 15.7 h in one inner solve. Segments 2 and 4–6 each stopped on the inner flat test (merit change
< 1e-4 relative over 10 accepted steps) after 11–14 steps with no gain; segments 3 and 7 restarted descent with a
fresh metric. Segment 5's solve finished normally, but its end-of-segment bookkeeping failed because `segment.sh` was
edited while it ran (bash syntax error, exit 2); its `summary.json` was written by hand and the chain resubmitted
(noted in `jobs.txt` and `segments/seg05/segment.txt`). No state was lost.

## Files

| file | contents |
|---|---|
| `report.md`, `F64L.png` | `chainlib.py report` output (chain table of finished segments, first→last diagnostics, figure) |
| `steps_F64L.csv`, `diag_F64L.csv` | every accepted inner step; every `[diag]` record (including segment 7 so far) |
| `chain.tsv`, `jobs.txt` | finished-segment table; Slurm job ids and resubmissions |
| `segments/segNN/` | `segment.txt` (command, start state, decision) and `summary.json` (finished segments) |
| `logs_and_json.tar.gz` | each segment's `run.log`, full `F64L.json` and `gpumem.txt` (segment 7 up to the snapshot) |
| `F64L_latest.npz` | newest checkpoint at the snapshot (seg07 `_latest`): `u` (28,750), `c_p`, `nu`, `rho`, `eps` |
