# Runbook: `speed elim` production A/B on Stellar AI (for the operator agent)

You are running a fixed, pre-planned set of GPU jobs on Princeton Stellar AI (B200) for the NEST code, and then
producing one report and one tarball to send back. Everything you need is scripted. **Do not modify code, flags or
inputs; do not diagnose physics; do not push to git.** When something goes wrong, use section 6, and record
what you did in the operator notes (section 7). If section 6 does not cover it, stop and send back what you have.

Total effort: ~15 min of your time, ~1 h of wall time (queue permitting), ~4 B200 GPU-hours.

---

## 0. What is being tested (enough to check the logs, no more)

NEST solves for a plasma current `u` and a pressure `p` by a Levenberg-Marquardt / trust-region least-squares
loop (`nest/harmonic_dense.py`, `HarmonicModel`). The new option `--speed ...,elim` removes `p` from the
unknowns: for every trial current it solves for the best `p` exactly (a weak Poisson equation of force balance),
so the solver only steps in `u`. Same answer, hopefully faster convergence. The runs compare:

| label | run name | what | expected time per iteration |
|---|---|---|---|
| gate | (tests) | `tests/test_pressure_elim.py` (8 exactness checks, small model) on 1 GPU | whole job ~5-15 min |
| abB | abN2B1000 | reference: today's solver (`phys,tr`), p free | ~12-15 s (measured in your earlier N = 2 runs) |
| abD | abN2D1000 | `elim,phys,tr`: p eliminated | ~13-17 s |
| abE | abN2E1000 | abD with a richer pressure space (`--p-factors 3 3 1`) | ~25-35 s |
| cond | (diagnostic) | condition numbers with p free vs eliminated, at the start state, 1 GPU | whole job ~15-30 min |
| cond_p3 | (diagnostic) | the same with `--p-factors 3 3 1` (larger: ~23k unknowns) | whole job ~20-45 min |

All A/B runs: N = 2, 1000 A, w_J 1e-7, 30 iterations, starting from the state `a120Cn2d1000` (iteration 225 of
your earlier run; the copy tracked in git is used). 16,457 unknowns (abE ~23k). Each A/B job uses 4 B200.

## 1. Rules

- Use only the scripts named here, unmodified: `scripts/lm/elim_preflight.sh`, `scripts/lm/elim_submit.sh`,
  `scripts/lm/elim_collect.sh` (they call `run_sai.sbatch`, `scripts/dshape/isl_scan.py`,
  `scripts/diagnostics/pressure_elim_conditioning_isl.py`, `scripts/lm/elim_report.py`).
- Work in a **separate git worktree** of branch `pressure-poisson` (section 2), so your usual checkout and its
  uncommitted work are not touched.
- Never delete or move anything in the NEST cache directory (`NEST_CACHE`). New files appearing there is normal.
- Do not run anything beyond the jobs in section 4 (and the section 6 fallbacks).

## 2. Setup (login node)

`ISL` below is your usual NEST checkout that ran `a120Cn2d1000` (its `.cache/` holds the N = 2 tables
`HarmTab_u0_*.npy`, `HarmTab_u1_*.npy`, `HarmTab_u2_*.npy`; ~20 GB). From the earlier run files it is probably
`/scratch/gpfs/EKOLEMEN/lk1158/NEST_isl`; confirm with `ls $ISL/.cache/HarmTab_u2_*.npy`. If your runs used
`NEST_CACHE=...` elsewhere, use that directory instead of `$ISL/.cache` below.

```bash
export ISL=/scratch/gpfs/EKOLEMEN/lk1158/NEST_isl          # adjust if different (see above)
cd $ISL
git fetch origin
git worktree add -b pressure-poisson ../NEST-pois origin/pressure-poisson \
  || { git worktree add ../NEST-pois pressure-poisson && git -C ../NEST-pois merge --ff-only origin/pressure-poisson; }
cd ../NEST-pois
export NEST_CACHE=$ISL/.cache                              # the cache with the N = 2 tables (read and appended to)
export JAX_COMPILATION_CACHE_DIR=$ISL/.cache/jax           # optional: reuses your compiled kernels
git log --oneline -1                                       # record this hash in the notes
```

Keep `NEST_CACHE` (and `JAX_COMPILATION_CACHE_DIR`) exported in the shell you submit from: `sbatch` passes the
environment to the jobs.

Why the cache matters: the tables are keyed by the content of the equilibrium file and of 7 source modules; those
modules are byte-identical to the code that built your tables, so they load in ~1 min. Without the cache, each
A/B job would spend ~1.5 h rebuilding them.

## 3. Preflight (login node, ~1 min)

```bash
bash scripts/lm/elim_preflight.sh
```

It must end with `PREFLIGHT PASSED`. WARN lines are fine. Fixes for FAIL lines:

| FAIL line | fix |
|---|---|
| `branch is ...` / `no _elim_step` | you are not in the `NEST-pois` worktree, or the fetch did not get the branch: redo section 2 |
| `tracked files modified` | `git -C ../NEST-pois status`; you must not edit files: `git checkout -- <file>` |
| `missing input ...` | the worktree is incomplete: `git -C ../NEST-pois status`, redo section 2 |
| `NEST_CACHE is not set` / `has no HarmTab_u0/u1/u2` | wrong cache directory: find the one with `HarmTab_u2_*.npy` (`ls <dir>/HarmTab_u2_*`), export `NEST_CACHE=<dir>` |
| `NEST_CACHE not writable` | use a cache you own, or ask before copying (~20 GB) |
| `no python at ...` | `export ENV=<your nest13 env>` (the one in `run_sai.sbatch`) and rerun |
| `sbatch not found` | run on a Stellar login node |

## 4. Submit (login node, ~1 min)

```bash
bash scripts/lm/elim_submit.sh
```

This submits 6 jobs and writes their ids to `results/isl/elim_jobs.txt`:
`gate` first, then `abB abD abE cond cond_p3`, each with `--dependency=afterok:<gate>`: they start only if the
gate tests pass. (Check the exact commands without submitting: `DRY=1 bash scripts/lm/elim_submit.sh`.)

## 5. Monitor (every ~10 min until all jobs end)

```bash
squeue -u $USER -o "%.10i %.10j %.8T %.10M %.20R"
cat results/isl/elim_jobs.txt
tail -n 5 logs/nest-<id>.log
```

What healthy logs look like:

- **gate**: ends with `all passed: T1 T2 T3 T4 T5 T6 T7 T8` (each `T<k>` line prints its measured number).
- **every A/B job, first minutes**: `JAX devices: [CudaDevice(id=0), ..., CudaDevice(id=3)]`, then
  `table u0: (97, 193, 6, 1, 2242) from the cache`, `table u1: ... from the cache`, `table u2: ... from the cache`
  (and p0, p1, p2). abE additionally builds new pressure tables (`table p0: (...), <n> s`; minutes, normal).
  **A line `... min estimated for ... box solves` means the cache was missed**: see section 6.
- **abD / abE start**: `speed elim, joint f before the start projection ...`, `elim: R_pp (...) built in ... s`,
  `elim: start projection p -> p*(u), f* ..., |dp|/|p| ...`.
- **iteration lines** (abD/abE format; abB has no `|c_p|` fields):

```
  it  2 QR   f 1.105147e-10  <|F|>/norm 1.2993e-05  |dx|/|x| 5.83e-02  rel 2.1e-02  mu/lam 9e-07  GN share 3.0e-01  ratio 0.99 tries 1  |c_p|/|r| 5e-11 f* chk 3e-11  p min/max -6.1e-06  6 s
```

  Healthy: `f` decreases every iteration; `|c_p|/|r|` <= 1e-8; `f* chk` <= 1e-8; `p min/max` small negative
  (around -1e-3 or closer to 0) or positive; the trailing seconds near the table in section 0.
- **cond / cond_p3**: lines `kappa joint_jacobi: ...`, `kappa reduced_jacobi: ...`, ... (5 of them), then `-> results/isl/cond_...json`.

You do not need to act on the numbers; the report flags them automatically.

## 6. Troubleshooting (do exactly this, then note it in section 7)

| symptom | action |
|---|---|
| jobs `PENDING` with reason `(DependencyNeverSatisfied)` | the gate failed: read the end of the gate log. (a) An `AssertionError` in a `test_T<k>` whose printed value is within 100x of its bound (e.g. `T5 ... rel 3e-6` against 1e-6): a GPU rounding difference. Cancel the pending jobs (`scancel <ids>`), then `SKIP_GATE=1 bash scripts/lm/elim_submit.sh`. (b) Anything else (ImportError, a traceback outside the tests, a value off by more than 100x): stop, run section 7, send. |
| a `... min estimated for ... box solves` line in an A/B or cond log (cache miss) | `scancel` all jobs in `elim_jobs.txt`; find the right `NEST_CACHE` (section 2), rerun the preflight, then `SKIP_GATE=1 bash scripts/lm/elim_submit.sh` if the gate already passed, else submit normally. |
| abD or abE fails at the start with an error mentioning devices / sharding / `committed` / `incompatible devices` (abB runs fine) | multi-GPU issue with elim (untested on 4 GPUs). Resubmit that job on 1 GPU: copy its line from `DRY=1 bash scripts/lm/elim_submit.sh`, replace `gpu:nvidia_b200:4` by `gpu:nvidia_b200:1`, `--mem=500G` by `--mem=300G`, remove `,gpus=4` from `--speed`, drop the `--dependency=...`; run it with `sbatch --parsable ...`, and replace the job's id in `results/isl/elim_jobs.txt` (`abD=<new id>`). Expect ~45 s/it (~25 min). |
| `out of memory` / `RESOURCE_EXHAUSTED` in abE | resubmit abE as above (4 GPUs) with `merge=50000` instead of `merge=100000` in `--speed`; if it fails again, with `--p-factors 2.5 2.5 1` instead of `3 3 1`. Update the id in `elim_jobs.txt`. |
| `out of memory` in cond_p3 | skip it (note it). |
| `AssertionError: speed elim: ...` | the flags were changed: you did not use `elim_submit.sh` unmodified. Resubmit with it. |
| `elim WARNING: A_p^T A_p changed ...` | nothing; it is in the report. |
| `it <k>: no decrease (QR); stopping` before iteration 30 | normal stop; nothing to do. |
| `DUE TO TIME LIMIT` | resubmit that job (as in the multi-GPU row, keeping its GPU count) with `--qos=gpu-medium --time=08:00:00`. |
| anything else | do not debug: run section 7 and send what exists, with the log. |

## 7. Collect, report, send (login node, after all jobs ended)

1. Write the operator notes (plain markdown; anything you did from section 6, the commit hash from section 2, queue
   delays, anything that looked odd), e.g.
   ```bash
   cat > results/isl/elim_operator_notes.md <<'NOTES'
   - code: <hash from git log --oneline -1>
   - all jobs ran as submitted; no fallbacks
   NOTES
   ```
2. Build the report and the return package:
   ```bash
   bash scripts/lm/elim_collect.sh
   ```
   It prints the gate result, the automatic FLAGS, and the tarball `results/isl/elim_return_<date>.tar.gz` with its
   file list (report, notes, job ids, the three runs' `*_eps1000.json/.npz`, the cond jsons, all 6 Slurm logs;
   a few MB).
3. Send back: the tarball, and paste `results/isl/elim_report.md` into the message.

If `elim_collect.sh` itself fails, send the files listed in its section 5 by hand (`results/isl/abN2?1000_eps1000.*`,
`results/isl/cond_a120Cn2d1000*.json`, `results/isl/elim_jobs.txt`, `logs/nest-<ids>.log`).

## 8. For the reader of the report (what the numbers mean)

- Section 1 (health): `max |c_p|/|r|` and `max f* chk` are exactness checks of the elimination (<= 1e-8 is
  healthy); `tables: cache u0,...` confirms no rebuild.
- Section 2 (convergence): f / f_ref per iteration, f_ref = the joint f at the common start (the last f of
  `a120Cn2d1000`; abB's iteration 0 must equal it, else a flag). The key comparison is abD vs abB at equal
  iterations and the wall time to reach abB's final f. On laptop models (L16-L32, up to 2,646 unknowns) abD reached
  f/f0 0.29-0.42 vs 0.46-0.53 for the reference after 30 iterations (docs/pressure_poisson_handoff.md).
- Section 3 (conditioning): on laptop models the eliminated (reduced) phys kappa was 17-80x below the joint one
  (the gap grew with resolution: 26x, 52x, 80x at L16, L24, L32 for w_J 1e-3; 17x, 34x, 54x for w_J 1e-7).
- Section 4 (flags): anything outside the expected ranges, or missing.
