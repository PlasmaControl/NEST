# Runbook: long fixed-boundary NEST runs on a GPU cluster (ripple case)

*Written for the agent operating the cluster (2026-11-16, branch `ripple-landscape`). Every command
is given exactly; nothing needs to be inferred. Background on the optimizer:
[`docs/optimizer_overview.md`](optimizer_overview.md). Scripts: `scripts/cluster/ripple_long/`.*

---

## 0. What these runs are for

NEST computes the current density **J** of a fixed-boundary MHD equilibrium without assuming nested
flux surfaces. The test case is a DESC tokamak equilibrium with a small resonant (2,1) boundary
ripple (R0 = 3.509 m, a = 1.198 m, |q| = 2 at ρ ≈ 0.63). DESC, which assumes nested surfaces,
*shields* the ripple at q = 2 with a current sheet. The question is whether NEST, pushed to the
lowest force residual it can reach, keeps that shielding or opens a q = 2 magnetic island.

The solver minimizes the change of current from DESC's (the misfit) subject to a bound on the
volume-mean force, M ≤ α·M_DESC. With α = 0.001 the bound cannot be met, so the run simply drives M
down as far as it can (the **floor**). Two runs:

| run | resolution (L/M/N), field grid | start | what it answers |
|---|---|---|---|
| **F64L** | 64/64/6, a/200, N_φ 25 | the best 64/64/6 state so far (M = 0.0161 M_DESC, M/ε = 16.14) | where the 64/64/6 floor is, and whether shielding survives there |
| **M30** | 30/30/8, a/100, N_φ 33 | DESC's current (J_fit); first an α sweep 0.1 → 0.05 → 0.02 → 0.01, then the α = 0.001 floor | the long-time behaviour at moderate resolution, quickly: does the shielding break, and when |

Both use the recommended solver flags (exact subspace search with previous-step and acceleration
directions, see `optimizer_overview.md` §7.3 and §9).

---

## 1. Quick start

```bash
git clone git@github.com:PlasmaControl/NEST.git nest-ripple && cd nest-ripple
git checkout ripple-landscape                     # commit at or after the one that added this runbook
cd scripts/cluster/ripple_long
cp site.sh.example site.sh && $EDITOR site.sh     # RUNS, SBATCH_OPTS, python env (section 2)
source site.sh && salloc $SBATCH_OPTS --gres=gpu:1 --time=01:00:00   # any GPU node, then on it:
bash preflight.sh                                 # must print PREFLIGHT PASS (about 10 min)
exit
bash submit.sh F64L                               # the chain resubmits itself until it finishes
bash submit.sh M30
bash status.sh                                    # live health digest, any time
tail -f $RUNS/F64L/current/run.log                # raw live log
bash collect.sh                                   # when both FINISHED (or any time): report + tarball
```

Send back the tarball `$RUNS/nest_ripple_long_<date>.tar.gz` (a few MB plus one state per run).

---

## 2. Setup

**Software.** Python with:
- `jax`/`jaxlib` with CUDA and float64 (the code enables `jax_enable_x64`); tested with jax 0.7.2;
- DESC (the equilibrium code; tested with DESC `0.17.3+96.g7e924cdfa`. A recent version should work;
  the preflight smoke run exercises every DESC call the solver makes);
- numpy, scipy; matplotlib is optional (report figures).

No installation of NEST itself: the scripts put `nest/` and `scripts/` on the path. If DESC is not
installed in the environment, add its source tree to `PYTHONPATH` in `site.sh`.

*B200 note:* some CUDA 12.x cuBLAS builds give wrong float64 results on B200. The preflight checks a
float64 Cholesky and matmul on the GPU; if that check fails, use an environment with a newer CUDA
(cuBLAS ≥ 13.2 is known good).

**`site.sh`** (copy from `site.sh.example`) sets:
- `RUNS`: output root on a large, fast file system not purged during the run (≈ 1 GB per run);
- `NEST_CACHE`: cache of the basis fits (default `$RUNS/cache`);
- `SBATCH_OPTS`: account, partition, QOS and `--time` (24 h recommended);
- the environment activation and `PY`.

**Resources per job:** 1 GPU, 8 CPU cores, 64 GB host RAM.

| run | unknowns n_u | collocation points | field-grid nodes | GPU memory peak | setup per segment |
|---|---|---|---|---|---|
| F64L | 28,750 | 337,552 | 10.96 M | ≈ 33 GB (measured on A100-80GB) | ≈ 15 min + 5 min metric |
| M30 | 8,958 | 166,991 | ≈ 3.6 M (estimate) | < 10 GB (estimate) | a few minutes |

Any GPU with ≥ 40 GB works for both; on a B200 they leave most of the card idle.

---

## 3. Preflight (`bash preflight.sh`, on a GPU node)

| check | PASS means | if FAIL |
|---|---|---|
| jax sees a GPU | a `gpu`/`cuda` device is listed | run on a GPU node; install a CUDA jaxlib |
| float64 Cholesky and matmul | relative errors < 1e-12 on a 4096² SPD matrix | newer CUDA/cuBLAS (B200 note above) |
| branch ripple-landscape | the checkout is on the right branch | `git checkout ripple-landscape` |
| input checksums | the DESC file and the F64L start state are byte-identical to ours | `git checkout -- <file>`; do not regenerate them |
| RUNS writable, ≥ 20 GB free | — | fix `RUNS` in `site.sh` |
| sbatch/squeue available | the chain can resubmit itself | see §5, "Other schedulers" |
| smoke segment 1, `[diag]` lines | an 8/8/4 α sweep runs, logs diagnostics and ends | read `$RUNS/_preflight/SMOKE/seg01/run.log` |
| signal → CONTROL STOP (segment 2) | a USR1 signal, as Slurm sends before the time limit, stops the floor solve cleanly with its state saved | read `seg02/run.log` |
| `_budget.npz`, `_latest.npz` | checkpoints are written | — |
| time budget (segment 3) | the `--time-budget` path stops cleanly with its state saved | read `seg03/run.log` |
| restart continuity | segment 3 starts exactly at segment 2's saved state (M agrees to 1e-8) | — |

Do not submit unless the last line is `PREFLIGHT PASS`.

---

## 4. The two runs in detail

Both run `scripts/meanforce/mf_fixed.py` with the DESC file
`desc_input/tokamak_boundary_perturbed/bp2/L32N4/tokamak_ripple_C_L32M32N4_x0.001.h5`,
`--bext desc-fit`, and the recommended flags

```
--subspace prev,accel --cg 1500 --no-rebuild-on-hit --f-noise auto --radius-by-delta --no-share-stop
--locate-chunk 131072 --diag-every 5 --save-every 1 --snap-outer
```

The exact command of every segment is in `$RUNS/<RUN>/segNN/segment.txt`.

**F64L** (one phase, `floor`): `--LMN 64 64 6 --N-phi 25 --frac 200 --alphas 0.001 --max-outer 1
--maxiter 100000`. The first segment starts from `results/ripple_landscape/valley/states/V_SA_budget.npz`.
Every segment starts from a copy of the newest checkpoint that keeps only the coefficients (`u`,
`c_p`). The penalty and multiplier are thereby reset, so the merit means the same thing in every
segment. (At an infeasible bound, outer updates would only raise the penalty tenfold; one long inner
solve per segment is used instead.)

**M30** (two phases):
- *Phase A, `sweep`:* `--LMN 30 30 8 --N-phi 33 --frac 100 --alphas 0.1 0.05 0.02 0.01 --stall-stop 0.01`
  with the default augmented-Lagrangian settings (`--max-outer 25 --maxiter 40`), from J_fit. Each
  α starts from the previous α's end state. The sweep ends at the first α that cannot be met
  (`--stall-stop` ends a stage that stops improving) or after 0.01. If a segment ends mid-sweep,
  the next one continues with the remaining α's from the newest checkpoint.
- *Phase B, `floor`:* as F64L, at 30/30/8, from the newest phase-A state.

**When a run finishes** (`$RUNS/<RUN>/FINISHED` holds the reason):
- *stalled:* two consecutive floor segments each lowered M/ε by less than 0.05% per hour of solve
  time (`STALL_RATE=5e-4`);
- *capped:* `MAX_SEG` segments (default 10, i.e. at most 10 × 24 h = 240 GPU-hours per run; raise it
  with `export MAX_SEG=20` in `site.sh` if a run is still descending clearly);
- *error:* the solver exited with an error (see §7).

---

## 5. Checkpointing and restarts

**Files in each segment directory `$RUNS/<RUN>/segNN/`:**

| file | written | contents |
|---|---|---|
| `run.log` | live, unbuffered, one line per event | the solver log (§6) |
| `<RUN>.json` | after every inner step (atomic) | every step's numbers, the `[diag]` records, setup timings, memory |
| `<RUN>_a<α>_latest.npz` | after every accepted step (`--save-every 1`) and at each outer end | checkpoint: `u`, `c_p`, `nu`, `rho`, `eps` |
| `<RUN>_a<α>_o<k>.npz` | at the end of outer iteration k | snapshot |
| `<RUN>_a<α>.npz` | when an α stage ends | that α's end state |
| `<RUN>_budget.npz` | when the time budget or a STOP ends the segment | the last accepted state |
| `<RUN>_bext.npz` | at setup | the fixed external field (large; can be deleted) |
| `segment.txt`, `summary.json`, `PHASE`, `gpumem.txt`, `start.npz` | — | command, summary, phase, GPU memory every 60 s, the start state used |

All state and JSON writes are atomic (write a temporary file, then rename), so a kill at any moment
leaves the previous checkpoint intact. A checkpoint is a few hundred kB; checkpointing every accepted
step costs nothing measurable. At most one inner step is lost (at 64/64/6 up to ≈ 15 min on an A100).

**How a segment ends cleanly.** Three mechanisms, in order:
1. `--time-budget`: `segment.sh` sets it to the job's remaining wall time minus `SAFETY` (45 min).
   The solver checks it after every inner step, saves `_budget.npz` and exits. The budget counts
   from program start, setup included; the step in progress finishes first (it can overshoot by one
   step).
2. Signal: `run.sbatch` asks Slurm for `USR1` 30 min before the limit (`--signal=B:USR1@1800`) and
   forwards it. `segment.sh` then writes `STOP` into `segNN/CONTROL`, which the solver reads after
   the next step.
3. Manual: `echo STOP > $RUNS/<RUN>/current/CONTROL` (stop now, state saved; the chain then
   resubmits). `echo NEXT > .../CONTROL` ends the current α stage (phase A only).

**Restart.** Each job runs `segment.sh`, which picks the newest valid checkpoint over all earlier
segments (`chainlib.py newest-state`) and starts a fresh process from it. The trust region, metric
and noise estimate are rebuilt (≈ 20 min at 64/64/6), so the first steps after a restart are short.
This is expected.

**Stopping a chain:** `touch $RUNS/<RUN>/NO_RESUBMIT` (the running job finishes its segment, no new
job is submitted), or `scancel` the job (the segment loses at most one step; resubmit later with
`bash submit.sh <RUN>`; it continues where it stopped).

**Other schedulers.** `segment.sh` does not depend on Slurm: run `bash segment.sh <RUN>` repeatedly
(e.g. from a PBS/LSF job, or a loop on an interactive node) with `RUNS`, `PY` and `SEG_SECONDS`
(the wall seconds available) set. Exit code 0 means "run another segment", 3 means "finished",
anything else is an error.

---

## 6. Live logging: reading the optimizer's health

`tail -f $RUNS/<RUN>/current/run.log` shows the log as it is written. `bash status.sh` prints a
digest for both runs:
- the chain table;
- the age of the last log line;
- the M/ε trend and rate per hour (whole segment and second half);
- boundary hits and poor-ratio counts;
- the last `[diag]`, `[metric]`, `[subspace]` and `[noise]` lines;
- any error lines;
- the last step lines.

**Line glossary** (examples from a 64/64/6 run):

```
[setup] Fields: 24.0 s; mem {'in use GB': 8.3, 'peak GB': 17.7, ...}     setup phases, GPU memory
[diag] start: M/M_desc 0.01614 |J-J_fit|/rms 1.2e-01 b21/vac 0.0208 phase -90.0 deg | band p16/DESC 0.18 ...
  [metric] refreshed at outer 0; Delta None -> 1.496e+00                   preconditioner (re)built
    [noise] merit at 6 points 1e-10 away: ... sigma_f 1.4e-06 (1.2e-08 of f)  merit noise, once per inner solve
      [beta] lanczos 111 (TR stop; theta [1.9e-03, 2.9e+07], hit True) t 66.3s   one trust-region subproblem
      [subspace] dirs d+accel, c* = (+3.175, +0.001), ... rel 1.6e-07; 5 evals, 24.3s   exact search on a boundary step
    it  12 f 1.2175e+02 -> 1.2174e+02 ratio +1.17 |y| 3.6e-05 Delta 1.2e+03 |g| 2.9e+01 GN 0.000 hit M/eps 16.6048 ... 1.5s
```

The `it` line is one inner step:
- `f` is the merit before → after (≈ ½(M/ε)² near the floor);
- `ratio` is actual/predicted decrease;
- `|y|` is the step length and `Delta` the trust radius (both in the preconditioner's metric);
- `|g|` is the gradient norm, and `hit` marks a step that reached the trust-region boundary;
- `M/ε` is the force over the bound (M/M_DESC = 0.001 × M/ε);
- the last number is the step's wall time.

**The `[diag]` line** is written every 5 accepted steps, at each outer end, at the start and at the
end. It is cheap: one extra Biot–Savart solve plus reductions over the points. Fields:

| field | meaning | how to read it |
|---|---|---|
| `M/M_desc` | volume-mean force / DESC's | the floor measure; lower is better |
| `\|J-J_fit\|/rms` | relative rms change of the current from DESC's (fitted) | how far the solution has moved |
| `b21/vac`, `phase` | resonant (2,1) radial-field harmonic at q = 2, of the change from J_fit, as a fraction of the vacuum (unshielded) value 3.78e-4 | **the shielding diagnostic.** Phase is ±90° by symmetry. −90° = anti-island side (over-shielded, as in every state so far: 2.1% at 64/64/6). **+90° and growth = shielding breaking.** Earlier states with real islands had b21 ≈ 6–12% of vacuum |
| `band p16/DESC`, `max/DESC` | p16-norm and max of \|F\| in the q = 2 band ρ ∈ [0.5675, 0.7075], / DESC's on the same points | whether residual force concentrates at the resonant surface (it rose 0.159 → 0.180 at 64/64/6 as the floor was approached) |
| `outside mean/DESC` | mean \|F\| outside the band / DESC's | the rest of the volume |

**Healthy vs unhealthy:**

| signal | healthy | warning sign → action |
|---|---|---|
| log age (`status.sh`) | new lines at least every ~15 min (64/64/6) | > 60 min with the job running: check `gpumem.txt` (is the GPU busy?) and `squeue`; if the process is gone, see §7 |
| M/ε trend | decreases in steps; plateaus of tens of minutes, then drops, are normal in curved valleys | rises over several accepted steps by more than the noise (≈ 1e-8 relative): report |
| `ratio` | mostly 0.5–2; occasional large values on noise-level steps | many consecutive ratios < 0.1 with Δ shrinking every step: report (radius collapse) |
| `Delta` | grows on `hit` steps, can reach 10²–10⁴ | Δ falling below ≈ 1e-8 and staying there: the inner solve will soon end (stationary or radius test), and the next segment restarts it with a fresh metric |
| `[beta] lanczos` | interior steps 10–200, boundary steps up to the 1500 cap | every step at the cap and `[metric] rebuilt` every step: report |
| `[subspace] rel` | < 1e-4 | larger is only a logged warning; the step is judged by its real evaluation |
| `[noise] sigma_f` | ≈ 1e-8 of f | > 1e-6 of f: report (precision problem) |
| `stall detector fired` | rare | repeatedly: report |
| GPU memory (`gpumem.txt`) | flat after setup | steady growth: report |
| `[diag] b21` | slowly varying, phase −90° | phase flips to +90°, or b21/vac rising past ≈ 0.05: **scientifically important; note the segment and step** |

---

## 7. Troubleshooting

| symptom | likely cause | action |
|---|---|---|
| `RESOURCE_EXHAUSTED` / out of memory in setup | too small a GPU, or another process on it | request a full ≥ 40 GB GPU; check `nvidia-smi` |
| `ModuleNotFoundError: desc` | DESC not on the path | install DESC or add it to `PYTHONPATH` in `site.sh` |
| preflight float64 check fails | CUDA/cuBLAS issue on B200 | use a newer CUDA environment |
| job ends after a few minutes, `FINISHED` says `error` | crash: read the end of `segNN/run.log` | fix and resubmit: `rm $RUNS/<RUN>/FINISHED; bash submit.sh <RUN>` (it restarts from the newest checkpoint) |
| no resubmission after a job | `sbatch` unavailable on compute nodes, or `NO_RESUBMIT`/`FINISHED` present | `cat $RUNS/<RUN>/jobs.txt`; resubmit by hand with `bash submit.sh <RUN>` |
| the job was killed (time limit, node failure) without `time budget reached` in the log | lost at most the step in progress | resubmit; the newest checkpoint is used |
| a segment ends with `stationary` after few steps | the inner stopping test fired | normal near a stall; the next segment restarts with a fresh metric; two slow segments end the chain |
| M30 phase A takes many outer iterations at one α | the augmented Lagrangian tightening toward the bound | normal; `--stall-stop` ends an α that cannot be met |
| want to inspect a state | — | `np.load('<file>.npz')['u']` holds the coefficients; evaluate on a finer grid with `mf_fixed.py ... --eval-only --force-norm p --b21-report --eval-state name=<file> --frac <finer>` (same `--LMN`) |

---

## 8. What to send back

`bash collect.sh` writes `$RUNS/report/report.md` with:
- the chain table per run;
- M/M_DESC and the diagnostics from first to last;
- CSVs of every accepted step (`steps_<RUN>.csv`) and every `[diag]` record (`diag_<RUN>.csv`);
- a figure per run.

It also packs everything needed to continue or re-analyse into `$RUNS/nest_ripple_long_<date>.tar.gz`:
logs, JSONs, summaries and the newest state of each run. Send that file; it is safe to send
intermediate tarballs while the runs continue.

---

## 9. Reproduction table

| item | value |
|---|---|
| code | branch `ripple-landscape`, scripts `scripts/cluster/ripple_long/` |
| DESC file | `tokamak_ripple_C_L32M32N4_x0.001.h5`, sha256 `921fac08…51fa69`, M_DESC = 4.9995e-7 |
| F64L start | `results/ripple_landscape/valley/states/V_SA_budget.npz`, sha256 `136ddab4…3d161e`, M/ε = 16.1438 at α = 0.001 (a/200) |
| solver flags | §4 |
| earlier context | 64/64/6 floor history (M/ε 25.8 → 16.14), valley comparison and the b21 diagnostic: `docs/optimizer_overview.md` §4.4, §7.4 |
