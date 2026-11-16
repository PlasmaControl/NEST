# Long ripple runs on a GPU cluster

Runbook: [`docs/ripple_long_runs_runbook.md`](../../../docs/ripple_long_runs_runbook.md).

| file | purpose |
|---|---|
| `site.sh.example` | site settings (copy to `site.sh`): RUNS, SBATCH_OPTS, environment |
| `preflight.sh` | environment, inputs, float64 GPU check, 3-segment smoke chain; PASS/FAIL |
| `submit.sh RUN` | submit the self-resubmitting chain of RUN (F64L or M30) |
| `run.sbatch` | Slurm wrapper: one segment per job, signal forwarding, resubmission |
| `segment.sh RUN` | one scheduler-independent segment: pick checkpoint, run, summarize, decide |
| `status.sh [RUN..]` | live health digest |
| `collect.sh [RUN..]` | report + tarball to send back |
| `chainlib.py` | checkpoint selection, summaries, stall decision, status, report |
