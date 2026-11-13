# Memory/timing probe and Biot-Savart grid check (2026-11-13)

Job script `scripts/modal_fixed/ripple/job_mem.sh` (P64: Modal job c6e574e5, A100-80GB, 2036 s; P80: job 01b5a67e, A100-80GB, 835 s, OOM).
New `mf_fixed.py` flags: `--locate-chunk` (locate transient = chunk x DESC modes x 8 B; 131072 -> 5.3 GB),
`--eval-dump` (J, B, F at the collocation points per evaluated state), `--mem-every` (not effective: the solver's step
log does not go through the wrapped logger).

- `memprobe_measured.csv`: per-phase JAX peaks of P64 (64/64/6 a/160) and P80 (80/80/6 a/200).
- `nest_memory_measured.csv`: per-phase peaks of 54 earlier runs (ripple + NCSX fixed-boundary).
- `nest_memory_projection.csv`: projected peaks, L = M, field grid a/(2.5 L), N_phi 4N+1, default collocation.
- `P64_grid_check.csv`, `P64_grid_check_by_rho.csv`: P64 start/end states evaluated at a/160, a/200, a/240
  (same collocation points). Dumps (80 MB each) are on Volume nest-container-mask at /results/fxr/mem/P64/.
- `states/P64_a0.05.npz`: 64/64/6 end state (load with `--LMN 64 64 6 --frac 160 --N-phi 25`).
- `runs/`: logs, JSON and nvidia-smi traces.
