# Modal GPU jobs for the cylindrical J -> B solver (stage D)

Image `im-85tKPZB30SmMyYst74flDg` (jax 0.7.2 CUDA; see `scripts/modal/README.md` on branch
`container-mask` for its build and the nvgpu fallback in `nest_gpu_entry.py`). Inputs per job
(staged flat into `/work`): these scripts, `commit.txt`, and three tarballs:

- `nest_src.tar.gz`: `git archive HEAD nest scripts tests` plus
  `desc_input/precise_QA_free_boundary_smooth_J/precise_QA_free_boundary_smooth_J_edgeJ_r0.3.h5` and
  `desc_input/precise_QA_free_boundary/precise_QA_helical_coils.h5`
- `desc_src.tar.gz`: the DESC tree's `desc/` (branch `js/proximal-auglag`)
- `deps_src.tar.gz`: `nufftax adv_jax_math interpax_fft optax absl` from the local env's site-packages

Results go to the Volume `nest-container-mask` under `/results/cyl/<job>/` as they are produced
(logs, per-size JSONs, `optimize_qa` iterates), and to `out/` at the end.
`common_cyl.sh` puts `nest/` and `tests/` on PYTHONPATH (`runpy` does not add the script's directory).

- `run_d1_mem.sh`: GPU memory model (`tests/cyl_memory.py`, compile only) and measured ladders
  (`tests/cyl_measure.py`, one process per size, device `peak_bytes_in_use`), cylindrical a/20-a/70.
  Default BFC allocator (the platform allocator reports no peak). A100-80GB.
- `run_d1_box.sh`: the same for main's VG box, a/20-a/40. A100-80GB.
- `run_d1_err.sh`: `tests/cyl_b4.py cyl` at a/30 and a/40 (exact J) at the B4 targets
  (`b4_targets.npz` = `results/cyl/b4/targets.npz`). A100-40GB.
- `run_d2.sh TAG <optimize_qa args>`: production `optimize_qa`, e.g.
  `--grid cylindrical --frac 20 --factors 3 3 2.5 --colloc 2 --check 3 --check-frac 20 --check-grid cartesian
  --maxiter 100 --cg 100 --time-limit 5400`. A100-40GB.

Note: S_m is built on the host CPU; on Modal it is ~4x slower than on an M-series laptop
(a/30: ~8 min), and grows ~h^-2.7.

## S_m cache and references (CPU containers)

`FreeSpaceAnnulus(cache=...)` / `NEST_S_CACHE`: S_m stored once in float64 under `S_<key>/`
(`S.npy`, `S_o.npy`, `meta.json`; key = axes, |m| set, z-fold, quadrature rule, `single_layer.S_NUMERICS`) and loaded afterwards. GPU jobs should set `NEST_S_CACHE=/results/cyl/S_cache` and
find their S_m built beforehand, so the GPU never waits on the host-CPU build. The references
built on 2026-10-17 are stored under keys of the earlier source-hash scheme; under the current
S_NUMERICS = 1 keys they are NOT found (a run rebuilds) until they are re-keyed on the Volume
(rename `S_<old>` to `S_<new>`, same entries; not yet done). Compare against them by path. See also
`docs/cylindrical_grid.md`, "Before a high-resolution run".

- `run_s_ref.sh FRAC JOBS`: `tests/cyl_s_build.py build` on a CPU container (`cpu` 32; memory
  ~2x the float64 rows: 16 GB to a/30, 24 GB a/40, 32 GB a/50, 48 GB a/60) for the production
  precise_QA grid of a/FRAC (`s_grids.json` from `tests/cyl_s_build.py grids`, local with DESC).
  It checks the process pool bitwise against the serial build, builds into
  `/results/cyl/S_cache`, and spot-checks the quadrature (32 points / ratio 0.08 against the
  default 16 / 0.15); summary in `/results/cyl/s_ref/s_ref_a<FRAC>.json`.
  a/20 on 32 processes: S 5 s (0.66 GB float64); pool bitwise; quadrature <= 4.5e-12 to |m| = 495.
