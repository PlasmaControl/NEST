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
