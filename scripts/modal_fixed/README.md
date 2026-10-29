# Modal job scripts for the fixed-boundary runs

Used on image `im-85tKPZB30SmMyYst74flDg` (jax 0.7.2 CUDA, DESC as a source tarball) with the Volume
`nest-container-mask` mounted at `/results`. Inputs are staged in `/work`: `nest_src.tar.gz`
(`tar --exclude=__pycache__ -czf nest_src.tar.gz nest scripts tests desc_input`), `desc_src.tar.gz`
(the DESC tree), `deps_src.tar.gz` (nufftax adv_jax_math interpax_fft optax absl) and `gpu_entry.py`
(the nvgpu fallback for driver-610 hosts).

- `fx_run.sh <tag> <mf_fixed args>`: `scripts/meanforce/mf_fixed.py`, log and per-step JSON/states
  mirrored to `/results/fx/<tag>`.
- `py_run.sh <tag> <script> <args>`: any diagnostic script; output `out/<tag>.npz`, mirrored.
- `test_run.sh <test.py>`: a CPU-JAX test script (works on a GPU container too).

Sizing (48/32/18): H200 for solves (a/40: ~20 GB in use, 53-56 GB peak at the force-metric build;
a/60: 66 GB device footprint), A100-80GB for diagnostics.
