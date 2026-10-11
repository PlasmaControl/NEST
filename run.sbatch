#!/bin/bash
# Single-A100 request on the stellar-debug QOS, same setup as NEMO/desc_input.
# Usage: sbatch [--time=..] run.sbatch SCRIPT.py ARGS
# NEST_DEVICE=gpu makes the NEST scripts call desc.set_device("gpu"); without
# it DESC hides the GPU (CUDA_VISIBLE_DEVICES="") and JAX runs on the CPU.
#SBATCH --account=mae
#SBATCH --partition=gpu
#SBATCH --qos=stellar-debug
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:a100:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=00:59:00
#SBATCH --job-name=nest
#SBATCH --output=logs/nest-%j.log
set -euo pipefail
cd "$SLURM_SUBMIT_DIR"
ENV=/scratch/gpfs/lk1158/.conda/envs/desc-env
export LD_PRELOAD=$ENV/lib/libstdc++.so.6
export NEST_DEVICE=gpu JAX_ENABLE_X64=true XLA_PYTHON_CLIENT_PREALLOCATE=false
# JAX may use up to 90% of the GPU (default 75%); the rest for cuFFT plans and the like
# compiled programs reused across jobs and tests (results unchanged; the first run fills it)
export JAX_COMPILATION_CACHE_DIR=${JAX_COMPILATION_CACHE_DIR:-$SLURM_SUBMIT_DIR/.cache/jax}
export JAX_PERSISTENT_CACHE_MIN_COMPILE_TIME_SECS=0 JAX_PERSISTENT_CACHE_MIN_ENTRY_SIZE_BYTES=-1
# the platform allocator (no pool): ~7% slower per iteration than the default pool,
# but long runs and large boxes cannot fail on a fragmented pool; override with
# --export=ALL,XLA_PYTHON_CLIENT_ALLOCATOR=default (e.g. for memory_stats)
export XLA_PYTHON_CLIENT_ALLOCATOR=${XLA_PYTHON_CLIENT_ALLOCATOR:-platform}
export XLA_PYTHON_CLIENT_MEM_FRACTION=${XLA_PYTHON_CLIENT_MEM_FRACTION:-0.90}  # override: sbatch --export=ALL,XLA_PYTHON_CLIENT_MEM_FRACTION=...
export PYTHONUNBUFFERED=1 MPLBACKEND=Agg OMP_NUM_THREADS=4
nvidia-smi --query-gpu=name,memory.total --format=csv
# stop if DESC/JAX do not get the GPU instead of running on the CPU
$ENV/bin/python -c "from desc import set_device; set_device('gpu'); import jax; d = jax.devices(); print('JAX devices:', d); assert d[0].platform == 'gpu'"
$ENV/bin/python "$@"
