#!/bin/bash
# One B200 (180 GB) on Stellar AI for the cylindrical-annulus runs (gpu partition; gpu-short
# QOS: 4 h, gpu-medium: 24 h). S_m and the plasma location cached on disk between jobs.
# Usage: sbatch [--time=..] [--qos=gpu-medium] run_sai_cyl.sbatch SCRIPT.py ARGS
#SBATCH --account=kolemen
#SBATCH --partition=gpu
#SBATCH --qos=gpu-short
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:nvidia_b200:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=200G
#SBATCH --time=04:00:00
#SBATCH --job-name=nestcyl
#SBATCH --output=logs/nestcyl-%j.log
set -euo pipefail
cd "$SLURM_SUBMIT_DIR"
ENV=/scratch/gpfs/EKOLEMEN/lk1158/envs/nest13  # CUDA 13: cuBLAS >= 13.2 (12.9 has a B200 TMEM bug)
export NEST_DEVICE=gpu JAX_ENABLE_X64=true XLA_PYTHON_CLIENT_PREALLOCATE=false
export JAX_COMPILATION_CACHE_DIR=${JAX_COMPILATION_CACHE_DIR:-$SLURM_SUBMIT_DIR/.cache/jax}
export JAX_PERSISTENT_CACHE_MIN_COMPILE_TIME_SECS=0 JAX_PERSISTENT_CACHE_MIN_ENTRY_SIZE_BYTES=-1
export XLA_PYTHON_CLIENT_ALLOCATOR=${XLA_PYTHON_CLIENT_ALLOCATOR:-platform}
export XLA_PYTHON_CLIENT_MEM_FRACTION=${XLA_PYTHON_CLIENT_MEM_FRACTION:-0.90}
export NEST_S_CACHE=${NEST_S_CACHE:-$SLURM_SUBMIT_DIR/.cache/S}
export NEST_EIG_THREADS=${NEST_EIG_THREADS:-8}
# threadpoolctl (in .pylib): one BLAS thread per eigendecomposition thread (else OpenBLAS segfaults)
export PYTHONPATH=$SLURM_SUBMIT_DIR/.pylib${PYTHONPATH:+:$PYTHONPATH}
export PYTHONUNBUFFERED=1 MPLBACKEND=Agg OMP_NUM_THREADS=8
nvidia-smi --query-gpu=name,memory.total --format=csv
$ENV/bin/python -c "from desc import set_device; set_device('gpu'); import jax; d = jax.devices(); print('JAX devices:', d); assert d[0].platform == 'gpu'"
nvidia-smi --query-gpu=timestamp,memory.used --format=csv,noheader -l 10 > logs/gpu-$SLURM_JOB_ID.txt &
SMI=$!
/usr/bin/time -v -o logs/time-$SLURM_JOB_ID.txt $ENV/bin/python "$@"
kill $SMI 2>/dev/null || true
sort -t, -k2 -n logs/gpu-$SLURM_JOB_ID.txt | tail -1 | sed 's/^/GPU memory peak (nvidia-smi): /'
grep -E "Maximum resident|Elapsed" logs/time-$SLURM_JOB_ID.txt
