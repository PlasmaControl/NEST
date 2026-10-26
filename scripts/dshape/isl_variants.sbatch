#!/bin/bash
# Solver-variant benchmark at N = 2 on 4 B200s: each variant ("name:speed options") runs 30 iterations from the
# same snapshot state, one after the other.
# Usage (repo root): sbatch scripts/dshape/isl_variants.sbatch EPS START NAME:SPEED [NAME:SPEED ...]
#SBATCH --account=kolemen
#SBATCH --partition=gpu
#SBATCH --qos=gpu-short
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:nvidia_b200:4
#SBATCH --cpus-per-task=16
#SBATCH --mem=500G
#SBATCH --time=04:00:00
#SBATCH --job-name=variants
#SBATCH --output=logs/nest-%j.log
set -uo pipefail
cd "$SLURM_SUBMIT_DIR"
EPS=$1; START=$2; shift 2
ENV=/scratch/gpfs/EKOLEMEN/lk1158/envs/nest13
export NEST_DEVICE=gpu JAX_ENABLE_X64=true XLA_PYTHON_CLIENT_PREALLOCATE=false
export JAX_COMPILATION_CACHE_DIR=${JAX_COMPILATION_CACHE_DIR:-$SLURM_SUBMIT_DIR/.cache/jax}
export JAX_PERSISTENT_CACHE_MIN_COMPILE_TIME_SECS=0 JAX_PERSISTENT_CACHE_MIN_ENTRY_SIZE_BYTES=-1
export XLA_PYTHON_CLIENT_ALLOCATOR=${XLA_PYTHON_CLIENT_ALLOCATOR:-platform}
export XLA_PYTHON_CLIENT_MEM_FRACTION=${XLA_PYTHON_CLIENT_MEM_FRACTION:-0.90}
export PYTHONUNBUFFERED=1 MPLBACKEND=Agg OMP_NUM_THREADS=8
for v in "$@"; do
  name=${v%%:*}; sp=${v#*:}
  echo "=== variant $name: $sp"
  $ENV/bin/python scripts/dshape/isl_scan.py --base results/gs/hN_a120D --N 2 --eps $EPS --j-mode full \
      --w-J 1e-7 --w-W 0 --w-pos 0 --steps qr --gn-tol 0 --rtol 1e-12 --iters 30 --batch 16 8 \
      --start-from $START --speed resident,chol,narrow,merge=100000,gpus=4,profile,$sp --name bn_$name
done
