#!/bin/bash
# One B200 on Stellar AI for a bash script of several python runs (NEST env).
# Usage: sbatch run_sai_bash.sbatch SCRIPT.sh ARGS
#SBATCH --account=kolemen
#SBATCH --partition=gpu
#SBATCH --qos=gpu-short
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:nvidia_b200:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=300G
#SBATCH --time=04:00:00
#SBATCH --job-name=post
#SBATCH --output=logs/post-%j.log
set -uo pipefail
cd "$SLURM_SUBMIT_DIR"
export NEST_DEVICE=gpu JAX_ENABLE_X64=true XLA_PYTHON_CLIENT_PREALLOCATE=false PYTHONUNBUFFERED=1 MPLBACKEND=Agg OMP_NUM_THREADS=8
export XLA_PYTHON_CLIENT_ALLOCATOR=platform TF_CPP_MIN_LOG_LEVEL=3
bash "$@"
