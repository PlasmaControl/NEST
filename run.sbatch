#!/bin/bash
# Single-A100 request on the stellar-debug QOS, same setup as NEMO/desc_input.
# Usage: sbatch [--time=..] run.sbatch SCRIPT.py ARGS
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
export JAX_ENABLE_X64=true XLA_PYTHON_CLIENT_PREALLOCATE=false
export PYTHONUNBUFFERED=1 MPLBACKEND=Agg OMP_NUM_THREADS=4
nvidia-smi --query-gpu=name,memory.total --format=csv
$ENV/bin/python "$@"
