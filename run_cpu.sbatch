#!/bin/bash
# CPU job on stellar (JAX on CPU).
# Usage: sbatch [--time=..] [--mem=..] run_cpu.sbatch SCRIPT.py ARGS
#SBATCH --account=mae
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=96G
#SBATCH --time=02:00:00
#SBATCH --job-name=nest-cpu
#SBATCH --output=logs/nest-%j.log
set -euo pipefail
cd "$SLURM_SUBMIT_DIR"
ENV=/scratch/gpfs/lk1158/.conda/envs/desc-env
export JAX_PLATFORMS=cpu JAX_ENABLE_X64=true
export PYTHONUNBUFFERED=1 MPLBACKEND=Agg
$ENV/bin/python "$@"
