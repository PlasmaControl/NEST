#!/bin/bash
# jaxmg (cuSOLVERMp) jobs on Stellar AI: one process per GPU.
# Usage: sbatch --gres=gpu:nvidia_b200:G --ntasks=G run_mg.sbatch SCRIPT.py ARGS
#SBATCH --account=kolemen
#SBATCH --partition=gpu
#SBATCH --qos=gpu-short
#SBATCH --nodes=1
#SBATCH --exclude=sai-n01g4  # GPU 2 there fails (CUDA_ERROR_UNKNOWN), 2026-10-26
#SBATCH --ntasks=4
#SBATCH --gres=gpu:nvidia_b200:4
#SBATCH --cpus-per-task=8
#SBATCH --mem=400G
#SBATCH --time=04:00:00
#SBATCH --job-name=mg
#SBATCH --output=logs/mg-%j.log
set -euo pipefail
cd "$SLURM_SUBMIT_DIR"
export ENV=/scratch/gpfs/EKOLEMEN/lk1158/envs/nest_mg
export JAX_ENABLE_X64=true XLA_PYTHON_CLIENT_PREALLOCATE=false PYTHONUNBUFFERED=1 MPLBACKEND=Agg OMP_NUM_THREADS=8
export XLA_PYTHON_CLIENT_MEM_FRACTION=0.90 MULTI=1 TF_CPP_MIN_LOG_LEVEL=3
nvidia-smi --query-gpu=name,memory.total --format=csv
srun --cpu-bind=none --gres=gpu:nvidia_b200:$SLURM_NTASKS bash -c 'echo "task $SLURM_PROCID local $SLURM_LOCALID CUDA_VISIBLE_DEVICES=$CUDA_VISIBLE_DEVICES"; exec $ENV/bin/python "$@"' _ "$@"
