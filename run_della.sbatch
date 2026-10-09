#!/bin/bash
# Single-A100 job on Della, same setup as run.sbatch on Stellar.
# Usage: sbatch [--time=..] run_della.sbatch SCRIPT.py ARGS
# Environment: /scratch/gpfs/EKOLEMEN/lk1158/conda-envs/nest-gpu (python 3.12,
# jax[cuda12] 0.7.2, DESC editable from /scratch/gpfs/EKOLEMEN/lk1158/DESC-nest).
#SBATCH --account=kolemen
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:1
#SBATCH --constraint=nomig
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=00:59:00
#SBATCH --job-name=nest
#SBATCH --output=logs/nest-%j.log
set -euo pipefail
cd "$SLURM_SUBMIT_DIR"
ENV=/scratch/gpfs/EKOLEMEN/lk1158/conda-envs/nest-gpu
export NEST_DEVICE=gpu JAX_ENABLE_X64=true XLA_PYTHON_CLIENT_PREALLOCATE=false
export PYTHONUNBUFFERED=1 MPLBACKEND=Agg OMP_NUM_THREADS=4
nvidia-smi --query-gpu=name,memory.total --format=csv
# stop if DESC/JAX do not get the GPU instead of running on the CPU
$ENV/bin/python -c "from desc import set_device; set_device('gpu'); import jax; d = jax.devices(); print('JAX devices:', d); assert d[0].platform == 'gpu'"
$ENV/bin/python "$@"
