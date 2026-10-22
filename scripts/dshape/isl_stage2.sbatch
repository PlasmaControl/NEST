#!/bin/bash
# One stage of the J-weight continuation at N = 2 on 4 B200s of Stellar AI, with the fast solver
# (resident tables, Cholesky damping scan, 100000-row QR blocks, one thread per GPU), then (eps > 0) the
# island and pressure diagnostics, then the next stage K + 1 is submitted. One job per chain in the queue.
# Stage 3 (w_J = 1e-3) starts from the N = 1 chain's state a120Ce<EPS>s3; later stages from the previous one.
# Usage (repo root): sbatch scripts/dshape/isl_stage2.sbatch EPS K [KMAX]
#SBATCH --account=kolemen
#SBATCH --partition=gpu
#SBATCH --qos=gpu-short
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:nvidia_b200:4
#SBATCH --cpus-per-task=16
#SBATCH --mem=500G
#SBATCH --time=04:00:00
#SBATCH --job-name=stage2
#SBATCH --output=logs/nest-%j.log
set -euo pipefail
cd "$SLURM_SUBMIT_DIR"
EPS=$1; K=$2; KMAX=${3:-10}
ENV=/scratch/gpfs/EKOLEMEN/lk1158/envs/nest13
export NEST_DEVICE=gpu JAX_ENABLE_X64=true XLA_PYTHON_CLIENT_PREALLOCATE=false
export JAX_COMPILATION_CACHE_DIR=${JAX_COMPILATION_CACHE_DIR:-$SLURM_SUBMIT_DIR/.cache/jax}
export JAX_PERSISTENT_CACHE_MIN_COMPILE_TIME_SECS=0 JAX_PERSISTENT_CACHE_MIN_ENTRY_SIZE_BYTES=-1
export XLA_PYTHON_CLIENT_ALLOCATOR=${XLA_PYTHON_CLIENT_ALLOCATOR:-platform}
export XLA_PYTHON_CLIENT_MEM_FRACTION=${XLA_PYTHON_CLIENT_MEM_FRACTION:-0.90}
export PYTHONUNBUFFERED=1 MPLBACKEND=Agg OMP_NUM_THREADS=8
NAME=a120Cn2e${EPS}s${K}
if [ "$K" -gt 3 ]; then START="--start-from a120Cn2e${EPS}s$((K-1))"; else START="--start-from a120Ce${EPS}s3"; fi
$ENV/bin/python scripts/dshape/isl_scan.py --base results/gs/hN_a120D --N 2 --eps $EPS --j-mode full \
    --w-J 1e-$K --w-W 0 --w-pos 0 --steps qr --gn-tol 0 --rtol 1e-9 --iters 200 --batch 16 8 --resume \
    --speed resident,chol,narrow,merge=100000,gpus=4 --name $NAME $START
if [ "$EPS" != "0" ]; then
  $ENV/bin/python scripts/dshape/isl_islands.py --name $NAME || echo "islands failed"
  $ENV/bin/python scripts/dshape/isl_pmap.py --name $NAME || echo "pmap failed"
fi
if [ "$K" -lt "$KMAX" ]; then
  sbatch --job-name=a120Cn2e${EPS}s$((K+1)) scripts/dshape/isl_stage2.sbatch $EPS $((K+1)) $KMAX
fi
