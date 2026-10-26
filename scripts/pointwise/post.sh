#!/bin/bash
# Post-processing of one al_run.py run: Hessians and Poincare sections of chosen alphas.
# Usage (inside a job, e.g. sbatch run_sai_bash.sbatch): scripts/pointwise/post.sh RUN TABLES "1 0.3 0.1"
#   RUN: results/pointwise/runs/qr_a40_f15 (without .json)
set -uo pipefail
RUN=$1; TABLES=$2; ALPHAS=$3
PY=${PY:-/scratch/gpfs/EKOLEMEN/lk1158/envs/nest13/bin/python}
states="--state start"
for a in $ALPHAS; do
  f=${RUN}_a${a}.npz
  [ -f "$f" ] || { echo "no $f"; continue; }
  $PY scripts/pointwise/hessian.py --tables "$TABLES" --state "$f" --out "${RUN}_a${a}_hess" | tail -3
  states="$states --state a${a}=$f"
done
$PY scripts/pointwise/poincare_state.py --tables "$TABLES" $states --periods 400 --out "${RUN}_poincare.npz"
