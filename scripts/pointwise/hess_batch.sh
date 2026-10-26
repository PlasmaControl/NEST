#!/bin/bash
# Hessians of several states of one run: scripts/pointwise/hess_batch.sh RUN TABLES "2 1 0.5"
set -uo pipefail
RUN=$1; TABLES=$2; ALPHAS=$3
PY=${PY:-/scratch/gpfs/EKOLEMEN/lk1158/envs/nest13/bin/python}
for a in $ALPHAS; do
  f=${RUN}_a${a}.npz
  [ -f "$f" ] || { echo "no $f"; continue; }
  $PY scripts/pointwise/hessian.py --tables "$TABLES" --state "$f" --out "${RUN}_a${a}_hess" | grep -E '"active points"|tangent|negative|near zero' | head -12
done
