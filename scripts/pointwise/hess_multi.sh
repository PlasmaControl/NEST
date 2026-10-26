#!/bin/bash
# Several hess_batch.sh in one job: scripts/pointwise/hess_multi.sh "run:tables:alphas" ...
set -uo pipefail
R=results/pointwise
for spec in "$@"; do
  IFS=: read run tab al <<< "$spec"
  echo "== $run ($al)"
  bash scripts/pointwise/hess_batch.sh $R/runs/$run $R/tables/$tab.npz "$al"
done
