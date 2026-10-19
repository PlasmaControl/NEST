#!/bin/bash
# Compact vs cached J sampling: time per optimizer step and GPU memory, production precise_QA
# (factors 3 3 2.5, float64 cylindrical, cold start, defaults), 5 iterations each: run_jc.sh "FRACS"
FRACS=${1:-"20 30"}
V=/results/cyl/jc; source /work/common_cyl.sh
export XLA_PYTHON_CLIENT_ALLOCATOR=platform  # nvidia-smi then shows the memory in use
log "commit $(cat /work/commit.txt); fracs $FRACS"
st=0
for F in $FRACS; do
  for M in compact cached; do
    run "a${F}_$M" scripts/optimize_qa.py --grid cylindrical --frac "$F" --maxiter 5 --no-check \
        --j-sampling "$M" --out "$V/a${F}_$M" || st=1
  done
done
finish; exit $st
