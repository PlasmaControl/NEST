#!/bin/bash
# D2: production precise_QA optimization with the cylindrical backend; run_d2.sh TAG <optimize_qa args>
TAG=$1; shift
V=/results/cyl/d2_$TAG; source /work/common_cyl.sh
export XLA_PYTHON_CLIENT_ALLOCATOR=platform  # as main's production runs
log "commit $(cat /work/commit.txt); args $*"
run opt scripts/optimize_qa.py --out "$V/run" "$@"
st=$?
finish; cp "$V"/run* /work/out/ 2>/dev/null; exit $st
