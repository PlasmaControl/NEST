#!/bin/bash
# D1-err: cylindrical B at the B4 targets (exact J at the nodes) at a/30 and a/40
V=/results/cyl/d1_err; source /work/common_cyl.sh
log "commit $(cat /work/commit.txt)"
cp /work/b4_targets.npz "$V/targets.npz"
st=0
for f in 30 40; do
  run b4_cyl_a$f tests/checks/cyl_b4.py cyl --frac $f --dir "$V" || { st=$?; log "stops at a/$f"; break; }
done
finish; exit $st
