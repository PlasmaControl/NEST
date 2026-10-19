#!/bin/bash
# D1-mem: GPU memory model (compile only) and measured ladders, one process per size
V=/results/cyl/d1_mem; source /work/common_cyl.sh
log "commit $(cat /work/commit.txt)"
run model_cyl tests/checks/cyl_memory.py --fracs 20 30 40 50 60 70 --methods cylindrical --cyl-dtype float32 --cyl-sym --cyl-zfold --out "$V/model_cyl_gpu.json"
run model_box tests/checks/cyl_memory.py --fracs 20 30 40 --methods box --out "$V/model_box_gpu.json"
for f in 20 30 40 50 60; do
  run cyl_a$f tests/checks/cyl_measure.py --method cylindrical --frac $f --sym --out "$V/cyl_a$f.json" || { log "cyl ladder stops at a/$f"; break; }
done
for f in 20 30 40; do
  run box_a$f tests/checks/cyl_measure.py --method box --frac $f --out "$V/box_a$f.json" || { log "box ladder stops at a/$f"; break; }
done
[ -f "$V/cyl_a60.json" ] && run cyl_a70 tests/checks/cyl_measure.py --method cylindrical --frac 70 --sym --out "$V/cyl_a70.json"
finish; exit 0
