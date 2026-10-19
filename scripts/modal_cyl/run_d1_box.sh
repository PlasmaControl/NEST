#!/bin/bash
# D1-box: main's VG box GPU memory model and measured ladder (split from D1-mem, whose S_m builds are slow)
V=/results/cyl/d1_box; source /work/common_cyl.sh
log "commit $(cat /work/commit.txt)"
run model_box tests/cyl_memory.py --fracs 20 30 40 --methods box --out "$V/model_box_gpu.json"
for f in 20 30 40; do
  run box_a$f tests/cyl_measure.py --method box --frac $f --out "$V/box_a$f.json" || { log "box ladder stops at a/$f"; break; }
done
finish; exit 0
