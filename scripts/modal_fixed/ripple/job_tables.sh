#!/bin/bash
# Tables of B, J, p, F on a DESC flux grid (391 x 128 x 32) for the highest-resolution beta = 0 / beta = 1 states
# (R1f: 40/40/6, solved at a/100 with 98 radii, alpha -> 0.05), DESC and the vacuum field; for offline tracing.
set -u
cd /work; for f in desc_src.tar.gz deps_src.tar.gz ripple_L32N4_x1e-3.h5 gpu_entry.py; do [ -f $f ] || cp /results/stage/$f .; done
rm -rf src/scripts src/nest; mkdir -p src out
tar -xzf nest_src.tar.gz -C src
if [ ! -f src/.deps ]; then tar -xzf desc_src.tar.gz -C src && tar -xzf deps_src.tar.gz -C src && touch src/.deps; fi
touch src/.unpacked
G=/results/fxr/res/R1g
bash rung_run.sh tables scripts/meanforce/state_tables.py --eq /work/ripple_L32N4_x1e-3.h5 --LMN 40 40 6 --frac 100 --N-phi 25 \
    --state S=$G/R1f_S_a0.05.npz --state b1=$G/R1f_b1.0_a0.05.npz --out /work/out/tables
mkdir -p /results/fxr/res/tables && cp /work/out/tables/* /results/fxr/res/tables/ 2>/dev/null; ls -la /work/out/tables; true
