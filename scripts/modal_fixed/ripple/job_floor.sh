#!/bin/bash
# Force floor at 64/64/6 (2026-11-13): alpha 0.001 (deliberately infeasible), mean bound, no b21 pin.
# Start: 64/64/6 alpha 0.05 end state P64_a0.05.npz (fresh multipliers). Solve grid a/200 (finer than a/160 because
# tighter bounds exploit field-grid error more), N_phi 25, default collocation. <= 8 outer x 80 inner, time budget.
# Then eval-only of the start, every outer snapshot and the end state at a/240 and a/280 (same collocation points).
set -u
cd /work; for f in desc_src.tar.gz deps_src.tar.gz ripple_L32N4_x1e-3.h5 gpu_entry.py; do [ -f $f ] || cp /results/stage/$f .; done
rm -rf src/scripts src/nest; mkdir -p src out
tar -xzf nest_src.tar.gz -C src
if [ ! -f src/.deps ]; then tar -xzf desc_src.tar.gz -C src && tar -xzf deps_src.tar.gz -C src && touch src/.deps; fi
touch src/.unpacked
EQ=/work/ripple_L32N4_x1e-3.h5; O=/results/fxr/floor/F64; mkdir -p $O
T0=$(date +%s); p() { echo "[floor] $1 at $(( $(date +%s) - T0 )) s" | tee -a $O/progress.txt; }
/opt/conda/bin/python -c "import numpy as np; d=np.load('/work/P64_a0.05.npz'); np.savez('/work/st64.npz', u=d['u'], c_p=d['c_p'])"
COMMON="--eq $EQ --bext desc-fit --LMN 64 64 6 --N-phi 25 --locate-chunk 131072 --mirror $O"
p "F64 solve begin"
bash rung_run.sh F64 scripts/meanforce/mf_fixed.py $COMMON --frac 200 --start /work/st64.npz \
     --alphas 0.001 --max-outer 8 --maxiter 80 --snap-outer --f-noise auto --radius-by-delta --no-share-stop \
     --time-budget ${TB:-13500} --out /work/out/F64
p "F64 solve end"
EV=""
for f in $(ls /work/out/F64_a0.001_o*.npz 2>/dev/null | sort -V) /work/out/F64_a0.001.npz /work/out/F64_budget.npz; do
  [ -f $f ] && EV="$EV --eval-state $(basename $f .npz)=$f"; done
for G in 240 280; do
  p "eval a/$G begin ($EV)"
  bash rung_run.sh F64_ev$G scripts/meanforce/mf_fixed.py $COMMON --frac $G --alphas 1 --model beta --force-norm p \
       --p-schedule 16 --p-band 0.5675 0.7075 --eval-only --b21-report --shell-max 0.30 0.995 \
       --rho-bins 0 0.3 0.5 0.6 0.7 0.8 0.85 0.9 0.95 1.0 --start /work/st64.npz $EV --out /work/out/F64_ev$G
done
p "all done"; true
