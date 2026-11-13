#!/bin/bash
# Memory/timing probe for higher-resolution ripple rungs, plus a Biot-Savart grid check (2026-11-13).
#   bash job_mem.sh P64   64/64/6, field grid a/160: 40 inner steps (alpha 0.05, mean bound) from the R1f shielded state
#                         embedded from 40/40/6 (fresh multipliers); then eval-only dumps of start + end state at
#                         a/160 (own grid), a/200 and a/240, same collocation (J, B, F per point, for differencing).
#   bash job_mem.sh P80   80/80/6, field grid a/200: setup + 10 inner steps (memory and step time only).
# All runs: --locate-chunk 131072 (locate transient ~5 GB, so later peaks are visible), --mem-every 1, N_phi 25.
set -u
MODE=$1
cd /work; for f in desc_src.tar.gz deps_src.tar.gz ripple_L32N4_x1e-3.h5 gpu_entry.py; do [ -f $f ] || cp /results/stage/$f .; done
rm -rf src/scripts src/nest; mkdir -p src out
tar -xzf nest_src.tar.gz -C src
if [ ! -f src/.deps ]; then tar -xzf desc_src.tar.gz -C src && tar -xzf deps_src.tar.gz -C src && touch src/.deps; fi
touch src/.unpacked
EQ=/work/ripple_L32N4_x1e-3.h5; O=/results/fxr/mem/$MODE; mkdir -p $O
T0=$(date +%s); p() { echo "[mem] $1 at $(( $(date +%s) - T0 )) s" | tee -a $O/progress.txt; }
/opt/conda/bin/python -c "import numpy as np; d=np.load('/work/R1f_S_a0.05.npz'); np.savez('/work/st40.npz', u=d['u'], c_p=d['c_p'])"
COMMON="--eq $EQ --bext desc-fit --N-phi 25 --locate-chunk 131072 --mem-every 1 --mirror $O"
EVAL="--alphas 1 --model beta --force-norm p --p-schedule 16 --p-band 0.5675 0.7075 --eval-only --b21-report --eval-dump"
if [ "$MODE" = P64 ]; then
  p "P64 solve begin"
  bash rung_run.sh P64 scripts/meanforce/mf_fixed.py $COMMON --LMN 64 64 6 --frac 160 --start /work/st40.npz --start-LMN 40 40 6 \
       --alphas 0.05 --max-outer 1 --maxiter 40 --snap-outer --f-noise auto --radius-by-delta --no-share-stop \
       --time-budget 5400 --out /work/out/P64
  END=$(ls -t /work/out/P64_a0.05.npz /work/out/P64_budget.npz /work/out/P64_o*.npz 2>/dev/null | head -1)
  p "P64 solve end, end state $END"
  for G in 160 200 240; do
    p "eval a/$G begin"
    bash rung_run.sh P64_ev$G scripts/meanforce/mf_fixed.py $COMMON --LMN 64 64 6 --frac $G $EVAL \
         --start /work/st40.npz --start-LMN 40 40 6 ${END:+--eval-state P64end=$END} --out /work/out/P64_ev$G
  done
elif [ "$MODE" = P80 ]; then
  p "P80 solve begin"
  bash rung_run.sh P80 scripts/meanforce/mf_fixed.py $COMMON --LMN 80 80 6 --frac 200 --start /work/st40.npz --start-LMN 40 40 6 \
       --alphas 0.05 --max-outer 1 --maxiter 10 --f-noise auto --radius-by-delta --no-share-stop \
       --time-budget 6000 --out /work/out/P80
fi
p "all done"; true
