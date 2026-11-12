#!/bin/bash
# Option-A continuation at a/60: alpha 0.3 (re-seat on the finer grid) -> 0.1 -> 0.03 -> 0.01, warm start from the a/40
# alpha 0.3 final state; 32/32/6, N_phi 25; <= 12 outer iterations per stage; 2 h budget; streamed + checkpointed per stage.
cd /work; mkdir -p src out
tar -xzf nest_src.tar.gz -C src
if [ ! -f src/.deps ]; then tar -xzf desc_src.tar.gz -C src && tar -xzf deps_src.tar.gz -C src && touch src/.deps; fi
touch src/.unpacked
EQ=/work/ripple_L32N4_x1e-3.h5
bash rung_run.sh A60_cont scripts/meanforce/mf_fixed.py --eq $EQ --bext desc-fit --LMN 32 32 6 --frac 60 --N-phi 25 \
    --alphas 0.3 0.1 0.03 0.01 --start /results/fxr/A32_a03/final.npz --max-outer 12 --time-budget 7200 --stage-flat 0.02 --stall-stop 1e-3 \
    --out /work/out/A60_cont --mirror /results/fxr/A60_cont
