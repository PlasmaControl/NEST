#!/bin/bash
# Option-A (desc-fit) single-alpha run: 32/32/6, a/40, N_phi 25, alpha 0.3, 40-min budget; streamed + checkpointed.
EQ=/work/ripple_L32N4_x1e-3.h5
bash rung_run.sh A32_a03 scripts/meanforce/mf_fixed.py --eq $EQ --bext desc-fit --LMN 32 32 6 --frac 40 \
    --N-phi 25 --alphas 0.3 --time-budget 2400 --out /work/out/A32_a03 --mirror /results/fxr/A32_a03
