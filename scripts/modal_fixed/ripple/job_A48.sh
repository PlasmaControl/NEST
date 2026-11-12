#!/bin/bash
# NEST 48/48/6 a/60 N_phi 25 option A: alpha 0.1 -> 0.05 -> 0.03 from the a/60 alpha 0.1 end state embedded from 32/32/6;
# stage-flat 2%, stall-stop 1e-3, <= 10 outers per stage, 2.5 h budget; CONTROL file NEXT/STOP in /results/fxr/A48_cont.
cd /work; mkdir -p src out
tar -xzf nest_src.tar.gz -C src
if [ ! -f src/.deps ]; then tar -xzf desc_src.tar.gz -C src && tar -xzf deps_src.tar.gz -C src && touch src/.deps; fi
touch src/.unpacked
EQ=/work/ripple_L32N4_x1e-3.h5
bash rung_run.sh A48_cont scripts/meanforce/mf_fixed.py --eq $EQ --bext desc-fit --LMN 48 48 6 --frac 60 --N-phi 25 \
    --alphas 0.1 0.05 0.03 --start /results/fxr/A60_cont/A60_cont_a0.1.npz --start-LMN 32 32 6 --max-outer 10 \
    --time-budget 9000 --stage-flat 0.02 --stall-stop 1e-3 --out /work/out/A48_cont --mirror /results/fxr/A48_cont
