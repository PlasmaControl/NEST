#!/bin/bash
# Max-bound run Q2 at 32/32/6 a/60 (97 GL radii): composite of the band p16 (band 0.5675 0.7075) <= alpha x the start's and the mean
# outside the band <= 1.044 x the start's; alpha 0.5 -> 0.25 -> 0.125; start = the a/60 alpha 0.1 end state (shielded);
# multiplier by plain first-order updates (no LS clip); snapshots every outer; CONTROL NEXT/STOP in /results/fxr/MQ2.
cd /work; for f in desc_src.tar.gz deps_src.tar.gz ripple_L32N4_x1e-3.h5 gpu_entry.py; do [ -f $f ] || cp /results/stage/$f .; done
rm -rf src/scripts src/nest; mkdir -p src out
tar -xzf nest_src.tar.gz -C src
if [ ! -f src/.deps ]; then tar -xzf desc_src.tar.gz -C src && tar -xzf deps_src.tar.gz -C src && touch src/.deps; fi
touch src/.unpacked
cp /results/fxr/A60_cont/A60_cont_a0.1.npz /work/start.npz
bash rung_run.sh MQ2 scripts/meanforce/mf_fixed.py --eq /work/ripple_L32N4_x1e-3.h5 --bext desc-fit --LMN 32 32 6 --frac 60 \
    --N-phi 25 --colloc-rho 3.0 --model beta --force-norm p --p 16 --p-band 0.5675 0.7075 --ref-start --edge-alpha 1.044 \
    --alphas 0.5 0.25 0.125 --start /work/start.npz --no-mult-ls --snap-outer --stage-flat 0.02 --stall-stop 1e-3 \
    --max-outer 8 --time-budget 10800 --out /work/out/MQ2 --mirror /results/fxr/MQ2
