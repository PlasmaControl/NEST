#!/bin/bash
# Tight max-bound run Q2b at 32/32/6 a/60 (97 GL radii): composite of the q = 2 band (0.5675-0.7075) p16 <= alpha x the start's
# and the WHOLE-volume mean <= the start's (strictly tighter than its mean bound); alpha 0.1 -> 0.05 -> 0.025 (0.05 x start ~ 1.9e-8,
# ~4x below the start's surface-averaged |F| at q = 2, so angular spreading cannot meet it); start = the shielded a/60 alpha 0.1 state;
# warm start from the Q2 alpha 0.25 state (reference stays the shielded start; alpha stepped to keep c O(1));
# plain first-order multiplier; snapshots every outer; CONTROL NEXT/STOP in /results/fxr/MQ2b.
cd /work; for f in desc_src.tar.gz deps_src.tar.gz ripple_L32N4_x1e-3.h5 gpu_entry.py; do [ -f $f ] || cp /results/stage/$f .; done
rm -rf src/scripts src/nest; mkdir -p src out
tar -xzf nest_src.tar.gz -C src
if [ ! -f src/.deps ]; then tar -xzf desc_src.tar.gz -C src && tar -xzf deps_src.tar.gz -C src && touch src/.deps; fi
touch src/.unpacked
cp /results/fxr/A60_cont/A60_cont_a0.1.npz /work/ref.npz; ls -la /results/fxr/MQ2/; (cp /results/fxr/MQ2/MQ2_budget.npz /work/start.npz || cp /results/fxr/MQ2/MQ2_p16_a0.25_latest.npz /work/start.npz)
bash rung_run.sh MQ2b scripts/meanforce/mf_fixed.py --eq /work/ripple_L32N4_x1e-3.h5 --bext desc-fit --LMN 32 32 6 --frac 60 \
    --N-phi 25 --colloc-rho 3.0 --model beta --force-norm p --p 16 --p-band 0.5675 0.7075 --ref-start --ref-state /work/ref.npz --edge-alpha 1.0 --mean-whole \
    --alphas 0.25 0.1 0.05 0.025 --start /work/start.npz --no-mult-ls --snap-outer --stage-flat 0.02 --stall-stop 1e-3 \
    --max-outer 10 --time-budget 12600 --out /work/out/MQ2b --mirror /results/fxr/MQ2b
