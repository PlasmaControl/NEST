#!/bin/bash
# Push-through test, arm A: one long inner solve from the Q2b alpha 0.1 stall state with the merit frozen
# (nu 13.94, rho 100), radius-by-delta and no share stop; arm B adds geodesic acceleration. 32/32/6 a/60.
cd /work; for f in desc_src.tar.gz deps_src.tar.gz ripple_L32N4_x1e-3.h5 gpu_entry.py; do [ -f $f ] || cp /results/stage/$f .; done
rm -rf src/scripts src/nest; mkdir -p src out
tar -xzf nest_src.tar.gz -C src
if [ ! -f src/.deps ]; then tar -xzf desc_src.tar.gz -C src && tar -xzf deps_src.tar.gz -C src && touch src/.deps; fi
touch src/.unpacked
cp /results/fxr/A60_cont/A60_cont_a0.1.npz /work/ref.npz; cp /results/fxr/MQ2b/MQ2b_p16_a0.1.npz /work/stall.npz
bash rung_run.sh PTA scripts/meanforce/mf_fixed.py --eq /work/ripple_L32N4_x1e-3.h5 --bext desc-fit --LMN 32 32 6 --frac 60 \
    --N-phi 25 --colloc-rho 3.0 --model beta --force-norm p --p 16 --p-band 0.5675 0.7075 --ref-start --ref-state /work/ref.npz \
    --edge-alpha 1.0 --mean-whole --alphas 0.1 --start /work/stall.npz --keep-start-mult --no-mult-ls --max-outer 1 --maxiter 100000 \
    --stat-tol 1e-12 --no-share-stop --radius-by-delta --f-noise auto  --time-budget 2700 --out /work/out/PTA --mirror /results/fxr/PTA
