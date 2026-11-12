#!/bin/bash
# Diagnostics of the Q2b alpha 0.1 stall state vs alpha 0.25 and the shielded start (32/32/6): q = 2 island / lambda / pressure /
# Poincare, and per-shell |F| max and mean at a/60 and a/80 (97 radii).
cd /work; for f in desc_src.tar.gz deps_src.tar.gz ripple_L32N4_x1e-3.h5 gpu_entry.py; do [ -f $f ] || cp /results/stage/$f .; done
rm -rf src/scripts src/nest; mkdir -p src out
tar -xzf nest_src.tar.gz -C src
if [ ! -f src/.deps ]; then tar -xzf desc_src.tar.gz -C src && tar -xzf deps_src.tar.gz -C src && touch src/.deps; fi
touch src/.unpacked
EQ=/work/ripple_L32N4_x1e-3.h5; O=/results/fxr/dqb; mkdir -p $O
cp /results/fxr/MQ2b/MQ2b_p16_a0.1.npz /work/q2b_a01.npz; cp /results/fxr/MQ2b/MQ2b_p16_a0.25.npz /work/q2b_a025.npz
cp /results/fxr/A60_cont/A60_cont_a0.1.npz /work/start.npz
for F in 60 80; do
bash rung_run.sh dqb_ev$F scripts/meanforce/mf_fixed.py --eq $EQ --bext desc-fit --LMN 32 32 6 --frac $F --N-phi 25 --colloc-rho 3.0 \
    --alphas 1 --model beta --force-norm p --p-schedule 16 --p-band 0.5675 0.7075 --eval-only --eval-state start=/work/start.npz \
    --eval-state q2b_a025=/work/q2b_a025.npz --eval-state q2b_a01=/work/q2b_a01.npz --shell-max 0.40 0.995 \
    --rho-bins 0 0.3 0.5 0.55 0.6 0.62 0.64 0.66 0.7 0.75 0.8 0.85 0.88 0.9 0.92 0.94 0.96 0.98 1.0 --out /work/out/dqb_ev$F --mirror $O
done
bash rung_run.sh dqb_m2 scripts/meanforce/start_check_bext.py --eq $EQ --LMN 32 32 6 --frac 60 --N-phi 25 --m 2 --k 1 \
    --rho-grid 0.45 0.82 186 --band 0.04 0.08 0.16 --state /work/start.npz --state /work/q2b_a025.npz --state /work/q2b_a01.npz \
    --perts vacuum start:A q2b_a025:A q2b_a01:A --turns 300 --lines 41 --seed-band 0.56 0.72 --seed-theta 0.0 --pressure \
    --out $O/dqb_m2
