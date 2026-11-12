#!/bin/bash
# (1) b21 functional check against start_check (start, stall, pt3, island seeds), (2) exact plane maps of the force
# from the shielded start along the q = 2 island-seed direction x {valley, q = 3 seed, back to J_fit}, (3) smoke test of
# the b21 pin (restore + 4 inner steps). 32/32/6 a/60.
cd /work; for f in desc_src.tar.gz deps_src.tar.gz ripple_L32N4_x1e-3.h5 gpu_entry.py; do [ -f $f ] || cp /results/stage/$f .; done
rm -rf src/scripts src/nest; mkdir -p src out
tar -xzf nest_src.tar.gz -C src
if [ ! -f src/.deps ]; then tar -xzf desc_src.tar.gz -C src && tar -xzf deps_src.tar.gz -C src && touch src/.deps; fi
touch src/.unpacked
EQ=/work/ripple_L32N4_x1e-3.h5; O=/results/fxr/pm; mkdir -p $O
cp /results/fxr/A60_cont/A60_cont_a0.1.npz /work/start.npz; cp /results/fxr/MQ2b/MQ2b_p16_a0.1.npz /work/stall.npz
cp /results/fxr/PTA/PTA_p16_a0.1.npz /work/pt3.npz
cp /results/fxr/pre32/seed32_r0.6375_w0.07.npz /work/sq2.npz; cp /results/fxr/pre32/seed32_r0.9131_w0.03.npz /work/sq3.npz
cp /results/fxr/pre32/seed32_all.npz /work/sall.npz
S=/work/start.npz
bash rung_run.sh pm_ev60 scripts/meanforce/mf_fixed.py --eq $EQ --bext desc-fit --LMN 32 32 6 --frac 60 --N-phi 25 --colloc-rho 3.0 \
    --alphas 1 --model beta --force-norm p --p-schedule 16 --p-band 0.5675 0.7075 --eval-only --b21-report \
    --eval-state start=$S --eval-state stall=/work/stall.npz --eval-state pt3=/work/pt3.npz \
    --eval-state seed_q2=/work/sq2.npz --eval-state seed_q3=/work/sq3.npz --eval-state seed_all=/work/sall.npz \
    --plane "isl2_valley|$S|/work/sq2.npz|J_fit|/work/pt3.npz|$S|-0.2,1.2,29|-3,6,37" \
    --plane "isl2_isl3|$S|/work/sq2.npz|J_fit|/work/sq3.npz|J_fit|-0.2,1.2,29|-0.2,1.2,29" \
    --plane "isl2_jfit|$S|/work/sq2.npz|J_fit|J_fit|$S|-0.2,1.2,29|-1,1.5,26" \
    --out /work/out/pm_ev60 --mirror $O
bash rung_run.sh pm_pin scripts/meanforce/mf_fixed.py --eq $EQ --bext desc-fit --LMN 32 32 6 --frac 60 --N-phi 25 \
    --alphas 0.3 --start $S --pin-b21 0.2 --b21-phase-state /work/sall.npz --max-outer 1 --maxiter 4 \
    --f-noise auto --radius-by-delta --no-share-stop --time-budget 1500 --out /work/out/pm_pin --mirror $O
