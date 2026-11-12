#!/bin/bash
# Max-bound pre-step at 32/32/6 (the grid-converged basis at a/60): island seeds; eval at a/60 and a/80 with 97 GL radii
# (colloc-rho 3.0): band p-norms (p 4, 16), band max, per-shell max for J_fit, alpha 0.3 / 0.1 end states and the seeds,
# q = 2 band [0.5675, 0.7075] and q = 3 band [0.868, 0.97]; island / pressure / Poincare diagnostics of the seeds.
cd /work; for f in desc_src.tar.gz deps_src.tar.gz ripple_L32N4_x1e-3.h5 gpu_entry.py; do [ -f $f ] || cp /results/stage/$f .; done
rm -rf src/scripts src/nest; mkdir -p src out
tar -xzf nest_src.tar.gz -C src
if [ ! -f src/.deps ]; then tar -xzf desc_src.tar.gz -C src && tar -xzf deps_src.tar.gz -C src && touch src/.deps; fi
touch src/.unpacked
EQ=/work/ripple_L32N4_x1e-3.h5; O=/results/fxr/pre32; mkdir -p $O
cp /results/fxr/A60_cont/A60_cont_a0.3.npz /work/a03_32.npz; cp /results/fxr/A60_cont/A60_cont_a0.1.npz /work/a01_32.npz
bash rung_run.sh seed32 scripts/meanforce/island_seed.py --eq $EQ --LMN 32 32 6 --frac 60 --N-phi 25 \
    --window 0.6375 0.07 0.03 --window 0.9131 0.03 0.015 --window -1 0 0 --cg 800 --out /work/out/seed32 || exit 1
cp /work/out/seed32* $O/
S2=/work/out/seed32_r0.6375_w0.07.npz; S3=/work/out/seed32_r0.9131_w0.03.npz; SA=/work/out/seed32_all.npz
BINS="0 0.3 0.5 0.55 0.6 0.62 0.64 0.66 0.7 0.75 0.8 0.85 0.88 0.9 0.92 0.94 0.96 0.98 1.0"
for F in 60 80; do
COMMON="--eq $EQ --bext desc-fit --LMN 32 32 6 --frac $F --N-phi 25 --colloc-rho 3.0 --alphas 1 --model beta --force-norm p \
    --p-schedule 4 16 --eval-only --eval-state a03_32=/work/a03_32.npz --eval-state a01_32=/work/a01_32.npz \
    --eval-state seed_q2=$S2 --eval-state seed_q3=$S3 --eval-state seed_all=$SA --mirror $O"
EXTRA=""; [ $F = 60 ] && EXTRA="--eval-save"
bash rung_run.sh pre32_a${F}_q2 scripts/meanforce/mf_fixed.py $COMMON --p-band 0.5675 0.7075 --shell-max 0.50 0.995 \
    --rho-bins $BINS $EXTRA --out /work/out/pre32_a${F}_q2
bash rung_run.sh pre32_a${F}_q3 scripts/meanforce/mf_fixed.py $COMMON --p-band 0.868 0.97 --out /work/out/pre32_a${F}_q3
done
cp /work/out/pre32_a60_q2_eval_seed_q2.npz /work/seed_q2.npz; cp /work/out/pre32_a60_q2_eval_seed_q3.npz /work/seed_q3.npz
cp /work/out/pre32_a60_q2_eval_seed_all.npz /work/seed_all.npz
bash rung_run.sh pre32_isl2 scripts/meanforce/start_check_bext.py --eq $EQ --LMN 32 32 6 --frac 60 --N-phi 25 --m 2 --k 1 \
    --rho-grid 0.45 0.82 186 --band 0.04 0.08 0.16 --state /work/seed_q2.npz --state /work/seed_all.npz \
    --perts vacuum seed_q2:A seed_all:A --turns 300 --lines 41 --seed-band 0.54 0.74 --seed-theta 0.0 --pressure \
    --out $O/pre32_isl2
bash rung_run.sh pre32_isl3 scripts/meanforce/start_check_bext.py --eq $EQ --LMN 32 32 6 --frac 60 --N-phi 25 --m 3 --k 1 \
    --rho-grid 0.80 0.995 196 --band 0.02 0.04 0.08 --state /work/seed_q3.npz --state /work/seed_all.npz \
    --perts vacuum seed_q3:A seed_all:A --turns 300 --lines 41 --seed-band 0.86 0.97 --seed-theta 0.7854 --pressure \
    --out $O/pre32_isl3
ls -la $O; du -sh /results/jaxcache
