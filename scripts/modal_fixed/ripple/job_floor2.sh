#!/bin/bash
# Force floor at 64/64/6, restart (2026-11-13): warm start from F64 step 29 (F64_restart_src.npz, its nu/rho),
# Lanczos cap 1500 and --no-rebuild-on-hit (no inner metric rebuild after trust-region boundary steps).
# Solve grid a/200 (tighter bounds exploit field-grid error more), N_phi 25, default collocation. <= 8 outer x 80 inner, time budget.
# Then eval-only of the start, every outer snapshot and the end state at a/240 and a/280 (same collocation points).
set -u
cd /work; for f in desc_src.tar.gz deps_src.tar.gz ripple_L32N4_x1e-3.h5 gpu_entry.py; do [ -f $f ] || cp /results/stage/$f .; done
rm -rf src/scripts src/nest; mkdir -p src out
tar -xzf nest_src.tar.gz -C src
if [ ! -f src/.deps ]; then tar -xzf desc_src.tar.gz -C src && tar -xzf deps_src.tar.gz -C src && touch src/.deps; fi
touch src/.unpacked
EQ=/work/ripple_L32N4_x1e-3.h5; O=/results/fxr/floor/F64b; mkdir -p $O
T0=$(date +%s); p() { echo "[floor] $1 at $(( $(date +%s) - T0 )) s" | tee -a $O/progress.txt; }
cp /results/fxr/floor/F64/F64_restart_src.npz /work/st64.npz; sha256sum /work/st64.npz | tee -a $O/progress.txt
COMMON="--eq $EQ --bext desc-fit --LMN 64 64 6 --N-phi 25 --locate-chunk 131072 --mirror $O"
p "F64b solve begin"
bash rung_run.sh F64b scripts/meanforce/mf_fixed.py $COMMON --frac 200 --start /work/st64.npz \
     --alphas 0.001 --max-outer 8 --maxiter 80 --cg 1500 --no-rebuild-on-hit --snap-outer --f-noise auto --radius-by-delta --no-share-stop \
     --time-budget ${TB:-13500} --out /work/out/F64b
p "F64b solve end"
EV=""
for f in $(ls /work/out/F64b_a0.001_o*.npz 2>/dev/null | sort -V) /work/out/F64b_a0.001.npz /work/out/F64b_budget.npz; do
  [ -f $f ] && EV="$EV --eval-state $(basename $f .npz)=$f"; done
for G in 240 280; do
  p "eval a/$G begin ($EV)"
  bash rung_run.sh F64b_ev$G scripts/meanforce/mf_fixed.py $COMMON --frac $G --alphas 1 --model beta --force-norm p \
       --p-schedule 16 --p-band 0.5675 0.7075 --eval-only --b21-report --shell-max 0.30 0.995 \
       --rho-bins 0 0.3 0.5 0.6 0.7 0.8 0.85 0.9 0.95 1.0 --start /work/st64.npz --eval-state P64_a0.05=/work/P64_a0.05.npz $EV --out /work/out/F64b_ev$G
done
p "all done"; true
