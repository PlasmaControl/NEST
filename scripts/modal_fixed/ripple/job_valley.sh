#!/bin/bash
# Valley-navigation A/B at the alpha 0.001 floor (2026-11-13). 64/64/6, field grid a/200, N_phi 25, mean bound, no pin.
# All arms start from F64b_budget.npz (M/eps 16.66) with the same settings as F64b (--cg 1500 --no-rebuild-on-hit,
# f-noise auto, radius-by-delta, no-share-stop) and the same time budget, one after another on ONE container (same GPU).
#   SA: --subspace prev,accel   S: --subspace prev   G: --geo-accel   C: control (last, so it gets the warmest XLA cache)
# Then one eval of all four end states at a/240 (same collocation points).
set -u
cd /work; for f in desc_src.tar.gz deps_src.tar.gz ripple_L32N4_x1e-3.h5 gpu_entry.py; do [ -f $f ] || cp /results/stage/$f .; done
rm -rf src/scripts src/nest; mkdir -p src out
tar -xzf nest_src.tar.gz -C src
if [ ! -f src/.deps ]; then tar -xzf desc_src.tar.gz -C src && tar -xzf deps_src.tar.gz -C src && touch src/.deps; fi
touch src/.unpacked
EQ=/work/ripple_L32N4_x1e-3.h5; O=/results/fxr/valley; mkdir -p $O
T0=$(date +%s); p() { echo "[valley] $1 at $(( $(date +%s) - T0 )) s" | tee -a $O/progress.txt; }
cp /results/fxr/floor/F64b/F64b_budget.npz /work/st.npz; sha256sum /work/st.npz | tee -a $O/progress.txt
COMMON="--eq $EQ --bext desc-fit --LMN 64 64 6 --N-phi 25 --locate-chunk 131072"
SOLVE="--frac 200 --start /work/st.npz --alphas 0.001 --max-outer 8 --maxiter 80 --cg 1500 --no-rebuild-on-hit \
       --f-noise auto --radius-by-delta --no-share-stop --time-budget ${TB:-3300}"
EV=""
for ARM in "V_SA --subspace prev,accel" "V_S --subspace prev" "V_G --geo-accel" "V_C"; do
  set -- $ARM; TAG=$1; shift
  p "$TAG begin ($*)"
  bash rung_run.sh $TAG scripts/meanforce/mf_fixed.py $COMMON $SOLVE "$@" --mirror $O/$TAG --out /work/out/$TAG
  F=$(ls -t /work/out/${TAG}_budget.npz /work/out/${TAG}_a0.001.npz 2>/dev/null | head -1)
  [ -n "$F" ] && EV="$EV --eval-state $TAG=$F"
  p "$TAG end ($F)"
done
p "eval a/240 begin ($EV)"
bash rung_run.sh V_ev240 scripts/meanforce/mf_fixed.py $COMMON --frac 240 --alphas 1 --model beta --force-norm p \
     --p-schedule 16 --p-band 0.5675 0.7075 --eval-only --b21-report --shell-max 0.30 0.995 \
     --rho-bins 0 0.3 0.5 0.6 0.7 0.8 0.85 0.9 0.95 1.0 --start /work/st.npz $EV --mirror $O --out /work/out/V_ev240
p "all done"; true
