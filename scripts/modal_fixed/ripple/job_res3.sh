#!/bin/bash
# R1 grid-convergence test (40/40/6): repeat the alpha 0.05 continuation of the R1 shielded (S) and b21-pinned beta=1
# states from the same R1 alpha 0.1 end states as R1b (warm multipliers), with matched effort (<= 2 outer x 40 = 80
# inner steps each), on two tighter grids:
#   R1c: field grid a/80 (as R1b), radial collocation 98 GL radii (--colloc-rho 2.7; default 55)
#   R1f: field grid a/100, 98 radii
# then one reference evaluation of all six states (R1b, R1c, R1f x S, b1.0) at a/120 with 120 radii (--colloc-rho 3.3).
set -u
cd /work; for f in desc_src.tar.gz deps_src.tar.gz ripple_L32N4_x1e-3.h5 gpu_entry.py; do [ -f $f ] || cp /results/stage/$f .; done
rm -rf src/scripts src/nest; mkdir -p src out
tar -xzf nest_src.tar.gz -C src
if [ ! -f src/.deps ]; then tar -xzf desc_src.tar.gz -C src && tar -xzf deps_src.tar.gz -C src && touch src/.deps; fi
touch src/.unpacked
EQ=/work/ripple_L32N4_x1e-3.h5; I=/results/fxr/res/R1; B=/results/fxr/res/R1b; O=/results/fxr/res/R1g; mkdir -p $O
T0=$(date +%s); p() { echo "[res3] $1 at $(( $(date +%s) - T0 )) s" | tee -a $O/progress.txt; }
BASE="--eq $EQ --bext desc-fit --LMN 40 40 6 --N-phi 25 --alphas 0.05 --max-outer 2 --maxiter 40 \
      --snap-outer --f-noise auto --radius-by-delta --no-share-stop --colloc-rho 2.7 --mirror $O"
endstate() { ls -t /work/out/${1}_a0.05.npz /work/out/${1}_budget.npz 2>/dev/null | head -1; }
for V in "R1c 80 2700" "R1f 100 3600"; do set -- $V; TAG=$1; FR=$2; TB=$3
  p "$TAG S begin"
  bash rung_run.sh ${TAG}_S scripts/meanforce/mf_fixed.py $BASE --frac $FR --start $I/R1_S_a0.1.npz --time-budget $TB --out /work/out/${TAG}_S
  p "$TAG b1.0 begin"
  bash rung_run.sh ${TAG}_b1.0 scripts/meanforce/mf_fixed.py $BASE --frac $FR --start $I/R1_b1.0_a0.1.npz --pin-b21 1.0 --time-budget $TB --out /work/out/${TAG}_b1.0
done
EV="--eval-state R1b_S=$B/R1b_S_budget.npz --eval-state R1b_b1.0=$B/R1b_b1.0_budget.npz"
for T in R1c_S R1c_b1.0 R1f_S R1f_b1.0; do F=$(endstate $T); [ -n "$F" ] && EV="$EV --eval-state $T=$F"; done
p "reference eval begin ($EV)"
bash rung_run.sh R1g_ev120 scripts/meanforce/mf_fixed.py --eq $EQ --bext desc-fit --LMN 40 40 6 --N-phi 25 --frac 120 --colloc-rho 3.3 \
    --alphas 1 --model beta --force-norm p --p-schedule 16 --p-band 0.5675 0.7075 --eval-only --b21-report $EV --out /work/out/R1g_ev120 --mirror $O
p "all done"; true
