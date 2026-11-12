#!/bin/bash
# R1 follow-up: (1) a/100 grid check of the R1 end states (OOM on 40 GB), (2) alpha 0.05 -> 0.03 continuation of the
# R1 shielded (S) and b21-pinned beta=1 states (warm multipliers), (3) a/80 + a/100 evals of the new states. 40/40/6.
set -u
cd /work; for f in desc_src.tar.gz deps_src.tar.gz ripple_L32N4_x1e-3.h5 gpu_entry.py; do [ -f $f ] || cp /results/stage/$f .; done
rm -rf src/scripts src/nest; mkdir -p src out
tar -xzf nest_src.tar.gz -C src
if [ ! -f src/.deps ]; then tar -xzf desc_src.tar.gz -C src && tar -xzf deps_src.tar.gz -C src && touch src/.deps; fi
touch src/.unpacked
EQ=/work/ripple_L32N4_x1e-3.h5; I=/results/fxr/res/R1; O=/results/fxr/res/R1b; mkdir -p $O
T0=$(date +%s); p() { echo "[res2] $1 at $(( $(date +%s) - T0 )) s" | tee -a $O/progress.txt; }
EVA="--eq $EQ --bext desc-fit --LMN 40 40 6 --N-phi 25 --alphas 1 --model beta --force-norm p --p-schedule 16 --p-band 0.5675 0.7075 --eval-only --b21-report"
EV=""; for T in S b0.2 b0.5 b1.0 rel0.5; do F=$(ls -t $I/R1_${T}_a0.1.npz $I/R1_${T}_budget.npz 2>/dev/null | head -1); [ -n "$F" ] && EV="$EV --eval-state $T=$F"; done
p "ev100 of R1 end states begin"
[ -f $O/R1b_ev100.json ] || bash rung_run.sh R1b_ev100 scripts/meanforce/mf_fixed.py $EVA --frac 100 $EV --out /work/out/R1b_ev100 --mirror $O
COMMON="--eq $EQ --bext desc-fit --LMN 40 40 6 --frac 80 --N-phi 25 --alphas 0.05 0.03 --max-outer 8 --maxiter 40 \
        --snap-outer --f-noise auto --radius-by-delta --no-share-stop --mirror $O"
p "S continuation begin"
bash rung_run.sh R1b_S scripts/meanforce/mf_fixed.py $COMMON --start $I/R1_S_a0.1.npz --time-budget 1500 --out /work/out/R1b_S
p "b1.0 continuation begin"
bash rung_run.sh R1b_b1.0 scripts/meanforce/mf_fixed.py $COMMON --start $I/R1_b1.0_a0.1.npz --pin-b21 1.0 --time-budget 3000 --out /work/out/R1b_b1.0
EV2=""; for f in $(ls /work/out/R1b_S_a0.0*.npz /work/out/R1b_S_budget.npz /work/out/R1b_b1.0_a0.0*.npz /work/out/R1b_b1.0_budget.npz 2>/dev/null | grep -vE "latest|_o[0-9]"); do
  EV2="$EV2 --eval-state $(basename $f .npz)=$f"; done
p "evals of new states begin ($EV2)"
for G in 100 80; do bash rung_run.sh R1b_ev2_$G scripts/meanforce/mf_fixed.py $EVA --frac $G $EV2 --out /work/out/R1b_ev2_$G --mirror $O; done
p "all done"; true
