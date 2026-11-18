#!/bin/bash
# Energy-objective plan, gates G1 and G2 (docs: energy_objective_plan.md), one step per container.
# Setting = R1f: 40/40/6, field grid a/100, 98 GL radii (--colloc-rho 2.7), option A (--bext desc-fit), alpha 0.05.
# New constraints in every solve: LCFS B.n modes |m| <= 40, |n| <= 6 (--bn-modes 40 6) and --pin-flux.
#
# usage: bash job_g12.sh STEP
#   g1a      exterior check (direct Biot-Savart) on random du and on the ORIGINAL R1f states (no B.n constraint)
#   g1a80    the same with the B.n cutoff doubled (--bn-modes 80 12)
#   S0       shielded reference: R1f_S restored onto the constraints, solved at lambda = 0
#   I<beta>  island reference: b21 pinned at beta x vacuum (island phase of R1f_b1.0), from R1f_b1.0, lambda = 0
#   g1b      exterior check on S0 / I<beta> and their differences; fine-grid (a/120, 120 radii) energy evaluation
# Inputs (flat in /work): nest_src.tar.gz rung_run.sh job_g12.sh R1f_S_fresh.npz R1f_b1_fresh.npz
# (desc_src, deps_src, gpu_entry.py and the DESC file come from /results/stage, as for the ripple runs).
# Outputs: /results/fxr/energy/<STEP>/ (mirror, fsynced per step) and out/.
set -u
STEP=$1
cd /work; for f in desc_src.tar.gz deps_src.tar.gz ripple_L32N4_x1e-3.h5 gpu_entry.py; do [ -f $f ] || cp /results/stage/$f .; done
rm -rf src/scripts src/nest; mkdir -p src out
tar -xzf nest_src.tar.gz -C src
if [ ! -f src/.deps ]; then tar -xzf desc_src.tar.gz -C src && tar -xzf deps_src.tar.gz -C src && touch src/.deps; fi
touch src/.unpacked
EQ=/work/ripple_L32N4_x1e-3.h5; E=/results/fxr/energy; mkdir -p $E
T0=$(date +%s); p() { echo "[g12] $1 at $(( $(date +%s) - T0 )) s" | tee -a $E/progress_$STEP.txt; }
GRID="--eq $EQ --bext desc-fit --LMN 40 40 6 --frac 100 --N-phi 25 --colloc-rho 2.7"
FLAGS="--subspace prev,accel --cg 1500 --no-rebuild-on-hit --f-noise auto --radius-by-delta --no-share-stop \
       --locate-chunk 131072 --diag-every 5 --save-every 5"
CONS="--bn-modes 40 6 --pin-flux"
SOLVE="$GRID $FLAGS $CONS --alphas 0.05 --max-outer 8 --maxiter 60 --energy-weight 0 --time-budget 3300"
endstate() { ls -t $E/$1/${1}_a0.05.npz $E/$1/${1}_budget.npz $E/$1/${1}_a0.05_latest.npz 2>/dev/null | head -1; }
case $STEP in
  g1a|g1a80)
    BN="40 6"; [ $STEP = g1a80 ] && BN="80 12"
    p "begin (B.n modes $BN)"
    bash rung_run.sh en_$STEP scripts/energy/exterior_check.py $GRID --bn-modes $BN --direct 32 4 --n-random 2 \
        --state R1f_S=/work/R1f_S_fresh.npz --state R1f_b1=/work/R1f_b1_fresh.npz \
        --pair R1f_island=/work/R1f_S_fresh.npz,/work/R1f_b1_fresh.npz --out $E/$STEP/$STEP
    ;;
  S0)
    p "begin"; mkdir -p $E/S0
    bash rung_run.sh en_S0 scripts/meanforce/mf_fixed.py $SOLVE --start /work/R1f_S_fresh.npz \
        --out /work/out/S0 --mirror $E/S0
    ;;
  I*)
    BETA=${STEP#I}; p "begin beta $BETA"; mkdir -p $E/$STEP
    bash rung_run.sh en_$STEP scripts/meanforce/mf_fixed.py $SOLVE --start /work/R1f_b1_fresh.npz \
        --pin-b21 $BETA --b21-phase-state /work/R1f_b1_fresh.npz --out /work/out/$STEP --mirror $E/$STEP
    ;;
  g1b)
    p "begin"; mkdir -p $E/g1b
    ST=""; PR=""; EV=""
    S=$(endstate S0)
    for T in S0 I0.25 I0.5 I1; do F=$(endstate $T); [ -n "$F" ] || continue
      ST="$ST --state $T=$F"; EV="$EV --eval-state $T=$F"
      [ "$T" != S0 ] && [ -n "$S" ] && PR="$PR --pair S0_to_$T=$S,$F"
    done
    p "states:$ST"
    bash rung_run.sh en_g1b scripts/energy/exterior_check.py $GRID --bn-modes 40 6 --direct 32 4 --n-random 0 \
        $ST $PR --out $E/g1b/g1b
    p "fine-grid energy evaluation"
    bash rung_run.sh en_g1b_ev120 scripts/meanforce/mf_fixed.py --eq $EQ --bext desc-fit --LMN 40 40 6 --N-phi 25 \
        --frac 120 --colloc-rho 3.3 --locate-chunk 131072 --bn-modes 40 6 --alphas 1 --model beta --force-norm p \
        --p-schedule 16 --p-band 0.5675 0.7075 --eval-only --b21-report $EV --out /work/out/g1b_ev120 --mirror $E/g1b
    ;;
  *) echo "unknown step $STEP"; exit 2;;
esac
p "done"; cp $E/progress_$STEP.txt out/ 2>/dev/null; true
