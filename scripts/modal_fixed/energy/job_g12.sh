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
# Inputs (flat in /work): nest_src.tar.gz rung_run.sh job_g12.sh R1f_S_fresh.npz R1f_b1_fresh.npz [S0_fresh.npz]
# Env: ALPHA (default 0.05). Step E<lambda>: energy weight lambda from S0a08_fresh.npz. S0 starts from S0_fresh.npz when it is staged.
# (desc_src, deps_src, gpu_entry.py and the DESC file come from /results/stage, as for the ripple runs).
# Outputs: /results/fxr/energy/<STEP>/ (mirror, fsynced per step) and out/.
set -u
STEP=$1
ALPHA=${ALPHA:-0.05}  # force bound (x M_DESC); outputs are tagged <STEP>_a<ALPHA> when ALPHA != 0.05
TAG=$STEP; [ "$ALPHA" != 0.05 ] && TAG=${STEP}_a$ALPHA
cd /work; for f in desc_src.tar.gz deps_src.tar.gz ripple_L32N4_x1e-3.h5 gpu_entry.py; do [ -f $f ] || cp /results/stage/$f .; done
rm -rf src/scripts src/nest; mkdir -p src out
tar -xzf nest_src.tar.gz -C src
if [ ! -f src/.deps ]; then tar -xzf desc_src.tar.gz -C src && tar -xzf deps_src.tar.gz -C src && touch src/.deps; fi
touch src/.unpacked
EQ=/work/ripple_L32N4_x1e-3.h5; E=/results/fxr/energy; mkdir -p $E
T0=$(date +%s); p() { echo "[g12] $1 at $(( $(date +%s) - T0 )) s" | tee -a $E/progress_$TAG.txt; }
GRID="--eq $EQ --bext desc-fit --LMN 40 40 6 --frac 100 --N-phi 25 --colloc-rho 2.7"
FLAGS="--subspace prev,accel --cg 1500 --no-rebuild-on-hit --f-noise auto --radius-by-delta --no-share-stop \
       --locate-chunk 131072 --diag-every 5 --save-every 5"
CONS="--bn-modes 40 6 --pin-flux"
SOLVE="$GRID $FLAGS $CONS --alphas $ALPHA --max-outer 8 --maxiter 60 --energy-weight 0 --time-budget 3300"
endstate() { ls -t $E/$1/${1}_a$ALPHA.npz $E/$1/${1}_budget.npz $E/$1/${1}_a${ALPHA}_latest.npz 2>/dev/null | head -1; }
case $STEP in
  g1a|g1a80)
    BN="40 6"; [ $STEP = g1a80 ] && BN="80 12"
    p "begin (B.n modes $BN)"
    bash rung_run.sh en_$STEP scripts/energy/exterior_check.py $GRID --bn-modes $BN --direct 32 4 --n-random 2 \
        --state R1f_S=/work/R1f_S_fresh.npz --state R1f_b1=/work/R1f_b1_fresh.npz \
        --pair R1f_island=/work/R1f_S_fresh.npz,/work/R1f_b1_fresh.npz --out $E/$STEP/$STEP
    ;;
  S0)
    ST0=/work/R1f_S_fresh.npz; [ -f /work/S0_fresh.npz ] && ST0=/work/S0_fresh.npz  # the constrained S0 (alpha 0.05 floor)
    p "begin (alpha $ALPHA, start $ST0)"; mkdir -p $E/$TAG
    bash rung_run.sh en_$TAG scripts/meanforce/mf_fixed.py $SOLVE --start $ST0 \
        --out /work/out/$TAG --mirror $E/$TAG
    ;;
  E*)
    # energy ladder: lambda = ${STEP#E} from the shielded alpha 0.8 state (S0a08_fresh.npz), at the ALPHA force bound
    LAM=${STEP#E}; p "begin lambda $LAM (alpha $ALPHA, start /work/S0a08_fresh.npz)"; mkdir -p $E/$TAG
    bash rung_run.sh en_$TAG scripts/meanforce/mf_fixed.py $SOLVE --energy-weight $LAM --start /work/S0a08_fresh.npz \
        --out /work/out/$TAG --mirror $E/$TAG
    ;;
  ladtab)
    # analysis of the energy ladder: DESC-flux-grid tables (field-line tracing, profiles) + LCFS-leak / shell dumps of the
    # shielded alpha-0.8 start and the E<lambda>_a1.5 end states
    p "begin"; D=$E/ladtab; mkdir -p $D
    last() { ls -t "$@" 2>/dev/null | head -1; }
    ST="--state S08=/work/S0a08_fresh.npz"
    for L in 1 10 100; do T=E${L}_a1.5; F=$(last $E/$T/${T}_a1.5.npz $E/$T/${T}_budget.npz $E/$T/${T}_a1.5_latest.npz); [ -n "$F" ] && ST="$ST --state E$L=$F"; done
    echo "states: $ST" | tee -a $E/progress_ladtab.txt
    bash rung_run.sh ladtab_bn scripts/energy/bn_spectrum.py $GRID --direct 32 4 $ST --dump --out $D/lad
    bash rung_run.sh ladtab_tab scripts/meanforce/state_tables.py --eq $EQ --LMN 40 40 6 --frac 100 --N-phi 25 $ST --out /work/out/tables
    cp $D/lad_* out/ 2>/dev/null; mkdir -p $D/tables; cp /work/out/tables/* $D/tables/ 2>/dev/null
    ;;
  I*)
    BETA=${STEP#I}; p "begin beta $BETA (alpha $ALPHA)"; mkdir -p $E/$TAG
    bash rung_run.sh en_$TAG scripts/meanforce/mf_fixed.py $SOLVE --start /work/R1f_b1_fresh.npz \
        --pin-b21 $BETA --b21-phase-state /work/R1f_b1_fresh.npz --out /work/out/$TAG --mirror $E/$TAG
    ;;
  g1b)
    p "begin"; mkdir -p $E/g1b
    ST=""; PR=""; EV=""
    SF=""; [ "$ALPHA" != 0.05 ] && SF=_a$ALPHA
    S=$(endstate S0$SF)
    for T0_ in S0 I0.25 I0.5 I1; do T=$T0_$SF; F=$(endstate $T); [ -n "$F" ] || continue
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
p "done"; cp $E/progress_$TAG.txt out/ 2>/dev/null; true
