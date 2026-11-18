#!/bin/bash
# B.n-leak study (docs/energy_objective_status_2026-11-18.md, section 5): how much of the force reduction of the
# fixed-boundary NEST states depends on magnetic flux crossing the LCFS. One step per container.
# Setting = R1f / S0: 40/40/6, field grid a/100, 98 GL radii, option A (--bext desc-fit), lambda = 0.
#
# usage: bash job_leak.sh STEP
#   spec    test 3: (m, n) spectrum of the LCFS flux leak of R1f_S, R1f_b1 and S0, and the energy error per mode
#   char    LCFS f dumps + per-shell force/current profiles of all states (bn_spectrum --dump)
#   A0      test 1: force-floor push from J_fit, no constraint
#   Aflux   test 1: the same with --pin-flux only
#   A40     test 1: --pin-flux + B.n modes |m| <= 40, |n| <= 6   (S0's constraint, but from J_fit)
#   A4 / A8 / A16   test 2: --pin-flux + B.n cutoffs 4/2, 8/4, 16/6
# Every arm: same start (J_fit satisfies every constraint exactly, so no projection), same flags, alpha 0.001
# (bound out of reach: the run minimizes force), one outer, 3300 s budget from script start (setup included).
# Inputs (flat in /work): nest_src.tar.gz rung_run.sh job_leak.sh R1f_S_fresh.npz R1f_b1_fresh.npz
# Outputs: /results/fxr/energy/leak/<STEP>/ (fsynced mirror), logs /results/fxr/lk_<STEP>/, and out/.
set -u
STEP=$1
cd /work; for f in desc_src.tar.gz deps_src.tar.gz ripple_L32N4_x1e-3.h5 gpu_entry.py; do [ -f $f ] || cp /results/stage/$f .; done
rm -rf src/scripts src/nest; mkdir -p src out
tar -xzf nest_src.tar.gz -C src
if [ ! -f src/.deps ]; then tar -xzf desc_src.tar.gz -C src && tar -xzf deps_src.tar.gz -C src && touch src/.deps; fi
touch src/.unpacked
EQ=/work/ripple_L32N4_x1e-3.h5; E=/results/fxr/energy/leak; mkdir -p $E/$STEP
T0=$(date +%s); p() { echo "[leak] $1 at $(( $(date +%s) - T0 )) s" | tee -a $E/progress_$STEP.txt; }
GRID="--eq $EQ --bext desc-fit --LMN 40 40 6 --frac 100 --N-phi 25 --colloc-rho 2.7"
FLAGS="--subspace prev,accel --cg 1500 --no-rebuild-on-hit --f-noise auto --radius-by-delta --no-share-stop \
       --locate-chunk 131072 --diag-every 5 --save-every 5"
FLOOR="$GRID $FLAGS --alphas 0.001 --max-outer 1 --maxiter 100000 --energy-weight 0 --time-budget 3300"
case $STEP in
  spec)
    p "begin"
    S0=/results/fxr/energy/S0/S0_budget.npz
    ST="--state R1f_S=/work/R1f_S_fresh.npz --state R1f_b1=/work/R1f_b1_fresh.npz"
    [ -f $S0 ] && ST="$ST --state S0=$S0"
    bash rung_run.sh lk_spec scripts/energy/bn_spectrum.py $GRID --direct 32 4 $ST --out $E/spec/spec
    cp $E/spec/spec_spec.json out/ 2>/dev/null
    ;;
  char)
    # B.n-leak characterization: LCFS f dumps (both routes) + per-shell |F|, |J - J_fit| profiles of every state
    p "begin"; mkdir -p $E/char
    last() { ls -t "$@" 2>/dev/null | head -1; }
    ST="--state Jfit=0 --state R1f_S=/work/R1f_S_fresh.npz --state R1f_b1=/work/R1f_b1_fresh.npz"
    for X in A0 Aflux A4 A8 A16 A40; do F=$(last $E/$X/${X}_a0.001.npz $E/$X/${X}_budget.npz $E/$X/${X}_a0.001_latest.npz); [ -n "$F" ] && ST="$ST --state $X=$F"; done
    F=$(last /results/fxr/energy/S0/S0_budget.npz); [ -n "$F" ] && ST="$ST --state S0=$F"
    for X in S0 I0.25 I0.5 I1; do T=${X}_a0.8; D=/results/fxr/energy/$T
      F=$(last $D/${T}_a0.8.npz $D/${T}_budget.npz $D/${T}_a0.8_latest.npz); [ -n "$F" ] && ST="$ST --state $T=$F"; done
    echo "states: $ST" | tee -a $E/progress_char.txt
    bash rung_run.sh lk_char scripts/energy/bn_spectrum.py $GRID --direct 32 4 $ST --dump --out $E/char/char
    cp $E/char/char_spec.json out/ 2>/dev/null; tar -C $E/char -czf out/char_dumps.tar.gz . 2>/dev/null
    ;;
  A0)    CONS="";;
  Aflux) CONS="--pin-flux";;
  A40)   CONS="--pin-flux --bn-modes 40 6";;
  A16)   CONS="--pin-flux --bn-modes 16 6";;
  A8)    CONS="--pin-flux --bn-modes 8 4";;
  A4)    CONS="--pin-flux --bn-modes 4 2";;
  *) echo "unknown step $STEP"; exit 2;;
esac
if [ $STEP != spec ] && [ $STEP != char ]; then
  p "begin ($CONS)"
  bash rung_run.sh lk_$STEP scripts/meanforce/mf_fixed.py $FLOOR $CONS --out /work/out/$STEP --mirror $E/$STEP
fi
p "done"; cp $E/progress_$STEP.txt out/ 2>/dev/null; true
