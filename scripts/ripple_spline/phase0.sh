# Phase 0: is the pressure basis what limits b.grad p? Re-solve only the least-squares p of the Fourier-Zernike
# states (same J) in the spline basis, on one common radial grid (composite Gauss over the spline breakpoints),
# and split the force into |b.F| = |b.grad p| and the perpendicular part.
set -u
source scripts/ripple_spline/env.sh
O=out/p0; mkdir -p $O
S="--eval-state start=$ST/shielded_start_A60_a0.1.npz --eval-state b0.05=$ST/beta_0.05.npz --eval-state b0.2=$ST/beta_0.2.npz \
   --eval-state b0.5=$ST/beta_0.5.npz --eval-state b1.0=$ST/beta_1.0.npz --eval-state rel0.05=$ST/release_from_beta0.05.npz \
   --eval-state rel0.5=$ST/release_from_beta0.5.npz"
B="--rho-bins 0 .4 .5 .55 .6 .65 .7 .75 .8 .85 .9 .95 1"
# (0) the report's grid, FZ p: reproduces Table 1 and splits it
$PY scripts/meanforce/mf_fixed.py $BASE --LMN 32 32 6 $EVAL $S $B --out $O/fzp_report
# (1) common grid, FZ p
$PY scripts/meanforce/mf_fixed.py $BASE --LMN 32 32 6 $EVAL $S $B $KNOTS --colloc-knots --out $O/fzp_common
# (2) common grid, spline p (rhom, bands), full M
$PY scripts/meanforce/mf_fixed.py $BASE --LMN 32 32 6 $EVAL $S $B $KNOTS --colloc-knots --pressure spline --out $O/splp_common
# (3) as (2) with the FZ-reach truncation of p (the run setting)
$PY scripts/meanforce/mf_fixed.py $BASE --LMN 32 32 6 $EVAL $S $B $KNOTS --colloc-knots --pressure spline --spl-mfz 32 --out $O/splp_mfz32
echo "phase0 done"
