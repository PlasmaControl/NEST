# Phase 0b: the FZ states' least-squares p re-solved in spline p spaces with finer outer knots (FZ spacing x C),
# common grid of each, force split. Does a richer p space lower |b.grad p| at fixed J (and keep perp at FZ level)?
set -u
source scripts/ripple_spline/env.sh
O=out/p0; mkdir -p $O
S="--eval-state start=$ST/shielded_start_A60_a0.1.npz --eval-state b0.2=$ST/beta_0.2.npz --eval-state b1.0=$ST/beta_1.0.npz"
B="--rho-bins 0 .4 .5 .55 .6 .65 .7 .75 .8 .85 .9 .95 1"
for C in 0.6 0.45; do
  K="--knots-fz 32 $C --spl-band 0.45 0.76 0.015 --spl-band 0.86 0.95 0.02 --spl-reg rhom --spl-qcap 6"
  $PY scripts/meanforce/mf_fixed.py $BASE --LMN 32 32 6 $EVAL $S $B $K --colloc-knots --out $O/fzp_c$C
  $PY scripts/meanforce/mf_fixed.py $BASE --LMN 32 32 6 $EVAL $S $B $K --colloc-knots --pressure spline --out $O/splp_c$C
  $PY scripts/meanforce/mf_fixed.py $BASE --LMN 32 32 6 $EVAL $S $B $K --colloc-knots --pressure spline \
      --spl-mtol 0.01 --spl-mmax 16 --out $O/splp_c${C}_t
done
echo "phase0b done"
