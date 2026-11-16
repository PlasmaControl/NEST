# Hybrid gates, then the report's FZ states on one grid: FZ T and p vs FZ T + hybrid p vs hybrid T + hybrid p
# (FZ states embedded: same J; the hybrid p space contains FZ's, so M can only fall). Force split by rho bin.
set -u; source scripts/ripple_spline/env.sh
$PY tests/test_hybrid_gates.py --mmax ${HMMAX:-16}
O=out/p0; mkdir -p $O
S="--eval-state start=$ST/shielded_start_A60_a0.1.npz --eval-state b0.2=$ST/beta_0.2.npz --eval-state b0.5=$ST/beta_0.5.npz \
   --eval-state b1.0=$ST/beta_1.0.npz --eval-state rel0.05=$ST/release_from_beta0.05.npz"
B="--rho-bins 0 .4 .5 .55 .6 .65 .7 .75 .8 .85 .9 .95 1"
$PY scripts/meanforce/mf_fixed.py $BASE --LMN 32 32 6 $EVAL $S $B $HYB --colloc-knots --out $O/hyb_fz
$PY scripts/meanforce/mf_fixed.py $BASE --LMN 32 32 6 $EVAL $S $B $HYB --pressure fz+spline --out $O/hyb_p
$PY scripts/meanforce/mf_fixed.py $BASE --LMN 32 32 6 $EVAL $S $B $HYB --current fz+spline --pressure fz+spline \
    --start-LMN 32 32 6 --out $O/hyb_tp
echo "hyb0 done"
