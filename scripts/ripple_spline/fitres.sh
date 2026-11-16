# Standard (T-potential) spline fit at much higher spline resolution: does its perpendicular error and start force
# go down? J_fit diagnostics only (--fit-split, eval-only), pure splines, same layout as s1 with finer knots.
#   bash fitres.sh NAME KC HB2 HB3 NRHO
set -u; source scripts/ripple_spline/env.sh
O=out/fitsplit; mkdir -p $O
B0="--eq $EQ --bext desc-fit --frac 100 --N-phi 25 --alphas 1 --model beta --force-norm p --p-schedule 16 --p-band 0.5675 0.7075 --eval-only --fit-split"
while [ $# -ge 5 ]; do
  n=$1; K="--knots-fz 32 $2 --spl-band 0.45 0.76 $3 --spl-band 0.86 0.95 $4 --spl-reg rhom --spl-qcap 6 --spl-mfz 32"
  echo "=== $n: KC $2, bands $3 / $4, fit nodes $5"
  $PY scripts/meanforce/mf_fixed.py $B0 --LMN 32 32 6 $K --current spline --pressure spline --jfit-nrho $5 --out $O/res_$n 2>&1 \
      | grep -E "current\] spline|pressure\] spline|collocation:|fit split|Traceback|Error|out of memory"
  shift 5
done
echo "fitres done"
