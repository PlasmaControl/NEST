# The convergence and localized-bound states of a configuration on the common grid (a/120): band p16, mean, b21,
# force split by rho bin. bash eval_extra.sh (TAG, KIND, LMN ... from the environment)
set -u; source scripts/ripple_spline/env.sh; source scripts/ripple_spline/cfg.sh
C=out/rs/common; mkdir -p $C
G="--eq $EQ --bext desc-fit --frac 120 --N-phi 25 $BAS $EVAL --rho-bins 0 .4 .5 .55 .6 .65 .7 .75 .8 .85 .9 .95 1"
ES="--eval-state start=$O/start.npz --eval-state b0.05=$O/b0.05.npz"
for f in $O/conv_S.npz $O/conv_R05.npz $O/conv_R10.npz $O/conv_P0.npz $O/conv_P05.npz $O/conv_xS.npz $O/conv_xR05.npz $O/conv_xP0.npz $O/conv_xP05.npz $O/loc_S.npz $O/loc_P05.npz $O/loc_P5.npz; do
  [ -f $f ] && ES="$ES --eval-state $(basename $f .npz)=$f"; done
$PY scripts/meanforce/mf_fixed.py $G $ES --shell-max 0.55 0.72 --out $C/${TAG}_extra
