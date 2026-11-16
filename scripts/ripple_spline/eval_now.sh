# Quick eval of a config's current continuation state(s) (force split by rho bin, b21, band p16): TAG=s1 bash eval_now.sh
set -u; source scripts/ripple_spline/env.sh; source scripts/ripple_spline/cfg.sh
EV=""; for f in $O/cont1_resume.npz $O/${CONT:-cont2}_a0.3_latest.npz $O/${CONT:-cont2}_a0.1_latest.npz; do [ -f $f ] && EV="$EV --eval-state $(basename $f .npz)=$f"; done
$PY scripts/meanforce/mf_fixed.py $RUN $EVAL $EV --rho-bins 0 .4 .5 .55 .6 .65 .7 .75 .8 .85 .9 .95 1 --out $O/evalnow
