# Common-grid comparison: this run's states at field grid a/120 (the run's own radial collocation: the hybrid band
# breakpoints for hybrid and fz). KIND=fz also evaluates the report's FZ states (a/60 runs) on that grid.
set -u; source scripts/ripple_spline/env.sh; source scripts/ripple_spline/cfg.sh
C=out/rs/common; mkdir -p $C
G="--eq $EQ --bext desc-fit --frac 120 --N-phi 25 $BAS $EVAL --rho-bins 0 .4 .5 .55 .6 .65 .7 .75 .8 .85 .9 .95 1"
ES="--eval-state start=$O/start.npz"
for B in 0.02 0.05 0.1 0.2 0.35 0.5 0.75 1.0; do [ -f $O/b$B.npz ] && ES="$ES --eval-state b$B=$O/b$B.npz"; done
for B in 0.05 0.5; do [ -f $O/release$B.npz ] && ES="$ES --eval-state rel$B=$O/release$B.npz"; done
$PY scripts/meanforce/mf_fixed.py $G $ES --out $C/$TAG
if [ $KIND = fz ] && [ "$LMN" = "32 32 6" ]; then  # the report states are 32/32/6
  EF="--eval-state start=$ST/shielded_start_A60_a0.1.npz"
  for B in 0.02 0.05 0.1 0.2 0.35 0.5 0.75 1.0; do EF="$EF --eval-state b$B=$ST/beta_$B.npz"; done
  EF="$EF --eval-state rel0.05=$ST/release_from_beta0.05.npz --eval-state rel0.5=$ST/release_from_beta0.5.npz"
  $PY scripts/meanforce/mf_fixed.py $G $EF --out $C/report_fz
fi
