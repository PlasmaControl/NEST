# Diagnostics on the run grid: b21, band p16, force split of every state; plane maps through the shielded start
# (the report's three planes); Poincare + pressure along lines (start_check_bext) of start and beta states.
set -u; source scripts/ripple_spline/env.sh; source scripts/ripple_spline/cfg.sh
S=$O/start.npz
EV="--eval-state start=$S"
for B in 0.02 0.05 0.1 0.2 0.35 0.5 0.75 1.0; do [ -f $O/b$B.npz ] && EV="$EV --eval-state b$B=$O/b$B.npz"; done
for f in $(ls $O/rel*_o*.npz $O/release*.npz 2>/dev/null); do EV="$EV --eval-state $(basename $f .npz)=$f"; done
$PY scripts/meanforce/mf_fixed.py $RUN $EVAL $EV --rho-bins 0 .4 .5 .55 .6 .65 .7 .75 .8 .85 .9 .95 1 \
    --shell-max 0.30 0.995 --out $O/eval
[ -f $O/release0.05.npz ] && VAL=$O/release0.05.npz || VAL=$O/cont_a0.1_latest.npz  # valley direction (report: the pushed state)
$PY scripts/meanforce/mf_fixed.py $RUN --alphas 1 --model beta --force-norm p --p-schedule 16 --p-band 0.5675 0.7075 \
    --eval-only --b21-report --eval-state start=$S --eval-state seed_q2=$O/seed_r0.6375_w0.07.npz \
    --eval-state seed_q3=$O/seed_r0.9131_w0.03.npz --eval-state seed_all=$O/seed_all.npz \
    --plane "isl2_valley|$S|$O/seed_r0.6375_w0.07.npz|J_fit|$VAL|$S|-0.2,1.2,29|-3,6,37" \
    --plane "isl2_isl3|$S|$O/seed_r0.6375_w0.07.npz|J_fit|$O/seed_r0.9131_w0.03.npz|J_fit|-0.2,1.2,29|-0.2,1.2,29" \
    --plane "isl2_jfit|$S|$O/seed_r0.6375_w0.07.npz|J_fit|J_fit|$S|-0.2,1.2,29|-1,1.5,26" --out $O/planes
ST="--state $S"; PT="vacuum start:A"
for B in 0.05 0.2 0.5 1.0; do [ -f $O/b$B.npz ] && { ST="$ST --state $O/b$B.npz"; PT="$PT b$B:A"; }; done
$PY scripts/meanforce/start_check_bext.py --eq $EQ --LMN $LMN --frac $FRAC --N-phi 25 $SEEDB --m 2 --k 1 --rho-grid 0.30 0.98 341 --band 0.04 0.08 0.16 $ST --perts $PT --turns 300 --lines 41 \
    --seed-band 0.50 0.78 --seed-theta 0.0 --pressure --out $O/diag_m2
