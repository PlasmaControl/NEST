# Where does the s1 core force (rho < 0.5) come from? Eval-only at J_fit and at s1's continuation state:
# (1) spline T with FZ p, (2) spline T and p without the m_fz truncation, (3) q_cap 10.
set -u; source scripts/ripple_spline/env.sh; export TAG=s1 KC=0.6 HB2=0.015 HB3=0.02 FRAC=100 SPLX="--spl-mfz 32"
source scripts/ripple_spline/cfg.sh
S=$O/cont2_a0.3_latest.npz
B="--rho-bins 0 .2 .3 .4 .45 .5 .55 .6 .7 .8 .9 1"
$PY scripts/meanforce/mf_fixed.py --eq $EQ --bext desc-fit --frac $FRAC --N-phi 25 --current spline --pressure fz --LMN 32 32 6 \
    $KNOTS --spl-mfz 32 --colloc-knots $EVAL --eval-state s1=$S $B --out $O/core_fzp
$PY scripts/meanforce/mf_fixed.py --eq $EQ --bext desc-fit --frac $FRAC --N-phi 25 --current spline --pressure spline --LMN 32 32 6 \
    $KNOTS $EVAL $B --out $O/core_nomfz
$PY scripts/meanforce/mf_fixed.py --eq $EQ --bext desc-fit --frac $FRAC --N-phi 25 --current spline --pressure spline --LMN 32 32 6 \
    ${KNOTS/--spl-qcap 6/--spl-qcap 10} --spl-mfz 32 $EVAL $B --out $O/core_q10
