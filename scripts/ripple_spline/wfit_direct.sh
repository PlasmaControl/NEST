# Does the current-fitting objective cause the spline fit's perpendicular error? Weighted re-fit of J at the
# collocation points, min |dJ_par|^2 + lambda |dJ_perp|^2 (lambda = 1: the plain pointwise J fit), for pure splines
# s1 / s2 and FZ (control), with the fit split of each and the start force it would give.
set -u; source scripts/ripple_spline/env.sh
O=out/fitsplit; mkdir -p $O
B0="--eq $EQ --bext desc-fit --frac 100 --N-phi 25 --alphas 1 --model beta --force-norm p --p-schedule 16 --p-band 0.5675 0.7075 --eval-only --fit-split"
K1="--knots-fz 32 0.6 --spl-band 0.45 0.76 0.015 --spl-band 0.86 0.95 0.02 --spl-reg rhom --spl-qcap 6 --spl-mfz 32"
K2="--knots-fz 32 0.45 --spl-band 0.45 0.76 0.01 --spl-band 0.86 0.95 0.015 --spl-reg rhom --spl-qcap 6 --spl-mfz 32"
run() { n=$1; shift; echo "=== $n"; $PY scripts/meanforce/mf_fixed.py $B0 "$@" --out $O/wfitd_$n 2>&1 | grep -E "fit split\] (B at|J_DESC itself|dG =|R \(not)|wfit|Traceback|Error"; }
run s1 --LMN 32 32 6 $K1 --current spline --pressure spline --wfit 1 100 10000 --wfit-direct
run fz --LMN 32 32 6 $K1 --colloc-knots --wfit 1 100 --wfit-direct
run s2 --LMN 32 32 6 $K2 --current spline --pressure spline --wfit 1 100 10000 --wfit-direct
echo "wfit direct done"
