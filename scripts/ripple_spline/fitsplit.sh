# Is the J_fit force error a non-gradient remainder of the current-fit error? At J_fit (option A: B = B_DESC):
# dG = (J_fit - J_DESC) x B_DESC = grad dp + R, R what the run's pressure space cannot absorb (--fit-split).
set -u; source scripts/ripple_spline/env.sh
O=out/fitsplit; mkdir -p $O
B0="--eq $EQ --bext desc-fit --frac 100 --N-phi 25 --alphas 1 --model beta --force-norm p --p-schedule 16 --p-band 0.5675 0.7075 --eval-only --fit-split"
K1="--knots-fz 32 0.6 --spl-band 0.45 0.76 0.015 --spl-band 0.86 0.95 0.02 --spl-reg rhom --spl-qcap 6 --spl-mfz 32"
K2="--knots-fz 32 0.45 --spl-band 0.45 0.76 0.01 --spl-band 0.86 0.95 0.015 --spl-reg rhom --spl-qcap 6 --spl-mfz 32"
HB="--spl-band 0.45 0.76 0.01 --spl-band 0.86 0.95 0.015 --spl-mmax 16"
run() { n=$1; shift; echo "=== $n"; $PY scripts/meanforce/mf_fixed.py $B0 "$@" --out $O/$n 2>&1 | grep -E "fit split|Traceback|Error|^start:"; }
run fzT_fzp     --LMN 32 32 6 $K1 --colloc-knots
run s1T_s1p     --LMN 32 32 6 $K1 --current spline --pressure spline
run s2T_s2p     --LMN 32 32 6 $K2 --current spline --pressure spline
run s1T_fzp     --LMN 32 32 6 $K1 --current spline
run fzT_s1p     --LMN 32 32 6 $K1 --pressure spline
run hybT_hybp   --LMN 32 32 6 $HB --current fz+spline --pressure fz+spline
echo "fitsplit done"
