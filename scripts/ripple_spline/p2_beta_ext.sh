# Extend a pinned-beta run to the FZ runs' stopping rule (12 outer iterations in all): restart from its end state
# (the same pin, the saved multiplier and penalty) with the remaining outers. The first run is kept as b<beta>_first.npz.
set -u; source scripts/ripple_spline/env.sh; source scripts/ripple_spline/cfg.sh
B=$1
DONE=$($PY -c "import json; d=json.load(open('$O/bs$B.json')); print(sum(len(a.get('outer', [])) for a in d['alphas']))")
MO=$((12 - DONE)); echo "beta $B: $DONE outers done, extending by $MO"
[ $MO -lt 1 ] && exit 0
cp $O/b$B.npz $O/b${B}_first.npz
$PY scripts/meanforce/mf_fixed.py $RUN --alphas ${BETA_ALPHA:-0.1} --start $O/b${B}_first.npz --pin-b21 $B \
    --b21-phase-state $O/seed_all.npz --max-outer $MO --stage-flat 0.02 $SOLVER --time-budget ${BUDGET_EXT:-12600} --out $O/bsx$B
S=$(pick $O/bsx$B ${BETA_ALPHA:-0.1}); echo "beta $B extended state: $S"; cp "$S" $O/b$B.npz
