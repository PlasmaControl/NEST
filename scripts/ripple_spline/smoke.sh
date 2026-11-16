# Smoke of the spline Phase 2 path at a/40 (minutes): seeds, a 2-trial continuation, a 2-trial pinned beta = 0.5
# run, the live log viewer. TAG=smoke.
set -u; source scripts/ripple_spline/env.sh; export TAG=smoke FRAC=40; source scripts/ripple_spline/cfg.sh
$PY scripts/meanforce/island_seed.py --eq $EQ --LMN 32 32 6 --frac $FRAC --N-phi 25 $KNOTS $SPLX --current spline \
    --p-basis spline --window 0.6375 0.07 0.03 --window -1 0 0 --cg 100 --out $O/seed || exit 1
$PY scripts/meanforce/mf_fixed.py $RUN --alphas 0.3 --max-outer 1 --maxiter 2 $SOLVER --out $O/cont || exit 1
S=$(pick $O/cont 0.3); echo "state $S"
$PY scripts/meanforce/mf_fixed.py $RUN --alphas 0.1 --start $S --pin-b21 0.5 --b21-phase-state $O/seed_all.npz \
    --max-outer 1 --maxiter 2 $SOLVER --out $O/bs0.5 || exit 1
$PY scripts/meanforce/tr_live.py $O/bs0.5_live.tsv -n 5; cat $O/bs0.5_status.json | head -40
$PY scripts/meanforce/tr_live.py $O
echo "smoke ok"
