# One pinned-b21 run (beta = $1) from the spline shielded start, alpha 0.1 mean bound, like job_bs*.sh.
set -u; source scripts/ripple_spline/env.sh; source scripts/ripple_spline/cfg.sh
B=$1
$PY scripts/meanforce/mf_fixed.py $RUN --alphas ${BETA_ALPHA:-0.1} --start $O/start.npz --pin-b21 $B --b21-phase-state $O/seed_all.npz \
    --max-outer 12 --stage-flat 0.02 $SOLVER --time-budget ${BUDGET_BETA:-1800} --out $O/bs$B
S=$(pick $O/bs$B ${BETA_ALPHA:-0.1}); echo "beta $B state: $S"; cp "$S" $O/b$B.npz
