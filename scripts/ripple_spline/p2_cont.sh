# Shielded start. Pure splines: alpha continuation from J_fit. Hybrid / FZ control: from the report's FZ shielded
# start (START, embedded with START_LMN=1), alpha 0.1 at this grid. One budget (BUDGET_CONT).
set -u; source scripts/ripple_spline/env.sh; source scripts/ripple_spline/cfg.sh
ST0=""; [ -n "${START:-}" ] && ST0="--start $START"
[ -n "${START_LMN:-}" ] && ST0="$ST0 --start-LMN 32 32 6"
$PY scripts/meanforce/mf_fixed.py $RUN $ST0 --alphas ${ALPHAS:-0.3 0.1 0.03 0.01} --max-outer 12 --stage-flat 0.02 --stall-stop 1e-3 \
    $SOLVER ${CONT_EXTRA:-} --snap-outer --time-budget ${BUDGET_CONT:-13300} --out $O/${CONT:-cont}
S=""; for al in ${ALPHAS:-0.3 0.1 0.03 0.01}; do s_=$(pick $O/${CONT:-cont} $al); [ -n "$s_" ] && S=$s_; done
[ -f $O/${CONT:-cont}_budget.npz ] && S=$O/${CONT:-cont}_budget.npz
echo "shielded start: $S"; cp "$S" $O/start.npz
