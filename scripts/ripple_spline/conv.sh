# Long convergence runs (item 1): bash conv.sh MODE [MODE ...], each sequentially, alpha BETA_ALPHA (unmet bound),
# up to MAXO outer iterations or BUDGET_CONV seconds, per-outer snapshots:
#   S   unpinned from the shielded start          R05 / R10  unpinned from the beta 0.05 / 0.1 state
#   P0  pinned b21 = 0 (no island-phase field)   P05        pinned beta 0.05
set -u; source scripts/ripple_spline/env.sh; source scripts/ripple_spline/cfg.sh
for MODE0 in "$@"; do
  PIN=""
  MODE=${MODE0#x}  # xMODE: continue conv_MODE from its end state (output conv_xMODE)
  case $MODE in
    S) STA=$O/start.npz ;; R05) STA=$O/b0.05.npz ;; R10) STA=$O/b0.1.npz ;;
    P0) STA=$O/start.npz; PIN="--pin-b21 0 --b21-phase-state $O/seed_all.npz" ;;
    P05) STA=$O/b0.05.npz; PIN="--pin-b21 0.05 --b21-phase-state $O/seed_all.npz" ;;
    *) echo "unknown mode $MODE"; continue ;;
  esac
  [ "$MODE0" != "$MODE" ] && STA=$O/conv_$MODE.npz
  $PY scripts/meanforce/mf_fixed.py $RUN --alphas ${BETA_ALPHA:-0.02} --start $STA $PIN --max-outer ${MAXO:-60} \
      --stage-flat 0.02 $SOLVER --rho-max ${RHOMAX:-1e8} --snap-outer --time-budget ${BUDGET_CONV:-43200} --out $O/conv_$MODE0
  S=$(pick $O/conv_$MODE0 ${BETA_ALPHA:-0.02}); echo "conv $MODE0 state: $S"; [ -n "$S" ] && cp "$S" $O/conv_$MODE0.npz
done
