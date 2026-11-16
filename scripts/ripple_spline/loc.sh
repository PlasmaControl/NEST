# Localized force bound (item 4): the band p16 of |F| over the q = 2 band (rho 0.5675-0.7075; a smooth maximum) bounded
# at alpha x DESC's band p16, alpha continued down LOC_ALPHAS until it stalls, together with the whole-volume mean
# <= LOC_EA x M_DESC (composite constraint, --mean-whole) so force cannot leave the band for the rest of the volume.
#   bash loc.sh S P05 P5   (S: unpinned from the shielded start; P05 / P5: pinned beta 0.05 / 0.5 from those states)
set -u; source scripts/ripple_spline/env.sh; source scripts/ripple_spline/cfg.sh
for MODE in "$@"; do
  PIN=""
  case $MODE in
    S) STA=$O/start.npz ;;
    P05) STA=$O/b0.05.npz; PIN="--pin-b21 0.05 --b21-phase-state $O/seed_all.npz" ;;
    P5) STA=$O/b0.5.npz; PIN="--pin-b21 0.5 --b21-phase-state $O/seed_all.npz" ;;
    *) echo "unknown mode $MODE"; continue ;;
  esac
  $PY scripts/meanforce/mf_fixed.py $RUN --start $STA $PIN --force-norm p --p-schedule 16 --p-band 0.5675 0.7075 \
      --edge-alpha ${LOC_EA:-0.25} --mean-whole --alphas ${LOC_ALPHAS:-0.3 0.1 0.05 0.03 0.02} --max-outer 12 \
      --stage-flat 0.02 --stall-stop 1e-3 $SOLVER --snap-outer --time-budget ${BUDGET_LOC:-14400} --out $O/loc_$MODE
  S=""; for al in ${LOC_ALPHAS:-0.3 0.1 0.05 0.03 0.02}; do
    for f in $O/loc_${MODE}_p16_a$al.npz $O/loc_${MODE}_p16_a${al}_latest.npz; do [ -f $f ] && { S=$f; break; }; done; done
  [ -f $O/loc_${MODE}_budget.npz ] && S=$O/loc_${MODE}_budget.npz
  echo "loc $MODE state: $S"; [ -n "$S" ] && cp "$S" $O/loc_$MODE.npz
done
