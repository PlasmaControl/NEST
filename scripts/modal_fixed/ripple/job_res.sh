#!/bin/bash
# Resolution rung of the (2,1) ripple landscape study (docs/ripple_landscape_report.md, Section 11).
# usage: bash job_res.sh <RUNG> <L> <M> <N> <FRAC> <FINE_FRAC> <RUN_BUDGET_S> [smoke]
# Five runs, each a fresh AL solve (nu, rho stripped), alpha 0.1 mean bound, <= 10 outer x 40 inner = 400 inner
# steps, hard cap RUN_BUDGET_S per run (setup included), snapshot every outer:
#   S      shielded floor (no pin), from the 32/32/6 a/60 shielded start
#   b1.0, b0.5, b0.2   b21 pinned at beta x vacuum (island phase), from the 32/32/6 beta-scan end states
#   rel0.5 release: the pin removed, from this rung's b0.5 end state
# then island / Poincare / pressure / J_par diagnostics of the end states and an eval on a finer field grid.
# All starts come from the 32/32/6 basis (embedded with --start-LMN 32 32 6 when L, M, N differ).
set -u
RUNG=$1; L=$2; M=$3; N=$4; FRAC=$5; FINE=$6; TB=$7; SMOKE=${8:-}
cd /work; for f in desc_src.tar.gz deps_src.tar.gz ripple_L32N4_x1e-3.h5 gpu_entry.py; do [ -f $f ] || cp /results/stage/$f .; done
rm -rf src/scripts src/nest; mkdir -p src out
tar -xzf nest_src.tar.gz -C src
if [ ! -f src/.deps ]; then tar -xzf desc_src.tar.gz -C src && tar -xzf deps_src.tar.gz -C src && touch src/.deps; fi
touch src/.unpacked
EQ=/work/ripple_L32N4_x1e-3.h5; O=/results/fxr/res/$RUNG; mkdir -p $O
PY=/opt/conda/bin/python
strip() { $PY -c "import numpy as np; d=np.load('$1'); np.savez('$2', u=d['u'], c_p=d['c_p'])"; }
strip /results/fxr/A60_cont/A60_cont_a0.1.npz /work/st_S.npz
for B in 0.2 0.5 1.0; do strip /results/fxr/bscan/bs${B}_budget.npz /work/st_b$B.npz; done
cp /results/fxr/pre32/seed32_all.npz /work/sall.npz
EMB=""; [ "$L $M $N" != "32 32 6" ] && EMB="--start-LMN 32 32 6"
COMMON="--eq $EQ --bext desc-fit --LMN $L $M $N --frac $FRAC --N-phi 25 --alphas 0.1 --max-outer 10 --maxiter 40 \
        --snap-outer --f-noise auto --radius-by-delta --no-share-stop --time-budget $TB --mirror $O"
endstate() { ls -t /work/out/${1}_a0.1.npz /work/out/${1}_budget.npz 2>/dev/null | head -1; }
T0=$(date +%s); echo "[res] $RUNG start $(date -u)" | tee -a $O/progress.txt
if [ -n "$SMOKE" ]; then
  bash rung_run.sh ${RUNG}_smoke scripts/meanforce/mf_fixed.py $COMMON $EMB --start /work/st_b1.0.npz --pin-b21 1.0 \
      --b21-phase-state /work/sall.npz --max-outer 1 --maxiter 6 --out /work/out/${RUNG}_smoke
  cp /work/out/${RUNG}_smoke* $O/ 2>/dev/null; echo "[res] smoke done $(( $(date +%s) - T0 )) s" | tee -a $O/progress.txt; exit 0
fi
run() { echo "[res] $1 begin at $(( $(date +%s) - T0 )) s" | tee -a $O/progress.txt
        bash rung_run.sh ${RUNG}_$1 scripts/meanforce/mf_fixed.py $COMMON "${@:2}" --out /work/out/${RUNG}_$1
        echo "[res] $1 end at $(( $(date +%s) - T0 )) s: $(endstate ${RUNG}_$1)" | tee -a $O/progress.txt; }
run S $EMB --start /work/st_S.npz
run b1.0 $EMB --start /work/st_b1.0.npz --pin-b21 1.0 --b21-phase-state /work/sall.npz
run b0.5 $EMB --start /work/st_b0.5.npz --pin-b21 0.5 --b21-phase-state /work/sall.npz
F=$(endstate ${RUNG}_b0.5); [ -n "$F" ] && strip $F /work/st_rel.npz && run rel0.5 --start /work/st_rel.npz
run b0.2 $EMB --start /work/st_b0.2.npz --pin-b21 0.2 --b21-phase-state /work/sall.npz
# diagnostics of the end states (this rung's basis)
ST=""; PT="vacuum"; EV=""
for T in S b0.2 b0.5 b1.0 rel0.5; do F=$(endstate ${RUNG}_$T); [ -n "$F" ] || continue
  ST="$ST --state $F"; PT="$PT $(basename $F .npz):A"; EV="$EV --eval-state $T=$F"; done
echo "[res] diagnostics begin at $(( $(date +%s) - T0 )) s" | tee -a $O/progress.txt
bash rung_run.sh ${RUNG}_diag scripts/meanforce/start_check_bext.py --eq $EQ --LMN $L $M $N --frac $FRAC --N-phi 25 --m 2 --k 1 \
    --rho-grid 0.30 0.98 341 --band 0.04 0.08 0.16 $ST --perts $PT --turns 300 --lines 41 --seed-band 0.50 0.78 \
    --seed-theta 0.0 --pressure --out /work/out/${RUNG}_diag
cp /work/out/${RUNG}_diag*.json /work/out/${RUNG}_diag*poincare*.npz $O/ 2>/dev/null
for G in $FINE $FRAC; do
  bash rung_run.sh ${RUNG}_ev$G scripts/meanforce/mf_fixed.py --eq $EQ --bext desc-fit --LMN $L $M $N --frac $G --N-phi 25 \
      --alphas 1 --model beta --force-norm p --p-schedule 16 --p-band 0.5675 0.7075 --eval-only --b21-report $EV \
      --out /work/out/${RUNG}_ev$G --mirror $O
done
echo "[res] $RUNG all done at $(( $(date +%s) - T0 )) s" | tee -a $O/progress.txt
true
