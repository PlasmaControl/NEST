#!/bin/bash
# One segment of a long NEST run: pick the newest checkpoint, run scripts/meanforce/mf_fixed.py until the
# time budget, summarize. Scheduler-independent (run.sbatch wraps it for Slurm; it also runs by hand).
#
# usage: bash segment.sh RUN          RUN = F64L | M30 | SMOKE   (see docs/ripple_long_runs_runbook.md)
# env:   RUNS        output root (required); the run lives in $RUNS/$RUN/segNN/
#        NEST_ROOT   repository root (default: three levels above this script)
#        PY          python executable (default: python)
#        SEG_SECONDS wall seconds this segment may use (default: Slurm's remaining time, else 86400)
#        TIME_BUDGET explicit --time-budget in seconds (overrides SEG_SECONDS/SAFETY; for tests)
#        SAFETY      seconds kept free at the end for the step in progress and the final save (default 2700)
#        STALL_RATE  floor phases stop after two segments each lowering M/eps by less than this, relative per
#                    solve hour (default 5e-4); MAX_SEG caps the number of segments (default 10)
# exit:  0 = segment done, continue the chain; 3 = run finished (stalled, done or capped); other = error
set -u
RUN=${1:?usage: segment.sh RUN}
HERE=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
NEST_ROOT=${NEST_ROOT:-$(cd "$HERE/../../.." && pwd)}
RUNS=${RUNS:?set RUNS to the output root}
PY=${PY:-python}
CL="$PY $HERE/chainlib.py"
R=$RUNS/$RUN; mkdir -p "$R"
if [ -f "$R/FINISHED" ]; then echo "[segment] $RUN already finished: $(cat "$R/FINISHED")"; exit 3; fi
if [ -f "$R/NO_RESUBMIT" ]; then echo "[segment] $RUN: NO_RESUBMIT present, not starting"; exit 3; fi
K=$(ls -d "$R"/seg[0-9][0-9] 2>/dev/null | wc -l); K=$((K + 1)); SEG=$(printf "seg%02d" "$K"); D=$R/$SEG
mkdir -p "$D"; ln -sfn "$SEG" "$R/current"

# time budget: mf_fixed.py counts it from program start (setup included) and stops at the first step boundary
# after it, so keep SAFETY for one long step plus the save
if [ -z "${SEG_SECONDS:-}" ] && [ -n "${SLURM_JOB_ID:-}" ]; then
  L=$(squeue -h -j "$SLURM_JOB_ID" -o %L 2>/dev/null)   # [d-]hh:mm:ss remaining
  SEG_SECONDS=$(echo "$L" | awk -F'[-:]' '{n=NF; s=$n+60*$(n-1)+3600*(n>2?$(n-2):0)+86400*(n>3?$(n-3):0); print s}')
fi
case ${SEG_SECONDS:-} in ''|0|*[!0-9]*) SEG_SECONDS=86400 ;; esac   # unset, UNLIMITED or unparsable
SAFETY=${SAFETY:-2700}
TB=$(( SEG_SECONDS - SAFETY )); [ "$TB" -lt 600 ] && TB=600
TB=${TIME_BUDGET:-$TB}   # explicit override (tests)

EQ=$NEST_ROOT/desc_input/tokamak_boundary_perturbed/bp2/L32N4/tokamak_ripple_C_L32M32N4_x0.001.h5
REC="--subspace prev,accel --cg 1500 --no-rebuild-on-hit --f-noise auto --radius-by-delta --no-share-stop --locate-chunk 131072"
LOGS="--diag-every ${DIAG_EVERY:-5} --save-every 1 --snap-outer"
FLOOR="--alphas 0.001 --max-outer 1 --maxiter 100000"

case $RUN in
  F64L)   # 64/64/6, field grid a/200: continue the alpha 0.001 floor from the valley A/B winner (M/eps 16.14)
    GRID="--LMN 64 64 6 --N-phi 25 --frac 200"
    PHASE=floor
    SRC=$($CL newest-state "$R")
    [ -z "$SRC" ] && SRC=$NEST_ROOT/results/ripple_landscape/valley/states/V_SA_budget.npz
    $CL strip "$SRC" "$D/start.npz"; START="--start $D/start.npz"; ALPHAS=$FLOOR ;;
  M30|SMOKE)  # 30/30/8 a/100: alpha continuation from J_fit (phase A), then the alpha 0.001 floor (phase B)
    if [ "$RUN" = SMOKE ]; then GRID="--LMN 8 8 4 --N-phi 17 --frac 20"; AL="0.1 0.05"; REC="--subspace prev,accel --cg 200 --no-rebuild-on-hit --f-noise auto --radius-by-delta --no-share-stop"
    else GRID="--LMN 30 30 8 --N-phi 33 --frac 100"; AL="0.1 0.05 0.02 0.01"; fi
    if [ ! -f "$R/PHASE_A_DONE" ]; then
      REM=$($CL remaining-alphas "$R" $AL)
      [ -z "$REM" ] && touch "$R/PHASE_A_DONE"
    fi
    if [ ! -f "$R/PHASE_A_DONE" ]; then
      PHASE=sweep
      SRC=$($CL newest-state "$R")
      START=""; [ -n "$SRC" ] && START="--start $SRC"
      ALPHAS="--alphas $REM --stall-stop 0.01"
      [ "$RUN" = SMOKE ] && ALPHAS="$ALPHAS --max-outer 2 --maxiter 6"
    else
      PHASE=floor
      SRC=$($CL newest-state "$R" floor); [ -z "$SRC" ] && SRC=$($CL newest-state "$R")
      $CL strip "$SRC" "$D/start.npz"; START="--start $D/start.npz"; ALPHAS=$FLOOR
    fi ;;
  *) echo "unknown RUN $RUN"; exit 2 ;;
esac
echo "$PHASE" > "$D/PHASE"

CMD="$PY $NEST_ROOT/scripts/meanforce/mf_fixed.py --eq $EQ --bext desc-fit $GRID $ALPHAS $REC $LOGS $START \
--time-budget $TB --out $D/$RUN"
{ echo "[segment] $RUN $SEG phase $PHASE on $(hostname) at $(date "+%Y-%m-%dT%H:%M:%S"); start state: ${SRC:-J_fit}"
  echo "[segment] git $(git -C "$NEST_ROOT" rev-parse --short HEAD 2>/dev/null) ; budget $TB s of $SEG_SECONDS s"
  echo "[segment] $CMD"; } | tee "$D/segment.txt"

export NEST_DEVICE=${NEST_DEVICE:-gpu} PYTHONUNBUFFERED=1 TF_CPP_MIN_LOG_LEVEL=1
export XLA_PYTHON_CLIENT_PREALLOCATE=false XLA_PYTHON_CLIENT_MEM_FRACTION=${XLA_PYTHON_CLIENT_MEM_FRACTION:-0.95}
export NEST_CACHE=${NEST_CACHE:-$RUNS/cache}; mkdir -p "$NEST_CACHE"
if command -v nvidia-smi >/dev/null && [ "$NEST_DEVICE" = gpu ]; then
  ( while true; do echo "$(date +%s) $(nvidia-smi --query-gpu=memory.used,utilization.gpu --format=csv,noheader,nounits | head -1)" \
      >> "$D/gpumem.txt"; sleep 60; done ) & MON=$!
fi
cd "$NEST_ROOT"
$CMD > "$D/run.log" 2>&1 &
PID=$!; echo $PID > "$D/PID"
# a USR1/TERM from the scheduler asks the solver to stop at the next step boundary (state saved)
trap 'echo "[segment] signal at $(date "+%Y-%m-%dT%H:%M:%S"): writing CONTROL STOP" >> "$D/segment.txt"; echo STOP > "$D/CONTROL"' USR1 TERM
# wait for the solver; a trapped signal interrupts 'wait' (status > 128) while the solver keeps running
while :; do
  wait $PID; RC=$?
  if [ $RC -gt 128 ] && kill -0 $PID 2>/dev/null; then continue; fi
  if [ $RC -gt 128 ]; then wait $PID 2>/dev/null; R2=$?; [ $R2 -ne 127 ] && RC=$R2; fi
  break
done
[ -n "${MON:-}" ] && kill $MON 2>/dev/null
rm -f "$D/PID"
echo "[segment] python exit $RC at $(date "+%Y-%m-%dT%H:%M:%S")" | tee -a "$D/segment.txt"
$CL summarize "$D" | tee -a "$D/segment.txt"
[ "$RC" -ne 0 ] && { echo "error: exit $RC" > "$R/FINISHED"; exit "$RC"; }
STAGE=$($PY -c "import json,sys; print(json.load(open(sys.argv[1]))['stage'])" "$D/summary.json")
if [ "$PHASE" = sweep ] && [ "$STAGE" = "sweep ended" ]; then touch "$R/PHASE_A_DONE"; echo "[segment] phase A done"; fi
DEC=$($CL decide "$R" "${STALL_RATE:-5e-4}" "${MAX_SEG:-10}")
echo "[segment] chain: $DEC" | tee -a "$D/segment.txt"
case $DEC in finished*) echo "${DEC#finished: }" > "$R/FINISHED"; exit 3 ;; esac
exit 0
