#!/bin/bash
# Generic NEST step on Modal with a live log on the Volume.
# usage: bash rung_run.sh <TAG> <script relative to src/> <args...>
# Inputs (flat in /work): nest_src.tar.gz desc_src.tar.gz deps_src.tar.gz gpu_entry.py (+ data files).
# Log: out/<TAG>.log and /results/fxr/<TAG>/<TAG>.log (streamed); GPU memory every 30 s to the same place.
set -u
TAG=$1; SCRIPT=$2; shift 2
cd /work
mkdir -p src out
if [ ! -f src/.unpacked ]; then
    tar -xzf nest_src.tar.gz -C src && tar -xzf desc_src.tar.gz -C src && tar -xzf deps_src.tar.gz -C src && touch src/.unpacked
fi
export PYTHONPATH=/work/src/nest:/work/src:${PYTHONPATH:-}
export NEST_DEVICE=gpu JAX_PLATFORMS=cuda,cpu PYTHONUNBUFFERED=1 TF_CPP_MIN_LOG_LEVEL=1
export XLA_PYTHON_CLIENT_PREALLOCATE=false XLA_PYTHON_CLIENT_MEM_FRACTION=0.95
export NEST_CACHE=/results/fx/cache
MIR=/results/fxr/$TAG
mkdir -p $MIR $NEST_CACHE
PY=/opt/conda/bin/python
nvidia-smi -L | tee out/nvsmi.txt
( while true; do echo "$(date +%s) $(nvidia-smi --query-gpu=memory.used,utilization.gpu --format=csv,noheader,nounits)" >> out/${TAG}_gpumem.txt; cp out/${TAG}_gpumem.txt $MIR/ 2>/dev/null; sleep 30; done ) &
MON=$!
cd src
$PY ../gpu_entry.py $SCRIPT "$@" 2>&1 | grep --line-buffered -v "SoL config" | tee ../out/$TAG.log | tee $MIR/$TAG.log
RC=${PIPESTATUS[0]}
kill $MON 2>/dev/null
cp ../out/${TAG}_gpumem.txt $MIR/ 2>/dev/null
echo "$TAG exit $RC"
exit $RC
