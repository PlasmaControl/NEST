#!/bin/bash
# Production: fixed-boundary matrix-free mean-force sweep. Usage: bash fx_run.sh <tag> <mf_fixed args...>
set -u
TAG=$1; shift
cd /work
mkdir -p src out
tar -xzf nest_src.tar.gz -C src && tar -xzf desc_src.tar.gz -C src && tar -xzf deps_src.tar.gz -C src
export PYTHONPATH=/work/src/nest:/work/src:${PYTHONPATH:-}
export NEST_DEVICE=gpu JAX_PLATFORMS=cuda,cpu PYTHONUNBUFFERED=1 TF_CPP_MIN_LOG_LEVEL=1
export XLA_PYTHON_CLIENT_PREALLOCATE=false XLA_PYTHON_CLIENT_MEM_FRACTION=0.95
export NEST_CACHE=/results/fx/cache
MIR=/results/fx/$TAG
mkdir -p $MIR $NEST_CACHE
PY=/opt/conda/bin/python
nvidia-smi -L | tee out/nvsmi.txt
( while true; do echo "$(date +%s) $(nvidia-smi --query-gpu=memory.used,utilization.gpu --format=csv,noheader,nounits)" >> out/gpumem.txt; sleep 15; done ) &
MON=$!
cd src
$PY ../gpu_entry.py scripts/meanforce/mf_fixed.py "$@" --out ../out/$TAG --mirror $MIR 2>&1 \
   | grep --line-buffered -v "SoL config" | tee ../out/$TAG.log
RC=${PIPESTATUS[0]}
kill $MON 2>/dev/null
cp ../out/$TAG.log ../out/gpumem.txt $MIR/ 2>/dev/null
echo "exit $RC"
exit $RC
