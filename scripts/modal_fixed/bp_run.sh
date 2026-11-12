#!/bin/bash
# DESC fixed-boundary (2,1) boundary-ripple ladder (DSHAPE island study, candidate C, NFP = 1).
# Usage: bash bp_run.sh <tag>. Inputs staged flat in /work: desc_src.tar.gz, deps_src.tar.gz, gpu_entry.py,
# make_bdry_ripple.py, tokamak_island_C_L{24,32}M*.h5. Outputs: out/<case>/ (h5 states, summary json),
# out/<case>.log; logs are also streamed to the Volume (/results/bp/<tag>/<case>.log) as the run goes.
set -u
TAG=$1
cd /work
mkdir -p src out
tar -xzf desc_src.tar.gz -C src && tar -xzf deps_src.tar.gz -C src
export PYTHONPATH=/work/src:${PYTHONPATH:-}
export JAX_PLATFORMS=cuda,cpu PYTHONUNBUFFERED=1 TF_CPP_MIN_LOG_LEVEL=1
export XLA_PYTHON_CLIENT_PREALLOCATE=false XLA_PYTHON_CLIENT_MEM_FRACTION=0.95
MIR=/results/bp/$TAG
mkdir -p $MIR
PY=/opt/conda/bin/python
nvidia-smi -L | tee out/nvsmi.txt
( while true; do echo "$(date +%s) $(nvidia-smi --query-gpu=memory.used,utilization.gpu --format=csv,noheader,nounits)" >> out/gpumem.txt; sleep 15; done ) &
MON=$!
AMPS="1e-4 3e-4 1e-3 3e-3"
RC=0
run() {
    local name=$1; shift
    echo "=== $name: $* ===" | tee -a $MIR/progress.txt
    $PY gpu_entry.py make_bdry_ripple.py "$@" --out out/$name --mirror $MIR/$name 2>&1 \
        | grep --line-buffered -v "SoL config" | tee out/$name.log > $MIR/$name.log
    local rc=${PIPESTATUS[0]}
    echo "=== $name exit $rc $(date) ===" | tee -a $MIR/progress.txt
    [ $rc -ne 0 ] && RC=$rc
    cp out/gpumem.txt $MIR/ 2>/dev/null
}
run L24N5      --base tokamak_island_C_L24M24.h5 --N 5 --amps $AMPS
run L24N5flip  --base tokamak_island_C_L24M24.h5 --N 5 --amps 1e-3 --helicity flip
run L32N5      --base tokamak_island_C_L32M32.h5 --N 5 --amps $AMPS
run L32N4      --base tokamak_island_C_L32M32.h5 --N 4 --amps $AMPS
kill $MON 2>/dev/null
echo "exit $RC"
exit $RC
