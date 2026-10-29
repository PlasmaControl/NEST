#!/bin/bash
# bash test_run.sh <script.py> [args]: run a test script from the NEST tree (CPU JAX by default)
set -u
cd /work
mkdir -p src out
tar -xzf nest_src.tar.gz -C src && tar -xzf desc_src.tar.gz -C src && tar -xzf deps_src.tar.gz -C src
export PYTHONPATH=/work/src/nest:/work/src:${PYTHONPATH:-}
export NEST_DEVICE=${NEST_DEVICE:-cpu} JAX_PLATFORMS=${JAX_PLATFORMS:-cpu} PYTHONUNBUFFERED=1 TF_CPP_MIN_LOG_LEVEL=1
export NEST_CACHE=/results/fx/cache
cd src
/opt/conda/bin/python "$@" 2>&1 | grep --line-buffered -v "SoL config" | tee ../out/test.log
RC=${PIPESTATUS[0]}
echo "exit $RC"
exit $RC
