#!/bin/bash
# S_m reference build on a CPU container: run_s_ref.sh FRAC JOBS
set -uo pipefail
FRAC=$1; JOBS=$2
V=/results/cyl/s_ref; mkdir -p /work/src/repo /work/out "$V" /results/cyl/S_cache
cd /work && tar --warning=no-unknown-keyword -xzf nest_src.tar.gz -C src/repo
export PYTHONPATH=/work/src/repo/nest:/work/src/repo/tests:${PYTHONPATH:-}
export PYTHONUNBUFFERED=1 OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 JAX_PLATFORMS=cpu
echo "[$(date +%H:%M:%S)] commit $(cat /work/commit.txt); a/$FRAC, $(nproc) CPUs, $(awk '/MemTotal/{print int($2/1048576)}' /proc/meminfo) GB" | tee -a "$V/a$FRAC.log" /work/out/log.txt
( cd /work/src/repo && /opt/conda/bin/python -u tests/cyl_s_build.py build --grids /work/s_grids.json --frac "$FRAC" \
    --cache /results/cyl/S_cache --jobs "$JOBS" --spot 24 --out "$V" ) 2>&1 | grep --line-buffered -v -i warning | tee -a "$V/a$FRAC.log" /work/out/log.txt
st=${PIPESTATUS[0]}
cp "$V/s_ref_a$FRAC.json" /work/out/ 2>/dev/null
echo "[$(date +%H:%M:%S)] exit $st" | tee -a "$V/a$FRAC.log" /work/out/log.txt
exit $st
