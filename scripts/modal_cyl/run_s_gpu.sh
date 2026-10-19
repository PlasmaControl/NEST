#!/bin/bash
# Device S_m build against the stored references (GPU container): run_s_gpu.sh "FRACS" [--rekey]
set -uo pipefail
FRACS=$1; shift
V=/results/cyl/s_gpu; mkdir -p /work/src/repo /work/out "$V"
cd /work && tar --warning=no-unknown-keyword -xzf nest_src.tar.gz -C src/repo
export PYTHONPATH=/work/src/repo/nest:/work/src/repo/tests:${PYTHONPATH:-}
# BLAS pinned (the eig threads); the eig threads = the container's CPUs (affinity shows the host's)
export PYTHONUNBUFFERED=1 OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 NEST_EIG_THREADS=${NEST_EIG_THREADS:-8}
echo "[$(date +%H:%M:%S)] commit $(cat /work/commit.txt); fracs $FRACS; $(awk '/MemTotal/{print int($2/1048576)}' /proc/meminfo) GB" | tee -a /work/out/log.txt
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader | tee -a /work/out/log.txt
( cd /work/src/repo && /opt/conda/bin/python -u tests/cyl_s_gpu.py --grids /work/s_grids.json --fracs $FRACS \
    --ref /results/cyl/S_cache --refjson /results/cyl/s_ref --out /work/out "$@" ) 2>&1 \
    | grep --line-buffered -v -i warning | tee -a /work/out/log.txt
st=${PIPESTATUS[0]}
cp /work/out/s_gpu.json "$V/s_gpu_$(date +%H%M%S).json" 2>/dev/null
echo "[$(date +%H:%M:%S)] exit $st" | tee -a /work/out/log.txt
exit $st
