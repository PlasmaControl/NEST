# shared setup for the cylindrical GPU jobs: sourced with V (Volume dir) set
set -uo pipefail
mkdir -p /work/src/repo /work/out "$V"
cd /work
tar --warning=no-unknown-keyword -xzf nest_src.tar.gz -C src/repo && tar --warning=no-unknown-keyword -xzf desc_src.tar.gz -C src && tar --warning=no-unknown-keyword -xzf deps_src.tar.gz -C src
export PYTHONPATH=/work/src:/work/src/repo/nest:/work/src/repo/tests:${PYTHONPATH:-}
export NEST_DEVICE=gpu JAX_ENABLE_X64=true JAX_PLATFORMS=cuda,cpu
export XLA_PYTHON_CLIENT_PREALLOCATE=false XLA_PYTHON_CLIENT_MEM_FRACTION=0.95
export PYTHONUNBUFFERED=1 MPLBACKEND=Agg OMP_NUM_THREADS=4
export NEST_CACHE=/results/cyl/cache JAX_COMPILATION_CACHE_DIR=/results/cyl/jaxcache
mkdir -p "$NEST_CACHE"
nvidia-smi --query-gpu=name,driver_version,memory.total --format=csv | tee "$V/gpu.txt" /work/out/gpu.txt
( while true; do echo "$(date +%s) $(nvidia-smi --query-gpu=memory.used,utilization.gpu --format=csv,noheader)" >> "$V/gpu_mem.txt"; sleep 5; done ) &
MON=$!
PY="/opt/conda/bin/python /work/nest_gpu_entry.py"
log() { echo "[$(date +%H:%M:%S)] $*" | tee -a "$V/log.txt" /work/out/log.txt; }
run() {  # run <name> <script and args...>: streams to the Volume, returns the python status
  local name=$1; shift
  log "=== $name start"
  ( cd /work/src/repo && $PY "$@" ) 2>&1 | grep --line-buffered -v -i "warning" | tee -a "$V/$name.log" "$V/log.txt" /work/out/log.txt
  local st=${PIPESTATUS[0]}
  log "=== $name exit $st"
  return $st
}
finish() { kill $MON 2>/dev/null; cp "$V"/*.json "$V"/*.npz "$V"/*.log "$V/gpu_mem.txt" /work/out/ 2>/dev/null; log "done"; }
