#!/bin/bash
# Preflight for the long ripple runs: environment, inputs, GPU float64 linear algebra, and a short smoke chain
# (8/8/4, three segments: sweep -> floor stopped by a signal -> floor restarted and stopped by its time budget).
# usage: bash preflight.sh            (on a GPU node: e.g. inside 'salloc --gres=gpu:1 ...', ~10-15 min)
#        SKIP_SMOKE=1 bash preflight.sh   (checks only);  SMOKE_DEVICE=cpu to run the smoke chain on CPU
# Prints PASS/FAIL per check and a final verdict; exit 0 only if every check passed.
set -u
HERE=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
export NEST_ROOT=${NEST_ROOT:-$(cd "$HERE/../../.." && pwd)}
[ -f "$HERE/site.sh" ] || { echo "FAIL site.sh missing (copy site.sh.example to site.sh and edit)"; exit 1; }
source "$HERE/site.sh"
PY=${PY:-python}; fail=0
ok() { echo "PASS $*"; }; bad() { echo "FAIL $*"; fail=1; }

# 1. python environment
OUT=$($PY - <<'PY' 2>&1
import jax, numpy, scipy, desc
jax.config.update("jax_enable_x64", True)
d = jax.devices()
print("jax", jax.__version__, "numpy", numpy.__version__, "scipy", scipy.__version__, "desc", getattr(desc, "__version__", "?"))
print("devices", [f"{x.platform}:{getattr(x, 'device_kind', '')}" for x in d])
print("GPU_OK" if any(x.platform in ("gpu", "cuda") for x in d) else "NO_GPU")
PY
)
echo "$OUT" | sed 's/^/     /'
echo "$OUT" | grep -q GPU_OK && ok "jax sees a GPU" || bad "jax does not see a GPU (wrong node, or CPU-only jaxlib)"

# 2. float64 dense linear algebra on the GPU (the metric is a float64 Cholesky; some cuBLAS builds are wrong on B200)
OUT=$($PY - <<'PY' 2>&1
import jax, jax.numpy as jnp, numpy as np
jax.config.update("jax_enable_x64", True)
rng = np.random.default_rng(0); n = 4096
A = rng.standard_normal((n, n)); H = A @ A.T / n + np.eye(n)
Hd = jnp.asarray(H); L = jnp.linalg.cholesky(Hd)
e1 = float(jnp.linalg.norm(L @ L.T - Hd) / jnp.linalg.norm(Hd))
e2 = float(np.linalg.norm(np.asarray(Hd @ Hd) - H @ H) / np.linalg.norm(H @ H))
print(f"chol rel residual {e1:.2e}, matmul rel error {e2:.2e}")
print("LA_OK" if e1 < 1e-12 and e2 < 1e-12 else "LA_BAD")
PY
)
echo "$OUT" | sed 's/^/     /'
echo "$OUT" | grep -q LA_OK && ok "float64 Cholesky and matmul on the GPU" || bad "float64 GPU linear algebra is wrong (check the CUDA/cuBLAS version)"

# 3. repository and inputs
B=$(git -C "$NEST_ROOT" rev-parse --abbrev-ref HEAD 2>/dev/null); C=$(git -C "$NEST_ROOT" rev-parse --short HEAD 2>/dev/null)
echo "     repo $NEST_ROOT branch $B commit $C"
[ "$B" = ripple-landscape ] && ok "branch ripple-landscape" || bad "branch is '$B' (expected ripple-landscape)"
chk() { local f=$NEST_ROOT/$1 want=$2; [ -f "$f" ] || { bad "missing $1"; return; }
  got=$( (sha256sum "$f" 2>/dev/null || shasum -a 256 "$f") | cut -d' ' -f1)
  [ "$got" = "$want" ] && ok "input $1" || bad "checksum of $1 ($got)"; }
chk desc_input/tokamak_boundary_perturbed/bp2/L32N4/tokamak_ripple_C_L32M32N4_x0.001.h5 921fac08865342d8a5bfd74259499c8281f4fbd6a5d2a1fd9f2b2f145451fa69
chk results/ripple_landscape/valley/states/V_SA_budget.npz 136ddab43072c7fb0c88b2aa0a3fc063ef8566221e2d1c54652415645a3d161e

# 4. output root and scheduler
mkdir -p "$RUNS" 2>/dev/null && touch "$RUNS/.w" 2>/dev/null && rm -f "$RUNS/.w" && ok "RUNS writable ($RUNS)" || bad "RUNS not writable ($RUNS)"
FREE=$(df -Pk "$RUNS" 2>/dev/null | awk 'NR==2{print int($4/1048576)}'); [ "${FREE:-0}" -ge 20 ] && ok "free space ${FREE} GB" || bad "free space ${FREE:-?} GB < 20 GB"
command -v sbatch >/dev/null && command -v squeue >/dev/null && ok "sbatch/squeue available (resubmission works from compute nodes only if sbatch is there too)" || bad "sbatch/squeue not found"

# 5. smoke chain: sweep -> floor stopped by a signal (as Slurm's USR1) -> floor stopped by the time budget
if [ -z "${SKIP_SMOKE:-}" ]; then
  export RUNS_REAL=$RUNS RUNS=$RUNS/_preflight NEST_DEVICE=${SMOKE_DEVICE:-gpu} SEG_SECONDS=100000
  rm -rf "$RUNS/SMOKE"; S=$RUNS/SMOKE; mkdir -p "$RUNS"
  bash "$HERE/segment.sh" SMOKE > "$RUNS/smoke1.txt" 2>&1; r1=$?
  [ $r1 -eq 0 ] && [ -f "$S/PHASE_A_DONE" ] && ok "smoke segment 1 (alpha sweep runs and ends)" || bad "smoke segment 1 rc $r1 (see $S/seg01/run.log)"
  grep -q "^\[diag\]" "$S/seg01/run.log" && ok "[diag] lines in the log" || bad "no [diag] lines"
  bash "$HERE/segment.sh" SMOKE > "$RUNS/smoke2.txt" 2>&1 & P2=$!
  for i in $(seq 1 180); do sleep 5; grep -q "^    it " "$S/seg02/run.log" 2>/dev/null && break; done
  kill -USR1 $P2 2>/dev/null; wait $P2; r2=$?
  grep -q "\[control\] STOP" "$S/seg02/run.log" && grep -q "time budget reached: state saved" "$S/seg02/run.log" \
    && ok "signal -> CONTROL STOP -> clean stop with state saved (segment 2)" || bad "signal stop not seen, rc $r2 (see $S/seg02/run.log)"
  ls "$S"/seg02/*_budget.npz "$S"/seg02/*_a0.001_latest.npz >/dev/null 2>&1 && ok "checkpoints _budget.npz and _latest.npz written" || bad "checkpoint files missing in seg02"
  TIME_BUDGET=1 bash "$HERE/segment.sh" SMOKE > "$RUNS/smoke3.txt" 2>&1; r3=$?
  grep -q "time budget reached: state saved" "$S/seg03/run.log" && ok "time budget -> clean stop with state saved (segment 3)" || bad "time-budget stop not seen, rc $r3 (see $S/seg03/run.log)"
  CONT=$($PY - "$S" <<'PY'
import json, sys, glob
s = sys.argv[1]
def dl(seg):
    f = [x for x in glob.glob(f"{s}/{seg}/*.json") if not x.endswith("summary.json")][0]
    return json.load(open(f))["diag log"]
a = dl("seg02")[-1]["M/M_desc"]
b = [d for d in dl("seg03") if d["where"] == "start"][0]["M/M_desc"]
print(f"seg02 last {a:.10f} seg03 start {b:.10f} rel {abs(a - b) / a:.1e}", "CONT_OK" if abs(a - b) <= 1e-8 * a else "CONT_BAD")
PY
)
  echo "     $CONT"; echo "$CONT" | grep -q CONT_OK && ok "restart continues from the saved state" || bad "restart does not reproduce the saved state"
  export RUNS=$RUNS_REAL
fi
echo; [ $fail -eq 0 ] && echo "PREFLIGHT PASS" || echo "PREFLIGHT FAIL"
exit $fail
