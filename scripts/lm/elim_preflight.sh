#!/bin/bash
# Login-node checks before the speed-elim production A/B (docs/pressure_poisson_runbook.md, step 3).
# Usage (repo root of the pressure-poisson worktree): NEST_CACHE=<cache dir> bash scripts/lm/elim_preflight.sh
# Prints PASS / FAIL per check; exit status 1 if any FAIL.
ENV=${ENV:-/scratch/gpfs/EKOLEMEN/lk1158/envs/nest13}
fail=0
ok()  { echo "PASS  $*"; }
bad() { echo "FAIL  $*"; fail=1; }
cd "$(dirname "$0")/../.." || exit 1
echo "repo: $(pwd)"
b=$(git rev-parse --abbrev-ref HEAD 2>/dev/null); h=$(git rev-parse --short HEAD 2>/dev/null)
[ "$b" = "pressure-poisson" ] && ok "branch $b at $h" || bad "branch is '$b' (expected pressure-poisson)"
grep -q "def _elim_step" nest/harmonic_dense.py && ok "speed elim present in nest/harmonic_dense.py" || bad "nest/harmonic_dense.py has no _elim_step (wrong code)"
[ -z "$(git status --porcelain --untracked-files=no)" ] && ok "no uncommitted changes to tracked files" || bad "tracked files modified: $(git status --porcelain --untracked-files=no | head -3 | tr '\n' ' ')"
for f in results/gs/hN_a120D_iter.json results/gs/hN_a120D_iter.npz results/isl/a120Cn2d1000_eps1000.json \
         results/isl/a120Cn2d1000_eps1000.npz results/lm/L16a20_full_base_wJ0.001.npz \
         results/lm/L16a20_full_stage_eps1000_wJ0.001.npz run_sai.sbatch scripts/dshape/isl_scan.py \
         scripts/diagnostics/pressure_elim_conditioning_isl.py scripts/lm/elim_report.py tests/test_pressure_elim.py; do
  [ -s "$f" ] && ok "input $f" || bad "missing input $f"
done
if [ -z "$NEST_CACHE" ]; then
  bad "NEST_CACHE is not set (export it to the cache directory that holds your HarmTab_u2_*.npy)"
else
  for t in u0 u1 u2; do
    n=$(ls "$NEST_CACHE"/HarmTab_${t}_*.npy 2>/dev/null | wc -l | tr -d ' ')
    [ "$n" -ge 1 ] && ok "NEST_CACHE has $n HarmTab_${t} table(s)" || bad "NEST_CACHE ($NEST_CACHE) has no HarmTab_${t}_*.npy: the N = 2 current tables would be rebuilt (~1.5 h of box solves); point NEST_CACHE at the cache of your a120Cn2d1000 runs"
  done
  for t in p0 p1 p2; do
    n=$(ls "$NEST_CACHE"/HarmTab_${t}_*.npy 2>/dev/null | wc -l | tr -d ' ')
    [ "$n" -ge 1 ] && ok "NEST_CACHE has $n HarmTab_${t} table(s)" || echo "WARN  NEST_CACHE has no HarmTab_${t}_*.npy (pressure tables are analytic: rebuilt in minutes, no action needed)"
  done
  [ -w "$NEST_CACHE" ] && ok "NEST_CACHE writable (new pressure tables for abE go there)" || bad "NEST_CACHE not writable"
  echo "      NEST_CACHE size: $(du -sh "$NEST_CACHE" 2>/dev/null | cut -f1); free: $(df -h "$NEST_CACHE" | awk 'NR==2{print $4}')"
fi
[ -x "$ENV/bin/python" ] && ok "python env $ENV" || bad "no python at $ENV/bin/python (set ENV=...)"
if [ -x "$ENV/bin/python" ]; then
  v=$(JAX_PLATFORMS=cpu "$ENV/bin/python" -c "import jax, desc, numpy; print('jax', jax.__version__, 'desc', desc.__version__, 'numpy', numpy.__version__)" 2>/dev/null | tail -1)
  [ -n "$v" ] && ok "imports: $v" || bad "jax/desc/numpy import failed in $ENV"
fi
mkdir -p logs && ok "logs/ exists (Slurm output goes to logs/nest-<jobid>.log)"
command -v sbatch >/dev/null && ok "sbatch available" || bad "sbatch not found (run on a Stellar login node)"
echo
[ $fail -eq 0 ] && echo "PREFLIGHT PASSED" || echo "PREFLIGHT FAILED: fix the FAIL lines before submitting"
exit $fail
