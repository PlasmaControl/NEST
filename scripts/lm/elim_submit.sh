#!/bin/bash
# Submit the speed-elim production A/B (docs/pressure_poisson_runbook.md, step 4): the GPU gate tests first, then
# abB, abD, abE, cond, cond_p3, each only if the gate tests pass (Slurm afterok dependency).
# Usage (repo root): NEST_CACHE=<cache dir> bash scripts/lm/elim_submit.sh      (DRY=1 prints the commands only;
#   SKIP_GATE=1 submits without the gate tests, only as the runbook's troubleshooting table allows)
# Writes the job ids to results/isl/elim_jobs.txt (label=jobid) for scripts/lm/elim_collect.sh.
set -euo pipefail
cd "$(dirname "$0")/../.."
: "${NEST_CACHE:?export NEST_CACHE first (see the runbook)}"
export NEST_CACHE
mkdir -p logs results/isl
COMMON="scripts/dshape/isl_scan.py --base results/gs/hN_a120D --N 2 --eps 1000 --w-J 1e-7 --j-mode full \
 --w-W 0 --w-pos 0 --iters 30 --rtol 1e-12 --gn-tol 0 --batch 16 8 --start-from a120Cn2d1000"
FAST="resident,chol,narrow,merge=100000,gpus=4"
CONDA="scripts/diagnostics/pressure_elim_conditioning_isl.py --base results/gs/hN_a120D --N 2 --eps 1000 --w-J 1e-7 \
 --j-mode full --w-W 0 --w-pos 0 --batch 16 8 --state results/isl/a120Cn2d1000_eps1000.npz --speed resident"
G4="--gres=gpu:nvidia_b200:4 --mem=500G"
G1="--gres=gpu:nvidia_b200:1 --mem=300G"
submit() {  # label, sbatch options..., -- , script args...
  local lab=$1; shift
  local opts=(); while [ "$1" != "--" ]; do opts+=("$1"); shift; done; shift
  if [ "${DRY:-0}" = "1" ]; then echo "sbatch --parsable --job-name=$lab ${opts[*]} run_sai.sbatch $*"; echo "$lab=DRY" >> results/isl/elim_jobs.txt; return; fi
  local id; id=$(sbatch --parsable --job-name="$lab" "${opts[@]}" run_sai.sbatch "$@")
  echo "$lab=$id" | tee -a results/isl/elim_jobs.txt
}
: > results/isl/elim_jobs.txt
if [ "${SKIP_GATE:-0}" = "1" ]; then
  echo "gate=skipped" >> results/isl/elim_jobs.txt; DEP="--time=04:00:00"   # (a harmless option in place of the dependency)
else
  submit gate $G1 --time=01:00:00 -- tests/test_pressure_elim.py
  GATE=$(grep '^gate=' results/isl/elim_jobs.txt | cut -d= -f2)
  DEP="--dependency=afterok:$GATE"
fi
submit abB $G4 $DEP -- $COMMON --speed $FAST,phys,tr --name abN2B1000
submit abD $G4 $DEP -- $COMMON --speed $FAST,elim,phys,tr --name abN2D1000
submit abE $G4 $DEP -- $COMMON --speed $FAST,elim,phys,tr --p-factors 3 3 1 --name abN2E1000
submit cond $G1 $DEP -- $CONDA --out results/isl/cond_a120Cn2d1000.json
submit cond_p3 $G1 $DEP -- $CONDA --p-factors 3 3 1 --out results/isl/cond_a120Cn2d1000_p3.json
echo "submitted; ids in results/isl/elim_jobs.txt; watch with: squeue -u \$USER; tail -f logs/nest-<id>.log"
