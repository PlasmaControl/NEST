#!/bin/bash
# After all jobs ended (docs/pressure_poisson_runbook.md, step 6): the report and the return package.
# Usage (repo root): bash scripts/lm/elim_collect.sh
set -uo pipefail
cd "$(dirname "$0")/../.."
ENV=${ENV:-/scratch/gpfs/EKOLEMEN/lk1158/envs/nest13}
PY=${PY:-$ENV/bin/python}
J=results/isl/elim_jobs.txt
id() { grep "^$1=" $J | cut -d= -f2; }
LOGS=""
for lab in gate abB abD abE cond cond_p3; do i=$(id $lab); [[ "$i" =~ ^[0-9]+$ ]] && LOGS="$LOGS $lab=logs/nest-$i.log"; done
echo "== gate tests:"; grep -E "^T[0-9]|all passed|Error|Traceback" logs/nest-$(id gate).log 2>/dev/null | tail -12 || echo "(no gate log)"
$PY scripts/lm/elim_report.py --runs abB=abN2B1000 abD=abN2D1000 abE=abN2E1000 --eps 1000 --ref-state a120Cn2d1000 \
  --cond results/isl/cond_a120Cn2d1000.json results/isl/cond_a120Cn2d1000_p3.json --logs $LOGS \
  --out results/isl/elim_report.md
OUT=results/isl/elim_return_$(date +%Y%m%d_%H%M).tar.gz
FILES="results/isl/elim_report.md $J"
[ -f results/isl/elim_operator_notes.md ] && FILES="$FILES results/isl/elim_operator_notes.md"
for n in abN2B1000 abN2D1000 abN2E1000; do for e in json npz; do f=results/isl/${n}_eps1000.$e; [ -f $f ] && FILES="$FILES $f"; done; done
for f in results/isl/cond_a120Cn2d1000.json results/isl/cond_a120Cn2d1000_p3.json; do [ -f $f ] && FILES="$FILES $f"; done
for lab in gate abB abD abE cond cond_p3; do i=$(id $lab); f=logs/nest-$i.log; [[ "$i" =~ ^[0-9]+$ ]] && [ -f $f ] && FILES="$FILES $f"; done
tar czf $OUT $FILES && echo "return package: $OUT ($(du -h $OUT | cut -f1))" && tar tzf $OUT
