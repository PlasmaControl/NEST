#!/bin/bash
# Build the report and a tarball to send back.  usage: bash collect.sh [RUN ...]   (default: F64L M30)
# Output: $RUNS/report/report.md (+ CSVs, figures) and $RUNS/nest_ripple_long_<date>.tar.gz containing the report,
# every segment's run.log, JSON, summary, segment.txt, gpumem.txt, and the newest state of each run.
set -u
HERE=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
[ -f "$HERE/site.sh" ] && source "$HERE/site.sh"
: "${RUNS:?}"; PY=${PY:-python}; RUNL=${@:-F64L M30}
$PY "$HERE/chainlib.py" report "$RUNS" $RUNL
cd "$RUNS"; LIST=report
for R in $RUNL; do
  [ -d "$R" ] || continue
  LIST="$LIST $(ls -d $R/chain.tsv $R/FINISHED $R/jobs.txt $R/seg*/run.log $R/seg*/*.json $R/seg*/segment.txt $R/seg*/gpumem.txt 2>/dev/null | tr '\n' ' ')"
  NS=$($PY "$HERE/chainlib.py" newest-state "$RUNS/$R"); [ -n "$NS" ] && LIST="$LIST ${NS#$RUNS/}"
done
T=nest_ripple_long_$(date +%Y%m%d_%H%M).tar.gz
tar -czf "$T" $LIST && echo "wrote $RUNS/$T ($(du -h "$T" | cut -f1))"
