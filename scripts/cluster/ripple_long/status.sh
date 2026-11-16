#!/bin/bash
# Live health digest of one or more runs.  usage: bash status.sh [RUN ...]   (default: F64L M30)
# For the raw log:  tail -f $RUNS/<RUN>/current/run.log
HERE=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
[ -f "$HERE/site.sh" ] && source "$HERE/site.sh"
: "${RUNS:?}"
for R in ${@:-F64L M30}; do ${PY:-python} "$HERE/chainlib.py" status "$RUNS/$R" ${N:-8}; echo; done
command -v squeue >/dev/null && squeue -u "$USER" -n nest-F64L,nest-M30 -o "%.10i %.12j %.8T %.10M %.10L %R" 2>/dev/null
