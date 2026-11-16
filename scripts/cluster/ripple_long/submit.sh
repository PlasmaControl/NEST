#!/bin/bash
# Submit (or resubmit) the segment chain of RUN.  usage: bash submit.sh RUN [--quiet]
# Reads site.sh for RUNS, NEST_ROOT and SBATCH_OPTS (account, partition, qos, time, ...).
set -u
RUN=${1:?usage: submit.sh RUN}
HERE=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
export NEST_ROOT=${NEST_ROOT:-$(cd "$HERE/../../.." && pwd)}
[ -f "$HERE/site.sh" ] || { echo "copy site.sh.example to site.sh and edit it first"; exit 1; }
source "$HERE/site.sh"
: "${RUNS:?site.sh must set RUNS}"
mkdir -p "$RUNS/$RUN"
J=$(sbatch --parsable ${SBATCH_OPTS:-} --job-name="nest-$RUN" --output="$RUNS/$RUN/slurm-%j.out" \
      --export=ALL,RUN=$RUN,RUNS=$RUNS,NEST_ROOT=$NEST_ROOT "$HERE/run.sbatch")
[ "${2:-}" = --quiet ] && echo "$J" || echo "submitted $RUN as job $J; live log: $RUNS/$RUN/current/run.log"
