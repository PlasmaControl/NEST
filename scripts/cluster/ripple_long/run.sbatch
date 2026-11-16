#!/bin/bash
# Slurm wrapper: one segment per job, then resubmit itself until the run finishes.
# Submit with submit.sh (it fills in the site-specific options), not directly.
#SBATCH --job-name=nest-ripple
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:1
#SBATCH --mem=64G
#SBATCH --signal=B:USR1@1800
set -u
: "${RUN:?}" "${RUNS:?}" "${NEST_ROOT:?}"
HERE=$NEST_ROOT/scripts/cluster/ripple_long
[ -f "$HERE/site.sh" ] && source "$HERE/site.sh"
mkdir -p "$RUNS/$RUN"
echo "$SLURM_JOB_ID $(date "+%Y-%m-%dT%H:%M:%S") start" >> "$RUNS/$RUN/jobs.txt"
# forward USR1 (sent 30 min before the time limit) to the segment, which asks the solver to stop cleanly
bash "$HERE/segment.sh" "$RUN" & SP=$!
trap 'kill -USR1 $SP 2>/dev/null' USR1
while :; do
  wait $SP; RC=$?
  if [ $RC -gt 128 ] && kill -0 $SP 2>/dev/null; then continue; fi
  if [ $RC -gt 128 ]; then wait $SP 2>/dev/null; R2=$?; [ $R2 -ne 127 ] && RC=$R2; fi
  break
done
echo "$SLURM_JOB_ID $(date "+%Y-%m-%dT%H:%M:%S") segment exit $RC" >> "$RUNS/$RUN/jobs.txt"
if [ "$RC" -eq 0 ] && [ ! -f "$RUNS/$RUN/FINISHED" ] && [ ! -f "$RUNS/$RUN/NO_RESUBMIT" ]; then
  J=$(bash "$HERE/submit.sh" "$RUN" --quiet)
  echo "$SLURM_JOB_ID resubmitted as $J" >> "$RUNS/$RUN/jobs.txt"
fi
