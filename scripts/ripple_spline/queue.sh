#!/bin/bash
# Submit the jobs of a list as QOS slots free (run with nohup on the login node):
#   nohup bash scripts/ripple_spline/queue.sh scripts/ripple_spline/jobs_followup.txt > logs/queue.log 2>&1 &
# Line format: NAME|TIME|AFTER(NAME,NAME or -)|ENV (space separated K=V)|SCRIPT ARGS
# QOS follows the time limit on Stellar-AI (> 4 h: gpu-medium, cap 10 submitted per user; else gpu-short, cap 10).
cd /scratch/gpfs/EKOLEMEN/lk1158/NEST_rspl
LIST=$1; IDS=${LIST%.txt}.ids; touch $IDS
cap=${CAP:-9}
while true; do
  pending=0
  while IFS='|' read -r name tl after envs cmd; do
    [ -z "$name" ] || [ "${name:0:1}" = "#" ] && continue
    grep -q "^$name " $IDS && continue
    pending=1
    dep=""
    if [ "$after" != "-" ]; then
      ok=1; ids=""
      for a in ${after//,/ }; do j=$(awk -v n=$a '$1==n {print $2}' $IDS); [ -z "$j" ] && ok=0; ids="$ids:$j"; done
      [ $ok = 0 ] && continue
      dep="--dependency=afterany$ids"
    fi
    h=${tl%%:*}; q=gpu-short; [ $((10#$h)) -ge 4 ] && [ "$tl" != "04:00:00" ] && q=gpu-medium
    n=$(squeue -u lk1158 -h -q $q | wc -l)
    [ $n -ge $cap ] && continue
    j=$(env $envs sbatch --parsable --export=ALL $dep --job-name=$name --time=$tl run_sai_bash.sbatch $cmd)
    [ -n "$j" ] && echo "$name $j" >> $IDS && echo "$(date '+%F %T') submitted $name $j ($q, $tl) $dep"
  done < $LIST
  [ $pending = 0 ] && { echo "$(date '+%F %T') all submitted"; break; }
  sleep 180
done
