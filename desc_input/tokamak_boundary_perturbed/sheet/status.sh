#!/bin/bash
# one line per sheet job: id, name, state, last [t] line or last iteration line
cd /scratch/gpfs/EKOLEMEN/lk1158/NEST_sheet
squeue -u lk1158 -h -o "%i %j %T %M" | grep " sh_" > /tmp/lk_sq.txt
for f in $(ls -t logs/nest-*.log | head -${1:-30}); do
  id=$(basename $f .log | cut -d- -f2)
  st=$(grep "^$id " /tmp/lk_sq.txt | cut -d" " -f3,4); st=${st:-DONE}
  nm=$(sacct -j $id -n -X -o jobname%12 2>/dev/null | tr -d " ")
  last=$(grep -E "^\[ +[0-9.]+ s\]|^ +[0-9]+ +[0-9]+ +[0-9.e+-]+ |Traceback|Error|RESOURCE_EXHAUSTED|Out of memory" $f | grep -v "cpu_aot\|pjrt" | tail -1 | cut -c1-150)
  echo "$id $nm [$st] $last"
done
