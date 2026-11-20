#!/bin/bash
# Submit split sheet diagnostics for every snapshot (out/*_it*.h5) that has no diag yet.
# One marker per file (diag/<tag>.submitted). Parts: 4 for L<=64 (gpu-test), 8 for larger (gpu-short, 2 h).
cd /scratch/gpfs/EKOLEMEN/lk1158/NEST_sheet
for F in $(ls -tr out/*_it*.h5 2>/dev/null); do
  tag=$(basename $F .h5)
  [ -e diag/$tag.submitted ] && continue
  L=$(echo $tag | sed -E "s/.*_L([0-9]+)N.*/\1/")
  if [ $L -le 64 ]; then NP=4; Q=gpu-test; T=00:30:00; else NP=4; Q=gpu-short; T=03:00:00; fi
  ids=""; ok=1
  for i in $(seq 0 $((NP-1))); do
    j=$(sbatch --parsable --qos=$Q --time=$T -J sh_g${L}_$i run_sai.sbatch sheet_diag.py $F --part $i --nparts $NP --chunk 16 --out diag 2>/dev/null) || { ok=0; break; }
    ids="$ids:$j"
  done
  if [ $ok = 0 ]; then [ -n "$ids" ] && scancel ${ids//:/ }; echo "no slots for $tag"; break; fi
  m=$(sbatch --parsable --dependency=afterok$ids --qos=gpu-medium --time=02:00:00 -J sh_m$L run_sai.sbatch sheet_diag.py $F --nparts $NP --merge --out diag 2>/dev/null) || m=FAILED
  echo "$ids merge $m" > diag/$tag.submitted; echo "submitted $tag: parts$ids merge $m"
done
# retry merges that could not be submitted
for S in diag/*.submitted; do
  grep -q "merge FAILED" $S || continue
  tag=$(basename $S .submitted); [ -e diag/${tag}_sheet.json ] && continue
  NP=$(ls diag/${tag}_part*of*.npz 2>/dev/null | head -1 | sed -E "s/.*of([0-9]+)\.npz/\1/"); [ -z "$NP" ] && continue
  [ $(ls diag/${tag}_part*of$NP.npz | wc -l) -lt $NP ] && continue
  m=$(sbatch --parsable --qos=gpu-test --time=00:20:00 -J sh_m run_sai.sbatch sheet_diag.py out/$tag.h5 --nparts $NP --merge --out diag 2>/dev/null) || continue
  sed -i "s/merge FAILED/merge $m/" $S; echo "submitted merge $tag $m"
done
