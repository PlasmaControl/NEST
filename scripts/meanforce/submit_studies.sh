#!/bin/bash
# The convergence studies of docs/mean_force_poisson_plan.md (sections 9, 10) as independent
# Stellar-AI jobs, each a full alpha sweep from DESC's state (run from the NEST copy's root).
#   study 1: annulus a/20, a/60, a/80, QR and GLTR, NEST 16/16/12
#   study 2: GLTR at L = M = 20, 24, 28 (N = 12) and 16/16/16, annulus a/40
#   probe:   a/100, a/120: setup, memory and one GLTR step (before their full runs)
set -euo pipefail
ALPHAS="0.7 0.5 0.3 0.2 0.1 0.05"
sub() {  # sub NAME QOS TIME ARGS...: one mf_run.py job
  local name=$1 qos=$2 t=$3
  shift 3
  sbatch --qos=$qos --time=$t --job-name=$name run_sai_cyl.sbatch scripts/meanforce/mf_run.py "$@"
}
for f in 20 60 80; do
  for m in qr gltr; do
    sub s1_${m}_a$f gpu-medium 24:00:00 --method $m --alphas $ALPHAS --frac $f --chunk 32 \
      --out results/meanforce/s1_${m}_a$f
  done
done
for lm in 20 24 28; do
  fac=$(python3 -c "print($lm / 16)")
  sub s2_L$lm gpu-medium 24:00:00 --method gltr --alphas $ALPHAS --frac 40 --chunk 16 \
    --factors $fac $fac 1 --out results/meanforce/s2_gltr_L${lm}N12
done
sub s2_N16 gpu-medium 24:00:00 --method gltr --alphas $ALPHAS --frac 40 --chunk 16 \
  --factors 1 1 1.3333333333333333 --out results/meanforce/s2_gltr_L16N16
for f in 100 120; do
  sub probe_a$f gpu-short 04:00:00 --method gltr --alphas 0.7 --frac $f --chunk 16 --maxiter 2 \
    --max-outer 1 --hessian-k 0 --out results/meanforce/probe_a$f
done
