# Submit the Phase 2 chain of one configuration (run on sai from the repository root), 6 jobs:
#   A (seeds + shielded start) -> 4 x (two pinned-beta runs) -> R (releases, diagnostics, common grid).
# QOS follows the time limit on Stellar-AI (> 4 h: gpu-medium): T_A, T_BETA, T_R.
#   KIND=hybrid TAG=h1 START=results/.../shielded_start_A60_a0.1.npz START_LMN=1 ALPHAS=0.1 bash scripts/ripple_spline/submit_p2.sh
set -eu
J="sbatch --parsable --export=ALL"
A=$($J --job-name=${TAG}_A --time=${T_A:-05:00:00} run_sai_bash.sbatch scripts/ripple_spline/p2_A.sh)
BS=""
for P in "0.02 1.0" "0.05 0.75" "0.1 0.5" "0.2 0.35"; do
  j=$($J --dependency=afterok:$A --job-name=${TAG}_b${P// /_} --time=${T_BETA:-04:30:00} \
      run_sai_bash.sbatch scripts/ripple_spline/p2_betas.sh $P)
  BS="$BS:$j"
done
R=$($J --dependency=afterany$BS --job-name=${TAG}_R --time=${T_R:-06:00:00} run_sai_bash.sbatch \
    scripts/ripple_spline/p2_R.sh)
echo "$TAG: A $A betas ${BS#:} R $R"
