# Resolution / band-layout chain (items 2, 3): seeds, shielded start from the report's FZ start, pinned beta runs
# (RES_BETAS, 12 outer iterations each: BUDGET_BETA large), then this configuration's states on the common grid.
set -u
bash scripts/ripple_spline/p2_seed.sh || exit 1
bash scripts/ripple_spline/p2_cont.sh || exit 1
for B in ${RES_BETAS:-0.05 0.2 1.0}; do bash scripts/ripple_spline/p2_beta.sh $B; done
bash scripts/ripple_spline/p3_common.sh
