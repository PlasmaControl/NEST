# Several pinned-beta runs in sequence: bash p2_betas.sh 0.02 1.0
set -u
for B in "$@"; do bash scripts/ripple_spline/p2_beta.sh $B; done
