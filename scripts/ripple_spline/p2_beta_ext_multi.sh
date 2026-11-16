# Several p2_beta_ext.sh runs in sequence: bash p2_beta_ext_multi.sh 0.05 0.2
set -u
for B in "$@"; do bash scripts/ripple_spline/p2_beta_ext.sh $B; done
