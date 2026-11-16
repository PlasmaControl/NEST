# Seeds, then the alpha continuation to the shielded start (one job).
set -u
bash scripts/ripple_spline/p2_seed.sh || exit 1
bash scripts/ripple_spline/p2_cont.sh
