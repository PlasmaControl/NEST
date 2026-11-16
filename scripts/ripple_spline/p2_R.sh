# Release tests, diagnostics and the common-grid comparison (after all beta runs).
set -u
bash scripts/ripple_spline/p2_release.sh 0.05
bash scripts/ripple_spline/p2_release.sh 0.5
bash scripts/ripple_spline/p2_eval.sh
bash scripts/ripple_spline/p3_common.sh
