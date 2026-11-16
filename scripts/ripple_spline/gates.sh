# Gates of the spline bases on the ripple equilibrium (tests/*), one GPU.
set -u
source scripts/ripple_spline/env.sh
rc=0
$PY tests/test_ripple_spline_gates.py --eq $EQ || rc=1
for t in test_spline_current test_spline_pressure test_spline_trunc; do
  echo "=== $t"; $PY tests/$t.py --eq $EQ || rc=1
done
echo "gates exit $rc"
