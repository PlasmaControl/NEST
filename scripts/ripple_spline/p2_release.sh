# Release test: the beta = $1 state with the pin removed (nu, rho reset), alpha 0.1, per-outer snapshots.
set -u; source scripts/ripple_spline/env.sh; source scripts/ripple_spline/cfg.sh
B=$1
$PY -c "import numpy as np; d=np.load('$O/b$B.npz'); np.savez('$O/rel_in$B.npz', u=d['u'], c_p=d['c_p'])"
$PY scripts/meanforce/mf_fixed.py $RUN --alphas ${BETA_ALPHA:-0.1} --start $O/rel_in$B.npz --max-outer 12 --snap-outer $SOLVER \
    --time-budget ${BUDGET_REL:-2400} --out $O/rel$B
S=$(pick $O/rel$B ${BETA_ALPHA:-0.1}); cp "$S" $O/release$B.npz
