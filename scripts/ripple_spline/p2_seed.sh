# Island seeds in the run's basis (q = 2 window, q = 3 window, all helical current = vacuum drive and the b21 phase).
set -u; source scripts/ripple_spline/env.sh; source scripts/ripple_spline/cfg.sh
$PY scripts/meanforce/island_seed.py --eq $EQ --LMN $LMN --frac $FRAC --N-phi 25 $SEEDB \
    --window 0.6375 0.07 0.03 --window 0.9131 0.03 0.015 --window -1 0 0 --cg 800 --out $O/seed
ls -la $O/seed*
