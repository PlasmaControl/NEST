# Common settings of the ripple spline study (sourced by the job scripts; run from the repository root).
# values with spaces can be passed through job lists with underscores (LMN=40_40_6, ALPHAS=0.1_0.05, B3=0.84_0.96_0.0075)
for v_ in LMN ALPHAS B2 B3 BE RES_BETAS LOC_ALPHAS SPLX CONT_EXTRA; do [ -n "${!v_:-}" ] && eval "$v_=\"\${$v_//_/ }\""; done
PY=${PY:-/scratch/gpfs/EKOLEMEN/lk1158/envs/nest13/bin/python}
EQ=desc_input/tokamak_boundary_perturbed/bp2/L32N4/tokamak_ripple_C_L32M32N4_x0.001.h5
ST=results/ripple_landscape/states
export NEST_CACHE=${NEST_CACHE:-$PWD/.cache}
# the report's run settings (32/32/6 option A, a/60, N_phi 25)
BASE="--eq $EQ --bext desc-fit --frac 60 --N-phi 25"
# eval-only measures (band p16 of the report, b21, force split)
EVAL="--alphas 1 --model beta --force-norm p --p-schedule 16 --p-band 0.5675 0.7075 --eval-only --b21-report --force-split"
# spline knots: FZ spacing at L = 32 (x KC) outside, 0.01 over the q = 2 island band, 0.015 over q = 3; rho^q_m regularity
KNOTS="--knots-fz 32 ${KC:-1} --spl-band 0.45 0.76 ${HB2:-0.01} --spl-band 0.86 0.95 ${HB3:-0.015} --spl-reg ${SPL_REG:-rhom} --spl-qcap 6"
SOLVER="--f-noise auto --radius-by-delta --no-share-stop --cg ${CG:-1500}"  # cap 400 made every cap hit a 10-16 min metric rebuild
# hybrid: FZ 32/32/6 + B-splines localized in the q = 2 band (0.45-0.76, spacing 0.01) and the q = 3 band
# (0.86-0.95, 0.015), |m| <= 16; the same bands give the radial collocation (--colloc-knots)
HYB="--spl-band ${B2:-0.45 0.76 ${HB2:-0.01}} --spl-band ${B3:-0.86 0.95 ${HB3:-0.015}} ${BE:+--spl-band $BE} --spl-mmax ${HMMAX:-16}"
# FZ resolution of the hybrid / FZ runs (LMN), and the basis of --start when it differs (START_LMN=1: the report's 32 32 6)
LMN=${LMN:-32 32 6}
