#!/bin/bash
# L-only resolution test: 96/64/6 warm-started from P64 (64/64/6, alpha 0.05), P64's solve flags otherwise.
# Then fine-radius eval (a/200, colloc-rho 3, eval-dump) and current-structure packing (pack_cur.py).
set -u
cd /work; for f in desc_src.tar.gz deps_src.tar.gz ripple_L32N4_x1e-3.h5 gpu_entry.py; do [ -f $f ] || cp /results/stage/$f .; done
mkdir -p src out; tar -xzf nest_src.tar.gz -C src
T0=$(date +%s); p() { echo "[l96] $1 at $(( $(date +%s) - T0 )) s" | tee -a /results/fxr/l96/progress.txt; }
mkdir -p /results/fxr/l96
p "start"
bash rung_run.sh S96 scripts/meanforce/mf_fixed.py --eq /work/ripple_L32N4_x1e-3.h5 \
  --coils /work/src/desc_input/vmec_free_boundary/NCSX/NCSX_coils.h5 --bext desc-fit --LMN 96 64 6 --colloc 1.5 \
  --frac 160 --N-phi 25 --pad 0.02 --mode-chunk 16 --rblock 32 --interp-order 8 --alphas 0.05 --rho0 1.0 --band 0.05 \
  --maxiter 40 --max-outer 1 --gtol 0.001 --cg 400 --test-delta 0.016 --line-tmax 2.0 --test-model gn --precond force \
  --hvp-far 0.1 --hvp-fallback 5 --beta 0.5 --outer-rebuild-factor 2.0 --beta-min 0.03125 --model beta --cg-tol 0.01 \
  --accel-amax 0.75 --accel-tol 0.0001 --accel-maxiter 300 --radius-by-delta --f-noise auto --no-share-stop \
  --stat-tol 0.001 --stat-window 10 --delta0 0.001 --gram-chunk 1024 --time-budget 4800 \
  --start /work/P64_a0.05.npz --start-LMN 64 64 6 --lanczos-stop tr --force-norm mean --snap-outer \
  --locate-chunk 131072 --mem-every 1 --b21-mk 2 1 --b21-vac 0.0003778 --hvp --diag-every 5 --save-every 5 \
  --out /work/out/S96 --mirror /results/fxr/l96/S96
p "solve exit $?"
ST=""; for c in S96_a0.05 S96_budget S96_a0.05_latest; do [ -f /work/out/$c.npz ] && { ST=/work/out/$c.npz; break; }; done
p "state for eval: $ST"
[ -n "$ST" ] || exit 3
bash rung_run.sh E96 scripts/meanforce/mf_fixed.py --eq /work/ripple_L32N4_x1e-3.h5 --bext desc-fit --N-phi 25 --frac 200 \
  --locate-chunk 131072 --colloc-rho 3 --alphas 1 --model beta --force-norm p --p-schedule 16 --p-band 0.5675 0.7075 \
  --eval-only --eval-dump --LMN 96 64 6 --start $ST --out /work/out/E96 --mirror /results/fxr/l96/E96
p "eval exit $?"
(cd /work/src && PYTHONPATH=/work/src/nest:/work/src NEST_DEVICE=gpu /opt/conda/bin/python /work/pack_cur.py E96) 2>&1 | tail -3
cp /work/out/cur_E96.npz /results/fxr/l96/ 2>/dev/null
rm -f /work/out/E96_dump.npz
p "done"; ls -la /work/out
