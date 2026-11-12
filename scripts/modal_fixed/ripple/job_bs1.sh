#!/bin/bash
# beta scan (part 1): hold the q = 2 resonant field b21 at beta x vacuum (island phase from the all-helical seed), start
# from the shielded start projected onto the constraint, alpha 0.1 (mean bound), equal time budget; then island,
# pressure and J_par diagnostics of the end states. 32/32/6 a/60 (the shielded start's settings).
cd /work; for f in desc_src.tar.gz deps_src.tar.gz ripple_L32N4_x1e-3.h5 gpu_entry.py; do [ -f $f ] || cp /results/stage/$f .; done
rm -rf src/scripts src/nest; mkdir -p src out
tar -xzf nest_src.tar.gz -C src
if [ ! -f src/.deps ]; then tar -xzf desc_src.tar.gz -C src && tar -xzf deps_src.tar.gz -C src && touch src/.deps; fi
touch src/.unpacked
EQ=/work/ripple_L32N4_x1e-3.h5; O=/results/fxr/bscan; mkdir -p $O
cp /results/fxr/A60_cont/A60_cont_a0.1.npz /work/start.npz; cp /results/fxr/pre32/seed32_all.npz /work/sall.npz
ST=""; PT="vacuum start:A"
for B in 0.02 0.1 0.35 0.75; do
  T=bs$B
  bash rung_run.sh $T scripts/meanforce/mf_fixed.py --eq $EQ --bext desc-fit --LMN 32 32 6 --frac 60 --N-phi 25 \
      --alphas 0.1 --start /work/start.npz --pin-b21 $B --b21-phase-state /work/sall.npz --max-outer 12 --stage-flat 0.02 \
      --f-noise auto --radius-by-delta --no-share-stop --time-budget 1800 --out /work/out/$T --mirror $O
  F=$(ls -t /work/out/${T}_a*.npz 2>/dev/null | grep -v latest | head -1)
  [ -n "$F" ] && cp $F /work/out/b$B.npz && ST="$ST --state /work/out/b$B.npz" && PT="$PT b$B:A"
done
bash rung_run.sh bsdiag1 scripts/meanforce/start_check_bext.py --eq $EQ --LMN 32 32 6 --frac 60 --N-phi 25 --m 2 --k 1 \
    --rho-grid 0.30 0.98 341 --band 0.04 0.08 0.16 --state /work/start.npz $ST --perts $PT --turns 300 --lines 41 --seed-band 0.50 0.78 \
    --seed-theta 0.0 --pressure --out /work/out/bsdiag1
cp /work/out/bsdiag1*.json /work/out/b*.npz $O/ 2>/dev/null; true
