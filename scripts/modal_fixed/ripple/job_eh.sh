#!/bin/bash
# Magnetic/thermal energy, helicity and fluxes of the ripple states (energy_helicity.py): R1 family (40/40/6, a/100)
# and R0 family (32/32/6, a/80, including the original beta scan and the shielded start).
set -u
cd /work; for f in desc_src.tar.gz deps_src.tar.gz ripple_L32N4_x1e-3.h5 gpu_entry.py; do [ -f $f ] || cp /results/stage/$f .; done
rm -rf src/scripts src/nest; mkdir -p src out
tar -xzf nest_src.tar.gz -C src
if [ ! -f src/.deps ]; then tar -xzf desc_src.tar.gz -C src && tar -xzf deps_src.tar.gz -C src && touch src/.deps; fi
touch src/.unpacked
EQ=/work/ripple_L32N4_x1e-3.h5; O=/results/fxr/res/eh; mkdir -p $O
R1=/results/fxr/res/R1; G=/results/fxr/res/R1g; R0=/results/fxr/res/R0; BS=/results/fxr/bscan
S1="--state R1_S=$R1/R1_S_a0.1.npz --state R1_b0.2=$R1/R1_b0.2_budget.npz --state R1_b0.5=$R1/R1_b0.5_a0.1.npz \
    --state R1_b1.0=$R1/R1_b1.0_a0.1.npz --state R1_rel0.5=$R1/R1_rel0.5_a0.1.npz \
    --state R1f_S=$G/R1f_S_a0.05.npz --state R1f_b1.0=$G/R1f_b1.0_a0.05.npz --state R1c_S=$G/R1c_S_a0.05.npz --state R1c_b1.0=$G/R1c_b1.0_a0.05.npz"
bash rung_run.sh eh_R1 scripts/meanforce/energy_helicity.py --eq $EQ --LMN 40 40 6 --frac 100 --N-phi 25 $S1 --out /work/out/eh_R1.json
cp /work/out/eh_R1.json $O/ 2>/dev/null
S0="--state start=/results/fxr/A60_cont/A60_cont_a0.1.npz"
for B in 0.02 0.05 0.1 0.2 0.35 0.5 0.75 1.0; do S0="$S0 --state bs$B=$BS/bs${B}_budget.npz"; done
for T in S b0.2 b0.5 b1.0 rel0.5; do S0="$S0 --state R0_$T=$R0/R0_${T}_budget.npz"; done
bash rung_run.sh eh_R0 scripts/meanforce/energy_helicity.py --eq $EQ --LMN 32 32 6 --frac 80 --N-phi 25 $S0 --out /work/out/eh_R0.json
cp /work/out/eh_R0.json $O/ 2>/dev/null; true
