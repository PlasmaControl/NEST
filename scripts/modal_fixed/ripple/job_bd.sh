#!/bin/bash
# beta-scan follow-up: (1) island/pressure/J_par diagnostics with Poincare for the shielded start and all 8 beta end states;
# (2) release tests: the beta 0.05 and 0.5 end states with the b21 pin removed (nu, rho reset), alpha 0.1 mean bound,
# per-outer snapshots, then b21 of every snapshot. 32/32/6 a/60.
cd /work; for f in desc_src.tar.gz deps_src.tar.gz ripple_L32N4_x1e-3.h5 gpu_entry.py; do [ -f $f ] || cp /results/stage/$f .; done
rm -rf src/scripts src/nest; mkdir -p src out
tar -xzf nest_src.tar.gz -C src
if [ ! -f src/.deps ]; then tar -xzf desc_src.tar.gz -C src && tar -xzf deps_src.tar.gz -C src && touch src/.deps; fi
touch src/.unpacked
EQ=/work/ripple_L32N4_x1e-3.h5; O=/results/fxr/bdiag; mkdir -p $O
cp /results/fxr/A60_cont/A60_cont_a0.1.npz /work/start.npz
ST="--state /work/start.npz"; PT="vacuum start:A"
for B in 0.02 0.05 0.1 0.2 0.35 0.5 0.75 1.0; do
  cp /results/fxr/bscan/bs${B}_budget.npz /work/b$B.npz && ST="$ST --state /work/b$B.npz" && PT="$PT b$B:A"
done
/opt/conda/bin/python - <<'PY' > /work/out/bd_dist.txt
import numpy as np, json
B=["0.02","0.05","0.1","0.2","0.35","0.5","0.75","1.0"]
u0=np.load("/work/start.npz")["u"]; U={b:np.load(f"/work/b{b}.npz")["u"] for b in B}
D={b:U[b]-u0 for b in B}; n0=np.linalg.norm(u0)
out={"|u_start|":float(n0)}
for i,b in enumerate(B):
    r={"|u_b-u_start|/|u_start|":float(np.linalg.norm(D[b])/n0)}
    if i: 
        pb=B[i-1]; r["|u_b-u_prev|/|u_start|"]=float(np.linalg.norm(U[b]-U[pb])/n0)
        r["cos(d_b, d_prev)"]=float(D[b]@D[pb]/np.linalg.norm(D[b])/np.linalg.norm(D[pb]))
    r["cos(d_b, d_0.02)"]=float(D[b]@D["0.02"]/np.linalg.norm(D[b])/np.linalg.norm(D["0.02"]))
    r["cos(d_b, d_1.0)"]=float(D[b]@D["1.0"]/np.linalg.norm(D[b])/np.linalg.norm(D["1.0"]))
    out[b]=r; print(b, {k:round(v,4) for k,v in r.items()})
json.dump(out, open("/work/out/bd_dist.json","w"), indent=1)
PY
cp /work/out/bd_dist.* $O/
bash rung_run.sh bd_m2 scripts/meanforce/start_check_bext.py --eq $EQ --LMN 32 32 6 --frac 60 --N-phi 25 --m 2 --k 1 \
    --rho-grid 0.30 0.98 341 --band 0.04 0.08 0.16 $ST --perts $PT --turns 300 --lines 41 --seed-band 0.50 0.78 \
    --seed-theta 0.0 --pressure --out /work/out/bd_m2
cp /work/out/bd_m2*.json /work/out/bd_m2*poincare*.npz $O/ 2>/dev/null
for B in 0.05 0.5; do
  /opt/conda/bin/python -c "import numpy as np; d=np.load('/work/b$B.npz'); np.savez('/work/rel$B.npz', u=d['u'], c_p=d['c_p'])"
  bash rung_run.sh rel$B scripts/meanforce/mf_fixed.py --eq $EQ --bext desc-fit --LMN 32 32 6 --frac 60 --N-phi 25 \
      --alphas 0.1 --start /work/rel$B.npz --max-outer 12 --snap-outer --f-noise auto --radius-by-delta --no-share-stop \
      --time-budget 1200 --out /work/out/rel$B --mirror $O
done
EV=""
for f in $(ls /work/out/rel*_o*.npz /work/out/rel*_budget.npz 2>/dev/null); do n=$(basename $f .npz); EV="$EV --eval-state $n=$f"; done
for B in 0.05 0.5; do EV="$EV --eval-state b$B=/work/b$B.npz"; done
bash rung_run.sh bd_b21 scripts/meanforce/mf_fixed.py --eq $EQ --bext desc-fit --LMN 32 32 6 --frac 60 --N-phi 25 \
    --alphas 1 --model beta --force-norm p --p-schedule 16 --p-band 0.5675 0.7075 --eval-only --b21-report $EV \
    --out /work/out/bd_b21 --mirror $O
true
