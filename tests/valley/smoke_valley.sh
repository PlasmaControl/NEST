#!/bin/bash
# Local CPU smoke test of the valley-navigation options (geodesic acceleration, exact subspace search).
# 8/8/4 a/20 ripple case, alpha 0.01 from J_fit, 12 inner steps per arm. Checks:
#   (1) default path unchanged: control step lines bit-identical to a reference tree (REF, e.g. a git archive of HEAD);
#   (2) [subspace] prediction vs real evaluation rel < 1e-10 and straight-step reconstruction < 1e-12;
#   (3) every arm runs 12 steps without error.
# usage: OUT=/tmp/valley REF=/path/to/ref_tree bash tests/valley/smoke_valley.sh   (run from the repo root)
set -u
OUT=${OUT:?}; REF=${REF:-}; S=$(pwd); mkdir -p $OUT/cache
export NEST_CACHE=$OUT/cache JAX_PLATFORMS=cpu JAX_ENABLE_X64=1 PYTHONUNBUFFERED=1
ARGS="--eq desc_input/tokamak_boundary_perturbed/bp2/L32N4/tokamak_ripple_C_L32M32N4_x0.001.h5 --bext desc-fit --LMN 8 8 4
      --frac 20 --N-phi 17 --alphas 0.01 --max-outer 1 --maxiter 12 --f-noise auto --radius-by-delta --no-share-stop --lanczos-stop tr"
run() { TAG=$1; SRC=$2; shift 2; PYTHONPATH=$SRC/nest:$SRC:${DESC_SRC:-$S/../../DESC} python $SRC/scripts/meanforce/mf_fixed.py $ARGS \
        --out $OUT/$TAG "$@" > $OUT/$TAG.log 2>&1 || { echo "FAIL $TAG"; exit 1; }; }
run ctl $S
[ -n "$REF" ] && run ref $REF
run acc $S --geo-accel
run subP $S --subspace prev
run subPA $S --subspace prev,accel
fail=0
if [ -n "$REF" ]; then
  diff <(grep -E "    it " $OUT/ctl.log | sed 's/ [0-9.]*s$//') <(grep -E "    it " $OUT/ref.log | sed 's/ [0-9.]*s$//') >/dev/null \
    && echo "PASS default path identical to REF" || { echo "FAIL default path differs from REF"; fail=1; }
fi
for T in ctl acc subP subPA; do n=$(grep -c "    it " $OUT/$T.log); [ "$n" -eq 12 ] && echo "PASS $T 12 steps" || { echo "FAIL $T $n steps"; fail=1; }; done
python - "$OUT" <<'PY' || fail=1
import re, sys, glob
bad = 0
for f in glob.glob(sys.argv[1] + "/sub*.log"):
    for m in re.finditer(r"rel ([0-9.e+-]+); straight recon ([0-9.e+-]+)", open(f).read()):
        r, s = float(m.group(1)), float(m.group(2))
        if not (r < 1e-10 and s < 1e-12):
            print(f"FAIL {f}: prediction {r:.1e}, straight recon {s:.1e}"); bad = 1
print("PASS subspace exactness" if not bad else ""); sys.exit(bad)
PY
exit $fail
