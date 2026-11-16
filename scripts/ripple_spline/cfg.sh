# Run configuration (sourced after env.sh). KIND: spline (pure B-splines, knots KNOTS + SPLX), hybrid (FZ 32/32/6 +
# band splines HYB), fz (FZ 32/32/6 on the hybrid's radial collocation: the control). TAG names the run directory.
TAG=${TAG:-s1}
FRAC=${FRAC:-100}
KIND=${KIND:-spline}
SPLX=${SPLX:---spl-mfz 32}
case $KIND in
  spline) BAS="--current spline --pressure spline --LMN 32 32 6 $KNOTS $SPLX"
          SEEDB="--current spline --p-basis spline $KNOTS $SPLX" ;;
  hybrid) BAS="--current fz+spline --pressure fz+spline --LMN $LMN $HYB"
          SEEDB="--current fz+spline --p-basis fz+spline $HYB" ;;
  fz)     BAS="--LMN $LMN $HYB --colloc-knots"
          SEEDB="$HYB --colloc-knots" ;;
esac
RUN="--eq $EQ --bext desc-fit --frac $FRAC --N-phi 25 $BAS"
O=out/rs/$TAG; mkdir -p $O
# the state of a run: its stage file if the stage ended, else its time-budget file, else its latest outer
pick() { for f in "$1_a$2.npz" "$1_budget.npz" "$1_a$2_latest.npz"; do [ -f "$f" ] && { echo "$f"; return; }; done; }
