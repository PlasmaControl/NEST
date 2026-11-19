# Using the exact B·n = 0 boundary constraint in fixed-boundary NEST (`mf_fixed.py`)

Branch `energy-objective`. This note is for collaborators who want to run fixed-boundary NEST with the plasma
boundary held as a magnetic surface. It is self-contained; background and results are in
`docs/energy_objective_status_2026-11-18.md` (sections 5-6) and `docs/energy_objective_status_2026-11-19.md`.

## 1. Why

Fixed-boundary NEST represents the plasma current J inside DESC's last closed flux surface (LCFS) and computes B by
Biot-Savart. Nothing in the original objective stops the solution from putting magnetic flux through the LCFS. In
the ripple test case this "leak" is not noise: the unconstrained force-floor state changes the normal field on the
LCFS by the equivalent of a 0.29 mm boundary displacement, mostly in toroidal mode n = 1, including -0.21 mm in the
(2,1) harmonic that is the applied ripple drive (+1.20 mm). In other words the optimizer weakens the external
perturbation it is supposed to respond to. Forbidding the leak raises the force floor from 0.062 to 0.58 M_DESC
(40/40/6, field grid a/100). Results obtained without the constraint, in particular low force floors and island
vs shielding comparisons, should be read with this in mind.

## 2. The flags

| flag | what it does | notes |
|---|---|---|
| `--bn-modes MC NC` | exact linear constraint: the Fourier modes \|m\| <= MC, \|n\| <= NC of f = sqrt(g) dB·e^rho on the LCFS vanish, with dB = B[J] - B[J_fit] | use the basis limits (MC = M, NC = N of `--LMN`); lower cutoffs leave a leak in the modes above the cutoff and give a lower, contaminated floor (4/2: 0.35, 8/4: 0.48, 16/6: 0.54, 40/6: 0.58 at 40/40/6) |
| `--pin-flux` | exact: the toroidally averaged toroidal flux stays at J_fit's | together with `--bn-modes` and the fixed net current I_p (built into the representation), the field outside the plasma is fixed |
| `--pin-current` | exact: the m = 0 modes of the potential `a` stay at J_fit's (enclosed toroidal current on every DESC surface) | optional; not needed for B·n work |
| `--freeze-n0-band R1 R2 --freeze-tau TAU --freeze-ref STATE` | exact: the toroidally symmetric (n = 0) part of J outside rho in [R1, R2] stays at STATE's values | experimental, for energy studies (section 5) |
| `--energy-weight LAMBDA` | adds LAMBDA (W - W(J_fit))/W(J_fit) to the objective (plasma-volume magnetic energy) | needs `--no-share-stop`; meaningful as a free energy only with `--bn-modes` and `--pin-flux` |

Implementation: `nest/meanforce_fixed.py` (`LCFSField`: B_plasma interpolated to rho = 1 on the collocation planes,
the sine-mode projection of f, and the constraint rows via the transpose of the field map; `LinearConstraints`:
projected trust-region metric plus `restore`). `scripts/meanforce/mf_fixed.py` wires the flags.

## 3. Minimal command (ripple example, as used for all results here)

```
python scripts/meanforce/mf_fixed.py --eq <DESC.h5> --bext desc-fit --LMN 40 40 6 --frac 100 --N-phi 25 \
    --colloc-rho 2.7 --bn-modes 40 6 --pin-flux \
    --subspace prev,accel --cg 1500 --no-rebuild-on-hit --f-noise auto --radius-by-delta --no-share-stop \
    --locate-chunk 131072 --diag-every 5 --save-every 5 \
    --alphas 0.8 --max-outer 8 --maxiter 60 --time-budget 3300 [--start state.npz] --out <out/TAG> [--mirror <dir>]
```

At 40/40/6 the B·n block is 526 rows; building it takes about a minute on an A100. The log prints the constraint
residuals every step (`[constraints] dPsi/Psi ..., Bn max|C du - d| ...`); they should stay at 1e-17 or below.
`[diag]` lines report `B.n max|b| ... rms ...` from a real field evaluation.

## 4. Pitfalls

1. **Starting from an old (leaky) state.** A `--start` state that violates the constraint is projected onto it by
   `LinearConstraints.restore`, which minimizes the change in the current-norm metric and ignores force. The
   projected state can have a much larger force error than the original (R1f shielded state: 0.06 -> 34 M_DESC;
   an island state: 0.08 -> ~425 M_DESC). The solver recovers, but within a time budget the end state may be
   path-dependent. Starting from J_fit satisfies every constraint exactly and needs no projection.
2. **The constrained force floor is much higher than the old floors** (0.58 vs 0.06 M_DESC at 40/40/6, a/100).
   Pick `--alphas` above the constrained floor, or the bound stays infeasible and the run stalls at the floor.
3. **Do not lower the B·n cutoff to recover a low floor.** The modes the leak uses are low-order n = 1 shape changes
   of the boundary; a cutoff like 4/2 just pushes the leak to m = 5-9 (same 0.26 mm displacement).
4. **Measuring the true leak.** The constraint acts on the annulus Biot-Savart field the solver uses. The direct
   Biot-Savart field differs by the grid error (~3e-7 to 7e-6 of the field scale at a/100); `scripts/energy/
   bn_spectrum.py` (with `--dump`) and `scripts/energy/bn_char.py` report the direct-field leak per (m, n), the
   equivalent boundary displacement in mm, and the energy error of using the plasma-volume energy.
5. **Saved-state names.** `mf_fixed` names the end state `<out>_a<alpha as float>.npz` (e.g. `_a3.0.npz`, `_a1.5.npz`);
   use a glob when scripting (see `lastst` in `scripts/modal_fixed/energy/job_g12.sh`).
6. With the force bound slack (alpha above the current force), `--subspace ...accel` used to raise inside its
   Lanczos solve and silently disable the whole subspace search; fixed in 3e01af8 (the accel direction is skipped
   and logged as `[subspace] accel direction skipped`).

## 5. Validation that was done

- CPU 8/8/2 (`scripts/energy/g0_checks.py`): constraint rows vs field evaluation 4e-16; adjoints <= 1.5e-14; energy
  gradient vs finite differences <= 1.6e-10; constraint kept to 3.5e-18 over 50 projected steps; the solver end state
  equals the closed-form constrained optimum to 1.7e-9 (re-run after the restore change, unchanged); with all new
  flags off the run is bit-identical to 6249d4e (verified at the build commit d7948ed; later changes touch only the
  constrained `restore` and the `--subspace` accel failure path).
- GPU 40/40/6 (`scripts/energy/exterior_check.py`, direct Biot-Savart): the constraint cuts the exterior field change
  30-60x at 0.1a, 0.5a and far points.
- `--freeze-n0-band`: `scripts/energy/freeze_check.py` (random feasible n = 0 directions put <= tau of their current
  norm outside the band); `restore` normalizes rows and refines iteratively so dependent B·n / freeze rows are
  satisfied exactly (G0 unchanged).

## 6. Job scripts (Modal; the mf_fixed command line carries over to Slurm unchanged)

`scripts/modal_fixed/energy/job_leak.sh` (leak study: `spec`, `char`, floor pushes `A0`, `Aflux`, `A4`-`A40`) and
`scripts/modal_fixed/energy/job_g12.sh` (energy work; env `ALPHA`, `TAU`; steps `S0`, `I<beta>`, `E<lambda>`,
`FZ<lambda>`, `FZALL<lambda>`, `FI<beta>`, `FR<lambda>`, `night2`). Both expect the Modal layout (/work, /results);
for a cluster, copy the `mf_fixed.py` command lines.
