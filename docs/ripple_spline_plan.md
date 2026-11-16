# Ripple landscape with B-spline T and p (branch `ripple-spline`, from `ripple-landscape`)

Goal: repeat the experiments of `docs/ripple_landscape_report.md` (shielded start, beta scan with b21 pinned, release
tests, plane maps, island and pressure diagnostics) with the current potential's a, b series and the pressure in radial
B-splines x Fourier instead of Fourier-Zernike, and compare the force landscape and the share of the force error in
b.grad p. Everything else is the report's setup (candidate C ripple equilibrium `bp2/L32N4 x0.001`, option A,
volume-mean bound alpha = 0.1, N_phi 25, noise-aware trust region).

## Representation

- **Same boundary conditions by construction.** Only the radial functions of a, b (``BoundaryPotential``) and of Q
  (p = (1 - rho^2)^3 Q) change; the window f = (1 - rho^2)^2, the chi terms, h_0 and the net current stay, so div J = 0,
  J = O(1 - rho) at the edge, I fixed and p = 0 on the edge hold for any coefficients (gate R3).
- **Axis regularity** (`--spl-reg rhom`): rho^q_m B_j(rho^2) scaled to unit maximum, q_m = |m| up to `--spl-qcap 6`, then
  capped with the parity of m (C^4 at the axis). The four axis-touching splines are kept for |m| <= 6 whatever the
  truncation (Zernike's low-order content; the m = 1 ~ rho of a shifted axis is exact, gate R2a). The old rule
  (`drop3`: drop the three axis splines) cannot represent the shifted axis near rho = 0 (error 5e-2).
- **Knots**: Fourier-Zernike's zero spacing at L = 32 times KC outside the bands (`--knots-fz 32 KC`: dense at the edge
  like FZ), and `--spl-band` bands 0.45-0.76 (the beta = 1 island 0.55-0.68 and both displaced lambda_21 peaks, 0.48
  and 0.70-0.73) and 0.86-0.95 (q = 3, where ~70% of the island's extra force sits).
- **Truncation**: `--spl-mfz 32` (drop m != 0 splines whose support ends before the first zero of FZ's top radial
  function of |m| at L = 32). `m_tol 1e-2 + m_max 16` costs 10-100x in the fit of DESC's J at rho 0.4-0.72 (knot
  scan): high-m content matters there.

## Gates (tests/test_ripple_spline_gates.py + the dshape-fixed spline tests on the ripple file): all pass

Fit of DESC's J, rms |J_fit - J_DESC| / rms J_DESC by rho band (0-0.4, 0.4-0.55, 0.55-0.72, 0.72-0.85, 0.85-0.95,
0.95-1), `out/knot_scan.json`:

| basis | n_u | by band |
|---|---|---|
| FZ 32/32/6 | 7726 | 2.5e-5 1.6e-5 1.4e-5 2.0e-5 4.5e-5 2.8e-4 |
| FZ 40/40/6 | 11734 | 1.9e-5 1.0e-5 8.9e-6 1.2e-5 3.0e-5 2.5e-4 |
| s1: KC 0.6, bands 0.015 / 0.02, m_fz 32 | 23694 | 7.1e-6 1.7e-5 4.1e-6 1.9e-5 2.6e-5 1.4e-4 |
| s2: KC 0.45, bands 0.01 / 0.015, m_fz 32 | 31212 | 2.3e-5 2.1e-5 4.7e-6 8.1e-6 5.5e-6 7.0e-5 |

Equal-count matching is not possible at FZ accuracy: cubic splines need ~3x FZ's unknowns to match FZ's spectral
accuracy away from the bands (KC = 1 with FZ spacing is worse than FZ 32/32/6 outside the bands). The resolution
control is therefore the hand-off's FZ 40/40/6 rung, not a count-matched spline.

## Phase 0: b.grad p in the FZ states, and spline p at fixed J

Force split (`--force-split`: |b.F| = |b.grad p| and the perpendicular part, volume means / M_DESC), report grid:

| state | M | parallel | perpendicular |
|---|---|---|---|
| shielded start | 0.124 | 0.079 | 0.079 |
| beta 0.2 | 0.179 | 0.118 | 0.106 |
| beta 1.0 | 0.246 | 0.154 | 0.160 |

The parallel force is half of M in every state; per volume it peaks at rho 0.9-0.95 (q = 3) and the edge, and grows in
the q = 2 band for the island states. Re-solving only the least-squares p of these states in spline p spaces leaves the
parallel part unchanged (0.080 -> 0.080-0.087) and raises the perpendicular part to 4-40 M_DESC (Galerkin 1e-16: the
solve is exact). The FZ states balance J x B against an FZ p to ~1e-7 relative, which cubic splines cannot reproduce:
the comparison has to optimize J with the spline p (Phase 2). For J_fit itself spline p lowers the force 4x (rms |F|
7.8e-6 -> 1.9e-6).

## Phase 2 (running): `scripts/ripple_spline/submit_p2.sh`, configurations s1 and s2 at a/100

A: island seeds (q = 2, q = 3, all helical) + alpha continuation 0.3 0.1 0.03 0.01 (budget 3.7 h) -> shielded start;
4 jobs x 2 pinned-beta runs (beta 0.02 ... 1.0, 90 min each including ~20 min setup); R: releases from beta 0.05 and
0.5 (60 min), diagnostics (b21, band p16, force split, shell max, the report's three planes, Poincare + pressure along
lines) and the common-grid comparison (`p3_common.sh`: FZ and spline states at a/120 on the same composite radial grid).

Solver note (first continuation, 2026-11-12): with the default Lanczos cap 400, every cap hit rebuilt the force
metric (10-16 min at n_u 24-31k); rebuilds took 75% (s1) and 90% (s2) of the trial time (s2: ~1000 s per trial). The
continuations were restarted from their last outer states with `--cg 1500` (`CG` in env.sh, all later runs) and a
7.5 h budget (`cont2`).

## Pure splines stall in the smooth core; switch to FZ + band-localized splines (2026-11-12, night)

The pure-spline continuations stalled at ~0.52 M_DESC (s1; FZ shielded start 0.124). Force split of the stalled state:
perpendicular force 0.6-1.1 M_DESC per volume at rho < 0.5 (FZ: 0.04), parallel 0.26 at rho < 0.4 (FZ: 0.017); in the
island band only ~4x FZ. The core error is already in J_fit (perpendicular 10-12 with KC 0.6, 75-88 with KC 1, at
rho 0.3-0.5) although the rms J fit there is better than FZ's (7e-6 vs 2.5e-5): the force at M ~ 0.1 M_DESC resolves
J to ~1e-7 relative, and the cubic-spline fit error oscillates on the knot scale, so its curl (which no p can balance)
is large, while FZ's spectral error is smooth. Neither the truncation nor the axis cap is the cause (q_cap 10, no
m_fz: same J_fit force). Pure cubic splines would need far finer knots everywhere to match FZ in the smooth regions.

Hybrid basis (`--current fz+spline --pressure fz+spline`, `nest/spline_basis.CompositeBasis`): FZ 32/32/6 plus
cubic B-splines in s = rho^2 whose support lies inside the refined bands (q = 2: 0.45-0.76 at 0.01; q = 3: 0.86-0.95 at
0.015; |m| <= 16, all n <= 6). Each spline and its first two derivatives vanish at the band edges, so the FZ part keeps
its spectral accuracy, axis regularity and edge refinement, and the boundary conditions are FZ's. n_u 21056 (FZ 7726),
n_p 10320 (FZ 3655). J_fit and the p fit are FZ's exactly (spline coefficients 0), so FZ states embed exactly and any
change comes from the band refinement alone. Gates `tests/test_hybrid_gates.py`.

Fixed J (the report's FZ states embedded, the same composite radial grid, a/60): the hybrid p lowers M by 6-9%
(shielded start 0.128 -> 0.120, beta 1: 0.255 -> 0.237 M_DESC), all from the perpendicular part; the parallel part is
unchanged to 3 digits (0.0803 -> 0.0803). At fixed J, b.grad p is set by J: only optimizing J can lower it.

Runs (a/100, both from the report's FZ shielded start, alpha 0.1, then the beta scan, releases, diagnostics, a/120
common grid): `h1` (hybrid) and `fzc` (FZ 32/32/6 on the hybrid's radial collocation: the control that separates the
grid from the basis). The pure-spline continuations (`s1/cont2`, `s2/cont2`) continue as the record of that basis.

Alpha targets: at a/100 both h1 and fzc met alpha = 0.1 within minutes (h1: 5 trials; the report's FZ run never met
it at a/60). With the bound met every beta state would sit at M = 0.1 and the landscape would not show, so the chains
were restarted (first attempt kept in `out/rs/old/`): shielded start = continuation 0.1 0.05 0.03 0.02 0.01 (stall-stop:
the deepest state reached), beta and release runs bounded at alpha = 0.02 (`BETA_ALPHA`), as in the report a bound
that is not met, so M is pushed down as far as the fixed effort allows.

Equal effort: the FZ beta runs stop on `--max-outer 12` (480 trials) in ~75 min; the hybrid's trials are slower
(n_u 21k), so at 90 min it had done 3-4 outers (beta <= 0.2) or 1-2 (beta >= 0.35, still far above their floor).
`p2_beta_ext.sh` extends every hybrid beta run from its end state (same pin, saved multiplier and penalty) by the
remaining outers up to 12 (budget 3.5 h); the first runs are kept as `b<beta>_first.npz`, and `collect.py` reports
both the 90-min and the extended values, with trial counts.

Live convergence: every run writes `<out>_live.tsv` (one line per trust-region trial) and `<out>_status.json` (window
statistics and warnings); `python scripts/meanforce/tr_live.py out/rs/` lists all runs, `tr_live.py <run>_live.tsv -f`
follows one.

## Follow-up (2026-11-13): the report's next steps, in the user's priority order

Job list `scripts/ripple_spline/jobs_followup.txt`, submitted as slots free by `scripts/ripple_spline/queue.sh` (nohup on
the login node; ids in `jobs_followup.ids`, log `logs/queue_followup.log`).

1. **Convergence and stationarity** (`conv.sh`, alpha 0.02, up to 60 outer iterations, 15 h hybrid / 9.5 h FZ each):
   unpinned from the shielded start (S) and from the beta 0.05 state (R05), pinned b21 = 0 in the island phase (P0)
   and pinned beta 0.05 (P05), for the hybrid (h1) and the FZ control (fzc). If S and R05 end at the same b21 and M,
   there is one minimum; P0 vs P05 at convergence gives the island cost at small beta free of effort differences.
   `converge_check.py` reads the live logs: dM/M over the last 20% of trials, |g|/|g0|, outer stop reasons, noise
   share, b21, and a verdict.
2. **Resolution** (`res.sh`: seeds, shielded start from the report's FZ start, pinned beta 0.05 / 0.2 / 1.0 with 12
   outer iterations each, common grid): h2 = FZ 40/40/6 + bands 0.01 (a/100); h3 = FZ 32/32/6 + bands 0.005 / 0.0075
   (a/120); fz40 = FZ 40/40/6 control (a/100).
3. **q = 3 and edge refinement**: h4 = FZ 32/32/6 + q = 2 band 0.01 + q = 3 band 0.84-0.96 at 0.0075 + edge band
   0.96-1.0 at 0.005.
4. **Localized bound** (`loc.sh`): band p16 of |F| over rho 0.5675-0.7075 (smooth max at q = 2) <= alpha x DESC's,
   alpha 0.3 -> 0.02 until it stalls, with the whole-volume mean <= 0.25 M_DESC in the same composite constraint;
   unpinned (S), pinned beta 0.05 (P05) and 0.5 (P5), hybrid and FZ control. Then `eval_extra.sh`: all convergence and
   localized states on the common grid (band p16, mean, b21, force split, shell max over 0.55-0.72).

Fit-split diagnostic (`--fit-split`, `fitsplit.sh`): at J_fit, dG = (J_fit - J_DESC) x B_DESC split into the part each
pressure space absorbs (grad dp) and the remainder R; FZ, pure splines s1 / s2, the hybrid, and the crosses (spline T
with FZ p, FZ T with spline p).
