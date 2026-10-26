# NEST without free pressure: p from the weak Poisson equation of force balance

Branch `pressure-poisson` = `origin/main` e762f37 fast-forwarded to `origin/lm-trust` 30fc957
(2026-10-26; the cylindrical grid is already in main). Worktree `~/PycharmProjects/NEST/NEST-pois`.

Conventions (as the earlier campaigns):
- Commits are local only; the user pushes. Author via `git -c user.name=... -c user.email=...`.
- Long runs stream per-iteration progress (no silent runs); pipes use `grep --line-buffered` only.
- No Modal / GPU / Stellar job without the user's approval. Production runs are handed off as sbatch lines.
- Scaling or conditioning questions: look at DESC first (`desc/optimize/tr_subproblems.py`,
  `desc/optimize/utils.py`), then ask.
- Main's default paths stay bit-identical: the new mode is opt-in (`--speed elim`, `HarmonicModel(..., speed={"elim": 1})`).

Status legend: `[ ]` open, `[x]` done, `[~]` done with a caveat recorded in the gate's notes.

---

## 0. The idea, and why it is variable projection

Force balance is affine in p: F = G(u) - grad p, G = J x B. With the force rows' own weights
w (= swF^2, i.e. dV/V / norm^2), the weak form of the divergence of force balance on the flux
domain rho <= 1 with p(1) = 0,

    integral w grad q . grad p  =  integral w grad q . G(u)      for all q in V_h,

is the normal equation of min_p || sqrt(w) (G(u) - grad p) ||. So "p solved from the Poisson
equation" is exactly the elimination of p by least squares (Golub-Pereyra variable projection),
with V_h the pressure space. The Galerkin weighted Laplacian is A_p^T A_p (A_p = -sqrt(w) grad
of the pressure columns, plus the constant <p> row when w_W > 0); it is constant, factored once.

Consequences used below:
- Reduced residual r*(u) = (I - P) r(u, 0), P = A_p (A_p^T A_p)^-1 A_p^T **constant** (A_p does
  not depend on u when par = 1 and w_pos = 0). Its exact Jacobian is (I - P) A_u: no
  derivative-of-projector term.
- Reduced Gram = Schur complement S = H_uu - H_up H_pp^-1 H_pu. Its eigenvalues lie inside the
  joint Gram's: kappa(S) <= kappa(H) (unscaled).
- Same minimiser as the joint problem for the same V_h. A richer V_h (a "resolved Poisson p")
  changes the physics only by allowing a better p.
- With QR: order the columns [A_p | A_u | r]; the trailing block of R is the QR of
  (I - P)[A_u | r]. R_pp is constant.

### Measured before planning (laptop lm_ab model: L16 a/20 N = 1, 516 u + 234 p; eps 1000 stage)

Scripts and numbers: workspace `diag/cond_elim2.py`, `cond_elim2.json` (artifacts).

| case | kappa joint (Jacobi) | kappa reduced (Jacobi) | kappa joint (phys) | kappa reduced (phys) | kappa u-only (phys) |
|---|---|---|---|---|---|
| w_J 1e-3 | 1.92e5 | 1.00e4 | 7.21e3 | 2.75e2 | 9.70e2 |
| w_J 1e-7 | 3.49e5 | 2.71e4 | 2.41e5 | 1.42e4 | 5.04e3 |
| w_J 1e-7, p factors 1.5 (n_p 494) | 6.64e5 | 4.58e4 | 8.84e5 | 2.56e4 | 5.04e3 |
| w_J 1e-7, p factors 2 (n_p 850) | 6.87e5 | 4.67e4 | 1.61e6 | 2.84e4 | 5.04e3 |

Reading: elimination removes the u-p collinearity (10-20x); with the physical metric on u the
reduced problem reaches 275 at w_J 1e-3. At w_J 1e-7 the floor is set by u directions whose
force change is a pure gradient (equilibrium families held only by w_J): elimination cannot
remove them. Production (L64, a/120) is unmeasured.

## 1. Design decisions (user, 2026-10-26)

| # | Decision |
|---|---|
| D1 | Host: `HarmonicModel` dense exact path (streamed TSQR + `speed tr`), new speed option `elim`. Matrix-free is deferred (P4). |
| D2 | p lives on the flux domain rho <= 1, p(1) = 0 (the window x Fourier-Zernike space, optionally richer via `--p-factors`). Not the annulus. |
| D3 | Weak form with the force rows' weights: orthogonal projection = least-squares elimination. Not the strong-form Laplacian. |
| D4 | v1 requires A_p constant: par = 1, w_pos = 0, no transport. w_W may be > 0 (its row is constant in p and joins A_p). Asserted, not silently ignored. Production island stages already use `--w-W 0 --w-pos 0`. |
| D5 | Scaling of the reduced problem: Jacobi on R_u, or `phys` = the u block of the physical metric (block diagonal, so L_uu = L[u, u]). |

## 2. Algorithm (speed `elim`)

Invariant at every accepted iterate: p = p*(u), i.e. c_p = Q_p^T r = 0.
1. Start: one projection p <- p*(u0) (a gradient pass; R_pp from the first TSQR, or `_pproj`'s factor).
2. Streamed TSQR of the chunk rows with columns permuted to [p | u | r] (and the global rows last).
   R_pp is checked against the first iteration's (constant; relative change < 1e-12 or rebuild + warn).
3. Trailing block T = [R_u c_u; 0 rho]. f_here = (|c_u|^2 + rho^2) / 2. GN share = |c_u|^2 / 2f.
4. Trust-region step on u only: the existing `_tr_step(RD = R_u D, c = c_u, ...)` (DESC exact-QR rule).
5. Trial at u + d_u: one pass gives |r(u + d_u, p)|^2 and g_p = A_p^T r; then
   f* = (|r|^2 - |R_pp^-T g_p|^2) / 2 and p' = p - R_pp^-1 R_pp^-T g_p (exact: r affine in p).
   Gain ratio on the reduced model: pred = f_here - (|R_u y + c_u|^2 + rho^2) / 2.
6. Accept z <- (u', p'); the invariant holds exactly by construction.

Cost per iteration vs today: TSQR identical (same width n_u + n_p); trust-region Cholesky on n_u
instead of n_u + n_p (production estimate ~11k vs ~16k: ~3x cheaper per mu); the trial pass adds an
n_p-vector accumulation. No extra field solves; p tables are analytic (no box solves).

If a richer V_h makes the width n_u + n_p too large (P2), the two-pass Schur variant replaces step 2:
pass 1 C = A_p^T [A_u | r] (n_p x (n_u + 1)), Y = (R_pp^T R_pp)^-1 C; pass 2 the TSQR of the
projected rows [A_u | r] - A_p Y (width n_u + 1). No normal-equation squaring.

## 3. Stages and gates

### P0. Branch baseline
- [~] P0.1 (f rel 1e-10 at its 1-3, 1.4e-9 at the start evaluation: reduction-order noise) `scripts/lm/lm_ab.py --variants tr --iters 3` on the merged branch reproduces the first
  3 iterations of `results/lm/abfix_L16a20_full_eps1000_wJ1e-07.json` (f to 1e-10 relative).
- [x] P0.2 (48c75a2) `scripts/diagnostics/pressure_elim_conditioning.py` (from `cond_elim2.py`) committed; reproduces the table above.

### P1. Elimination in HarmonicModel (laptop, 750 unknowns)
Implementation: `speed elim` in `_qr_step` / `_tsqr` / `_trials` / `lm`; asserts of D4; per-iteration
log adds `|c_p|/|c|`, kappa estimate of R_u D (cheap: diag ratio), f*.
Gates (tests/test_pressure_elim.py, `PYTHONPATH=nest:$DESC JAX_PLATFORMS=cpu`):
- [x] T1 (1.3e-14) T^T T equals H_uu - H_up H_pp^-1 H_pu (relative 1e-10).
- [x] T2 (<= 7e-12) After the start projection, |c_p| / |c| < 1e-10; again after every accepted step.
- [x] T3 (1.7e-15; grad_p f at p* 1.9e-14) f* from the trial pass equals `evaluate(u', p*(u'))` with an explicit p solve (1e-12 relative).
- [x] T4 (2.9e-11) The undamped reduced GN step equals the u part of the undamped joint GN step (1e-8 relative).
- [x] T5 (3.2e-10, Richardson) Finite differences of f*(u) along random d match the reduced gradient R_u^T c_u (exact VarPro).
- [~] T6 A converged joint solution (lm_ab base, wJ 1e-3) is a fixed point: reduced GN step |d_u|/|u| = 1.8e-7 (not < 1e-8) with reduced GN share 4.9e-14, i.e. a step along flat directions that removes nothing; test bound 1e-6.
- [x] T7 (5.2e-11; f 1.1e-2 -> 1.500369e-10 restored) Manufactured Poisson: G = grad q, q in V_h -> p* = q, r* = 0 (1e-12).
Experiment E1 (A/B, lm_ab harness, 20 its, w_J 1e-7 eps 1000, and w_J 1e-3): variants `tr`, `elim,tr`,
`phys,tr`, `elim,phys,tr`. Report f/f0, <|F|>/norm, GN share, time per iteration, accepted/rejected tries.
E1 result (30 its; `results/lm/e1_*.json`):

| case | variant | its | f end | f / f0(joint) end | GN share end | time |
|---|---|---|---|---|---|---|
| A w_J 1e-7, from stage | tr | 30 | 1.058e-10 | 0.705 | 0.24 | 97 s |
| | elim,tr | 30 | 7.338e-11 | 0.489 | 0.056 | 87 s |
| | phys,tr | 30 | 7.906e-11 | 0.527 | 0.069 | 105 s |
| | elim,phys,tr | 30 | 6.358e-11 | 0.424 | 0.011 | 107 s |
| B w_J 1e-3, from base | tr | 30 (not converged) | 3.0970981936e-10 | 3.17e-3 | 2.6e-9 | 200 s |
| | elim,tr | 20 | 3.0970981870e-10 | 3.17e-3 | 1.0e-12 | 71 s |
| | phys,tr | 21 | 3.0970981870e-10 | 3.17e-3 | 8.4e-13 | 65 s |
| | elim,phys,tr | 11 | 3.0970981870e-10 | 3.17e-3 | 1.1e-14 | 48 s |

- [x] E1 done.
**STOP S1**: results to the user.

### P2. Pressure space resolution (the "resolved Poisson" p)
- [x] `--p-factors` decoupled from the current factors (`isl_scan.py`, `lm_ab.py`; p tables only, cached by key).
- [x] T8 Manufactured q not in V_h (q = (1-rho^2)^3 (1 + 0.2 x cos zeta + 0.2 y sin zeta)/(1.15 - x)): |grad(p* - q)|_w/|grad q|_w = 2.3e-3, 2.4e-4, 2.8e-5, 3.9e-7, 5.6e-9 at L 8, 12, 16, 24, 32 (0.11-0.12 per +4 in L, the pole's rate); kappa(A_p) 1.8e2 -> 4.2e4, full rank at L 32 on 33 radii.
- [x] E2 Ladder p factors 1, 1.5, 2, 3 at fixed u basis (factors 1; laptop model; `results/lm/e2_*.json`, `e2p_*`):

| case | p factors (n_p) | variant | its | f end | <\|F\|>/norm | s/it |
|---|---|---|---|---|---|---|
| w_J 1e-3 from base, converged | 1 (234) | elim,phys,tr / phys,tr | 11 / 21 | 3.0971e-10 | 1.485e-5 | 4.4 / 3.1 |
| | 1.5 (494) | elim,phys,tr / phys,tr | 16 / 8 | 2.8957e-10 | 1.386e-5 | 5.8 / 6.4 |
| | 2 (850) | elim,phys,tr | 10 | 2.8914e-10 | 1.386e-5 | 16.2 |
| | 3 (1850) | elim,phys,tr | 11 | 2.8821e-10 | 1.383e-5 | 28.8 |
| w_J 1e-7 from stage, 30 its | 1 | elim,phys,tr / phys,tr | 30 | 6.36e-11 / 7.91e-11 | 9.94e-6 / 1.11e-5 | 3.6 / 3.5 |
| | 1.5 | elim,phys,tr / phys,tr | 30 | 5.67e-11 / 6.69e-11 | 9.30e-6 | 6.4 / 8.1 |
| | 2 | elim,phys,tr / phys,tr | 30 | 5.66e-11 / 7.78e-11 | 9.29e-6 / 1.10e-5 | 9.9 / 9.7 |
| | 3 | elim,phys,tr | 30 | 5.67e-11 | 9.30e-6 | 28.5 |

  Reading: the pressure space saturates at ~1.5x the current factors (the default costs ~7% in f and
  <|F|>; beyond 1.5 the change is <= 0.5%: the u basis / collocation limits). Reduced kappa saturates
  (phys 2.8e4, Jacobi 4.7e4 at p 2-3) while the joint phys kappa grows (8.8e5, 1.6e6, 5.0e6 at p 1.5, 2, 3).
  At w_J 1e-7 elim reaches a lower f in 30 its at every p (gain 1.2-1.4x in f). At w_J 1e-3, p 1.5 the
  joint phys,tr converged in fewer iterations (8 vs 16): elim is not uniformly faster to the end.
  p >= 0: the converged states have min p / max p = -1.4e-4 (w_J 1e-3) and -3e-4 (w_J 1e-7), the same in
  the joint and elim solutions (same minimiser): a property of the solution, not of elim.
  `|c_p|/|r|` grows with kappa(A_p): <= 1e-10 at p 1-2, 1e-8 at p 3.
  Two-pass Schur variant: not implemented. Its flop saving over the [p|u|r] TSQR is 2 m n_p^2 of
  2 m (n_u + n_p)^2, i.e. n_p^2/(n_u + n_p)^2 = 24% at p 1.5 here and ~10% at production (n_p ~5k of
  16k, estimate), against forming the rows twice.
**STOP S2**: recommendation `--p-factors` = 1.5 x the current factors with `--speed elim,phys,tr`.

### P3. Production measurement
- [ ] Extend `scripts/diagnostics/base_conditioning.py` with the reduced kappa (Jacobi and phys) at the
  L32 / L64 a/120 production bases (16k unknowns).
- [ ] sbatch lines for the collaborator: A/B `tr` vs `elim,tr` vs `elim,phys,tr` from a120Cn2d1000,
  w_J 1e-7, 30 its, streaming logs (as `docs/lm_solver_handoff.md`).
**STOP S3** before any GPU or cluster job.

### P4. Deferred (not in scope until S3)
Matrix-free reduced trust region (LSQR / Steihaug on (I - P) A_u, preconditioned by the reduced phys
factor) for the table-free path; the annulus/strong-form Poisson only there.

## 4. Risks
- The w_J floor: at small w_J the reduced kappa is set by equilibrium-family directions; elimination
  does not fix the non-uniqueness (measured: 1.4e4 phys at w_J 1e-7).
- p >= 0 is not enforced in v1 (w_pos = 0); a negative p* is reported per iteration.
- If production kappa is dominated by within-u radial collinearity (2026-10-22 diagnosis), elimination
  gives ~10x and the phys metric is what matters.
