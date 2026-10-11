# NEST optimization: status (2026-10-09)

Case: smooth-current precise_QA (free boundary) in its coils, box h = a/10, stellarator symmetry.

## Method

- **Unknowns x = (u, c_p).**
  - u: the BoundaryPotential coefficients, J = curl T. Fitted at 3× the equilibrium resolution, giving L, M, N = 48, 48, 35 and 90,408 unknowns.
  - c_p: the pressure Fourier–Zernike coefficients (43,500 unknowns).
  - The runs below optimize u only.
- **Fields at the collocation points (`nest/objective.py`, `Fields`).**
  - J, p, ∇p: tensor-product evaluation on a Gauss (ρ) × uniform (θ, ζ) grid, with cached basis factors and geometry.
  - B = B_ext + B_plasma:
    1. J is evaluated on a uniform flux grid.
    2. Cached 4th-order Lagrange interpolation takes it to the Cartesian box points.
    3. FFT and a free-space kernel (Hockney–Eastwood, or the Vico–Greengard truncated kernel) give B on the box.
    4. A type-2 NUFFT (`nufftax`) takes B back to the collocation points.
  - F = J × B − ∇p.
- **Stellarator symmetry.** The collocation grid, flux grid and box points cover half the domain; the other half follows from the field parities.
- **Objectives (DESC-like `ObjectiveFunction`).**
  - Terms: ForceBalance (F, normalized by ⟨|∇B²/2μ0|⟩), FieldError (B − B_DESC), CurrentError (J − J_DESC), PressureError (p − p_DESC).
  - Each term is weighted and scaled by √(dV/V).
- **Derivatives.**
  - One `jax.vjp` per iterate. Its pullback (a pytree) gives Jᵀw, and its `linear_transpose` gives J v.
  - The Jacobian is never formed.
  - All caches are pytrees passed to jit, not compiled in as constants.
- **Optimizer (`nest/optimize.py`).**
  - Levenberg–Marquardt: matrix-free PCG on (JᵀJ + λD)d = −Jᵀr.
  - D = diag(JᵀJ), estimated with Hutchinson probes; it also serves as the Jacobi preconditioner.
  - Exact line search: r is quadratic in x, so |r(x + a d)|² is a quartic in a.

## Tests (all pass)

| Test | Result |
|---|---|
| J v vs finite differences; adjoint ⟨w, Jv⟩ = ⟨Jᵀw, v⟩ | 1e-14; 2e-14 |
| Symmetric vs full setup at x0 | 3.5% |
| Truncated vs Hockney–Eastwood kernel | within 1.3% |
| Force residual at x0 vs DESC | 4.8e-4 (DESC's own: 4.24e-5); dominated by DESC's free-boundary field mismatch (3.9e-4) |
| Optimized x0 scored with J on the box interpolated vs exact vs a 1.5× finer flux grid | identical: the box interpolation is not a limit |

### Time and memory per call (one A100 40 GB)

| Box | f | J v | Jᵀw | GPU peak |
|---|---|---|---|---|
| a/10 | 0.03 s | 0.07 s | 0.07 s | 4.5 GB |
| a/20 | 0.13 s | 0.29 s | 0.27 s | 15 GB |

At a/20 the caches are about 4.9 GB, the largest being the J factors and geometry on the flux grid (2.2 GB).

## Optimization results, ⟨|F|⟩/norm (DESC: 4.24e-5)

| Run | Collocation grid | Independent finer check grid |
|---|---|---|
| Colloc 4× eq. resolution, 30 iterations, CG ≤ 200 | 4.62e-4 → 1.71e-4 | 4.63e-4 → 2.58e-4, worse for ρ > 0.8 |
| Same, + CurrentError (weight 0.01) | 4.62e-4 → 2.13e-4 | 4.63e-4 → 2.73e-4 |
| Same, 4 iterations, CG ≤ 2000 | 4.62e-4 → 1.56e-4 | 4.63e-4 → 2.29e-4 |
| Colloc 2× resolution of F (1.65M points, 55 residuals per unknown), 6 iterations, CG ≤ 300 | 4.62e-4 → 1.93e-4 (ρ > 0.95: 6.4e-4 → 4.9e-4) | timed out before the check |
| Same, 15 iterations, CG ≤ 400 (11.5 min per iteration) | 4.62e-4 → 1.85e-4, about 0.3% per iteration | timed out before the check |

## Problems fixed along the way

- **Host OOM and slow compiles:** caches were embedded as constants in the compiled programs. Fixed by passing them as pytrees.
- **GPU OOM in J v (a/20):** forward mode keeps both value and derivative box arrays alive. Fixed by computing J v as the transpose of the reverse pass.
- **GPU OOM in Jᵀw (5 to 9 GB):** the chunked NUFFT was a Python loop, so XLA kept every chunk's spreading buffers alive at once. Fixed with a sequential `lax.scan`, which cut the memory 20×.
  - The `custom_jvp` around the NUFFT had to go, because JAX 0.9.2 cannot reverse-differentiate a `custom_jvp` whose rule contains a scan.

## Where we are stuck

1. **Overfitting the collocation grid.** At 4× equilibrium resolution, F is barely sampled at Nyquist. Its resolution is about potential + equilibrium = (64, 64, 49). The optimizer zeroes F at the nodes while F grows between them, which the check grid shows near the edge. The 2× resolution-of-F grid should fix this, but it costs about 1.7 s per CG iteration, roughly 8.5 minutes per LM iteration.
2. **Slow convergence: an ill-conditioned linear solve.**
   - CG does not converge even in 2000 iterations, and the Jacobi preconditioner does not help.
   - Likely cause: J × B only constrains J⊥. The current along B is fixed only through ∇·J = 0, i.e. B·∇σ = −∇·J⊥, a magnetic differential equation with small (mι − n) factors. Those directions are weakly determined.
   - The CurrentError regularization did not help.
   - The force error stalls around 1.5–2e-4, 4–5× above DESC.

## Possible next steps

- **Physics-based preconditioner:** split J into J⊥ and J∥, or block preconditioning per toroidal mode.
- **Continuation in the basis resolution:** solve coarse first, then refine.
- **Krylov reuse:** keep the Krylov space across LM iterations (r is nearly linear), or switch to LSQR.
- **Separate the residual floor:** decide how much of the remaining error is the box resolution (B at a/10) and how much is the solver, by rerunning at a/20.

## Update (2026-10-10, overnight)

### Overfitting: fixed
- The collocation grid is now set relative to F's resolution, not the equilibrium's. At 2× that resolution, the 36-iteration run gives the same force error on the collocation grid and on the 3× check grid: 1.81e-4 on both.
- Per-iteration saves and a time budget (`--time-limit`) were added, so a time-out no longer loses the result.

### The 1.8e-4 plateau is a floor of the fixed-pressure problem, not the solver
`scripts/spectrum.py` forms JᵀJ exactly at a smaller resolution and takes exact Gauss–Newton steps.

| Setup (potential L, M, N = 16, 16, 14; 4,910 unknowns) | Exact least-squares floor ⟨\|F\|⟩/norm |
|---|---|
| Hockney–Eastwood kernel, a/10 | 1.83e-4 (check grid: same) |
| Truncated kernel, a/10 | 1.89e-4 |
| Pressure free as well (7,133 unknowns) | the linear model drops \|r\|² 38×, but the first step is cut to 0.175 by the nonlinearity; running |
| Hockney–Eastwood, a/20 | running |

- The kernel does not set the floor.
- The 90k-unknown runs stall at 1.81e-4, essentially the same floor as the 4.9k-unknown exact solve.
- At L, M, N = 24, 24, 20 (14k unknowns) the linear model floor is lower, but the nonlinearity limits the step (exact GN step cut to 0.53).

### Conditioning and the new block preconditioner
- With Jacobi scaling the condition number grows with resolution: 4e4 (200 unknowns), 5e7 (4.9k), 8e8 (14k).
- The unknowns sharing one (|m|, |n|) Fourier mode (all radial l of a and b, plus χ) are nearly collinear.
- `nest/precondition.py` (`BlockPreconditioner`) inverts those blocks, built from the local columns (J_i × B, −∇p_k, ...) without any Biot–Savart products. It takes 13 s at 4.9k unknowns.
  - Condition number 4.6e7 → 2.6e4, the same as with the exact blocks.
  - CG reaches 99.8% of the optimal decrease in 100 iterations.
- Probing the blocks with J v fails: neighbouring modes couple too strongly.
- Use it with `optimize_qa.py --precond block`.

### Next
- A run with both potential and pressure free, the block preconditioner, at 2× collocation (queued).
- Decide whether the remaining gap to DESC (4.2e-5) is the fixed pressure or the edge structure of the potential.

## Update (2026-10-10, speed and memory)

Production setup: a/10 box, 2× collocation (1.65M points, 5M residuals), 90,408 free unknowns, one A100.

| | Before | Now |
|---|---|---|
| Fits | 176 s | 36 s; 1.5 s from the cache (`nest/fit_cache.py`) |
| Collocation geometry | 451 s (DESC pointwise) | 14 s (tensor evaluation, checked against DESC on a sample) |
| DESC targets (B, J, p objectives) | ~1,170 s | ~25 s (DESC on a tensor `LinearGrid`) |
| One CG iteration (J v + Jᵀw) | ~1.7 s | 0.126 s |
| GPU peak (a/10, 2× collocation) | 17.7 GB | 6.0 GB |
| a/20 box | out of memory | 12.9 GB peak; J v 0.08 s, Jᵀw 0.27 s |
| Block preconditioner build | – | 213 s (574 s in the first version) |
| **Full optimization, to 1.0e-4** | 7.5 h (and stuck at 1.8e-4) | **~16 min** |

### Changes (each checked against the previous path)
- **Interpolation of B to the collocation points.**
  - The fine-grid FFT runs once per evaluation, not once per chunk.
  - Kernel weights are cached at setup.
  - The gather indices and weights are recomputed in the reverse pass instead of stored, about 6.8 GB saved.
  - Deconvolution uses pad and ifftshift, so no zero fine grid is kept.
  - Chunks of 262k points.
- **Explicit linearization.**
  - dF = J_lin(v) × B + J × B_lin(v) − ∇p(v): J v is one evaluation and Jᵀw its exact transpose.
  - The linearization keeps only J and B at the collocation points: 4 MB, against GBs of cache copies before.
- **Box and kernel.**
  - J on the box is built by one gather.
  - The Biot–Savart solve is checkpointed, so its kernel slices are not stored.
  - J is interpolated to the box in the box precision.
- **Fit.** One unoptimized `einsum`: 140 s → 0.02 s.
- **Optimizer.**
  - Block preconditioner by default.
  - Stops on stagnation (`--ftol`, relative decrease per iteration over 5).
  - Exact line search capped at 10.
  - Saves x every iteration; `--start` restarts from a saved x.
- **Checks.**
  - J v vs finite differences: 1e-13; adjoint identity: 1e-15.
  - Objective values identical to before.
  - Fast preconditioner blocks equal the generic ones to 1e-14 in float64.

### Result with the defaults
⟨|F|⟩/norm 4.62e-4 → **1.00e-4**, the same on the 3× check grid.
- Edge band ρ > 0.95: 6.4e-4 → 7.4e-5.
- DESC's own value: 4.24e-5.
- The Jacobi preconditioner had stalled at 1.8e-4.

## Update (2026-10-10): below DESC's force error

### Why the force error stalled around 1e-4
At the fixed-pressure optimum (1.09e-4), F splits along b = B/|B| into two parts:
- F_par = −b·∇p, which no current can change, is 8.5e-5: most of the total.
- F_perp, the part J balances, is 4.8e-5, already at DESC's level.

The pressure was frozen to DESC's profile, so its isosurfaces were DESC's flux surfaces. DESC's own field lies on them exactly (b_DESC·∇p ≈ 1e-17). The self-consistent coil-plus-plasma field differs from DESC's B by about 2e-3, and its field lines cross those surfaces, mostly in the core (`scripts/decompose_opt.py`).

A finer box (a/20) and the truncated kernel do not lower this floor; a larger potential basis (205k unknowns) lowers it only to 9.5e-5.

### Solution
Free the pressure together with the current, and hold its volume average (the stored energy) with a `PressureEnergy` objective. That objective is one residual with weight 10; without it, F also drops by deflating the pressure.

Run: `optimize_qa.py --free all --w-W 10`, started from the fixed-pressure optimum, a/10 box, truncated kernel, 100 iterations, 29 min.

| | ⟨\|F\|⟩/norm on its own a/10 grid | Independent check: a/20 box, truncated kernel, 3× grid |
|---|---|---|
| Start (fits) | 4.68e-4 | 4.67e-4 |
| Fixed pressure, optimized | 1.09e-4 | ~1.8e-4 |
| **Free pressure + energy held** | **1.69e-5** | **2.75e-5** |
| DESC (its own field) | 4.24e-5 | |

On the check, every radial band is below DESC, from 2.0e-5 to 3.2e-5.

What changed in the free-pressure solution:
- F_par: 8.5e-5 → 9.6e-6.
- F_perp: 4.8e-5 → 1.2e-5.
- Pressure: rms change 2.7% (max 3.6%), ⟨p⟩ conserved exactly, axis pressure × 0.9993.
- p is no longer constant on DESC's surfaces (b_DESC·∇p = 1.3e-4): it follows the field lines of the self-consistent field.

Running: the same optimization continued on the a/20 box, to bring down the error it shows on the a/20 check.

## Update (2026-10-11): continued on a/20, validation, tests

### Continued on the a/20 box (free pressure, stored energy held)
`--start results/freeP.npz --frac 20`, 100 iterations, 78 min.

| | ⟨\|F\|⟩/norm |
|---|---|
| On its own a/20 box | 8.21e-6 |
| On the 3× collocation check grid | 8.26e-6 |
| Evaluated on an a/25 box (box convergence) | 8.81e-6 |
| DESC | 4.24e-5 |

Every radial band is below 1.1e-5.

### Validation (`scripts/validate_opt.py`)
- **Current:** the net toroidal current through two cross-sections equals I to 4e-15, and J = 0 on the LCFS.
- **Exact J:** F with J evaluated exactly at the box points agrees with the objective's interpolated J to 0.2% (Hockney–Eastwood: 2.584e-5 vs 2.589e-5).
- **Force split:** F_par = 5.9e-6 and F_perp = 4.6e-6, against 1.3e-4 and 4.5e-4 at the start.
- **Field lines:** the Poincaré plot (`figures/freeP20_poincare.png`) shows nested surfaces lying on DESC's flux surfaces. The field of the start (fits) does not: its surfaces are shifted and the edge is partly chaotic.
- **Pressure:** a ±2% change, lower on the inboard side and higher on the outboard side.
- **Pressure sign:** p < 0 at 3,228 of 663k points, all at ρ ≥ 0.95, min −2.6e-4 of max p. Fixed for later runs by `PressurePositivity`, a one-sided penalty on min(p, 0) whose derivative uses the active set. The preconditioner leaves it out.

### Default workflow (`optimize_qa.py`)
- Defaults: pressure free, stored energy held (`--w-W 10`), p ≥ 0 (`--w-pos 10`), truncated kernel, block preconditioner, up to 100 iterations per box.
- `--refine 20` continues on finer boxes in the same run.
- Block preconditioner:
  - pressure blocks also use restricted tables;
  - Gram matrices are accumulated over sequential radial slabs, with an optimization barrier;
  - each block's table columns are gathered outside the Gram program, so XLA keeps no copies of the full tables.

### Test suite (`scripts/time_tests.py`, one A100)
All 12 test files pass, in 29 min. `test_reconstruction` needs the platform allocator for its a/30 box; the runner now sets it.

| Test | Time |
|---|---|
| test_smooth_edge | 322 s |
| test_boundary_potential | 284 s |
| test_reconstruction | 284 s |
| test_magnetic_field | 180 s |
| test_objective | 133 s |
| test_pressure | 115 s |
| test_cartesian_grid | 113 s |
| test_force | 89 s |
| test_fourier | 78 s |
| test_current_potential | 74 s |
| test_grid | 63 s |
| test_desc_grid | 55 s |

About 50 s of every test is fixed startup: importing DESC, loading the equilibrium, compiling. The slowest tests refit at full resolution, since the fit is what they test.

## Update (2026-10-11): other equilibria, basis, robustness

### Four equilibria in the same coils, all below DESC's own force error
Default workflow from the fits: pressure free, stored energy held, p ≥ 0, a/10 then `--refine 20`. Each result is checked independently on the a/20 box with a 3× collocation grid; the checks agree with the optimization grid to the digits shown.

| Equilibrium | DESC ⟨\|F\|⟩/norm | NEST (a/20 check) | Ratio |
|---|---|---|---|
| `smooth_J_edgeJ_r0.3` | 4.24e-5 | 8.26e-6 | 5.1× |
| `smooth_J` | 8.32e-5 | 1.42e-5 | 5.9× |
| `smooth_J_bdry_k5` | 3.73e-5 | 1.40e-5 | 2.7× |
| `L16N14` (original; J ≠ 0 at the edge in DESC) | 6.14e-5 | 1.26e-5 | 4.9× |

The first row was started from the fixed-pressure optimum. Its check box is the same a/20 box; on an a/25 box it gives 8.8e-6.

Each run takes about 1.5 h on one A100:
- about 4 min of preconditioner build per box;
- 100 iterations at 11 s on a/10;
- 100 iterations at 40 s on a/20.

### Basis
The factor-4 potential (205k unknowns, free pressure, a/10) reaches 1.78e-5 on its own box but 5.6e-5 on the a/20 check. The factor-3 basis at the same stage gives 1.69e-5 and 2.75e-5. A larger basis fits more of the box's discretization error, so it needs a finer box. Factor 3 stays the default.

### Robustness
- **Final check:** evaluated a few radii at a time, sharing the box side across chunks (`Fields(share=)`, `ForceResidual(rho_slice=)`). It is identical to the full-grid metrics to 2e-16, and x is saved before the check.
- **Preconditioner build:** a scan over equal radial slabs, padded with zero weights, holds one slab's buffers at a time. Each block's table columns are gathered outside the Gram program. The factor-4 build, 6,110 blocks, now fits in memory.
- **Allocator:** `run.sbatch` uses the platform allocator by default. It is about 7% slower per CG iteration (0.116 vs 0.108 s), but long runs cannot fail on a fragmented pool.
- **Compilation:** `run.sbatch` keeps a persistent JAX compilation cache (`.cache/jax`). The test suite takes 17.5 min with a warm cache, against 29 min before.
