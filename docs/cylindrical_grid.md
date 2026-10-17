# Cylindrical-annulus J → B solver (branch `cylindrical-grid`)

Status 2026-10-17. Stages A–C done locally; B5 (local memory push) done; stage D (GPU) awaits approval (stop point S3).
Plan and check-list: `docs/cylindrical_grid_plan.md`. Stage reports: `docs/cylindrical_grid_stageA.md`, `docs/cylindrical_grid_B4.md`.

## What it is

B_plasma = curl A with ΔA = −μ0 J in free space, solved on one field period of an annulus R_in ≤ r ≤ R_out, |z| ≤ z0 around the plasma (walls 2h outside it):
Fourier in φ (N_φ odd planes per period), Chebyshev in r and z, a Dirichlet solve per toroidal mode by matrix diagonalization, and James' correction u = u_D + S_m[σ] with the toroidal single-layer matrices S_m.
A_z takes the modes j·NFP and A₊ = e^{iφ}(A_r + iA_φ) the modes j·NFP + 1, so both are NFP-periodic scalars.
J reaches the nodes through main's flux-grid `PointInterpolator`; B goes to the collocation points by barycentric (r, z) and exact Fourier (φ) interpolation.

Use: `optimize_qa.py --grid cylindrical` (same `--frac`, `--margin`, check options), or `Fields(force, current, pressure)` with a `CylindricalGrid` box.
Options: `dtype` (float32 default, as main), `S_dtype` (float32), `cyl_half` (stellarator-symmetric half sampling, default on), `FreeSpaceAnnulus(zsym="auto")` (z-folded S_m).
`DenseFields` works unchanged. Main's paths are bitwise unchanged (C4).

## Gates (all pass)

| Gate | Result |
|---|---|
| A1–A5 | Chebyshev operators; κ(V_r) ≤ 4 for \|m\| ≤ 738; Q_{m−½} vs mpmath 5e-11 at m = 1000; S_m vs a 3-D integral 4.3e-15 |
| A5b | z-folded S_m vs dense 2e-15, storage exactly ½ |
| A6 | scalar free-space solve vs Gaussian charges 1.1e-12 (n = 96, N_φ = 251) |
| B1–B3 | JAX vs NumPy 1e-15; manufactured field 1.8e-12; div B 1e-15; adjoint 4e-15; `check_grads` |
| B4 | NEST's J on precise_QA: at equal h, 0.7–0.8× the box's interior error and ~0.6× its exterior error, with ~7× fewer points |
| C1, C1b | inside-node volume 1.8e-3, section current 3.1e-3 (a/6); symmetric locate identical in half the time |
| C2, C2b, C2c | `Fields` vs standalone 1e-14; half sampling 7e-15; float32 vs float64 B 1.6e-6 rms |
| C3 | Jv vs `jax.jvp` 5e-16; adjoint 2e-15; optimizer's `linear_transpose` JᵀW vs vjp 1.5e-16 |
| C4 | 17/17 arrays bitwise identical to 263bfad |
| C5 | 20 iterations at a/6 (factors 1): see below |
| C6 | dense G vs solve 4e-14; u0ᵀG cylindrical vs box 1.0e-2 at a/6 (box error dominates: 1.0e-2 vs 4.2e-3 against a cylindrical a/10 G) |

C5 (precise_QA smooth J, a/6, factors 1, 20 iterations, cg 100; check on the VG box at a/10 with 3× collocation):

| backend | ⟨\|F\|⟩/norm own grid | check | s/iteration (median of last 10) |
|---|---|---|---|
| cylindrical | 4.27e-4 → 1.28e-4 | 1.31e-4 | 39 |
| VG box | 4.28e-4 → 1.31e-4 | 1.41e-4 | 141 |

## B5: memory and time

Measures: the J → B_plasma pipeline of one optimizer product (J at the sampled nodes → B at main's production collocation, 1.65M points), forward and JᵀW (`jax.linear_transpose`).
Model: XLA's compiled memory analysis, args + output + temporaries (`tests/cyl_memory.py`); not included are the h-independent collocation-side caches shared by both methods (~4.5 GB at production on the GPU).
"Optimised" cylindrical = float32 arithmetic, stellarator-symmetric half sampling, z-folded S_m (S_m in parentheses, already inside the JᵀW figure).

| h | cylindrical, optimised: JᵀW + sampling (S_m) | cylindrical, float64 unfolded | main's box, float32 | box / cyl optimised |
|---|---|---|---|---|
| a/10 | 0.23 + 0.04 = **0.27 GB** (0.05) | 0.55 GB | 1.96 GB | 7.2× |
| a/15 | 0.58 + 0.14 = **0.72 GB** (0.15) | 1.55 GB | 4.41 GB | 6.1× |
| a/20 | 1.23 + 0.32 = **1.55 GB** (0.33) | 3.40 GB | 8.62 GB | 5.6× |
| a/30 | 3.85 + 1.03 = **4.88 GB** (1.08) | 10.82 GB | 29.92 GB | 6.1× |
| a/40 | 8.86 + 2.39 = **11.25 GB** (2.51) | 25.05 GB | 75.37 GB | 6.7× |
| a/60 | 29.09 + 7.90 = **36.99 GB** (8.29) | 82.50 GB | int32 overflow | — |
| a/80 | 68.32 + 18.55 = **86.87 GB** (19.45) | — | int32 overflow | — |

Measured (`tests/cyl_measure.py`, laptop CPU, process peak RSS including ~0.3 GB JAX runtime, synthetic J of production size):

| run | grid | nodes | J rows | build peak | fwd peak | JᵀW peak | fwd | JᵀW |
|---|---|---|---|---|---|---|---|---|
| cylindrical a/10 (float32, sym, z-fold) | 251x71x82 | 1.5M | 58,452 | 0.70 GB | 0.82 GB | 0.88 GB | 0.51 s | 0.44 s |
| cylindrical a/15 (float32, sym, z-fold) | 373x103x119 | 4.6M | 182,874 | 0.84 GB | 1.03 GB | 1.24 GB | 1.05 s | 1.01 s |
| cylindrical a/15 (float64, full, unfolded) | 373x103x119 | 4.6M | 365,748 | 0.72 GB | 1.28 GB | 1.46 GB | 1.88 s | 1.92 s |
| cylindrical a/20 (float32, sym, z-fold) | 495x134x156 | 10.3M | 413,899 | 1.38 GB | 1.59 GB | 1.78 GB | 1.57 s | 1.62 s |
| cylindrical a/25 (float32, sym, z-fold) | 615x166x193 | 19.7M | 788,134 | 1.38 GB | 2.07 GB | 2.19 GB | 2.91 s | 3.49 s |
| box a/10 (float32) | 315x315x99 | 9.8M | 61,395 | 1.53 GB | 2.19 GB | 2.19 GB | 3.89 s | 5.86 s |

The model matches the box (2.19 GB measured vs 1.91 + 0.3) and the cylindrical a/20 (1.78 vs 1.23 + 0.3 + build transient); at a/10–a/15 the cylindrical peak is set by the build transients, not the pipeline.
Not done locally: the per-stage breakdown (gather, FFT, mode solves, S matvec, curl, interpolation) that the plan asks for; it goes into D1 on the GPU, where `cyl_measure.py` records the device peak.

What changed in B5: float32 path (C2c), symmetric half sampling (C2b), z-folded S_m (A5b), symmetric locate (C1b), and the S_m build batched over targets (3–4× faster, bitwise identical).
Banded S_m was measured and dropped: in the nodal basis about a third of the entries of S_m stay above 1e-10 even at high \|m\| (each face's global Lagrange interpolation spreads the local kernel), so banding saves at most ~3× with ragged storage.

![error vs memory](cyl_verdict.png)

(a) Exterior B error (rms over 177 converged points 0.15–0.41 a outside the plasma, `results/cyl/b4/exterior_fine.json`) against the pipeline memory of the JᵀW product plus J sampling (model). Filled: measured errors; open/dashed: O(h^p) extrapolation with the fitted orders (cylindrical p = 2.16 from a/10–a/20, box p = 1.71 from a/10–a/15). Dotted: A100 budgets less ~4.5 GB shared. (b) Measured laptop CPU time per JᵀW.

## Verdict

Yes, it is a big win over the box, in memory, accuracy and time together:
1. At equal h the cylindrical pipeline needs 5.6–7.2× less memory than main's float32 box (a/10–a/40), is more accurate (B4: 0.6× the exterior error), and is 7.6× (f) / 13× (JᵀW) faster on the laptop CPU.
2. At equal memory it reaches ~1.9× finer h. Extrapolated to one A100 (budget less 4.5 GB shared): 40 GB → box a/32 (1.3e-04) vs cylindrical a/59 (1.3e-05); 80 GB → box a/40 (8.7e-05; the NUFFT's int32 offsets overflow near a/42) vs cylindrical a/76 (7.2e-06). About 10–12× lower B error per GPU.
3. On the laptop: cylindrical a/25 (20M nodes) runs JᵀW in 3.5 s at a 2.2 GB process peak; box a/10 already peaks at 2.2 GB (box a/15, model 4.4 GB, was not run: it would swap).

Caveats: the GPU numbers are model extrapolations (memory) and O(h^p) extrapolations (error) from three (cylindrical) or two (box) measured sizes; GPU times are unmeasured — the box's FFT path is GPU-friendly, the cylindrical solve is a sequential scan over mode chunks of small batched matmuls. The errors are B_plasma alone, from exact J at the nodes for both methods (B4); in `Fields`, main's Lagrange flux-grid J adds a bias shared by both (1.1e-4 rms at a/10) that is not in these curves. S_m precompute grows ~h^-2.7 on the host CPU (a/25: 195 s measured before batching; batching measured 3× faster at a/15, so a/60 extrapolates to ~12 min).

## Proposed GPU jobs (S3: needs approval)

1. D1 memory push (A100-80GB, cap 2 h): `cyl_measure.py` ladders, cylindrical a/20, 30, 40, 50, 60, 70 and box a/20, 30, 40, with device peaks and times for f and JᵀW; plus cylindrical B at a/30 and a/40 against the exterior reference (`tests/cyl_b4.py`, reference files shipped) to test the O(h^2.2) extrapolation. Estimate ~1.5 h, dominated by S_m builds (~45 min host CPU).
2. D2 production (A100-40GB, cap 2 h each, two jobs): precise_QA `optimize_qa --grid cylindrical` at factors 3 3 2.5, colloc 2, maxiter 100, at a/20 and a/30; check on the VG a/20 box with 3× collocation and on cylindrical a/40. Compare with main's production anchor (a/10 box, 16 min on one A100).
3. D3 DSHAPE axisymmetric: needs local preparation first (N_φ = 1 path, DSHAPE files from branch `optimization`); not proposed yet.

## Notes for main

- `convergence_b_plasma.py`'s exterior reference compares the continuous Biot–Savart at a basis resolution with one at 2/3 of it; for precise_QA's (48, 48, 35) the 2/3 level under-resolves N = 35 and gives a 1.1e-2 self-difference. Use (64, 64, 56) vs (48, 48, 42).
- The J-sampling cache (`PointInterpolator`: 64 indices and weights per point) is a large part of both methods' memory at fine h (cylindrical a/40: 2.4 GB of 11 GB); computing the weights on the fly would remove it for both.

## Files

`nest/`: chebyshev.py, annulus_poisson.py, toroidal_kernel.py, single_layer.py, annulus_freespace.py (FreeSpaceAnnulus, ZFold), cylindrical_bs.py (CylindricalSolver), cylindrical_grid.py (CylindricalGrid, CylindricalInterpolator); objective.py, force.py, dense.py (cylindrical branches).
`tests/`: test_cylindrical_bs.py (A1–B3), test_cylindrical_integration.py (C1–C6), cyl_a6.py, cyl_manufactured.py, cyl_b4.py, cyl_g0_probe.py, cyl_memory.py, cyl_measure.py.
Results (gitignored): `results/cyl/` (a6.json, b4/, c5/, c6/, memory.json, memory_opt.json, measure/).
