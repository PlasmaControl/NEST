# To do

Open work on the solve path, grouped by kind. Each item says what, where, and how
it will be checked. When an item is done, tick it in the commit that does it and
record the numbers in that commit's UPDATES.md entry. Remove ticked items once
they are no longer useful as a reference.

## Validation (not run yet)

- [ ] **Production-resolution solve on a GPU.** `nest/run_solve.py` has only run at
  bases 4, 4, 3 and h = a/4. At the equilibrium resolution of the smooth-current
  precise_QA (16, 16, 14), the unknowns number about 9,800 (about 4,900 with
  `sym=True`), and D_B is of order 15 GB in float64. Run it full and symmetric
  with the memory lines on. Check: peak memory per D_B batch and per GN step; the
  forces on the solve grid and on the independent evaluation grid; that the end
  states converge with h. This run decides whether the memory items below are
  needed.
- [ ] **`tests/test_reconstruction.py::test_qa`** (a/10, a/20, a/30) is too heavy
  for an 8 GB laptop; it did not finish in 1.5 h. Run it on a cluster node and
  record the numbers.
- [ ] **Box sampling vs fftq.** NEST samples J at the box nodes, spacing h. fftq's
  node path samples at h/2 (q = 2 sub-nodes per cell). Compare B_plasma and the
  end states at NEST h = a/(2F) against fftq at h = a/F, to separate source
  quadrature error from the band limit.

## Studies

- [ ] **Edge power s = 3 or 4** (the README question). With the general edge factor
  h_q, compare the fit misfit to DESC's J against the convergence of B_plasma in
  h, for s = 2, 3 and 4.

## Optional features

- [ ] **Carried net toroidal current.** `BoundaryPotential` holds I at DESC's value.
  fftq's `net_current="carried"` makes I one extra unknown: one column of
  (I / 2 pi) h_0 in T_theta. Where: `boundary_potential.py` (structure, `split`,
  closed-form columns), `fields.py`. Check: with I fixed at its fitted value the
  carried solve reproduces the fixed one; the Jacobian column against a central
  difference.
- [ ] **Report inverse-map failures.** `grid.map_to_flux` silently counts inside
  points where Newton did not converge as outside. Return the count (and the
  points) in an info dict, print it in `locate_plasma`, and check it against a
  fine LCFS table. With symmetry, a missing point's mirror can fill it, as fftq
  does.
- [ ] **Disk cache of `locate_plasma`** (inside indices and flux coordinates), keyed
  by the equilibrium, h and the box factor, like the D_B cache. One Newton solve
  per box point inside the plasma bounding box, repeated for every run at the same h.
- [ ] **Field-line tracing and Poincaré sections** of B = B_ext + B_plasma at end
  states. The purpose is to see whether islands open at iota = 1/2; DESC's field
  is traced as the control. A new `diagnostics.py`.
- [ ] **Uniqueness / Hessian check at end states.** The residual is quadratic in x,
  so the exact Hessian is J^T J plus a constant second-order term, cheap to form. Report
  the smallest eigenvalues and their eigenvectors (fftq found soft directions of
  about 1e-11 lambda_max with the force term alone).

## Memory (if the production run needs it)

- [ ] **Hockney kernel storage.** `biot_savart_kernel_hat` is complex,
  (Nx, Ny, Nz // 2 + 1, 3): 12 times the Vico-Greengard octant. It is imaginary
  and odd per component, so a real octant with sign folds would store 1/8 of it.
  Only needed if Hockney is used at production resolution.
- [ ] **Type-2 NUFFT from the half spectrum by default.** `interpolate_hat` rebuilds
  the full spectrum (`half=False`), because the half-spectrum NUFFT needs
  eps = 1e-14 at upsampfac 1.25. Either accept that cost, or centre the k_z
  range differently so that the k_z = 0 modes are not at the band edge.
- [ ] **Two real fields in one complex NUFFT** (fftq's packing): B_x + i B_y in one
  type-2, so 2 type-2 transforms instead of 3.
- [ ] **Coset splitting of the padded box** (fftq's packed layout, x and y). Exact.
  Spectra and the NUFFT's fine grid shrink by the number of cosets, and spreading
  work grows by the same factor.

## Known limitations (no action planned)

- The precise_QA helical coils are stellarator symmetric only to 1.4e-4 of max |B|
  at the collocation points. With `sym=True` and the coils, the solve is of the
  symmetrized problem; use `DESCExternalField` for exact symmetric comparisons.
