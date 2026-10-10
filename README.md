# NEST

Fixed-boundary MHD solver with non-nested flux surfaces, starting from a DESC
free-boundary equilibrium.

# To do

- [x] Fit DESC current density J as a Fourier-Zernike series (`nest/current_fz.py`)
- [x] Evaluate J on the flux grid G1, a `ConcentricGrid` (`nest/desc_grid.py`)
- [x] Map J to a uniform Cartesian grid G2 with a 2-D Newton inverse map (`nest/cartesian_grid.py`)
- [x] Tests vs DESC: fit on G1, current through phi = 0, int |J| dV (`tests/`)
- [x] Use the free-boundary equilibrium from `desc_input` instead of the precise_QA example
- [x] Extend to the periodic computational box (step 3), with free-space handling of B (`nest/magnetic_field.py`)
- [x] FFT of J and B_plasma = i mu0 k x J / |k|^2 (steps 4-5) (`nest/fourier.py`, `nest/magnetic_field.py`)
- [x] Pressure as a Fourier-Zernike field, optimized with T (`nest/pressure.py`)
- [x] Math representation of J as potential T, with smoothness at the edge (`nest/boundary_potential.py`)
- [x] Grid class and DESC common input class (`nest/grid.py`, `nest/desc_input.py`)
- [x] B = B_ext + B_plasma and force residual J x B - grad p inside the plasma (`nest/force.py`)
- [x] Jacobian construction: matrix-free J v and J^T w, explicit columns (`nest/objective.py`, `nest/jacobian.py`)
- [x] Optimization loop over the J coefficients: Levenberg-Marquardt with a block preconditioner (`nest/optimize.py`, `nest/precondition.py`, `scripts/optimize_qa.py`)
- [x] Force balance below DESC's own: 2.75e-5 (checked on a finer box) against 4.24e-5, with the pressure free and the stored energy held (`OPTIMIZATION_STATUS.md`)

# Questions?
- [ ] Should we raise s in (1-rho^2)^s to 3 or 4?
- [ ] How do we do the reconstruction of the potential current T (not simply a FZ series anymore) with .fit from DESC?
- [ ] Does the way we find the current density on cartesian grid has to remain this one: each Cartesian point is mapped back to flux coordinates by Newton and J is evaluated there exactly ; the nearest flux-grid node is only the Newton starting guess?
- [ ] Should we smooth out the B Fourier modes with a kernel G(k) (remove the periodicity)?
