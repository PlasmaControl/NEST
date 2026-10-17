# NEST

Fixed-boundary MHD solver with non-nested flux surfaces, starting from a DESC free-boundary equilibrium.

# To do

- [x] Fit DESC current density J as a Fourier-Zernike series (`nest/current_fz.py`)
- [x] Evaluate J on the flux grid G1, a `ConcentricGrid` (`nest/desc_grid.py`)
- [x] Map J to a uniform Cartesian grid G2 with a 2-D Newton inverse map (`nest/cartesian_grid.py`)
- [x] Extend to the periodic computational box (step 3), with free-space handling of B (`nest/magnetic_field.py`)
- [x] FFT of J and B_plasma = i mu0 k x J / |k|^2 (steps 4-5) (`nest/fourier.py`, `nest/magnetic_field.py`)
- [x] Pressure as a Fourier-Zernike field, optimized with T (`nest/pressure.py`)
- [x] Math representation of J as potential T, with smoothness at the edge (`nest/boundary_potential.py`)
- [x] Grid class and DESC common input class (`nest/grid.py`, `nest/desc_input.py`)
- [x] B = B_ext + B_plasma and force residual J x B - grad p inside the plasma (`nest/force.py`)
- [x] Jacobian construction: matrix-free J v and J^T w, explicit columns (`nest/objective.py`, `nest/jacobian.py`)
- [x] Optimization loop over the J coefficients: Levenberg-Marquardt with a block preconditioner (`nest/optimize.py`, `nest/precondition.py`, `scripts/optimize_qa.py`)

- [ ] For the dshape tokamak, we need make a correction to the initial B_plasma computed from FFT (intrinsic error due to the grid resolution h). So using a biot-savart calculation for B_plasma from J_DESC. The only error will in the computation of delta_B during the optimization. Eventually the cylindric grid can solve this issue and we don't need to do that.
- [ ] For the dshape tokamak, we need to introduce a specific perturbation and scale it to see how the island grows (look at Matt's paper on islands/other papers for this scaling law).
- [ ] Add the free-boundary to these two tokamaks tests. See how things change.
- [ ] For NCSX, do two cases: one with and one without the correction on the initial B_plasma, to see a) how the size of the islands are at the initialization b) if the end result is different. A validation case would be that the final islands are physical if with two very different initializations, the result is the same.
- [ ] Add the free-boundary NCSX. See how the islands change.
- [ ] Ultimately for all these cases, we should make sure there is a convergence with the resolution, that is the fields change are getting more and more smaller, reaching zero with the resolution.
- [ ] Add simple and more complex tests for the FFTs and iFFTs to re-construct precisely J and B.
- [ ] Look at other solvers like QR/SVD on multiple GPUs (https://github.com/flatironinstitute/jaxmg)


# Questions?
- [ ] Should we raise s in (1-rho^2)^s to 3 or 4?
- [ ] How do we do the reconstruction of the potential current T (not simply a FZ series anymore) with .fit from DESC?
- [ ] Does the way we find the current density on cartesian grid has to remain this one: each Cartesian point is mapped back to flux coordinates by Newton and J is evaluated there exactly ; the nearest flux-grid node is only the Newton starting guess?
- [ ] Should we smooth out the B Fourier modes with a kernel G(k) (remove the periodicity)?
