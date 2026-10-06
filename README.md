# NEST

Fixed-boundary MHD solver with non-nested flux surfaces, starting from a DESC
free-boundary equilibrium.

# To do

- [x] Fit DESC current density J as a Fourier-Zernike series (`nest/current_fz.py`)
- [x] Evaluate J on the flux grid G1, a `ConcentricGrid` (`nest/desc_grid.py`)
- [x] Map J to a uniform Cartesian grid G2 with a 2-D Newton inverse map (`nest/cartesian_grid.py`)
- [x] Tests vs DESC: fit on G1, current through phi = 0, int |J| dV (`tests/`)
- [x] Use the free-boundary equilibrium from `desc_input` instead of the precise_QA example
- [ ] Extend to the periodic computational box (step 3), with free-space handling of B
- [ ] FFT of J and B_plasma = i mu0 k x J / |k|^2 (steps 4-5)
- [ ] find a way to compute the pressure
- [ ] B = B_ext + B_plasma and force residual J x B - grad p inside the plasma
- [ ] Optimization loop over the J coefficients
