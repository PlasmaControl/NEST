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
- [x] B = B_ext + B_plasma and force residual J x B - grad p inside the plasma (`nest/force.py`, `nest/fields.py`)
- [x] Jacobian construction (`nest/fields.py`: affine J, B, p; `nest/objectives.py`)
- [x] Optimization loop over the J coefficients (`nest/optimize.py`: Gauss-Newton trust region, lambda continuation; driver `nest/run_solve.py`)

Open items (validation, studies, optional features, memory): [TODO.md](TODO.md).

# Solve path

The modules in `nest/`, run from there (as the tests do):

```python
from boundary_potential import BoundaryPotential
from desc_input import DESCInput
from fields import Fields
from grid import CartesianGrid
from objectives import FieldDistance, ForceBalance, ObjectiveFunction
from optimize import gauss_newton
from pressure import Pressure

inp = DESCInput.load(eq_file, field=coils_file)
current = BoundaryPotential.fit(inp.eq, L, M, N)          # J = curl T; sym=True: stellarator symmetric
pressure = Pressure.fit(inp.eq, L, M, N, edge_order=2)
box = CartesianGrid.periodic_box(inp, inp.a / frac).locate_plasma(inp)
F = Fields(inp, current, pressure, box)                     # J, B, p, grad p affine in x; D_B costs one
                                                            # Biot-Savart per current coefficient
obj = ObjectiveFunction(F, [FieldDistance(F), ForceBalance(F, weight=1 / lam)])
res = gauss_newton(obj, F.fit_pressure(F.initial_state()))  # res["x"]; F.objects(res["x"]) -> series
```

On a cluster: `sbatch run.sbatch nest/run_solve.py --frac 10 --cache-dir DIR` (lambda
continuation, every end state checked on an independent grid; see the docstring of
`nest/run_solve.py`). Each change is recorded in [UPDATES.md](UPDATES.md).

# Questions?
- [ ] Should we raise s in (1-rho^2)^s to 3 or 4?
- [ ] How do we do the reconstruction of the potential current T (not simply a FZ series anymore) with .fit from DESC?
- [ ] Does the way we find the current density on cartesian grid has to remain this one: each Cartesian point is mapped back to flux coordinates by Newton and J is evaluated there exactly ; the nearest flux-grid node is only the Newton starting guess?
- [ ] Should we smooth out the B Fourier modes with a kernel G(k) (remove the periodicity)?
