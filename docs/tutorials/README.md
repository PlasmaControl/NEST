# NEST tutorials

Jupyter notebooks that walk through NEST, in the style of DESC's `docs/notebooks/tutorials`.
They are stored with their outputs; run them from this folder (they put `../../nest` on the path).

| notebook | what it covers |
|---|---|
| [`basic_solve.ipynb`](basic_solve.ipynb) | One solve from start to finish at low resolution (about 2 minutes on a laptop CPU). It covers the DESC input and its coils, the current potential J = curl T and its parameters, the pressure, the Cartesian box, the force residual of the starting point, the affine field maps, objective terms, Gauss-Newton with lambda continuation, checking on an independent grid, saving, and the stellarator-symmetry option. It ends with notes on production runs. |
