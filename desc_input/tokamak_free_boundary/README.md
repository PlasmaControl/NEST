# Axisymmetric free-boundary tokamak with J → 0 at the LCFS

A smooth Grad–Shafranov equilibrium for testing NEST in 2D: the force-balance
Jacobian is small enough to form and decompose exactly (`scripts/gs_*.py`).

## Equilibrium (`make_tokamak.py`)

- **Boundary:** taken from DESC's `DSHAPE_CURRENT` example: D-shaped, R0 = 3.51 m,
  a = 1 m (DESC's a = 1.20 m), elongation 1.47, weakly triangular. Ψ = 1 Wb, NFP = 1,
  N = 0, stellarator symmetric.
- **Profiles:** DSHAPE's profiles are replaced by smooth ones. Both are exact
  polynomials, `PowerSeriesProfile` of degree 6.
  - p = p0 (1 − ρ²)³ with p0 = 1.6 kPa, so p′ and p″ vanish at the edge.
  - I(s) = I0 (1 − (1 − s)³) with s = ρ² and I0 = −120 kA. Then dI/ds ∝ (1 − s)², so
    the toroidal current density and its slope vanish at the edge.
- **Edge current:** in axisymmetry J → 0 at the LCFS with zero slope, but only down
  to DESC's resolution floor (`edge_J_convergence.png`). The floor is 1 % of the
  peak at L = 16 and 0.05 % at L = 24.

| file | force error | B·n/\|B\| rms (max) | ⟨β⟩ | q(0) → q(1) |
|---|---|---|---|---|
| `tokamak_L12M12.h5` | 7.2e-5 | 4.9e-7 (1.5e-6) | 2.39 % | 1.18 → 7.43 |
| `tokamak_L16M16.h5` | 5.9e-6 | 4.8e-7 (1.6e-6) | 2.39 % | 1.18 → 7.46 |
| `tokamak_L24M24.h5` | 1.1e-7 | 4.8e-7 (1.5e-6) | 2.39 % | 1.18 → 7.46 |
| `tokamak_L32M32.h5` | 1.3e-8 | 4.8e-7 (1.5e-6) | 2.39 % | 1.18 → 7.46 |

The force error is ⟨|J×B − ∇p|⟩ / ⟨|∇B²/2μ0|⟩.

- **Main case:** `tokamak_L24M24.h5`, used by NEST's drift and null-space study.
- **Reference:** `tokamak_L32M32.h5`, for DESC's own error (`scripts/gs_check_end.py`).

Each `*_field.h5` is that equilibrium's external field.

## External field

With J → 0 at the LCFS there is no surface current. A fixed-boundary solution is
then also a free-boundary one, with

    B_ext = B_DESC − B_plasma(J_DESC)  in the plasma.

That field is curl-free and divergence-free there, (B_ext + B_plasma)·n = 0 on the
LCFS, and B is continuous across it.

`*_field.h5` realizes this B_ext with a `FourierCurrentPotentialField`:
- the winding surface is offset 0.8 m from the LCFS;
- G = −3.84 MA is set by the toroidal field;
- the single-valued Φ(θ), with |m| ≤ 16, comes from REGCOIL at λ = 1e-20, which
  minimizes B·n on the LCFS (virtual casing).

`DESCInput.B_ext` evaluates a surface current on a source grid of at least
(M, N) = (64, 128), and in chunks of 2000 points.

**Pressure jump.** The `[[B²+2μ0p]]/B²` values in `*_diagnostics.json` (about 6e-5
rms) use DESC's default quadrature for the surface current. That quadrature is too
coarse in ζ and overstates the jump. On a fine source grid the jump is about 1e-5,
and 2e-5 at most, at the outboard tip (`free_boundary_tokamak.py`).

## Free-boundary check (`free_boundary_tokamak.py`)

The LCFS of `tokamak_L16M16.h5` is released in its own field, with Φ and G fixed.
`BoundaryError` is then minimized under force balance, with p, I and Ψ fixed, by
`proximal-lsq-exact`. The LCFS modes are released in stages |m| ≤ 4, 8, 12.

- **Displacement:** the LCFS moves by at most 2.5 mm (2e-3 a), mostly as a small
  inward shift.
- **Residuals:** the pressure jump drops to about 5e-6, while B·n/|B| rises to
  about 1e-5.
- **Convergence:** each stage stops after 2–3 iterations ("bad approximation"),
  because the objective is already near its floor.

Outputs: `tokamak_L16M16_fb.h5`, `_fb_diagnostics.json` and `_fb.png`.

## Figures and running

- `plot_tokamak.py L M` draws `tokamak_L{L}M{M}_overview.png`: flux surfaces and
  winding surface, profiles, the edge current, |J| and the local force error, B·n
  and the pressure jump on the LCFS, and the current potential.
- To build an equilibrium on a CPU node (about 3 min at L = M = 16):

```bash
sbatch --time=01:00:00 --mem=32G -o logs/tokamak-%j.log ../../run_cpu.sbatch make_tokamak.py --LM 24 24 --device cpu
```
