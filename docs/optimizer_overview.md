# How the NEST fixed-boundary optimizer works

*Branch `ripple-landscape`, state of 2026-11-16 (commit `d3fb93b` and later). Driver:
`scripts/meanforce/mf_fixed.py`. Solver code: `nest/meanforce.py` (trust region, `Solver`),
`nest/meanforce_fixed.py` (fixed-boundary anchor, rows, metric, outer loop), `nest/cylindrical_bs.py`
(Biot–Savart).*

This note is meant to be read without the branch history. It explains what problem the optimizer
solves, every method it uses to solve it, the two newest options (`--subspace prev` and
`--subspace accel`), and which changes made it fast enough to run at 64/64/6. Section 8 lists the
recommended flags. Each method has a short worked example that can be checked by hand.

---

## Contents

1. [The problem](#1-the-problem)
2. [The structure that everything relies on](#2-the-structure-that-everything-relies-on)
3. [Outer loop: the augmented Lagrangian](#3-outer-loop-the-augmented-lagrangian)
4. [Inner loop: one trust-region step](#4-inner-loop-one-trust-region-step)
5. [The metric (preconditioner) and when it is rebuilt](#5-the-metric-preconditioner-and-when-it-is-rebuilt)
6. [Valley navigation: geodesic acceleration and the exact subspace search](#6-valley-navigation-geodesic-acceleration-and-the-exact-subspace-search)
7. [The changes that mattered most for speed](#7-the-changes-that-mattered-most-for-speed)
8. [Recommended flags](#8-recommended-flags)
9. [Glossary and code map](#9-glossary-and-code-map)

---

## 1. The problem

### 1.1 What is being computed

We start from a DESC equilibrium, which assumes nested flux surfaces. NEST looks for a nearby
magnetohydrodynamic (MHD) equilibrium that does **not** assume nested surfaces. The question is
posed as a constrained least-squares problem:

> Find the plasma current density **J** closest to DESC's current **J**_fit, subject to the
> force-balance residual being at most a chosen fraction α of DESC's own residual.

Written out,

$$
\min_u \ \phi(u) = \tfrac12 \lVert g(u) \rVert^2
\qquad \text{subject to} \qquad M(u) \le \varepsilon = \alpha \, M_\text{DESC}.
$$

Every symbol is defined below.

### 1.2 Unknowns: the current potential *u*

The current is written as a curl, $\mathbf J = \nabla \times \mathbf T$, with $\mathbf T$ expanded in a
Fourier–Zernike basis in DESC's flux coordinates $(\rho, \theta, \zeta)$ (class `BoundaryPotential`,
`nest/boundary_potential.py`). The coefficient vector is called **u**. It has three parts:
- surface-potential modes $\chi_{mn}$;
- the coefficients $a_{lmn}$;
- the coefficients $b_{lmn}$.

Writing $\mathbf J = \nabla\times\mathbf T$ guarantees three things for **every** u:
- $\nabla\cdot\mathbf J = 0$ exactly;
- $\mathbf J = 0$ on the last closed flux surface (LCFS), falling off as $(1-\rho)^{s-1}$ with `edge_power` $s = 2$;
- the net toroidal current $I$ is fixed at DESC's value.

**J is linear in u.** The number of unknowns, $n_u$, is set by the resolution `--LMN L M N`. For
L = M it is about (a fit to the measured counts, accurate to 0.1%) $\tfrac{(M+1)(M+2)}{2}(2N+1) + (M+1)(2N+1)$: 7,726 at 32/32/6, 11,734 at 40/40/6
and 28,750 at 64/64/6.

The geometry (the map from $(\rho,\theta,\zeta)$ to space) is DESC's and stays **fixed**. This is
the *fixed-boundary* problem.

### 1.3 The magnetic field

The field has two parts:

$$
\mathbf B = \mathbf B_\text{ext} + \mathbf B_\text{ann}[\mathbf J].
$$

- **$\mathbf B_\text{ann}[\mathbf J]$, the plasma's own field.** It comes from the free-space
  Biot–Savart law, evaluated on a cylindrical *annulus* grid that surrounds the plasma (§1.6). It is
  linear in J, and so linear in u.
- **$\mathbf B_\text{ext}$, the external field ("option A", `--bext desc-fit`).** It is defined as
  $\mathbf B_\text{ext} = \mathbf B_\text{DESC} - \mathbf B_\text{ann}[\mathbf J_\text{fit}]$, i.e.
  DESC's field minus the plasma field of DESC's current computed *on our own grid*. It is a fixed
  vector at every point.

With option A, the start $u = u_\text{fit}$ reproduces DESC's field exactly, and any grid error
enters only through $\mathbf B_\text{ann}[\mathbf J - \mathbf J_\text{fit}]$. That is why the
field-grid convergence checks (`--eval-only` on finer `--frac`) look at states that have moved away
from $J_\text{fit}$.

### 1.4 The pressure is eliminated, not solved for

For a given J and B we choose the pressure $p$ that best balances the Lorentz force
$\mathbf J\times\mathbf B$ in the least-squares sense:

$$
p^\star = \arg\min_p \sum_i w_i \,\lvert \mathbf J\times\mathbf B - \nabla p \rvert_i^2 .
$$

Here $i$ runs over the **collocation points**, a Gauss–Legendre grid in ρ times uniform grids in θ
and ζ. The weights $w_i = dV_i/V$ are volume fractions, so $\sum_i w_i = 1$. The pressure is
expanded in its own basis with $n_p$ coefficients ($p = 0$ on the LCFS).

Discretely, let $\mathsf P$ be the matrix whose columns are $\sqrt{w}\,\nabla p_k$ for the pressure
basis functions, and $a = \sqrt w\,(\mathbf J\times\mathbf B)$, stacked over points. Then

$$
c^\star = G^{-1}\mathsf P^T a, \qquad G = \mathsf P^T \mathsf P \ (n_p\times n_p),
\qquad f = a - \mathsf P c^\star = (I - \mathsf P G^{-1} \mathsf P^T)\, a .
$$

This is the weak form of $\nabla^2 p = \nabla\cdot(\mathbf J\times\mathbf B)$, which is why the code
calls it "weak Poisson". $G$ depends only on the fixed geometry, so it is built once and Cholesky
factored once (`Lg`). The projection is applied matrix-free, by one evaluation of $\nabla p$ and its
transpose, with one refinement pass. The residual force at point $i$ is
$\mathbf F_i = f_i/\sqrt{w_i}$.

*Why eliminate p?* If $p$ were an extra unknown, the conditioning of the combined problem would grow
with spectral resolution, which rules out iterative (Krylov) methods. Eliminating it keeps the
problem in u alone, and keeps it well conditioned enough for the methods below.

A consequence to keep in mind: NEST's pressure is whatever best balances $\mathbf J\times\mathbf B$.
It is not tied to flux surfaces and carries no physical constraint, such as fixed mass per flux tube.

### 1.5 The objective and the constraint

- **Misfit:** $g = \sqrt w\,(\mathbf J - \mathbf J_\text{fit})/\text{rms}(\mathbf J_\text{fit})$,
  stacked over points, and $\phi = \tfrac12\lVert g\rVert^2$. So $\lVert g\rVert$ is the relative rms
  current change $|J - J_\text{fit}|/\text{rms}$ quoted in every log.
- **Force measure:** the volume mean $M = \sum_i w_i\lvert\mathbf F_i\rvert/\text{norm}$, with norm $=$
  DESC's $\langle|\nabla B^2/2\mu_0|\rangle$. $M_\text{DESC}$ is DESC's own value of $M$; for the
  ripple study it is $4.9995\times10^{-7}$ in code units. Forces are always quoted as
  $M/M_\text{DESC}$.
- **Bound:** $M \le \varepsilon = \alpha M_\text{DESC}$ (`--alphas`). The logs print $M/\varepsilon$,
  so $M/\varepsilon = 16.66$ at $\alpha = 0.001$ means $M = 0.0167\,M_\text{DESC}$.
- **Alternative bound (`--force-norm p`):** an interior p-norm
  $M_p = (\sum_\text{int} w|F|^p/\sum_\text{int} w)^{1/p}/\text{norm}$, used for max-type
  experiments. Everything below holds for either measure.

### 1.6 The field grid (Biot–Savart)

$\mathbf B_\text{ann}$ is computed in the Coulomb-gauge vector potential, $\nabla^2\mathbf A = -\mu_0\mathbf J$,
on an annulus $R_\text{in}\le r\le R_\text{out}$, $|z|\le z_0$, with N_φ planes per field period
(`--N-phi`) and node spacing about $a/\text{frac}$ (`--frac`). The steps are:
1. Locate every annulus node in flux coordinates (Newton inversion of DESC's map; `locate_plasma_planes`).
2. Evaluate J there ($\mathbf J = 0$ outside the plasma).
3. Fourier transform in φ.
4. Solve one 2-D free-space problem per toroidal mode in $(r,z)$: a Dirichlet solve by
   eigendecomposition, plus a single-layer boundary correction ("James' method").
5. Take the curl and interpolate **B** back to the collocation points.

This is `CylindricalSolver` in `nest/cylindrical_bs.py`. Its cost and memory scale with the number
of nodes, about 7 M at 64/64/6 a/160.

---

## 2. The structure that everything relies on

Three facts make the problem tractable.

1. **J is linear in u, B is affine in u, the pressure projection is linear.** So
   $a = \sqrt w\,(\mathbf J\times\mathbf B)$ is **exactly quadratic** in u, and so is the force vector
   $\mathbf F_i(u)$ at every point. The misfit $g(u)$ is **affine**.
2. **Products are cheap; matrices are not.** A Jacobian–vector product $J v$ costs about one
   Biot–Savart solve, and its transpose $J^T w$ about two to three. A dense Jacobian would be
   $3n_\text{pts}\times n_u$, roughly $10^6\times 3\times10^4$ at 64/64/6, which we never form.
   Everything iterative is therefore **matrix-free**.
3. **The force is a small difference of large terms.** $\mathbf J\times\mathbf B$ and $\nabla p$
   nearly cancel, $|F| \sim 10^{-7}$ of $|\nabla B^2/2\mu_0|$ near the optimum. So the merit is
   known only to about $10^{-9}$–$10^{-7}$ relative. This *evaluation noise* $\sigma_f$ matters for
   step acceptance (§4.5).

**Worked example (quadratic structure).** Along any line $u = u_0 + t\,d$,
$\mathbf F_i(t) = \mathbf F_i^{(0)} + t\,\mathbf F_i^{(1)} + t^2\,\mathbf F_i^{(2)}$ exactly. Three
evaluations, at $t = 0$ and $t = \pm1$, determine it:

$$
\mathbf F^{(1)} = \tfrac12\left[\mathbf F(1) - \mathbf F(-1)\right], \qquad
\mathbf F^{(2)} = \tfrac12\left[\mathbf F(1)+\mathbf F(-1)\right] - \mathbf F(0).
$$

Check with a scalar: $F(t) = 1 - 3t + t^2$ gives $F(0)=1$, $F(1)=-1$, $F(-1)=5$, so
$F^{(1)} = (-1-5)/2 = -3$ and $F^{(2)} = (-1+5)/2 - 1 = 1$. ✓

With these coefficients, $|F(t)|$ and hence $M(t)$ is known **exactly for every t**, with no further
Biot–Savart solves. The plane maps (`plane_map.py`) and the subspace search (§6.3) both use this.

---

## 3. Outer loop: the augmented Lagrangian

The constraint $M\le\varepsilon$ is handled by a Powell–Hestenes–Rockafellar (PHR) augmented
Lagrangian (`Weights` in `meanforce.py`, `FixedSolver.solve_alpha` in `meanforce_fixed.py`).

### 3.1 The merit

Write the constraint as $c_F = M/\varepsilon - 1 \le 0$. With a multiplier $\nu \ge 0$ and a
penalty $\rho > 0$, the merit minimized at fixed $(\nu,\rho)$ is

$$
\mathcal L_A(u) = \phi(u) + \psi(c_F), \qquad
\psi(c) = \frac{\rho}{2}\left[\max\!\left(0,\ c + \frac{\nu}{\rho}\right)^2 - \left(\frac{\nu}{\rho}\right)^2\right].
$$

Its slope is $\psi'(c) = \max(0,\ \nu + \rho c)$, the *effective* multiplier, called psi′ in logs.
When the bound is satisfied with room to spare ($c < -\nu/\rho$), $\psi$ is flat and only $\phi$ is
minimized. When violated, the merit pushes M down with strength $\nu + \rho c$.

**Worked example.** At $\nu = 0$, $\rho = 1$ and $c_F = 0.2$ (M is 20% over the bound):
$\psi = \tfrac12(0.2^2 - 0) = 0.02$ and $\psi' = 0.2$. At $c_F = -0.1$ (10% under):
$\psi = \tfrac12(\max(0,-0.1)^2 - 0) = 0$ and $\psi' = 0$.

### 3.2 One outer iteration

1. **Start of a bound** (first outer iteration of each α): set ν by least squares (`ls_multiplier`;
   on by default, off with `--no-mult-ls`). Choose the ν that makes the gradient of the Lagrangian
   $\phi + \nu c_F$ smallest in the metric norm, clipped at 0. This is the same idea as DESC's
   `lsq-auglag` initial multipliers.
2. **Metric refresh** (§5): rebuild the preconditioner, unless psi′ changed by less than a factor
   `--outer-rebuild-factor` (2) since the last build. After a rebuild the trust radius is set to the
   gradient length in the new metric, $|L^{-1}\nabla\mathcal L_A|$. If the metric is kept, the radius
   carries over. (With `--precond hj` there is no refresh, and the radius is reset to at least
   `--delta0`.)
3. **Inner solve:** trust-region steps on $\mathcal L_A$ at fixed $(\nu,\rho)$, up to `--maxiter`
   steps (§4).
4. **Multiplier update:** $\nu \leftarrow \max(0, \nu + \rho c_F)$. With the LS multiplier on, the
   update is clipped to $[0.25, 4]\times$ the LS estimate.
5. **Penalty update:** if the violation did not fall below $0.25\times$ its previous value,
   $\rho \leftarrow 10\rho$ (up to $10^8$).
6. **Stop** when $c_F \le 0.01$ (M within 1% of the bound), the inner solve was stationary, and the
   bound is *active* whenever $\nu > 0$ ($|c_F| \le 0.01$). The last condition, "complementarity",
   stops the sweep from moving on from a state that is feasible but slack and not optimal.

Several values in `--alphas` form a continuation: each α starts from the previous end state.

**When the bound is infeasible** (our α = 0.001 floor runs): $c_F \gg 1$ stays large, ν and ρ grow,
and the penalty term dominates the merit. The inner loop then effectively *minimizes M*, with $\phi$
as a weak regularizer. Those runs end on the time budget (`--time-budget`, or a `STOP` written to the
`CONTROL` file in the mirror directory), not on the outer stop test.

---

## 4. Inner loop: one trust-region step

At fixed $(\nu,\rho)$ each inner step does four things:
1. build a quadratic model of $\mathcal L_A$ around the current u;
2. minimize the model inside a trust region (an ellipsoid) with a Krylov method;
3. evaluate the true merit at the trial point;
4. accept or reject it and resize the region.

The code is `Solver.inner` in `meanforce.py`.

### 4.1 The model: rows whose sum of squares approximates the merit

The solver works with a **residual vector** r(u) and its Jacobian, so that the model is
Gauss–Newton: $m(d) = \tfrac12\lVert r + J_r d\rVert^2$. The rows (`FixedRows`) are:
- **misfit rows** g, exact because g is affine;
- **force rows** $s_i\,\mathbf z_i$, one 3-vector per point, with $\mathbf z_i = w_i\mathbf F_i$;
- **one penalty-curvature row** $a_F\cdot d$.

The force rows are chosen so that their squares reproduce the *linear* part of $\psi$ in M. Since
$M\propto\sum_i|\mathbf z_i|$ and

$$
s_i^2 = \frac{\psi'}{\varepsilon\,\text{norm}\,t_i},\qquad t_i = \max(|\mathbf z_i|,\ w_i\tau),
$$

we get $\tfrac12\sum_i s_i^2|\mathbf z_i|^2 = \tfrac{\psi'}{2}\cdot\tfrac{M}{\varepsilon}$ (with
$t_i = |\mathbf z_i|$). Its gradient is the exact gradient $\psi'\,\nabla c_F$. The floor
$t_i\ge w_i\tau$ (`tau_frac` $=10^{-3}$ of the bound) avoids dividing by zero where $F_i = 0$.

Because $\lvert \mathbf z + \delta\rvert \le \frac{\lvert\mathbf z + \delta\rvert^2 + \lvert\mathbf z\rvert^2}{2\lvert \mathbf z\rvert}$
(the arithmetic–geometric mean inequality), this square-of-norm model is a **majorizer**: it never
under-predicts $|\mathbf z+\delta|$.

The curvature of $\psi$ itself, $\rho\,(\nabla c_F)(\nabla c_F)^T$, enters through the extra row
$a_F = \sqrt{\rho}\,\nabla c_F$. It is also kept within 5% of the bound (`band`), so that the model
can see a step that crosses the bound.

### 4.2 The β model (`--model beta`, the default)

The majorizer is safe but pessimistic along $\hat{\mathbf z}$ (the direction of $\mathbf F_i$
itself). Split a change $\delta$ into its part along $\hat{\mathbf z}$, $\delta_\parallel$, and the
rest, $\delta_\perp$. Then

$$
|\mathbf z+\delta| \approx |\mathbf z| + \delta_\parallel + \frac{|\delta_\perp|^2}{2|\mathbf z|}
\quad\text{(true, to second order)},
$$

$$
\text{majorizer:}\quad |\mathbf z| + \delta_\parallel + \frac{\delta_\parallel^2 + |\delta_\perp|^2}{2|\mathbf z|},
\qquad
\beta\text{ model:}\quad |\mathbf z| + \delta_\parallel + \frac{\beta\,\delta_\parallel^2 + |\delta_\perp|^2}{2|\mathbf z|}.
$$

The true norm has *no* curvature along $\hat{\mathbf z}$ (a norm is linear along its own direction
until it reaches zero). The majorizer adds curvature 1 there; the β model adds β. `FixedRowsB`
implements this by projecting each force row with
$(I - (1-\sqrt\beta)\,\hat{\mathbf z}\hat{\mathbf z}^T)$ and rescaling the residual so the gradient
stays exact. β = 1/2 is the most optimistic choice that never predicts $|\mathbf z+\delta| < 0$. The
model stays positive semidefinite, so the subproblem never sees negative curvature.

**Worked example.** Take $|\mathbf z| = 2$.
- *Along* $\hat{\mathbf z}$, $\delta_\parallel = 0.5$: the truth is 2.5; the majorizer predicts
  $2.5 + 0.25/4 = 2.5625$; the β = ½ model predicts $2.5 + 0.125/4 = 2.53125$.
- *Perpendicular*, $|\delta_\perp| = 1$: the truth is $\sqrt5 = 2.2361$; both models predict
  $2 + 1/4 = 2.25$.

The β model is closer along $\hat{\mathbf z}$ and identical across.

(The exact Hessian model, `--model hvp`, was 2.8× faster at α = 0.7 but collapsed the trust radius at
α = 0.05, where the kink of $|\mathbf z|$ at zero matters. The β model is the default for that reason.)

### 4.3 The trust region and the metric

The step is measured in a **metric**: $y = L^T d$, where $L$ is a lower-triangular Cholesky factor
(§5). The trust region is $\lVert y\rVert\le\Delta$, i.e. the ellipsoid $d^T(LL^T)d \le \Delta^2$
in the original coordinates. In y the model is $m(y) = f_0 + g_y\cdot y + \tfrac12 y^T B y$, with
$g_y = L^{-1}J_r^T r$ and $B = L^{-1}J_r^T J_r L^{-T}$. A good L makes B close to the identity, so
the Krylov solve below converges in few iterations. Every product goes through
`met.back(y)` $= L^{-T}y$ and `met.back_T(d)` $= L^{-1}d$, which are triangular solves.

### 4.4 Solving the subproblem: GLTR (Lanczos)

$\min_{\lVert y\rVert\le\Delta} m(y)$ is solved by the generalized Lanczos trust-region method
(GLTR; `lanczos_tr_step_H`):
1. Build an orthonormal Krylov basis $Q_k$ of $\{g_y, Bg_y, B^2 g_y,\dots\}$, with full
   reorthogonalization. In that basis B becomes a $k\times k$ tridiagonal matrix $T_k$.
2. Solve the small problem $\min_{\lVert h\rVert\le\Delta}\ \lVert g_y\rVert h_1 + \tfrac12 h^T T_k h$
   exactly (Moré–Sorensen on the eigendecomposition of $T_k$, including the "hard case").
3. Set $y = Q_k h$.

The step is *interior* if the unconstrained minimum lies inside the ellipsoid, and a *boundary step*
("hit" in logs) otherwise.

**Stopping rule (`--lanczos-stop tr`, default).** Stop when the trust-region residual
$\beta_k|h_k| \le \texttt{cg\_tol}\cdot\lVert g_y\rVert$ (`--cg-tol` $10^{-2}$), or at the cap
`--cg`. The older rule (`both`) also waited for the *unconstrained* minimizer to converge. That
value only feeds a diagnostic, but it converges slowly in the soft directions, so boundary steps ran
to the cap (§7).

Each Lanczos iteration costs one $J_r v$ and one $J_r^T w$, i.e. about one forward and one adjoint
Biot–Savart solve.

### 4.5 Accepting the step and resizing the region

The **ratio** compares the actual decrease with the model's prediction:
$\text{ratio} = (f - f_\text{new})/(f - m(y))$. In the base rules:
- **Acceptance:** a step is accepted if it lowers the merit at all ($f_\text{new} < f$).
- **Poor ratio:** if ratio < 0.25, the radius becomes $0.25|y|$.
- **Good boundary step:** if ratio > 0.75 and the step hit the boundary, the radius grows to
  $\Delta \leftarrow \max(2|y|, \Delta)$.

Three refinements were needed for runs near a floor (all opt-in, all recommended):

- **Noise-tolerant ratio (`--f-noise auto`).** At every inner solve the solver measures $\sigma_f$,
  the spread of the merit at six points $10^{-10}$ away. The ratio then becomes
  $(f - f_\text{new} + 2\sigma_f)/(\text{pred} + 2\sigma_f)$. A step is accepted if this ratio
  exceeds 0.1; a rejection sets $\Delta = 0.25\min(|y|,\Delta)$. A step whose predicted decrease is
  below $10\sigma_f$ (and that does not raise the merit by more than $4\sigma_f$) is accepted without
  changing the radius. Since the merit is deterministic, rejecting such a step would only repeat it.
- **Radius by step (`--radius-by-delta`).** An accepted *interior* step with a poor ratio shrinks the
  radius to $\max(0.25|y|, 0.25\Delta)$, not to $0.25|y|$. A single tiny wall-correction step then
  cannot collapse the radius by orders of magnitude.
- **No share stop (`--no-share-stop`).** Disables the "stationary" exit triggered when the predicted
  relative decrease of the unconstrained Krylov minimizer falls below $10^{-8}$. In long shallow
  valleys that test fired after short Lanczos runs that had only seen stiff directions.

**Other inner stops:**
- the relative decrease falls below `ftol` (without `--f-noise`);
- $|g_y|$ falls below `--gtol` times its first value;
- the merit is flat over `stat_window` (10) accepted steps;
- the step budget (`--maxiter`) is reached;
- the radius vanishes.

**Worked example (ratio).** Suppose $f = 122.6416$, the model predicts $m(y) = 122.6375$
(pred $= 0.0041$), and the trial gives $f_\text{new} = 122.6358$ (actual decrease 0.0058). Then
ratio $= 0.0058/0.0041 = 1.41$, the step is accepted, and since it hit the boundary the radius
doubles.

---

## 5. The metric (preconditioner) and when it is rebuilt

### 5.1 What the metric is

With `--precond force` (default), the metric is

$$
LL^T = H_J + H_{F,\text{loc}} + a_F a_F^T .
$$

- $H_J = J_g^TJ_g$ is the Gram matrix of the misfit rows. It is constant, built once.
- $H_{F,\text{loc}}$ is the Gauss–Newton Gram matrix of the force rows, keeping only the **local**
  part of the linearized force $\sqrt w\,(\delta\mathbf J\times\mathbf B_0)$. The non-local
  Biot–Savart part $\mathbf J_0\times\delta\mathbf B$ is dropped, so the matrix can be accumulated
  point by point over radial slices of the collocation grid. The pressure projection
  $I - \mathsf P G^{-1}\mathsf P^T$ is kept exactly.
- $a_F a_F^T$ is the penalty curvature of §4.1.

$L$ is a dense $n_u\times n_u$ Cholesky factor. Applying $L^{-1}$ or $L^{-T}$ is a triangular solve
costing $O(n_u^2)$, small next to a Biot–Savart solve.

*Why not just $H_J$?* $H_J$ ignores the force-bound curvature, which dominates when α is small and
the multiplier is large. With the $H_J$-only metric the subproblem at α = 0.05 was far from
converged at the 400-iteration cap (§7).

### 5.2 How it is built (the "lean" build)

The matrices are accumulated slice by slice, with no padded $n_u^2$ arrays, on donated buffers.
$H_J$ is kept in host RAM and added in row blocks (`force_metric_lean`). Peak device memory:
- **H_J phase:** about $3\times 8n_u^2$ bytes on top of what is resident;
- **force-metric phase:** about $8(n_u^2 + 2n_un_p + 2n_p^2)$ bytes plus one slice of columns.

At 64/64/6 ($n_u = 28{,}750$) the measured peaks are 29.5 GB and 32.4 GB, and a force-metric
rebuild takes about 290–380 s on an A100-80GB. Memory projections for larger rungs are in
`results/ripple_landscape/memprobe/README.md`.

### 5.3 When it is rebuilt

- **At each outer iteration**, unless psi′ changed by less than a factor of 2 since the last build.
- **Inside the inner loop** (`use_models`), before a subproblem, when the previous subproblem
  - hit the Lanczos cap; or
  - accumulated more "extra" Lanczos time, i.e. iterations above the first post-rebuild count, than
    one rebuild costs ("rent-or-buy").

  The radius is then mapped to the new metric by the ratio of $|L^{-1}\nabla\mathcal L_A|$.
- **`--no-rebuild-on-hit` (recommended):** a subproblem whose step hit the trust-region boundary
  triggers neither inner rule, and its time is not counted. On a boundary step a cap hit only means
  Lanczos reached the edge of the ellipsoid before converging; it does not mean the metric is stale.
  Interior cap hits still trigger a rebuild.

---

## 6. Valley navigation: geodesic acceleration and the exact subspace search

### 6.1 The problem these solve

Near the α = 0.001 floor the log has a characteristic rhythm:
1. a long boundary step ($|y|\sim 10^2$–$10^3$, 250–900 s of Lanczos, ratio ≈ 1);
2. a sharp jump in the gradient ($|g|$ from about 1 to about 100);
3. several tiny interior steps ($|y|\sim 0.1$) that each recover about $10^{-3}$ in $M/\varepsilon$;
4. occasionally a rejected boundary step at the largest radius (ratio −0.4 to −0.7), throwing away
   a 400–900 s Lanczos solve.

This is a **curved valley**. The model is a quadratic in u, so its minimizer lies along a straight
line from the current point. The floor of the valley bends. A long straight step leaves the floor
and lands on the valley wall, and the short steps climb back down. The bend is the second-order term
of the force: since $\mathbf F$ is exactly quadratic,

$$
\mathbf F(u+d) = \mathbf F(u) + \mathcal J d + \mathbf q(d,d),
$$

with $\mathcal J$ the force Jacobian and $\mathbf q(d,d)$ the exact quadratic part (in the field
picture, $\mathbf q = \Pi[\sqrt w\,(\delta\mathbf J\times\delta\mathbf B)]$). The Gauss–Newton model
drops $\mathbf q$. Along long steps, $\mathbf q$ is what makes the prediction wrong.

### 6.2 Geodesic acceleration (`--geo-accel`; also the `accel` direction below)

The idea, from Transtrum and Sethna's geodesic Levenberg–Marquardt: follow the curve along which
the *linearized* residual would move, rather than the straight line. To second order the curved
step is $u + d + a$, with the correction a chosen so that the linear term cancels the quadratic
error:

$$
\mathcal J a \approx -\mathbf q(d,d)
\quad\Longrightarrow\quad
a = -(\mathcal J^T\mathcal J)^{-1}\mathcal J^T\mathbf q(d,d).
$$

In practice (`Solver._geodesic`, `Solver._accel_dir`):
1. **Exact second-order term.** From the trial $\mathbf F(u+d)$ and one more evaluation $\mathbf F(u-d)$,
   $\mathbf q(d,d) = \tfrac12[\mathbf F(u+d)+\mathbf F(u-d)] - \mathbf F(u)$, as in §2. It is written
   as the weighted rows $q$.
2. **Solve for a** with the same Lanczos machinery and metric, inside a radius
   $\texttt{accel\_amax}\cdot|y|$ (0.75), to tolerance $10^{-4}$ and at most 300 iterations.
3. **With `--geo-accel` alone:** evaluate $u+d+a$ and keep it if its merit beats the straight trial.

**Cost:** one extra state evaluation, one extra Lanczos solve, one more evaluation.

**What we measured at 64/64/6:** a removes 77–84% of $\mathbf q$, but it is tiny, with
$|a|/|y| \approx 10^{-3}$. Each boundary step gains only a little over the straight step, and the
extra Lanczos solve costs more than that gain (0.79× the control at matched time). On its own it is
not recommended. Its *direction*, however, is valuable inside the subspace search.

### 6.3 Exact subspace search (`--subspace prev`, `accel`, `prev,accel`)

The idea: on each boundary step, do not trust the quadratic model's step length or its straight
line. Instead, find the best point exactly in a small subspace that contains the Lanczos step.

Because $\mathbf F$ is exactly quadratic and $g$ is affine (§2), on any K-dimensional affine
subspace $u(c) = u + \sum_{i=1}^K c_i e_i$,

$$
\mathbf F(c) = \mathbf F_0 + \sum_i c_i\,\mathbf F_i + \sum_i c_i^2\,\mathbf F_{ii} + \sum_{i<j} c_ic_j\,\mathbf F_{ij},
\qquad g(c) = g_0 + \sum_i c_i\,g_i,
$$

at every collocation point. The coefficients come from evaluations at $u\pm e_i$ and $u + e_i + e_j$.
The trial $u + d$ is already evaluated, so the new evaluations number $2K - 1 + K(K-1)/2$: **4 for
K = 2, 8 for K = 3**. With them, the true merit $\mathcal L_A(c)$ is available exactly for every c,
with no Biot–Savart solve:
- $M(c) = \sum_i w_i|\mathbf F_i(c)|/\text{norm}$ is one weighted sum on the GPU;
- $\phi(c)$ is a quadratic in c;
- $\psi$ is a scalar function of $M(c)$.

**The directions:**

| name | $e_1$ | extra direction | why |
|---|---|---|---|
| (always) | $d$, the Lanczos step | — | the model's best straight step |
| `prev` | | the previous accepted step | in a bending valley, the last two steps span the local plane of the bend (a "momentum" direction) |
| `accel` | | the acceleration direction a of §6.2 | points along the curvature of the force map itself |

The extra directions are rescaled to the same metric length $|y|$ as d, so that the coefficients are
comparable.

**The search** (`Solver._subspace`):
1. Evaluate a grid: $c_1\in[0, 4]$ (`--subspace-smax`) and the others in $[-2, 2]$
   (`--subspace-tmax`); 33×17 points for K = 2 and 33×9×9 for K = 3. The straight step
   $c = (1,0,\dots)$ and the current point $c = 0$ are always included.
2. Polish the best grid point with Nelder–Mead, at most 200 function calls.
3. **Evaluate the minimizer for real** (one more state evaluation). Take it if its merit is below
   both the current point and the straight trial.
4. Set the radius from the metric length $|y^\star|$ of the step actually taken:
   - $2|y^\star|$ if $|y^\star|\ge 0.9\Delta$, i.e. the step went to the edge or beyond;
   - otherwise $\max(|y^\star|, 0.25\Delta)$.

What this buys:
- **Steps can be longer than the trust radius:** $c_1$ up to 4, and in practice up to 14.
  The radius grows as fast as the valley allows.
- **No boundary step is wasted.** If the straight step overshoots, the search finds the best point
  along the same directions. In one A/B step the straight trial had merit 255.8 against 117.2 at the
  current point; the search chose $c_1 = -0.51$, half a step *backwards*, and gained 0.11 in
  $M/\varepsilon$ at once.
- **The bend is followed,** through the `prev` and `accel` directions.

**A subtlety: rounding grows with the extrapolation.** The coefficients come from differences of
nearly equal force fields, and those carry rounding at the $\sigma_f$ level. Extrapolating to
$|c|\sim 10$ multiplies that error by about $|c|^2$. At 64/64/6 the predicted and actual merits
differ by $10^{-7}$–$5\times10^{-5}$ relative, rising roughly as $|c|^2$. That error is logged
("rel …"), but **the step is judged only by its real evaluation**. (An early version rejected any
step whose prediction was off by more than $10^{-6}$, and so rejected every improvement; fixed in
`02fe740`.)

**Cost per boundary step:**
- `prev`: 5 state evaluations;
- `prev,accel`: 9 state evaluations plus the acceleration Lanczos solve, about 100 s at 64/64/6.

Both are small next to the 250–900 s of the boundary step's own Lanczos solve.

**Worked example (1-D, by hand).** Use the scalar $F(t) = 1 - 3t + t^2$ from §2, and let the merit
be $|F|$. A trust region capped at $t \le 1$ gives the straight step $t = 1$, $|F| = 1$. The exact
reconstruction knows $|F(t)|$ for all t. Its minimum is at the root $t = (3-\sqrt5)/2 = 0.382$, with
$|F| = 0$. If the cap were instead $t\le 3$, the straight step $t = 3$ would give $|F(3)| = 1$ again
(an overshoot past the minimum), while the search would still find $t = 0.382$. The subspace search
is this, in two or three dimensions, with the true merit.

### 6.4 Measured effect (64/64/6, a/200, α = 0.001)

Four arms started from the same state ($M/\varepsilon = 16.6615$) and ran with the same flags
otherwise, one after another on one A100-80GB, control last. Data: `results/ripple_landscape/valley/`.

| arm | $M/\varepsilon$ at 2280 s of solve | decrease | × control | × control per Lanczos iteration | rejected steps |
|---|---|---|---|---|---|
| `--subspace prev,accel` | 16.3101 | 0.351 | **2.03** | 2.99 | 1 |
| `--subspace prev` | 16.4032 | 0.258 | 1.49 | 1.49 | 0 |
| `--geo-accel` | 16.5246 | 0.137 | 0.79 | 0.40 | 0 |
| control | 16.4886 | 0.173 | 1.00 | 1.00 | 1 |

Notes:
- One run per arm; the lead of `prev,accel` comes from a few large steps, so read the factor as
  indicative.
- The end states, re-evaluated on a finer field grid (a/240), agree with the solve grid
  (0.0162–0.0165 $M_\text{DESC}$), so no arm gained by exploiting grid error.

---

## 7. The changes that mattered most for speed

Ordered by impact, with the measurement behind each.

| # | change | what it fixed | measured effect |
|---|---|---|---|
| 1 | **Pressure eliminated by the weak-Poisson projection** (§1.4), unknowns u only | with p as an unknown, conditioning grows with resolution | made iterative (Krylov) solves possible at all |
| 2 | **Matrix-free products + GLTR trust region** (§4.4), instead of a dense Jacobian with QR | dense J and QR are $O(n_\text{rows}n_u)$ memory and $O(n_\text{rows}n_u^2)$ time | the fixed-boundary driver runs only the matrix-free path (the dense QR route was dropped for cost); every result on this branch uses it |
| 3 | **Cylindrical annulus Biot–Savart** (§1.6), instead of a Cartesian FFT box | box points wasted outside the torus | about 7× fewer grid points at equal h, with 0.7–0.8× the box's interior error |
| 4 | **Force preconditioner** $H_J + H_{F,\text{loc}} + a_Fa_F^T$ (§5.1), instead of $H_J$ alone | $H_J$ ignores the force-bound curvature | gate: Lanczos 41 → 13; at α = 0.05 with $H_J$ alone the subproblem was unconverged at the 400 cap (residual/$\lVert g\rVert$ = 1.21) |
| 5 | **Lanczos stop on the trust-region residual** (`--lanczos-stop tr`, §4.4) | the old rule waited for a diagnostic to converge | boundary steps 400 (cap) → 79–102 iterations with identical steps, about 3–5× fewer near the bound |
| 6 | **Exact subspace search, prev + accel** (§6.3) | straight steps cut across a curved valley; overshoots were thrown away | 2.0× the control's descent at matched time, 3.0× per Lanczos iteration |
| 7 | **Stall fixes:** noise-tolerant ratio, radius by step, no share stop (§4.5) | radius collapse on merit noise; false "stationary" exits | the Q2b "floor" at α = 0.1 turned out to be solver-limited; the push-through kept descending at a steady rate (0.0023 in $M/\varepsilon$ per 12 min) where the old rules stopped |
| 8 | **Lean metric build** (§5.2) | padded $n_u^2$ copies; $H_J$ on the device | at $n_u = 41{,}206$ the metric peak fell from about 101 GB (projected) to 53 GB; at 64/64/6 the measured peak is 32.4 GB |
| 9 | **`--locate-chunk 131072`** | the locate step's last batch allocated chunk × DESC modes × 8 B (up to 42 GB) | locate transient about 5 GB; it was the peak of every earlier ripple run, and the cause of the a/100 OOM on 40 GB cards |
| 10 | **`--cg 1500 --no-rebuild-on-hit`** (§5.3) | each boundary step hit the 400 cap and triggered a 335 s inner rebuild that bought about 0.1% | removed the wasted rebuilds (about three-quarters of wall time in the first α = 0.001 run); on its own it did **not** speed up the descent per minute, but the subspace search builds on it |

Two changes tried and found **not** to help in this regime:
- **Geodesic acceleration alone (0.79×).** The correction is too small at the step lengths the trust
  region allows.
- **The exact-Hessian model (`--model hvp`).** It was faster far from the bound (2.8× at α = 0.7) but
  collapsed the radius near it.

---

## 8. Recommended flags

For tight-bound, long-valley runs (e.g. 64/64/6 at α = 0.001):

```
python scripts/meanforce/mf_fixed.py --eq <desc.h5> --bext desc-fit --LMN 64 64 6 --N-phi 25 --frac 200 \
    --alphas 0.001 --max-outer 8 --maxiter 80 \
    --subspace prev,accel --cg 1500 --no-rebuild-on-hit \
    --f-noise auto --radius-by-delta --no-share-stop \
    --locate-chunk 131072 --snap-outer --mirror <dir> --out <dir>/<tag> [--start <state.npz>] [--time-budget S]
```

- **Leave these at their defaults:** `--model beta`, `--precond force` (lean build),
  `--lanczos-stop tr`, `--cg-tol 1e-2`, `--delta0 1e-3`, and `--accel-amax 0.75 --accel-tol 1e-4
  --accel-maxiter 300` (these also set the subspace acceleration direction).
- **Do not add `--geo-accel` on top of `--subspace …accel`.** The subspace search already uses the
  acceleration direction.
- **Field grid:** keep $h\,L/a \lesssim 0.4$, i.e. `--frac` ≳ 2.5 L. Evaluate the end states one grid
  finer (`--eval-only --frac <finer>`), and compare states only on a common grid.
- **Warm starts:** `--start file.npz` loads u, ν and ρ. Strip ν and ρ for fresh multipliers.
  `--start-LMN L M N` embeds a lower-resolution state.
- **Control:** write `STOP` (save and exit) or `NEXT` (next α) to `<mirror>/CONTROL` while running.
- **Memory** at 64/64/6 a/160–a/200 is about 33 GB peak. 80/80/6 a/200 needs more than an 80 GB card
  with the current Cholesky (see `results/ripple_landscape/memprobe/README.md`).

---

## 9. Glossary and code map

| term | meaning |
|---|---|
| u, $n_u$ | current-potential coefficients (the unknowns) and their number |
| $n_p$ | pressure coefficients (eliminated) |
| $w_i$, $\sqrt w$ | collocation volume weights ($\sum w_i = 1$) and their square roots |
| $\mathbf z_i = w_i \mathbf F_i$ | weighted force at point $i$ |
| $M$, $M_\text{DESC}$, $\varepsilon$ | volume-mean force / norm, DESC's value of it, the bound $\alpha M_\text{DESC}$ |
| $c_F$ | $M/\varepsilon - 1$ (log: `M/eps` $= 1 + c_F$) |
| $\nu$, $\rho$, psi′ | multiplier, penalty, effective multiplier $\max(0,\nu+\rho c_F)$ |
| φ | $\tfrac12 |g|^2$, the current misfit term |
| $L$, metric | Cholesky factor defining $y = L^Td$ and the ellipsoidal trust region |
| Δ, $|y|$ | trust radius and step length in the metric |
| hit / boundary step | the subproblem solution lies on $|y| = \Delta$ |
| ratio | actual / predicted decrease of the merit |
| $\sigma_f$ | measured evaluation noise of the merit |
| `a/N` | field-grid spacing a/N (a = minor radius) |

| file | contents |
|---|---|
| `scripts/meanforce/mf_fixed.py` | driver: setup, flags, alpha sweep, logging, eval-only, mirrors |
| `nest/meanforce_fixed.py` | `FixedAnchor` (state, pressure projection), `FixedRows`/`FixedRowsB`, `FixedSolver` (outer loop), `lanczos_tr_step_H`, `use_models` (rebuild rules), lean metric builds |
| `nest/meanforce.py` | `Weights` (AL), `Solver.inner` (trust region, acceptance, radius), `_geodesic`, `_accel_dir`, `_subspace`, `_subspace_M` |
| `nest/cylindrical_bs.py`, `nest/annulus_freespace.py` | annulus Biot–Savart |
| `nest/cylindrical_grid.py` | annulus grid, `locate_plasma_planes` |
| `nest/boundary_potential.py` | $\mathbf J = \nabla\times\mathbf T$ representation |
| `tests/valley/smoke_valley.sh` | local CPU check of the valley options (exactness, default path unchanged) |
