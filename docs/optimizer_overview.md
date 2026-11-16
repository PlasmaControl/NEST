# How the NEST fixed-boundary optimizer works

*A self-contained explanation for a reader who knows vector calculus, linear algebra and basic MHD,
but has not seen this code or its history. State of 2026-11-16, branch `ripple-landscape`. Code
locations are collected in §11.*

---

## Contents

0. [Background: what is being computed, and why](#0-background-what-is-being-computed-and-why)
1. [Notation](#1-notation)
2. [The problem](#2-the-problem)
3. [The structure that everything relies on](#3-the-structure-that-everything-relies-on)
4. [Outer loop: the augmented Lagrangian](#4-outer-loop-the-augmented-lagrangian)
5. [Inner loop: one trust-region step](#5-inner-loop-one-trust-region-step)
6. [The metric (preconditioner) and when it is rebuilt](#6-the-metric-preconditioner-and-when-it-is-rebuilt)
7. [Valley navigation: geodesic acceleration and the exact subspace search](#7-valley-navigation-geodesic-acceleration-and-the-exact-subspace-search)
8. [The changes that mattered most for speed](#8-the-changes-that-mattered-most-for-speed)
9. [Recommended flags](#9-recommended-flags)
10. [Appendix: background methods, with references](#10-appendix-background-methods-with-references)
11. [Code map](#11-code-map)

---

## 0. Background: what is being computed, and why

**DESC** (Dudt & Kolemen, 2020) is a spectral code that computes three-dimensional MHD equilibria,
$\mathbf J\times\mathbf B = \nabla p$. It assumes that the magnetic field lines lie on nested
toroidal *flux surfaces*. Each surface is labelled by a radius-like coordinate $\rho\in[0,1]$:
$\rho = 0$ is the magnetic axis and $\rho = 1$ is the last closed flux surface (**LCFS**), the
plasma boundary. Angles $\theta$ (poloidal) and $\zeta$ (toroidal; in DESC $\zeta$ is the geometric
toroidal angle $\phi$) complete the coordinates.

The **safety factor** $q(\rho)$ is the number of toroidal turns a field line makes per poloidal
turn on the surface ρ. On a surface with rational $q = m/n$ the field lines close on themselves
after $m$ toroidal turns, and a perturbation with poloidal and toroidal mode numbers $(m,n)$ is
called **resonant** there: it is constant along those field lines, so its effect accumulates.

Real equilibria need not have nested surfaces. A small resonant perturbation of the boundary can
make field lines on a rational surface reconnect into **magnetic islands**. Ideal MHD with nested
surfaces cannot do that. With nested surfaces $\mathbf B\cdot\nabla\rho = 0$ everywhere, so the
resonant $(m,n)$ part of the radial field must be exactly cancelled at the $q = m/n$ surface. The
plasma does this with a thin *shielding* current sheet at that surface (in the limit, a surface
current; Hahm & Kulsrud, 1985). DESC represents that sheet only to its resolution. **NEST** (this code) asks what the best nearby
equilibrium looks like *without* assuming nested surfaces. Its unknown is the current density
itself, from which the field follows by Biot–Savart; nothing forces the field lines onto surfaces.

The current application (the "ripple study") is a tokamak-like equilibrium with a small resonant
(2,1) boundary ripple. Its central question is whether NEST keeps DESC's shielding or opens the
q = 2 island, and how that depends on resolution. The test case, from the DESC file:

| quantity | value |
|---|---|
| major radius $R_0$, minor radius $a$ | 3.509 m, 1.198 m |
| field on axis | 0.214 T |
| volume-mean β | 0.0214 |
| net toroidal current $\lvert I_\text{tor}\rvert$ | 0.169 MA |
| $\lvert q\rvert$ on axis, at the edge | 1.37, 4.22; $\lvert q\rvert = 2$ at $\rho\approx0.63$ |
| field periods $N_\text{FP}$ | 1 (a tokamak plus a non-axisymmetric ripple) |
| boundary ripple | resonant (2,1), amplitude $\xi/a = 10^{-3}$ |
| DESC resolution, residual | L/M/N = 32/32/4, $M_\text{DESC} = 5.0\times10^{-7}$ (§2.5) |

(DESC's sign convention gives negative $q$ and $I_\text{tor}$ here; magnitudes are quoted.)

The problem is posed as **"change DESC's current as little as possible, subject to a force-balance
residual at most α times DESC's own"**, for three reasons:
1. Minimizing the force residual alone is poorly posed. It is one scalar of about $3\times10^4$
   unknowns, and the runs of §7 show long, nearly flat valleys in which very different currents
   give almost the same residual. The minimizer would then be set by the discretization.
2. Staying close to DESC's current regularizes the problem and makes the answer interpretable: we
   see which changes the force bound *forces*. The anchor is $\mathbf J_\text{fit}$, DESC's current
   expressed in NEST's basis (§2.2), not DESC's current itself: only the former can be reached
   exactly, so the misfit can be zero.
3. Sweeping α traces the trade-off between "change in current" and "residual force". When α is
   below what any current at this resolution can reach, the run finds the lowest reachable residual,
   which we call the **floor** (§4.4).

**Resolution** is written L/M/N, the maximum radial, poloidal and toroidal mode numbers of the
current's spectral basis. For example "64/64/6" means L = M = 64, N = 6. The resolutions run so far are 32/32/6, 40/40/6
and 64/64/6. (These L, M, N are resolutions. In formulas they appear only in mode counts and grid
sizes; elsewhere $L$ is the metric's Cholesky factor and $M$ the force measure, §1.)

The field is computed on a separate grid around the plasma. Its node spacing is written **a/frac**
(e.g. a/200), with a the minor radius (1.198 m here).

All forces are reported relative to DESC's own residual, $M/M_\text{DESC}$, or relative to the bound,
$M/\varepsilon$ (§2.5).

---

## 1. Notation

The code uses some letters for several things. This note uses one symbol per object; where the
code name differs, it is given in the last column.

| symbol | meaning | first defined | code / log name |
|---|---|---|---|
| $\rho,\theta,\zeta$ | DESC flux coordinates (radius-like, poloidal, toroidal) | §0 | |
| L/M/N, $N_\text{FP}$ | basis resolution (max radial, poloidal, toroidal mode numbers); field periods | §0 | `--LMN`, `NFP` |
| $\sqrt g$, $J^\rho, J^\theta, J^\zeta$ | coordinate Jacobian $1/(\nabla\rho\cdot\nabla\theta\times\nabla\zeta)$; contravariant components $J^\rho = \mathbf J\cdot\nabla\rho$ etc. | §2.2 | |
| $T_\theta, T_\zeta$ | covariant components $\mathbf T\cdot\partial\mathbf x/\partial\theta$, $\mathbf T\cdot\partial\mathbf x/\partial\zeta$ of the current potential | §2.2 | |
| $e(\rho)$, $\mathsf a$, $\mathsf b$, $\chi$, $h_{\lvert m\rvert}$ | edge factor $(1-\rho^2)^{s_e}$; the two Fourier–Zernike series; surface potential; its radial functions | §2.2 | `f`, `a`, `b`, `chi` |
| $K_c$ | number of modes in one cosine-type Fourier–Zernike series | §2.2 | |
| $u$, $n_u$ | current-potential coefficients (the unknowns), their number | §2.2 | `u`, `n_u` |
| $u_\text{fit}$, $\mathbf J_\text{fit}$ | the fit of DESC's current in NEST's basis, $\mathbf J(u_\text{fit})$ | §2.2 | `u0`, `Jfit` |
| $\mathbf J$, $\mathbf B$, $p$ | current density, magnetic field, pressure | §2 | |
| $I_\text{tor}$ | net toroidal current (fixed) | §2.2 | `I` |
| $s_e$ | edge exponent of the current potential (= 2) | §2.2 | `edge_power` |
| $\mathbf B_\text{ann}[\mathbf J]$, $\mathbf B_\text{ext}$ | plasma field from Biot–Savart; fixed external field | §2.3 | `B_plasma`, `B_ext` |
| $i$, $n_\text{pts}$, $w_i$ | collocation point index, number of points, volume weights ($\sum w_i = 1$) | §2.4 | `sw` $=\sqrt w$ |
| $\boldsymbol\ell$ | weighted Lorentz force, $\sqrt{w_i}\,(\mathbf J\times\mathbf B)_i$ stacked | §2.4 | `a` |
| $\mathsf P$, $\mathsf G$, $\Pi$ | pressure-gradient matrix, Gram $\mathsf P^T\mathsf P$, projector $\mathsf I - \mathsf P\mathsf G^{-1}\mathsf P^T$ | §2.4 | `P`, `G`, `_project` |
| $\mathbf c_p$, $n_p$ | pressure coefficients (eliminated), their number | §2.4 | `c`, `n_p` |
| $\mathbf f$, $\mathbf F_i$ | projected weighted force, $\mathbf f = \Pi\boldsymbol\ell$; force at point i, $\mathbf F_i = \mathbf f_i/\sqrt{w_i}$ | §2.4 | `f`, `F` |
| $\mathbf z_i$ | $w_i\mathbf F_i$ (weighted force, the quantity the rows use) | §5.1 | `z` |
| $N_B$ | force normalization $\langle\lvert\nabla(B^2/2\mu_0)\rvert\rangle$ (DESC) | §2.5 | `norm` |
| $M$, $M_\text{DESC}$, $\alpha$, $\varepsilon$ | mean force / $N_B$; DESC's value; bound fraction; bound $\alpha M_\text{DESC}$ | §2.5 | `M`, `M_desc`, `alphas`, `eps` |
| $\mathbf g$, $\phi$ | current misfit vector; $\phi=\tfrac12\lVert\mathbf g\rVert^2$ | §2.5 | `g`, `phi` |
| $c_F$ | constraint $M/\varepsilon - 1$ | §4.1 | log `M/eps` $= 1+c_F$ |
| $\nu$, $\mu$ | AL multiplier, AL penalty | §4.1 | `nu`, **`rho`** |
| $\psi$, $\psi'$ | AL penalty function, its slope $\max(0,\nu+\mu c_F)$ | §4.1 | `psi`, `psi'`, `d1` |
| $\mathcal L_A$ | the merit (augmented Lagrangian) | §4.1 | log `f` |
| $\mathbf r$, $\mathsf K$ | residual rows of the step model, their Jacobian $\partial\mathbf r/\partial u$ | §5.1 | `rows.r`, `rows.J` |
| $\mathsf D$, $\mathsf K_z$ | force Jacobian $\partial\mathbf F/\partial u$; weighted-force Jacobian $\partial\mathbf z/\partial u$ (rows $w_i\mathsf D_i$) | §5.1 | |
| $\mathsf P_\beta$ | β projection $\mathsf I-(1-\sqrt\beta)\hat{\mathbf z}\hat{\mathbf z}^T$ | §5.2 | |
| pred, ratio, share | predicted decrease $m(0)-m(\mathbf y)$; actual/predicted; relative decrease of the unconstrained Krylov step | §5.5 | |
| $c_F^+$ | $\max(0,c_F)$ | §4.2 | |
| $s_i$, $\bar z_i$, $\tau$ | row weight, floored norm $\max(\lvert\mathbf z_i\rvert, w_i\tau)$, floor | §5.1 | `sF`, `t`, `tau` |
| $\mathbf k_F$ | penalty-curvature row | §5.1 | `aF` |
| $\beta$ | β-model curvature along $\hat{\mathbf z}_i$ (= 1/2) | §5.2 | `--beta` |
| $m(\cdot)$ | the quadratic step model | §5.1 | |
| $L$, $\mathbf y$, $\Delta$ | metric Cholesky factor, step in metric coordinates $\mathbf y = L^T\mathbf d$, trust radius | §5.3 | `met.L`, `y`, `Delta` |
| $\boldsymbol\gamma$, $\mathsf H_y$ | model gradient and Hessian in metric coordinates | §5.3 | log `|g|` $=\lVert\boldsymbol\gamma\rVert$ |
| $\eta_k$, $\omega_k$, $\mathsf T_k$ | Lanczos diagonal and off-diagonal, tridiagonal matrix | §5.4 | `alpha`, `beta` |
| $\sigma_f$ | measured evaluation noise of the merit | §5.5 | `sigma_f` |
| $\mathbf q(\mathbf d,\mathbf d)$, $\mathbf q(\mathbf a,\mathbf b)$ | exact second-order part of the force along a step; its symmetric bilinear form | §7.1 | |
| $\mathbf a$ | geodesic-acceleration correction | §7.2 | |
| $\mathbf v_j$, $\mathbf t$, $\mathbf t^\star$, $\mathbf y^\star$ | subspace directions and coordinates; the chosen point; its step in metric coordinates | §7.3 | `E`, `c` |
| $\sigma$ | rounding level of one force evaluation (distinct from $\sigma_f$) | §7.3 | |

*Letters that keep two meanings, always distinguishable by context:* $q$ is the safety factor, but
$\mathbf q_k$ are Lanczos vectors (§5.4) and $\mathbf q(\cdot,\cdot)$ the second-order force (§7.1);
$\phi$ is the misfit objective, while the toroidal angle is always written ζ (in DESC ζ is the
geometric angle); $\mathsf Q$ is the point-weight matrix of §6.1, $Q$ the pressure series of §2.4
and $\mathsf Q_k$ the Lanczos basis; $k$ indexes outer iterations (§4), Lanczos iterations (§5.4)
and pressure basis functions (§2.4); $\mathbf h$ is the small trust-region unknown (§5.4) and $h$
the field-grid spacing (§2.6). The cylindrical radius is written $R$.

A **state evaluation** means computing, for one $u$ (each item is defined in §2):
1. $\mathbf J(u)$ at the field-grid nodes and at the collocation points;
2. $\mathbf B_\text{ann}$ by Biot–Savart;
3. $\mathbf B$ at the collocation points;
4. the pressure projection;
5. $\mathbf F$, $M$, $\mathbf g$ and $\mathcal L_A$.

At 64/64/6 a/200 on an A100-80GB it takes about 0.11 s (measured: 0.114 s), comparable to one $\mathsf K\mathbf v$ product.

---

## 2. The problem

$$
\min_u\ \phi(u) = \tfrac12\lVert\mathbf g(u)\rVert^2
\qquad\text{subject to}\qquad M(u)\le\varepsilon = \alpha\,M_\text{DESC}.
\tag{2.1}
$$

### 2.1 Fixed boundary

The geometry is DESC's and is **fixed**: the map $(\rho,\theta,\zeta)\mapsto$ space, and therefore
the LCFS shape. The current lives inside the LCFS. The normal field $\mathbf B\cdot\mathbf n$ on
the LCFS is **not** imposed and changes as $\mathbf J$ changes. "Fixed boundary" here means only
that the plasma region does not move.

### 2.2 Unknowns: the current potential

Representing $\mathbf J$ by coefficients directly would need $\nabla\cdot\mathbf J = 0$ as a
constraint. Writing the current as a curl, $\mathbf J = \nabla\times\mathbf T$, builds it in. Here
$\mathbf T$ is a vector potential for the current, described by its covariant components
$T_\rho, T_\theta, T_\zeta$ ($T_\theta = \mathbf T\cdot\partial\mathbf x/\partial\theta$, etc.).
Adding a gradient $\nabla\Lambda$ to $\mathbf T$ does not change $\mathbf J$; choosing $\Lambda$
with $\partial_\rho\Lambda = -T_\rho$ makes $T_\rho = 0$ (the *radial gauge*). In this gauge

$$
T_\theta = \frac{I_\text{tor}}{2\pi}\,h_0(\rho) + \partial_\theta\chi + \rho^2 e(\rho)\,\mathsf a(\rho,\theta,\zeta),\qquad
T_\zeta = \partial_\zeta\chi + e(\rho)\,\mathsf b(\rho,\theta,\zeta),
\qquad e(\rho) = (1-\rho^2)^{s_e}.
$$

The ingredients:
- **$\mathsf a$ and $\mathsf b$** are *Fourier–Zernike series* (DESC's basis), sums of
  $c_{lmn}\,\mathcal Z_l^m(\rho,\theta)\,\mathcal F_n(\zeta)$ with
  $\mathcal Z_l^m = R_l^{|m|}(\rho)\,\mathcal F_m(\theta)$. Here $R_l^{|m|}$ is the radial Zernike
  polynomial (degree $l$, behaving as $\rho^{|m|}$ near the axis), and
  $\mathcal F_m(\theta) = \cos m\theta$ for $m\ge0$, $\sin|m|\theta$ for $m<0$; likewise
  $\mathcal F_n(\zeta) = \cos nN_\text{FP}\zeta$ or $\sin|n|N_\text{FP}\zeta$. Indices run over
  $|m|\le l\le L$ with $l - |m|$ even, $|m|\le M$ and $|n|\le N$.
- **$\chi$** is a "surface potential" $\sum_{m\ne0,n}\chi_{mn}\,h_{|m|}(\rho)\,\mathcal F_m(\theta)\mathcal F_n(\zeta)$,
  with radial functions (for $s_e = 2$, with $q_m = |m|+2$)
  $h_{|m|}(\rho) = \tfrac12\rho^{q_m}\big[(q_m+2) - q_m\rho^2\big]$.
  Check: $h_{|m|}(1) = 1$ and $h_{|m|}'(1) = \tfrac12[q_m(q_m+2) - q_m(q_m+2)] = 0$.
- **$h_0(\rho) = \rho^2(2-\rho^2)$**, the same formula with $q = 2$.
- *Why the powers of ρ.* Near the axis a smooth vector field has $T_\zeta$ mode $m$ behaving like
  $\rho^{|m|}$ (true of any Zernike series) and $T_\theta$ mode $m$ like $\rho^{|m|+2}$, because
  $\partial\mathbf x/\partial\theta$ itself vanishes like ρ. The factor $\rho^2$ on $\mathsf a$ and the
  $\rho^{|m|+2}$ in $h$ supply exactly that, so $\mathbf J$ is regular on the axis for every $u$.

The coefficient vector $u$ holds the $\chi_{mn}$ and the coefficients of $\mathsf a$ and $\mathsf b$. **Any** $u$ is
admissible, and three properties hold for every u:

- **$\nabla\cdot\mathbf J = 0$**, because $\mathbf J$ is a curl.
- **$\mathbf J = 0$ on the LCFS.** The contravariant components are
  $\sqrt g\,J^\rho = e\,(\partial_\theta \mathsf b - \rho^2\partial_\zeta \mathsf a)$ (the $\chi$ terms cancel),
  $\sqrt g\,J^\theta = -\partial_\rho T_\zeta$ and $\sqrt g\,J^\zeta = \partial_\rho T_\theta$.
  - $e = O((1-\rho)^{s_e})$, so its $\rho$-derivative is $O((1-\rho)^{s_e-1})$.
  - $h_{|m|}'(1) = 0$ for $s_e = 2$, and $h_0'(1) = 4\rho(1-\rho^2)|_{\rho=1} = 0$.
  - Every component is therefore $O((1-\rho)^{s_e-1})$, which is zero at $\rho = 1$ for $s_e\ge 2$.
- **The net toroidal current is fixed.** On a cross-section $\zeta = $ const the unit normal is
  $\nabla\zeta/\lvert\nabla\zeta\rvert$ and the area element is $\sqrt g\,\lvert\nabla\zeta\rvert\,d\rho\,d\theta$, so
  $\mathbf J\cdot d\mathbf A = \sqrt g\,J^\zeta\,d\rho\,d\theta$ and the current through it is
  $\iint \sqrt g J^\zeta\, d\rho\, d\theta = \int_0^{2\pi}[T_\theta(1) - T_\theta(0)]\,d\theta$.
  - $T_\theta(0) = 0$, since every term carries $\rho^2$ or $h(\rho)\propto\rho^{|m|+2}$.
  - At $\rho = 1$, $e = 0$ and $\partial_\theta\chi$ integrates to zero over $\theta$.
  - So the integral is $2\pi\cdot I_\text{tor}h_0(1)/2\pi = I_\text{tor}$ for every $u$.

**$\mathbf J$ is affine in $u$:** a part linear in u, plus the fixed $I_\text{tor}h_0$ term.

**$u_\text{fit}$** is the least-squares fit of this representation to DESC's own potential. That
potential is obtained by integrating DESC's current from the axis in the radial gauge,
$T_\theta^\text{DESC} = \int_0^\rho\sqrt g\,J^\zeta\,d\rho'$ and
$T_\zeta^\text{DESC} = -\int_0^\rho\sqrt g\,J^\theta\,d\rho'$, on a tensor grid
(`BoundaryPotential.fit`). The fit minimizes $\sum_p w_p\lvert\mathbf T(u)-\mathbf T^\text{DESC}\rvert^2$
with weights uniform in θ, ζ and $\rho\,d\rho$; a function $\alpha(\zeta)$ that may be added to
$T_\zeta$ without changing $\mathbf J$ is left free and then dropped. We define
$\mathbf J_\text{fit} = \mathbf J(u_\text{fit})$, so the misfit $\mathbf g(u_\text{fit}) = 0$ exactly.
$\mathbf J_\text{fit}$ differs from DESC's current because the bases differ (the size of that
difference is not quantified in this note).

**Counting unknowns.** *Stellarator symmetry* means invariance under $(\rho,\theta,\zeta)\to(\rho,-\theta,-\zeta)$
(with $R$ even and $Z$ odd). DESC's ripple equilibrium has it, and then $T_\theta$, $T_\zeta$ are
even, so $\mathsf a$, $\mathsf b$ keep only the *cosine-type* products
($\mathcal F_m\mathcal F_n$ = cos·cos or sin·sin, i.e. $m$, $n$ of equal sign), and $\chi$ only the
*sine-type* ones ($m$, $n$ of opposite sign).

- *Fourier–Zernike modes.* For L = M there are $(M+1)(M+2)/2$ pairs $(l,m)$, so
  $(2N+1)(M+1)(M+2)/2$ modes in all. Symmetry splits them evenly into cosine and sine type,
  except the $m = n = 0$ modes ($l = 0, 2, \dots$: $\lfloor M/2\rfloor + 1$ of them), which are all
  cosine type. Hence $2K_c = (2N+1)\tfrac{(M+1)(M+2)}{2} + \lfloor M/2\rfloor + 1$.
- *χ modes.* $m = \pm1,\dots,\pm M$ and $n = -N,\dots,N$, of which the sine-type half survives:
  $M\cdot N$ with $m>0$, $n<0$ and $M(N+1)$ with $m<0$, $n\ge0$, i.e. $M(2N+1)$.

$$
n_u = M(2N+1) + 2K_c .
$$

*Example, 32/32/6:* $32\cdot13 + 13\cdot561 + 17 = 416 + 7293 + 17 = 7{,}726$. The formula gives 7,726 at 32/32/6, 11,734 at 40/40/6 and 28,750 at 64/64/6, which are the counts in the
run logs. (Without symmetry, $n_u$ roughly doubles.)

### 2.3 The magnetic field

The field has two parts:

$$
\mathbf B = \mathbf B_\text{ext} + \mathbf B_\text{ann}[\mathbf J].
$$

- **$\mathbf B_\text{ann}[\mathbf J]$, the plasma's own field.** It comes from the free-space
  Biot–Savart law, evaluated on a grid around the plasma (§2.6). It is linear in $\mathbf J$.
- **$\mathbf B_\text{ext}$, the external field**, a fixed vector at every collocation point. The driver
  offers three choices (`--bext`):
  - `coils`: the field of a coil set;
  - `desc-J`: DESC's field minus the plasma field of DESC's exact current;
  - **`desc-fit` (used throughout):**

$$
\mathbf B_\text{ext} = \mathbf B_\text{DESC} - \mathbf B_\text{ann}[\mathbf J_\text{fit}],
$$

i.e. DESC's field minus the plasma field of $\mathbf J_\text{fit}$ computed on *our own* grid. With
this choice the start $u_\text{fit}$ reproduces $\mathbf B_\text{DESC}$ exactly, and any field-grid
error enters only through $\mathbf B_\text{ann}[\mathbf J - \mathbf J_\text{fit}]$.

Two consequences:
1. **Grid convergence can only be tested at optimized states.** A grid-convergence test at
   $u_\text{fit}$ is meaningless, because the grid error cancels there by construction. We test at
   optimized states, where $\mathbf J\ne\mathbf J_\text{fit}$.
2. **$\mathbf B_\text{ext}$ is only approximately curl-free.** Since $\nabla\times\mathbf B_\text{DESC} = \mu_0\mathbf J_\text{DESC}$
   and $\nabla\times\mathbf B_\text{ann}[\mathbf J_\text{fit}]\approx\mu_0\mathbf J_\text{fit}$, we get
   $\nabla\times\mathbf B_\text{ext}\approx\mu_0(\mathbf J_\text{DESC}-\mathbf J_\text{fit})$ plus the field-grid error.
   The first part is the basis-fit residual of §2.2 and does not shrink with the grid; `--bext desc-J`
   removes it. Both parts are acceptable because $\mathbf B_\text{ext}$ is held fixed.

### 2.4 The pressure is eliminated, not solved for

**Collocation points.** Forces are evaluated at the points of a tensor grid in $(\rho,\theta,\zeta)$:
Gauss–Legendre nodes in $\rho$, uniform in $\theta$ and $\zeta$, over half a field period (by
symmetry; for $N_\text{FP} = 1$ that is $\zeta\in[0,\pi]$). With $\text{rd}(x)$ the nearest
integer and $\lfloor\cdot\rfloor$ integer division, the default sizes are
$n_\rho = \lfloor\text{rd}(1.5(L+L_\text{eq}))/2\rfloor + 1$,
$n_\theta = 2\,\text{rd}(1.5(M+M_\text{eq})) + 1$ and $n_\zeta = \text{rd}(1.5(N+N_\text{eq})) + 1$
(`--colloc` = 1.5), where $L_\text{eq}$, $M_\text{eq}$, $N_\text{eq}$ is DESC's own resolution
(32/32/4 here). At 64/64/6: $\text{rd}(144)=144$, so $73\times289\times16 = 337{,}552$ points. The
weights are $w_i = \sqrt g_i\,\Delta_i/V$, with $\Delta_i$ the quadrature weight in $(\rho,\theta,\zeta)$
and $V$ the plasma volume, so $\sum_i w_i = 1$ and $\sum_i w_i(\cdot)_i\approx\tfrac1V\int(\cdot)\,dV$.

**Pressure basis.** $p = (1-\rho^2)^3 Q$, with $Q$ a Fourier–Zernike series of the same size as one
of $\mathsf a$, $\mathsf b$, so $n_p = K_c$ (13,959 at 64/64/6). Thus $p = 0$ on the LCFS.

**Least-squares pressure.** For given $\mathbf J$ and $\mathbf B$, choose the pressure that best
balances the Lorentz force in the weighted least-squares sense. Let $\mathsf P$ be the
$3n_\text{pts}\times n_p$ matrix whose $k$-th column is $\sqrt{w_i}\,\nabla p_k(\mathbf x_i)$
stacked over points. Then

$$
\mathbf c_p^\star = \arg\min_{\mathbf c}\lVert\boldsymbol\ell - \mathsf P\mathbf c\rVert^2 = \mathsf G^{-1}\mathsf P^T\boldsymbol\ell,
\qquad \mathsf G = \mathsf P^T\mathsf P,
\qquad \mathbf f = \boldsymbol\ell - \mathsf P\mathbf c_p^\star = \Pi\,\boldsymbol\ell,
\qquad \Pi = \mathsf I - \mathsf P\mathsf G^{-1}\mathsf P^T .
\tag{2.2}
$$

$\Pi$ is the orthogonal projector onto the complement of the pressure-gradient space. The force at
point $i$ is $\mathbf F_i = \mathbf f_i/\sqrt{w_i}$.

*Why "weak Poisson".* The normal equations $\mathsf P^T(\boldsymbol\ell - \mathsf P\mathbf c) = 0$ read,
for every basis function $p_k$,

$$
\sum_i w_i\,\nabla p_k\cdot(\mathbf J\times\mathbf B - \nabla p)\big|_{\mathbf x_i} = 0
\ \approx\ \frac1V\int \nabla p_k\cdot(\mathbf J\times\mathbf B - \nabla p)\,dV .
$$

With $\mathbf v = \mathbf J\times\mathbf B - \nabla p$, integration by parts reads
$\int\nabla p_k\cdot\mathbf v\,dV = \oint p_k\,\mathbf v\cdot\mathbf n\,dS - \int p_k\,\nabla\cdot\mathbf v\,dV$.
The surface term vanishes because $p_k = 0$ on the LCFS, leaving
$\int p_k\,[\nabla^2 p - \nabla\cdot(\mathbf J\times\mathbf B)]\,dV = 0$ for every $k$. A *Galerkin*
method makes the residual of an equation orthogonal to the same functions used to expand the
unknown. So this is the Galerkin (weak) form of $\nabla^2 p = \nabla\cdot(\mathbf J\times\mathbf B)$.

*How it is computed.* $\mathsf G$ depends only on the fixed geometry. It is formed once (summed over
radial slices of the grid) and Cholesky factored once. Applying $\Pi$ needs one evaluation of
$\mathsf P^T$ (a transposed $\nabla p$ evaluation), two triangular solves and one evaluation of
$\mathsf P$. Nothing of size $3n_\text{pts}\times n_p$ is stored. Because $\mathbf f$ is a small
difference of large terms (§3), the state evaluation applies the projection twice: $\Pi$ applied to
$\mathbf f$ again removes what the first solve left over, a step of iterative refinement. The
linearization used by the step model applies it once. Since $\Pi^2 = \Pi$ in exact arithmetic,
the two agree to rounding level.

*Why eliminate $p$ at all?* This was a design decision, argued rather than measured here. Treating
$\mathbf c_p$ as extra unknowns adds columns $\nabla p_k$ whose size grows with the mode number
(roughly as $l/a$). Their column norms then spread like $L$, and the condition number of the
normal equations like $L^2$ or worse, which would slow Krylov solves without a much better
preconditioner. Eliminating $p$ keeps the unknowns to $u$ alone, with $\mathsf G$ factored once.

*Why the mean of |F|?* It is the residual DESC reports for its own equilibria (volume-mean $|\mathbf F|$
over the magnetic-pressure gradient), so $M/M_\text{DESC}$ compares like with like in form, and it
weights small residuals more evenly than a sum of squares. The price is that $M$ has kinks where
some $\mathbf F_i = 0$, which is why §5.1–5.2 model it with reweighted squares, and why §7.3 uses a
derivative-free search.

*A mismatch to be aware of.* $p^\star$ minimizes the **sum of squares** of the force, an L2 norm,
while the constraint (§2.5) uses the **mean of |F|**, an L1-type norm. So for given $u$ the pressure
is not the one that minimizes $M$. This is accepted for two reasons:
1. The L2 projection is linear, which keeps $\mathbf F$ exactly quadratic in $u$ (§3), and it is cheap.
2. An L1-optimal pressure would be a non-smooth function of $u$.

*NEST's pressure is not a physical constraint.* It is whatever best balances $\mathbf J\times\mathbf B$,
not tied to flux surfaces or to conserved mass.

**Worked example (projection).** Take two points with $w_1 = w_2$ (absorbed into the vectors), one
pressure function with $\mathsf P = (1, 1)^T$, and $\boldsymbol\ell = (3, 1)^T$. Then
$\mathsf G = 2$, $\mathsf P^T\boldsymbol\ell = 4$, $c_p^\star = 2$ and
$\mathbf f = (3,1) - 2(1,1) = (1,-1)$. Check: $\mathsf P^T\mathbf f = 0$, so the remaining force has
no component the pressure could remove. ✓

### 2.5 The objective and the constraint

- **Misfit:** $\mathbf g = \sqrt{w}\,(\mathbf J - \mathbf J_\text{fit})/J_\text{rms}$, stacked over
  points, with $J_\text{rms} = (\sum_i w_i\lvert\mathbf J_\text{fit}(\mathbf x_i)\rvert^2)^{1/2}$.
  So $\lVert\mathbf g\rVert$ is the relative rms change of the current, printed as `|J - J_fit|/rms`.
  $\phi = \tfrac12\lVert\mathbf g\rVert^2$. Since $\mathbf J$ is affine in $u$, $\mathbf g$ is affine.
- **Normalization:** $N_B = \langle\lvert\nabla(B^2/2\mu_0)\rvert\rangle_\text{vol}$ from DESC, the
  volume-mean magnetic-pressure gradient (DESC's force normalization). In this low-β tokamak most of
  $\nabla(B^2/2\mu_0)$ is balanced by field-line curvature, so the two terms that cancel in
  $\mathbf F$, $\mathbf J\times\mathbf B$ and $\nabla p$, are smaller than $N_B$: about
  $\beta R_0/(2a)\approx0.03\,N_B$ (an order-of-magnitude estimate from ⟨β⟩ = 0.0214, not a measurement).
- **Force measure:** $M(u) = \sum_i w_i\lvert\mathbf F_i\rvert/N_B$, dimensionless.
- **Reference:** $M_\text{DESC}$ is DESC's own volume-mean force residual divided by $N_B$, computed by
  DESC on its quadrature grid with DESC's pressure; $4.9995\times10^{-7}$ for the ripple equilibrium.
  It is **not** $M(u_\text{fit})$. The start $u_\text{fit}$ has a much larger $M$ in NEST's
  discretization ($M(u_\text{fit})/M_\text{DESC} = 9.68$ at 64/64/6), so $\alpha = 1$ is not
  satisfied at the start. The factor 9.68 has three sources: the basis change
  ($\mathbf J_\text{DESC}\to\mathbf J_\text{fit}$), the different pressure (least-squares in NEST,
  DESC's own profile in DESC) and the different quadrature. So $M_\text{DESC}$ is a convenient
  *scale* for the bound, not a like-for-like target: "$M = 0.016\,M_\text{DESC}$" means NEST's
  measure is 1.6% of DESC's reported residual.
- **Bound:** $M\le\varepsilon = \alpha M_\text{DESC}$ (`--alphas`). Logs print $M/\varepsilon$; for
  example $M/\varepsilon = 16.66$ at $\alpha = 0.001$ means $M = 0.0167\,M_\text{DESC}$.

**Alternative bound (`--force-norm p`, `--p`, `--p-band`).** A $p_\text{exp}$-norm over an interior
set of points, typically a *band* $\rho\in[\rho_\text{lo},\rho_\text{hi}]$ around a surface of
interest: $M_p = F_p/N_B$ with $F_p = (\sum_\text{int}w\lvert\mathbf F\rvert^{p_\text{exp}}/W_\text{int})^{1/p_\text{exp}}$
and $W_\text{int} = \sum_\text{int}w$. It approaches the maximum of $|F|$ on the set as
$p_\text{exp}\to\infty$, and is used for max-type experiments. **Sections 5.1–5.2 and 7.3 are
written for the mean bound** (read them first). Under the $p_\text{exp}$-norm:
- the row weights carry an extra factor $\kappa_i = (\lvert\mathbf F_i\rvert/F_p)^{p_\text{exp}-1}/W_\text{int}$,
  because $\partial M_p/\partial\lvert\mathbf F_i\rvert = w_i\kappa_i/N_B$ replaces $w_i/N_B$;
- the β model uses $\beta = p_\text{exp}-1$, the curvature of $t\mapsto t^{p_\text{exp}}$ at $t=|z|$
  relative to its slope there ($t^p$ has second/first derivative ratio $(p-1)/t$);
- the subspace search is disabled;
- the exact-Hessian model is refused.

### 2.6 The field grid (Biot–Savart)

$\mathbf B_\text{ann}$ is computed from the Coulomb-gauge vector potential:
$\nabla^2\mathbf A = -\mu_0\mathbf J$, with $\mathbf A\to0$ at infinity (free space), and
$\mathbf B = \nabla\times\mathbf A$. The domain is an annulus $R_\text{in}\le R\le R_\text{out}$,
$|Z|\le Z_0$ ($R$ the cylindrical radius), whose walls lie `--pad` metres (default 0.02 m) beyond
the plasma's $(R,Z)$ extent. It has $N_\phi$ planes per field period (`--N-phi`, 25 here) and
Chebyshev-type nodes in $R$ and $Z$, placed so that the node spacing over the plasma is at most
$h = a/\text{frac}$ (Chebyshev nodes cluster towards the walls, where they are denser).

The steps:
1. **Locate** every node in flux coordinates by Newton inversion of DESC's map
   (`locate_plasma_planes`). Nodes outside the LCFS get $\mathbf J = 0$.
2. **Evaluate** $\mathbf J$ at the inside nodes.
3. **Fourier transform** in the toroidal angle. Each toroidal mode then gives a 2-D problem in $(R,Z)$.
4. **Solve each mode in free space by James' method** (James, 1977):
   1. solve with $\mathbf A = 0$ on the annulus boundary (eigendecomposition in $R$ and $Z$);
   2. outside the annulus the true potential is source-free, so the jump between the boundary
      condition used ($\mathbf A = 0$) and the true one is carried by an equivalent surface source,
      given by the normal derivative of that solution on the boundary;
   3. the free-space potential of that surface source on the boundary itself is computed with a
      precomputed *single-layer matrix* (a boundary-integral operator, stored in float32);
   4. re-solve inside with that boundary data; the result is the free-space solution.
5. **Curl and interpolate:** take the curl, and interpolate $\mathbf B$ to the collocation points.

At 64/64/6 and a/160 the grid has about 7.1 M nodes. The rule of thumb $h L/a\lesssim 0.4$ (with
$h = a/\text{frac}$; here L = M, so either can be used) comes from grid-convergence tests in which
refining the grid changed the end states' $M$ by about 3% or less. The default `--frac 40` is far
too coarse at high resolution (64/40 = 1.6); it must be raised with L.

---

## 3. The structure that everything relies on

1. **$\mathbf F$ is exactly quadratic in $u$.**
   - $\mathbf J$ is affine in $u$.
   - $\mathbf B$ is affine: $\mathbf B_\text{ext}$ is fixed and $\mathbf B_\text{ann}$ is linear.
   - Every interpolation is a fixed linear map, and so is the projection $\Pi$ (twice).
   - Hence $\boldsymbol\ell = \sqrt w\,(\mathbf J\times\mathbf B)$ and $\mathbf f = \Pi\boldsymbol\ell$
     are exactly quadratic in $u$, and so is every $\mathbf F_i$. The misfit $\mathbf g$ is affine.

   "Exactly" holds in exact arithmetic. In floating point, rounding adds a small non-quadratic part
   (item 3).
2. **Products are cheap; matrices are not.** Let $\mathsf K$ be the Jacobian of the model rows
   (§5.1), with about $6n_\text{pts}+1\approx2\times10^6$ rows and $n_u\approx3\times10^4$ columns at
   64/64/6.
   - Stored densely, $\mathsf K$ would take about $2\times10^6\times2.9\times10^4\times8$ B ≈ 470 GB, far
     more than a GPU holds.
   - A product $\mathsf K\mathbf v$ costs 0.094 s and $\mathsf K^T\mathbf x$ 0.112 s at 64/64/6 a/200
     (measured). $\mathsf K\mathbf v$ is a forward-mode derivative of the state evaluation and
     $\mathsf K^T\mathbf x$ a reverse-mode one (vector–Jacobian product, i.e. the adjoint of every
     step, including the Biot–Savart solve), both by automatic differentiation in JAX. Each is
     essentially one Biot–Savart solve plus local work.

   Everything iterative is therefore **matrix-free**.
3. **The force is a small difference of large terms.** $\mathbf J\times\mathbf B$ and $\nabla p$ are
   each about $0.03\,N_B$ (§2.5), but near the optimum their difference is about $10^{-8}N_B$, a
   cancellation of 6–7 digits. Each carries a rounding error of order $10^{-16}$ of its size per operation, and the errors accumulate through the
   Biot–Savart solve, the interpolations and the projection. So $M$ is known only to a relative
   accuracy of order $10^{-8}$. Near the floor the merit is dominated by $\psi\approx\tfrac12\mu c_F^2$
   (§4.4), whose relative error is then about twice that of $M$ (it is quadratic in $M/\varepsilon$).
   Measured: $\sigma_f/\mathcal L_A = 1.2\times10^{-8}$ at 64/64/6. This noise matters for step acceptance (§5.5) and for long extrapolations (§7.3).

**Worked example (the quadratic structure).** Along any line $u = u_0 + t\,\mathbf d$,
$\mathbf F_i(t) = \mathbf F_i^{(0)} + t\,\mathbf F_i^{(1)} + t^2\,\mathbf F_i^{(2)}$ exactly. Three
evaluations, at $t = 0$ and $t = \pm1$, determine it:

$$
\mathbf F^{(1)} = \tfrac12[\mathbf F(1) - \mathbf F(-1)],\qquad
\mathbf F^{(2)} = \tfrac12[\mathbf F(1) + \mathbf F(-1)] - \mathbf F(0).
$$

Check with a scalar: $F(t) = 1 - 3t + t^2$ gives $F(0)=1$, $F(1)=-1$, $F(-1)=5$, so
$F^{(1)} = (-1-5)/2 = -3$ and $F^{(2)} = (-1+5)/2 - 1 = 1$. ✓ With these coefficients, $|F(t)|$,
and hence $M(t)$, is known for every $t$ without further Biot–Savart solves (up to rounding).

---

## 4. Outer loop: the augmented Lagrangian

### 4.1 Why an augmented Lagrangian, and the merit

*Why not a plain penalty?* One could minimize $\phi + \tfrac{\mu}{2}\max(0,c_F)^2$ with
$c_F = M/\varepsilon - 1$. But a plain penalty reaches $c_F\le0$ only as $\mu\to\infty$, and a huge
$\mu$ makes the problem badly conditioned. The **augmented Lagrangian (AL)** adds a multiplier
estimate $\nu\ge0$. At a solution of (2.1) the *KKT conditions* hold:
$\nabla\phi + \nu^*\nabla c_F = 0$, $c_F\le0$, $\nu^*\ge0$, $\nu^* c_F = 0$, for some
multiplier $\nu^*$. With $\nu = \nu^*$ the solution is a minimizer of the AL for every $\mu$ above
a finite threshold, so the AL reaches feasibility at finite $\mu$ by updating $\nu$ towards $\nu^*$
(Nocedal & Wright, ch. 17; Powell 1969, Hestenes 1969, Rockafellar 1973).

*Derivation of the inequality form* (Powell–Hestenes–Rockafellar, PHR). Write $c_F\le0$ as
$c_F + s = 0$ with a slack $s\ge0$, and form the equality AL
$\nu(c_F+s) + \tfrac{\mu}{2}(c_F+s)^2$, a parabola in $s$ with vertex at $c_F+s = -\nu/\mu$.
Minimizing over $s\ge0$ gives $s = \max(0, -c_F - \nu/\mu)$. Two cases:
- $s>0$ (when $c_F < -\nu/\mu$): $c_F + s = -\nu/\mu$, and the value is $-\nu^2/(2\mu)$.
- $s=0$: the value is $\nu c_F + \tfrac{\mu}{2}c_F^2 = \tfrac{\mu}{2}[(c_F+\nu/\mu)^2 - (\nu/\mu)^2]$.

Both cases are

$$
\psi(c_F) = \frac{\mu}{2}\left[\max\!\left(0, c_F + \frac{\nu}{\mu}\right)^2 - \left(\frac{\nu}{\mu}\right)^2\right],
\qquad \mathcal L_A(u) = \phi(u) + \psi(c_F(u)),
\qquad \psi' = \max(0,\ \nu + \mu c_F).
\tag{4.1}
$$

The slope $\psi'$ is the *effective multiplier* (`psi'` in logs). The weight (pull) of the merit on
$M$ is $\partial\psi/\partial M = \psi'/\varepsilon$.
- If $c_F < -\nu/\mu$, the bound is comfortably met: $\psi$ is constant and only $\phi$ is minimized.
- Otherwise the merit pushes $M$ down.

**Worked examples.**
1. $\nu = 0$, $\mu = 1$, $c_F = 0.2$ (M is 20% over the bound): $\psi = \tfrac12(0.2^2 - 0) = 0.02$, $\psi' = 0.2$.
2. $\nu = 0$, $\mu = 1$, $c_F = -0.1$ (10% under): $\psi = 0$, $\psi' = 0$.
3. $\nu = 1$, $\mu = 10$, $c_F = -0.05$: $\max(0, -0.05 + 0.1) = 0.05$,
   $\psi = 5(0.0025 - 0.01) = -0.0375$, $\psi' = 1 - 0.5 = 0.5$.

   A feasible state still "feels" the bound when the multiplier is positive. This is what makes
   the bound active at the solution.

### 4.2 One outer iteration

The outer loop drives the iterate towards the KKT conditions of §4.1. It starts from $\nu = 0$ (or
the value in a `--start` file) and $\mu = $ `--rho0` (default 1). Each outer iteration $k$ does the
following (`FixedSolver.solve_alpha`):

1. **Least-squares multiplier** (first outer iteration of each α; on by default, `--no-mult-ls` off).
   Set $\nu$ to the value that makes the Lagrangian's gradient smallest in the metric norm
   $\lVert x\rVert_{L} = \lVert L^{-1}x\rVert$ (the metric $L$ is defined in §5.3). It is computed
   before this iteration's metric refresh, so it uses the metric current at that moment (the
   $\mathsf H_g$-only metric for the first bound). Setting the derivative of
   $\lVert L^{-1}(\nabla\phi + \nu\nabla c_F)\rVert^2$ with respect to ν to zero, and clipping at 0
   because a KKT multiplier is non-negative:

$$
\nu_\text{LS} = \max\!\left(0,\ -\frac{(L^{-1}\nabla\phi)\cdot(L^{-1}\nabla c_F)}{\lVert L^{-1}\nabla c_F\rVert^2}\right).
$$

   Its cost is two gradients and two triangular solves.

   This is the same idea as DESC's `lsq-auglag` initial multipliers.
2. **Metric refresh** (§6.3). Rebuild the preconditioner unless the keep test passes: both the old
   and new $\psi'$ are positive and within a factor `--outer-rebuild-factor` (2) of each other.
   - After an *outer* rebuild, the trust radius is reset to $\lVert L^{-1}\nabla\mathcal L_A\rVert$,
     the length of the unconstrained model step if the model Hessian were the identity (§5.3).
     (Rebuilds *inside* the inner loop instead rescale the existing radius, §6.3.)
   - If the metric is kept, the radius carries over.

   *Caveat (known, not fixed):* the keep test sees only $\psi'$. The metric's force block scales
   like $\psi'/\varepsilon$ and its penalty block like $\mu$ (§6.1). So a change of $\varepsilon$
   (a new α) or a 10× increase of $\mu$ can leave $\psi'$ within a factor 2 while those blocks are
   off by that factor; the Krylov solve then needs more iterations until the next rebuild.
3. **Inner solve:** trust-region steps on $\mathcal L_A$ at fixed $(\nu,\mu)$, up to `--maxiter`
   steps (§5).
4. **Multiplier update:** $\nu\leftarrow\max(0,\nu+\mu c_F)$, i.e. $\nu\leftarrow\psi'$. With the
   LS multiplier on, $\nu_\text{LS}$ is recomputed at the new point after *every* inner solve (as a
   safeguard; it *replaces* ν only at the first outer iteration, step 1). If $\nu_\text{LS}>0$, the
   update is clipped to $[0.25, 4]\times\nu_\text{LS}$. This stops a first-order update from overshooting when $\mu$ is
   large.
5. **Penalty update.** Let the violation be $v_k = \max(c_F^+, v_\text{comp})$, where the
   complementarity violation is $v_\text{comp} = \max(0, -c_F - 0.01)$ if $\nu > 0$ and 0 otherwise
   (a positive multiplier with a slack bound violates $\nu c_F = 0$). If $v_k > 0.25\,v_{k-1}$
   (the violation did not fall by a factor 4; $v_{-1} = \infty$), $\mu < 10^8$, and the bound is not
   already met within tolerance ($c_F\le0.01$ with complementarity), then $\mu\leftarrow10\mu$.
   The constants 0.25 and 10 are the standard choices (Conn, Gould & Toint; Nocedal & Wright, §17.4).
6. **Stop** when all of the following hold:
   - $c_F\le0.01$ (M within 1% of the bound);
   - $k>0$ (at least two outer iterations);
   - the inner solve was **stationary**: it stopped on its own test (§5.5), or $\phi$ was flat over
     the last 10 accepted steps with $c_F\ge-0.01$. "Flat" means $\lvert\phi_{-11}-\phi_{-1}\rvert\le10^{-3}\lvert\phi_{-1}\rvert$
     (`--stat-tol`), and needs at least 11 accepted steps and a next radius above $10^{-8}$;
   - **complementarity:** if $\nu>0$, the bound is active ($c_F\ge-0.01$).

   The complementarity check matters. If $\nu$ overshoots $\nu^*$, the inner optimum satisfies
   $\nu + \mu c_F\approx\nu^*$, so it is feasible but slack, $c_F\approx(\nu^*-\nu)/\mu<0$. That
   point minimizes $\phi$ less well than it could; without the check the sweep would accept it.

The outer loop also ends at `--max-outer` iterations, or on `--stall-stop` if that is given.

### 4.3 Several bounds: continuation

`--alphas a1 a2 ...` runs the bounds in sequence, each starting from the previous end state. The
sweep stops at the first α whose bound is not met ($c_F > 0.01$ when its outer loop ends, for any
reason); the remaining α's are not run. `NEXT` in the control file (§9) moves on regardless.

### 4.4 Infeasible bounds and the "floor"

When α is below what any current can reach at this resolution, $c_F\gg1$ throughout. The penalty
term $\tfrac12\mu c_F^2$ then dominates the merit even at $\mu = 1$: in the runs of §7.4,
$c_F\approx15.6$ gives $\tfrac12\mu c_F^2\approx122$, against $\phi\ll1$ and $\nu_F\approx10^{-11}$.
(Over further outer iterations $\nu$ and $\mu$ would also grow; those runs never got past the first.)
So the inner loop effectively **minimizes M**, with $\phi$ acting as a weak regularizer. The lowest
$M$ such a run reaches is what we call the floor. At 64/64/6 the floor is $\le 0.0161\,M_\text{DESC}$
on the solve grid a/200 ($\le0.0162$ on a/240): that is the lowest value reached so far, by runs
that were still descending. Such runs end on the time budget (`--time-budget`, counted from program start including
setup) or on a `STOP` command (§9), not on the outer stop test.

---

## 5. Inner loop: one trust-region step

At fixed $(\nu,\mu)$, each inner step does four things (`Solver.inner`):
1. build a quadratic model of $\mathcal L_A$ around the current $u$;
2. minimize the model inside a trust region (an ellipsoid) with a Krylov method;
3. evaluate the true merit at the trial point;
4. accept or reject the trial and resize the region.

### 5.1 The model: residual rows whose sum of squares models the merit

The solver uses a Gauss–Newton model: a residual vector $\mathbf r$, its Jacobian $\mathsf K$, and

$$
m(\mathbf d) = \tfrac12\lVert\mathbf r + \mathsf K\mathbf d\rVert^2 .
$$

The rows (`FixedRows`), with all weights **frozen at the current point**, are:

$$
\mathbf r = \begin{pmatrix}\mathbf g\\ s_1\mathbf z_1\\ \vdots\\ s_{n_\text{pts}}\mathbf z_{n_\text{pts}}\\ 0\end{pmatrix},\qquad
\mathsf K = \begin{pmatrix}\partial\mathbf g/\partial u\\ s_1\,\partial\mathbf z_1/\partial u\\ \vdots\\ s_{n_\text{pts}}\,\partial\mathbf z_{n_\text{pts}}/\partial u\\ \mathbf k_F^T\end{pmatrix},
\qquad
s_i^2 = \frac{\psi'}{\varepsilon N_B\,\bar z_i},\quad \bar z_i = \max(\lvert\mathbf z_i\rvert, w_i\tau),
\quad \mathbf k_F = \sqrt{\mu}\,\nabla c_F .
$$

Here $\mathbf z_i = w_i\mathbf F_i$, so $M = \sum_i\lvert\mathbf z_i\rvert/N_B$. Choosing the
weights $s_i$ this way, recomputed at every step, is *iteratively reweighted least squares* (IRLS):
a sum of norms is minimized through a sequence of weighted sums of squares (motivated by the
majorizer below). The floor $\tau = 10^{-3}\,\varepsilon N_B$ (`tau_frac`) avoids dividing by zero
where $\mathbf F_i = 0$; $\lvert\mathbf z_i\rvert < w_i\tau$ is the same as $\lvert\mathbf F_i\rvert<\tau$,
i.e. points whose force is below 0.1% of the bound. $\mathsf K_z = \partial\mathbf z/\partial u$
(rows $w_i\mathsf D_i$, with $\mathsf D = \partial\mathbf F/\partial u$) is the exact derivative of
the quadratic force: it includes both $\delta\mathbf J\times\mathbf B$ and
$\mathbf J\times\mathbf B_\text{ann}[\delta\mathbf J]$, followed by $\Pi$.

**The gradient is exact.** With the $s_i$ frozen, at points above the floor ($\bar z_i = \lvert\mathbf z_i\rvert$):

$$
\nabla\Big[\tfrac12\sum_i s_i^2\lvert\mathbf z_i\rvert^2\Big]
= \sum_i s_i^2\,\lvert\mathbf z_i\rvert\,\nabla\lvert\mathbf z_i\rvert
= \frac{\psi'}{\varepsilon N_B}\sum_i\nabla\lvert\mathbf z_i\rvert
= \frac{\psi'}{\varepsilon}\nabla M = \psi'\,\nabla c_F,
$$

which is the gradient of $\psi(c_F)$. With $\nabla\tfrac12\lVert\mathbf g\rVert^2 = \nabla\phi$,
the model's gradient $\mathsf K^T\mathbf r$ equals $\nabla\mathcal L_A$. (At floored points the
contribution is scaled by $\lvert\mathbf z_i\rvert/\bar z_i<1$.)

**The model's value is not the merit.** $\tfrac12\lVert\mathbf r\rVert^2$ is not $\mathcal L_A$;
for example the force rows give $\tfrac{\psi'}{2}\tfrac{M}{\varepsilon}$, not $\psi$. Only the
model's *changes* matter. The predicted decrease of a step is therefore $m(0) - m(\mathbf d)$, never
$\mathcal L_A - m(\mathbf d)$.

**Why squares of norms: a majorizer.** A *majorizer* of a function at a point is a function that is
everywhere $\ge$ it and equal to it at that point; minimizing a majorizer guarantees progress on the
original (the *majorize–minimize*, MM, principle; Hunter & Lange, 2004). For one point, with $\mathbf x = \mathbf z+\delta$,

$$
\lvert\mathbf x\rvert \le \frac{\lvert\mathbf x\rvert^2 + \lvert\mathbf z\rvert^2}{2\lvert\mathbf z\rvert}
\iff (\lvert\mathbf x\rvert - \lvert\mathbf z\rvert)^2\ge0,
\qquad
\frac{\lvert\mathbf z+\delta\rvert^2+\lvert\mathbf z\rvert^2}{2\lvert\mathbf z\rvert} = \lvert\mathbf z\rvert + \hat{\mathbf z}\cdot\delta + \frac{\lvert\delta\rvert^2}{2\lvert\mathbf z\rvert}.
$$

The row model is this right-hand side, up to the factor $s_i^2\lvert\mathbf z_i\rvert = \psi'/(\varepsilon N_B)$
and an additive constant:
$\tfrac12 s_i^2\lvert\mathbf z+\delta\rvert^2 = s_i^2\lvert\mathbf z\rvert\cdot\frac{\lvert\mathbf z+\delta\rvert^2+\lvert\mathbf z\rvert^2}{2\lvert\mathbf z\rvert} - \tfrac12 s_i^2\lvert\mathbf z\rvert^2$.
**Caveat:** in the model, $\delta$ is the *linearized* change $\mathsf K_z\mathbf d$, whereas the
true change of $\mathbf z$ also contains the quadratic term $w_i\mathbf q_i(\mathbf d,\mathbf d)$
(§7.1). So the model majorizes $\lvert\mathbf z+\mathsf K_z\mathbf d\rvert$, not the true merit, and
MM's guarantee of monotone descent does not apply. That is why the trust-region ratio test is still
needed.

**The penalty-curvature row.** By the chain rule,
$\nabla^2\psi(c_F(u)) = \psi''\,\nabla c_F\nabla c_F^T + \psi'\,\nabla^2 c_F$, with $\psi'' = \mu$
where $\nu+\mu c_F>0$. The force rows model the second term (through the majorizer of each
$\lvert\mathbf z_i\rvert$). The row $\mathbf k_F$, with residual 0, supplies the first: a zero
residual adds $\mathbf k_F\mathbf k_F^T = \mu\nabla c_F\nabla c_F^T$ to the model Hessian but
nothing to its gradient $\mathsf K^T\mathbf r$. The true curvature is present only where $\nu+\mu c_F>0$. The row is also kept when
$c_F+\nu/\mu>-0.05$ (`band`), i.e. within 5% of the bound when $\nu = 0$. Without that, a model built
just inside the bound would not see that a step can cross it, and its radius collapsed.

### 5.2 The β model (`--model beta`, the default)

**The idea.** Split a change $\delta$ at point $i$ into $\delta_\parallel = \hat{\mathbf z}\cdot\delta$,
along the current force direction, and $\delta_\perp$, the rest. To second order,

$$
\text{true:}\ \lvert\mathbf z\rvert + \delta_\parallel + \frac{\lvert\delta_\perp\rvert^2}{2\lvert\mathbf z\rvert},
\qquad
\text{majorizer:}\ \lvert\mathbf z\rvert + \delta_\parallel + \frac{\delta_\parallel^2+\lvert\delta_\perp\rvert^2}{2\lvert\mathbf z\rvert},
\qquad
\beta\text{ model:}\ \lvert\mathbf z\rvert + \delta_\parallel + \frac{\beta\delta_\parallel^2+\lvert\delta_\perp\rvert^2}{2\lvert\mathbf z\rvert}.
$$

A norm has no curvature along its own direction: $\lvert\mathbf z + t\hat{\mathbf z}\rvert = \lvert\mathbf z\rvert + t$
until it reaches zero. The majorizer adds curvature $1/\lvert\mathbf z\rvert$ there; the β model adds
$\beta/\lvert\mathbf z\rvert$.

**How it is implemented** (`FixedRowsB`):
1. Replace each force row's Jacobian by its projection with $\mathsf P_\beta = \mathsf I - (1-\sqrt\beta)\hat{\mathbf z}\hat{\mathbf z}^T$,
   which scales the $\hat{\mathbf z}$ component by $\sqrt\beta$.
2. Scale the residual to $\mathbf z/\sqrt\beta$, so that the linear term stays exact.

Check: per point, the row model is
$\tfrac12 s^2\big[(\lvert\mathbf z\rvert/\sqrt\beta + \sqrt\beta\,\delta_\parallel)^2 + \lvert\delta_\perp\rvert^2\big]
= \tfrac{\psi'}{\varepsilon N_B}\big[\tfrac{\lvert\mathbf z\rvert}{2\beta} + \delta_\parallel + \tfrac{\beta\delta_\parallel^2+\lvert\delta_\perp\rvert^2}{2\lvert\mathbf z\rvert}\big]$.
That is the β model up to a constant, with the exact gradient. The projection is applied only at
points above the floor.

**Why β = 1/2.** Along $\hat{\mathbf z}$ the 1-D model $\lvert\mathbf z\rvert + \delta + \beta\delta^2/(2\lvert\mathbf z\rvert)$
is minimized at $\delta = -\lvert\mathbf z\rvert/\beta$, with minimum value $\lvert\mathbf z\rvert(1 - 1/2\beta)$.
That is $\ge0$ if and only if $\beta\ge1/2$. So β = 1/2 is the most optimistic choice that never
predicts a negative norm. (Separately, like any Gauss–Newton model, it is positive semidefinite for
every β ≥ 0, so the subproblem never sees negative curvature.)

**Worked examples** with $\lvert\mathbf z\rvert = 2$:

| change | truth | majorizer | β = ½ model |
|---|---|---|---|
| $\delta_\parallel = +0.5$ | 2.5 | 2.5625 | 2.53125 |
| $\delta_\parallel = -1$ | 1 | 1.25 | 1.125 |
| $\delta_\parallel = -4$ | 2 (norm reaches 0 at −2, then grows back to 2) | 2 (touches the truth) | 0 (the model's minimum) |
| $\lvert\delta_\perp\rvert = 1$ | $\sqrt5 = 2.2361$ | 2.25 | 2.25 |

The third row shows what "optimistic" means: the β = ½ model reaches the right minimum value, 0,
but at $\delta_\parallel = -4$, twice as far as the true zero at −2. The majorizer is never below
the truth, so it is too pessimistic for long steps along $\hat{\mathbf z}$.

**Why not the exact Hessian?** The exact model (`--model hvp`, Hessian–vector products) includes the
second derivatives of $\lvert\mathbf z_i\rvert$ and of $\mathbf z_i$ itself, and can be indefinite.
In one measured case (an earlier ripple run at α = 0.7, while $M$ was still far above the bound,
$\lvert c_F\rvert>0.1$) it reached the same optimum with 2.8× less solve time. Near the bound (α = 0.05) it collapsed the trust radius: its curvature changes abruptly where
some $\lvert\mathbf z_i\rvert\to0$. The β model is the default for robustness.

### 5.3 The trust region and the metric

The step is measured in a **metric**. Let $L$ be a lower-triangular Cholesky factor (§6) and set
$\mathbf y = L^T\mathbf d$. The trust region is $\lVert\mathbf y\rVert\le\Delta$, which in the
original coordinates is the ellipsoid $\mathbf d^T(LL^T)\mathbf d\le\Delta^2$. In $\mathbf y$ the model is

$$
m(\mathbf y) = m_0 + \boldsymbol\gamma\cdot\mathbf y + \tfrac12\mathbf y^T\mathsf H_y\mathbf y,
\qquad m_0 = \tfrac12\lVert\mathbf r\rVert^2,
\quad \boldsymbol\gamma = L^{-1}\mathsf K^T\mathbf r,
\quad \mathsf H_y = L^{-1}\mathsf K^T\mathsf K L^{-T}.
$$

If $LL^T$ equalled $\mathsf K^T\mathsf K$, then $\mathsf H_y$ would be the identity and the problem
trivial. A good metric makes $\mathsf H_y$ close to the identity, so the Krylov solve below
converges in few iterations; this is *preconditioning*. A product $\mathsf H_y\mathbf v$ takes
$L^{-T}\mathbf v$ (to a step in $u$), then $\mathsf K$ and $\mathsf K^T$, then $L^{-1}(\cdot)$ (back
to metric coordinates): two triangular solves.

### 5.4 Solving the subproblem: GLTR (Lanczos)

We need $\min_{\lVert\mathbf y\rVert\le\Delta} m(\mathbf y)$ with $\mathsf H_y$ available only
through products. This is done with the *generalized Lanczos trust-region* method (GLTR; Gould,
Lucidi, Roma & Toint, 1999), in `lanczos_tr_step_H`.

1. **Lanczos.** Start from $\mathbf q_1 = \boldsymbol\gamma/\lVert\boldsymbol\gamma\rVert$ and build an
   orthonormal basis of the *Krylov space* $\{\boldsymbol\gamma, \mathsf H_y\boldsymbol\gamma, \mathsf H_y^2\boldsymbol\gamma,\dots\}$:

$$
\mathsf H_y\mathbf q_k = \omega_{k-1}\mathbf q_{k-1} + \eta_k\mathbf q_k + \omega_k\mathbf q_{k+1}.
$$

   Each new vector is reorthogonalized against all previous ones (full reorthogonalization), which
   costs $k\,n_u$ memory: 0.35 GB at $k = 1500$, $n_u = 28{,}750$ (about twice that transiently,
   because the basis is re-stacked each iteration; this copy is also why the time per iteration
   grows with $k$, from ≈0.24 s at $k<100$ to ≈0.48 s at $k>600$). In the basis
   $\mathsf Q_k = [\mathbf q_1\dots\mathbf q_k]$, $\mathsf H_y$ becomes the tridiagonal
   $\mathsf T_k$ with diagonal $\eta$ and off-diagonal $\omega$.
2. **Small trust-region problem.** Solve
   $\min_{\lVert\mathbf h\rVert\le\Delta}\lVert\boldsymbol\gamma\rVert h_1 + \tfrac12\mathbf h^T\mathsf T_k\mathbf h$ exactly.
   (In the basis $\mathsf Q_k$, $\boldsymbol\gamma = \lVert\boldsymbol\gamma\rVert\mathbf q_1$, hence the
   $\lVert\boldsymbol\gamma\rVert h_1$ term.) Its solution satisfies

$$
(\mathsf T_k + \lambda\mathsf I)\mathbf h = -\lVert\boldsymbol\gamma\rVert\mathbf e_1,\qquad
\lambda\ge0,\qquad \lambda(\lVert\mathbf h\rVert - \Delta) = 0,\qquad \mathsf T_k+\lambda\mathsf I\succeq0
$$

   (Moré & Sorensen, 1983). The first three are the Lagrange conditions for a ball constraint, with
   λ its multiplier: stationarity, a non-negative multiplier, and either an interior solution
   (λ = 0) or one on the boundary. The fourth, positive semidefiniteness, makes them sufficient for
   a global minimizer even when $\mathsf T_k$ is indefinite. With the eigendecomposition
   $\mathsf T_k = \mathsf V\Lambda\mathsf V^T$ (eigenvalues $\lambda_j$, eigenvectors $\mathbf v_j$,
   the columns of $\mathsf V$; not the subspace directions of §7.3):
   - If $\lambda = 0$ gives $\lVert\mathbf h\rVert\le\Delta$, that is the solution: an **interior** step.
   - Otherwise $\lambda>0$ solves the scalar *secular equation*
     $\lVert\mathbf h(\lambda)\rVert^2 = \sum_j(\mathbf v_j\cdot\lVert\boldsymbol\gamma\rVert\mathbf e_1)^2/(\lambda_j+\lambda)^2 = \Delta^2$.
     The code solves it by bisection to $10^{-8}$ relative, on the bracket
     $[\max(0,-\lambda_\min),\ \max(0,-\lambda_\min) + \lVert\boldsymbol\gamma\rVert/\Delta + \lambda_\max]$.
     This is a **boundary step** ("hit" in logs).
   - The **hard case** is when $\mathbf e_1$ has no component along the eigenvector of the smallest
     eigenvalue and $\lVert\mathbf h(-\lambda_\min)\rVert<\Delta$, so the equation has no root above
     $\max(0,-\lambda_\min)$. The solution then adds a multiple
     of that eigenvector. For an exact unreduced Lanczos $\mathsf T_k$ this cannot happen ($\mathbf e_1$
     has a component on every eigenvector); in floating point the code treats $\mathsf T_k$ as
     singular when $\lambda_\min\le10^{-14}\lambda_\max$. It then skips the interior test and always
     returns a boundary step ("hit"), by bisection or the hard-case formula. This is **not rare**: on
     the longest valley steps $\lambda_\min/\lambda_\max\approx7\times10^{-15}$ (e.g. $[1.95\times10^{-7}, 2.73\times10^{7}]$
     at 1102 iterations), so there "hit", which drives §6.3 and §7.3, is set by this threshold.

   The small problem is re-solved after every Lanczos iteration (tridiagonal eigensolver, $O(k^2)$).

   *Worked example (2×2).* $\mathsf T = \text{diag}(1,4)$, linear term $(1,1)$, $\Delta = 0.5$. The
   unconstrained minimizer $\mathbf h = (-1, -0.25)$ has norm 1.031 > Δ, so λ > 0 solves
   $1/(1+\lambda)^2 + 1/(4+\lambda)^2 = 0.25$, giving λ = 1.169 and $\mathbf h = (-0.461, -0.193)$.
   Check: $0.461^2 + 0.193^2 = 0.250$ ✓. The model value is −0.473, against −0.625 for the
   (inadmissible) unconstrained step.
3. **Step:** $\mathbf y = \mathsf Q_k\mathbf h$ and $\mathbf d = L^{-T}\mathbf y$.

**Stopping rule (`--lanczos-stop tr`, the default).** The trust-region residual of the current
iterate is $\lVert(\mathsf H_y+\lambda\mathsf I)\mathbf y + \boldsymbol\gamma\rVert = \omega_k|h_k|$.
Derivation: the recurrence in matrix form is
$\mathsf H_y\mathsf Q_k = \mathsf Q_k\mathsf T_k + \omega_k\mathbf q_{k+1}\mathbf e_k^T$. With
$\mathbf y = \mathsf Q_k\mathbf h$ and $\boldsymbol\gamma = \lVert\boldsymbol\gamma\rVert\mathsf Q_k\mathbf e_1$,
$(\mathsf H_y+\lambda\mathsf I)\mathbf y + \boldsymbol\gamma = \mathsf Q_k[(\mathsf T_k+\lambda\mathsf I)\mathbf h + \lVert\boldsymbol\gamma\rVert\mathbf e_1] + \omega_k h_k\mathbf q_{k+1} = \omega_k h_k\mathbf q_{k+1}$,
whose norm is $\omega_k|h_k|$. Lanczos stops when $\omega_k|h_k|\le\texttt{cg\_tol}\cdot\lVert\boldsymbol\gamma\rVert$
(`--cg-tol`, $10^{-2}$), when $\omega_k\approx0$ (the Krylov space is exhausted), or at the cap
`--cg` (default 400; 1500 recommended). (The flag names say "cg" for historical reasons; the method
is Lanczos.)

The older rule (`both`) also waited for the unconstrained minimizer of the Krylov model,
$\mathbf y_\text{unc} = -\mathsf Q_k\mathsf T_k^{-1}\lVert\boldsymbol\gamma\rVert\mathbf e_1$, to converge.
That value feeds only the "share" $(m(0)-m(\mathbf y_\text{unc}))/m(0)$, used by one stopping test
(§5.5). It converges slowly in
the soft directions, so boundary steps ran to the cap (§8).

**Cost of an iteration:** one $\mathsf K\mathbf v$ and one $\mathsf K^T\mathbf x$ (0.2 s together at
64/64/6), two triangular solves ($O(n_u^2)$ each), the reorthogonalization against $k$ stored
vectors ($O(k\,n_u)$, including re-stacking the basis) and the small tridiagonal eigenproblem
($O(k^2)$, negligible). Measured, 0.24 s per iteration at small $k$ and 0.48 s at $k>600$; the
growth is the reorthogonalization.

### 5.5 Accepting the step and resizing the region

The **predicted decrease** is $\text{pred} = m(0) - m(\mathbf y)$, and the **ratio** is
$\text{ratio} = (\mathcal L_A - \mathcal L_A^\text{new})/\text{pred}$. The base rules (accepting any
decrease is a common, permissive variant of the textbook threshold) are:
- **Acceptance:** a step is accepted if it lowers the merit at all ($\mathcal L_A^\text{new}<\mathcal L_A$).
- **Poor ratio:** if ratio < 0.25, $\Delta\leftarrow0.25\lVert\mathbf y\rVert$.
- **Good boundary step:** if ratio > 0.75 and the step hit the boundary, $\Delta\leftarrow\max(2\lVert\mathbf y\rVert,\Delta)$.
- **Otherwise** $\Delta$ is unchanged.

Three refinements were needed near the floor (opt-in, recommended):

- **Noise-tolerant ratio (`--f-noise auto`).** At the start of every inner solve the solver measures
  $\sigma_f$, the sample standard deviation of $\mathcal L_A$ over the current point and six random
  points at metric distance $10^{-10}$. (At that distance the smooth change,
  $\approx\lVert\boldsymbol\gamma\rVert\times10^{-10}$, is far below the noise.) The rules become:
  The rules, **checked in this order**:
  1. *Small prediction:* a step with $\text{pred}<10\sigma_f$ that does not raise the merit by more
     than $4\sigma_f$ is accepted with $\Delta$ unchanged. At that size its outcome says nothing
     reliable about the model, and rejecting it would shrink Δ on the basis of noise alone;
     repeated, that collapses the radius.
  2. Otherwise $\text{ratio} = (\mathcal L_A - \mathcal L_A^\text{new} + 2\sigma_f)/(\text{pred} + 2\sigma_f)$,
     and the step is accepted if ratio > 0.1, with the base radius rules. A rejection sets
     $\Delta\leftarrow0.25\min(\lVert\mathbf y\rVert,\Delta)$.

  The constants are heuristic: adding $2\sigma_f$ to both differences keeps the ratio near 1 when
  both are at noise level (the difference of two noisy values has spread $\sqrt2\sigma_f$); the
  thresholds $10\sigma_f$ and $4\sigma_f$ separate "resolvable" from "noise". Accepting a rise of
  up to $4\sigma_f$ gives up strict monotone descent. For the theory of trust regions with noisy
  values see Sun & Nocedal (2023).
- **Radius by step (`--radius-by-delta`).** An accepted *interior* step with ratio < 0.25 shrinks
  the radius to $0.25\Delta$ instead of $0.25\lVert\mathbf y\rVert$, which can be orders of
  magnitude smaller (for an interior step $\lVert\mathbf y\rVert<\Delta$).
  A single very short step, such as the short "wall-correction" steps of §7.1, then cannot collapse
  the radius by orders of magnitude.
- **No share stop (`--no-share-stop`).** The "share" is the relative model decrease that the
  *unconstrained* Krylov minimizer would achieve, $(m(0) - m(\mathbf y_\text{unc}))/m(0)$. One
  stopping test declares the inner solve stationary when share < $10^{-8}$. In long shallow valleys
  that test fired after short Lanczos runs that had only seen stiff directions; this flag disables it.

**When a subspace step is taken** (`--subspace`, §7.3), these rules are bypassed for that step: it is
accepted and the radius follows the rule of §7.3.

**Other inner stops** (merit $\mathcal L_A$, relative tests):
- the relative decrease of an accepted step falls below $10^{-12}$ (only without `--f-noise`);
- $\lVert\boldsymbol\gamma\rVert$ falls below `--gtol` ($10^{-3}$) times its first value;
- the merit $\mathcal L_A$ changed by less than $10^{-4}$ relative over the last 10 accepted steps
  (needs at least 11 accepted steps and $\Delta>10^{-8}$). (The outer test of §4.2 instead looks at
  $\phi$, with a looser $10^{-3}$, because it asks whether the *current change* has settled.)
- the step budget (`--maxiter`) is reached;
- $\Delta<10^{-14}$ (tested only after a rejected step).

The $\lVert\boldsymbol\gamma\rVert$ test compares with the first value of the inner solve even after an
inner metric rebuild, which changes the norm in which $\boldsymbol\gamma$ is measured.

**Worked example.** Take illustrative numbers $\mathcal L_A = 122.6416$, $m(0) - m(\mathbf y) = 0.0041$,
a boundary step, and $\mathcal L_A^\text{new} = 122.6358$.
- **Base rules:** ratio = 0.0058/0.0041 = 1.41, so the step is accepted and $\Delta\leftarrow2\lVert\mathbf y\rVert$.
- **With $\sigma_f = 10^{-4}$:** pred = 0.0041 ≥ $10\sigma_f$ = 0.001, so the ratio test applies:
  ratio = (0.0058 + 0.0002)/(0.0041 + 0.0002) = 1.40 > 0.1. Accepted, and $\Delta\leftarrow2\lVert\mathbf y\rVert$ as before.
- **With $\sigma_f = 10^{-3}$:** pred = 0.0041 < $10\sigma_f$ = 0.01, and the merit did not rise by
  more than $4\sigma_f$, so the noise-level rule applies: accepted with $\Delta$ unchanged. (The ratio,
  1.28, is not used.) Even $\mathcal L_A^\text{new} = 122.6450$, a rise of 0.0034 ≤ $4\sigma_f$, would be
  accepted this way.
- **A rejection:** with $\sigma_f = 10^{-4}$ and $\mathcal L_A^\text{new} = 122.6450$, the ratio is
  (−0.0034 + 0.0002)/(0.0041 + 0.0002) = −0.74 < 0.1, so the step is rejected and
  $\Delta\leftarrow0.25\min(\lVert\mathbf y\rVert,\Delta)$.

---

## 6. The metric (preconditioner) and when it is rebuilt

### 6.1 What the metric is

With `--precond force` (default),

$$
LL^T = \mathsf H_g + \mathsf H_{F,\text{loc}} + \mathbf k_F\mathbf k_F^T + 10^{-10}\,\overline{\text{diag}}\,\mathsf I .
$$

- **$\mathsf H_g = (\partial\mathbf g/\partial u)^T(\partial\mathbf g/\partial u)$**, the Gram matrix
  of the misfit rows. A *Gram matrix* of rows $\mathsf A$ is $\mathsf A^T\mathsf A$. It is constant
  because $\mathbf g$ is affine, so it is built once.
- **$\mathsf H_{F,\text{loc}}$**, the Gram matrix of the force rows with two simplifications:
  1. Only the **local** part of the linearized force is kept,
     $\mathsf C = [\sqrt w\,(\delta\mathbf J_j\times\mathbf B_0)]_j$, one column per unknown, with
     $\delta\mathbf J_j = \partial\mathbf J/\partial u_j$ and $\mathbf B_0$, $\mathbf J_0$ the field
     and current at the current point. The non-local Biot–Savart part
     $\mathbf J_0\times\mathbf B_\text{ann}[\delta\mathbf J]$ is dropped.
  2. The majorizer weights $s_i^2$ are used, without the β projection.

  With $\mathsf Q = \text{diag}(s_i^2 w_i)$, each entry repeated over the three components (the
  force row at point $i$ is $s_i\mathbf z_i = s_i\sqrt{w_i}\,\mathbf f_i$),
  $\mathsf H_{F,\text{loc}} = \mathsf C^T\Pi^T\mathsf Q\,\Pi\mathsf C$. Expanding
  $\Pi = \mathsf I - \mathsf P\mathsf G^{-1}\mathsf P^T$ (Π is symmetric),
  $\Pi\mathsf Q\Pi = \mathsf Q - \mathsf Q\mathsf P\mathsf G^{-1}\mathsf P^T - \mathsf P\mathsf G^{-1}\mathsf P^T\mathsf Q + \mathsf P\mathsf G^{-1}(\mathsf P^T\mathsf Q\mathsf P)\mathsf G^{-1}\mathsf P^T$,
  so

$$
\mathsf C^T\Pi\mathsf Q\,\Pi\mathsf C = \mathsf C^T\mathsf Q\mathsf C - \mathsf C^T\mathsf P\,\mathsf G^{-1}\mathsf P^T\mathsf Q\mathsf C - \big(\mathsf C^T\mathsf P\,\mathsf G^{-1}\mathsf P^T\mathsf Q\mathsf C\big)^T + \mathsf C^T\mathsf P\,\mathsf G^{-1}(\mathsf P^T\mathsf Q\mathsf P)\,\mathsf G^{-1}\mathsf P^T\mathsf C .
$$

  The projector couples all points, but each of the four blocks ($\mathsf C^T\mathsf Q\mathsf C$,
  $\mathsf C^T\mathsf P$, $\mathsf C^T\mathsf Q\mathsf P$, $\mathsf P^T\mathsf Q\mathsf P$) is a sum
  over points, because $\mathsf Q$ is diagonal. They are accumulated slice by slice (sizes
  $n_u\times n_u$, $n_u\times n_p$, $n_u\times n_p$, $n_p\times n_p$); the $\mathsf G^{-1}$ solves
  are applied afterwards to these small blocks. So the projection is still applied exactly.
- **$\mathbf k_F\mathbf k_F^T$**, the penalty curvature of §5.1.
- **The small diagonal shift** $10^{-10}\,\overline{\text{diag}}$, where $\overline{\text{diag}}$ is
  the mean of the diagonal of the matrix being factored, guards the Cholesky factorization.

The β projection and the floor τ are left out of $\mathsf H_{F,\text{loc}}$; this only changes how
well the metric approximates the model Hessian, not the model itself.

$L$ is a dense $n_u\times n_u$ factor (6.6 GB in double precision at 64/64/6). Triangular solves with
it cost $O(n_u^2)$, small next to a Biot–Savart solve.

*Why the local part?* It is the part that can be accumulated point by point, and it carries the
large weights $s_i^2$ that dominate the model when α is small and $\psi'$ is large. With
$\mathsf H_g$ alone, the subproblem was far from converged at the 400-iteration cap (NCSX 32/32/24
at α = 0.05, §8). What the local part misses, the non-local response, is left to the Krylov solve.

### 6.2 How it is built (the "lean" build) and its memory

The blocks are accumulated slice by slice into arrays of their exact size, not padded to a block
multiple. Each compiled GPU kernel reuses its input buffer for its output (*buffer donation*, a JAX
feature), so that no second copy exists. $\mathsf H_g$ is kept in host (CPU) memory and added in
row blocks.

*Resident* memory means what stays on the GPU for the whole run: mainly the field-grid and
collocation-point tables of the basis functions and their derivatives, plus the factor of
$\mathsf G$ ($8n_p^2$ = 1.56 GB at 64/64/6). Empirically it is about 0.93 GB per $10^9$
(collocation points × $n_u$).

Measured device (GPU) peaks at 64/64/6 ($n_u = 28{,}750$, $n_p = 13{,}959$):
- **$\mathsf H_g$ phase:** 29.5 GB. That is the resident 9.7 GB plus about $3\times8n_u^2$ bytes.
  The factor 3 is measured ((29.52 − 9.68)/6.61 = 3.0); plausibly $\mathsf H_g$, its Cholesky factor
  and the solver's workspace, but this was not traced.
- **Force-metric phase:** 32.4 GB, against an empirical fit (within 7% on the runs measured) of about
  $1.075\times[\text{resident} + 8(n_u^2 + 2n_un_p + 2n_p^2) + \text{one slice of columns}]$ bytes.
  The terms are the blocks of §6.1: $n_u^2$ for the $n_u\times n_u$ accumulator,
  $2n_un_p$ for $\mathsf C^T\mathsf P$ and $\mathsf C^T\mathsf Q\mathsf P$, $2n_p^2$ for
  $\mathsf P^T\mathsf Q\mathsf P$ and the factor of $\mathsf G$. The slice term is about 4.3 GB at
  64/64/6 (by difference: 32.42/1.075 − 9.68 − 16.15).

A force-metric rebuild takes 290–380 s on an A100-80GB. At 80/80/6 a/200 ($n_u = 44{,}254$,
resident 22.1 GB measured) the $\mathsf H_g$ phase model predicts $22.1 + 3\times8n_u^2/10^9 = 69$ GB,
but the run failed on an 80 GB card with an out-of-memory error at a 31.3 GB request inside the
Cholesky. So the factor 3 is not a safe margin at that size.

### 6.3 When it is rebuilt

- **At each outer iteration**, unless the keep test of §4.2 passes.
- **Inside the inner loop**, before a subproblem, when the previous subproblem either:
  - hit the Lanczos cap; or
  - tipped a *rent-or-buy* counter. Let $k_0$ be the Lanczos count of the first subproblem after a
    rebuild. Each later subproblem adds $\max(0, k - k_0)\times(\text{its time per iteration})$
    to a counter (never subtracts), and a rebuild happens when the counter reaches the measured
    time of the last rebuild; the counter then resets to 0. The rule is borrowed from the
    *ski-rental* problem (Karlin, Manasse, Rudolph & Sleator, 1988): keep paying the extra
    iterations (renting) until they add up to the one-off cost of a rebuild (buying). In that
    idealized setting (fixed prices) the cost is never more than twice the best choice made in
    hindsight; here the prices vary, so no guarantee is claimed.
    *Example:* rebuild 300 s, each subproblem 100 iterations above $k_0$ at 0.5 s each, i.e. 50 s
    per subproblem: the counter reaches 300 s after the 6th, and the next subproblem rebuilds first.
- **Radius after an inner rebuild** (outer rebuilds reset it, §4.2):
  $\Delta\leftarrow\max\big(\lVert L_\text{new}^{-1}\nabla\mathcal L_A\rVert,\ \Delta\,\lVert L_\text{new}^{-1}\nabla\mathcal L_A\rVert/\lVert L_\text{old}^{-1}\nabla\mathcal L_A\rVert\big)$.
  This keeps the radius at the same multiple of the gradient length, but never below one new
  gradient length.
- **`--no-rebuild-on-hit` (recommended).** A subproblem whose step hit the trust-region boundary
  triggers neither inner rule, and its time is not counted. In the long valleys of §7, boundary
  steps routinely run hundreds of Lanczos iterations while still giving ratios near 1. The step was
  good; Lanczos was still refining it inside a radius that bound anyway. In the first α = 0.001 run
  at 64/64/6, a rebuild there cost 335 s, and the following steps lowered $M/\varepsilon$ by only
  about 0.1%. Interior cap hits still trigger a rebuild.

---

## 7. Valley navigation: geodesic acceleration and the exact subspace search

### 7.1 The problem these solve

Near the floor (α = 0.001 at 64/64/6), the inner loop repeats a characteristic cycle:
1. a long boundary step: $\lVert\mathbf y\rVert\sim10^2$–$10^3$, 250–900 s of Lanczos, ratio ≈ 1;
2. a jump of the model gradient, $\lVert\boldsymbol\gamma\rVert$, from about 1 to about 100;
3. several very short interior steps ($\lVert\mathbf y\rVert\sim0.1$, a few seconds each), each
   recovering about $10^{-3}$ in $M/\varepsilon$;
4. now and then, a boundary step at the largest radius that is rejected (ratio −0.4 to −0.7),
   throwing away a 400–900 s Lanczos solve.

This is a **curved valley**: a curved set of points along which the merit decreases slowly, with
steep growth across it. Its *bottom* is the curve of lowest merit across sections; its *walls* are
the steep sides. (The Rosenbrock function $(1-x)^2 + 100(y-x^2)^2$, whose bottom is the parabola
$y = x^2$, is the textbook example.) Each trial is a straight segment $u\to u+\mathbf d$. The
Gauss–Newton model treats the force as linear in the step, but it is quadratic, so the bottom
bends away from the straight segment. A long segment leaves the bottom and ends on a wall: hence
the jump of the gradient, and the short "wall-correction" steps that climb back down.

The bend is the second-order term of the force. Since $\mathbf F$ is exactly quadratic,

$$
\mathbf F(u+\mathbf d) = \mathbf F(u) + \mathsf D\mathbf d + \mathbf q(\mathbf d,\mathbf d),
\qquad
\mathbf q_i = \frac{1}{\sqrt{w_i}}\Big(\Pi\big[\sqrt w\,(\delta\mathbf J\times\mathbf B_\text{ann}[\delta\mathbf J])\big]\Big)_i,
\quad \delta\mathbf J = \mathbf J(u+\mathbf d) - \mathbf J(u),
$$

with $\mathsf D = \partial\mathbf F/\partial u$ the force Jacobian; the force block of $\mathsf K$ is
$s_iw_i\mathsf D_i$ per point. The symmetric bilinear form behind $\mathbf q(\mathbf d,\mathbf d)$ is
$\mathbf q_i(\mathbf a,\mathbf b) = \frac{1}{2\sqrt{w_i}}\Big(\Pi\big[\sqrt w\,(\delta\mathbf J_a\times\mathbf B_\text{ann}[\delta\mathbf J_b] + \delta\mathbf J_b\times\mathbf B_\text{ann}[\delta\mathbf J_a])\big]\Big)_i$,
with $\delta\mathbf J_a$ the linear change of $\mathbf J$ along $\mathbf a$; $\mathbf q(\mathbf d,\mathbf d)$ is its
diagonal.
The Gauss–Newton model drops $\mathbf q$.

### 7.2 Geodesic acceleration (`--geo-accel`; also the `accel` direction below)

**Idea** (Transtrum & Sethna, 2012). Picture the set of all force fields $\mathbf F(u)$ as a curved
surface (the *model manifold*). A straight step in $u$ moves along a curve on that surface, while
the Gauss–Newton step assumes a straight line in force space. A *geodesic* step follows the surface
instead: keep the step's first-order direction, and add a second-order correction so that the
force changes, to second order, as the linear model predicted. Expand the force at
$u + \mathbf d + \mathbf a$ with a small correction $\mathbf a$:

$$
\mathbf F(u+\mathbf d+\mathbf a) = \mathbf F + \mathsf D\mathbf d + \mathsf D\mathbf a + \mathbf q(\mathbf d,\mathbf d) + 2\mathbf q(\mathbf d,\mathbf a) + \mathbf q(\mathbf a,\mathbf a).
$$

For $\lVert\mathbf a\rVert\ll\lVert\mathbf d\rVert$ the last two terms are higher order. Choosing
$\mathsf D\mathbf a\approx-\mathbf q(\mathbf d,\mathbf d)$ cancels the leading error, i.e.
$\mathbf a\approx-(\mathsf D^T\mathsf D)^{-1}\mathsf D^T\mathbf q$. (Transtrum and Sethna write the
step as $\mathbf d + \tfrac12\mathbf a_\text{TS}$, with $\mathbf a_\text{TS}$ built from the second
directional derivative $2\mathbf q$; that is the same step.)

**What the code solves** (`Solver._geodesic`, `Solver._accel_dir`):
1. **The exact second-order term.** Compute
   $\mathbf q(\mathbf d,\mathbf d) = \tfrac12[\mathbf F(u+\mathbf d) + \mathbf F(u-\mathbf d)] - \mathbf F(u)$
   (as in §3). This takes one extra state evaluation, since $u+\mathbf d$ is the trial already done.
2. **Solve for a regularized correction.** Form the weighted rows
   $\tilde{\mathbf q} = (0;\ s_iw_i\mathbf q_i;\ 0)$ and solve
   $\min\tfrac12\lVert\mathsf K\mathbf a+\tilde{\mathbf q}\rVert^2$ subject to
   $\lVert L^T\mathbf a\rVert\le0.75\lVert\mathbf y\rVert$ (`--accel-amax`). The same GLTR solver is
   used, to tolerance $10^{-4}$ with at most 300 iterations. The formula above is the unconstrained,
   force-rows-only limit of this.
3. **With `--geo-accel` alone:** evaluate $u+\mathbf d+\mathbf a$ and keep it if its merit is lower
   than the straight trial's.

*Do not combine `--geo-accel` with `--subspace …accel`.* Both compute the same acceleration solve,
so the cost is paid twice; the subspace search already contains $\mathbf a$ as a direction, and it
would compare against the curved trial instead of the straight one.

*A known imperfection:* under the β model, $\mathsf K$ carries the β projection along
$\hat{\mathbf z}_i$, but $\tilde{\mathbf q}$ does not.

**Measured at 64/64/6.** The correction removes 10–84% of $\tilde{\mathbf q}$, measured as
$1-\lVert\mathsf K\mathbf a+\tilde{\mathbf q}\rVert/\lVert\tilde{\mathbf q}\rVert$ (rising along the run;
77–84% on the later steps), but it is tiny in metric length:
$\lVert L^T\mathbf a\rVert/\lVert\mathbf y\rVert\lesssim6\times10^{-3}$. Each corrected step is only slightly
better than the straight one, and the extra Lanczos solve costs more than the gain (0.79× the
control in §7.4).

*Interpretation (not proven):* most of the straight step's error at these lengths is not removable
by a small correction at fixed direction. What helps is changing the step *length* along
$\mathbf d$, which the subspace search does. Whether the extra directions help is not settled: in
the prev,accel arm their optimal coefficients were small (prev 0.000–0.007, accel 0.001–0.128),
so most of the gain came from $t_1$, an exact line search along $\mathbf d$ with extrapolation. A
$t_1$-only arm would separate the two effects.

### 7.3 Exact subspace search (`--subspace prev`, `accel`, `prev,accel`)

**Idea.** On each boundary step, do not trust the model's step length or its straight line. Find the
best point *exactly* in a small subspace that contains the Lanczos step.

**The reconstruction.** Because $\mathbf F$ is exactly quadratic and $\mathbf g$ affine (§3), on a
K-dimensional affine subspace $u(\mathbf t) = u + \sum_{j=1}^K t_j\mathbf v_j$, at every collocation point,

$$
\mathbf F(\mathbf t) = \mathbf F^{(0)} + \sum_j t_j\mathbf F^{(j)} + \sum_j t_j^2\mathbf F^{(jj)} + \sum_{j<k}t_jt_k\mathbf F^{(jk)},
\qquad \mathbf g(\mathbf t) = \mathbf g^{(0)} + \sum_j t_j\mathbf g^{(j)},
$$

with coefficients obtained from state evaluations at $u\pm\mathbf v_j$ and $u+\mathbf v_j+\mathbf v_k$:

$$
\mathbf F^{(j)} = \tfrac12[\mathbf F(\mathbf v_j) - \mathbf F(-\mathbf v_j)],\quad
\mathbf F^{(jj)} = \tfrac12[\mathbf F(\mathbf v_j) + \mathbf F(-\mathbf v_j)] - \mathbf F^{(0)},\quad
\mathbf F^{(jk)} = \mathbf F(\mathbf v_j+\mathbf v_k) - \mathbf F^{(0)} - \mathbf F^{(j)} - \mathbf F^{(k)} - \mathbf F^{(jj)} - \mathbf F^{(kk)},
$$

and $\mathbf g^{(j)} = \mathbf g(\mathbf v_j) - \mathbf g^{(0)}$. That is $2K$ evaluations for the $\pm$
points plus $K(K-1)/2$ for the pairs, minus one because $u+\mathbf d$ is the trial already
evaluated: $2K - 1 + K(K-1)/2$, i.e. **4 for K = 2 and 8 for K = 3**. After that, the true merit
$\mathcal L_A(\mathbf t)$ costs no Biot–Savart solve:
- $M(\mathbf t) = \sum_iw_i\lvert\mathbf F_i(\mathbf t)\rvert/N_B$ is a weighted sum on the GPU;
- $\phi(\mathbf t) = \tfrac12\lVert\mathbf g(\mathbf t)\rVert^2$;
- $\psi$ is a scalar function of $M(\mathbf t)$.

**The directions.** $\mathbf v_1 = \mathbf d$, the Lanczos step, always. Then:
- `prev`: the previous accepted step of the same inner solve. In a bending valley the last two steps
  span the local plane of the bend; this is a "momentum" direction. It is unavailable on the first
  step of each inner solve, where K drops by one; with `prev` alone the search is then skipped
  for that step.
- `accel`: the acceleration direction $\mathbf a$ of §7.2. It points along the curvature of the
  force map itself.

The extra directions are rescaled to the same metric length $\lVert\mathbf y\rVert$ as $\mathbf d$, so
that their coefficients are comparable. The reconstruction is exact at any scale; the scale only
decides which region the grid box below covers.

**The search** (`Solver._subspace`):
1. **Grid.** Evaluate $\mathcal L_A$ on a grid: $t_1\in[0,4]$ (`--subspace-smax`) and the other
   coordinates in $[-2,2]$ (`--subspace-tmax`); 33×17 points for K = 2, 33×9×9 for K = 3. The
   straight step $\mathbf t = (1,0,\dots)$ and the current point $\mathbf t = 0$ are always included.
2. **Polish.** Run Nelder–Mead from the best grid point, at most 200 function calls. It is
   *unconstrained* and may leave the grid box; that is how $t_1 = 7.5$, or $t_1 = -0.51$, can occur.
   A derivative-free method is used because $M(\mathbf t)$ has kinks where some
   $\mathbf F_i(\mathbf t) = 0$, and K ≤ 3 is small.
3. **Evaluate for real.** Evaluate the minimizer $\mathbf t^\star$ with one more state evaluation.
   - Take it if its merit is below both the current point's and the straight trial's.
   - Otherwise fall back to the straight trial, under the rules of §5.5.
4. **Radius.** Set it from the metric length $\lVert\mathbf y^\star\rVert$, with
   $\mathbf y^\star = L^T\sum_jt^\star_j\mathbf v_j$, of the step actually taken:
   $\Delta\leftarrow2\lVert\mathbf y^\star\rVert$ if $\lVert\mathbf y^\star\rVert\ge0.9\Delta$
   (the step went to the edge or beyond, in any direction), else
   $\Delta\leftarrow\max(\lVert\mathbf y^\star\rVert, 0.25\Delta)$. In the backward example below,
   $\mathbf t^\star\approx(-0.51,0,0)$ on a boundary step ($\lVert\mathbf y\rVert = \Delta$) gives
   $\lVert\mathbf y^\star\rVert\approx0.51\Delta$, so $\Delta\leftarrow0.51\Delta$.

**What this buys:**
- **Longer steps than the trust radius.** $t_1$ reached 7.5 in the A/B runs (14 in an earlier,
  cancelled run), so the radius grows as fast as the valley allows.
- **No boundary step is wasted.** If the straight step overshoots, the search finds the best point
  along the same directions.
  - *Example from §7.4:* the straight trial had merit 255.8 against 117.2 at the current point
    ($\lVert\mathbf y\rVert\approx4\times10^3$). The search chose $\mathbf t^\star\approx(-0.51, 0, 0)$,
    half a step *backwards* along $\mathbf d$, and gained 0.11 in $M/\varepsilon$ at once.
  - At that length the merit along $\mathbf d$ is far from its quadratic model: it has its minimum on
    the other side of $u$. $-\mathbf d$ is not a first-order descent direction, but at half a
    step length the higher-order terms dominate the slope.
  - *A two-point illustration.* Let the merit be $|F_1(t)| + |F_2(t)|$ with $F_1 = 0.5 + t$ and
    $F_2 = 0.5 - 1.1t - 4.2t^2$. At $t = 0$ the merit is 1 with slope $1 - 1.1 = -0.1$, so $+t$ is the
    descent direction. The straight step $t = 1$ gives $1.5 + 4.8 = 6.3$, far worse. Forward there
    is only a shallow minimum, 0.738 at $t = 0.238$ (where $F_2 = 0$). Behind, at $t = -0.5$, both
    terms vanish: the merit is 0. An exact search along the line finds it; a descent method that
    only looks forward cannot.
- **The bend can be followed** through the `prev` and `accel` directions; in the A/B runs their
  coefficients were small after the first steps (see §7.2), so this is the design intent rather
  than a measured effect.

**Rounding grows with the extrapolation.** Each coefficient comes from differences of nearly equal
force fields, so it carries an absolute error of order $\sigma$, the rounding level of the force.
In $\mathbf F(\mathbf t)$ the $t_j^2$ terms dominate for large $|\mathbf t|$, so the error is expected
to grow like $\sigma\lvert\mathbf t\rvert^2$. At 64/64/6 the predicted and actual merits differed by
$10^{-8}$ to $6\times10^{-6}$ relative in the A/B runs ($t_1\le7.5$; no clean trend with $|t|$), and
by $1.4\times10^{-6}$ to $4.9\times10^{-5}$ in the earlier cancelled run as $|t|$ went from 3 to 14
(a factor 35 for a factor 4.7 in $|t|$, i.e. about $|t|^{2.3}$). That
error is logged ("rel …"). **The step is judged only by its real evaluation**, which makes the
search safe. An early version rejected any step whose prediction was off by more than $10^{-6}$, and
so rejected every real improvement.

**Cost per boundary step at 64/64/6:**
- `prev`: 5 state evaluations (≈0.1 s each) plus the grid and Nelder–Mead, about 5 s in total.
- `prev,accel`: 9 evaluations plus the acceleration Lanczos solve, 35–100 s.

Both are small next to the 250–900 s of the boundary step's own Lanczos solve.

**Worked example (1-D, by hand).** Take $F(t) = 1 - 3t + t^2$ (from §3) and the merit $|F|$, at
$t = 0$, where $|z| = 1$ and the linear change is $\delta = -3t$ along $\hat{\mathbf z}$.
- *Majorizer model (β = 1):* $1 - 3t + \tfrac{9}{2}t^2$, minimized at $t = 1/3$, where the true
  $\lvert F(1/3)\rvert = 1/9 = 0.111$.
- *β = ½ model (the default):* $1 - 3t + \tfrac94t^2$, minimized at $t = 2/3$, where
  $\lvert F(2/3)\rvert = \lvert1 - 2 + 4/9\rvert = 5/9 = 0.556$: it overshoots the root.
- *Exact search:* the reconstruction knows $|F(t)|$ for every t. On the grid $t\in[0,4]$ (spacing
  0.125) the best points are $t = 0.375$ and $t = 2.625$, both with $|F| = 0.015625$ (an exact tie);
  Nelder–Mead then polishes to a root, $t = (3-\sqrt5)/2 = 0.382$ or $t = (3+\sqrt5)/2 = 2.618$,
  where $|F| = 0$. Both are minima of $|F|$.
- *Overshoot:* even a step of $t = 1$, where $|F(1)| = 1$ (no better than the start, so a
  trust-region test would reject it), contains the information needed: the search along the same
  direction finds a root.

### 7.4 Measured effect (64/64/6, a/200, α = 0.001)

**Setup.** Four runs ("arms") started from the same state, $M/\varepsilon = 16.6615$. All used the
recommended flags of §9, except that the valley option was varied. They ran one after another on one
A100-80GB, with the control last, as separate processes. Running them on one card removes
hardware differences. (No persistent compile cache was configured, and the time axis counts
inner-step time, so the order should not matter.)

**Comparison points.** *Solve time* is the cumulative wall time of the inner steps (Lanczos,
evaluations, subspace search, inner metric rebuilds), excluding setup. The arms had different
lengths because each ran to the same overall budget (`--time-budget 3300`, i.e. 55 min counted
from program start, so including its own setup).
- Time: 2280 s of solve time, the length of the shortest arm (`--geo-accel`).
- Iterations: 6185 Lanczos iterations, the smallest arm total (also `--geo-accel`), counting every
  Lanczos solve including the acceleration solves.

$M/\varepsilon$ changes only at accepted steps, so it is piecewise constant in time and in
iterations. Where the two "decrease" columns agree, no step was accepted between the arm's
2280-s iteration count and 6185 iterations.

| arm | $M/\varepsilon$ at 2280 s | decrease at 2280 s | × control (time) | Lanczos iterations by 2280 s | decrease at 6185 iterations | × control (iterations) | rejected steps by 2280 s |
|---|---|---|---|---|---|---|---|
| `--subspace prev,accel` | 16.3101 | 0.351 | **2.03** | 3918 | 0.490 | **2.83** | 0 |
| `--subspace prev` | 16.4032 | 0.258 | 1.49 | 5642 | 0.258 | 1.49 | 0 |
| `--geo-accel` | 16.5246 | 0.137 | 0.79 | 6185 | 0.137 | 0.79 | 0 |
| control | 16.4886 | 0.173 | 1.00 | 5825 | 0.173 | 1.00 | 0 |

**End states on two field grids.** The end state is each arm's last accepted state, at the end of
its run (later than 2280 s for all but geo-accel). Values are $M/M_\text{DESC}$ on the solve grid
a/200 and on a finer a/240. The two grids agree, so no arm gained by exploiting field-grid error:

| arm | end of run (solve time) | final $M/\varepsilon$ | a/200 | a/240 |
|---|---|---|---|---|
| prev,accel | 2592 s | 16.1438 | 0.0161 | 0.0162 |
| prev | 2636 s | 16.3063 | 0.0163 | 0.0163 |
| geo-accel | 2280 s | 16.5246 | 0.0165 | 0.0164 |
| control | 2643 s | 16.4572 | 0.0165 | 0.0165 |

All end states keep the shielding at the q = 2 surface. The diagnostic is $b_{21}$: the (2,1)
harmonic, in straight-field-line (PEST) angles, of $b^\rho = \delta\mathbf B\cdot\nabla\rho/\mathbf B\cdot\nabla\zeta$
on the q = 2 surface, where $\delta\mathbf B = \mathbf B_\text{ann}[\mathbf J - \mathbf J_\text{fit}]$ is the
field change from the start (DESC's own resonant field is zero, because its surfaces are nested).
It is quoted as a fraction of the *vacuum* value $3.78\times10^{-4}$, the resonant field the ripple
would produce with no shielding current. For the four end states it is 2.08–2.14%, phase −90°: about
98% of the vacuum resonant field is still cancelled. An open island would need $b_{21}$ to grow
toward the vacuum value (`--b21-report` prints it in an `--eval-only` run).

**Caveats.**
- One run per arm, so the factors have no error bar; run-to-run variability (for example from the
  random points used for $\sigma_f$) has not been measured.
- The lead of `prev,accel` comes from a few large steps, so read the factors as indicative.
- The runs were still descending, so the floor is not yet measured.

---

## 8. The changes that mattered most for speed

Ordered by impact. The measurements come from earlier development runs and were not repeated for
this note; the case is named for each.

| # | change | what it fixed | measured effect (case) |
|---|---|---|---|
| 1 | **Pressure eliminated by the weak-Poisson projection** (§2.4) | with $p$ as an unknown, conditioning is expected to grow with resolution (argued, §2.4) | design choice, not measured; unknowns are $u$ only |
| 2 | **Matrix-free products + GLTR** (§5.4), instead of a dense Jacobian with QR | the dense Jacobian alone needs ≈ 470 GB at 64/64/6, and a QR needs $\approx 2n_\text{rows}n_u^2\approx3\times10^{15}$ flops | the only route that fits; every result here uses it |
| 3 | **Annulus Biot–Savart** (§2.6), instead of a Cartesian FFT box around the plasma | a box spends most of its points outside the torus | ≈ 7× fewer grid points at equal spacing, with 0.7–0.8× the box's rms field error inside the plasma (smooth-current test case) |
| 4 | **Force preconditioner** (§6.1), instead of $\mathsf H_g$ alone | $\mathsf H_g$ ignores the force-bound curvature | a small development test problem: Lanczos iterations 41 → 13. NCSX 32/32/24 at α = 0.05 with $\mathsf H_g$ alone: TR residual/$\lVert\boldsymbol\gamma\rVert$ = 1.21 at the 400 cap (target $10^{-2}$) |
| 5 | **Lanczos stop on the trust-region residual** (§5.4) | the old rule waited for a quantity that only feeds a stopping test | NCSX 48/32/18, α = 0.005: boundary steps 400 (cap) → 79–102 iterations, identical steps |
| 6 | **Exact subspace search, prev + accel** (§7.3) | straight steps cut across a curved valley; overshoots were thrown away | ripple 64/64/6, α = 0.001: 2.03× the control's decrease at equal time, 2.83× at equal Lanczos iterations |
| 7 | **Stall fixes:** noise-tolerant ratio, radius by step, no share stop (§5.5) | radius collapse on merit noise; false "stationary" exits | ripple 32/32/6, a band $p_\text{exp}$-norm bound (§2.5) at α = 0.1: a run that had stopped improving at $M_p/\varepsilon = 1.124$ (it looked converged) kept descending with the fixes, by a steady 0.0023 in $M_p/\varepsilon$ per 12.7 min of solve time |
| 8 | **Lean metric build** (§6.2) | padded copies of $n_u^2$ arrays; $\mathsf H_g$ on the device | NCSX 48/32/18 ($n_u = 41{,}206$): metric peak 53 GB, against ≈ 101 GB projected for the old build |
| 9 | **Smaller locate batches** (`--locate-chunk 131072`) | the last Newton batch of the locate step evaluated DESC's 5,049 basis modes for $2^{20}$ points at once (≈ 42 GB) | locate transient ≈ 5 GB. It had been the peak of every earlier ripple run but one, and caused an out-of-memory failure on 40 GB cards |
| 10 | **`--cg 1500 --no-rebuild-on-hit`** (§6.3) | each boundary step hit the 400 cap and triggered a 335 s rebuild buying ≈ 0.1% | removed the wasted rebuilds (≈ ¾ of wall time in the first α = 0.001 run); on its own it did **not** speed up the descent per minute, but the subspace search builds on it |

(NCSX is a stellarator used as an earlier test case, with three field periods; 48/32/18 denotes
L = 48, M = 32, N = 18.)

Two changes were tried and **did not** help in this regime:
- **Geodesic acceleration alone (0.79×).** The correction is too small at the step lengths the trust
  region allows.
- **The exact-Hessian model.** It was faster far from the bound but collapsed the radius near it
  (§5.2).

---

## 9. Recommended flags

A run for a tight, long-valley bound (64/64/6, α = 0.001):

```
python scripts/meanforce/mf_fixed.py --eq <desc.h5> --bext desc-fit --LMN 64 64 6 --N-phi 25 --frac 200 \
    --alphas 0.001 --max-outer 8 --maxiter 80 \
    --subspace prev,accel --cg 1500 --no-rebuild-on-hit \
    --f-noise auto --radius-by-delta --no-share-stop \
    --locate-chunk 131072 --snap-outer --mirror <dir> --out <dir>/<tag> [--start <state.npz>] [--time-budget S]
```

Such a run does not meet its bound (α = 0.001 is below the floor, §4.4). It ends on `--time-budget`
or `STOP`, normally inside the first outer iteration (80 inner steps take longer than the budgets
used). Setup takes about 15 min at 64/64/6 on an A100-80GB; the §7.4 arms had a 55 min budget and
used 38–44 min of it as solve time.

| flag | default | meaning | here |
|---|---|---|---|
| `--eq`, `--bext` | NCSX test file, coils | DESC equilibrium file; external field (§2.3) | ripple file, desc-fit |
| `--LMN L M N` | DESC's L, M, N × `--factors` (1, 1, 1) | resolution of the current basis | 64 64 6 |
| `--pad` | 0.02 m | wall margin of the field grid (§2.6) | default |
| `--frac` | 40 | field-grid spacing $h = a/\text{frac}$; keep $hL/a\lesssim0.4$ | 200 |
| `--N-phi` | from the arc spacing $h$ (odd); 1 if axisymmetric | field-grid planes per period; must be odd and resolve the toroidal modes of $\mathbf J$ (the default gives thousands at a/200) | 25 for N = 6 (convention $4N+1$) |
| `--alphas` | 0.7 | bound fractions, run in sequence (§4.3) | 0.001 |
| `--max-outer`, `--maxiter` | 25, 40 | outer iterations; inner steps per outer iteration | 8, 80 (long inner solves at a fixed merit) |
| `--rho0` | 1 | initial penalty μ (§4.2) | default |
| `--no-mult-ls` | off | disable the least-squares multiplier (§4.2) | off |
| `--outer-rebuild-factor` | 2 | metric keep test (§4.2) | default |
| `--gtol`, `--stat-tol`, `--stat-window` | $10^{-3}$, $10^{-3}$, 10 | inner and outer stationarity tests (§4.2, §5.5) | default |
| `--stall-stop` | none | end the sweep when the bound is unmet and $M/\varepsilon$ changed by less than this fraction over an outer iteration | |
| `--force-norm`, `--p`, `--p-band` | mean, 8, none | the bound's measure (§2.5) | mean |
| `--model`, `--beta` | beta, 0.5 | step model (§5.2) | default |
| `--precond` | force | metric (§6.1); `hj` = $\mathsf H_g$ only | default |
| `--lanczos-stop`, `--cg-tol` | tr, $10^{-2}$ | Lanczos stopping rule and tolerance (§5.4) | default |
| `--cg` | 400 | Lanczos iteration cap | 1500 (boundary steps need 400 to ≥1500; the cap was reached once) |
| `--no-rebuild-on-hit` | off | no inner rebuild after boundary steps (§6.3) | on |
| `--subspace` | off | exact subspace search (§7.3) | prev,accel |
| `--subspace-smax`, `--subspace-tmax` | 4, 2 | subspace grid extents | default |
| `--accel-amax`, `--accel-tol`, `--accel-maxiter` | 0.75, $10^{-4}$, 300 | acceleration solve (§7.2); also used by `--subspace accel` | default |
| `--geo-accel` | off | curved step on its own (§7.2) | **off** (do not combine with `--subspace …accel`) |
| `--f-noise`, `--radius-by-delta`, `--no-share-stop` | off | noise and radius rules (§5.5) | auto, on, on |
| `--delta0` | $10^{-3}$ | first trust radius and per-outer floor; used only with `--precond hj` (with the force metric the first radius is $\lVert L^{-1}\nabla\mathcal L_A\rVert$) | default |
| `--locate-chunk` | $2^{20}$ | points per Newton batch when locating grid nodes | 131072 |
| `--start file.npz` | — | warm start (below) | |
| `--start-LMN L M N` | — | the start is from a lower basis; matching $(l,m,n)$ coefficients are copied, new modes start at 0 | |
| `--snap-outer` | off | save a state after every outer iteration | on |
| `--mirror` | — | directory for live copies of logs and states (and the `CONTROL` file) | |
| `--out` | — | output prefix | |
| `--time-budget` | none | wall-time limit in seconds, counted from program start (including setup) | |
| `--eval-only` | off | evaluate given states (`--eval-state name=file.npz`) on this run's grids and exit; requires `--force-norm p` (any `--p`; the mean $M$ is reported as `mean/M_desc`) | for grid checks |

**Warm starts.** A state file holds $u$, the pressure coefficients $\mathbf c_p$ (stored for
analysis; they are recomputed from $u$), ν and μ. `--start` loads $u$ and takes
$\mu = \max(\text{saved}, \texttt{--rho0})$. It also loads ν, but with the least-squares multiplier on
(the default) ν is recomputed at the first outer iteration. To restart with a *small* μ, save a copy
of the file with only `u` and `c_p`. The trust radius is not stored; it is reset by the first metric
build.

**Control while running.** Write `STOP` (save the latest accepted state and exit) or `NEXT` (move to the next α)
into a file named `CONTROL` in the `--mirror` directory (otherwise next to `--out`). It is read after
each inner step, so it acts within one step, which can take several minutes.

**Grid checks.** Evaluate end states one grid finer (`--eval-only --force-norm p --frac <finer>`), and compare
different states only on a common grid.

**Memory.** About 33 GB peak at 64/64/6 and a/160–a/200. 80/80/6 a/200 needs more than an 80 GB
card with the current Cholesky.

---

## 10. Appendix: background methods, with references

**Least squares and Gauss–Newton.** For $\min\tfrac12\lVert\mathbf r(u)\rVert^2$, the Gauss–Newton
model replaces $\mathbf r(u+\mathbf d)$ by its linearization $\mathbf r + \mathsf K\mathbf d$. Its
Hessian $\mathsf K^T\mathsf K$ is positive semidefinite and needs only first derivatives. It is
accurate when the residual's second-order terms are small relative to $\mathsf K\mathbf d$. That
condition fails along the long valley steps of §7, which is the point of §7. (Nocedal & Wright,
ch. 10.)

**Trust-region methods.** Instead of choosing a step length along a direction (a line search), a
trust-region method minimizes the model within a region $\lVert\mathbf y\rVert\le\Delta$ where it is
trusted. It compares the actual and predicted decrease (the ratio), and grows or shrinks $\Delta$
accordingly. Global convergence follows if each step achieves at least a fraction of the decrease
along the steepest-descent direction (the *Cauchy decrease*), which GLTR's first iteration already
does; later GLTR iterates only lower the model further. That guarantee assumes a smooth objective
evaluated exactly and the textbook acceptance rules. NEST departs from each: $M$ has kinks where
some $\mathbf F_i = 0$, the model weights are floored (τ), the merit carries noise, noise-level steps
may raise it, subspace steps bypass the ratio test and radius-by-delta changes the radius rule (§5.5,
§7.3). These are heuristics outside the theory. (Nocedal & Wright, ch. 4; Conn, Gould & Toint, 2000.)

**Krylov methods and Lanczos.** For a symmetric $\mathsf H$ available only through products
$\mathsf H\mathbf v$, the Krylov space $\mathcal K_k = \text{span}\{\mathbf b,\mathsf H\mathbf b,\dots,\mathsf H^{k-1}\mathbf b\}$
contains good approximate solutions after few iterations when $\mathsf H$ is well conditioned.
Lanczos builds an orthonormal basis of $\mathcal K_k$ in which $\mathsf H$ is tridiagonal.
Conjugate gradients is the same process applied to a linear system. The eigenvalues of
$\mathsf T_k$ (*Ritz values*) approximate those of $\mathsf H$; their spread indicates how hard the
problem is. The logs print the extreme Ritz values of each subproblem (`theta [min, max]`), which is
how the near-singular case of §5.4 was found. (Golub & Van Loan, ch. 10.)

**Preconditioning by a Cholesky factor.** If $\mathsf A\approx LL^T$ with $L$ cheap to apply, working
in $\mathbf y = L^T\mathbf d$ replaces $\mathsf H$ by $L^{-1}\mathsf HL^{-T}\approx\mathsf I$, which
Krylov methods solve quickly. In a trust-region method the same change of variables makes the trust
region an ellipsoid adapted to the problem's scales.

**Augmented Lagrangian.** See §4.1, and Nocedal & Wright, ch. 17; Bertsekas, *Constrained
Optimization and Lagrange Multiplier Methods* (1982).

**References**
- F. Dudt and E. Kolemen, "DESC: A stellarator equilibrium solver", *Phys. Plasmas* **27**, 102513 (2020).
- R. A. James, "The solution of Poisson's equation for isolated source distributions", *J. Comput. Phys.* **25**, 71 (1977).
- J. J. Moré and D. C. Sorensen, "Computing a trust region step", *SIAM J. Sci. Stat. Comput.* **4**, 553 (1983).
- N. I. M. Gould, S. Lucidi, M. Roma and Ph. L. Toint, "Solving the trust-region subproblem using the Lanczos method", *SIAM J. Optim.* **9**, 504 (1999).
- M. K. Transtrum and J. P. Sethna, "Improvements to the Levenberg–Marquardt algorithm for nonlinear least-squares minimization", arXiv:1201.5885 (2012).
- J. Nocedal and S. J. Wright, *Numerical Optimization*, 2nd ed., Springer (2006).
- A. R. Conn, N. I. M. Gould and Ph. L. Toint, *Trust-Region Methods*, SIAM (2000).
- G. H. Golub and C. F. Van Loan, *Matrix Computations*, 4th ed., Johns Hopkins (2013).
- T. S. Hahm and R. M. Kulsrud, "Forced magnetic reconnection", *Phys. Fluids* **28**, 2412 (1985).
- M. J. D. Powell, "A method for nonlinear constraints in minimization problems", in *Optimization*, ed. R. Fletcher, Academic Press (1969); M. R. Hestenes, "Multiplier and gradient methods", *J. Optim. Theory Appl.* **4**, 303 (1969); R. T. Rockafellar, "A dual approach to solving nonlinear programming problems by unconstrained optimization", *Math. Programming* **5**, 354 (1973).
- D. R. Hunter and K. Lange, "A tutorial on MM algorithms", *Am. Stat.* **58**, 30 (2004).
- A. R. Karlin, M. S. Manasse, L. Rudolph and D. D. Sleator, "Competitive snoopy caching", *Algorithmica* **3**, 79 (1988).
- J. A. Nelder and R. Mead, "A simplex method for function minimization", *Comput. J.* **7**, 308 (1965).
- S. Sun and J. Nocedal, "A trust region method for noisy unconstrained optimization", *Math. Programming* **202**, 445 (2023).

---

## 11. Code map

| file | contents |
|---|---|
| `scripts/meanforce/mf_fixed.py` | driver: flags, setup, α sweep, metric refresh wiring, logging, eval-only, control file |
| `nest/meanforce_fixed.py` | `FixedAnchor` (state evaluation, pressure projection), `FixedRows`/`FixedRowsB` (§5.1–5.2), `FixedSolver` (§4), `lanczos_tr_step_H` and `_tridiag_tr_indef` (§5.4), `use_models` (inner rebuild rules, §6.3), lean metric builds (§6.2) |
| `nest/meanforce.py` | `Weights` (AL, §4.1), `Solver.inner` (acceptance and radius, §5.5), `_geodesic`, `_accel_dir` (§7.2), `_subspace`, `_subspace_M` (§7.3) |
| `nest/boundary_potential.py` | the representation $\mathbf J = \nabla\times\mathbf T$ and the fit $u_\text{fit}$ (§2.2) |
| `nest/pressure.py` | the pressure basis (§2.4) |
| `nest/cylindrical_bs.py`, `nest/annulus_freespace.py`, `nest/cylindrical_grid.py` | annulus Biot–Savart and node location (§2.6) |
| `tests/valley/smoke_valley.sh` | local CPU check of the valley options (exactness of the reconstruction; default path unchanged) |
| `results/ripple_landscape/valley/` | the §7.4 comparison (summary, per-step table, logs, end states) |
| `results/ripple_landscape/memprobe/` | memory measurements and projections (§6.2) |
