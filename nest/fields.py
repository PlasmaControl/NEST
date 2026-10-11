"""Step 7 of NEST: J, B, p and grad p at the collocation points as affine maps of the state.

The state is x = (u, x_p): u the free coefficients of the current potential
(``BoundaryPotential``, net toroidal current I fixed) and x_p the free coefficients of
the pressure (``Pressure.free_coefficients``). At fixed collocation points, with the
geometry of the fixed equilibrium,

    J(x)      = J_0 + D_J u            J_0 the net-current term (I / 2pi) h_0
    B(x)      = B_0 + D_B u            B_0 = B_ext + BS_h(J_0),  D_B = BS_h(D_J on the box)
    p(x)      = D_p x_p
    grad p(x) = D_gp x_p

exactly: J is linear in u, BS_h (``magnetic_field.BiotSavart``) is linear, p is linear in
x_p. The D's are constant, so once they are built an objective made of these fields
needs no Biot-Savart call: the force residual J x B - grad p is quadratic in x and its
Jacobian is D_J x B + J x D_B - D_gp at any x.

Columns. Column j of D_J is the current of the single coefficient u_j, in closed form
(``boundary_potential._sqrtg_J_columns``): one radial and two Fourier factors per
column, then J = sum_i sqrt(g) J^i e_i / sqrt(g) with the geometry (e_rho, e_theta,
e_zeta, sqrt(g)) of the points computed once. Column j of D_B is BS_h of the same
current on the box points inside the plasma, again from the closed form with the box
geometry computed once: exact (no interpolation) and the J part costs little next to
the FFTs. D_p, D_gp likewise (``Pressure.p_columns``), with grad p = p_rho grad(rho) +
p_theta grad(theta) + p_zeta grad(zeta).

``Fields`` is a ``force.ForceResidual``: the collocation points are its DESC grid over
one field period (here by default QuadratureGrid(2 L, 2 M, 2 N) of the current basis,
the 2x rule), with its volume weights dV = |sqrt(g)| w, B_ext and normalization; B
there is exact because the box holds the whole torus. ``state(x, data)`` gives the
fields at x; ``objects(x)`` the series of x for ``ForceResidual.evaluate`` and the
rest. Vectors are Cartesian, shape (n_points, 3); the D's are (n_points, 3,
n_columns), JAX arrays stored once.

D_B costs one Biot-Savart application per column (``progress`` lines by elapsed
time). With ``cache_dir`` its columns are saved in blocks, keyed by everything they
depend on (equilibrium geometry, box, kernel, points, basis and edge power), and a
build that stops (e.g. a job time limit) resumes from the saved blocks.
"""

import functools
import hashlib
import json
import os
import time

import jax
import jax.numpy as jnp
import numpy as np
from desc.grid import QuadratureGrid

from cartesian_grid import evaluate_chunked
from desc_input import current_from_sqrtg
from force import ForceResidual
from grid import device_memory, progress
from pressure import Pressure


@functools.partial(jax.jit, donate_argnums=0)
def _put_columns(D, block, i0):
    """D[:, :, i0 : i0 + k] = block, in place (D is donated)."""
    return jax.lax.dynamic_update_slice(D, block.astype(D.dtype), (0, 0, i0))


def _columns(f, rtz, extra, n_cols, budget):
    """f(rho, theta, zeta, *extra) -> (n, ..., n_cols) on the points, in chunks of about
    ``budget / n_cols`` points: about ``budget`` values per component per chunk,
    whatever the number of columns."""
    return evaluate_chunked(f, rtz, max(1, budget // max(n_cols, 1)), extra=extra)


class Fields(ForceResidual):
    """J, B, p, grad p at the collocation points as affine maps of x = (u, x_p).

    A ``ForceResidual`` (the points, volume weights dV, B_ext, DESC's normalization,
    ``metrics``, ``decompose``, ``desc_fields``) with the constant matrices D_J, J_0,
    D_B, B_0, D_p, D_gp added; they are stored once, as JAX arrays.

    Parameters
    ----------
    inp : DESCInput
        Equilibrium and external field (``inp.field``: coils, a field file, or a
        ``DESCExternalField``).
    current : BoundaryPotential
    pressure : Pressure
    box : CartesianGrid
        Periodic box with the plasma located.
    grid : desc Grid, optional
        Collocation points over one field period; default QuadratureGrid(2 L, 2 M,
        2 N) of the current basis.
    kernel, bs_kw : Biot-Savart kernel and other ``BiotSavart`` arguments (float64).
    build_D_B : bool
        Build D_B (the Jacobian of B). Without it, B(x) needs ``B_direct`` (one
        Biot-Savart application): for an independent evaluation grid.
    col_chunk : int
        D_B columns per batch (memory of the batched Biot-Savart).
    cache_dir : str or None
        Save / resume D_B column blocks there (opt-in).
    chunk : int
        Points per call of the series evaluators (the column builds use chunks of
        8 chunk / n_columns points).
    verbose : bool
        Progress lines.

    With stellarator-symmetric bases (sym="cos", ``BoundaryPotential.fit(sym=True)``,
    ``Pressure.fit(sym=True)``) the points are the half theta in [0, pi] of the grid
    (``ForceResidual(sym=True)``): half the rows and half the columns.
    """

    def __init__(self, inp, current, pressure, box, grid=None, kernel="vico_greengard",
                 bs_kw=None, build_D_B=True, col_chunk=4, cache_dir=None, chunk=20000,
                 verbose=True):
        b = current.basis
        sym = b.sym == "cos"
        if (pressure.basis.sym == "cos") != sym:
            raise ValueError("current and pressure bases must both be symmetric or both not")
        grid = grid or QuadratureGrid(L=2 * b.L, M=2 * b.M, N=2 * b.N, NFP=inp.eq.NFP)
        super().__init__(inp, box, grid, eval_chunk=chunk, kernel=kernel, dtype=np.float64,
                         bs_kw=bs_kw, sym=sym)
        self.current, self.pressure = current, pressure
        self.chunk, self.verbose = chunk, verbose
        geo = inp.basis_vectors(self.rtz)
        self.E, self.G, self.sqrt_g = geo["E"], geo["G"], geo["sqrt_g"]
        self.n_u = current.n_free
        self.n_p = len(pressure.free_coefficients()[0])
        self.n = self.n_u + self.n_p
        self.bs_kw = dict(kernel=kernel, **(bs_kw or {}))
        self._log(f"collocation: {len(self.x):,} points, {self.n_u} current + {self.n_p} pressure "
                  f"coefficients; box {box.shape}, {len(box.inside):,} inside points, "
                  f"kernel {kernel}")
        t0 = time.time()
        self.D_J = self._current_columns()
        self.J_0 = jnp.asarray(self._J(self.rtz, self.E, self.sqrt_g))
        self.D_p, self.D_gp = self._pressure_columns()
        self._log(f"D_J, D_p, D_gp built ({time.time() - t0:.1f} s): "
                  f"{(self.D_J.nbytes + self.D_p.nbytes + self.D_gp.nbytes) / 1e9:.2f} GB")
        # the box: geometry (once) and the net-current term
        t0 = time.time()
        bg = inp.basis_vectors(box.rtz)
        self.E_box, self.sqrt_g_box = jnp.asarray(bg["E"]), jnp.asarray(bg["sqrt_g"])
        self._log(f"box geometry at {len(box.inside):,} points ({time.time() - t0:.1f} s)")
        J0_box = self._J(box.rtz, self.E_box, self.sqrt_g_box)
        self.B_0 = jnp.asarray(self.B_ext) + self.bs.apply(J0_box)
        self.D_B = self._build_D_B(col_chunk, cache_dir) if build_D_B else None

    def _log(self, msg):
        if self.verbose:
            print(f"Fields: {msg}", flush=True)

    # ----------------------------------------------------------- columns
    def _J_columns(self, cols, rho, theta, zeta, E, sqrt_g):
        """Cartesian J of the unit coefficients ``cols`` at points with tangent basis E
        and Jacobian sqrt_g, (n, 3, len(cols)): the closed-form sqrt(g) J^i
        (``BoundaryPotential.sqrtg_J_columns``), then J = sum_i sqrt(g) J^i e_i / sqrt(g)."""
        sJ = self.current.sqrtg_J_columns(cols, rho, theta, zeta)    # (n, k, 3)
        return jnp.swapaxes(current_from_sqrtg(sJ, E, sqrt_g), 1, 2)

    def _current_columns(self):
        """D_J at the collocation points, (n, 3, n_u)."""
        cols = np.arange(self.n_u)
        f = functools.partial(self._J_columns, cols)
        return jnp.asarray(_columns(f, self.rtz, (jnp.asarray(self.E), jnp.asarray(self.sqrt_g)),
                                    self.n_u, self.chunk * 8))

    def _J(self, rtz, E, sqrt_g, u=None):
        """Cartesian J(u) at the points, (n, 3); default u = 0, the net-current term J_0."""
        u = jnp.zeros(self.n_u) if u is None else jnp.asarray(u)

        def f(rho, theta, zeta, E, sqrt_g):
            return current_from_sqrtg(self.current.sqrtg_J(u, rho, theta, zeta), E, sqrt_g)
        return evaluate_chunked(f, rtz, self.chunk, extra=(jnp.asarray(E), jnp.asarray(sqrt_g)))

    def _pressure_columns(self):
        """D_p (n, n_p) and D_gp (n, 3, n_p): closed-form columns and the dual basis."""
        cols = np.arange(self.n_p)

        def f(rho, theta, zeta, G):
            c = self.pressure.p_columns(cols, rho, theta, zeta)            # (n, k, 4)
            return jnp.concatenate([c[:, None, :, 0], jnp.einsum("nki,nia->nak", c[..., 1:], G)],
                                   axis=1)
        out = jnp.asarray(_columns(f, self.rtz, (jnp.asarray(self.G),), self.n_p, self.chunk * 8))
        return out[:, 0], out[:, 1:]

    def _cache_key(self):
        eq, b, box = self.inp.eq, self.current.basis, self.box
        h = hashlib.sha256()
        for a in (eq.R_lmn, eq.Z_lmn, np.asarray(box.axes[0]), np.asarray(box.axes[1]),
                  np.asarray(box.axes[2]), box.inside, self.rtz, self.dV, b.modes,
                  self.current.chi_modes):
            h.update(np.ascontiguousarray(np.asarray(a)).tobytes())
        h.update(json.dumps([self.current.edge_power, b.NFP, sorted(
            (k, str(v)) for k, v in self.bs_kw.items())]).encode())
        return h.hexdigest()[:16]

    def _build_D_B(self, col_chunk, cache_dir):
        """D_B (n, 3, n_u): BS_h of the current of each coefficient on the box."""
        n_t = len(self.x)
        D_B = jnp.zeros((n_t, 3, self.n_u))   # one copy, updated in place (donated)
        done = np.zeros(self.n_u, bool)
        path = None
        if cache_dir is not None:
            path = os.path.join(cache_dir, f"D_B_{self._cache_key()}")
            os.makedirs(path, exist_ok=True)
            for f in sorted(os.listdir(path)):
                if f.startswith("cols_") and f.endswith(".npy"):
                    i0, i1 = (int(v) for v in f[5:-4].split("_"))
                    D_B = _put_columns(D_B, jnp.asarray(np.load(os.path.join(path, f))), i0)
                    done[i0:i1] = True
            if done.any():
                self._log(f"D_B: {done.sum()} of {self.n_u} columns restored from {path}")
        C = self.bs.consts()
        apply = jax.jit(jax.vmap(lambda J, C: self.bs.apply(J, C), in_axes=(2, None),
                                 out_axes=2))
        rtz_box = [jnp.asarray(self.box.rtz[:, j]) for j in range(3)]
        todo = [i for i in range(0, self.n_u, col_chunk) if not done[i:i + col_chunk].all()]
        n_todo = sum(min(i + col_chunk, self.n_u) - i for i in todo)
        t0, t_last, n_done = time.time(), time.time(), 0
        for k, i0 in enumerate(todo):
            i1 = min(i0 + col_chunk, self.n_u)
            J = self._J_columns(np.arange(i0, i1), *rtz_box, self.E_box, self.sqrt_g_box)
            block = apply(J, C)
            D_B = _put_columns(D_B, block, i0)
            if path is not None:
                np.save(os.path.join(path, f"cols_{i0}_{i1}.npy"), np.asarray(block))
            n_done += i1 - i0
            now = time.time()
            if self.verbose and (k == 0 or now - t_last > 30 or k + 1 == len(todo)):
                progress("D_B", k + 1, len(todo), n_done, n_todo, t0,
                         f"{(now - t0) / n_done:.2f} s/column, memory {device_memory()}",
                         unit="columns")
                t_last = now
        return D_B

    # ------------------------------------------------------------ fields
    def split(self, x):
        """x -> (u, x_p): the current and the pressure coefficients."""
        return x[: self.n_u], x[self.n_u :]

    def data(self):
        """The arrays the fields read, as a pytree of JAX arrays (pass as a jit argument);
        the stored arrays themselves, not copies. Every one has the points along the
        first axis."""
        return {k: getattr(self, k) for k in ("D_J", "J_0", "D_p", "D_gp", "B_0", "D_B")}

    def state(self, x, data):
        """J, B, p, grad p at the points for state x: pure JAX, linear / affine in x."""
        u, xp = self.split(x)
        if data["D_B"] is None:
            raise RuntimeError("no D_B: use B_direct, or build_D_B=True")
        return {"J": data["J_0"] + data["D_J"] @ u, "B": data["B_0"] + data["D_B"] @ u,
                "p": data["D_p"] @ xp, "grad_p": data["D_gp"] @ xp}

    def J_box(self, u):
        """Cartesian J of coefficients u at the box points inside the plasma, (n_inside, 3)."""
        return self._J(self.box.rtz, self.E_box, self.sqrt_g_box, u)

    def B_direct(self, u):
        """B = B_ext + BS_h(J(u) on the box) at the points, without D_B, (n, 3)."""
        return self.B_ext + np.asarray(self.bs.apply(self.J_box(u)))

    def initial_state(self):
        """x of the current's and the pressure's own coefficients."""
        return np.concatenate([self.current.u, self.pressure.free_values()])

    def objects(self, x):
        """The ``BoundaryPotential`` and ``Pressure`` of state x (for ``ForceResidual``
        methods, saving, ...)."""
        u, xp = (np.asarray(v) for v in self.split(x))
        p = self.pressure
        return (self.current.with_u(u),
                Pressure(p.basis, np.asarray(p.to_coefficients(xp)), p.eq, p.edge_order))

    def report(self, x, data=None):
        """``ForceResidual.metrics`` of F = J x B - grad p at state x (DESC's measure)."""
        f = self.state(jnp.asarray(x), self.data() if data is None else data)
        return self.metrics(np.asarray(jnp.cross(f["J"], f["B"]) - f["grad_p"]))

    # ---------------------------------------------------------- pressure fit
    def fit_pressure(self, x):
        """x with the pressure part replaced by the force-optimal one at the current of x
        (``Pressure.fit_force`` with J x B of x at the collocation points)."""
        u = jnp.asarray(self.split(x)[0])
        J = self.J_0 + self.D_J @ u
        B = (self.B_0 + self.D_B @ u) if self.D_B is not None else jnp.asarray(self.B_direct(u))
        p = self.pressure.fit_force(self.rtz, self.dV, jnp.cross(J, B), self.G)
        return np.concatenate([np.asarray(u), p.free_values()])
