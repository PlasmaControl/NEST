"""Step 7: J, B, p and grad p at the collocation points as affine maps of the state.

The state is x = (u, x_p): u the free coefficients of the current potential
(``BoundaryPotential``, net toroidal current I fixed) and x_p the free coefficients of
the pressure (``Pressure.free_coefficients``). At fixed collocation points, with the
geometry of the fixed equilibrium,

    J(x)      = J_0 + D_J u            J_0 the net-current term (I / 2pi) h_0
    B(x)      = B_0 + D_B u            B_0 = B_ext + BS_h(J_0),  D_B = BS_h(D_J on the box)
    p(x)      = D_p x_p
    grad p(x) = D_gp x_p

exactly: J is linear in u, BS_h (``biot_savart.BiotSavart``) is linear, p is linear in
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

The collocation points are a DESC grid over one field period (default
QuadratureGrid(2 L, 2 M, 2 N) of the current basis, the 2x rule), with volume weights
dV = |sqrt(g)| w; B there is exact because the box holds the whole torus. Vectors are
Cartesian, shape (n_points, 3); the D's are (n_points, 3, n_columns).

D_B costs one Biot-Savart application per column (``progress`` lines by elapsed
time). With ``cache_dir`` its columns are saved in blocks, keyed by everything they
depend on (equilibrium geometry, box, kernel, points, basis and edge power), and a
build that stops (e.g. a job time limit) resumes from the saved blocks.
"""

import hashlib
import json
import os
import time

import jax
import jax.numpy as jnp
import numpy as np
from desc.grid import QuadratureGrid

from magnetic_field import BiotSavart
from grid import progress
from separable import Separable


def _geometry_chunk(RZ, rho, theta, zeta):
    """e_rho, e_theta, e_zeta (rows) and sqrt(g) at the points, (R, phi, Z) components."""
    X, Xr = RZ.evaluate(rho, theta, zeta), RZ.evaluate(rho, theta, zeta, dr=1)
    Xt, Xz = RZ.evaluate(rho, theta, zeta, dt=1), RZ.evaluate(rho, theta, zeta, dz=1)
    R, zero = X[:, 0], jnp.zeros_like(rho)
    E = jnp.stack([jnp.stack([Xr[:, 0], zero, Xr[:, 1]], -1),
                   jnp.stack([Xt[:, 0], zero, Xt[:, 1]], -1),
                   jnp.stack([Xz[:, 0], R, Xz[:, 1]], -1)], axis=1)
    sqrt_g = R * (Xt[:, 0] * Xr[:, 1] - Xr[:, 0] * Xt[:, 1])
    return E, sqrt_g, X


_geometry_jit = jax.jit(_geometry_chunk, static_argnums=0)


def geometry(eq, rtz, chunk=50000):
    """Geometry of the fixed equilibrium at flux coordinates ``rtz`` (n, 3).

    Returns a dict of NumPy arrays: E (n, 3, 3) the tangent basis e_rho, e_theta,
    e_zeta (rows) and G (n, 3, 3) the dual basis grad(rho), grad(theta), grad(zeta),
    both in Cartesian components; sqrt_g (n,); x (n, 3) the Cartesian positions;
    phi (n,) = zeta.
    """
    RZ = Separable([eq.R_basis, eq.Z_basis], [eq.R_lmn, eq.Z_lmn], eq.NFP)
    rtz = np.asarray(rtz, dtype=float)
    E, sg, X = [], [], []
    for i in range(0, len(rtz), chunk):
        n = min(chunk, len(rtz) - i)
        pad = np.pad(np.arange(i, i + n), (0, chunk - n), mode="edge") if len(rtz) > chunk \
            else np.arange(i, i + n)
        e, s, x = _geometry_jit(RZ, *(jnp.asarray(rtz[pad, j]) for j in range(3)))
        E.append(np.asarray(e)[:n]), sg.append(np.asarray(s)[:n]), X.append(np.asarray(x)[:n])
    E, sg, X = np.concatenate(E), np.concatenate(sg), np.concatenate(X)
    phi = rtz[:, 2]
    c, s = np.cos(phi), np.sin(phi)
    rot = np.zeros((len(phi), 3, 3))  # (R, phi, Z) -> (x, y, z)
    rot[:, 0, 0], rot[:, 0, 1], rot[:, 1, 0], rot[:, 1, 1], rot[:, 2, 2] = c, -s, s, c, 1
    E = np.einsum("nab,nib->nia", rot, E)
    G = np.stack([np.cross(E[:, 1], E[:, 2]), np.cross(E[:, 2], E[:, 0]),
                  np.cross(E[:, 0], E[:, 1])], axis=1) / sg[:, None, None]
    x = np.column_stack([X[:, 0] * c, X[:, 0] * s, X[:, 1]])
    return {"E": E, "G": G, "sqrt_g": sg, "x": x, "phi": phi}


@jax.jit
def _to_cartesian(sJ, E, sqrt_g):
    """(n, k, 3) sqrt(g) J^i and the basis E (n, 3, 3) -> Cartesian J, (n, 3, k)."""
    return jnp.einsum("nki,nia->nak", sJ, E) / sqrt_g[:, None, None]


class Fields:
    """J, B, p, grad p at collocation points as affine maps of x = (u, x_p).

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
    kernel, bs_kw : Biot-Savart kernel and other ``BiotSavart`` arguments.
    build_D_B : bool
        Build D_B (the Jacobian of B). Without it, B(x) needs ``B_direct`` (one
        Biot-Savart application): for an independent evaluation grid.
    col_chunk : int
        D_B columns per batch (memory of the batched Biot-Savart).
    cache_dir : str or None
        Save / resume D_B column blocks there (opt-in).
    verbose : bool
    """

    def __init__(self, inp, current, pressure, box, grid=None, kernel="vico_greengard",
                 bs_kw=None, build_D_B=True, col_chunk=4, cache_dir=None, chunk=20000,
                 verbose=True):
        if box.inside is None:
            raise RuntimeError("locate the plasma on the box first (box.locate_plasma)")
        eq = inp.eq
        self.inp, self.current, self.pressure, self.box = inp, current, pressure, box
        self.chunk, self.verbose = chunk, verbose
        b = current.basis
        self.grid = grid or QuadratureGrid(L=2 * b.L, M=2 * b.M, N=2 * b.N, NFP=eq.NFP)
        self.rtz = np.asarray(self.grid.nodes)
        geo = geometry(eq, self.rtz)
        self.x, self.phi, self.E, self.G, self.sqrt_g = (geo[k] for k in ("x", "phi", "E", "G",
                                                                          "sqrt_g"))
        self.dV = np.abs(self.sqrt_g) * np.asarray(self.grid.weights)
        self.n_u = current.n_free
        self.free_p, _ = pressure.free_coefficients()
        self.n_p = len(self.free_p)
        self.n = self.n_u + self.n_p
        self.bs_kw = dict(kernel=kernel, **(bs_kw or {}))
        self.bs = BiotSavart(box, self.x, **self.bs_kw)
        self._log(f"collocation: {len(self.x):,} points, {self.n_u} current + {self.n_p} pressure "
                  f"coefficients; box {box.shape}, {len(box.inside):,} inside points, "
                  f"kernel {kernel}")
        t0 = time.time()
        self.D_J = self._current_columns(self.rtz, self.E, self.sqrt_g)
        self.J_0 = self._net_current(self.rtz, self.E, self.sqrt_g)
        self.D_p, self.D_gp = self._pressure_columns()
        self._log(f"D_J, D_p, D_gp built ({time.time() - t0:.1f} s): "
                  f"{(self.D_J.nbytes + self.D_p.nbytes + self.D_gp.nbytes) / 1e9:.2f} GB")
        # the box: geometry (once) and the net-current term
        t0 = time.time()
        bg = geometry(eq, box.rtz)
        self.E_box, self.sqrt_g_box = jnp.asarray(bg["E"]), jnp.asarray(bg["sqrt_g"])
        self._log(f"box geometry at {len(box.inside):,} points ({time.time() - t0:.1f} s)")
        self.B_ext = self._external_field()
        J0_box = self._net_current(box.rtz, self.E_box, self.sqrt_g_box)
        self.B_0 = self.B_ext + np.asarray(self.bs.apply(J0_box))
        self.D_B = self._build_D_B(col_chunk, cache_dir) if build_D_B else None

    def _log(self, msg):
        if self.verbose:
            print(f"Fields: {msg}", flush=True)

    # ----------------------------------------------------------- columns
    def _current_columns(self, rtz, E, sqrt_g, cols=None):
        """D_J at the points ``rtz``, (n, 3, len(cols)), chunked over points."""
        cols = np.arange(self.n_u) if cols is None else np.asarray(cols)
        step = max(1, self.chunk * 8 // max(len(cols), 1))
        out = []
        for i in range(0, len(rtz), step):
            r = [jnp.asarray(rtz[i : i + step, j]) for j in range(3)]
            sJ = self.current.sqrtg_J_columns(cols, *r)
            out.append(np.asarray(_to_cartesian(sJ, jnp.asarray(E[i : i + step]),
                                                jnp.asarray(sqrt_g[i : i + step]))))
        return np.concatenate(out)

    def _net_current(self, rtz, E, sqrt_g):
        """J_0 = J(u = 0) (the net-current term), (n, 3)."""
        out = []
        for i in range(0, len(rtz), self.chunk):
            r = [jnp.asarray(rtz[i : i + self.chunk, j]) for j in range(3)]
            sJ = self.current.sqrtg_J(np.zeros(self.n_u), *r)[:, None, :]
            out.append(np.asarray(_to_cartesian(sJ, jnp.asarray(E[i : i + self.chunk]),
                                                jnp.asarray(sqrt_g[i : i + self.chunk])))[..., 0])
        return np.concatenate(out)

    def _pressure_columns(self):
        """D_p (n, n_p) and D_gp (n, 3, n_p)."""
        Dp, Dgp = [], []
        step = max(1, self.chunk * 8 // max(self.n_p, 1))
        for i in range(0, len(self.rtz), step):
            r = [jnp.asarray(self.rtz[i : i + step, j]) for j in range(3)]
            c = np.asarray(self.pressure.p_columns(np.arange(self.n_p), *r))  # (n, k, 4)
            Dp.append(c[:, :, 0])
            Dgp.append(np.einsum("nki,nia->nak", c[:, :, 1:], self.G[i : i + step]))
        return np.concatenate(Dp), np.concatenate(Dgp)

    def _external_field(self):
        f = self.inp.field
        if f is None:
            raise ValueError(f"{self.inp.name}: no external field (DESCInput field=...)")
        if hasattr(f, "at_flux"):  # DESCExternalField: the flux coordinates are known
            return np.asarray(f.at_flux(self.rtz, self.x))
        return self.inp.B_ext(self.x)

    def _cache_key(self):
        eq, b, box = self.inp.eq, self.current.basis, self.box
        h = hashlib.sha256()
        for a in (eq.R_lmn, eq.Z_lmn, np.asarray(box.axes[0]), np.asarray(box.axes[1]),
                  np.asarray(box.axes[2]), box.inside, self.rtz, b.modes,
                  self.current.chi_modes):
            h.update(np.ascontiguousarray(np.asarray(a)).tobytes())
        h.update(json.dumps([self.current.edge_power, b.NFP, sorted(
            (k, str(v)) for k, v in self.bs_kw.items())]).encode())
        return h.hexdigest()[:16]

    def _build_D_B(self, col_chunk, cache_dir):
        """D_B (n, 3, n_u): BS_h of the current of each coefficient on the box."""
        n_t = len(self.x)
        D_B = np.zeros((n_t, 3, self.n_u))
        done = np.zeros(self.n_u, bool)
        path = None
        if cache_dir is not None:
            path = os.path.join(cache_dir, f"D_B_{self._cache_key()}")
            os.makedirs(path, exist_ok=True)
            for f in sorted(os.listdir(path)):
                if f.startswith("cols_") and f.endswith(".npy"):
                    i0, i1 = (int(v) for v in f[5:-4].split("_"))
                    D_B[:, :, i0:i1] = np.load(os.path.join(path, f))
                    done[i0:i1] = True
            if done.any():
                self._log(f"D_B: {done.sum()} of {self.n_u} columns restored from {path}")
        C = self.bs.consts()
        apply = jax.jit(jax.vmap(lambda J, C: self.bs.apply(J, C), in_axes=(2, None),
                                 out_axes=2))
        rho, theta, zeta = (jnp.asarray(self.box.rtz[:, j]) for j in range(3))
        todo = [i for i in range(0, self.n_u, col_chunk) if not done[i:i + col_chunk].all()]
        t0, t_last, n_done = time.time(), time.time(), 0
        for k, i0 in enumerate(todo):
            i1 = min(i0 + col_chunk, self.n_u)
            sJ = self.current.sqrtg_J_columns(np.arange(i0, i1), rho, theta, zeta)
            J = _to_cartesian(sJ, self.E_box, self.sqrt_g_box)       # (n_inside, 3, k)
            D_B[:, :, i0:i1] = np.asarray(apply(J, C))
            if path is not None:
                np.save(os.path.join(path, f"cols_{i0}_{i1}.npy"), D_B[:, :, i0:i1])
            n_done += i1 - i0
            now = time.time()
            if self.verbose and (k == 0 or now - t_last > 30 or k + 1 == len(todo)):
                progress("D_B", k + 1, len(todo), n_done, sum(min(i + col_chunk, self.n_u) - i
                                                              for i in todo), t0,
                         f"{(now - t0) / n_done:.2f} s/column", unit="columns")
                t_last = now
        return D_B

    # ------------------------------------------------------------ fields
    def split(self, x):
        return x[: self.n_u], x[self.n_u :]

    def data(self):
        """The arrays the fields read, as a pytree of JAX arrays (pass as a jit argument)."""
        d = {k: jnp.asarray(getattr(self, k)) for k in ("D_J", "J_0", "D_p", "D_gp", "B_0")}
        d["D_B"] = None if self.D_B is None else jnp.asarray(self.D_B)
        d["dV"] = jnp.asarray(self.dV)
        return d

    def evaluate(self, x, data):
        """J, B, p, grad p at the points for state x: pure JAX, linear / affine in x."""
        u, xp = x[: self.n_u], x[self.n_u :]
        if data["D_B"] is None:
            raise RuntimeError("no D_B: use B_direct, or build_D_B=True")
        return {"J": data["J_0"] + data["D_J"] @ u, "B": data["B_0"] + data["D_B"] @ u,
                "p": data["D_p"] @ xp, "grad_p": data["D_gp"] @ xp}

    def J_box(self, u):
        """Cartesian J of coefficients u at the box points inside the plasma, (n_inside, 3)."""
        rtz, out = self.box.rtz, []
        for i in range(0, len(rtz), self.chunk):
            r = [jnp.asarray(rtz[i : i + self.chunk, j]) for j in range(3)]
            sJ = self.current.sqrtg_J(u, *r)[:, None, :]
            out.append(np.asarray(_to_cartesian(sJ, self.E_box[i : i + self.chunk],
                                                self.sqrt_g_box[i : i + self.chunk]))[..., 0])
        return np.concatenate(out)

    def B_direct(self, u):
        """B = B_ext + BS_h(J(u) on the box) at the points, without D_B, (n, 3)."""
        return self.B_ext + np.asarray(self.bs.apply(self.J_box(u)))

    def initial_state(self):
        """x of the current's and the pressure's own coefficients."""
        return np.concatenate([self.current.u, self.pressure.free_values()])

    # ---------------------------------------------------------- pressure fit
    def fit_pressure(self, x):
        """x with the pressure part replaced by the force-optimal one at the current of x:
        min over x_p of sum dV |J x B - grad p(x_p)|^2 (linear least squares), the
        pressure block of the Gauss-Newton problem."""
        u = np.asarray(x[: self.n_u])
        J = self.J_0 + self.D_J @ u
        B = (self.B_0 + self.D_B @ u) if self.D_B is not None else self.B_direct(u)
        w = np.sqrt(self.dV)[:, None]
        A = (w[:, :, None] * self.D_gp).reshape(-1, self.n_p)
        y = (w * np.cross(J, B)).ravel()
        s = np.linalg.norm(A, axis=0)
        s = np.where(s > 0, s, 1.0)
        xp = np.linalg.lstsq(A / s, y, rcond=None)[0] / s
        return np.concatenate([u, xp])
