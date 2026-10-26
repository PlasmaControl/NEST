"""Exact dense Levenberg-Marquardt for a 3-D NEST state on an axisymmetric equilibrium,
from 2-D tables of the toroidal harmonics.

The flux coordinates and metric of the equilibrium are axisymmetric (zeta = phi), so
every linear map of the model keeps the toroidal harmonic of a coefficient: a basis
function of toroidal mode n (cos or sin type) gives, in cylindrical components,

    V(rho, theta, zeta) = V^c(rho, theta) cos(n zeta) + V^s(rho, theta) sin(n zeta)

for V = J (u -> J), B_plasma (u -> annulus solve -> points), p and grad p (c_p ->).
V^c = V(zeta = 0) and V^s = V(zeta = pi / 2n) are tabulated once per column on the
(rho, theta) nodes of the collocation grid; the fields at any zeta and the exact
Jacobian of the residual on the 3-D collocation grid then follow by synthesis, with no
further box solve. B_plasma columns of harmonic n come from the annulus solve on
N_phi = 2n + 1 planes (exact mode by mode for that harmonic, ``CylindricalGrid.around``).

Stellarator symmetry halves the tables: every column is symmetric, so with theta_j =
2 pi j / n_theta (n_theta odd) the values at theta_(n_theta - j) are those at theta_j
times a sign per component and per cos/sin plane (``_PARITY``): only j <= n_theta // 2
is stored.

The objective (``HarmonicModel``), with F = J x B - grad p, B = B_coils + eps B_pert +
B_plasma, all in cylindrical components:
  force   sqrt(dV/V) F / norm at every collocation point (3 per point);
  J       w_J sqrt(dV_2D/V) (<J>_zeta - J_DESC) / rms J_DESC for rho < rho_max: the
          toroidal average only (the n = 0 harmonic), so the axisymmetric problem keeps
          the regularization of the 2-D runs and the 3-D response is not pulled to 0;
  W       w_W (<p>_vol - W_t) / W_t (W_t: the pressure fit's, as in the 2-D runs);
  p<0     w_pos sqrt(dV/V) min(p, 0) / rms p_DESC.
For an axisymmetric x it equals the 2-D runs' objective exactly.

Each LM iteration streams the tables through the device in chunks of radii: the rows
of the exact Jacobian [A | r] of a chunk are formed there and accumulated into the Gram
matrix H = A^T A and g = A^T r (float64). The step for a damping mu is the Cholesky
solve of (D H D + mu I) y = -D g, d = D y (D: Jacobi scaling, diag(D H D) = 1), for
mu over decades; every trial step is then evaluated exactly (the fields are affine in
x, so all trials take one more pass) and the best kept.
"""

import functools
import hashlib
import inspect
import os
import sys
import time

import jax
import jax.numpy as jnp
import numpy as np

import fit_cache
from flux_interp import TensorCurrent, TensorGeometry, TensorGrid, TensorPressure
from separable import Separable

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
# theta -> -theta (with zeta -> -zeta) of a symmetric column: the sign of the cos part per
# component; the sin part has the opposite sign. J, B: (-, +, +); p: +; grad p: (+, -, -)
_PARITY = {"u": np.array([-1.0, 1.0, 1.0, -1.0, 1.0, 1.0]),  # J (3), B_plasma (3)
           "p": np.array([1.0, 1.0, -1.0, -1.0])}             # p, grad p (3)


def harmonics(cur, pre):
    """|n| of every unknown x = (u, c_p): u = (chi, a, b), c_p (in modes' order)."""
    S = cur.S
    nu = np.concatenate([np.abs(S.chi_modes[:, 1]), np.abs(cur.basis.modes[:, 2]),
                         np.abs(cur.basis.modes[:, 2])])
    return nu.astype(int), np.abs(pre.basis.modes[:, 2]).astype(int)


def embed(x2, cur2, pre2, cur3, pre3):
    """The coefficients x2 of a run with smaller bases (e.g. N = 0 or 1) in the larger ones (the
    modes it lacks set to 0), matched mode by mode."""
    S2, S3 = cur2.S, cur3.S
    out = np.zeros(cur3.n_free + pre3.basis.num_modes)
    x2 = np.asarray(x2)
    chi3 = {tuple(m): i for i, m in enumerate(S3.chi_modes)}
    ab3 = {tuple(m): i for i, m in enumerate(cur3.basis.modes)}
    p3 = {tuple(m): i for i, m in enumerate(pre3.basis.modes)}
    for i, m in enumerate(S2.chi_modes):
        out[chi3[tuple(m)]] = x2[i]
    for i, m in enumerate(cur2.basis.modes):
        out[S3.n_chi + ab3[tuple(m)]] = x2[S2.n_chi + i]
        out[S3.n_chi + S3.K + ab3[tuple(m)]] = x2[S2.n_chi + S2.K + i]
    for i, m in enumerate(pre2.basis.modes):
        out[cur3.n_free + p3[tuple(m)]] = x2[cur2.n_free + i]
    return out


def to_cyl(V, phi):
    """Cartesian (..., 3) -> (R, phi, Z) components at angles phi (broadcast)."""
    c, s = np.cos(phi), np.sin(phi)
    return np.stack([V[..., 0] * c + V[..., 1] * s, -V[..., 0] * s + V[..., 1] * c, V[..., 2]], -1)


def _src_hash(*mods):
    return hashlib.sha256("".join(inspect.getsource(m) for m in mods).encode()).hexdigest()[:16]


def _cache_dir():
    return os.environ.get("NEST_CACHE", os.path.join(ROOT, ".cache"))


def _cached_points(name, key_parts, fn):
    """fn() stored as .cache/<name>_<hash>.npy."""
    where = _cache_dir()
    h = hashlib.sha256(repr(key_parts).encode()).hexdigest()[:24]
    path = os.path.join(where, f"{name}_{h}.npy")
    if os.path.exists(path):
        return np.load(path)
    v = np.asarray(fn())
    os.makedirs(where, exist_ok=True)
    tmp = path + f".{os.getpid()}.tmp.npy"
    np.save(tmp, v)
    os.replace(tmp, path)
    return v


class _PlanePoints:
    """The annulus' modal field (``CylindricalSolver.field_modes``, a batch of columns) at
    fixed (R, Z) points on a few planes phi_p: exact Fourier synthesis in phi, then the
    barycentric Chebyshev interpolation in (R, Z) as dense matrix products with the rows
    I_r (points x n_r) and I_z (points x n_z) formed once (``CylindricalInterpolator``
    forms them again for every call and every column). Same values."""

    def __init__(self, box, R, Z, planes, chunk=4096):
        from cylindrical_grid import _bary_rows
        N, NFP = box.N_phi, box.NFP
        npts = len(R)
        self.npts, self.chunk = npts, chunk
        pad = -npts % chunk
        R = np.concatenate([R, np.full(pad, R[-1])])
        Z = np.concatenate([Z, np.full(pad, Z[-1])])
        ra, za = box.r_axis, box.z_axis
        Ir = _bary_rows(jnp.asarray(ra.to_y(R)), jnp.asarray(ra.y), jnp.asarray(ra.w_bary))
        Iz = _bary_rows(jnp.asarray(za.to_y(Z)), jnp.asarray(za.y), jnp.asarray(za.w_bary))
        nc = len(R) // chunk
        self.Ir, self.Iz = Ir.reshape(nc, chunk, -1), Iz.reshape(nc, chunk, -1)
        jp = np.fft.fftfreq(N, 1.0 / N)
        jz = np.arange(N // 2 + 1)
        wz = np.where(jz == 0, 1.0, 2.0)
        self.Ep = jnp.asarray(np.exp(1j * np.outer(planes, jp * NFP)) / N)
        self.Ez = jnp.asarray(wz * np.exp(1j * np.outer(planes, jz * NFP)) / N)

    def __call__(self, Bp, Bz):
        """(k, P, points, 3) cylindrical components from slabs Bp (k, N, n_r, n_z), Bz (k, N//2+1, ...)."""
        return _plane_points(self.Ir, self.Iz, self.Ep, self.Ez, Bp, Bz)[:, :, : self.npts]


@jax.jit
def _plane_points(Ir, Iz, Ep, Ez, Bp, Bz):
    Pc = jnp.einsum("pj,kjab->kpab", Ep, Bp)
    Pz = jnp.real(jnp.einsum("pj,kjab->kpab", Ez, Bz))

    def one(args):
        ir, iz = args  # (chunk, n_r), (chunk, n_z)
        cr = jnp.einsum("kpib,ib->kpi", jnp.einsum("ia,kpab->kpib", ir, Pc.real), iz)
        ci = jnp.einsum("kpib,ib->kpi", jnp.einsum("ia,kpab->kpib", ir, Pc.imag), iz)
        cz = jnp.einsum("kpib,ib->kpi", jnp.einsum("ia,kpab->kpib", ir, Pz), iz)
        return jnp.stack([cr, ci, cz], -1)

    out = jax.lax.map(one, (Ir, Iz))  # (n_chunks, k, P, chunk, 3)
    return jnp.moveaxis(out, 0, 2).reshape(out.shape[1], out.shape[2], -1, 3)


@jax.jit
def _qr_r(M):
    return jnp.linalg.qr(M, mode="r")


@jax.jit
def _damped_qr(RD, c, sm):
    """y minimizing |R D y + c|^2 + mu |y|^2 by the QR of [R D, c; sqrt(mu) I, 0], and the factor."""
    n = RD.shape[0]
    M = jnp.concatenate([jnp.concatenate([RD, c[:, None]], 1),
                         jnp.concatenate([sm * jnp.eye(n, dtype=RD.dtype), jnp.zeros((n, 1), RD.dtype)], 1)], 0)
    Rt = jnp.linalg.qr(M, mode="r")
    y = -jax.scipy.linalg.solve_triangular(Rt[:n, :n], Rt[:n, n], lower=False)
    return y, Rt[:n, :n]


@functools.partial(jax.jit, donate_argnums=(0,))
def _add_outer(H, a):
    return H + a[:, None] * a[None, :]


@functools.partial(jax.jit, donate_argnums=(0,))
def _scale(H, d):
    return d[:, None] * H * d[None, :]


def _device_bytes():
    try:
        return float(jax.devices()[0].memory_stats()["bytes_limit"])
    except Exception:  # noqa: BLE001
        return 40e9


@jax.jit
def _damped_step(Hs, mu, gs):
    """y = -(Hs + mu I)^-1 gs by Cholesky, and whether it is finite."""
    i = jnp.arange(Hs.shape[0])
    L = jnp.linalg.cholesky(Hs.at[i, i].add(mu))
    y = jax.scipy.linalg.cho_solve((L, True), -gs)
    return y, jnp.all(jnp.isfinite(y))


class _Scale:
    """The variables of the damped step: z-step d = S y, damping mu |y|^2.
    Jacobi (default): S = diag(D). Physical metric (speed phys): S = L^-T with L L^T = M, the Gram
    matrix of the columns' (J / rms J, p / rms p) at the collocation points, so that |y| is the
    volume-weighted rms change of J and p. M is fixed by the tables (no field solves); it removes the
    radial collinearity of the basis that a diagonal scaling cannot (2026-10-23, lm-trust)."""

    def __init__(self, D=None, L=None):
        self.D, self.L = D, L

    def cols(self, R):  # R S
        if self.L is None:
            return R * self.D[None, :]
        return jax.scipy.linalg.solve_triangular(self.L, R.T, lower=True).T

    def back(self, y):  # S y
        if self.L is None:
            return self.D * y
        return jax.scipy.linalg.solve_triangular(self.L, y, lower=True, trans="T")

    def grad(self, g):  # S^T g
        if self.L is None:
            return self.D * g
        return jax.scipy.linalg.solve_triangular(self.L, g, lower=True)

    def inv(self, d):  # S^-1 d
        if self.L is None:
            return d / self.D
        return self.L.T @ d


class HarmonicModel:
    """Tables, constants and the dense LM (module docstring).

    Parameters
    ----------
    inp : DESCInput (axisymmetric equilibrium, with its coils)
    cur, pre : BoundaryPotential, Pressure with N >= 1 (3-D bases)
    fr : ForceResidual of the 3-D collocation grid (box None)
    frac, margin : the annulus spacing h = a / frac and its margin
    pert : callable, Cartesian points -> B_pert per unit eps (Cartesian), or None
    weights : dict with "J", "J_rho", "W", "pos"
    """

    def __init__(self, inp, cur, pre, fr, frac, margin=2.0, pert=None, weights=None,
                 chunk=2, batch_u=(4, 2), verbose=True, only=None, j_target=None, j_nodes="direct",
                 j_mode="full", transport=None, rough=None, par=None, moments=None, speed=None):
        t0 = time.time()
        self.inp, self.cur, self.pre, self.fr = inp, cur, pre, fr
        self.frac, self.margin, self.chunk, self.verbose = frac, margin, int(chunk), verbose
        w = dict({"J": 1e-3, "J_rho": 0.9, "W": 10.0, "pos": 10.0}, **(weights or {}))
        self.weights = w
        # the J term: None -> its toroidal average against J_DESC (rho < J_rho); an array (n_rho,
        # n_theta, 3) of cylindrical J (axisymmetric, e.g. J of the start) -> the full J against it
        # at every collocation point (the 2-D runs' CurrentError with j_target "start")
        self.j_nodes = j_nodes
        # j_mode "n0": the J term on the toroidal average only, against j_target (or J_DESC)
        self.j_full = j_target is not None and j_mode == "full"
        # pressure transport (location-free flattening in islands): weak form of
        # div(kappa_par grad_par p + grad_perp p) = S on the pressure basis, S from the eps = 0 state
        self.transport = None if transport is None else dict({"kappa": 1e4, "w": 1.0}, **transport)
        self._kappa = 1.0 if self.transport is None else float(self.transport["kappa"])
        # weight of the force component along B (= B . grad p, the equation that makes the surfaces):
        # rows swF (F + (par - 1) b b.F), b = B/|B| of the iterate (its derivative is of the order of F)
        self._par = 1.0 if par is None else float(par)
        # solver speed options (each independent): chol, narrow, newton, gpus=G, reuse=k, fp32, profile
        self.speed = dict(speed or {})
        self._kd_last, self._chord, self._steps, self._gn_last = None, None, "qr", float("nan")
        self._Delta, self._mu_tr = None, None
        self._tcache, self._loaded = {}, False  # speed resident: the tables stay on the device(s)
        # global moments of J_phi (objective.CurrentMoments: I_p, R_geo, Z_geo, V_eff / V) held at their
        # targets (the start's): 4 rows w (m - target) / scale, weighted sums over all collocation points
        self.mom = None if moments is None else dict({"w": 1.0, "which": ("I", "R", "Z", "V")}, **moments)
        # smoothness of the n >= 1 current: rows w (l / L)^q |J_k| / rms J_DESC u_k (no location)
        self.rough = None if rough is None else dict({"w": 1e-3, "q": 2}, **rough)
        self._j_target = None if j_target is None else np.asarray(j_target, dtype=float)
        eq = inp.eq
        assert int(eq.N) == 0, "the equilibrium must be axisymmetric"
        self.NFP = int(eq.NFP)
        g = fr.tgrid
        self.rho, self.theta, self.zeta = np.asarray(g.rho), np.asarray(g.theta), np.asarray(g.zeta)
        n_r, n_t, n_z = len(self.rho), len(self.theta), len(self.zeta)
        assert n_t % 2 == 1 and fr.sym, "needs stellarator symmetry and an odd theta grid"
        self.n_th = n_t // 2 + 1  # stored theta: j = 0 .. n_t // 2
        # the full theta from the stored half: source index and mirror flag
        j = np.arange(n_t)
        self.src = np.where(j < self.n_th, j, n_t - j)
        self.mirror = (j >= self.n_th).astype(float)
        self.n_u, self.n_p = cur.n_free, pre.basis.num_modes
        self.n = self.n_u + self.n_p
        nu, npp = harmonics(cur, pre)
        self.N = int(max(nu.max(), npp.max()))
        # blocks (kind, n, global indices); the internal order z = x[perm] puts them in a row
        self.blocks = []
        for kind, nh, off in (("u", nu, 0), ("p", npp, self.n_u)):
            for n in range(self.N + 1):
                idx = off + np.flatnonzero(nh == n)
                if len(idx):
                    self.blocks.append({"kind": kind, "n": n, "idx": idx})
        self.perm = np.concatenate([b["idx"] for b in self.blocks])
        self.iperm = np.argsort(self.perm)
        o = 0
        for b in self.blocks:
            b["sl"] = slice(o, o + len(b["idx"]))
            b["P"] = 1 if b["n"] == 0 else 2
            b["planes"] = np.array([0.0]) if b["n"] == 0 else np.array([0.0, np.pi / (2 * b["n"] * self.NFP)])
            # synthesis weights at the collocation zetas: (n_z, P)
            b["Wz"] = (np.ones((n_z, 1)) if b["n"] == 0 else
                       np.stack([np.cos(b["n"] * self.NFP * self.zeta), np.sin(b["n"] * self.NFP * self.zeta)], 1))
            o += len(b["idx"])
        self._log(f"harmonic model: N = {self.N}, collocation {g.shape}, {self.n:,} unknowns in blocks "
                  + ", ".join(f"{b['kind']}{b['n']} {len(b['idx'])}" for b in self.blocks))
        if only:  # build (and cache) these tables only, e.g. in separate jobs
            for b in self.blocks:
                if f"{b['kind']}{b['n']}" in only:
                    self._table(b, batch_u)
            self.tables = None
            return
        self._constants(pert)
        self._log(f"  constants: {time.time() - t0:.0f} s")
        self.tables = [self._table(b, batch_u) for b in self.blocks]
        self._log(f"  tables: {sum(t.nbytes for t in self.tables) / 1e9:.1f} GB on the host; "
                  f"model ready, {time.time() - t0:.0f} s")
        self._compile()

    # ------------------------------------------------------------------ setup
    def _log(self, s):
        if self.verbose:
            print(s, flush=True)

    def _key(self, *extra):
        eq, b, pb = self.inp.eq, self.cur.basis, self.pre.basis
        return (fit_cache._file_hash(self.inp.path), (b.L, b.M, b.N, self.cur.edge_power, str(b.sym)),
                (pb.L, pb.M, pb.N, self.pre.edge_order), tuple(np.round(self.rho, 14)), len(self.theta),
                float(self.cur.I)) + extra

    def _constants(self, pert):
        fr = self.fr
        shape3 = fr.tgrid.shape
        phi = np.asarray(fr.phi)
        dV = np.asarray(fr.dV).reshape(shape3)
        w3 = dV / dV.sum()
        self.sw3 = np.sqrt(w3)
        self.sw2 = np.sqrt(w3.sum(2))
        self.norm = float(fr.norm)
        self._Vtot = float(dV.sum())
        self._geoR = np.asarray(fr.geo.R).reshape(shape3)
        self._geoZ = np.asarray(fr.geo.Z).reshape(shape3)
        d = fr.desc_fields()
        w = w3.ravel()
        self.rms = {k: float(np.sqrt(np.sum(w * np.sum(np.reshape(d[k], (len(w), -1)) ** 2, 1))))
                    for k in ("J", "B", "p")}
        # DESC's J (axisymmetric: any zeta) and B, cylindrical
        self.JD = to_cyl(np.asarray(d["J"]), phi).reshape(shape3 + (3,))[:, :, 0]
        self.BD = to_cyl(np.asarray(d["B"]), phi).reshape(shape3 + (3,))
        key = self._key(tuple(np.round(self.zeta, 14)))
        fld = self.inp.field
        fkey = hashlib.sha256(repr((type(fld).__name__, float(getattr(fld, "G", 0.0)), float(getattr(fld, "I", 0.0)),
                                    np.round(np.asarray(getattr(fld, "Phi_mn", [])), 12).tolist(),
                                    np.round(np.asarray(getattr(fld, "R_lmn", [])), 12).tolist(),
                                    np.round(np.asarray(getattr(fld, "Z_lmn", [])), 12).tolist())).encode()).hexdigest()
        key = key + (fkey,)
        Bc = _cached_points("HarmBcoil", key, lambda: fr.B_ext)
        self.B_coil = to_cyl(Bc, phi).reshape(shape3 + (3,))
        self.B_pert = np.zeros_like(self.B_coil)
        if pert is not None:
            Bp = _cached_points("HarmBpert", key + (repr(pert),), lambda: pert(fr.x))
            self.B_pert = to_cyl(Bp, phi).reshape(shape3 + (3,))
        # the net-current parts (axisymmetric): J_I on the grid, B_I from the n = 0 annulus
        g2 = TensorGrid(self.rho, self.theta, np.array([0.0]), self.NFP)
        tc = TensorCurrent(self.cur.S, self.cur._RZ, g2, factored=True)
        self.J_I = np.asarray(tc(jnp.zeros(self.n_u), self.cur.I))[:, :, 0]
        import annulus_freespace, cylindrical_bs, cylindrical_grid, objective
        def B_I():
            fields0, pts0 = self._annulus(0, full_theta=True)
            Bp, Bz = fields0.solver.field_modes(fields0.J_box(jnp.zeros(self.n_u), self.cur.I))
            return np.asarray(_PlanePoints(pts0.box, pts0.R, pts0.Z, np.array([0.0]))(Bp[None], Bz[None]))[0, 0]
        import node_current
        self.B_I = _cached_points("HarmBI", self._key(self.frac, self.margin, self.j_nodes, "S64", _src_hash(
            annulus_freespace, cylindrical_bs, cylindrical_grid, objective, node_current)), B_I).reshape(
            len(self.rho), len(self.theta), 3)
        # W's target: <p> of the pressure fit (as the 2-D runs' PressureEnergy)
        tp = TensorPressure(Separable([self.pre.basis], [self.pre.c], self.NFP), self.pre.edge_order,
                            self.cur._RZ, fr.tgrid, fr.geo, factored=True)
        p_fit = np.asarray(tp(jnp.asarray(self.pre.c))[0])
        self.W_t = float(np.sum(w3 * p_fit))
        wJ = self.weights["J"]
        mask = (self.rho < self.weights["J_rho"]) if self.weights["J_rho"] is not None else np.ones(len(self.rho), bool)
        if self.j_full:
            self.swJ = wJ * self.sw3 * mask[:, None, None] / self.rms["J"]
            self.JD = self._j_target[:, :, None, :]
        else:
            self.swJ = wJ * self.sw2 * mask[:, None] / self.rms["J"]
            if self._j_target is not None:  # the toroidal average against the target (e.g. J of the start)
                self.JD = self._j_target
        self.swF = self.sw3 / self.norm
        self.swP = self.weights["pos"] * self.sw3 / self.rms["p"]
        self.wW = self.weights["W"] / self.W_t
        jax.clear_caches()

    def _annulus(self, n, full_theta=False):
        """Fields on the annulus of N_phi = 2n + 1 planes (only its box side is used) and
        the interpolator to the table points (rho, theta stored or all, plane zetas)."""
        from boxes import make_box
        from cylindrical_grid import CylindricalInterpolator
        from force import ForceResidual
        from objective import Fields
        b = self.cur.basis
        N_phi = 2 * n + 1
        box = make_box(self.inp, self.frac, "cylindrical", True, self.margin, verbose=self.verbose, N_phi=N_phi)
        res = (2 * b.L, 2 * b.M, 2 * max(b.N, 1))
        frs = ForceResidual(self.inp, box, resolution=res, rho_slice=slice(0, 1))
        fg = TensorGrid.uniform(4 * b.L, 2 * (3 * b.M), 4 * N_phi, self.NFP, sym=True)  # zeta hits the planes
        fields = Fields(frs, self.cur, self.pre, kernel="cylindrical", dtype=jnp.float64, flux_grid=fg,
                        S_dtype=jnp.float64,
                        factored=True)
        if self.j_nodes == "direct":  # J at the annulus nodes from the basis (node_current), as the 2-D runs
            from node_current import NodeFields
            fields = NodeFields.from_fields(fields, self.cur, box)
        theta = self.theta if full_theta else self.theta[: self.n_th]
        g = TensorGrid(self.rho, theta, np.array([0.0]), self.NFP)  # R, Z: the same on every plane
        geo = TensorGeometry(self.cur._RZ, g)

        class Points:
            pass
        pts = Points()
        pts.box, pts.R, pts.Z = box, np.asarray(geo.R).ravel(), np.asarray(geo.Z).ravel()
        return fields, pts

    def _table(self, blk, batch_u):
        """(n_r, n_th, comps, P, cols) of a block: u: J (3) and B_plasma (3); p: p and grad p."""
        n, kind = blk["n"], blk["kind"]
        cols = blk["idx"] - (0 if kind == "u" else self.n_u)
        P, n_r, n_th = blk["P"], len(self.rho), self.n_th
        import annulus_freespace, cylindrical_bs, cylindrical_grid, objective, boundary_potential, pressure
        import node_current
        # keyed by the block's own functions (its modes in column order), not the whole basis: a
        # run with larger N reuses the tables of the harmonics it shares with a smaller one
        if kind == "u":
            S, md = self.cur.S, self.cur.basis.modes
            allm = np.concatenate([np.column_stack([np.zeros(S.n_chi), S.chi_modes]), md, md])
            tag = np.concatenate([np.repeat([0, 1, 2], [S.n_chi, S.K, S.K])[:, None], allm], 1)[cols]
            fn = (self.cur.basis.L, self.cur.basis.M, self.cur.edge_power, str(self.cur.basis.sym))
        else:
            tag = self.pre.basis.modes[cols]
            fn = (self.pre.basis.L, self.pre.basis.M, self.pre.edge_order)
        key = (fit_cache._file_hash(self.inp.path), fn, hashlib.sha256(np.ascontiguousarray(tag, float).tobytes()).hexdigest(),
               tuple(np.round(self.rho, 14)), len(self.theta), kind, n, self.frac, self.margin,
               _src_hash(annulus_freespace, cylindrical_bs, cylindrical_grid, objective,
                         boundary_potential, pressure, node_current), self.j_nodes, "S64")
        h = hashlib.sha256(repr(key).encode()).hexdigest()[:24]
        path = os.path.join(_cache_dir(), f"HarmTab_{kind}{n}_{h}.npy")
        if os.path.exists(path):
            T = np.load(path)
            self._log(f"  table {kind}{n}: {T.shape} from the cache")
            return T
        t0 = time.time()
        g = TensorGrid(self.rho, self.theta[:n_th], blk["planes"], self.NFP)
        nc = len(cols)
        T = np.empty((n_r, n_th, 6 if kind == "u" else 4, P, nc))
        if kind == "p":
            tp = TensorPressure(Separable([self.pre.basis], [self.pre.c], self.NFP), self.pre.edge_order,
                                self.cur._RZ, g, factored=True)
            f = jax.jit(jax.vmap(lambda e, t: (lambda pg: jnp.concatenate([pg[0][..., None], pg[1]], -1))(t(e)),
                                 in_axes=(0, None)))
            bt = 64
            for i in range(0, nc, bt):
                c = cols[i : i + bt]
                E = np.zeros((bt, self.n_p))
                E[np.arange(len(c)), c] = 1.0
                T[..., i : i + len(c)] = np.asarray(f(jnp.asarray(E), tp))[: len(c)].transpose(1, 2, 4, 3, 0)
            self._log(f"  table p{n}: {T.shape}, {time.time() - t0:.0f} s")
        else:
            tc = TensorCurrent(self.cur.S, self.cur._RZ, g, factored=True)
            fJ = jax.jit(jax.vmap(lambda e, t: t(e, 0.0), in_axes=(0, None)))
            bt = 64
            for i in range(0, nc, bt):
                c = cols[i : i + bt]
                E = np.zeros((bt, self.n_u))
                E[np.arange(len(c)), c] = 1.0
                T[:, :, :3, :, i : i + len(c)] = np.asarray(fJ(jnp.asarray(E), tc))[: len(c)].transpose(1, 2, 4, 3, 0)
            del tc
            jax.clear_caches()
            self._log(f"  table u{n}: J {time.time() - t0:.0f} s; B_plasma on the annulus N_phi = {2 * n + 1}",)
            fields, interp = self._annulus(n)
            pp = _PlanePoints(interp.box, interp.R, interp.Z, blk["planes"])
            modes = jax.jit(jax.vmap(lambda e, fl: fl.solver.field_modes(fl.J_box(e, 0.0)), in_axes=(0, None)))
            bt = batch_u[0] if n == 0 else batch_u[1]
            part = path + ".partial.npz"
            start, t_save, t1 = 0, time.time(), time.time()
            if os.path.exists(part):
                with np.load(part) as d:
                    start = int(d["done"])
                    T[:, :, 3:, :, :start] = d["B"]
                self._log(f"    resuming at column {start} of {nc}")
            for i in range(start, nc, bt):
                c = cols[i : i + bt]
                E = np.zeros((bt, self.n_u))
                E[np.arange(len(c)), c] = 1.0
                Bp, Bz = modes(jnp.asarray(E), fields)
                B = np.asarray(pp(Bp, Bz))[: len(c)]  # (b, P, n_r n_th, 3)
                T[:, :, 3:, :, i : i + len(c)] = B.reshape(len(c), P, n_r, n_th, 3).transpose(2, 3, 4, 1, 0)
                done = i + len(c)
                if i == start + 4 * bt:
                    self._log(f"    {(time.time() - t1) / (done - start) * (nc - start) / 60:.1f} min estimated "
                              f"for {nc - start} box solves ({bt} at a time)")
                if time.time() - t_save > 600 and done < nc:
                    tmp = part + f".{os.getpid()}.tmp.npz"
                    np.savez(tmp, B=T[:, :, 3:, :, :done], done=done)
                    os.replace(tmp, part)
                    t_save = time.time()
            solve = modes
            del fields, interp, solve, pp
            jax.clear_caches()
            if os.path.exists(part):
                os.remove(part)
            self._log(f"  table u{n}: {T.shape}, {time.time() - t0:.0f} s")
        self._check_parity(blk, T)
        os.makedirs(_cache_dir(), exist_ok=True)
        tmp = path + f".{os.getpid()}.tmp.npy"
        np.save(tmp, T)
        os.replace(tmp, path)
        return T

    def _check_parity(self, blk, T):
        """The stored half reproduces the other half: J and p (cheap) at the mirrored theta
        of a few columns, against the parity signs."""
        kind, n = blk["kind"], blk["n"]
        cols = (blk["idx"] - (0 if kind == "u" else self.n_u))[:: max(1, len(blk["idx"]) // 6)][:6]
        n_t = len(self.theta)
        jm = n_t - np.arange(1, self.n_th)  # mirrors of the stored j >= 1
        g = TensorGrid(self.rho, self.theta[jm], blk["planes"], self.NFP)
        if kind == "u":
            tc = TensorCurrent(self.cur.S, self.cur._RZ, g, factored=True)
            E = np.zeros((len(cols), self.n_u))
            E[np.arange(len(cols)), cols] = 1.0
            V = np.asarray(jax.vmap(lambda e: tc(e, 0.0))(jnp.asarray(E)))  # (c, r, t, P, 3)
            ref = T[:, 1:, :3][..., [np.searchsorted(blk["idx"], c) for c in cols]]
            sgn = _PARITY["u"][:3]
        else:
            tp = TensorPressure(Separable([self.pre.basis], [self.pre.c], self.NFP), self.pre.edge_order,
                                self.cur._RZ, g, factored=True)
            E = np.zeros((len(cols), self.n_p))
            E[np.arange(len(cols)), cols] = 1.0
            V = np.asarray(jax.vmap(lambda e: (lambda pg: jnp.concatenate([pg[0][..., None], pg[1]], -1))(
                tp(jnp.asarray(e))))(jnp.asarray(E)))
            ref = T[:, 1:][..., [np.searchsorted(blk["idx"] - self.n_u, c) for c in cols]]
            sgn = _PARITY["p"]
        V = V.transpose(1, 2, 4, 3, 0)  # (r, t, comp, P, c)
        ps = np.array([1.0, -1.0])[: blk["P"]]
        pred = ref * sgn[None, None, :, None, None] * ps[None, None, None, :, None]
        err = np.abs(V - pred).max() / max(np.abs(V).max(), 1e-300)
        if err > 1e-10:
            raise RuntimeError(f"table {kind}{n}: the theta mirror misses by {err:.1e}")

    # ------------------------------------------------------------------ device side
    def _compile(self):
        n_r = len(self.rho)
        c = self.chunk
        self.n_chunks = -(-n_r // c)
        pad = self.n_chunks * c - n_r
        padr = lambda a: np.concatenate([a, np.zeros((pad,) + a.shape[1:])]) if pad else a  # noqa: E731
        self._const = {k: padr(v) for k, v in (("sw3", self.sw3), ("swF", self.swF), ("swP", self.swP),
                                                ("swJ", self.swJ), ("JD", self.JD), ("J_I", self.J_I),
                                                ("B_I", self.B_I), ("B_coil", self.B_coil),
                                                ("B_pert", self.B_pert), ("iR", 1.0 / self._geoR),
                                                ("Zg", self._geoZ))}
        self._pad = pad
        self._sgn = {k: np.stack([v, -v], -1) for k, v in _PARITY.items()}  # (comps, P=2)
        blocks = [(b["kind"], b["n"], b["P"], b["sl"].start, b["sl"].stop) for b in self.blocks]
        self._blocks_static = tuple(blocks)
        self._Wz = tuple(jnp.asarray(b["Wz"]) for b in self.blocks)
        self._src_j = jnp.asarray(self.src)
        self._mir = jnp.asarray(self.mirror)
        # tables kept on the device while they fit (beside the Gram matrix, its Cholesky factor
        # and a chunk's work); the rest streams from the host chunk by chunk
        self._dev = [None] * len(self.tables)
        used = self._reload()
        self._log(f"  tables on the device: {used / 1e9:.1f} GB; streamed per pass: "
                  f"{sum(T.nbytes for T, d in zip(self.tables, self._dev) if d is None) / 1e9:.1f} GB")
        # the gradient of <p> (linear in z: a constant row)
        self._gW = None
        self.n_pc = sum(b["sl"].stop - b["sl"].start for b in self.blocks if b["kind"] == "p")
        self.n_uc = self.n - self.n_pc
        self._dvec = np.zeros(self.n)
        if self.rough is not None:
            S, md = self.cur.S, self.cur.basis.modes
            l_u = np.concatenate([np.abs(S.chi_modes[:, 0]), md[:, 0], md[:, 0]]).astype(float)
            L = float(self.cur.basis.L)
            for blk, T in zip(self.blocks, self.tables):
                if blk["kind"] != "u" or blk["n"] == 0:
                    continue
                coln = np.sqrt(np.mean(T[:, :, :3] ** 2, axis=(0, 1, 2, 3)))  # |J| of each column
                self._dvec[blk["sl"]] = (self.rough["w"] * (l_u[blk["idx"]] / L) ** self.rough["q"] * coln
                                         / self.rms["J"])
        self._dvec_j = jnp.asarray(self._dvec)
        self._rough_idx = np.flatnonzero(self._dvec)
        self._s, self._snorm = None, 1.0

    def _devs(self):
        g = int(self.speed.get("gpus", 1))
        return jax.devices()[:g]

    def _chunk_tables(self, k, dev=None):
        if self.speed.get("resident") and dev is None and len(self._devs()) > 1:
            dev = self._devs()[0]  # passes on the first device (gradient, evaluation): tables cached there
        if self.speed.get("resident") and dev is not None:
            key = (k, dev.id)
            if key not in self._tcache:
                self._tcache[key] = self._chunk_tables_build(k, dev)
            return self._tcache[key]
        return self._chunk_tables_build(k, dev)

    def _chunk_tables_build(self, k, dev=None):
        r0, r1 = k * self.chunk, min((k + 1) * self.chunk, len(self.rho))
        out = []
        for T, D in zip(self.tables, self._dev):
            if D is not None:
                sl = D[k * self.chunk : (k + 1) * self.chunk]
                out.append(sl if dev is None else jax.device_put(sl, dev))
                continue
            t = T[r0:r1]
            if r1 - r0 < self.chunk:
                t = np.concatenate([t, np.zeros((self.chunk - (r1 - r0),) + t.shape[1:])])
            out.append(jnp.asarray(t) if dev is None else jax.device_put(t, dev))
        return tuple(out)

    def _chunk_const(self, k, dev=None):
        r0, r1 = k * self.chunk, (k + 1) * self.chunk
        if dev is None:
            return {key: jnp.asarray(v[r0:r1]) for key, v in self._const.items()}
        return {key: jax.device_put(v[r0:r1], dev) for key, v in self._const.items()}

    @functools.partial(jax.jit, static_argnums=(0,))
    def _synth(self, tables):
        """Column fields of a chunk at the collocation points, per block: (c, n_t, n_z, comps, cols)."""
        out = []
        for (kind, n, P, a, b), T, Wz in zip(self._blocks_static, tables, self._Wz):
            sgn = jnp.asarray(self._sgn[kind][:, :P])  # (comps, P)
            full = T[:, self._src_j]  # (c, n_t, comps, P, cols)
            f = 1.0 + self._mir[None, :, None, None, None] * (sgn[None, None, :, :, None] - 1.0)
            out.append(jnp.einsum("rtcpk,zp->rtzck", full * f, Wz))
        return out

    def _fields(self, cols, z, const, eps):
        """J, B, p, grad p of z (c, n_t, n_z, .) and J0 (the n = 0 J, (c, n_t, 3))."""
        J = const["J_I"][:, :, None]
        Bp = const["B_I"][:, :, None]
        p, gp, J0 = 0.0, 0.0, const["J_I"]
        for (kind, n, P, a, b), V in zip(self._blocks_static, cols):
            v = V @ z[a:b]
            if kind == "u":
                J, Bp = J + v[..., :3], Bp + v[..., 3:]
                if n == 0:
                    J0 = J0 + v[:, :, 0, :3]
            else:
                p, gp = p + v[..., 0], gp + v[..., 1:]
        B = const["B_coil"] + eps * const["B_pert"] + Bp
        return J, B, p, gp, (J if self.j_full else J0)

    @staticmethod
    def _cross(a, b):
        return jnp.stack([a[..., 1] * b[..., 2] - a[..., 2] * b[..., 1], a[..., 2] * b[..., 0] - a[..., 0] * b[..., 2],
                          a[..., 0] * b[..., 1] - a[..., 1] * b[..., 0]], -1)

    def _wF(self, const, B, X):
        """swF P X for force-like X (leading axes before B's allowed): P = I + (par - 1) b b^T."""
        pad = (None,) * (X.ndim - B.ndim)
        if self._par != 1.0:
            Bn = jnp.linalg.norm(B, axis=-1, keepdims=True)
            b = jax.lax.stop_gradient(B / jnp.where(Bn > 0, Bn, 1.0))[pad]
            X = X + (self._par - 1.0) * b * jnp.sum(b * X, -1, keepdims=True)
        return const["swF"][..., None][pad] * X

    def _residual(self, J, B, p, gp, J0, const):
        """The local rows (force, J, p<0) of a chunk and its share of <p>."""
        F = self._cross(J, B) - gp
        rF0 = const["swF"][..., None] * F
        rF = self._wF(const, B, F)
        rJ = const["swJ"][..., None] * (J0 - const["JD"])
        rP = const["swP"] * jnp.minimum(p, 0.0)
        Wp = jnp.sum(const["sw3"] ** 2 * p)
        return rF, rJ, rP, Wp, rF0

    def _rows(self, tables, z, const, eps):
        """The exact Jacobian rows of a chunk, A (n, rows), its residual r (rows), its share of
        <p> and of the gradient of <p>."""
        cols = self._synth(tables)
        J, B, p, gp, J0 = self._fields(cols, z, const, eps)
        rF, rJ, rP, Wp, rF0 = self._residual(J, B, p, gp, J0, const)
        act = (p < 0).astype(p.dtype)
        rows_F, rows_J, rows_P, gW = [], [], [], []
        for (kind, n, P, a, b), V in zip(self._blocks_static, cols):
            if kind == "u":
                dJ, dB = jnp.moveaxis(V[..., :3, :], -1, 0), jnp.moveaxis(V[..., 3:, :], -1, 0)  # (k, c, t, z, 3)
                dF = self._cross(dJ, B[None]) + self._cross(J[None], dB)
                rows_F.append(self._wF(const, B, dF))
                if self.j_full:
                    rows_J.append(const["swJ"][None, ..., None] * dJ)
                else:
                    rows_J.append(const["swJ"][None, ..., None] * dJ[:, :, :, 0] if n == 0 else
                                  jnp.zeros((b - a,) + rJ.shape))
                rows_P.append(jnp.zeros((b - a,) + rP.shape))
                gW.append(jnp.zeros(b - a))
            else:
                dp, dgp = jnp.moveaxis(V[..., 0, :], -1, 0), jnp.moveaxis(V[..., 1:, :], -1, 0)
                rows_F.append(-self._wF(const, B, dgp))
                rows_J.append(jnp.zeros((b - a,) + rJ.shape))
                rows_P.append(const["swP"][None] * act[None] * dp)
                gW.append(jnp.sum(const["sw3"][None] ** 2 * dp, axis=(1, 2, 3)))
        A = jnp.concatenate([jnp.concatenate([r.reshape(r.shape[0], -1) for r in rows], 0)
                             for rows in (rows_F, rows_J, rows_P)], 1)  # (n, rows)
        r = jnp.concatenate([rF.ravel(), rJ.ravel(), rP.ravel()])
        return A, r, Wp, jnp.concatenate(gW)

    @functools.partial(jax.jit, static_argnums=(0,), donate_argnums=(1, 2))
    def _gram_chunk(self, H, g, tables, z, const, eps):
        A, r, Wp, gW = self._rows(tables, z, const, eps)
        return H + A @ A.T, g + A @ r, 0.5 * jnp.sum(r**2), Wp, gW

    @functools.partial(jax.jit, static_argnums=(0,))
    def _rows_chunk(self, tables, z, const, eps):
        """[A^T | r] of a chunk (rows, n + 1), its share of <p> and of the gradient of <p>."""
        A, r, Wp, gW = self._rows(tables, z, const, eps)
        return jnp.concatenate([A.T, r[:, None]], 1), Wp, gW

    @functools.partial(jax.jit, static_argnums=(0,))
    def _Av_chunk(self, tables, z, d, const, eps):
        """The rows of A d of a chunk (flat) and its share of <p> along d."""
        cols = self._synth(tables)
        J, B, p, gp, J0 = self._fields(cols, z, const, eps)
        act = (p < 0).astype(p.dtype)
        dJ, dB, dp, dgp, dJ0 = 0.0, 0.0, 0.0, 0.0, 0.0
        for (kind, n, P, a, b), V in zip(self._blocks_static, cols):
            v = V @ d[a:b]
            if kind == "u":
                dJ, dB = dJ + v[..., :3], dB + v[..., 3:]
                if n == 0 and not self.j_full:
                    dJ0 = dJ0 + v[:, :, 0, :3]
            else:
                dp, dgp = dp + v[..., 0], dgp + v[..., 1:]
        qF = self._wF(const, B, self._cross(dJ, B) + self._cross(J, dB) - dgp)
        qJ = const["swJ"][..., None] * (dJ if self.j_full else dJ0) * jnp.ones_like(J0)
        qP = const["swP"] * act * dp
        return jnp.concatenate([qF.ravel(), qJ.ravel(), qP.ravel()]), jnp.sum(const["sw3"] ** 2 * dp)

    @functools.partial(jax.jit, static_argnums=(0,))
    def _ATu_chunk(self, tables, z, u, const, eps):
        """A^T u of the local rows of a chunk (u flat, as ``_Av_chunk`` returns)."""
        cols = self._synth(tables)
        J, B, p, gp, J0 = self._fields(cols, z, const, eps)
        act = (p < 0).astype(p.dtype)
        sF, sJ = J.shape, J0.shape
        nF, nJ = int(np.prod(sF)), int(np.prod(sJ))
        return self._AT(cols, J, B, u[:nF].reshape(sF), u[nF:nF + nJ].reshape(sJ), u[nF + nJ:].reshape(p.shape),
                        act, const)

    @functools.partial(jax.jit, static_argnums=(0,))
    def _r_chunk(self, tables, z, const, eps):
        cols = self._synth(tables)
        J, B, p, gp, J0 = self._fields(cols, z, const, eps)
        rF, rJ, rP, Wp, rF0 = self._residual(J, B, p, gp, J0, const)
        return jnp.concatenate([rF.ravel(), rJ.ravel(), rP.ravel()]), Wp

    @functools.partial(jax.jit, static_argnums=(0,))
    def _trial_chunk(self, tables, z, D, const, eps):
        """Sum of squares of the local rows and the share of <p>, at z + D[:, k] for each k."""
        cols = self._synth(tables)

        def one(d):
            J, B, p, gp, J0 = self._fields(cols, z + d, const, eps)
            rF, rJ, rP, Wp, rF0 = self._residual(J, B, p, gp, J0, const)
            Fa = jnp.sum(const["sw3"] * jnp.linalg.norm(rF0, axis=-1))
            return jnp.sum(rF**2) + jnp.sum(rJ**2) + jnp.sum(rP**2), Wp, Fa

        return jax.lax.map(one, D.T)

    def _AT(self, cols, J, B, qF, qJ, qP, act, const):
        """A^T q of a chunk for row values q (force qF (c,t,z,3), J qJ (c,t,3), p<0 qP (c,t,z))."""
        wF = self._wF(const, B, qF)
        BxW, WxJ = self._cross(B, wF), self._cross(wF, J)
        wJ = const["swJ"][..., None] * qJ
        wP = const["swP"] * act * qP
        out = []
        for (kind, n, P, a, b), V in zip(self._blocks_static, cols):
            if kind == "u":
                o = jnp.einsum("rtzck,rtzc->k", V[..., :3, :], BxW) + jnp.einsum("rtzck,rtzc->k", V[..., 3:, :], WxJ)
                if self.j_full:
                    o = o + jnp.einsum("rtzck,rtzc->k", V[..., :3, :], wJ)
                elif n == 0:
                    o = o + jnp.einsum("rtck,rtc->k", V[:, :, 0, :3, :], wJ)
            else:
                o = -jnp.einsum("rtzck,rtzc->k", V[..., 1:, :], wF) + jnp.einsum("rtzk,rtz->k", V[..., 0, :], wP)
            out.append(o)
        return jnp.concatenate(out)

    @functools.partial(jax.jit, static_argnums=(0,))
    def _grad_chunk(self, tables, z, const, eps):
        """A^T r, |r|^2 of the local rows, the share of <p> and of <|F|>, and of the gradient of <p>."""
        cols = self._synth(tables)
        J, B, p, gp, J0 = self._fields(cols, z, const, eps)
        rF, rJ, rP, Wp, rF0 = self._residual(J, B, p, gp, J0, const)
        act = (p < 0).astype(p.dtype)
        g = self._AT(cols, J, B, rF, rJ, rP, act, const)  # rows are already weighted
        gW = jnp.concatenate([jnp.sum(const["sw3"] ** 2 * V[..., 0, :].transpose(3, 0, 1, 2), axis=(1, 2, 3))
                              if kind == "p" else jnp.zeros(b - a)
                              for (kind, n, P, a, b), V in zip(self._blocks_static, cols)])
        Fa = jnp.sum(const["sw3"] * jnp.linalg.norm(rF0, axis=-1))
        return g, jnp.sum(rF**2) + jnp.sum(rJ**2) + jnp.sum(rP**2), Wp, Fa, gW

    @functools.partial(jax.jit, static_argnums=(0,))
    def _normal_chunk(self, tables, z, d, const, eps):
        """A^T (A d) of the local rows of a chunk, and its share of <p> along d."""
        cols = self._synth(tables)
        J, B, p, gp, J0 = self._fields(cols, z, const, eps)
        act = (p < 0).astype(p.dtype)
        dJ, dB, dp, dgp, dJ0 = 0.0, 0.0, 0.0, 0.0, 0.0
        for (kind, n, P, a, b), V in zip(self._blocks_static, cols):
            v = V @ d[a:b]
            if kind == "u":
                dJ, dB = dJ + v[..., :3], dB + v[..., 3:]
                if n == 0 and not self.j_full:
                    dJ0 = dJ0 + v[:, :, 0, :3]
            else:
                dp, dgp = dp + v[..., 0], dgp + v[..., 1:]
        qF = self._wF(const, B, self._cross(dJ, B) + self._cross(J, dB) - dgp)
        qJ = const["swJ"][..., None] * (dJ if self.j_full else dJ0)
        qP = const["swP"] * act * dp
        return self._AT(cols, J, B, qF, qJ, qP, act, const), jnp.sum(const["sw3"] ** 2 * dp)

    # ------------------------------------------------------------------ global rows
    def _p_grads(self, cols):
        """grad of every pressure column at the chunk's points: (c, t, z, 3, n_pc), and p of them."""
        G = [V[..., 1:, :] for (kind, *_), V in zip(self._blocks_static, cols) if kind == "p"]
        P = [V[..., 0, :] for (kind, *_), V in zip(self._blocks_static, cols) if kind == "p"]
        return jnp.concatenate(G, -1), jnp.concatenate(P, -1)

    def _transport_R(self, cols, B, gp, w):
        """sum_points w grad(phi_k) . M grad p, M = I + (kappa - 1) b b."""
        Gp, _ = self._p_grads(cols)
        Bn = jnp.linalg.norm(B, axis=-1, keepdims=True)
        b = B / jnp.where(Bn > 0, Bn, 1.0)  # (padded radii: B = 0, weight 0)
        Mgp = gp + (self._kappa - 1.0) * b * jnp.sum(b * gp, -1, keepdims=True)
        return jnp.einsum("rtzck,rtzc->k", Gp, w[..., None] * Mgp)

    @functools.partial(jax.jit, static_argnums=(0, 5))
    def _global_chunk(self, tables, z, const, eps, jac):
        """A chunk's share of <p> and its gradient row, and (with transport) of R and its Jacobian."""
        cols = self._synth(tables)
        J, B, p, gp, J0 = self._fields(cols, z, const, eps)
        w = const["sw3"] ** 2
        Gp, Pp = self._p_grads(cols)
        Wp, gWp = jnp.sum(w * p), jnp.einsum("rtzk,rtz->k", Pp, w)
        if self.transport is None:
            return Wp, gWp, None, None
        Bn = jnp.linalg.norm(B, axis=-1, keepdims=True)
        Bn = jnp.where(Bn > 0, Bn, 1.0)  # (padded radii: B = 0, weight 0)
        b = B / Bn
        k1 = self._kappa - 1.0
        bp = jnp.sum(b * gp, -1, keepdims=True)
        wG = w[..., None, None] * Gp
        R = jnp.einsum("rtzck,rtzc->k", wG, gp + k1 * b * bp)
        if not jac:
            return Wp, gWp, R, None
        MGp = Gp + k1 * b[..., None] * jnp.sum(b[..., None] * Gp, -2, keepdims=True)
        JRp = jnp.einsum("rtzck,rtzcj->kj", wG, MGp)
        dB = jnp.concatenate([V[..., 3:, :] for (kind, *_), V in zip(self._blocks_static, cols) if kind == "u"], -1)
        db = (dB - b[..., None] * jnp.sum(b[..., None] * dB, -2, keepdims=True)) / Bn[..., None]
        dMgp = k1 * (db * bp[..., None] + b[..., None] * jnp.sum(db * gp[..., None], -2, keepdims=True))
        JRu = jnp.einsum("rtzck,rtzcj->kj", wG, dMgp)
        return Wp, gWp, R, jnp.concatenate([JRu, JRp], 1)

    @functools.partial(jax.jit, static_argnums=(0,))
    def _trial_R_chunk(self, tables, z, D, const, eps):
        cols = self._synth(tables)
        w = const["sw3"] ** 2

        def one(d):
            J, B, p, gp, J0 = self._fields(cols, z + d, const, eps)
            return self._transport_R(cols, B, gp, w)

        return jax.lax.map(one, D.T)

    def _global(self, z, eps, jac=True):
        """The global residuals g (the <p> row, the transport rows, the roughness rows) and, with
        ``jac``, their Jacobian G (n_g, n)."""
        Wp, gWp, R, JR = 0.0, jnp.zeros(self.n_pc), None, None
        for k in range(self.n_chunks):
            w_, gw, r_, jr = self._global_chunk(self._chunk_tables(k), z, self._chunk_const(k), eps, jac)
            Wp, gWp = Wp + float(w_), gWp + gw
            if r_ is not None:
                R = r_ if R is None else R + r_
                if jr is not None:
                    JR = jr if JR is None else JR + jr
        self._gW = jnp.concatenate([jnp.zeros(self.n_uc), gWp])
        gv = [jnp.asarray([self.weights["W"] * (Wp - self.W_t) / self.W_t])]
        rows = [self.wW * self._gW[None]] if jac else []
        if self.transport is not None:
            if self._s is None:
                raise RuntimeError("transport: set_source(x) first (the eps = 0 state)")
            sc = self.transport["w"] / self._snorm
            gv.append(sc * (R - self._s))
            if jac:
                rows.append(sc * JR)
        if self.mom is not None:
            r_m, G_m = self._mom_rows(z, eps, jac)
            gv.append(r_m)
            if jac:
                rows.append(G_m)
        if self.rough is not None:
            ri = self._rough_idx
            gv.append(self._dvec_j[ri] * z[ri])
            if jac:
                rows.append(jnp.zeros((len(ri), self.n)).at[jnp.arange(len(ri)), ri].set(self._dvec_j[ri]))
        return jnp.concatenate(gv), (jnp.concatenate(rows) if jac else None)

    # --- current moments (a global, nonlinear-in-J row set)
    _MOM_KEYS = ("I", "R", "Z", "V")

    def _mom_sums(self, J, const):
        """sum_points (w J_phi / R, w J_phi, w Z J_phi / R, w J_phi^2) of a chunk (..., 4)."""
        w = const["sw3"] ** 2
        Jp = J[..., 1]
        return jnp.stack([jnp.sum(w * Jp * const["iR"]), jnp.sum(w * Jp),
                          jnp.sum(w * const["Zg"] * Jp * const["iR"]), jnp.sum(w * Jp**2)])

    def _mom_values(self, S):
        """The moments (I, R, Z, V) from the sums S (..., 4), the selected ones (..., n_which)."""
        s0, s1, sz, s2 = S[..., 0], S[..., 1], S[..., 2], S[..., 3]
        m = {"I": self._Vtot / (2 * np.pi) * s0, "R": s1 / s0, "Z": sz / s0, "V": s1**2 / s2}
        return jnp.stack([m[k] for k in self.mom["which"]], -1)

    def _mom_resid(self, S):
        return self.mom["w"] * (self._mom_values(S) - self._mom_t) / self._mom_scale

    @functools.partial(jax.jit, static_argnums=(0, 5))
    def _mom_chunk(self, tables, z, const, eps, jac):
        cols = self._synth(tables)
        J, B, p, gp, J0 = self._fields(cols, z, const, eps)
        S = self._mom_sums(J, const)
        if not jac:
            return S, None
        w = const["sw3"] ** 2
        Jp = J[..., 1]
        rows = []
        for (kind, n, P, a, b), V in zip(self._blocks_static, cols):
            if kind == "u":
                dJp = V[..., 1, :]  # (c, t, z, k)
                rows.append(jnp.stack([jnp.einsum("rtzk,rtz->k", dJp, w * const["iR"]),
                                       jnp.einsum("rtzk,rtz->k", dJp, w),
                                       jnp.einsum("rtzk,rtz->k", dJp, w * const["Zg"] * const["iR"]),
                                       2.0 * jnp.einsum("rtzk,rtz->k", dJp, w * Jp)]))
            else:
                rows.append(jnp.zeros((4, b - a)))
        return S, jnp.concatenate(rows, 1)

    @functools.partial(jax.jit, static_argnums=(0,))
    def _mom_trial_chunk(self, tables, z, D, const, eps):
        cols = self._synth(tables)

        def one(d):
            J, B, p, gp, J0 = self._fields(cols, z + d, const, eps)
            return self._mom_sums(J, const)

        return jax.lax.map(one, D.T)  # (K, 4)

    def _mom_total(self, z, eps, jac):
        S, JS = 0.0, None
        for k in range(self.n_chunks):
            s_, js = self._mom_chunk(self._chunk_tables(k), z, self._chunk_const(k), eps, jac)
            S = S + s_
            if jac:
                JS = js if JS is None else JS + js
        return S, JS

    def _mom_rows(self, z, eps, jac):
        """The moment residuals r (n_which) and, with ``jac``, their Jacobian (n_which, n)."""
        S, JS = self._mom_total(z, eps, jac)
        r = self._mom_resid(S)
        if not jac:
            return r, None
        return r, jax.jacfwd(self._mom_resid)(S) @ JS

    def _set_moment_target(self, x):
        z = self.to_z(x)
        S, _ = self._mom_total(z, 0.0, False)
        which = self.mom["which"]
        t = np.asarray(self._mom_values(S))
        self._mom_t = jnp.asarray(t)
        a = float(self.inp.a)
        sc = {"I": abs(t[which.index("I")]) if "I" in which else 1.0, "R": a, "Z": a,
              "V": abs(t[which.index("V")]) if "V" in which else 1.0}
        self._mom_scale = jnp.asarray([sc[k] for k in which])
        self._log("  moments " + ", ".join(f"{k} {v:.6g}" for k, v in zip(which, t))
                  + f"; weight {self.mom['w']:g}")

    def set_source(self, x):
        """The transport source: the weak residual of the eps = 0 state x (so that it satisfies the
        transport equation exactly)."""
        if self.mom is not None:
            self._set_moment_target(x)
        if self.transport is None:
            return
        z = self.to_z(x)
        R = None
        for k in range(self.n_chunks):
            _, _, r_, _ = self._global_chunk(self._chunk_tables(k), z, self._chunk_const(k), 0.0, False)
            R = r_ if R is None else R + r_
        self._s = R
        self._snorm = float(jnp.linalg.norm(R))
        self._log(f"  transport: kappa_par / kappa_perp {self._kappa:g}, weight {self.transport['w']:g}; "
                  f"source |s| {self._snorm:.3e}")

    # ------------------------------------------------------------------ public
    def to_z(self, x):
        return jnp.asarray(np.asarray(x)[self.perm])

    def to_x(self, z):
        return np.asarray(z)[self.iperm]

    def evaluate(self, x, eps):
        """f = |r|^2 / 2 and <|F|>/norm at x (one pass, no Gram)."""
        z = self.to_z(x)
        f, Wp, Fsum = 0.0, 0.0, 0.0
        for k in range(self.n_chunks):
            const = self._chunk_const(k)
            s, w_, Fa = self._eval_chunk(self._chunk_tables(k), z, const, eps)
            f, Wp, Fsum = f + float(s), Wp + float(w_), Fsum + float(Fa)
        gv, _ = self._global(z, eps, jac=False)
        return 0.5 * (f + float(gv @ gv)), Fsum

    @functools.partial(jax.jit, static_argnums=(0,))
    def _eval_chunk(self, tables, z, const, eps):
        cols = self._synth(tables)
        J, B, p, gp, J0 = self._fields(cols, z, const, eps)
        rF, rJ, rP, Wp, rF0 = self._residual(J, B, p, gp, J0, const)
        # <|F|>/norm: sum sqrt(dV/V)^2 |F| / norm = sum sw3 |rF|
        Fa = jnp.sum(const["sw3"] * jnp.linalg.norm(rF0, axis=-1))
        return jnp.sum(rF**2) + jnp.sum(rJ**2) + jnp.sum(rP**2), Wp, Fa

    @functools.partial(jax.jit, static_argnums=(0,))
    def _bgp_chunk(self, tables, z, const, eps):
        cols = self._synth(tables)
        J, B, p, gp, J0 = self._fields(cols, z, const, eps)
        Bn = jnp.linalg.norm(B, axis=-1)
        return jnp.sum(const["sw3"] ** 2 * jnp.abs(jnp.sum(B * gp, -1)) / jnp.where(Bn > 0, Bn, 1.0))

    def bgradp(self, x, eps):
        """<|B . grad p| / |B|>_vol / norm at x: the part of the force along B, in the units of <|F|>/norm."""
        z = self.to_z(x)
        tot = 0.0
        for k in range(self.n_chunks):
            tot += float(self._bgp_chunk(self._chunk_tables(k), z, self._chunk_const(k), eps))
        return tot / self.norm

    def fields(self, x, eps):
        """J, B, p, grad p (cylindrical) and F at all collocation points, numpy, (n_r, n_t, n_z, .)."""
        z = self.to_z(x)
        out = {k: [] for k in ("J", "B", "p", "grad_p", "F")}
        for k in range(self.n_chunks):
            const = self._chunk_const(k)
            J, B, p, gp, _ = self._fields(self._synth(self._chunk_tables(k)), z, const, eps)
            for key, v in zip(("J", "B", "p", "grad_p", "F"), (J, B, p, gp, self._cross(J, B) - gp)):
                out[key].append(np.asarray(v))
        n_r = len(self.rho)
        return {k: np.concatenate(v)[:n_r] for k, v in out.items()}

    # ------------------------------------------------------------------ the solver
    def _gram(self, z, eps):
        """Full pass: H = A^T A (with the <p> row), g = A^T r, f."""
        n = self.n
        H, g = jnp.zeros((n, n)), jnp.zeros(n)
        fl, Wp, gW = 0.0, 0.0, jnp.zeros(n)
        for k in range(self.n_chunks):
            H, g, s, w_, gw = self._gram_chunk(H, g, self._chunk_tables(k), z, self._chunk_const(k), eps)
            fl, Wp, gW = fl + float(s), Wp + float(w_), gW + gw
        self._gW = gW
        rW = self.weights["W"] * (Wp - self.W_t) / self.W_t
        aW = self.wW * gW
        return _add_outer(H, aW), g + aW * rW, 0.5 * (fl + rW**2)

    def _grad(self, z, eps):
        """Pass: g = A^T r and f (exact, matrix-free)."""
        g, fl, Wp, Fs = jnp.zeros(self.n), 0.0, 0.0, 0.0
        gW = jnp.zeros(self.n)
        for k in range(self.n_chunks):
            gk, s, w_, Fa, gw = self._grad_chunk(self._chunk_tables(k), z, self._chunk_const(k), eps)
            g, fl, Wp, Fs, gW = g + gk, fl + float(s), Wp + float(w_), Fs + float(Fa), gW + gw
        gv, G = self._global(z, eps)
        return g + G.T @ gv, 0.5 * (fl + float(gv @ gv)), Fs

    def _normal(self, z, d, eps):
        """Pass: A^T A d (exact, matrix-free, with the <p> row)."""
        out, Sd = jnp.zeros(self.n), 0.0
        for k in range(self.n_chunks):
            o, sd = self._normal_chunk(self._chunk_tables(k), z, d, self._chunk_const(k), eps)
            out, Sd = out + o, Sd + sd
        return out + self.wW**2 * self._gW * Sd

    def _trials(self, z, D, eps):
        K = D.shape[1]
        fs, Ws, Fs = (np.zeros(K) for _ in range(3))
        R = 0.0
        devs = self._devs() if (self.transport is None and self.mom is None and self.rough is None) else self._devs()[:1]
        nd = len(devs)
        zD = [(jax.device_put(z, d), jax.device_put(D, d)) for d in devs] if nd > 1 else [(z, D)]
        outs = []
        for k in range(self.n_chunks):
            g = k % nd
            dev = devs[g] if nd > 1 else None
            tab, con = self._chunk_tables(k, dev), self._chunk_const(k, dev)
            outs.append(self._trial_chunk(tab, zD[g][0], zD[g][1], con, eps))  # async: summed below
            if self.transport is not None:
                R = R + self._trial_R_chunk(tab, z, D, con, eps)  # (K, n_pc)
        for s, w_, Fa in outs:
            fs, Ws, Fs = fs + np.asarray(s), Ws + np.asarray(w_), Fs + np.asarray(Fa)
        g2 = (self.weights["W"] * (Ws - self.W_t) / self.W_t) ** 2
        if self.transport is not None:
            sc = self.transport["w"] / self._snorm
            g2 = g2 + np.asarray(jnp.sum((sc * (R - self._s[None])) ** 2, 1))
        if self.mom is not None:
            S = 0.0
            for k in range(self.n_chunks):
                S = S + self._mom_trial_chunk(self._chunk_tables(k), z, D, self._chunk_const(k), eps)
            g2 = g2 + np.asarray(jnp.sum(self._mom_resid(S) ** 2, -1))
        if self.rough is not None:
            ri = self._rough_idx
            g2 = g2 + np.asarray(jnp.sum((self._dvec_j[ri, None] * (z[ri, None] + D[ri])) ** 2, 0))
        return 0.5 * (fs + g2), Fs

    def _scaling(self, diag, floor=1e-8):
        """Column scaling D = 1 / sqrt(max(H_ii, floor x median of its kind)): Jacobi, except that a
        column that barely moves the residual is not given a huge scale (with the response driven,
        its gradient is no longer 0, and a damped step would throw it far out)."""
        dg = np.asarray(diag)
        out = np.zeros_like(dg)
        for kind in ("u", "p"):
            idx = np.concatenate([np.arange(b["sl"].start, b["sl"].stop) for b in self.blocks if b["kind"] == kind])
            pos = dg[idx][dg[idx] > 0]
            fl = floor * (np.median(pos) if len(pos) else 1.0)
            out[idx] = 1.0 / np.sqrt(np.maximum(dg[idx], fl))
        return jnp.asarray(out)

    def _evict(self):
        """Free the resident tables (the Gram step needs the room); ``_reload`` puts them back."""
        if self.speed.get("resident"):
            return
        self._dev = [None] * len(self._dev)

    def _reload(self):
        if self.speed.get("resident") and self._loaded:
            return
        self._loaded = True
        if len(self._devs()) > 1:  # several devices: the tables are streamed from the host to each
            self._dev = [None] * len(self.tables)
            return 0.0
        n_r = len(self.rho)
        pad = self.n_chunks * self.chunk - n_r
        budget = 0.9 * _device_bytes() - self.n**2 * 8 - 6e9  # beside the kept Cholesky factor
        used, dev = 0.0, []
        for T in self.tables:
            if used + T.nbytes <= budget:
                dev.append(jnp.asarray(np.concatenate([T, np.zeros((pad,) + T.shape[1:])]) if pad else T))
                used += T.nbytes
            else:
                dev.append(None)
        self._dev = dev
        return used

    def _gram_step(self, z, eps, decades):
        """The exact Gram matrix at z, steps for mu over decades by Cholesky, each evaluated
        exactly; keeps the Cholesky factor of the best as the PCG preconditioner."""
        n = self.n
        self._prec = None
        self._evict()
        H, g, f_here = self._gram(z, eps)
        Dsc = self._scaling(jnp.diag(H))
        Hs = _scale(H, Dsc)  # in place (H donated)
        del H
        gs = Dsc * g
        v = jnp.ones(n) / np.sqrt(n)
        for _ in range(30):  # lambda_max of Hs: the damping scale
            v = Hs @ v
            lam = float(jnp.linalg.norm(v))
            v = v / lam
        host = 3 * n * n * 8 > 0.6 * _device_bytes()  # large n: Hs waits on the host
        if host:
            Hs = np.asarray(Hs)
        steps, mus = [], []
        for kd in np.arange(decades[0] - 3, decades[1] + 0.5, 1.0):
            mu = lam * 10.0 ** (-kd)
            y, ok = _damped_step(jnp.asarray(Hs) if host else Hs, mu, gs)
            if bool(ok):
                steps.append(Dsc * y)
                mus.append(mu)
        D = jnp.stack(steps, 1)
        fs, Fs = self._trials(z, D, eps)
        j = int(np.argmin(fs))
        dz = D[:, j]
        i = jnp.arange(n)
        self._prec = {"L": jnp.linalg.cholesky((jnp.asarray(Hs) if host else Hs).at[i, i].add(mus[j])),
                      "Dsc": Dsc, "mu": mus[j], "lam": lam}
        del Hs, D
        self._reload()
        return dz, f_here, fs[j], Fs[j], {"mu/lambda_max": mus[j] / lam, "|g|": float(jnp.linalg.norm(gs))}

    def _pcg_step(self, z, eps, maxit=30, tol=1e-6):
        """The LM step at z for the kept damping, by PCG on (D A^T A D + mu I) y = -D g with
        exact matrix-free products and the kept Cholesky factor as preconditioner; then the
        step and its halvings evaluated exactly."""
        P = self._prec
        Dsc, mu, L = P["Dsc"], P["mu"], P["L"]
        g, f_here, _ = self._grad(z, eps)
        b = -Dsc * g
        M = lambda r: jax.scipy.linalg.cho_solve((L, True), r)  # noqa: E731
        Op = lambda y: Dsc * self._normal(z, Dsc * y, eps) + mu * y  # noqa: E731
        y = M(b)
        r = b - Op(y)
        zr = M(r)
        pdir = zr
        rz = float(r @ zr)
        bn = float(jnp.linalg.norm(b))
        k, res = 0, float(jnp.linalg.norm(r)) / bn
        while res > tol and k < maxit:
            Ap = Op(pdir)
            alpha = rz / float(pdir @ Ap)
            y, r = y + alpha * pdir, r - alpha * Ap
            zr = M(r)
            rz_new = float(r @ zr)
            pdir = zr + (rz_new / rz) * pdir
            rz = rz_new
            k += 1
            res = float(jnp.linalg.norm(r)) / bn
        d = Dsc * y
        D = jnp.stack([d, 0.5 * d, 0.25 * d, 0.125 * d], 1)
        fs, Fs = self._trials(z, D, eps)
        j = int(np.argmin(fs))
        return D[:, j], f_here, fs[j], Fs[j], {"pcg": k + 1, "pcg res": res, "scale": 0.5**j,
                                                 "mu/lambda_max": mu / P["lam"], "|g|": float(jnp.linalg.norm(b))}

    # --- QR (no normal equations: sigma_min / sigma_max ~ 1e-13 here, squared it is lost)
    def _tsqr(self, z, eps, merge_rows=None, cperm=None):
        """R of [A | r] (n + 1, n + 1) by a streamed QR of the row blocks (with the <p> row), and f.
        speed merge=M: rows per factorization (the QR runs at ~24 TFLOP/s whatever the block, and each
        merge re-factors the n rows of R); gpus=G: one Python thread per device (the cuSOLVER calls
        block the host thread, so one thread would run them one after the other), chunks round-robin,
        the R factors merged on the first device; fp32: single precision. cperm: a column order of
        [A | r] for the factorization (speed elim: [p | u | r])."""
        from concurrent.futures import ThreadPoolExecutor
        merge_rows = int(self.speed.get("merge", 22000)) if merge_rows is None else merge_rows
        self._evict()
        prof = bool(self.speed.get("profile"))
        self._tt = {}
        tg = time.time()
        gv, G = self._global(z, eps)  # the global rows (<p>, transport, roughness), appended last
        if prof:
            self._tt["global"] = time.time() - tg
        devs = self._devs()
        nd = len(devs)
        dt = jnp.float32 if self.speed.get("fp32") else None
        cast = (lambda a: a.astype(dt)) if dt is not None else (lambda a: a)
        zs = [jax.device_put(z, d) for d in devs] if nd > 1 else [z]

        def leaf(g):
            dev = devs[g] if nd > 1 else None
            R, buf, nb = None, [], 0
            t_rows = t_qr = 0.0
            t_start = time.time()
            for k in range(g, self.n_chunks, nd):
                t0 = time.time()
                blk, _, _ = self._rows_chunk(self._chunk_tables(k, dev), zs[g], self._chunk_const(k, dev), eps)
                if cperm is not None:
                    blk = blk[:, cperm]
                if prof:
                    jax.block_until_ready(blk)
                    t_rows += time.time() - t0
                buf.append(cast(blk))
                nb += blk.shape[0]
                if nb >= merge_rows:
                    t0 = time.time()
                    R = _qr_r(jnp.concatenate(([R] if R is not None else []) + buf))
                    if prof:
                        jax.block_until_ready(R)
                        t_qr += time.time() - t0
                    buf, nb = [], 0
            if buf:
                t0 = time.time()
                R = _qr_r(jnp.concatenate(([R] if R is not None else []) + buf))
                if prof:
                    jax.block_until_ready(R)
                    t_qr += time.time() - t0
            if prof:
                self._tt[f"dev{g}"] = f"rows {t_rows:.2f} qr {t_qr:.2f} span {t_start % 1000:.1f}-{time.time() % 1000:.1f} nrows {nb}"
            return jax.block_until_ready(R) if R is not None else None

        tq = time.time()
        if nd > 1:
            with ThreadPoolExecutor(nd) as ex:
                Rs = list(ex.map(leaf, range(nd)))
        else:
            Rs = [leaf(0)]
        if prof:
            self._tt["leaves"] = time.time() - tq
        parts = [jax.device_put(R, devs[0]) if nd > 1 else R for R in Rs if R is not None]
        Gr = jnp.concatenate([G, gv[:, None]], 1)
        parts.append(cast(Gr if cperm is None else Gr[:, cperm]))
        del G
        R = _qr_r(jnp.concatenate(parts))
        return R.astype(jnp.float64) if dt is not None else R

    @functools.partial(jax.jit, static_argnums=(0,))
    def _hess2_chunk(self, tables, z, const, eps):
        """sum over the chunk's points of J_a . (B_b x v), v = swF^2 F: half of the residual-curvature
        term sum_i r_i grad^2 r_i of the force rows (J x B is bilinear in the unknowns)."""
        cols = self._synth(tables)
        J, B, p, gp, J0 = self._fields(cols, z, const, eps)
        v = const["swF"][..., None] ** 2 * (self._cross(J, B) - gp)
        Jm = jnp.moveaxis(jnp.concatenate([V[..., :3, :] for (kind, *_), V in zip(self._blocks_static, cols)
                                           if kind == "u"], -1), -1, 0)  # (K, c, t, z, 3)
        Bm = jnp.moveaxis(jnp.concatenate([V[..., 3:, :] for (kind, *_), V in zip(self._blocks_static, cols)
                                           if kind == "u"], -1), -1, 0)
        X = self._cross(Bm, v[None])
        return jnp.einsum("arstj,brstj->ab", Jm, X)

    def _hess2(self, z, eps):
        """The residual-curvature term of the Hessian on the u block, (n_uc, n_uc)."""
        M = 0.0
        for k in range(self.n_chunks):
            M = M + self._hess2_chunk(self._chunk_tables(k), z, self._chunk_const(k), eps)
        return M + M.T

    @functools.partial(jax.jit, static_argnums=(0,))
    def _metric_chunk(self, tables, const):
        """A chunk's share of M = sum_points w (J_i . J_j / rms_J^2 + p_i p_j / rms_p^2), (n, n)."""
        cols = self._synth(tables)
        X = []
        for (kind, n, P, a, b), V in zip(self._blocks_static, cols):
            if kind == "u":  # (c, t, z, 3, k) -> J / rms J, p part 0
                X.append(jnp.concatenate([V[..., :3, :] / self.rms["J"], jnp.zeros_like(V[..., :1, :])], -2))
            else:            # (c, t, z, 4, k): p, grad p -> J part 0, p / rms p
                X.append(jnp.concatenate([jnp.zeros_like(V[..., 1:, :]), V[..., :1, :] / self.rms["p"]], -2))
        X = jnp.concatenate(X, -1)
        w = const["sw3"] ** 2
        return jnp.einsum("rtzck,rtzcl->kl", X * w[..., None, None], X)

    def _phys_L(self, jit=1e-10):
        """Cholesky factor of the physical metric (built once: it depends only on the tables)."""
        if getattr(self, "_Lm", None) is None:
            t0 = time.time()
            order = [a for (_, _, _, a, _) in self._blocks_static]
            assert order == sorted(order) and self._blocks_static[-1][4] == self.n, "blocks not contiguous"
            M = 0.0
            for k in range(self.n_chunks):
                M = M + self._metric_chunk(self._chunk_tables(k), self._chunk_const(k))
            M = 0.5 * (M + M.T)
            d = jnp.diag(M)
            M = M.at[jnp.arange(self.n), jnp.arange(self.n)].add(jit * float(jnp.mean(d)))
            self._Lm = jnp.linalg.cholesky(M)
            ok = bool(jnp.all(jnp.isfinite(self._Lm)))
            self._log(f"    physical metric: built in {time.time() - t0:.1f} s, diag min/max "
                      f"{float(d.min()):.2e}/{float(d.max()):.2e}, jitter {jit:g} x mean, cholesky ok {ok}")
            assert ok, "physical metric not positive definite: raise the jitter"
            del M
        return self._Lm

    def _tr_step(self, z, eps, RD, c, Hs, gs, Sc, lam, f_here, max_tries=8, trials=None):
        """Trust-region step (speed tr): y(mu) = -(Hs + mu I)^-1 gs with |y(mu)| = Delta in the scaled
        variables of Sc (the physical rms change of J and p with speed phys), mu by safeguarded Newton on
        the secular equation (one Cholesky per mu); the step evaluated exactly; DESC's radius rule (x2 above
        ratio 0.75 on the boundary, 0.25 |y| below 0.25); a rejected step shrinks Delta and is retried.
        As DESC: the undamped Gauss-Newton step when it fits in the radius (mu = 0), no floor on mu.
        trials (speed elim): z, D (steps of the reduced variables) -> f, F and the full z-steps."""
        n = Hs.shape[0]
        Delta = self._Delta
        mu = self._mu_tr if self._mu_tr else 1e-8 * lam
        rho2 = max(2.0 * f_here - float(c @ c), 0.0)
        ii = jnp.arange(n)

        def solve(m):
            L = jnp.linalg.cholesky(Hs.at[ii, ii].add(m))
            return L, -jax.scipy.linalg.cho_solve((L, True), gs)

        # DESC's exact-QR rule (desc.optimize.tr_subproblems.trust_region_step_exact_qr): the undamped
        # Gauss-Newton step y_N = -(R D)^-1 c (one triangular solve; R D is upper triangular for both
        # scalings) is taken whenever |y_N| <= Delta; otherwise mu on the boundary |y(mu)| = Delta. There is
        # no floor on mu: the former floor 1e-13 lambda_max held every step of the production closure runs
        # (2026-10-23) there and damped exactly the soft directions that carry K.
        dr = jnp.diag(RD)
        dri = jnp.where(dr == 0, 0.0, 1.0 / jnp.where(dr == 0, 1.0, dr))  # as solve_triangular_regularized
        yN = -dri * jax.scipy.linalg.solve_triangular(RD * dri[:, None], c, lower=False, unit_diagonal=True)
        nN = float(jnp.linalg.norm(yN))
        mu_chol = 1e-10 * lam  # below this the Cholesky of Hs + mu I is not trusted (Hs squares kappa)
        tries = 0
        for _ in range(max_tries):
            tries += 1
            if np.isfinite(nN) and nN <= Delta:
                y, ny, mu, hit = yN, nN, 0.0, False
            else:
                hit = True
                L, y = solve(mu_chol)
                ny = float(jnp.linalg.norm(y))
                if np.isfinite(ny) and ny <= Delta:  # the root lies in (0, mu_chol): log bisection by QR
                    lo, hi, yh, nh = 1e-12 * mu_chol, mu_chol, y, ny
                    for _ in range(10):
                        mid = np.sqrt(lo * hi)
                        ym, _Rt = _damped_qr(RD, c, np.sqrt(mid))
                        del _Rt
                        nm = float(jnp.linalg.norm(ym))
                        if np.isfinite(nm) and nm <= Delta:
                            hi, yh, nh = mid, ym, nm
                        else:
                            lo = mid
                        if nh >= 0.95 * Delta:
                            break
                    y, ny, mu = yh, nh, hi
                else:  # the root lies above mu_chol: safeguarded Newton on the secular equation
                    mu, lo, hi = max(mu, mu_chol), mu_chol, np.inf
                    for _ in range(25):
                        L, y = solve(mu)
                        ny = float(jnp.linalg.norm(y))
                        if not np.isfinite(ny):
                            lo = mu
                            mu = np.sqrt(mu * hi) if np.isfinite(hi) else 10.0 * mu
                            continue
                        phi = ny - Delta
                        if abs(phi) <= 0.05 * Delta:
                            break
                        if phi > 0:
                            lo = mu
                        else:
                            hi = mu
                        q = jax.scipy.linalg.solve_triangular(L, y, lower=True)
                        mu_new = mu + (ny / float(jnp.linalg.norm(q))) ** 2 * phi / Delta
                        if not (np.isfinite(mu_new) and lo < mu_new < hi):
                            mu_new = np.sqrt(lo * hi) if np.isfinite(hi) else 10 * mu
                        mu = mu_new
            d = Sc.back(y)
            if trials is None:
                fs, Fs = self._trials(z, d[:, None], eps)
            else:
                fs, Fs, Dz = trials(z, d[:, None], eps)
                d = Dz[:, 0]
            f_new, F_new = float(fs[0]), float(Fs[0])
            r_lin = RD @ y + c
            pred = f_here - 0.5 * (float(r_lin @ r_lin) + rho2)
            ratio = (f_here - f_new) / pred if pred > 0 else 0.0
            if not np.isfinite(ratio) or ratio < 0.25:
                Delta = 0.25 * ny
            elif ratio > 0.75 and hit:
                Delta = max(2.0 * ny, Delta)
            if f_new < f_here:
                break
        self._Delta, self._mu_tr = Delta, mu
        return d, f_new, F_new, mu, {"tr ratio": float(ratio), "tr tries": tries, "Delta": float(Delta)}

    def _qr_step(self, z, eps, decades):
        """Exact LM steps from the QR of [A | r]: for each mu, the QR of [R D, c; sqrt(mu) I, 0];
        each step evaluated exactly, the best kept (with its factor, the LSQR preconditioner).
        Speed options: chol (Cholesky of (RD)^T RD + mu I for kd <= 10 instead of a QR per mu),
        narrow (mu scan only around the last best), newton (+ D H2 D, the residual curvature)."""
        if self.speed.get("elim"):
            return self._elim_step(z, eps, decades)
        n = self.n
        sp = self.speed
        prof = bool(sp.get("profile"))
        tm = {}

        def tick(name, t0, *arrs):
            if prof:
                for a_ in arrs:
                    jax.block_until_ready(a_)
                tm[name] = tm.get(name, 0.0) + time.time() - t0

        self._prec = None
        t0 = time.time()
        Ra = self._tsqr(z, eps)
        tick("tsqr", t0, Ra)
        R, c = Ra[:n, :n], Ra[:n, n]
        f_here = 0.5 * float(c @ c + Ra[n, n] ** 2)
        # what a full Gauss-Newton step could still remove: f - |r_perp|^2 / 2, r_perp the part of r
        # outside the range of A (the exact linear model's minimum)
        gn_share = float(c @ c) / float(c @ c + Ra[n, n] ** 2)
        self._gn_last = gn_share
        Dsc = self._scaling(jnp.sum(R**2, axis=0))
        if sp.get("phys"):  # damping in the physical (J, p) metric instead of Jacobi
            assert self._steps == "qr" and not sp.get("newton"), "phys: QR steps only, without newton"
            Sc = _Scale(L=self._phys_L())
        else:
            Sc = _Scale(D=Dsc)
        RD = Sc.cols(R)
        del Ra, R
        v = jnp.ones(n) / np.sqrt(n)
        for _ in range(30):  # lambda_max of (R D)^T (R D)
            v = RD.T @ (RD @ v)
            lam = float(jnp.linalg.norm(v))
            v = v / lam
        chol = bool(sp.get("chol") or sp.get("newton") or sp.get("reuse") or sp.get("soc"))
        Hs = gs = None
        if chol:
            t0 = time.time()
            Hs = RD.T @ RD
            gs = RD.T @ c
            if sp.get("newton"):
                assert self._par == 1.0
                H2 = self._hess2(z, eps)
                K = H2.shape[0]
                Hs = Hs.at[:K, :K].add(Dsc[:K, None] * H2 * Dsc[None, :K])
                del H2
            tick("gram", t0, Hs)
        use_tr = bool(sp.get("tr")) and self._Delta is not None
        if use_tr:  # trust region: mu from |y(mu)| = Delta (the metric of Sc), exact trial, DESC radius rule
            t0 = time.time()
            if Hs is None:
                Hs, gs = RD.T @ RD, RD.T @ c
            d, fbest, Fbest, mu, trinfo = self._tr_step(z, eps, RD, c, Hs, gs, Sc, lam, f_here)
            kdl, j = [float("nan")], 0
            tick("tr", t0)
        else:
            t0 = time.time()
            kds = list(np.arange(decades[0] - 3, decades[1] + 0.5, 2.0))
            if sp.get("narrow") and self._kd_last is not None:
                kds = [k for k in kds if abs(k - self._kd_last) <= 4.0]
            steps, mus, kdl = [], [], []
            for kd in kds:
                mu = lam * 10.0 ** (-kd)
                if chol and (kd <= 10 or sp.get("newton")):
                    if sp.get("newton") and kd > 12:
                        continue
                    y, ok = _damped_step(Hs, mu, gs)
                    ok = bool(ok)
                else:
                    y, Rt = _damped_qr(RD, c, np.sqrt(mu))
                    del Rt
                    ok = bool(jnp.all(jnp.isfinite(y)))
                if ok:
                    steps.append(Sc.back(y))
                    mus.append(mu)
                    kdl.append(kd)
            D = jnp.stack(steps, 1)
            tick("scan", t0, D)
            t0 = time.time()
            fs, Fs = self._trials(z, D, eps)
            tick("trials", t0)
            j = int(np.argmin(fs))
            fbest, Fbest, d, mu = fs[j], Fs[j], D[:, j], mus[j]
            self._kd_last = kdl[j]
            del D
            if sp.get("tr"):  # first iteration: the scan initializes the trust radius
                self._Delta, self._mu_tr = float(jnp.linalg.norm(Sc.inv(d))), mu
            trinfo = {}
        if sp.get("soc"):  # second-order correction with the kept factor (the exact quadratic part q(d, d))
            t0 = time.time()
            L = jnp.linalg.cholesky(Hs.at[jnp.arange(n), jnp.arange(n)].add(mu))
            y0, dcur = Sc.inv(d), d
            rs0, gv0 = self._rvec(z, eps)
            Gz = self._Gz
            cands = []
            for _ in range(int(sp["soc"]) if int(sp["soc"]) > 1 else 2):
                rs1, gv1 = self._rvec(z + dcur, eps)
                self._Gz = Gz
                aq, aW = self._Av(z, dcur, eps)
                q = [r1 - r0 - a_ for r1, r0, a_ in zip(rs1, rs0, aq)]
                qg = gv1 - gv0 - aW
                t = self._ATu(z, q, qg, eps)
                dcur = Sc.back(y0 - jax.scipy.linalg.cho_solve((L, True), Sc.grad(t)))
                cands.append(dcur)
            del L, rs0, rs1, q
            Dc = jnp.stack(cands + [0.5 * cands[-1]], 1)
            fc, Fc = self._trials(z, Dc, eps)
            jc = int(np.argmin(fc))
            if fc[jc] < fbest:
                fbest, Fbest, d = fc[jc], Fc[jc], Dc[:, jc]
            tick("soc", t0)
        if self._steps != "qr":  # the LSQR preconditioner
            _, Rt = _damped_qr(RD, c, np.sqrt(mu))
            self._prec = {"Rt": Rt, "Dsc": Dsc, "mu": mu, "lam": lam}
        if sp.get("reuse"):
            i = jnp.arange(n)
            self._chord = {"L": jnp.linalg.cholesky(Hs.at[i, i].add(mu)), "Sc": Sc, "mu": mu, "lam": lam,
                           "gn": gn_share, "age": 0}
        del RD, Hs, gs
        self._reload()
        if prof:
            self._log("    tsqr parts: " + ", ".join(f"{k} {v_ if isinstance(v_, str) else format(v_, '.2f')}"
                                                    for k, v_ in getattr(self, "_tt", {}).items()))
            self._log("    time: " + ", ".join(f"{k} {v_:.1f} s" for k, v_ in tm.items())
                      + f"; kd {kdl[j]:g} of {len(kdl)} scanned")
        return d, f_here, fbest, Fbest, {"mu/lambda_max": mu / lam, "GN share": gn_share, **trinfo}

    # ------------------------------------------------------------------ speed elim: p eliminated
    # The force rows are affine in p with constant coefficients (par = 1, w_pos = 0, no transport), so
    # min_p |r(u, p)| is the weak Poisson equation of force balance with the force rows' weights,
    #   sum_points w grad(phi_i) . grad p = sum_points w grad(phi_i) . (J x B)   (+ the constant <p> row),
    # on the pressure space (p(1) = 0). The LM / trust-region step is taken on u alone with the reduced
    # Jacobian (I - P) A_u (exact: P is constant): the trailing block of the QR of [A_p | A_u | r]. Every
    # trial re-solves p exactly, so every accepted iterate has p = p*(u) (variable projection).
    def _elim_check(self, steps="qr"):
        sp = self.speed
        bad = [k for k in ("newton", "soc", "reuse", "pproj", "accel") if sp.get(k)]
        assert not bad, f"speed elim: not with {bad}"
        assert steps == "qr", "speed elim: QR steps only"
        assert self._par == 1.0, "speed elim: needs par = 1 (A_p depends on b otherwise)"
        assert self.transport is None, "speed elim: not with transport (A_p depends on b)"
        assert float(self.weights["pos"]) == 0.0, "speed elim: needs w_pos = 0 (p<0 rows are not affine in p)"
        if not hasattr(self, "_elim_cols"):
            nu, n = self.n_uc, self.n
            self._elim_cols = np.r_[np.arange(nu, n), np.arange(nu), [n]]  # numpy: valid on every device (gpus=G)
            self._Rpp, self._Gpp0 = None, None

    def _elim_Rpp_build(self, z, eps):
        """R of the constant A_p (force rows' p columns and the <p> row) by a streamed QR, p columns only."""
        nu, n = self.n_uc, self.n
        t0 = time.time()
        self._global(z, eps, jac=False)  # sets self._gW
        R, buf, nb = None, [], 0
        for k in range(self.n_chunks):
            blk, _, _ = self._rows_chunk(self._chunk_tables(k), z, self._chunk_const(k), eps)
            buf.append(blk[:, nu:n])
            nb += blk.shape[0]
            if nb >= 22000:
                R, buf, nb = _qr_r(jnp.concatenate(([R] if R is not None else []) + buf)), [], 0
        buf.append(self.wW * self._gW[None, nu:])
        R = _qr_r(jnp.concatenate(([R] if R is not None else []) + buf))
        self._log(f"    elim: R_pp ({self.n_pc}) built in {time.time() - t0:.1f} s, "
                  f"diag min/max {float(jnp.abs(jnp.diag(R)).min()):.2e}/{float(jnp.abs(jnp.diag(R)).max()):.2e}")
        return R

    def _elim_set_Rpp(self, Rpp):
        """Keep R_pp; check that A_p^T A_p is unchanged (it must be constant)."""
        G = Rpp.T @ Rpp
        if self._Gpp0 is None:
            self._Gpp0 = G
        else:
            dev = float(jnp.linalg.norm(G - self._Gpp0) / jnp.linalg.norm(self._Gpp0))
            if dev > 1e-10:
                self._log(f"    elim WARNING: A_p^T A_p changed by {dev:.1e} (relative); is A_p constant?")
        self._Rpp = Rpp

    @functools.partial(jax.jit, static_argnums=(0,))
    def _elim_trial_chunk(self, tables, z, D, const, eps):
        """Per trial z + D[:, k]: |local rows|^2, the share of <p> and <|F|>, and A_p^T r of the local rows."""
        cols = self._synth(tables)

        def one(d):
            J, B, p, gp, J0 = self._fields(cols, z + d, const, eps)
            rF, rJ, rP, Wp, rF0 = self._residual(J, B, p, gp, J0, const)
            Fa = jnp.sum(const["sw3"] * jnp.linalg.norm(rF0, axis=-1))
            wF = self._wF(const, B, rF)
            g = [-jnp.einsum("rtzck,rtzc->k", V[..., 1:, :], wF)
                 for (kind, *_), V in zip(self._blocks_static, cols) if kind == "p"]
            return jnp.sum(rF**2) + jnp.sum(rJ**2) + jnp.sum(rP**2), Wp, Fa, jnp.concatenate(g)

        return jax.lax.map(one, D.T)

    def _elim_trials(self, z, Du, eps):
        """Trials u + Du[:, k] with p re-solved exactly: f*(u + Du_k), <|F|>/norm before the p update
        (the accepted step is re-evaluated), and the full z-steps [Du_k; dp_k] (n, K)."""
        nu = self.n_uc
        K = Du.shape[1]
        D = jnp.concatenate([Du, jnp.zeros((self.n_pc, K), Du.dtype)], 0)
        fs, Ws, Fs, Gp = 0.0, 0.0, 0.0, 0.0
        for k in range(self.n_chunks):
            s, w_, Fa, g = self._elim_trial_chunk(self._chunk_tables(k), z, D, self._chunk_const(k), eps)
            fs, Ws, Fs, Gp = fs + s, Ws + w_, Fs + Fa, Gp + g
        fs, Ws, Fs = np.asarray(fs), np.asarray(Ws), np.asarray(Fs)
        rW = self.weights["W"] * (Ws - self.W_t) / self.W_t
        g2 = rW**2
        if self.mom is not None:
            S = 0.0
            for k in range(self.n_chunks):
                S = S + self._mom_trial_chunk(self._chunk_tables(k), z, D, self._chunk_const(k), eps)
            g2 = g2 + np.asarray(jnp.sum(self._mom_resid(S) ** 2, -1))
        if self.rough is not None:
            ri = self._rough_idx
            g2 = g2 + np.asarray(jnp.sum((self._dvec_j[ri, None] * (z[ri, None] + D[ri])) ** 2, 0))
        gP = Gp + (self.wW * self._gW[nu:])[None, :] * jnp.asarray(rW)[:, None]  # (K, n_pc): A_p^T r
        y = jax.scipy.linalg.solve_triangular(self._Rpp, gP.T, lower=False, trans="T")  # R_pp^-T A_p^T r
        dP = -jax.scipy.linalg.solve_triangular(self._Rpp, y, lower=False)
        f_star = 0.5 * (fs + g2) - 0.5 * np.asarray(jnp.sum(y**2, 0))
        return f_star, Fs, jnp.concatenate([Du, dP], 0)

    def _elim_project(self, z, eps):
        """p <- p*(u) (the weak Poisson solve) at the start; returns z."""
        self._elim_set_Rpp(self._elim_Rpp_build(z, eps))
        f_star, _, Dz = self._elim_trials(z, jnp.zeros((self.n_uc, 1)), eps)
        self._log(f"    elim: start projection p -> p*(u), f* {float(f_star[0]):.6e}, "
                  f"|dp|/|p| {float(jnp.linalg.norm(Dz[self.n_uc:, 0]) / max(float(jnp.linalg.norm(z[self.n_uc:])), 1e-300)):.2e}")
        return z + Dz[:, 0]

    @functools.partial(jax.jit, static_argnums=(0,))
    def _prange_chunk(self, tables, z, const, eps):
        J, B, p, gp, J0 = self._fields(self._synth(tables), z, const, eps)
        return jnp.min(p), jnp.max(p)

    def p_range(self, z, eps):
        """min p and max p over the collocation points (p >= 0 is not enforced with speed elim)."""
        lo, hi = np.inf, -np.inf
        for k in range(self.n_chunks):
            a, b = self._prange_chunk(self._chunk_tables(k), z, self._chunk_const(k), eps)
            lo, hi = min(lo, float(a)), max(hi, float(b))
        return lo, hi

    def _elim_step(self, z, eps, decades):
        """One LM / trust-region step on u with p eliminated (see above); returns as ``_qr_step``."""
        n, nu, npc = self.n, self.n_uc, self.n_pc
        sp = self.speed
        t0 = time.time()
        Ra = self._tsqr(z, eps, cperm=self._elim_cols)
        t_qr = time.time() - t0
        self._elim_set_Rpp(Ra[:npc, :npc])
        cp = Ra[:npc, n]
        T = Ra[npc:, npc:]  # QR of (I - P)[A_u | r]
        R, c, rho = T[:nu, :nu], T[:nu, nu], float(T[nu, nu])
        del Ra
        cc = float(c @ c)
        f_here = 0.5 * (cc + rho**2)
        cp_rel = float(jnp.linalg.norm(cp)) / np.sqrt(float(cp @ cp) + cc + rho**2)
        gn_share = cc / (cc + rho**2)
        self._gn_last = gn_share
        Dsc = self._scaling(jnp.concatenate([jnp.sum(R**2, axis=0), jnp.ones(npc)]))[:nu]
        Sc = _Scale(L=self._phys_L()[:nu, :nu]) if sp.get("phys") else _Scale(D=Dsc)  # metric block diagonal
        RD = Sc.cols(R)
        del R
        v = jnp.ones(nu) / np.sqrt(nu)
        for _ in range(30):
            v = RD.T @ (RD @ v)
            lam = float(jnp.linalg.norm(v))
            v = v / lam
        Hs, gs = RD.T @ RD, RD.T @ c
        trinfo = {}
        if sp.get("tr") and self._Delta is not None:
            d, fbest, Fbest, mu, trinfo = self._tr_step(z, eps, RD, c, Hs, gs, Sc, lam, f_here,
                                                        trials=self._elim_trials)
        else:
            kds = list(np.arange(decades[0] - 3, decades[1] + 0.5, 2.0))
            if sp.get("narrow") and self._kd_last is not None:
                kds = [k for k in kds if abs(k - self._kd_last) <= 4.0]
            steps, mus, kdl = [], [], []
            for kd in kds:
                mu = lam * 10.0 ** (-kd)
                if sp.get("chol") and kd <= 10:
                    y, ok = _damped_step(Hs, mu, gs)
                    ok = bool(ok)
                else:
                    y, _Rt = _damped_qr(RD, c, np.sqrt(mu))
                    del _Rt
                    ok = bool(jnp.all(jnp.isfinite(y)))
                if ok:
                    steps.append(Sc.back(y))
                    mus.append(mu)
                    kdl.append(kd)
            fs, Fs, Dz = self._elim_trials(z, jnp.stack(steps, 1), eps)
            j = int(np.argmin(fs))
            fbest, d, mu = float(fs[j]), Dz[:, j], mus[j]
            self._kd_last = kdl[j]
            if sp.get("tr"):
                self._Delta, self._mu_tr = float(jnp.linalg.norm(Sc.inv(d[:nu]))), mu
        del RD, Hs, gs
        self._reload()
        f_chk, F_chk = self.evaluate(self.to_x(z + d), eps)  # the accepted point, p re-solved
        p_lo, p_hi = self.p_range(z + d, eps)
        info = {"mu/lambda_max": mu / lam, "GN share": gn_share, "|c_p|/|r|": cp_rel, "p min/max": p_lo / p_hi,
                "f* check": abs(f_chk - fbest) / max(abs(fbest), 1e-300), "qr time": t_qr, **trinfo}
        return d, f_here, f_chk, F_chk, info

    def _chord_step(self, z, eps):
        """A step from the exact gradient at z with the kept Cholesky factor of the last QR step's
        (scaled) Gauss-Newton matrix, evaluated exactly with its halvings."""
        P = self._chord
        g, f_here, _ = self._grad(z, eps)
        d = P["Sc"].back(-jax.scipy.linalg.cho_solve((P["L"], True), P["Sc"].grad(g)))
        D = jnp.stack([d, 0.5 * d, 0.25 * d, 0.125 * d], 1)
        fs, Fs = self._trials(z, D, eps)
        j = int(np.argmin(fs))
        P["age"] += 1
        return D[:, j], f_here, fs[j], Fs[j], {"mu/lambda_max": P["mu"] / P["lam"], "GN share": P["gn"],
                                               "chord": P["age"]}

    def _pproj(self, z, eps, f_new, F):
        """The exact minimization over p alone (the force rows are affine in p): z_p -= (A_p^T A_p)^-1 g_p,
        the Cholesky factor of A_p^T A_p built once (A_p is constant); returns z, f, <|F|>/norm."""
        n, n_uc = self.n, self.n_uc
        if getattr(self, "_Hpp_L", None) is None:
            t0 = time.time()
            H = jnp.zeros((n - n_uc, n - n_uc))
            for k in range(self.n_chunks):
                blk, _, _ = self._rows_chunk(self._chunk_tables(k), z, self._chunk_const(k), 0.0)
                Ap = blk[:, n_uc:n]
                H = H + Ap.T @ Ap
            H = H + self.wW**2 * jnp.outer(self._gW[n_uc:], self._gW[n_uc:]) if self._gW is not None else H
            jit = 1e-9 * float(jnp.mean(jnp.diag(H)))
            self._Hpp_L = jnp.linalg.cholesky(H + jit * jnp.eye(n - n_uc))
            self._log(f"    pproj: A_p^T A_p built in {time.time() - t0:.1f} s")
        g, f0, _ = self._grad(z, eps)
        dp = -jax.scipy.linalg.cho_solve((self._Hpp_L, True), g[n_uc:])
        z2 = z.at[n_uc:].add(dp)
        f1, F1 = self.evaluate(self.to_x(z2), eps)
        if f1 < f_new:
            return z2, f1, F1
        return z, f_new, F

    def _rvec(self, z, eps):
        """The local residuals (per chunk) and the global ones; keeps the global Jacobian at z."""
        rs = []
        for k in range(self.n_chunks):
            r, _ = self._r_chunk(self._chunk_tables(k), z, self._chunk_const(k), eps)
            rs.append(r)
        gv, self._Gz = self._global(z, eps)
        return rs, gv

    def _Av(self, z, d, eps):
        out = []
        for k in range(self.n_chunks):
            q, _ = self._Av_chunk(self._chunk_tables(k), z, d, self._chunk_const(k), eps)
            out.append(q)
        return out, self._Gz @ d

    def _ATu(self, z, u, ug, eps):
        out = jnp.zeros(self.n)
        for k in range(self.n_chunks):
            out = out + self._ATu_chunk(self._chunk_tables(k), z, u[k], self._chunk_const(k), eps)
        return out + self._Gz.T @ ug

    def _lsqr_step(self, z, eps, maxit=20, tol=1e-6):
        """The LM step for the kept damping by LSQR on [A D; sqrt(mu) I] Rt^-1 w = [-r; 0] (right
        preconditioned with the kept QR factor: exact products, no normal equations)."""
        from jax.scipy.linalg import solve_triangular
        P = self._prec
        Dsc, mu, Rt = P["Dsc"], P["mu"], P["Rt"]
        sm = np.sqrt(mu)
        Ri = lambda w: solve_triangular(Rt, w, lower=False)  # noqa: E731
        RiT = lambda w: solve_triangular(Rt, w, lower=False, trans="T")  # noqa: E731

        def op(w):
            y = Ri(w)
            a, aW = self._Av(z, Dsc * y, eps)
            return (a, aW, sm * y)

        def opT(u):
            a, aW, v = u
            return RiT(Dsc * self._ATu(z, a, aW, eps) + sm * v)

        dot = lambda u, v: sum(float(x @ y) for x, y in zip(u[0], v[0])) + float(u[1] @ v[1]) + float(u[2] @ v[2])  # noqa: E731
        scal = lambda s_, u: ([s_ * x for x in u[0]], s_ * u[1], s_ * u[2])  # noqa: E731
        add = lambda u, v: ([x + y for x, y in zip(u[0], v[0])], u[1] + v[1], u[2] + v[2])  # noqa: E731
        rs, rW = self._rvec(z, eps)
        f_here = 0.5 * (sum(float(x @ x) for x in rs) + float(rW @ rW))
        b = ([-x for x in rs], -rW, jnp.zeros(self.n))
        beta = np.sqrt(dot(b, b))
        u = scal(1 / beta, b)
        v = opT(u)
        alpha = float(jnp.linalg.norm(v))
        v = v / alpha
        w = v
        x = jnp.zeros(self.n)
        phibar, rhobar = beta, alpha
        k, res = 0, 1.0
        anorm = alpha
        while k < maxit:
            u = add(op(v), scal(-alpha, u))
            beta = np.sqrt(dot(u, u))
            u = scal(1 / beta, u)
            v = opT(u) - beta * v
            alpha = float(jnp.linalg.norm(v))
            v = v / alpha
            anorm = max(anorm, np.hypot(alpha, beta))
            rho = np.hypot(rhobar, beta)
            cs, sn = rhobar / rho, beta / rho
            theta = sn * alpha
            rhobar = -cs * alpha
            phi = cs * phibar
            phibar = sn * phibar
            x = x + (phi / rho) * w
            w = v - (theta / rho) * w
            k += 1
            res = abs(phibar * alpha * cs) / (anorm * max(phibar, 1e-300))  # |A^T r| / (|A| |r|)
            if res < tol:
                break
        d = Dsc * Ri(x)
        D = jnp.stack([d, 0.5 * d, 0.25 * d, 0.125 * d], 1)
        fs, Fs = self._trials(z, D, eps)
        j = int(np.argmin(fs))
        return D[:, j], f_here, fs[j], Fs[j], {"lsqr": k, "lsqr res": float(res), "scale": 0.5**j,
                                                 "mu/lambda_max": mu / P["lam"]}

    def lm(self, x, eps, iters=20, rtol=1e-7, decades=(1, 17), callback=None, steps="qr", gn_tol=1e-6):
        """Levenberg-Marquardt from x at amplitude eps.

        ``steps="qr"``: an exact QR step every iteration, and a stop once that step's exact linear
        model says a full Gauss-Newton step could remove less than ``gn_tol`` of f (or f drops by
        less than ``rtol``). ``"lsqr"``: QR steps only to refresh the preconditioner of LSQR steps
        (for problems too large to factor every iteration)."""
        z = self.to_z(x)
        hist = []
        self._steps, self._chord, self._kd_last, self._zhist = steps, None, None, []
        self._Delta, self._mu_tr = None, None  # speed tr: the trust radius (scaled variables), last mu
        if self.speed.get("elim"):  # p eliminated: start from p = p*(u)
            self._elim_check(steps)
            fj, _ = self.evaluate(self.to_x(z), eps)
            self._log(f"  eps {eps:g}: speed elim, joint f before the start projection {fj:.6e}")
            z = self._elim_project(z, eps)
        f0, F0 = self.evaluate(self.to_x(z), eps)
        hist.append({"it": 0, "f": f0, "<|F|>/norm": F0, "time": 0.0})
        self._log(f"  eps {eps:g}: start f {f0:.6e}, <|F|>/norm {F0:.4e}")
        if not hasattr(self, "_prec"):
            self._prec = None
        for it in range(1, iters + 1):
            t0 = time.time()
            kind = "QR" if steps == "qr" or self._prec is None or "Rt" not in self._prec else "LSQR"
            if kind == "LSQR":
                last = hist[-1].get("rel decrease", 1.0) if len(hist) > 1 else 1.0
                dz, f_here, f_new, F, info = self._lsqr_step(z, eps, tol=float(np.clip(last, 1e-8, 1e-3)))
                if not f_new < f_here:  # a stale preconditioner: refactor now
                    kind = "QR"
                elif info["lsqr"] >= 15:  # still a decrease, but refactor next time
                    self._prec = None
            if kind == "QR" and self.speed.get("reuse") and self._chord is not None \
                    and self._chord["age"] < int(self.speed["reuse"]) - 1:
                dz, f_here, f_new, F, info = self._chord_step(z, eps)
                if f_new < f_here:
                    kind = "CHD"
                else:  # a stale factor: a full QR step instead
                    self._chord = None
            if kind == "QR":
                dz, f_here, f_new, F, info = self._qr_step(z, eps, decades)
            if not f_new < f_here:
                self._log(f"  it {it}: no decrease ({kind}); stopping")
                break
            rel = (f_here - f_new) / f_here
            z = z + dz
            if self.speed.get("pproj"):  # p re-solved exactly for the new J (the force is affine in p)
                z, f_new, F = self._pproj(z, eps, f_new, F)
            if int(self.speed.get("accel", 0)) > 0:  # extrapolate along the last m steps, exact search
                m_ = int(self.speed["accel"])
                self._zhist = (getattr(self, "_zhist", []) + [z])[-(m_ + 1):]
                if len(self._zhist) > m_:
                    pdir = z - self._zhist[0]
                    Dx = jnp.stack([al * pdir for al in (0.5, 1.0, 2.0, 4.0, 8.0)], 1)
                    fx, Fx = self._trials(z, Dx, eps)
                    jx = int(np.argmin(fx))
                    if fx[jx] < f_new:
                        z = z + Dx[:, jx]
                        f_new, F = fx[jx], Fx[jx]
                        self._zhist[-1] = z
            rel = (f_here - f_new) / f_here
            h = {"it": it, "f": float(f_new), "<|F|>/norm": float(F), "|d|/|x|": float(jnp.linalg.norm(dz) / jnp.linalg.norm(z)),
                 "rel decrease": float(rel), "step": kind, "time": time.time() - t0, **info}
            hist.append(h)
            self._log(f"  it {it:2d} {kind:4s} f {f_new:.6e}  <|F|>/norm {F:.4e}  |dx|/|x| {h['|d|/|x|']:.2e}  "
                      f"rel {rel:.1e}  " + (f"lsqr {info['lsqr']} res {info['lsqr res']:.0e}  " if kind == "LSQR" else
                                            f"mu/lam {info['mu/lambda_max']:.0e}  GN share {info['GN share']:.1e}  "
                                            + (f"ratio {info['tr ratio']:.2f} tries {info['tr tries']}  " if 'tr ratio' in info else "")
                                            + (f"|c_p|/|r| {info['|c_p|/|r|']:.0e} f* chk {info['f* check']:.0e}  "
                                               f"p min/max {info['p min/max']:.1e}  "
                                               if '|c_p|/|r|' in info else ""))
                      + f"{h['time']:.0f} s")
            if callback is not None:
                callback(self.to_x(z), hist)
            if rel < rtol or info.get("GN share", 1.0) < gn_tol:
                break
        return self.to_x(z), hist
