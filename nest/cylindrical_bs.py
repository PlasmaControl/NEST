"""Free-space Biot-Savart J -> B on one field period of a toroidal annulus (JAX).

Grid: N_phi (odd) planes phi_k = 2 pi k / (NFP N_phi), Chebyshev nodes in r on
[R_in, R_out] and in z on [-z0, z0]; fields have shape (N_phi, n_r + 1, n_z + 1)
with cylindrical components (V_r, V_phi, V_z) last.

With A the Coulomb-gauge vector potential, Delta A = -mu0 J, B = curl A, and the
complex combination A_+ = A_x + i A_y = e^{i phi} (A_r + i A_phi):

    A_z    real, modes m = j NFP        (rfft of -mu0 J_z over the period)
    A_+    modes m = j NFP + 1          (fft of -mu0 (J_r + i J_phi) over the period)

each mode a 2-D free-space problem in (r, z) solved with James' correction
(``annulus_freespace``, gate A6). The curl, per FFT index j:

    B_+^(j) = i d_z A_+^(j) - i (d_r - j NFP / r) A_z^(j)
    psi^(j) = (d_r + (j NFP + 1) / r) A_+^(j),     B_z^(j) = (psi^(j) - conj psi^(-j)) / 2i

(A_z^(-j) = conj A_z^(j)); the inverse FFT of B_+^ is B_r + i B_phi and the
inverse rfft of B_z^ is B_z.

The precomputation (eigendecompositions, single-layer matrices) is NumPy
(``FreeSpaceAnnulus``); ``CylindricalSolver`` holds it as a pytree, so that a
jitted call takes the operators as inputs. The mode solves run as a
``lax.map`` over chunks of ``chunk`` modes: transient memory is that of one
chunk (its V_r, V_r^-1 and S_m gathered from the distinct-|m| tables).
Real operators act on complex data as two real products.

On several devices (``mesh``): the mode pairs +-j of ``planes_F`` are split into
one block per device, which holds only the per-mode tables of its own |m|, forms
only its own source slabs and solves its pairs; the planes are summed over the
devices (``planes_sharded``, a ``shard_map``).
"""

import functools

import jax
import jax.numpy as jnp
import numpy as np
from scipy.constants import mu_0

from annulus_freespace import FreeSpaceAnnulus, mode_indices
from flux_interp import _Pytree


def _lmul(V, X):
    """Batched real V (c, a, b) times complex X (c, b, d)."""
    return (jnp.einsum("kab,kbd->kad", V, X.real) + 1j * jnp.einsum("kab,kbd->kad", V, X.imag))


def _rmul(X, M):
    """Complex X (..., a, b) times real M (b, d)."""
    return X.real @ M + 1j * (X.imag @ M)


def _along(D, X, axis):
    """Real matrix D applied along ``axis`` of complex or real X."""
    if jnp.iscomplexobj(X):
        return _along(D, X.real, axis) + 1j * _along(D, X.imag, axis)
    return jnp.moveaxis(jnp.tensordot(D, X, axes=(1, axis)), 0, axis)


def _varying(x, axis):
    """x typed as different on each device of the mesh axis (inside ``shard_map``: a scan
    carry started from zeros and summed into per device)."""
    if hasattr(jax.lax, "pcast"):
        return jax.lax.pcast(x, axis, to="varying")
    return jax.lax.pvary(x, axis)


def replicate(tree, mesh, _seen=None):
    """tree with its arrays on every device of ``mesh``: those already on exactly these
    devices are kept (e.g. a solver's split tables), the others copied once (a jitted
    call would otherwise copy each uncommitted input there at every call). Arrays shared
    between leaves stay shared."""
    from jax.sharding import NamedSharding, PartitionSpec
    rep = NamedSharding(mesh, PartitionSpec())
    devs = set(mesh.devices.flat)
    seen = {} if _seen is None else _seen

    def put(a):
        if not isinstance(a, (jax.Array, np.ndarray)):
            return a
        if isinstance(a, jax.Array) and set(a.sharding.device_set) == devs:
            return a
        if id(a) not in seen:
            seen[id(a)] = (a, jax.device_put(a, rep))
        return seen[id(a)][1]
    return jax.tree_util.tree_map(put, tree)


def pair_blocks(N_phi, chunk, n_dev, kz, kp):
    """The blocks of mode pairs of ``CylindricalSolver._distribute``: jv, ok (n_dev, L)
    the pairs j of each device (L a multiple of chunk // 2; padded with the block's first
    pair, ok False), and use[d] the sorted distinct-|m| indices device d needs (of
    kz[j], kp[j], kp[-j]; kz, kp as ``CylindricalSolver.kz``, ``.kp``)."""
    M = N_phi // 2
    c = max(1, chunk // 2)
    per = -(-(M + 1) // n_dev)  # pairs per device
    L = -(-per // c) * c
    jv = np.zeros((n_dev, L), int)
    ok = np.zeros((n_dev, L), bool)
    for d in range(n_dev):
        js = np.arange(d * per, min((d + 1) * per, M + 1))
        jv[d] = js[0] if len(js) else 0  # padding: a pair of the block (no extra rows)
        jv[d, :len(js)], ok[d, :len(js)] = js, True
    use = [np.unique(np.concatenate([kz[jv[d]], kp[jv[d]], kp[(-jv[d]) % N_phi]]))
           for d in range(n_dev)]
    return jv, ok, use


class CylindricalSolver(_Pytree):
    """J (cylindrical components on the annulus grid) -> A or B, free space.

    Parameters
    ----------
    r_axis, z_axis : ChebAxis
    NFP, N_phi : int (N_phi odd)
    chunk : int
        Modes per step of the mode loop.
    fs : FreeSpaceAnnulus, optional
        A precomputation to reuse (shifts 0 and 1).
    S_dtype : float64 or float32
        Storage of the single-layer matrices (the largest precomputed array);
        products are formed in ``dtype``.
    dtype : float64 or float32
        Of the operators, the FFTs and the mode solves (float32: complex64 slabs).
    keep_numpy : bool
        Keep the NumPy S_m of ``fs`` (for the NumPy reference solve); by default it
        is released once the JAX copy exists, so only one copy is held.
    mesh : jax.sharding.Mesh, optional
        Devices to split the mode pairs of ``planes_F`` over (along ``mesh_axis``,
        default the mesh's first axis): each holds the V_r, V_r^-1, lam_r, a_r and S_m
        rows of its own |m| only (``_distribute``), the small operators replicated.
        Only ``planes_F`` / ``planes_sharded`` (``cyl_fused``) run on such a solver.
    """

    _dynamic = ("Vr", "Vr_inv", "lam_r", "lam_s", "pair_r", "ar", "Vz", "Vz_inv", "lam_z", "ez", "S", "S_o",
                "Dr", "Dz", "r", "kz", "kp", "mz", "mp", "pairs", "sel")

    def __init__(self, r_axis, z_axis, NFP, N_phi, chunk=16, fs=None, verbose=False,
                 S_dtype=np.float64, keep_numpy=False, dtype=jnp.float64, mesh=None,
                 mesh_axis=None):
        fs = fs or FreeSpaceAnnulus(r_axis, z_axis, NFP, N_phi, shifts=(0, 1), verbose=verbose,
                                    S_dtype=S_dtype)
        ap = fs.ap
        self.NFP, self.N_phi, self.chunk = int(NFP), int(N_phi), int(chunk)
        self.shape = (self.N_phi,) + ap.shape
        self.nr, self.nz = ap.nr, ap.nz
        self.face_sizes = ap.face_sizes
        self.dtype = jnp.dtype(dtype)  # of the operators and the arithmetic (float32: complex64)
        jz = mode_indices(N_phi, real=True)
        jp = mode_indices(N_phi, real=False)
        self.jz, self.jp = jz, jp
        kz, kp = ap.k_of(jz * NFP), ap.k_of(jp * NFP + 1)
        # complex-pair blocks of mapped axes (``annulus_poisson._block_eig``); has_pairs is static
        self.has_pairs = bool(getattr(ap, "has_pairs", False))
        if self.has_pairs and mesh is not None:
            raise NotImplementedError("mapped axes with complex pairs are not distributed over a mesh")
        self.lam_s = self.pair_r = None
        self.zfold = fs.zfold
        self.S_nbytes = fs.S_nbytes
        self.mesh, self.mesh_axis, self.pairs, self.sel = mesh, None, None, None
        put = jnp.asarray
        if mesh is None:
            f = lambda a: jnp.asarray(a, self.dtype)  # noqa: E731
            self.Vr, self.Vr_inv = f(ap.Vr), f(ap.Vr_inv)
            self.lam_r, self.ar = f(ap.lam_r), f(ap.ar)
            self.lam_s = f(ap.lam_s) if self.has_pairs else None
            self.pair_r = jnp.asarray(ap.pair_r, jnp.int32) if self.has_pairs else None
            # S_m, or its even block (S_o the odd one) when z-folded; in its own dtype
            self.S = jnp.asarray(fs.S)
            self.S_o = None if fs.S_o is None else jnp.asarray(fs.S_o)
        else:  # the per-mode tables split over the devices, the rest replicated on them
            self._distribute(ap, fs, kz, kp, mesh, mesh_axis)
            put = functools.partial(replicate, mesh=mesh)
        if not keep_numpy:
            fs.S = fs.S_o = None
        f = lambda a: put(jnp.asarray(a, self.dtype))  # noqa: E731
        if getattr(ap, "z_complex", False):  # mapped z axis with complex pairs: the complex eigenbasis
            cdt = jnp.complex128 if self.dtype == jnp.float64 else jnp.complex64
            fc = lambda a: put(jnp.asarray(a, cdt))  # noqa: E731
            self.Vz, self.Vz_inv, self.lam_z = fc(ap.Vz), fc(ap.Vz_inv), fc(ap.lam_z)
            self.ez = f(ap.ez)
        else:
            self.Vz, self.Vz_inv, self.lam_z, self.ez = f(ap.Vz), f(ap.Vz_inv), f(ap.lam_z), f(ap.ez)
        self.Dr, self.Dz, self.r = f(r_axis.D), f(z_axis.D), f(r_axis.s)
        self.mz, self.mp = f(jz * NFP), f(jp * NFP + 1)
        self.kz, self.kp = put(jnp.asarray(kz, jnp.int32)), put(jnp.asarray(kp, jnp.int32))
        self.r_axis, self.z_axis = r_axis, z_axis
        self.fs = fs

    # ------------------------------------------------------------ per mode
    def _dirichlet(self, FI, k):
        W = _rmul(_lmul(self.Vr_inv[k], FI), self.Vz_inv.T)
        if self.has_pairs:
            a = self.lam_r[k][:, :, None] + self.lam_z[None, None, :]
            sk = self.lam_s[k][:, :, None]
            Wp = jnp.take_along_axis(W, jnp.broadcast_to(self.pair_r[k][:, :, None], W.shape), axis=1)
            W = (a * W - sk * Wp) / (a * a + sk * sk)
        else:  # (bitwise the unmapped result)
            W = W / (self.lam_r[k][:, :, None] + self.lam_z[None, None, :])
        return _rmul(_lmul(self.Vr[k], W), self.Vz.T)

    def _solve_chunk(self, F, k):
        """Free-space u for one chunk of modal slabs F (c, n_r + 1, n_z + 1), indices k."""
        FI = F[:, 1:-1, 1:-1]
        uI = self._dirichlet(FI, k)
        u = jnp.pad(uI, ((0, 0), (1, 1), (1, 1)))
        # outward normal derivative on the faces (in, out, bot, top), packed
        ur = _along(self.Dr[jnp.array([0, -1])], u, 1)  # (c, 2, n_z + 1)
        uz = _along(self.Dz[jnp.array([0, -1])], u, 2)  # (c, n_r + 1, 2)
        sigma = jnp.concatenate([-ur[:, 0], ur[:, 1], -uz[:, :, 0], uz[:, :, 1]], axis=-1)
        def mv(S, x):
            return jnp.einsum("kij,kj->ki", S, x.real) + 1j * jnp.einsum("kij,kj->ki", S, x.imag)

        S = self.S[k].astype(self.dtype)
        if getattr(self, "zfold", None) is None:
            g = mv(S, sigma)
        else:
            g = self.zfold.apply(S, self.S_o[k].astype(self.dtype), sigma, mv, jnp)
        n1, n2 = self.nz + 1, self.nr + 1
        g_in, g_out = g[:, :n1], g[:, n1:2 * n1]
        g_bot, g_top = g[:, 2 * n1:2 * n1 + n2], g[:, 2 * n1 + n2:]
        ar = self.ar[k]
        rhs = (FI - ar[:, :, 0, None] * g_in[:, None, 1:-1] - ar[:, :, 1, None] * g_out[:, None, 1:-1]
               - g_bot[:, 1:-1, None] * self.ez[None, :, 0] - g_top[:, 1:-1, None] * self.ez[None, :, 1])
        uI = self._dirichlet(rhs, k)
        # boundary values around the interior by concatenation (transposable, unlike
        # a scatter): rows r = R_in, R_out are the lateral faces, columns the caps
        mid = jnp.concatenate([g_bot[:, 1:-1, None], uI, g_top[:, 1:-1, None]], axis=2)
        return jnp.concatenate([g_in[:, None, :], mid, g_out[:, None, :]], axis=1)

    def solve_modes(self, F, k):
        """Free-space u for modal slabs F (K, n_r + 1, n_z + 1) with distinct-|m| indices k."""
        if getattr(self, "mesh", None) is not None:
            raise ValueError("the per-mode tables are split over devices: only planes_F "
                             "(cyl_fused) runs on a solver with a mesh")
        K, c = F.shape[0], self.chunk
        n = -(-K // c)
        pad = n * c - K
        Fp = jnp.pad(F, ((0, pad), (0, 0), (0, 0))).reshape((n, c) + F.shape[1:])
        kp = jnp.pad(k, (0, pad)).reshape(n, c)

        # the linear data as the scan's carry, read chunk by chunk, and only the
        # constant mode indices scanned over: the form JAX transposes (as in
        # objective.SpectralInterpolator); checkpointed, so the reverse pass
        # recomputes a chunk instead of storing it
        @functools.partial(jax.checkpoint, policy=jax.checkpoint_policies.nothing_saveable)
        def step(Fc, ik):
            i, kk = ik
            return Fc, self._solve_chunk(jax.lax.dynamic_index_in_dim(Fc, i, keepdims=False), kk)

        out = jax.lax.scan(step, Fp, (jnp.arange(n), kp))[1]
        return out.reshape((n * c,) + F.shape[1:])[:K]

    # ------------------------------------------------------------ fields
    def sources(self, J):
        """FFT-index slabs of -mu0 J_z (rfft half) and of -mu0 (J_r + i J_phi) (full)."""
        J = J.astype(getattr(self, "dtype", J.dtype))
        Fz = jnp.fft.rfft(-mu_0 * J[..., 2], axis=0)
        Fp = jnp.fft.fft(-mu_0 * (J[..., 0] + 1j * J[..., 1]), axis=0)
        return Fz, Fp

    def potential_modes(self, J):
        """FFT-index slabs of A_z (rfft half) and of A_+'s periodic factor (full)."""
        return self.potential_modes_F(*self.sources(J))

    def potential_modes_F(self, Fz, Fp):
        """``potential_modes`` from the source slabs (``sources``)."""
        return self.solve_modes(Fz, self.kz), self.solve_modes(Fp, self.kp)

    def potential(self, J):
        """A on the grid: (A_r, A_phi, A_z), shape J.shape."""
        Az, Ap = self.potential_modes(J)
        Ac = jnp.fft.ifft(Ap, axis=0)
        return jnp.stack([Ac.real, Ac.imag, jnp.fft.irfft(Az, n=self.N_phi, axis=0)], axis=-1)

    def field_modes(self, J):
        """FFT-index slabs of B_+ / e^{i phi} (full) and of B_z (rfft half)."""
        return self.field_modes_F(*self.sources(J))

    def field_modes_F(self, Fz, Fp):
        """``field_modes`` from the source slabs (``sources``; e.g. the exact low
        toroidal modes of a J sampled on more planes, ``objective.Fields``)."""
        Az, Ap = self.potential_modes_F(Fz, Fp)
        N = self.N_phi
        jz = np.asarray(self.jz)
        # A_z at every signed j (FFT order): A_z^(-j) = conj A_z^(j)
        # FFT order (0, 1, .., M, -M, .., -1): the negative half is conj A_z^(M..1)
        Az_full = jnp.concatenate([Az, jnp.conj(Az[1:])[::-1]], axis=0)
        inv_r = (1.0 / self.r)[None, :, None]
        Bp = (1j * _along(self.Dz, Ap, 2)
              - 1j * (_along(self.Dr, Az_full, 1) - self.mz_full()[:, None, None] * inv_r * Az_full))
        psi = _along(self.Dr, Ap, 1) + self.mp[:, None, None] * inv_r * Ap
        psi_neg = psi[(-jz) % N]  # psi at -j for j >= 0
        Bz = (psi[jz] - jnp.conj(psi_neg)) / 2j
        return Bp, Bz

    def planes_F(self, Fz, Fp, Ep):
        """B on the (r, z) nodes of the planes phi_k of the points, from the source slabs
        (``sources``), one chunk of mode pairs +-j at a time: (B_r + i B_phi, B_z), shapes
        (n_planes, n_r + 1, n_z + 1). Ep (n_planes, N_phi): exp(i j NFP phi_k) / N_phi in
        FFT order (``CylindricalInterpolator.Ep``).

        Per pair j >= 0: A_z^(j) (A_z^(-j) = conj A_z^(j)), A_+^(j) and A_+^(-j) (the same
        mode solves as ``field_modes``), their curl, and their sum on the planes,
        B_+ = sum_j Ep_kj B_+^(j) and, from B_z^(j) = (psi^(j) - conj psi^(-j)) / 2i,
        B_z = Im sum_j Ep_kj psi^(j). A and B of all modes are never held (the memory of
        the sources, the planes and one chunk).

        On a solver with a mesh: ``planes_sharded`` with the full slabs replicated (each
        device picks its own); ``objective.Fields`` forms only each device's slabs."""
        if getattr(self, "mesh", None) is not None:
            return self.planes_sharded(lambda sz, sp, Fz, Fp: (Fz[sz], Fp[sp]), Ep, Fz, Fp)
        N = self.N_phi
        M = N // 2
        c = max(1, self.chunk // 2)
        n = -(-(M + 1) // c)
        jv = np.concatenate([np.arange(M + 1), np.zeros(n * c - M - 1, int)]).reshape(n, c)
        ok_p = (np.arange(n * c) <= M).reshape(n, c)
        ok_n = ok_p & (jv > 0)  # j = 0 has no separate -j
        ip, im = jv, (-jv) % N  # FFT indices of +j and -j
        dt = self.dtype
        kz, kp = jnp.asarray(self.kz), jnp.asarray(self.kp)
        args = (jnp.asarray(jv), jnp.asarray(ip), jnp.asarray(im),
                kz[jnp.asarray(jv)], kp[jnp.asarray(ip)], kp[jnp.asarray(im)],
                jnp.asarray(jv * self.NFP, dt), jnp.asarray(ok_p, dt), jnp.asarray(ok_n, dt),
                jnp.asarray(ip), jnp.asarray(im))
        return self._planes(Fz, Fp, Ep, args)

    def _planes(self, Fz, Fp, Ep, args, axis=None):
        """``planes_F``'s sum over chunks of pairs; args per chunk (n, c): the rows of Fz
        (A_z^(j)) and of Fp (A_+^(j), A_+^(-j)), the table rows of the three solves, j NFP,
        the masks of +j and -j, and Ep's columns of +j and -j. ``axis``: inside
        ``planes_sharded``, the mesh axis the sums vary along."""
        dt = self.dtype
        inv_r = (1.0 / self.r)[None, :, None]
        pc = jnp.zeros((Ep.shape[0],) + self.shape[1:], Fp.dtype)
        pz = jnp.zeros((Ep.shape[0],) + self.shape[1:], dt)
        if axis is not None:  # the carry's type: different on each device
            pc, pz = _varying(pc, axis), _varying(pz, axis)

        # the sources (linear) and the sums as the scan's carry, the chunk's constants
        # scanned over (the form JAX transposes); checkpointed: the reverse pass
        # recomputes a chunk's solves instead of storing them
        @functools.partial(jax.checkpoint, policy=jax.checkpoint_policies.nothing_saveable)
        def step(carry, a):
            Fz, Fp, pc, pz = carry
            j, ip, im, kz, kpp, kpm, m, okp, okn, ep, en = a
            Az = self._solve_chunk(Fz[j], kz)
            Ap = self._solve_chunk(Fp[ip], kpp)
            An = self._solve_chunk(Fp[im], kpm)
            dAz = _along(self.Dr, Az, 1)
            mi = m[:, None, None] * inv_r
            Bp = 1j * _along(self.Dz, Ap, 2) - 1j * (dAz - mi * Az)
            Bn = 1j * _along(self.Dz, An, 2) - 1j * (jnp.conj(dAz) + mi * jnp.conj(Az))
            psp = _along(self.Dr, Ap, 1) + (mi + inv_r) * Ap
            psn = _along(self.Dr, An, 1) + (inv_r - mi) * An
            Epj = Ep[:, ep] * okp[None, :]
            Enj = Ep[:, en] * okn[None, :]
            pc = pc + jnp.einsum("kc,cab->kab", Epj, Bp) + jnp.einsum("kc,cab->kab", Enj, Bn)
            pz = pz + jnp.imag(jnp.einsum("kc,cab->kab", Epj, psp)
                               + jnp.einsum("kc,cab->kab", Enj, psn))
            return (Fz, Fp, pc, pz), None

        (_, _, pc, pz), _ = jax.lax.scan(step, (Fz, Fp, pc, pz), args)
        return pc, pz

    # ------------------------------------------------------------ devices
    def _distribute(self, ap, fs, kz, kp, mesh, axis):
        """The per-mode tables over the devices of ``mesh`` along ``axis``: device d takes
        the d-th block of the pairs j = 0 .. (N_phi - 1) / 2 of ``planes_F`` and holds the
        rows of V_r, V_r^-1, lam_r, a_r and S_m (S_o) of the |m| they use (j NFP, j NFP + 1,
        |j NFP - 1|: disjoint between pairs for NFP >= 3), stacked on a leading device
        axis and padded to the longest block by repeating a row. ``pairs``: the per-chunk
        indices of ``_planes`` for each device (its source rows, table rows, j NFP, the
        masks, Ep's columns); ``sel``: the FFT indices of its source slabs (of Fz, and of
        Fp for +j then -j). The tables go from the host straight to their device."""
        from jax.sharding import NamedSharding, PartitionSpec
        axis = axis or mesh.axis_names[0]
        self.mesh_axis = axis
        n_dev = mesh.shape[axis]
        jv, ok, use = pair_blocks(self.N_phi, self.chunk, n_dev, kz, kp)
        L, c = jv.shape[1], max(1, self.chunk // 2)
        n = L // c  # chunks per device
        ip, im = jv, (-jv) % self.N_phi
        K = max(len(u) for u in use)
        rows = np.stack([np.pad(u, (0, K - len(u)), mode="edge") for u in use])  # (n_dev, K)
        loc = lambda k: np.stack([np.searchsorted(u, kd) for u, kd in zip(use, k)])  # noqa: E731
        sh = NamedSharding(mesh, PartitionSpec(axis))

        def split(a, dtype):
            """a's rows of each device's |m| on that device: (n_dev, K) + a.shape[1:]."""
            dtype = np.dtype(dtype)
            return jax.make_array_from_callback(
                (n_dev, K) + a.shape[1:], sh,
                lambda idx: np.asarray(a[rows[idx[0]]], dtype))

        dt = self.dtype
        self.Vr, self.Vr_inv = split(ap.Vr, dt), split(ap.Vr_inv, dt)
        self.lam_r, self.ar = split(ap.lam_r, dt), split(ap.ar, dt)
        self.S = split(fs.S, fs.S.dtype)
        self.S_o = None if fs.S_o is None else split(fs.S_o, fs.S_o.dtype)
        ch = lambda a: a.reshape(n_dev, n, c)  # noqa: E731
        put = lambda a, t: jax.device_put(np.asarray(a, t), sh)  # noqa: E731
        self.pairs = tuple(put(ch(a), t) for a, t in (
            (loc(kz[jv]), np.int32), (loc(kp[ip]), np.int32), (loc(kp[im]), np.int32),
            (jv * self.NFP, dt), (ok, dt), (ok & (jv > 0), dt), (ip, np.int32), (im, np.int32)))
        self.sel = (put(jv, np.int32), put(np.concatenate([ip, im], axis=1), np.int32))
        self.n_pairs = L

    def planes_sharded(self, sources, Ep, *data):
        """``planes_F`` with the mode pairs split over the devices of ``mesh``: each device
        forms its own source slabs, sources(sel_z, sel_p, *data) -> (Fz[sel_z], Fp[sel_p])
        (``data`` replicated: e.g. J's rows, ``objective.Fields._cyl_sources``), solves its
        pairs with its tables, and the planes are summed over the devices (psum: the
        result replicated on every device). Linear in data when sources is; JAX
        transposes it (the planes' cotangent reaches every device, the data's are summed).

        Every array the body uses comes in as an input (this solver's leaves, Ep, data):
        sources must not close over arrays on the mesh (JAX's sharding-in-types refuses
        to index them inside the body)."""
        from jax.sharding import PartitionSpec
        ax, Pd, Pr = self.mesh_axis, PartitionSpec(self.mesh_axis), PartitionSpec()
        L = self.n_pairs
        n, c = self.pairs[0].shape[1:]
        own = np.arange(L).reshape(n, c)  # rows of the device's slabs: Fz (j), Fp (+j, -j)
        leaves, tree = jax.tree_util.tree_flatten(self)
        mine = {id(a) for a in jax.tree_util.tree_leaves(
            (self.S, self.S_o, self.Vr, self.Vr_inv, self.lam_r, self.ar, self.pairs, self.sel))}
        split = [id(a) in mine for a in leaves]

        def body(leaves, Ep, data):
            # this device's solver: its tables (the device axis dropped), the rest replicated
            dev = jax.tree_util.tree_unflatten(tree, [a[0] if s else a
                                                      for a, s in zip(leaves, split)])
            Fz, Fp = sources(*dev.sel, *data)
            args = (own, own, own + L) + tuple(dev.pairs)
            pc, pz = dev._planes(Fz, Fp, Ep, args, axis=ax)
            return jax.lax.psum(pc, ax), jax.lax.psum(pz, ax)

        specs = [Pd if s else Pr for s in split]
        return jax.shard_map(body, mesh=self.mesh, in_specs=(specs, Pr, Pr),
                             out_specs=(Pr, Pr))(leaves, Ep, data)

    def mz_full(self):
        return jnp.asarray(np.asarray(self.jp) * self.NFP, getattr(self, "dtype", jnp.float64))

    def field(self, J):
        """B on the grid: (B_r, B_phi, B_z), shape J.shape [T] for J [A/m^2]."""
        Bp, Bz = self.field_modes(J)
        Bc = jnp.fft.ifft(Bp, axis=0)
        return jnp.stack([Bc.real, Bc.imag, jnp.fft.irfft(Bz, n=self.N_phi, axis=0)], axis=-1)

    # ------------------------------------------------------------ diagnostics
    def _dphi(self, f):
        """d/dphi of a real field on the planes (spectral, N_phi odd)."""
        k = 1j * self.NFP * jnp.asarray(np.fft.fftfreq(self.N_phi, 1.0 / self.N_phi))
        return jnp.fft.ifft(k[:, None, None] * jnp.fft.fft(f, axis=0), axis=0).real

    def div_grid(self, V):
        """div V of cylindrical components V on the grid (spectral in all directions)."""
        r = self.r[None, :, None]
        return (_along(self.Dr, r * V[..., 0], 1) / r + self._dphi(V[..., 1]) / r
                + _along(self.Dz, V[..., 2], 2))

    def curl_grid(self, V):
        """curl V of cylindrical components V on the grid."""
        r = self.r[None, :, None]
        Vr, Vp, Vz = V[..., 0], V[..., 1], V[..., 2]
        return jnp.stack([self._dphi(Vz) / r - _along(self.Dz, Vp, 2),
                          _along(self.Dz, Vr, 2) - _along(self.Dr, Vz, 1),
                          (_along(self.Dr, r * Vp, 1) - self._dphi(Vr)) / r], axis=-1)

    # ------------------------------------------------------------ grid
    @property
    def phi(self):
        return 2 * np.pi * np.arange(self.N_phi) / (self.NFP * self.N_phi)

    def nodes(self):
        """(r, phi, z) of every node, shape (N_phi, n_r + 1, n_z + 1, 3)."""
        P, R, Z = np.meshgrid(self.phi, self.r_axis.s, self.z_axis.s, indexing="ij")
        return np.stack([R, P, Z], axis=-1)


@functools.partial(jax.jit, static_argnums=())
def field(solver, J):
    """Jitted B = solver.field(J), the operators passed as inputs."""
    return solver.field(J)


@jax.jit
def potential(solver, J):
    return solver.potential(J)
