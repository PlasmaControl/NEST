"""Per-toroidal-mode Poisson problems on the meridional rectangle of the annulus.

For a field u(r, z) e^{i m phi} the Laplacian is

    L_m u = u_rr + u_r / r - m^2 u / r^2 + u_zz,

on [R_in, R_out] x [-z0, z0], discretized by collocation on a Chebyshev axis in
r and in z (``chebyshev.ChebAxis``, n_r + 1 and n_z + 1 nodes). Arrays u have
shape (..., n_r + 1, n_z + 1), r first. The interior equations are

    (L_r,m u)[I, J] + (u D_zz^T)[I, J] = f[I, J],     I = 1..n_r - 1, J = 1..n_z - 1,

and the interior blocks are diagonalized once, L_r,m[I, I] = V_r Lambda_r V_r^-1
per distinct |m| and D_zz[J, J] = V_z Lambda_z V_z^-1, so a Dirichlet solve is

    u_I = V_r [(V_r^-1 f_I V_z^-T) / (lambda_r,i + lambda_z,j)] V_z^T

(the matrix-diagonalization method of Haidvogel and Zang, 1979). Boundary
values g enter the interior equations through the boundary columns of the two
operators only (corners never do):

    rhs_I = f_I - L_r[I, 0] g_in - L_r[I, n_r] g_out - g_bot D_zz[J, 0] - g_top D_zz[J, n_z].

The boundary is stored face by face, each face with its own end nodes (the
corners appear twice):

    in   r = R_in,  all z     (n_z + 1)       outward normal -r
    out  r = R_out, all z     (n_z + 1)       outward normal +r
    bot  z = -z0,   all r     (n_r + 1)       outward normal -z
    top  z = +z0,   all r     (n_r + 1)       outward normal +z

``flux`` gives the outward normal derivative on the faces, the density sigma of
the James correction.
"""

import numpy as np

FACES = ("in", "out", "bot", "top")


def _n_threads(K):
    """Threads for the per-mode eigendecompositions: $NEST_EIG_THREADS, else the usable CPUs
    (at most 32, at most one per 8 modes) when BLAS can be held to one thread per call
    (threadpoolctl, or OMP/OPENBLAS/MKL/VECLIB thread counts of 1 in the environment), else 1:
    with multithreaded BLAS calls the threads oversubscribe (a/30 on 8 cores: 36 s against 9 s
    serial; pinned: 1.3 s against 6.4 s)."""
    import os
    env = os.environ.get("NEST_EIG_THREADS")
    if env is not None:
        return max(1, int(env))
    pinned = any(os.environ.get(k) == "1" for k in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS",
                                                   "MKL_NUM_THREADS", "VECLIB_MAXIMUM_THREADS"))
    if not pinned:
        try:
            import threadpoolctl  # noqa: F401
        except ImportError:
            return 1
    try:
        n = len(os.sched_getaffinity(0))
    except AttributeError:
        n = os.cpu_count() or 1
    return max(1, min(32, n, K // 8))


def _pin_worker():
    try:
        from threadpoolctl import threadpool_limits
        threadpool_limits(1)
    except ImportError:
        pass


def _blas_single_thread():
    """Context limiting BLAS/LAPACK to one thread per caller (threadpoolctl if available)."""
    try:
        from threadpoolctl import threadpool_limits
        return threadpool_limits(1)
    except ImportError:
        import contextlib
        return contextlib.nullcontext()


def _real_eig(A, what):
    lam, V = np.linalg.eig(A)
    if np.max(np.abs(lam.imag) / np.abs(lam)) > 1e-10 or np.max(np.abs(V.imag)) > 1e-10:
        raise RuntimeError(f"{what}: complex eigenpairs")
    lam, V = lam.real, V.real
    if lam.max() >= 0:
        raise RuntimeError(f"{what}: non-negative eigenvalue {lam.max():.3e}")
    order = np.argsort(lam)[::-1]
    return lam[order], V[:, order]




def _complex_eig(A, what):
    """Eigendecomposition of A (real, with complex pairs) in complex arithmetic, ordered by
    descending real part. Used for the shared z operator of a mapped axis: the solver's data
    are complex Fourier coefficients anyway."""
    lam, V = np.linalg.eig(A)
    if lam.real.max() >= 0:
        raise RuntimeError(f"{what}: non-negative eigenvalue (real part) {lam.real.max():.3e}")
    order = np.argsort(lam.real)[::-1]
    return lam[order], V[:, order]


def _block_eig(A, what):
    """Real block diagonalization A V = V B, for an A with real or complex-pair eigenvalues.

    B is block diagonal: a real eigenvalue a is a 1x1 block; a pair a +- i b (b > 0) is the
    2x2 block [[a, b], [-b, a]] on the columns (Re v, Im v) of the eigenvector v of a + i b.
    Returns (a, s, pair, V): a the real parts (descending), s the off-diagonal entries
    (b on the first column of a pair, -b on the second, 0 for real eigenvalues), pair the
    partner column (itself for real ones). All arithmetic stays real. Complex pairs come
    from the non-normality of mapped (KTE) operators; they are well conditioned there
    (kappa(V) <= ~40 for the production grids to alpha = 0.99)."""
    lam, W = np.linalg.eig(A)
    scale = np.abs(lam).max()
    cx = np.abs(lam.imag) > 1e-10 * scale
    blocks = []
    for j in np.flatnonzero(~cx):
        if np.abs(W[:, j].imag).max() > 1e-10 * np.abs(W[:, j]).max():
            raise RuntimeError(f"{what}: complex eigenvector of a real eigenvalue")
        blocks.append((lam[j].real, [W[:, j].real], [0.0]))
    for j in np.flatnonzero(cx & (lam.imag > 0)):
        b = lam[j].imag
        blocks.append((lam[j].real, [W[:, j].real, W[:, j].imag], [b, -b]))
    if 2 * int(np.sum(cx & (lam.imag > 0))) != int(cx.sum()):
        raise RuntimeError(f"{what}: unpaired complex eigenvalues")
    blocks.sort(key=lambda t: -t[0])
    n = A.shape[0]
    a, s_, pair, V = np.empty(n), np.empty(n), np.empty(n, np.int64), np.empty((n, n))
    c = 0
    for re, cols, ss in blocks:
        for q, (v, sq) in enumerate(zip(cols, ss)):
            a[c + q], s_[c + q], V[:, c + q] = re, sq, v
            pair[c + q] = c + (1 - q if len(cols) == 2 else 0)
        c += len(cols)
    if a.max() >= 0:
        raise RuntimeError(f"{what}: non-negative eigenvalue (real part) {a.max():.3e}")
    return a, s_, pair, V


def _block_divide(W, a, s, pair, axis=-2):
    """Solve (B + mu) X = W in the eigen coordinates of ``_block_eig``: ``a`` already holds
    a + mu (broadcast against W), ``s`` the pair entries, ``pair`` the partner rows (along
    ``axis``). X_k = ((a_k + mu) W_k - s_k W_p(k)) / ((a_k + mu)^2 + s_k^2)."""
    import numpy as _np
    Wp = _np.take_along_axis(W, pair, axis=axis)
    return (a * W - s * Wp) / (a * a + s * s)


class AnnulusPoisson:
    """Dirichlet core and boundary operators for the modes |m| in ``ms``.

    Parameters
    ----------
    r_axis, z_axis : ChebAxis
        On [R_in, R_out] (R_in > 0) and [-z0, z0].
    ms : sequence of int
        The distinct |m| that will be solved for (index k in ``ms`` selects one).
    """

    def __init__(self, r_axis, z_axis, ms):
        if r_axis.a <= 0:
            raise ValueError("annulus needs R_in > 0")
        self.r_axis, self.z_axis = r_axis, z_axis
        self.ms = np.asarray(sorted(set(int(abs(m)) for m in ms)))
        self.index = {int(m): k for k, m in enumerate(self.ms)}
        nr, nz = r_axis.n, z_axis.n
        self.nr, self.nz = nr, nz
        self.shape = (nr + 1, nz + 1)
        r = r_axis.s
        self.r = r
        I, J = slice(1, nr), slice(1, nz)
        # z: shared by every mode
        Dzz = z_axis.D2
        try:
            self.lam_z, self.Vz = _real_eig(Dzz[J, J], "D_zz")
            self.z_complex = False
        except RuntimeError as e:  # complex pairs (mapped z axis): the complex eigenbasis
            if "complex eigenpairs" not in str(e):
                raise
            self.lam_z, self.Vz = _complex_eig(Dzz[J, J], "D_zz")
            self.z_complex = True
        self.Vz_inv = np.linalg.inv(self.Vz)
        self.ez = Dzz[J][:, [0, nz]]  # (n_z - 1, 2): bot, top columns on interior rows
        # r: per |m|
        Lr0 = r_axis.D2 + r_axis.D / r[:, None]
        K = len(self.ms)
        self.lam_r = np.empty((K, nr - 1))  # real parts of the eigenvalues
        self.lam_s = np.zeros((K, nr - 1))  # complex pairs: +-b (``_block_eig``), else 0
        self.pair_r = np.tile(np.arange(nr - 1), (K, 1))  # partner column (itself if real)
        self.Vr = np.empty((K, nr - 1, nr - 1))
        self.Vr_inv = np.empty((K, nr - 1, nr - 1))
        self.ar = np.empty((K, nr - 1, 2))  # in, out columns on interior rows

        def radial(k):
            m = self.ms[k]
            Lr = Lr0 - np.diag(m**2 / r**2)
            lam, ls, pr, V = _block_eig(Lr[I, I], f"L_r, m = {m}")
            self.lam_r[k], self.lam_s[k], self.pair_r[k], self.Vr[k] = lam, ls, pr, V
            self.Vr_inv[k] = np.linalg.inv(V)
            self.ar[k] = Lr[I][:, [0, nr]]

        n_threads = _n_threads(K)
        if n_threads > 1:  # independent modes; LAPACK releases the GIL
            from concurrent.futures import ThreadPoolExecutor
            # the limit in each worker too: OpenMP thread counts (OpenMP OpenBLAS) are per thread
            with _blas_single_thread(), ThreadPoolExecutor(n_threads, initializer=_pin_worker) as ex:
                list(ex.map(radial, range(K)))
        else:
            for k in range(K):
                radial(k)
        self.Lr0 = Lr0
        # modes whose radial operator has complex pairs (mapped axes); none unmapped
        self.has_pairs = bool(np.any(self.lam_s != 0))

    @property
    def cond_Vr(self):
        """2-norm condition numbers of the radial eigenvector matrices (computed on demand:
        an SVD per mode, ~1/4 of the setup if done eagerly)."""
        if not hasattr(self, "_cond_Vr"):
            self._cond_Vr = np.array([np.linalg.cond(V) for V in self.Vr])
        return self._cond_Vr

    # ------------------------------------------------------------ operators
    def k_of(self, m):
        """Index into ``ms`` of each m (array-like)."""
        return np.array([self.index[int(abs(x))] for x in np.atleast_1d(m)])

    def apply(self, u, m):
        """L_m u on all nodes (rows r, columns z), for one m."""
        Lr = self.Lr0 - np.diag(m**2 / self.r**2)
        return Lr @ u + u @ self.z_axis.D2.T

    def dirichlet(self, f, k):
        """Interior solution of L_m u = f with u = 0 on the boundary.

        f: (..., n_r + 1, n_z + 1) (only the interior is used) or interior-sized;
        k: mode index per leading entry (int or array). Returns u with zero
        boundary, the full node array.
        """
        f = np.asarray(f)
        fI = f[..., 1:-1, 1:-1] if f.shape[-2:] == self.shape else f
        k = np.broadcast_to(np.asarray(k), fI.shape[:-2])
        W = np.einsum("...ij,...jl->...il", self.Vr_inv[k], fI) @ self.Vz_inv.T
        if self.has_pairs:
            W = _block_divide(W, self.lam_r[k][..., :, None] + self.lam_z[None, :], self.lam_s[k][..., :, None],
                              np.broadcast_to(self.pair_r[k][..., :, None], W.shape))
        else:  # (bitwise the unmapped result)
            W = W / (self.lam_r[k][..., :, None] + self.lam_z[None, :])
        uI = np.einsum("...ij,...jl->...il", self.Vr[k], W) @ self.Vz.T
        if self.z_complex and not np.iscomplexobj(fI):
            uI = uI.real  # real data: real solution (to rounding) through the complex z basis
        u = np.zeros(fI.shape[:-2] + self.shape, dtype=np.result_type(fI, float))
        u[..., 1:-1, 1:-1] = uI
        return u

    def solve(self, f, g, k):
        """L_m u = f inside, u = g on the boundary (g: dict of faces or packed vector)."""
        g = self.unpack(g) if not isinstance(g, dict) else g
        f = np.asarray(f)
        fI = f[..., 1:-1, 1:-1] if f.shape[-2:] == self.shape else f
        k = np.broadcast_to(np.asarray(k), fI.shape[:-2])
        ar = self.ar[k]  # (..., n_r - 1, 2)
        rhs = (fI
               - ar[..., :, 0, None] * g["in"][..., None, 1:-1]
               - ar[..., :, 1, None] * g["out"][..., None, 1:-1]
               - g["bot"][..., 1:-1, None] * self.ez[None, :, 0]
               - g["top"][..., 1:-1, None] * self.ez[None, :, 1])
        u = self.dirichlet(rhs, k)
        u[..., 0, :] = g["in"]
        u[..., -1, :] = g["out"]
        u[..., 1:-1, 0] = g["bot"][..., 1:-1]
        u[..., 1:-1, -1] = g["top"][..., 1:-1]
        return u

    # ------------------------------------------------------------ boundary
    @property
    def face_sizes(self):
        return {"in": self.nz + 1, "out": self.nz + 1, "bot": self.nr + 1, "top": self.nr + 1}

    @property
    def n_boundary(self):
        return 2 * (self.nz + 1) + 2 * (self.nr + 1)

    def faces(self, u):
        """Boundary values of u, face by face (each with its end nodes)."""
        return {"in": u[..., 0, :], "out": u[..., -1, :], "bot": u[..., :, 0], "top": u[..., :, -1]}

    def pack(self, d):
        return np.concatenate([d[f] for f in FACES], axis=-1)

    def unpack(self, v):
        out, i = {}, 0
        for f in FACES:
            n = self.face_sizes[f]
            out[f] = v[..., i:i + n]
            i += n
        return out

    def flux(self, u):
        """Outward normal derivative on each face (dict), from the nodal u."""
        ur_in = np.einsum("j,...jl->...l", self.r_axis.D[0], u)
        ur_out = np.einsum("j,...jl->...l", self.r_axis.D[-1], u)
        uz_bot = u @ self.z_axis.D[0]
        uz_top = u @ self.z_axis.D[-1]
        return {"in": -ur_in, "out": ur_out, "bot": -uz_bot, "top": uz_top}

    def boundary_points(self):
        """(r, z) of the packed boundary nodes, and the face id of each."""
        r, z = self.r_axis.s, self.z_axis.s
        pts = {"in": np.column_stack([np.full_like(z, r[0]), z]),
               "out": np.column_stack([np.full_like(z, r[-1]), z]),
               "bot": np.column_stack([r, np.full_like(r, z[0])]),
               "top": np.column_stack([r, np.full_like(r, z[-1])])}
        return self.pack({f: pts[f].T for f in FACES}).T, np.concatenate(
            [np.full(self.face_sizes[f], i) for i, f in enumerate(FACES)])
