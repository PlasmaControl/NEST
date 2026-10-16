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


def _real_eig(A, what):
    lam, V = np.linalg.eig(A)
    if np.max(np.abs(lam.imag) / np.abs(lam)) > 1e-10 or np.max(np.abs(V.imag)) > 1e-10:
        raise RuntimeError(f"{what}: complex eigenpairs")
    lam, V = lam.real, V.real
    if lam.max() >= 0:
        raise RuntimeError(f"{what}: non-negative eigenvalue {lam.max():.3e}")
    order = np.argsort(lam)[::-1]
    return lam[order], V[:, order]


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
        self.lam_z, self.Vz = _real_eig(Dzz[J, J], "D_zz")
        self.Vz_inv = np.linalg.inv(self.Vz)
        self.ez = Dzz[J][:, [0, nz]]  # (n_z - 1, 2): bot, top columns on interior rows
        # r: per |m|
        Lr0 = r_axis.D2 + r_axis.D / r[:, None]
        K = len(self.ms)
        self.lam_r = np.empty((K, nr - 1))
        self.Vr = np.empty((K, nr - 1, nr - 1))
        self.Vr_inv = np.empty((K, nr - 1, nr - 1))
        self.ar = np.empty((K, nr - 1, 2))  # in, out columns on interior rows
        self.cond_Vr = np.empty(K)
        for k, m in enumerate(self.ms):
            Lr = Lr0 - np.diag(m**2 / r**2)
            lam, V = _real_eig(Lr[I, I], f"L_r, m = {m}")
            self.lam_r[k], self.Vr[k] = lam, V
            self.Vr_inv[k] = np.linalg.inv(V)
            self.ar[k] = Lr[I][:, [0, nr]]
            self.cond_Vr[k] = np.linalg.cond(V)
        self.Lr0 = Lr0

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
        W = W / (self.lam_r[k][..., :, None] + self.lam_z[None, :])
        uI = np.einsum("...ij,...jl->...il", self.Vr[k], W) @ self.Vz.T
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
