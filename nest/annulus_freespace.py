"""Free-space Poisson solve for an NFP-periodic source in a toroidal annulus (NumPy).

The annulus is Omega = [R_in, R_out] x [-z0, z0] in (r, z) and one field period
0 <= phi < 2 pi / NFP, with N_phi (odd) uniform planes phi_k = 2 pi k / (NFP N_phi).
For a source f supported inside Omega (and vanishing near its boundary), the
free-space solution of Delta u = f, u -> 0 at infinity, is computed in Omega
mode by mode (James' method):

    u_D     Dirichlet solution (u_D = 0 on the boundary), ``AnnulusPoisson``
    sigma   outward normal derivative of u_D on the four faces
    g       S_m sigma on the boundary nodes, ``single_layer``
    u       L_m u = f inside, u = g on the boundary

since u_free = u_D + S[sigma] in Omega and S[sigma] is harmonic there.

Fields carry a mode shift s: a field whose physical modes are m = j NFP + s,
with j the FFT index over the period. s = 0 is a scalar such as A_z (real, rfft);
s = 1 is A_+ = A_x + i A_y, whose FFT is that of the NFP-periodic A_r + i A_phi
(see docs/cylindrical_grid_plan.md, D5).

This is the reference implementation of stage A (gate A6); the JAX apply path of
stage B is checked against it.
"""

import time

import numpy as np

from annulus_poisson import AnnulusPoisson
from single_layer import single_layer_matrix


def mode_indices(N_phi, real):
    """FFT indices j of a period with N_phi (odd) planes: 0..(N-1)/2 (rfft) or all
    j in FFT order (0, 1, .., (N-1)/2, -(N-1)/2, .., -1)."""
    if N_phi % 2 == 0:
        raise ValueError("N_phi must be odd")
    if real:
        return np.arange(N_phi // 2 + 1)
    return np.fft.fftfreq(N_phi, 1.0 / N_phi).astype(int)


def z_reflection(ap, tol=1e-12):
    """The permutation P of the packed boundary nodes under z -> -z (the lateral faces
    reversed, the caps exchanged), or None if the z axis is not symmetric about 0."""
    z = ap.z_axis.s
    if np.abs(z[::-1] + z).max() > tol * np.abs(z).max():
        return None
    from single_layer import FACES
    sizes = ap.face_sizes
    offs = dict(zip(FACES, np.cumsum([0] + [sizes[f] for f in FACES])[:-1]))
    perm = np.empty(ap.n_boundary, int)
    for f in FACES:
        n, o = sizes[f], offs[f]
        if f in ("in", "out"):
            perm[o:o + n] = o + np.arange(n)[::-1]
        else:
            perm[o:o + n] = offs["top" if f == "bot" else "bot"] + np.arange(n)
    return perm


class ZFold:
    """S_m in the even and odd parts of the boundary under z -> -z.

    On a z-symmetric annulus S_m = P S_m P (P the node reflection), so in the
    orthonormal basis u_a = c_a (e_i + e_Pi) (c = 1/sqrt 2 for a pair, 1/2 for a node
    on z = 0) and v_a = (e_i - e_Pi) / sqrt 2 it is block diagonal:

        S_e[a, b] = 2 c_a c_b (S[i, j] + S[i, Pj]),   S_o[a, b] = S[i, j] - S[i, Pj]

    with i, j the representatives (i <= Pi). Only the rows S[rep, :] are computed, and
    S_e, S_o together hold half of S_m. ``apply`` forms S_m sigma with gathers only
    (transposable by JAX).
    """

    def __init__(self, perm):
        n = len(perm)
        idx = np.arange(n)
        self.rep = idx[idx <= perm]
        self.partner = perm[self.rep]
        fixed = self.rep == self.partner
        self.c = np.where(fixed, 0.5, 1 / np.sqrt(2))
        self.odd_rows = np.nonzero(~fixed)[0]  # rows of rep that have an odd part
        self.odd, self.podd = self.rep[~fixed], self.partner[~fixed]
        n_e, n_o = len(self.rep), len(self.odd)
        a_of = np.empty(n, int)
        a_of[self.rep] = np.arange(n_e)
        a_of[self.partner] = np.arange(n_e)
        self.a_of = a_of
        self.ce = np.where(fixed, 1.0, 1 / np.sqrt(2))[a_of]
        b_of = np.full(n, n_o)  # a zero slot for the nodes on z = 0
        b_of[self.odd] = np.arange(n_o)
        b_of[self.podd] = np.arange(n_o)
        self.b_of = b_of
        so = np.zeros(n)
        so[self.odd], so[self.podd] = 1 / np.sqrt(2), -1 / np.sqrt(2)
        self.so = so
        self.n_e, self.n_o = n_e, n_o

    def fold(self, S_rows, dtype=None):
        """(K, n_e, N_b) rows S[:, rep, :] -> S_e (K, n_e, n_e), S_o (K, n_o, n_o)."""
        K = S_rows.shape[0]
        dtype = dtype or S_rows.dtype
        S_e = np.empty((K, self.n_e, self.n_e), dtype)
        S_o = np.empty((K, self.n_o, self.n_o), dtype)
        cc = 2 * self.c[:, None] * self.c[None, :]
        for k in range(K):  # mode by mode: no full-size temporaries
            R = np.asarray(S_rows[k], np.float64)
            S_e[k] = cc * (R[:, self.rep] + R[:, self.partner])
            S_o[k] = (R[self.odd_rows][:, self.odd] - R[self.odd_rows][:, self.podd])
        return S_e, S_o

    def apply(self, S_e, S_o, sigma, mv, xp=np):
        """S_m sigma for packed sigma (..., N_b); mv(S, x) the batched product."""
        rdt = sigma.real.dtype  # the coefficients in sigma's precision (no promotion)
        c, ce, so = (xp.asarray(a, rdt) for a in (self.c, self.ce, self.so))
        s_e = c * (sigma[..., self.rep] + sigma[..., self.partner])
        s_o = (sigma[..., self.odd] - sigma[..., self.podd]) * xp.asarray(1 / np.sqrt(2), rdt)
        y_e, y_o = mv(S_e, s_e), mv(S_o, s_o)
        y_o = xp.concatenate([y_o, xp.zeros(y_o.shape[:-1] + (1,), y_o.dtype)], axis=-1)
        return ce * y_e[..., self.a_of] + so * y_o[..., self.b_of]


def S_backend():
    """'jax' or 'numpy' for the S_m build: $NEST_S_BACKEND ('jax', 'numpy', or 'auto', the
    default: JAX when its default backend is a GPU with x64 enabled, else NumPy)."""
    import os
    want = os.environ.get("NEST_S_BACKEND", "auto").lower()
    if want in ("jax", "numpy"):
        return want
    try:
        import jax
        if jax.default_backend() == "gpu" and jax.config.jax_enable_x64:
            return "jax"
    except Exception:
        pass
    return "numpy"


def _S_rows(ap, ms, pts, verbose=False, dtype=np.float64, **sl_kw):
    """S_m rows at pts by the backend of ``S_backend`` (entries agree to rounding: the device build
    against the float64 references a/20-a/60, <= 4e-14 row-relative)."""
    if S_backend() == "jax":
        from single_layer_jax import single_layer_matrix_jax
        kw = {k: v for k, v in sl_kw.items() if k in ("n_gl", "ratio")}
        return single_layer_matrix_jax(ap, ms, pts, verbose=verbose, dtype=dtype, **kw)
    return single_layer_matrix(ap, ms, pts, verbose=verbose, dtype=dtype, **sl_kw)


def S_key(ap, zsym, n_gl=16, ratio=0.15):
    """Cache key of the single-layer matrices of ``ap`` (AnnulusPoisson): the axes, the
    |m| set, the z-fold, the quadrature rule, and ``single_layer.S_NUMERICS``, the version
    of the numerics. Edits that do not change the entries (parallelism, batching,
    documentation, a GPU path that reproduces them) keep the key, so caches survive them;
    a change of the entries must bump S_NUMERICS (``tests/cyl_s_build.py`` compares a
    build against the stored references)."""
    import hashlib

    from single_layer import S_NUMERICS
    ax = lambda a: (a.a, a.b, a.n, a.alpha)  # noqa: E731
    rep = (ax(ap.r_axis), ax(ap.z_axis), tuple(int(m) for m in ap.ms), bool(zsym), int(n_gl), float(ratio),
           "S_NUMERICS", int(S_NUMERICS))
    return hashlib.sha256(repr(rep).encode()).hexdigest()[:24]


def _S_cache_dir(cache):
    """The S_m cache directory: ``cache`` if given, else $NEST_S_CACHE; None or 'off': none."""
    import os
    where = cache if cache is not None else os.environ.get("NEST_S_CACHE")
    return None if where in (None, "", "off") else where


def save_S(where, key, S, S_o, meta):
    """S (and S_o when z-folded) in float64 under where/S_<key>/ (atomic: a temporary
    directory renamed into place). Returns the directory."""
    import json
    import os
    path = os.path.join(where, f"S_{key}")
    if os.path.isdir(path):
        return path
    tmp = f"{path}.tmp{os.getpid()}"
    os.makedirs(tmp, exist_ok=True)
    np.save(os.path.join(tmp, "S.npy"), np.asarray(S, np.float64))
    if S_o is not None:
        np.save(os.path.join(tmp, "S_o.npy"), np.asarray(S_o, np.float64))
    with open(os.path.join(tmp, "meta.json"), "w") as f:
        json.dump(meta, f, indent=1)
    try:
        os.replace(tmp, path)
    except OSError:  # another writer finished first
        pass
    return path


def load_S(where, key, dtype):
    """(S, S_o, meta) from where/S_<key>/ in ``dtype`` (cast mode by mode from the
    float64 files, memory-mapped), or None if absent."""
    import json
    import os
    path = os.path.join(where, f"S_{key}")
    if not os.path.isfile(os.path.join(path, "meta.json")):
        return None

    def read(name):
        f = os.path.join(path, name)
        if not os.path.exists(f):
            return None
        mm = np.load(f, mmap_mode="r")
        out = np.empty(mm.shape, dtype)
        for k in range(mm.shape[0]):
            out[k] = mm[k]
        return out
    with open(os.path.join(path, "meta.json")) as f:
        meta = json.load(f)
    return read("S.npy"), read("S_o.npy"), meta


class FreeSpaceAnnulus:
    """Precomputed per-mode operators for the shifts in ``shifts`` (0, 1 or both).

    ``cache`` (or $NEST_S_CACHE): a directory where the single-layer matrices are stored
    in float64 after their first build and loaded afterwards (keyed by ``S_key``);
    None/'off' (the default when the variable is unset) builds them every time.

    Parameters
    ----------
    r_axis, z_axis : ChebAxis
    NFP, N_phi : int
    shifts : tuple of int
    """

    def __init__(self, r_axis, z_axis, NFP, N_phi, shifts=(0,), verbose=False, S_dtype=np.float64,
                 zsym="auto", cache=None, **sl_kw):
        self.NFP, self.N_phi, self.shifts = int(NFP), int(N_phi), tuple(shifts)
        ms = set()
        for s in self.shifts:
            ms |= set(np.abs(mode_indices(N_phi, real=(s == 0)) * NFP + s).tolist())
        t0 = time.time()
        self.ap = AnnulusPoisson(r_axis, z_axis, sorted(ms))
        t1 = time.time()
        pts, _ = self.ap.boundary_points()
        perm = z_reflection(self.ap) if zsym else None
        if zsym is True and perm is None:
            raise ValueError("zsym=True needs a z axis symmetric about 0")
        self.zfold = None if perm is None else ZFold(perm)
        where = _S_cache_dir(cache)
        self.S_cache = None
        self.S_built_by = None  # 'jax' / 'numpy' when built here
        if where is not None:
            key = S_key(self.ap, self.zfold is not None, sl_kw.get("n_gl", 16), sl_kw.get("ratio", 0.15))
            got = load_S(where, key, S_dtype)
            if got is not None:
                self.S, self.S_o, _ = got
                self.S_cache = ("loaded", key)
        if self.S_cache is None:
            # with a cache the build is float64 (the stored copy), else in S_dtype
            build_dtype = np.float64 if where is not None else S_dtype
            self.S_built_by = S_backend()
            if self.zfold is None:
                self.S = _S_rows(self.ap, self.ap.ms, pts, verbose=verbose, dtype=build_dtype,
                                             **sl_kw)
                self.S_o = None
            else:  # half the rows, folded into the even and odd blocks
                rows = _S_rows(self.ap, self.ap.ms, pts[self.zfold.rep], verbose=verbose,
                                           dtype=build_dtype, **sl_kw)
                self.S, self.S_o = self.zfold.fold(rows)
                del rows
            if where is not None:
                meta = dict(key=key, r_axis=[r_axis.a, r_axis.b, r_axis.n, r_axis.alpha],
                            z_axis=[z_axis.a, z_axis.b, z_axis.n, z_axis.alpha], NFP=self.NFP,
                            N_phi=self.N_phi, shifts=list(self.shifts), ms=[int(m) for m in self.ap.ms],
                            zsym=self.zfold is not None, n_gl=int(sl_kw.get("n_gl", 16)),
                            ratio=float(sl_kw.get("ratio", 0.15)), build_s=time.time() - t1)
                save_S(where, key, self.S, self.S_o, meta)
                self.S_cache = ("built", key)
                if np.dtype(S_dtype) != np.float64:
                    self.S = self.S.astype(S_dtype)
                    self.S_o = None if self.S_o is None else self.S_o.astype(S_dtype)
        self.timing = {"eig": t1 - t0, "S": time.time() - t1}
        if verbose:
            print(f"  FreeSpaceAnnulus: {len(self.ap.ms)} distinct |m| <= {self.ap.ms.max()}, "
                  f"N_b = {self.ap.n_boundary}{' (z-folded)' if self.zfold else ''}; eig "
                  f"{self.timing['eig']:.1f} s, S {self.timing['S']:.1f} s"
                  + (f" [{self.S_built_by}]" if self.S_built_by else "") + " "
                  f"({self.S_nbytes / 1e6:.0f} MB"
                  + (f"; cache {self.S_cache[0]} S_{self.S_cache[1]}" if self.S_cache else "") + ")",
                  flush=True)

    @property
    def S_nbytes(self):
        return sum(a.nbytes for a in (self.S, self.S_o) if a is not None)

    def apply_S(self, k, sigma):
        """S_m sigma for packed sigma (n_modes, N_b), distinct-|m| indices k."""
        if self.zfold is None:
            return np.einsum("kij,kj->ki", self.S[k], sigma)
        return self.zfold.apply(self.S[k], self.S_o[k], sigma,
                                lambda S, x: np.einsum("kij,kj->ki", S, x))

    @property
    def phi(self):
        return 2 * np.pi * np.arange(self.N_phi) / (self.NFP * self.N_phi)

    def solve_modes(self, F, m):
        """Free-space u for modal slabs F (n_modes, n_r + 1, n_z + 1) with modes m."""
        ap = self.ap
        k = ap.k_of(m)
        uD = ap.dirichlet(F, k)
        sigma = ap.pack(ap.flux(uD))
        g = self.apply_S(k, sigma)
        return ap.solve(F, g, k), {"uD": uD, "sigma": sigma, "g": g}

    def solve(self, f, shift=0, return_parts=False):
        """Free-space u of f on the grid (N_phi, n_r + 1, n_z + 1).

        shift 0: f real, rfft in phi. shift s != 0: f is the NFP-periodic factor of a
        field with modes j NFP + s (complex), full FFT.
        """
        real = shift == 0
        j = mode_indices(self.N_phi, real)
        F = np.fft.rfft(f, axis=0) if real else np.fft.fft(f, axis=0)
        U, parts = self.solve_modes(F, j * self.NFP + shift)
        u = np.fft.irfft(U, n=self.N_phi, axis=0) if real else np.fft.ifft(U, axis=0)
        return (u, parts) if return_parts else u
