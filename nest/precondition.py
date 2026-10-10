"""Block preconditioner for the Gauss-Newton system of ``optimize``.

The normal matrix A = J^T J of the force balance is far from diagonal: the
unknowns of one Fourier mode (every radial l of a and b, and chi, of the
current potential) are nearly collinear in J-space, so the Jacobi scaling
leaves a condition number of ~1e7 already at the equilibrium resolution
(``scripts/spectrum.py``). The diagonal blocks of A over the groups of
unknowns sharing (|m|, |n|) bring it to ~1e4 (``scripts/precond_test.py``).

A's blocks would cost one J v per unknown. Instead they are taken from the
local part of the residual, which is cheap column by column:

    ForceBalance   d F / d u_i   ~ J_i x B      (J x dB_i, through Biot-Savart, dropped)
                   d F / d c_k   = -grad p_k
    CurrentError   d J / d u_i   = J_i
    PressureError  d p / d c_k   = p_k
    FieldError     (nonlocal, dropped)

each with the objective's weight sqrt(dV / V) / normalization. The block
(Gram matrix of these columns) is eigen-decomposed once; the preconditioner
applies V (w + lam d)^-1 V^T per block (d: the block's mean diagonal, for
the Levenberg-Marquardt damping lam D with D the diagonal of the blocks).
"""

import time

import jax
import jax.numpy as jnp
import numpy as np

from objective import _cartesian


def mode_labels(fields, current, pressure):
    """(kind, l, m, n) of every unknown x = (u, c_p): kind 0 for chi, a, b and 1 for p."""
    S, modes = current.S, np.asarray(current.basis.modes, int)
    chi = np.column_stack([np.abs(S.chi_modes[:, 0]), S.chi_modes])
    u = np.concatenate([chi, modes, modes])
    p = np.asarray(pressure.basis.modes, int)
    return np.concatenate([np.column_stack([np.zeros(len(u), int), u]),
                           np.column_stack([np.ones(len(p), int), p])])


def mode_groups(labels, free=None):
    """Index arrays of the free unknowns sharing (kind, |m|, |n|)."""
    free = np.ones(len(labels), bool) if free is None else np.asarray(free, bool)
    keys = {}
    for i in np.flatnonzero(free):
        k, _, m, n = labels[i]
        keys.setdefault((k, abs(m), abs(n)), []).append(i)
    return [np.array(v) for v in keys.values()]


def _columns(obj, x):
    """f(e) -> the local columns of every objective for unit vectors e (x-sized)."""
    fields = obj.fields
    B = fields.compute(x)["B"]
    n_u = fields.n_u
    sw = fields.sw[:, None]

    def cols(e, fl, B):
        u, c = e[:n_u], e[n_u:]
        J = _cartesian(fl.J_col(u, 0.0).reshape(-1, 3), fl.phi)
        p, gp = fl.p_col(c)
        gp = _cartesian(gp.reshape(-1, 3), fl.phi)
        # p_col(c) is affine only through the fixed edge order: linear here
        out = []
        for o in obj.objectives:
            if o.dim_f == 1:  # a single scalar (PressureEnergy): rank one, left out
                continue
            s = o.weight / o.normalization
            if o.quantity == "F":
                out.append(s * sw * (jnp.cross(J, B) - gp))
            elif o.quantity == "J":
                out.append(s * sw * J)
            elif o.quantity == "p":
                out.append(s * sw[:, 0] * p.ravel())
        return jnp.concatenate([v.ravel() for v in out])

    return jax.jit(jax.vmap(cols, in_axes=(0, None, None))), B


class _View:
    """Attribute holder: a restricted stand-in for ``TensorFactors`` and
    ``TensorCurrent`` (their methods are called on it unbound)."""


def _u_block_gram(obj, B, pot_struct, k_max, max_bytes=1e9):
    """g(block arrays) -> the Gram matrix of the local columns of a block of the
    current potential, in one compiled function per block size.

    A column is J of one unit coefficient: only the factor-table columns of the
    block's own modes take part, so the tables are restricted to them (padded to
    fixed sizes) before the contraction, instead of contracting the full tables
    with a one-hot coefficient array. The Gram matrix (a sum over the points) is
    accumulated over radial slabs of the tensor grid, so that the columns of a
    slab take at most ~``max_bytes``."""
    from flux_interp import TensorCurrent, TensorFactors, TensorGeometry
    fields = obj.fields
    tc = fields.J_col
    n_r, n_t, n_z = tc.grid.shape
    scale = [(o.quantity, o.weight / o.normalization) for o in obj.objectives
             if o.quantity in ("F", "J")]
    rows = 3 * n_t * n_z * len(scale)  # residual rows per radius
    per = max(1, int(max_bytes // (4 * k_max * rows)))
    slabs = [(r, min(r + per, n_r)) for r in range(0, n_r, per)]

    class FactorView:
        __call__ = TensorFactors.__call__

    class GeoView:
        vector = TensorGeometry.vector

    def factor_view(tf, r0, r1, cols_p, cols_q):
        v = FactorView()
        v.grid = _View()
        v.grid.shape = (r1 - r0, n_t, n_z)
        v.P = {d: P[r0 * n_t : r1 * n_t, cols_p] for d, P in tf.P.items()}
        v.F = {d: F[:, cols_q] for d, F in tf.F.items()}
        return v

    def gram(ab_p, ab_q, chi_m, chi_n, kind, pa, pb, tcd, B, sw, phi):
        n_p, n_q, n_m, n_n = len(ab_p), len(ab_q), len(chi_m), len(chi_n)

        class Split:
            @staticmethod
            def split(e):  # e: one-hot over the block's columns
                Cchi = jnp.zeros((n_m, n_n)).at[pa, pb].add(e * (kind == 0))
                c = jnp.where(kind == 2, 1, 0)
                Cab = jnp.zeros((n_p, n_q, 2)).at[pa, pb, c].add(e * (kind > 0))
                return Cchi, Cab

        k = len(kind)
        G = jnp.zeros((k, k), jnp.float32)
        for r0, r1 in slabs:  # radial slabs (static)
            sub = _View()
            sub.S = Split
            sub.w = [w[r0:r1] for w in tcd.w]
            sub.h0r, sub.rho2 = tcd.h0r[r0:r1], tcd.rho2[r0:r1]
            geo = GeoView()
            g = tcd.geo
            geo.R, geo.Xr, geo.Xt, geo.Xz, geo.sqrt_g = (
                g.R[r0:r1], g.Xr[r0:r1], g.Xt[r0:r1], g.Xz[r0:r1], g.sqrt_g[r0:r1])
            sub.geo = geo
            sub.ab = factor_view(tcd.ab, r0, r1, ab_p, ab_q)
            sub.chi = factor_view(tcd.chi, r0, r1, chi_m, chi_n)
            a, b = r0 * n_t * n_z, r1 * n_t * n_z  # the slab's points (C order)
            Bs, sws, phis = B[a:b], sw[a:b, None], phi[a:b]

            def column(e):
                J = _cartesian(TensorCurrent.__call__(sub, e, 0.0).reshape(-1, 3), phis)
                out = [s * sws * (jnp.cross(J, Bs) if q == "F" else J) for q, s in scale]
                return jnp.concatenate([v.ravel() for v in out]).astype(jnp.float32)

            Phi = jax.lax.map(column, jnp.eye(k), batch_size=16)
            G = G + jnp.matmul(Phi, Phi.T, precision=jax.lax.Precision.HIGHEST)
        return G

    return jax.jit(gram)


def _u_block_arrays(S, groups, n_u):
    """Per block of u: the restricted columns of the factor tables and, per
    unknown, its kind (0 chi, 1 a, 2 b) and positions in them; padded."""
    (pa_, qa_), (pb_, qb_) = S.ab._pq
    rows = []
    for g in groups:
        kind, ia, ib = [], [], []
        for i in g:
            if i < S.n_chi:
                kind.append(0); ia.append(S.chi_idx[0][i]); ib.append(S.chi_idx[1][i])
            elif i < S.n_chi + S.K:
                k = i - S.n_chi
                kind.append(1); ia.append(pa_[k]); ib.append(qa_[k])
            else:
                k = i - S.n_chi - S.K
                kind.append(2); ia.append(pb_[k]); ib.append(qb_[k])
        kind, ia, ib = np.array(kind), np.array(ia), np.array(ib)
        chi, ab = kind == 0, kind > 0
        cm, pos_cm = np.unique(ia[chi], return_inverse=True)
        cn, pos_cn = np.unique(ib[chi], return_inverse=True)
        ap, pos_ap = np.unique(ia[ab], return_inverse=True)
        aq, pos_aq = np.unique(ib[ab], return_inverse=True)
        pa, pb = np.zeros(len(g), int), np.zeros(len(g), int)
        pa[chi], pb[chi], pa[ab], pb[ab] = pos_cm, pos_cn, pos_ap, pos_aq
        rows.append((ap, aq, cm, cn, kind, pa, pb))
    size = [max(max(len(r[j]), 1) for r in rows) for j in range(4)]
    # the columns of a block padded to the next of a few sizes (one compilation each)
    k_max = max(len(r[4]) for r in rows)
    buckets = [b for b in (8, 16, 32, 64) if b < k_max] + [k_max]
    out = []
    for ap, aq, cm, cn, kind, pa, pb in rows:
        k = next(b for b in buckets if b >= len(kind))
        pad = lambda a, n, v=0: np.concatenate([a, np.full(n - len(a), v, int)]).astype(np.int32)
        out.append((pad(ap, size[0]), pad(aq, size[1]), pad(cm, size[2]), pad(cn, size[3]),
                    pad(kind, k, -1), pad(pa, k), pad(pb, k)))
    return out


class BlockPreconditioner:
    """(blocks of A + lam D)^-1 from the local columns, for ``levenberg_marquardt``.

    Parameters
    ----------
    obj : ObjectiveFunction
    x : array
        Where the columns are taken (B of the force balance).
    groups : list of index arrays
        The blocks (``mode_groups``).
    chunk : int
        Columns evaluated together.
    floor : float
        Eigenvalues of a block floored at floor x its largest (the columns are in
        single precision: eigenvalues below ~1e-7 of the largest are not resolved,
        which only affects how well, not whether, the preconditioner works).
    """

    def __init__(self, obj, x, groups, chunk=16, floor=1e-7, verbose=True, structure=None):
        t0 = time.time()
        n_x = obj.dim_x
        cols, B = _columns(obj, jnp.asarray(x))
        fields = obj.fields
        # blocks of the current potential: restricted tables (``_u_block_gram``)
        fast = {}
        # (objectives of p alone do not depend on u; FieldError, nonlocal, is dropped)
        if structure is not None:
            ug = [gi for gi, g in enumerate(groups) if g.max() < fields.n_u]
            if ug:
                arrays = _u_block_arrays(structure, [groups[i] for i in ug], fields.n_u)
                gram = _u_block_gram(obj, B, structure, max(len(a[4]) for a in arrays))
                for gi, arrs in zip(ug, arrays):
                    fast[gi] = (gram, arrs)
        k_max = max(len(g) for g in groups)
        idx = np.full((len(groups), k_max), n_x)
        V = np.zeros((len(groups), k_max, k_max))
        w = np.ones((len(groups), k_max))
        d = np.zeros(n_x + 1)
        for gi, g in enumerate(groups):
            k = len(g)
            if gi in fast:
                gram, arrs = fast[gi]
                G = np.asarray(gram(*arrs, fields.J_col, B, fields.sw, fields.phi),
                               dtype=float)[:k, :k]
            else:
                G = self._generic(cols, fields, B, g, n_x, chunk)
            lam, U = np.linalg.eigh(G)
            lam = np.maximum(lam, floor * max(lam[-1], 1e-300))
            idx[gi, :k] = g
            V[gi, :k, :k] = U
            V[gi, k:, k:] = np.eye(k_max - k)
            w[gi, :k] = lam
            d[g] = np.diag(G)
            if verbose and gi == 9:
                print(f"  block preconditioner: {(time.time() - t0) / 10 * len(groups):.0f} s "
                      f"estimated for {len(groups)} blocks (largest {k_max})", flush=True)
        self.idx, self.V, self.w = jnp.asarray(idx), jnp.asarray(V), jnp.asarray(w)
        self.dbar = jnp.asarray([d[g].mean() for g in groups])
        self.diag = jnp.asarray(d[:n_x])
        self.n_x = n_x
        if verbose:
            print(f"  block preconditioner: {len(groups)} blocks, {time.time() - t0:.0f} s",
                  flush=True)

    @staticmethod
    def _generic(cols, fields, B, g, n_x, chunk):
        """Gram matrix of a block from the full evaluation of unit vectors."""
        Phi = []
        for i in range(0, len(g), chunk):
            gg = g[i : i + chunk]
            E = jnp.zeros((chunk, n_x)).at[jnp.arange(len(gg)), gg].set(1.0)
            # columns in single precision (a preconditioner), their Gram matrix at
            # full matmul precision
            Phi.append(cols(E, fields, B)[: len(gg)].astype(jnp.float32))
        Phi = jnp.concatenate(Phi)
        return np.asarray(jnp.matmul(Phi, Phi.T, precision=jax.lax.Precision.HIGHEST),
                          dtype=float)

    def __call__(self, r, lam=0.0):
        return _apply_blocks(self.idx, self.V, self.w, self.dbar, r, lam)


@jax.jit
def _apply_blocks(idx, V, w, dbar, r, lam):
    re = jnp.concatenate([r, jnp.zeros(1, r.dtype)])[idx]  # (groups, k)
    t = jnp.einsum("gkj,gk->gj", V, re) / (w + lam * dbar[:, None])
    z = jnp.einsum("gkj,gj->gk", V, t)
    out = jnp.zeros(r.shape[0] + 1, r.dtype).at[idx].add(z)
    return out[:-1]
