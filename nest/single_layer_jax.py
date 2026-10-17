"""S_m on an accelerator: ``single_layer.single_layer_matrix`` with the toroidal-kernel
recurrences and the face products in JAX.

The numerics are those of ``single_layer`` / ``toroidal_kernel`` (same quadrature, same start
values, same recurrences, same Lagrange maps), so the entries agree to rounding and the S_m
cache key (``single_layer.S_NUMERICS``) is shared. What differs is the organisation:

* host (NumPy, per target): the panel rules, the lane data (chi - 1, the weight factor, the
  face coordinate y of the Lagrange map), the start values Q_{-1/2}, Q_{1/2} (SciPy elliptic
  integrals, as on the CPU) and the choice forward / Miller with its start length
  K = ceil(40 / eta) (Miller start N = m_max + K);
* device: lanes are grouped by K on the ladder 64 * 2^j (a longer start only lowers the start
  error below e^{-80}) and run in fixed-shape blocks of B lanes: the start sweep (K steps), the
  stored sweep of the ratios (m_max steps), the upward product; forward lanes run the forward
  recurrence. The Lagrange matrices of each face are formed on the device (barycentric, as
  ``chebyshev.barycentric_matrix``) and S_m[:, t, face] = kw @ L is one batched einsum per
  chunk of targets.

Needs ``jax_enable_x64``. Memory: about ``mem_bytes`` of device memory (Q of one chunk of
targets, one lane block, the Lagrange matrices of a face); the result is copied to the host
chunk by chunk.
"""

import time

import numpy as np

from annulus_poisson import FACES
from single_layer import _face_geometry, _panel_rule
from toroidal_kernel import _Q_start

_PAD_Y = 3.0  # face coordinate of padding lanes: outside [-1, 1], never a node


def _round_up(n, k):
    return (int(n) + k - 1) // k * k


def _device_fns(m_max, am):
    """Jitted device functions for one m_max and |m| set (cached per call)."""
    import jax
    import jax.numpy as jnp
    from jax import lax

    am = jnp.asarray(am)

    def upward(q0, rat):  # Q_{m+1/2} = rho_m Q_{m-1/2}, sequential as on the CPU
        def step(q, r):
            q = q * r
            return q, q
        _, qs = lax.scan(step, q0, rat)
        return jnp.concatenate([q0[None], qs])[am]

    def miller(c, q0, K):
        def start(i, rho):  # m = m_max + K .. m_max + 1
            m = (m_max + K - i).astype(c.dtype)
            return (m - 0.5) / (2 * m * c - (m + 0.5) * rho)
        rho = lax.fori_loop(0, K, start, jnp.zeros_like(c))

        def store(rho, m):  # m = m_max .. 1: rho becomes rho_{m-1}
            rho = (m - 0.5) / (2 * m * c - (m + 0.5) * rho)
            return rho, rho
        _, rat = lax.scan(store, rho, jnp.arange(m_max, 0, -1, dtype=c.dtype))
        return upward(q0, rat[::-1])

    def forward(c, q0, q1):
        def step(carry, m):
            qm, q = carry
            qn = (2 * m * c * q - (m - 0.5) * qm) / (m + 0.5)
            return (q, qn), qn
        _, qs = lax.scan(step, (q0, q1), jnp.arange(1, m_max, dtype=c.dtype))
        return jnp.concatenate([q0[None], q1[None], qs])[am] if m_max > 1 else \
            jnp.concatenate([q0[None], q1[None]])[am]

    def scatter(Qc, idx, vals):
        return Qc.at[:, idx].set(vals)

    def face(Qf, pref, y, y_nodes, w_bary):
        # Qf (n_m, T, P), pref / y (T, P); Lagrange rows as chebyshev.barycentric_matrix
        d = y[:, :, None] - y_nodes[None, None, :]
        exact = d == 0
        c = w_bary / jnp.where(exact, 1.0, d)
        L = c / c.sum(axis=2, keepdims=True)
        L = jnp.where(exact.any(axis=2, keepdims=True), exact.astype(L.dtype), L)
        return jnp.einsum("mtp,tpn->mtn", Qf * pref[None], L, precision=lax.Precision.HIGHEST)

    return dict(miller=jax.jit(miller, static_argnums=2), forward=jax.jit(forward),
                scatter=jax.jit(scatter, donate_argnums=0), face=jax.jit(face))


def single_layer_matrix_jax(ap, ms, targets, n_gl=16, ratio=0.15, verbose=False, dtype=np.float64,
                            mem_bytes=4e9, out=None):
    """As ``single_layer.single_layer_matrix`` (same arguments and result), on the default JAX
    device. ``out``: an optional preallocated host array (len(ms), n_t, ap.n_boundary)."""
    import jax
    import jax.numpy as jnp
    if not jax.config.jax_enable_x64:
        raise RuntimeError("single_layer_matrix_jax needs jax_enable_x64")
    targets = np.atleast_2d(np.asarray(targets, dtype=float))
    ms = np.asarray(ms)
    am = np.abs(ms)
    m_max = int(am.max())
    n_m, n_t = len(ms), len(targets)
    if m_max < 1:
        raise ValueError("m_max >= 1 expected")
    geo = _face_geometry(ap)
    sizes = ap.face_sizes
    offs = np.cumsum([0] + [sizes[f] for f in FACES])
    S = out if out is not None else np.zeros((n_m, n_t, ap.n_boundary), dtype=dtype)
    fns = _device_fns(m_max, am)
    nodes = {f: (jnp.asarray(geo[f][0].y), jnp.asarray(geo[f][0].w_bary)) for f in FACES}
    # lane block: the (m_max + 1, B) intermediates of a block (ratios, Q) in ~mem_bytes / 4
    B = int(2 ** np.floor(np.log2(max(4096, mem_bytes / 4 / (16 * (m_max + 1))))))

    def lanes(rt, zt):
        """Per face: (y, chi - 1, weight factor) of the quadrature of one target."""
        res = []
        for f in FACES:
            axis, (which, c) = geo[f]
            t_along = zt if which == "r" else rt
            t_norm = rt - c if which == "r" else zt - c
            p = min(max(t_along, axis.a), axis.b)
            delta = np.hypot(t_norm, t_along - p)
            l, w, off = _panel_rule(axis.s, p, delta, n_gl, ratio)
            rp = np.full_like(l, c) if which == "r" else l
            d2 = t_norm**2 + ((t_along - p) - off) ** 2
            assert (d2 > 0).all()
            rr = rt * rp
            res.append((axis.to_y(l), d2 / (2 * rr), -0.5 * w * rp / (np.pi * np.sqrt(rr))))
        return res

    # targets per chunk: Q of the chunk (n_m, T * P) in ~mem_bytes / 4
    probe = lanes(*targets[0])
    P_est = sum(len(x[0]) for x in probe) + 4 * 512
    T = int(max(1, min(n_t, mem_bytes / 4 / (8 * n_m * P_est))))
    t0, t_dev = time.time(), 0.0
    for c0 in range(0, n_t, T):
        tgt = targets[c0:c0 + T]
        per = [lanes(*t) for t in tgt]
        P = [_round_up(max(len(p[fi][0]) for p in per), 512) for fi in range(len(FACES))]
        # lanes in face-major blocks (T, P_f); padding lanes: chi - 1 = 1, weight 0, y = 3
        Y, X, W = ([np.full((T, Pf), v) for Pf in P] for v in (_PAD_Y, 1.0, 0.0))
        for ti, p in enumerate(per):
            for fi, (y, x, w) in enumerate(p):
                Y[fi][ti, :len(y)], X[fi][ti, :len(y)], W[fi][ti, :len(y)] = y, x, w
        x = np.concatenate([a.ravel() for a in X])
        chi = 1.0 + x
        eta = np.log1p(x + np.sqrt(x * (chi + 1.0)))
        Q0, Q1 = _Q_start(x)
        fwd = (m_max * eta <= 2.0) & (eta <= 0.5)
        K = np.where(fwd, 0, np.ceil(40.0 / eta)).astype(np.int64)
        Kb = np.where(fwd, -1, 64 * 2 ** np.ceil(np.log2(np.maximum(K, 64) / 64)).astype(np.int64))
        n_lane = len(x)
        td = time.time()
        Qc = jnp.zeros((n_m, n_lane + 1))  # last column: sink of padding lanes
        for kb in np.unique(Kb):
            idx = np.flatnonzero(Kb == kb)
            for b0 in range(0, len(idx), B):
                ib = idx[b0:b0 + B]
                pad = B - len(ib)
                cb = np.concatenate([chi[ib], np.full(pad, 2.0)])
                q0 = np.concatenate([Q0[ib], np.ones(pad)])
                if kb < 0:
                    q1 = np.concatenate([Q1[ib], np.ones(pad)])
                    vals = fns["forward"](jnp.asarray(cb), jnp.asarray(q0), jnp.asarray(q1))
                else:
                    vals = fns["miller"](jnp.asarray(cb), jnp.asarray(q0), int(kb))
                Qc = fns["scatter"](Qc, jnp.asarray(np.concatenate([ib, np.full(pad, n_lane)])), vals)
        o = 0
        for fi, f in enumerate(FACES):
            n = T * P[fi]
            Sf = fns["face"](Qc[:, o:o + n].reshape(n_m, T, P[fi]), jnp.asarray(W[fi]), jnp.asarray(Y[fi]),
                             *nodes[f])
            S[:, c0:c0 + len(tgt), offs[fi]:offs[fi + 1]] = np.asarray(Sf)[:, :len(tgt)]
            o += n
        t_dev += time.time() - td
        if verbose and ((c0 // T + 1) % max(1, (n_t // T) // 10) == 0 or c0 + T >= n_t):
            print(f"    S_m (jax {jax.default_backend()}): {min(c0 + T, n_t)}/{n_t} targets "
                  f"({time.time() - t0:.0f} s; device {t_dev:.0f} s; chunk {T}, block {B})", flush=True)
    return S
