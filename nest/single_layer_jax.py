"""S_m on an accelerator: ``single_layer.single_layer_matrix`` with the toroidal-kernel
recurrences and the face products in JAX.

The numerics are those of ``single_layer`` / ``toroidal_kernel`` (same quadrature, same start
values, same recurrences, same Lagrange maps), so the entries agree to rounding and the S_m
cache key (``single_layer.S_NUMERICS``) is shared. What differs is the organisation:

* host (NumPy, per target): the panel rules, the lane data (chi - 1, the weight factor, the
  face coordinate y of the Lagrange map), the start values Q_{-1/2}, Q_{1/2} (SciPy elliptic
  integrals, as on the CPU) and the choice forward / Miller with its start length
  K = ceil(40 / eta) (Miller start N = m_max + K);
* device, per chunk of targets: (1) the Miller start sweep (K steps, a vector per lane) for the
  lanes grouped by K on the ladder 64 * 2^j (a longer start only lowers the start error below
  e^{-80}), giving rho_{m_max}; (2) over all lanes of the chunk at once, the stored sweep of the
  ratios, the upward product, and the forward recurrence, selected per lane; (3) the Lagrange
  matrices of each face (barycentric, as ``chebyshev.barycentric_matrix``) and
  S_m[:, t, face] = kw @ L as one batched einsum. Shapes are fixed for the whole build (the
  lane counts per face padded to the largest), so each function compiles once.

Needs ``jax_enable_x64``. Memory: about ``mem_bytes`` of device memory (Q of one chunk of
targets, one lane block, the Lagrange matrices of a face); the result is copied to the host
chunk by chunk.
"""

import time

import numpy as np

from annulus_poisson import FACES
from single_layer import _face_geometry, _panel_rule
from toroidal_kernel import _Q_start

# padding lanes: weight 0 at the face node y = -1 (an exact hit, so a one-hot Lagrange row; a
# point outside [-1, 1] makes the barycentric denominator ~5.8^-n, lost to cancellation: 0 * inf)
_PAD_Y = -1.0
_B1 = 1 << 16  # lanes per block of the start sweep


def _round_up(n, k):
    return (int(n) + k - 1) // k * k


def _device_fns(m_max, am, unroll=8):
    import jax
    import jax.numpy as jnp
    from jax import lax

    am = jnp.asarray(am)

    def start(c, K):  # m = m_max + K .. m_max + 1, from rho = 0: rho_{m_max}
        def step(i, rho):
            m = (m_max + K - i).astype(c.dtype)
            return (m - 0.5) / (2 * m * c - (m + 0.5) * rho)
        return lax.fori_loop(0, K, step, jnp.zeros_like(c), unroll=unroll)

    def Q(c, q0, q1, rho, fwd):
        def store(r, m):  # m = m_max .. 1: r becomes rho_{m-1}
            r = (m - 0.5) / (2 * m * c - (m + 0.5) * r)
            return r, r
        _, rat = lax.scan(store, rho, jnp.arange(m_max, 0, -1, dtype=c.dtype), unroll=unroll)

        def up(q, r):  # Q_{m+1/2} = rho_m Q_{m-1/2}, sequential as on the CPU
            q = q * r
            return q, q
        _, qm = lax.scan(up, q0, rat[::-1], unroll=unroll)

        def fw(carry, m):
            a, b = carry
            n = (2 * m * c * b - (m - 0.5) * a) / (m + 0.5)
            return (b, n), n
        _, qf = lax.scan(fw, (q0, q1), jnp.arange(1, m_max, dtype=c.dtype), unroll=unroll)
        Qm = jnp.concatenate([q0[None], qm])
        Qf = jnp.concatenate([q0[None], q1[None], qf])
        return jnp.where(fwd[None], Qf, Qm)[am]

    def face(Qf, pref, y, y_nodes, w_bary):
        d = y[:, :, None] - y_nodes[None, None, :]
        exact = d == 0
        c = w_bary / jnp.where(exact, 1.0, d)
        L = c / c.sum(axis=2, keepdims=True)
        L = jnp.where(exact.any(axis=2, keepdims=True), exact.astype(L.dtype), L)
        return jnp.einsum("mtp,tpn->mtn", Qf * pref[None], L, precision=lax.Precision.HIGHEST)

    return dict(start=jax.jit(start, static_argnums=1), Q=jax.jit(Q), face=jax.jit(face))


def _default_mem():
    import jax
    try:
        lim = jax.devices()[0].memory_stats()["bytes_limit"]
        return 0.3 * lim
    except Exception:
        return 2e9


def single_layer_matrix_jax(ap, ms, targets, n_gl=16, ratio=0.15, verbose=False, dtype=np.float64,
                            mem_bytes=None, out=None):
    """As ``single_layer.single_layer_matrix`` (same arguments and result), on the default JAX
    device. ``mem_bytes``: device working set (default 0.3 of the device memory limit, else 2 GB);
    ``out``: an optional preallocated host array (len(ms), n_t, ap.n_boundary)."""
    import jax
    import jax.numpy as jnp
    if not jax.config.jax_enable_x64:
        raise RuntimeError("single_layer_matrix_jax needs jax_enable_x64")
    mem_bytes = mem_bytes or _default_mem()
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
    t0 = time.time()

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

    per_all = [lanes(*t) for t in targets]  # host: panel rules (lane counts fix the shapes)
    P = [_round_up(max(len(p[fi][0]) for p in per_all), 256) for fi in range(len(FACES))]
    t_host = time.time() - t0
    # targets per chunk: ~5 arrays (m_max + 1, T * sum(P)) on the device
    # and the Lagrange matrices of a face, (T, P_f, n_f), with its weighted product (x3): at a/320
    # these, not the recurrences, set the chunk (70 GB requested before they were counted)
    lag = max(P[fi] * sizes[f] for fi, f in enumerate(FACES))
    T = int(max(1, min(n_t, mem_bytes / (5 * 8 * (m_max + 1) * sum(P)), mem_bytes / (3 * 8 * (lag + n_m * P[0])))))
    t_dev = 0.0
    for c0 in range(0, n_t, T):
        per = per_all[c0:c0 + T]
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
        Kb = np.where(fwd, 0, 64 * 2 ** np.ceil(np.log2(np.maximum(K, 64) / 64)).astype(np.int64))
        td = time.time()
        rho = np.zeros_like(x)
        for kb in np.unique(Kb[~fwd]):
            idx = np.flatnonzero(Kb == kb)
            for b0 in range(0, len(idx), _B1):
                ib = idx[b0:b0 + _B1]
                cb = np.concatenate([chi[ib], np.full(_B1 - len(ib), 2.0)])
                rho[ib] = np.asarray(fns["start"](jnp.asarray(cb), int(kb)))[:len(ib)]
        Qc = fns["Q"](jnp.asarray(chi), jnp.asarray(Q0), jnp.asarray(Q1), jnp.asarray(rho), jnp.asarray(fwd))
        o = 0
        for fi, f in enumerate(FACES):
            n = T * P[fi]
            Sf = fns["face"](Qc[:, o:o + n].reshape(n_m, T, P[fi]), jnp.asarray(W[fi]), jnp.asarray(Y[fi]),
                             *nodes[f])
            S[:, c0:c0 + len(per), offs[fi]:offs[fi + 1]] = np.asarray(Sf)[:, :len(per)]
            o += n
        del Qc
        t_dev += time.time() - td
        if verbose and ((c0 // T + 1) % max(1, (n_t // T) // 10) == 0 or c0 + T >= n_t):
            print(f"    S_m (jax {jax.default_backend()}): {min(c0 + T, n_t)}/{n_t} targets "
                  f"({time.time() - t0:.0f} s; host lanes {t_host:.0f} s, device {t_dev:.0f} s; chunk {T}, "
                  f"lanes/target {sum(P)})", flush=True)
    return S
