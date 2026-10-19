"""Fast single-layer matrices S_m of the annulus boundary (``single_layer``), on the GPU.

``single_layer.single_layer_matrix`` builds, for every target and every face, a
target-specific composite rule over the whole face (the face nodes plus points graded
towards the target) and its Lagrange matrix: O(n_t n_q n_f) work in NumPy, ~30 min at
a/320 on one CPU and ~h^-3.

Here the same integral is split:

  regular  every face's panels between consecutive face nodes, n_gl Gauss-Legendre
           points each: the points, weights and their Lagrange matrix L do not depend
           on the target, so S_reg = K(targets, points) W L is one kernel evaluation and
           one matrix product per face (JAX, on the device);
  near     only for the (target, panel) pairs closer than ``c_near`` panel lengths,
           the panel's regular contribution is replaced by ``single_layer._panel_rule``
           on that panel alone (graded towards the target's closest point, the u^q
           singular panels for a target on the face): the rule of the slow path,
           restricted to where the regular rule is not converged.

Far from a panel (distance >= c_near x its length) the n_gl-point rule on it converges
like rho^(-2 n_gl), rho the Bernstein-ellipse parameter (~2.4 at c_near = 1: 1e-12 for
n_gl = 16; ~3.6 at 1.5).  The kernel: Q_{-1/2} = k K(k), Q_{1/2} = chi k K - 2E / k near
(eta = arccosh chi <= 1/2), and Q_{1/2} = rho_0 Q_{-1/2} from the backward (Miller) ratio
sweep elsewhere, as ``toroidal_kernel.toroidal_Q``; K, E by the AGM of the complementary
parameter.  Only |m| <= 1 (axisymmetric fields: N_phi = 1); ``single_layer_matrix``
dispatches here then.
"""

import time

import jax
import jax.numpy as jnp
import numpy as np

from annulus_poisson import FACES
from single_layer import _face_geometry, _panel_rule
from toroidal_kernel import toroidal_Q


def _ellipke(m1, n_iter=24):
    """K(m), E(m) from the complementary parameter m1 = 1 - m (AGM)."""
    a, g = jnp.ones_like(m1), jnp.sqrt(m1)
    s, p = 0.5 * (1.0 - m1), 0.5
    for _ in range(n_iter):
        a, g, c = 0.5 * (a + g), jnp.sqrt(a * g), 0.5 * (a - g)
        p = 2 * p
        s = s + p * c * c
    K = jnp.pi / (2 * a)
    return K, K * (1 - s)


def q_half(chi_m1, n_miller=96):
    """(Q_{-1/2}, Q_{1/2})(chi) from chi - 1 > 0, as toroidal_Q (JAX, any shape)."""
    x = chi_m1
    chi = 1.0 + x
    p = x / (chi + 1.0)  # 1 - k^2
    k = jnp.sqrt(2.0 / (chi + 1.0))
    K, E = _ellipke(p)
    Q0 = k * K
    eta = jnp.log1p(x + jnp.sqrt(x * (chi + 1.0)))
    near = chi * k * K - (2.0 / k) * E
    rho = jnp.zeros_like(x)  # rho_{m-1} = (m - 1/2) / (2 m chi - (m + 1/2) rho_m), from rho_N = 0
    for m in range(n_miller, 0, -1):
        rho = (m - 0.5) / (2 * m * chi - (m + 0.5) * rho)
    far = rho * Q0
    return Q0, jnp.where(eta <= 0.5, near, far)


@jax.jit
def _regular_block(rt, zt, rq, zq, wrq, L):
    """K W L for a block of targets: (2, n_t, n_f) for |m| = 0, 1."""
    d2 = (rt[:, None] - rq[None, :]) ** 2 + (zt[:, None] - zq[None, :]) ** 2
    rr = rt[:, None] * rq[None, :]
    Q0, Q1 = q_half(d2 / (2 * rr))
    c = -0.5 * wrq[None, :] / (jnp.pi * jnp.sqrt(rr))
    return jnp.stack([(Q0 * c) @ L, (Q1 * c) @ L])


def single_layer_matrix_fast(ap, ms, targets, n_gl=16, ratio=0.15, c_near=1.5, dtype=np.float64,
                             verbose=False, block=512):
    """As ``single_layer.single_layer_matrix`` (same shape and order), |m| <= 1."""
    targets = np.atleast_2d(np.asarray(targets, dtype=float))
    am = np.abs(np.asarray(ms))
    if am.max() > 1:
        raise ValueError("the fast path holds |m| <= 1 only")
    geo = _face_geometry(ap)
    sizes = ap.face_sizes
    offs = np.cumsum([0] + [sizes[f] for f in FACES])
    n_t = len(targets)
    S = np.zeros((2, n_t, ap.n_boundary))
    xg, wg = np.polynomial.legendre.leggauss(n_gl)
    t0 = time.time()
    rt_d, zt_d = jnp.asarray(targets[:, 0]), jnp.asarray(targets[:, 1])
    n_near = 0
    for fi, f in enumerate(FACES):
        axis, (which, c) = geo[f]
        s = axis.s
        lo, hi = s[:-1], s[1:]
        q = ((lo + hi) / 2)[:, None] + ((hi - lo) / 2)[:, None] * xg[None, :]  # (n_panels, n_gl)
        wq = ((hi - lo) / 2)[:, None] * wg[None, :]
        L = axis.interp_matrix(q.ravel())  # (n_panels n_gl, n_f)
        rq = np.full(q.size, c) if which == "r" else q.ravel()
        zq = q.ravel() if which == "r" else np.full(q.size, c)
        wrq = wq.ravel() * rq
        Ld, rqd, zqd, wd = jnp.asarray(L), jnp.asarray(rq), jnp.asarray(zq), jnp.asarray(wrq)
        cols = slice(offs[fi], offs[fi + 1])
        for i in range(0, n_t, block):
            S[:, i:i + block, cols] = np.asarray(_regular_block(rt_d[i:i + block], zt_d[i:i + block],
                                                                rqd, zqd, wd, Ld))
        # near corrections: per target, the panels closer than c_near panel lengths
        t_along = targets[:, 1] if which == "r" else targets[:, 0]
        t_norm = targets[:, 0] - c if which == "r" else targets[:, 1] - c
        gap = np.maximum(0.0, np.maximum(lo[None, :] - t_along[:, None], t_along[:, None] - hi[None, :]))
        dist = np.hypot(t_norm[:, None], gap)
        near = dist < c_near * (hi - lo)[None, :]
        for t, ip in zip(*np.nonzero(near)):
            rt, ta, tn = targets[t, 0], t_along[t], t_norm[t]
            p = min(max(ta, lo[ip]), hi[ip])
            l, w, off = _panel_rule(np.array([lo[ip], hi[ip]]), p, dist[t, ip], n_gl, ratio)
            rp = np.full_like(l, c) if which == "r" else l
            d2 = tn**2 + ((ta - p) - off) ** 2
            Qa = toroidal_Q(d2 / (2 * rt * rp), 1)
            acc = (Qa * (-0.5 * w * rp / (np.pi * np.sqrt(rt * rp)))[None, :]) @ axis.interp_matrix(l)
            # minus the panel's regular contribution (its n_gl points)
            rq_p = rq.reshape(-1, n_gl)[ip]
            zq_p = zq.reshape(-1, n_gl)[ip]
            d2r = (rt - rq_p) ** 2 + (targets[t, 1] - zq_p) ** 2
            Qr = toroidal_Q(d2r / (2 * rt * rq_p), 1)
            reg = (Qr * (-0.5 * wrq.reshape(-1, n_gl)[ip] / (np.pi * np.sqrt(rt * rq_p)))[None, :]) \
                @ L.reshape(len(lo), n_gl, -1)[ip]
            S[:, t, cols] += acc - reg
        n_near += int(near.sum())
        if verbose:
            print(f"    S_m (fast): face {f}: {q.size} regular points, {int(near.sum())} near panels, "
                  f"{time.time() - t0:.0f} s", flush=True)
    out = S[am].astype(dtype)
    if verbose:
        print(f"    S_m (fast): {n_t} targets, {n_near} near (target, panel) pairs, "
              f"{time.time() - t0:.0f} s", flush=True)
    return out
