"""Single-layer potential of the annulus boundary, one toroidal mode at a time.

For a density sigma(l') e^{i m phi'} on the boundary of the annulus (its four
faces, ``annulus_poisson.FACES``), the potential

    S[sigma](x) = -1 / (4 pi) int sigma(y) / |x - y| dS_y

is S_m[sigma](r, z) e^{i m phi} with

    S_m[sigma](r, z) = -1/2 int_faces r' k_m(r, z; r', z') sigma(l') dl',

k_m the toroidal kernel (``toroidal_kernel``): dS = r' dphi' dl' on every face
(dl' = dz' on the lateral faces, dr' on the caps), and the phi' integral of
e^{i m phi'} / |x - y| is 2 pi k_m e^{i m phi}.

``single_layer_matrix`` gives the matrix from the face-by-face nodal values of
sigma (each face's Chebyshev interpolant, packed as in ``AnnulusPoisson``) to
S_m[sigma] at target points (r, z). The l'-integral of each face is a composite
Gauss-Legendre rule on panels bounded by the face nodes and by points graded
geometrically towards the point p* of the face closest to the target, down to a
quarter of the target's distance from the face; for a target on the face (the
log singularity of k_m) down to 1e-11 of the face length, the two panels at p*
with a u^6 substitution that makes the log integrand smooth. The kernel is the same
for m and -m, so only |m| enters. The default rule (16 points, grading ratio
0.15) is converged to ~1e-12 up to |m| = 300 on the A5 test box (gate A5).
"""

import numpy as np

from annulus_poisson import FACES
from toroidal_kernel import toroidal_Q


def _face_geometry(ap):
    """Per face: its axis (the Chebyshev axis of the face coordinate), the fixed
    coordinate, and which of (r, z) varies."""
    r, z = ap.r_axis, ap.z_axis
    return {"in": (z, ("r", r.a)), "out": (z, ("r", r.b)),
            "bot": (r, ("z", z.a)), "top": (r, ("z", z.b))}


def _panel_rule(nodes, p_star, delta, n_gl, ratio=0.15, eps=1e-11, q=6):
    """Composite Gauss-Legendre points and weights on [nodes[0], nodes[-1]].

    Panels: the face nodes, and points p* +- L ratio^k graded down to
    stop = max(delta / 4, eps L). If the target is within stop of the face, the
    two panels [p* - stop, p*] and [p*, p* + stop] carry the log singularity:
    they use s = p* -+ stop u^q (u in (0, 1), Gauss-Legendre), which makes
    log|s - p*| ds smooth in u. (Grading on to round-off would put Gauss points
    on the target itself: the face coordinate has ulp ~ 1e-16.)
    """
    a, b = nodes[0], nodes[-1]
    L = b - a
    stop = max(delta / 4, eps * L)
    br = [nodes]
    s = L * ratio
    while s > stop:
        br.append([p_star - s, p_star + s])
        s *= ratio
    br.append([p_star - stop, p_star, p_star + stop])
    br = np.unique(np.clip(np.concatenate([np.ravel(x) for x in br]), a, b))
    x, w = np.polynomial.legendre.leggauss(n_gl)
    lo, hi = br[:-1], br[1:]
    keep = hi > lo
    lo, hi = lo[keep], hi[keep]
    sing = np.zeros(len(lo), bool)
    if delta < stop:  # the target is (practically) on the face: singular end panels
        sing = (lo == p_star) | (hi == p_star)
    # offsets from p* (kept exactly: near p* they are far below the ulp of p*)
    off = (lo - p_star)[:, None] + (hi - lo)[:, None] / 2 * (x[None, :] + 1)
    wts = (hi - lo)[:, None] / 2 * w[None, :]
    if sing.any():
        u = (x + 1) / 2
        for i in np.flatnonzero(sing):
            h = hi[i] - lo[i]
            sign = 1.0 if lo[i] == p_star else -1.0
            off[i] = sign * h * u**q
            wts[i] = h * q * u ** (q - 1) * w / 2
    return p_star + off.ravel(), wts.ravel(), off.ravel()


def _block(geo_ns, ms, targets, n_gl, ratio, dtype):
    """One block of targets (process-pool worker; spawn-safe, module level)."""
    return single_layer_matrix(geo_ns, ms, targets, n_gl, ratio, dtype=dtype, n_jobs=1)


def single_layer_matrix(ap, ms, targets, n_gl=16, ratio=0.15, verbose=False, dtype=np.float64,
                        n_jobs=None):
    """S_m from the packed face values of sigma to S_m[sigma] at targets.

    Parameters
    ----------
    ap : AnnulusPoisson
    ms : sequence of int (|m| is used); the result is in this order.
    targets : (n_t, 2) array of (r, z).
    n_jobs : int or None
        Worker processes over blocks of targets (each row is independent, and a
        block is computed exactly as serially, so the result is bitwise the same).
        None: the env NEST_S_JOBS, else 1 (serial). Opt-in: not measured faster on
        the laptop (the serial path batches the Q recurrence over targets instead).

    Returns (len(ms), n_t, ap.n_boundary), of ``dtype`` (float32 halves the largest
    precomputed array; entries are computed in float64 either way).
    """
    import os
    targets = np.atleast_2d(np.asarray(targets, dtype=float))
    if n_jobs is None:
        n_jobs = int(os.environ.get("NEST_S_JOBS", 1))
    n_jobs = max(1, min(int(n_jobs), len(targets) // 8))
    if n_jobs > 1:
        import multiprocessing as mp
        import time
        from concurrent.futures import ProcessPoolExecutor
        from types import SimpleNamespace
        geo_ns = SimpleNamespace(r_axis=ap.r_axis, z_axis=ap.z_axis, face_sizes=dict(ap.face_sizes),
                                 n_boundary=ap.n_boundary)
        blocks = np.array_split(np.arange(len(targets)), 4 * n_jobs)
        S = np.zeros((len(ms), len(targets), ap.n_boundary), dtype=dtype)
        t0 = time.time()
        # one BLAS thread per worker (the children inherit the environment at spawn)
        pin = {k: "1" for k in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS",
                                "VECLIB_MAXIMUM_THREADS")}
        saved = {k: os.environ.get(k) for k in pin}
        os.environ.update(pin)
        try:  # processes (spawn: safe with JAX threads); threads where the OS forbids them
            ex = ProcessPoolExecutor(n_jobs, mp_context=mp.get_context("spawn"))
            kind = "processes"
        except (OSError, PermissionError, NotImplementedError):
            from concurrent.futures import ThreadPoolExecutor
            ex, kind = ThreadPoolExecutor(n_jobs), "threads"
        with ex:
            futs = [(b, ex.submit(_block, geo_ns, np.asarray(ms), targets[b], n_gl, ratio, dtype))
                    for b in blocks]
            for k, v in saved.items():  # the workers have started
                if v is None:
                    os.environ.pop(k, None)
                else:
                    os.environ[k] = v
            for i, (b, f) in enumerate(futs):
                S[:, b] = f.result()
                if verbose and (i + 1) % max(1, len(futs) // 10) == 0:
                    print(f"    S_m: {b[-1] + 1}/{len(targets)} targets ({n_jobs} {kind}, "
                          f"{time.time() - t0:.0f} s)", flush=True)
        return S
    ms = np.asarray(ms)
    am = np.abs(ms)
    m_max = int(am.max())
    geo = _face_geometry(ap)
    sizes = ap.face_sizes
    offs = np.cumsum([0] + [sizes[f] for f in FACES])
    S = np.zeros((len(ms), len(targets), ap.n_boundary), dtype=dtype)
    def geometry(rt, zt):
        """The quadrature of every face for one target: (rp, d2, w) and the Lagrange maps."""
        rows, cols, lags = [], [], []
        for fi, f in enumerate(FACES):
            axis, (which, c) = geo[f]
            # closest point of the face to the target, and its distance
            t_along = zt if which == "r" else rt
            t_norm = rt - c if which == "r" else zt - c
            p = min(max(t_along, axis.a), axis.b)
            delta = np.hypot(t_norm, t_along - p)
            l, w, off = _panel_rule(axis.s, p, delta, n_gl, ratio)
            rp = np.full_like(l, c) if which == "r" else l
            # distance from the exact offsets: along the face (t_along - p) - off, across t_norm
            d2 = t_norm**2 + ((t_along - p) - off) ** 2
            assert (d2 > 0).all()
            rows.append((rp, d2, w))
            lags.append(axis.interp_matrix(l))
            cols.append(slice(offs[fi], offs[fi + 1]))
        return (np.concatenate([x[0] for x in rows]), np.concatenate([x[1] for x in rows]),
                np.concatenate([x[2] for x in rows]), cols, lags)

    # targets in blocks sharing one toroidal_Q call: its recurrences are elementwise, so
    # this is bitwise the per-target result with the Python loop amortized over the block
    block_bytes = float(os.environ.get("NEST_S_BLOCK_BYTES", 2e8))
    it, done = 0, 0
    while it < len(targets):
        geos, n_q = [], 0
        while it < len(targets) and (not geos or 16 * (m_max + 1) * n_q < block_bytes):
            g = geometry(*targets[it])
            geos.append((it, g))
            n_q += len(g[0])
            it += 1
        rt_all = np.concatenate([np.full(len(g[0]), targets[j][0]) for j, g in geos])
        rp_all = np.concatenate([g[0] for _, g in geos])
        rr_all = rt_all * rp_all
        Q_all = toroidal_Q(np.concatenate([g[1] for _, g in geos]) / (2 * rr_all), m_max)[am]
        q0 = 0
        for j, (rp, d2, w, cols, lags) in geos:
            nq = len(rp)
            Q, rr = Q_all[:, q0:q0 + nq], rr_all[q0:q0 + nq]
            q0 += nq
            kw = Q * (-0.5 * w * rp / (np.pi * np.sqrt(rr)))[None, :]
            i0 = 0
            for col, Lm in zip(cols, lags):
                nq_f = Lm.shape[0]
                S[:, j, col] = kw[:, i0:i0 + nq_f] @ Lm
                i0 += nq_f
        if verbose and (it - done) and (it * 10 // len(targets) > done * 10 // len(targets)):
            print(f"    S_m: {it}/{len(targets)} targets", flush=True)
        done = it
    return S
