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


def single_layer_matrix(ap, ms, targets, n_gl=16, ratio=0.15, verbose=False, dtype=np.float64):
    """S_m from the packed face values of sigma to S_m[sigma] at targets.

    Parameters
    ----------
    ap : AnnulusPoisson
    ms : sequence of int (|m| is used); the result is in this order.
    targets : (n_t, 2) array of (r, z).

    Returns (len(ms), n_t, ap.n_boundary), of ``dtype`` (float32 halves the largest
    precomputed array; entries are computed in float64 either way).
    """
    ms = np.asarray(ms)
    am = np.abs(ms)
    m_max = int(am.max())
    targets = np.atleast_2d(np.asarray(targets, dtype=float))
    geo = _face_geometry(ap)
    sizes = ap.face_sizes
    offs = np.cumsum([0] + [sizes[f] for f in FACES])
    S = np.zeros((len(ms), len(targets), ap.n_boundary), dtype=dtype)
    for it, (rt, zt) in enumerate(targets):
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
        rp = np.concatenate([x[0] for x in rows])
        d2 = np.concatenate([x[1] for x in rows])
        w = np.concatenate([x[2] for x in rows])
        rr = rt * rp
        Q = toroidal_Q(d2 / (2 * rr), m_max)[am]  # (n_m, n_q)
        kw = Q * (-0.5 * w * rp / (np.pi * np.sqrt(rr)))[None, :]
        i0 = 0
        for col, Lm in zip(cols, lags):
            nq = Lm.shape[0]
            S[:, it, col] = kw[:, i0:i0 + nq] @ Lm
            i0 += nq
        if verbose and (it + 1) % max(1, len(targets) // 10) == 0:
            print(f"    S_m: {it + 1}/{len(targets)} targets", flush=True)
    return S
