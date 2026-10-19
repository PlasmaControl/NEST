"""Toroidal harmonics: the toroidal Fourier modes of the free-space kernel 1 / |x - y|.

For x = (r, phi, z) and y = (r', phi', z'), with d^2 = (r - r')^2 + (z - z')^2,

    1 / |x - y| = sum_m k_m e^{i m (phi - phi')},
    k_m = (1 / 2 pi) int_0^{2 pi} cos(m t) / sqrt(d^2 + 4 r r' sin^2(t / 2)) dt
        = Q_{|m| - 1/2}(chi) / (pi sqrt(r r')),          chi = 1 + d^2 / (2 r r'),

with Q_{m - 1/2} the Legendre function of the second kind of half-integer degree.
``toroidal_Q`` returns Q_{m - 1/2}(chi), m = 0..m_max, from chi - 1 (passed
directly, so that near-coincident points keep their digits):

* Q_{-1/2} = k K(k), k^2 = 2 / (chi + 1), with K from ``scipy.special.ellipkm1``
  of p = 1 - k^2 = (chi - 1) / (chi + 1) (accurate as p -> 0, the log singularity);
* where m_max eta <= 2 and eta <= 1/2 (eta = arccosh chi, near-coincident points), the forward
  recurrence (m + 1/2) Q_{m+1/2} = 2 m chi Q_{m-1/2} - (m - 1/2) Q_{m-3/2} from
  Q_{-1/2} and Q_{1/2} = chi k K - (2 / k) E; its error grows like e^{2 m eta}
  <= e^4 there;
* elsewhere, Q is the minimal solution of the recurrence: the ratios
  rho_m = Q_{m+1/2} / Q_{m-1/2} come from the backward (Miller) sweep
  rho_{m-1} = (m - 1/2) / (2 m chi - (m + 1/2) rho_m) started at rho_N = 0,
  N = m_max + 40 / eta (start error e^{-2 eta (N - m)} <= e^{-80}), and
  Q_{m+1/2} = rho_m Q_{m-1/2}. Points are grouped by N.
"""

import numpy as np
from scipy.special import ellipe, ellipkm1


def _Q_start(chi_m1):
    """Q_{-1/2} and Q_{1/2} from chi - 1 (complete elliptic integrals)."""
    chi_m1 = np.asarray(chi_m1, dtype=float)
    chi = 1.0 + chi_m1
    p = chi_m1 / (chi + 1.0)  # 1 - k^2
    k = np.sqrt(2.0 / (chi + 1.0))
    K = ellipkm1(p)
    Q0 = k * K
    Q1 = chi * k * K - (2.0 / k) * ellipe(1.0 - p)
    return Q0, Q1


def toroidal_Q(chi_m1, m_max, group=256):
    """Q_{m - 1/2}(chi) for m = 0..m_max at chi = 1 + chi_m1 (chi_m1 > 0).

    Returns an array of shape (m_max + 1,) + chi_m1.shape.
    """
    chi_m1 = np.asarray(chi_m1, dtype=float)
    shape = chi_m1.shape
    x = chi_m1.ravel()
    if np.any(x <= 0):
        raise ValueError("chi - 1 must be positive (coincident points)")
    chi = 1.0 + x
    # arccosh(chi) without cancellation: log(chi + sqrt((chi - 1)(chi + 1)))
    eta = np.log1p(x + np.sqrt(x * (chi + 1.0)))
    Q0, Q1 = _Q_start(x)
    out = np.empty((m_max + 1, x.size))
    out[0] = Q0
    if m_max == 0:
        return out.reshape((1,) + shape)
    # (eta <= 0.5 also keeps Q_{1/2} = chi k K - 2E / k free of cancellation)
    fwd = (m_max * eta <= 2.0) & (eta <= 0.5)
    # forward recurrence (near-coincident)
    if fwd.any():
        i = np.flatnonzero(fwd)
        c = chi[i]
        qm, q = Q0[i], Q1[i]
        out[1, i] = q
        for m in range(1, m_max):
            qn = (2 * m * c * q - (m - 0.5) * qm) / (m + 0.5)
            out[m + 1, i] = qn
            qm, q = q, qn
    # Miller backward recurrence of the ratios, grouped by start index N
    if (~fwd).any():
        i_all = np.flatnonzero(~fwd)
        N_all = m_max + np.ceil(40.0 / eta[i_all]).astype(int)
        Nb = (N_all + group - 1) // group * group  # buckets of equal N
        for N in np.unique(Nb):
            i = i_all[Nb == N]
            c = chi[i]
            rho = np.zeros(len(i))
            ratios = np.empty((m_max, len(i)))  # rho_0 .. rho_{m_max - 1}
            for m in range(N, 0, -1):
                rho = (m - 0.5) / (2 * m * c - (m + 0.5) * rho)  # this is rho_{m-1}
                if m - 1 < m_max:
                    ratios[m - 1] = rho
            q = Q0[i]
            for m in range(m_max):
                q = q * ratios[m]
                out[m + 1, i] = q
    return out.reshape((m_max + 1,) + shape)


def toroidal_kernel(r, z, rp, zp, m_max):
    """k_m(r, z; r', z') for m = 0..m_max (broadcast over the point arrays).

    Shape (m_max + 1,) + broadcast shape. 1 / |x - y| = sum_m k_m e^{i m (phi - phi')}.
    """
    r, z, rp, zp = np.broadcast_arrays(*(np.asarray(a, dtype=float) for a in (r, z, rp, zp)))
    d2 = (r - rp) ** 2 + (z - zp) ** 2
    rr = r * rp
    Q = toroidal_Q(d2 / (2 * rr), m_max)
    return Q / (np.pi * np.sqrt(rr))
