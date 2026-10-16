"""Chebyshev-Gauss-Lobatto axes for the cylindrical J -> B solver.

An axis is the interval [a, b] with n + 1 nodes s_j = a + (b - a) (x_j + 1) / 2,
where x_j = g(y_j) and y_j = -cos(pi j / n) are the Chebyshev-Gauss-Lobatto
points in ascending order. The map g is the identity (``alpha = 0``) or the
Kosloff-Tal-Ezer map

    g(y) = arcsin(alpha y) / arcsin(alpha),    0 < alpha < 1,

which spreads the nodes towards uniform spacing (alpha -> 1) at the price of a
slower spectral convergence rate. Every field is a polynomial in y, so the
barycentric weights, the interpolation and the Clenshaw-Curtis quadrature are
those of the plain Chebyshev points in y; derivatives carry the chain rule
d/ds = (2 / (b - a)) (1 / g'(y)) d/dy.

All operators are dense NumPy float64 arrays.
"""

import numpy as np


def cgl_points(n):
    """y_j = -cos(pi j / n), j = 0..n, ascending, in the symmetric form
    sin(pi (2 j - n) / (2 n)) (exactly antisymmetric, no cancellation)."""
    j = np.arange(n + 1)
    return np.sin(np.pi * (2 * j - n) / (2 * n))


def cgl_weights(n):
    """Barycentric weights (-1)^j delta_j of the CGL points (delta = 1/2 at the ends)."""
    w = (-1.0) ** np.arange(n + 1)
    w[0] *= 0.5
    w[-1] *= 0.5
    return w


def _differences(n):
    """y_i - y_j for the CGL points, by the product formula (no cancellation).

    With y_k = -cos(pi k / n):  y_i - y_j = 2 sin(pi (i + j) / 2n) sin(pi (i - j) / 2n).
    """
    i = np.arange(n + 1)[:, None]
    j = np.arange(n + 1)[None, :]
    return 2 * np.sin(np.pi * (i + j) / (2 * n)) * np.sin(np.pi * (i - j) / (2 * n))


def cheb_D(n):
    """Differentiation matrix d/dy on the n + 1 ascending CGL points.

    Barycentric form D_ij = (w_j / w_i) / (y_i - y_j) for i != j, and the
    diagonal from the row sums (D applied to a constant is exactly zero).
    """
    w = cgl_weights(n)
    dy = _differences(n)
    np.fill_diagonal(dy, 1.0)
    D = (w[None, :] / w[:, None]) / dy
    np.fill_diagonal(D, 0.0)
    np.fill_diagonal(D, -D.sum(axis=1))
    return D


def clenshaw_curtis(n):
    """Clenshaw-Curtis weights on the n + 1 CGL points (integral over [-1, 1])."""
    theta = np.pi * np.arange(n + 1) / n
    w = np.zeros(n + 1)
    v = np.ones(n - 1)
    if n % 2 == 0:
        w[0] = w[n] = 1.0 / (n**2 - 1)
        for k in range(1, n // 2):
            v -= 2 * np.cos(2 * k * theta[1:-1]) / (4 * k**2 - 1)
        v -= np.cos(n * theta[1:-1]) / (n**2 - 1)
    else:
        w[0] = w[n] = 1.0 / n**2
        for k in range(1, (n - 1) // 2 + 1):
            v -= 2 * np.cos(2 * k * theta[1:-1]) / (4 * k**2 - 1)
    w[1:-1] = 2 * v / n
    return w  # symmetric, so the ascending order needs no flip


def barycentric_matrix(y_nodes, w, y):
    """Rows of Lagrange basis values l_j(y_t) for targets y (polynomial in y)."""
    y = np.atleast_1d(np.asarray(y, dtype=float))
    d = y[:, None] - y_nodes[None, :]
    exact = d == 0
    d = np.where(exact, 1.0, d)
    c = w[None, :] / d
    M = c / c.sum(axis=1, keepdims=True)
    hit = exact.any(axis=1)
    if hit.any():
        M[hit] = exact[hit].astype(float)
    return M


class ChebAxis:
    """Chebyshev-Gauss-Lobatto axis on [a, b], n + 1 nodes, optional KTE map.

    Attributes
    ----------
    s : (n + 1,) nodes in [a, b], ascending, s[0] = a and s[-1] = b.
    D, D2 : d/ds and d^2/ds^2 on the nodes.
    w_int : quadrature weights for the integral over [a, b] (Clenshaw-Curtis in y).
    """

    def __init__(self, a, b, n, alpha=0.0):
        if not (0.0 <= alpha < 1.0):
            raise ValueError("alpha must be in [0, 1)")
        self.a, self.b, self.n, self.alpha = float(a), float(b), int(n), float(alpha)
        self.y = cgl_points(n)
        self.w_bary = cgl_weights(n)
        x = self._g(self.y)
        x[0], x[-1] = -1.0, 1.0
        self.s = self.a + (self.b - self.a) * (x + 1) / 2
        self.s[0], self.s[-1] = self.a, self.b
        # ds/dy at the nodes
        self.dsdy = (self.b - self.a) / 2 * self._dg(self.y)
        Dy = cheb_D(n)
        self.D = Dy / self.dsdy[:, None]
        self.D2 = self.D @ self.D
        self.w_int = clenshaw_curtis(n) * self.dsdy

    # ------------------------------------------------------------ the map
    def _g(self, y):
        if self.alpha == 0.0:
            return np.array(y, dtype=float)
        return np.arcsin(self.alpha * y) / np.arcsin(self.alpha)

    def _dg(self, y):
        if self.alpha == 0.0:
            return np.ones_like(np.asarray(y, dtype=float))
        return self.alpha / (np.arcsin(self.alpha) * np.sqrt(1 - (self.alpha * y) ** 2))

    def to_y(self, s):
        """Inverse map: the y in [-1, 1] of points s in [a, b]."""
        x = 2 * (np.asarray(s, dtype=float) - self.a) / (self.b - self.a) - 1
        if self.alpha == 0.0:
            return x
        return np.sin(x * np.arcsin(self.alpha)) / self.alpha

    # ------------------------------------------------------- evaluation
    def interp_matrix(self, s):
        """(n_targets, n + 1) matrix taking node values to values at s."""
        return barycentric_matrix(self.y, self.w_bary, self.to_y(s))

    def spacing(self):
        """Node spacings s[j + 1] - s[j]."""
        return np.diff(self.s)

    def max_spacing_in(self, lo, hi):
        """Largest node spacing among the intervals that overlap [lo, hi]."""
        d = np.diff(self.s)
        sel = (self.s[1:] > lo) & (self.s[:-1] < hi)
        return float(d[sel].max())

    @classmethod
    def for_spacing(cls, a, b, lo, hi, h, alpha=0.0, n_min=4, n_max=4096):
        """Smallest axis on [a, b] whose spacing over [lo, hi] is at most h."""
        for n in range(n_min, n_max):
            ax = cls(a, b, n, alpha)
            if ax.max_spacing_in(lo, hi) <= h:
                return ax
        raise ValueError("no n below n_max meets the spacing")
