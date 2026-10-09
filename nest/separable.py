"""Fourier-Zernike series regrouped by the toroidal factor, evaluated with JAX.

A series f(rho, theta, zeta) = sum_k c_k Z_k(rho) T_k(theta) F_k(zeta) in
DESC's basis is stored as a tensor C[p, q, c] over the distinct (l, m) pairs
p, the distinct n values q and the components c, so that

    f_c = sum_p P_p(rho, theta) sum_q C[p, q, c] F_q(zeta),

with P_p = Z_l^|m|(rho) T_m(theta). At fixed zeta the inner sum collapses to a
2-D Zernike series with coefficients a[p, c] = sum_q C[p, q, c] F_q.

``scatter`` builds C from coefficient arrays with JAX and ``evaluate`` accepts
it as an argument, so a series can be evaluated (and differentiated) for
coefficients that are traced JAX arrays, with the regrouping fixed.
"""

import jax.numpy as jnp
import numpy as np

from desc.basis import fourier, zernike_radial


class Separable:
    """Several Fourier-Zernike series (components) sharing one regrouping."""

    def __init__(self, bases, coefs, NFP):
        modes = np.concatenate([b.modes for b in bases])
        self.lm, p = np.unique(modes[:, :2], axis=0, return_inverse=True)
        self.n, q = np.unique(modes[:, 2], return_inverse=True)
        self.NFP = NFP
        bounds = np.cumsum([0] + [b.num_modes for b in bases])
        self._pq = [(p.ravel()[i:j], q.ravel()[i:j]) for i, j in zip(bounds[:-1], bounds[1:])]
        self.C = self.scatter(coefs)

    def scatter(self, coefs):
        """C[p, q, c] from one coefficient array per basis (JAX, differentiable)."""
        C = jnp.zeros((len(self.lm), len(self.n), len(self._pq)))
        for c, ((p, q), coef) in enumerate(zip(self._pq, coefs)):
            C = C.at[p, q, c].set(jnp.asarray(coef, dtype=C.dtype))
        return C

    def poloidal(self, rho, theta, dr=0, dt=0):
        """P_p(rho, theta) and derivatives, shape (n_points, n_lm)."""
        l, m = self.lm[:, 0], self.lm[:, 1]
        return zernike_radial(rho[:, None], l, m, dr=dr) * fourier(theta[:, None], m, dt=dt)

    def toroidal(self, zeta, dz=0):
        """F_q(zeta) and derivatives, shape (n_points, n_n)."""
        return fourier(zeta[:, None], self.n, NFP=self.NFP, dt=dz)

    def collapse(self, zeta):
        """2-D coefficients a[i, p, c] at each point's zeta."""
        return jnp.einsum("iq,pqc->ipc", self.toroidal(zeta), self.C)

    def evaluate(self, rho, theta, zeta, dr=0, dt=0, dz=0, C=None):
        """f_c (or a derivative) at the points, shape (n_points, n_components).

        ``C`` (from ``scatter``) replaces the stored coefficients.
        """
        C = self.C if C is None else C
        # (P C) first, then the toroidal factor: never forms (points, p, q, c)
        PC = jnp.einsum("ip,pqc->iqc", self.poloidal(rho, theta, dr, dt), C)
        return jnp.einsum("iqc,iq->ic", PC, self.toroidal(zeta, dz))
