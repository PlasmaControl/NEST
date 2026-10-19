"""Manufactured exact fields for the cylindrical J -> B solver (gates B2, B3).

psi is a smooth, NFP-periodic vector field in the annulus (Gaussian bumps in
(r, z) times functions of phi; no z- or stellarator symmetry; one component with
a von Mises factor exp(kappa cos(NFP (phi - phi0))), a broad toroidal spectrum).
Then

    A = curl psi   (div A = 0),   B = curl A,   J = curl B / mu0 = -Delta A / mu0,

so A is the free-space Coulomb-gauge solution of Delta A = -mu0 J (it decays and
J has the same Gaussian support), and B is exact everywhere, including inside
the current. chi is a Gaussian scalar for the gradient test of B3 (grad chi has
no curl, so B(J + grad chi) = B(J) and curl B - mu0 (J + grad chi) = -mu0 grad chi).
All derivatives are JAX autodiff (float64).
"""

import jax
import jax.numpy as jnp
import numpy as np
from scipy.constants import mu_0

jax.config.update("jax_enable_x64", True)

# (r0, z0, s) of the bump of each cylindrical component of psi
BUMPS = {"r": (1.00, 0.03, 0.036), "phi": (0.97, -0.04, 0.034), "z": (1.04, 0.02, 0.038)}


def _psi_cyl(r, phi, z, NFP):
    def b(c):
        r0, z0, s = BUMPS[c]
        return jnp.exp(-((r - r0) ** 2 + (z - z0) ** 2) / (2 * s**2))
    vm = jnp.exp(6.0 * (jnp.cos(NFP * (phi - 0.4)) - 1.0))  # von Mises, kappa = 6
    p_r = b("r") * (0.6 + jnp.cos(NFP * phi + 0.3)) * 0.01
    p_phi = b("phi") * (0.4 + vm) * 0.01
    p_z = b("z") * (1.0 + 0.5 * jnp.sin(2 * NFP * phi + 1.1) + 0.3 * jnp.cos(NFP * phi)) * 0.01
    return p_r, p_phi, p_z


def _psi(x, NFP):
    r = jnp.sqrt(x[0] ** 2 + x[1] ** 2)
    phi = jnp.arctan2(x[1], x[0])
    pr, pp, pz = _psi_cyl(r, phi, x[2], NFP)
    c, s = jnp.cos(phi), jnp.sin(phi)
    return jnp.stack([pr * c - pp * s, pr * s + pp * c, pz])


def _curl(f):
    def g(x, *a):
        Jm = jax.jacfwd(f)(x, *a)
        return jnp.stack([Jm[2, 1] - Jm[1, 2], Jm[0, 2] - Jm[2, 0], Jm[1, 0] - Jm[0, 1]])
    return g


def _chi(x, NFP):
    """Gaussian in (r, z) at (0.95, -0.05) times a von Mises factor in phi (NFP-periodic,
    resolved like psi; an isotropic 3-D Gaussian is ~0.04 rad wide in phi, finer
    than the planes of the test grids)."""
    r = jnp.sqrt(x[0] ** 2 + x[1] ** 2)
    phi = jnp.arctan2(x[1], x[0])
    g = jnp.exp(-((r - 0.95) ** 2 + (x[2] + 0.05) ** 2) / (2 * 0.036**2))
    return 0.02 * g * (0.3 + jnp.exp(6.0 * (jnp.cos(NFP * (phi - 1.0)) - 1.0)))


def fields(nodes, NFP, chunk=20000):
    """J, B, A, grad chi (cylindrical components) at nodes (..., 3) = (r, phi, z)."""
    A_f = _curl(_psi)
    B_f = _curl(A_f)
    J_f = _curl(B_f)
    gchi = jax.grad(_chi)

    @jax.jit
    def one(x):
        return jnp.stack([J_f(x, NFP) / mu_0, B_f(x, NFP), A_f(x, NFP), gchi(x, NFP)])

    pts = np.asarray(nodes).reshape(-1, 3)
    X = np.column_stack([pts[:, 0] * np.cos(pts[:, 1]), pts[:, 0] * np.sin(pts[:, 1]), pts[:, 2]])
    v = jax.vmap(one)
    out = np.concatenate([np.asarray(v(jnp.asarray(X[i:i + chunk]))) for i in range(0, len(X), chunk)])
    c, s = np.cos(pts[:, 1]), np.sin(pts[:, 1])

    def cyl(V):
        return np.stack([V[:, 0] * c + V[:, 1] * s, -V[:, 0] * s + V[:, 1] * c, V[:, 2]], axis=-1)

    shape = np.shape(nodes)
    return {k: cyl(out[:, i]).reshape(shape) for i, k in enumerate(("J", "B", "A", "grad_chi"))}


def wall_values(F, solver_shape=None):
    """max |F| on the four walls of the (N_phi, n_r + 1, n_z + 1, 3) grid."""
    return max(np.abs(F[:, 0]).max(), np.abs(F[:, -1]).max(), np.abs(F[:, :, 0]).max(),
               np.abs(F[:, :, -1]).max())
