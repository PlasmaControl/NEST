"""The Jacobian of the objective's residual: explicit columns and matrix-free tools.

The full Jacobian dr/dx has about (3 n_collocation) x (n_u + n_p) ~ 10^11
entries and is never formed. ``Objective.jvp`` (J v, forward mode) and
``Objective.vjp`` (J^T w, reverse mode) give its action; here:

``jacobian_columns``  explicit columns J e_k for chosen unknowns k (batched J v),
                      e.g. a small problem or a check
``dense_jacobian``    all columns, for small problems only
``diag_JTJ``          an estimate of diag(J^T J) by random probing (Hutchinson:
                      E[z * J^T J z] for Rademacher z), the column scaling and
                      Jacobi preconditioner of the optimizer
``normal_matvec``     v -> M J^T J M v + lam D v on the free unknowns (mask M), the
                      operator of the damped Gauss-Newton (Levenberg-Marquardt) step
"""

import jax
import jax.numpy as jnp
import numpy as np


def jacobian_columns(obj, x, cols, batch=16):
    """J e_k for the unknowns ``cols``, shape (n_residual, len(cols))."""
    cols = np.asarray(cols)
    out = []
    for i in range(0, len(cols), batch):
        c = cols[i : i + batch]
        E = jnp.zeros((len(c), obj.n)).at[jnp.arange(len(c)), c].set(1.0)
        out.append(jax.vmap(lambda v: obj.jvp(x, v))(E))
    return jnp.concatenate(out).T


def dense_jacobian(obj, x, batch=16):
    """The whole Jacobian (n_residual, n): small problems only."""
    return jacobian_columns(obj, x, np.arange(obj.n), batch)


def diag_JTJ(obj, x, mask=None, n_probes=8, seed=0):
    """Hutchinson estimate of diag(J^T J) on the free unknowns (``mask``)."""
    mask = jnp.ones(obj.n) if mask is None else jnp.asarray(mask, float)
    key = jax.random.PRNGKey(seed)
    d = jnp.zeros(obj.n)
    for _ in range(n_probes):
        key, sub = jax.random.split(key)
        z = jax.random.rademacher(sub, (obj.n,), dtype=jnp.float64) * mask
        d = d + z * obj.vjp(x, obj.jvp(x, z))
    return jnp.abs(d) / n_probes * mask


def normal_matvec(obj, x, mask, lam, D):
    """v -> M J^T J M v + lam D v (Levenberg-Marquardt normal operator)."""
    def mv(v):
        v = v * mask
        return mask * obj.vjp(x, obj.jvp(x, v)) + lam * D * v
    return mv
