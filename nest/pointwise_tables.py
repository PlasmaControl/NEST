"""Column tables of the pointwise problem (``pointwise.DenseModel``) from ``objective.Fields``.

At the collocation points (Cartesian components), per unit coefficient:

    A (n_u, n, 3), J_I     J of each potential coefficient (no field solve), net-current part
    G (n_u, n, 3), B0      B_plasma of each coefficient (one annulus solve each), and
                           B0 = B_ext + B_plasma(0, I)
    gP (n_p, n, 3), P      grad p and p of each pressure coefficient

and the weights, normalizations, the start x0 = (u of the fit of DESC's J, c of DESC's p),
J0 = J(u0), DESC's fields, and per point rho, R, phi and the direction of grad rho of
DESC's surfaces (for the diagnostics). Saved as one .npz (``save``), loaded by the dense
solver in another environment.
"""

import time

import jax
import jax.numpy as jnp
import numpy as np

from dense import J_columns
from objective import _cartesian


def g_columns(fields, batch=16, verbose=True):
    """B_plasma at the points per unit u: (n_u, n, 3), by batched annulus solves."""
    n_u, t0 = fields.n_u, time.time()
    solve = jax.jit(jax.vmap(lambda e, fl: fl.B_plasma(e, 0.0), in_axes=(0, None)))
    out = np.empty((n_u, len(fields.phi), 3))
    for i in range(0, n_u, batch):
        E = np.zeros((batch, n_u))
        k = min(batch, n_u - i)
        E[np.arange(k), i + np.arange(k)] = 1.0
        out[i : i + k] = np.asarray(solve(jnp.asarray(E), fields))[:k]
    if verbose:
        print(f"  G: {n_u} annulus solves, {time.time() - t0:.0f} s", flush=True)
    return out


def p_columns(fields, batch=64):
    """p and grad p (Cartesian) at the points per unit pressure coefficient."""
    phi = fields.phi

    def one(e, pc):
        p, gp = pc(e)
        return p.ravel(), _cartesian(gp.reshape(-1, 3), phi)

    f = jax.jit(jax.vmap(one, in_axes=(0, None)))
    n_p = fields.n_p
    P, gP = [], []
    for i in range(0, n_p, batch):
        E = np.zeros((batch, n_p))
        k = min(batch, n_p - i)
        E[np.arange(k), i + np.arange(k)] = 1.0
        p, gp = f(jnp.asarray(E), fields.p_col)
        P.append(np.asarray(p)[:k])
        gP.append(np.asarray(gp)[:k])
    return np.concatenate(P), np.concatenate(gP)


def build(fields, cur, pre, with_G=True, g_batch=16):
    """The tables (dict of NumPy arrays and scalars)."""
    force = fields.force
    A, J_I = J_columns(force, cur)
    P, gP = p_columns(fields)
    w = np.asarray(fields.sw) ** 2
    u0, c0 = np.asarray(cur.u), np.asarray(pre.c)
    J0 = np.einsum("k,knc->nc", u0, A) + J_I
    p0 = c0 @ P
    B_I = np.asarray(fields.B_plasma(jnp.zeros(fields.n_u), fields.I))
    T = {"A": A, "J_I": J_I, "gP": gP, "P": P, "w": w, "norm": float(fields.norm),
         "u0": u0, "c0": c0, "J0": J0, "rmsJ0": float(np.sqrt(np.sum(w * np.sum(J0**2, 1)))),
         "rmsp0": float(np.sqrt(np.sum(w * p0**2))), "B_ext": np.asarray(fields.B_ext),
         "B_I": B_I, "B0": np.asarray(fields.B_ext) + B_I}
    if with_G:
        T["G"] = g_columns(fields, batch=g_batch)
    # geometry per point
    g = force.geo
    R, ph = np.asarray(g.R).ravel(), np.asarray(fields.phi)
    Xt, Xz = (np.asarray(v).reshape(-1, 2) for v in (g.Xt, g.Xz))
    gr = np.stack([-R * Xt[:, 1], Xt[:, 1] * Xz[:, 0] - Xt[:, 0] * Xz[:, 1], R * Xt[:, 0]], 1)
    T["grad_rho"] = np.asarray(_cartesian(gr, ph))
    T["rtz"], T["R"], T["phi"] = np.asarray(force.rtz), R, ph
    T["Z"] = np.asarray(g.Z).ravel() if hasattr(g, "Z") else np.zeros_like(R)
    T["tgrid_shape"] = np.array(force.tgrid.shape)
    d = {k: np.asarray(v) for k, v in fields.desc.items()}
    T["J_desc"], T["B_desc"], T["p_desc"] = d["J"], d["B"], d["p"].ravel()
    T["grad_p_desc"] = d["grad(p)"]
    return T


def save(path, T):
    np.savez(path, **T)


def load(path):
    with np.load(path) as f:
        return {k: (f[k].item() if f[k].ndim == 0 else f[k]) for k in f.files}
