"""Chunking and batching: bounded memory for evaluations over many points.

Two patterns recur in NEST:

``host_chunks``   a jitted function applied to slices of NumPy arrays from a
                  Python loop, every slice padded to the same length so that the
                  function compiles once; results are gathered on the host. For
                  one-off evaluations (fits, diagnostics) whose output can leave
                  the GPU.

``map_chunks``    the same inside JAX: ``lax.map`` over padded chunks, each chunk
                  checkpointed so that reverse-mode differentiation keeps the
                  intermediates of one chunk at a time. For evaluations inside
                  a differentiated function (the residual and its Jacobian).

``batched_vmap``  ``jax.vmap`` over a leading axis in batches (e.g. Jacobian
                  columns from ``jvp``), to bound the memory of vectorized
                  forward-mode derivatives.
"""

import jax
import jax.numpy as jnp
import numpy as np


def _pad_rows(x, m):
    """Pad the leading axis of x to length m by repeating its last row."""
    n = x.shape[0]
    if m == n:
        return x
    reps = (m - n,) + (1,) * (x.ndim - 1)
    return jnp.concatenate([x, jnp.tile(x[-1:], reps)])


def host_chunks(f, *arrays, chunk=50000, progress=None):
    """f(*slices) over the leading axis of ``arrays``, ``chunk`` rows at a time.

    Every call gets exactly ``chunk`` rows (the last chunk padded by repeating
    its last row), so a jitted f compiles once; the outputs (arrays with the
    same leading axis) are concatenated as NumPy arrays. ``progress(k, n_chunks,
    done)`` is called after each chunk if given.
    """
    n = len(arrays[0])
    chunk = min(chunk, max(n, 1))
    out, n_chunks = [], -(-n // chunk)
    for k, i in enumerate(range(0, n, chunk)):
        m = min(chunk, n - i)
        idx = np.pad(np.arange(i, i + m), (0, chunk - m), mode="edge")
        res = f(*(jnp.asarray(np.asarray(a)[idx]) for a in arrays))
        out.append(jax.tree_util.tree_map(lambda r: np.asarray(r)[:m], res))
        if progress is not None:
            progress(k + 1, n_chunks, i + m)
    return jax.tree_util.tree_map(lambda *r: np.concatenate(r), *out)


def map_chunks(f, *arrays, chunk=16384, checkpoint=True, carry=None):
    """f(*slices) over the leading axis of JAX ``arrays``, inside JAX.

    ``lax.map`` over chunks of ``chunk`` rows (padded), each chunk checkpointed
    (``checkpoint=True``) to bound reverse-mode memory. f returns an array (or
    pytree of arrays) with a leading axis of length ``chunk``. With ``carry``,
    f(carry, *slices): the value is passed through the scan as its carry, which
    is how JAX can transpose a map that is linear in it (``jax.linear_transpose``
    fails on a scan closing over a linear value).
    """
    n = arrays[0].shape[0]
    chunk = min(chunk, n)
    m = -(-n // chunk) * chunk
    padded = [_pad_rows(a, m).reshape((m // chunk, chunk) + a.shape[1:]) for a in arrays]
    if carry is None:
        g = (lambda args: f(*args))
        g = jax.checkpoint(g) if checkpoint else g
        out = jax.lax.map(g, padded)
    else:
        g = (lambda c, args: (c, f(c, *args)))
        g = jax.checkpoint(g) if checkpoint else g
        out = jax.lax.scan(g, carry, padded)[1]
    return jax.tree_util.tree_map(lambda r: r.reshape((m,) + r.shape[2:])[:n], out)


def batched_vmap(f, xs, batch=64):
    """jax.vmap(f) over the leading axis of xs, ``batch`` rows at a time (Python loop)."""
    vf = jax.jit(jax.vmap(f))
    return jnp.concatenate([vf(xs[i : i + batch]) for i in range(0, xs.shape[0], batch)])
