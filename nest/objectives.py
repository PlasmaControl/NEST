"""Step 8: objectives built from the fields J, B, p, grad p of ``fields.Fields``.

An objective is a list of terms; term i is a residual r_i(x) at the collocation
points and a weight w_i, and the optimizer minimizes

    f(x) = sum_i w_i ||r_i(x)||^2.

Every term carries the volume weights sqrt(dV / V), so ||r_i||^2 is a volume average
over the plasma. Terms given here (targets default to DESC's fields at the points):

    ForceBalance      (J x B - grad p) / F_0,     F_0 = rms |grad(|B|^2)| / 2 mu0 of DESC
                      (the rms form of DESC's force normalization)
    FieldDistance     (B - B_target) / rms |B_target|
    CurrentDistance   (J - J_target) / rms |J_target|
    PressureDistance  (p - p_target) / rms p_target

e.g. the combined objective ||r_D||^2 + lambda^-1 ||r_F||^2:

    obj = ObjectiveFunction(F, [FieldDistance(F), ForceBalance(F, weight=1 / lam)])

A new term subclasses ``Term`` and defines ``residual(f, d)``: f the fields at the
points (dict of JAX arrays J, B, grad_p (n, 3), p (n,)), d the term's own arrays (each
with the points along the first axis), returning (n, k). Terms are pointwise, so the
rows of every term can be split into blocks of points (``blocks``, for the row-chunked
Gauss-Newton step).

Jacobian: forward-mode products (jvp) through the affine fields, in chunks of
columns (``jac_chunk``, as DESC's jac_chunk_size): with B = B_0 + D_B u there is no
Biot-Savart call, and the Jacobian of a term is exact.
"""

import functools

import jax
import jax.numpy as jnp
import numpy as np

from cartesian_grid import cylindrical_to_cartesian


class Term:
    """One residual block: weight, name, and arrays (``data``) along the points.

    Parameters
    ----------
    fields : Fields
    weight : float
    name : str
    """

    name = "term"

    def __init__(self, fields, weight=1.0, name=None):
        self.weight = float(weight)
        self.name = name or self.name
        self.V = float(np.sum(fields.dV))
        self._data = {"sw": jnp.asarray(np.sqrt(fields.dV / self.V))}

    def data(self):
        """The term's arrays, each with the points along the first axis."""
        return self._data

    def residual(self, f, d):
        raise NotImplementedError


def _desc_at_points(fields, names):
    """DESC quantities at the collocation points; vectors in Cartesian components."""
    d = fields.inp.compute(names, fields.grid)
    return {k: cylindrical_to_cartesian(v, fields.phi) if v.ndim == 2 and v.shape[1] == 3 else v
            for k, v in d.items()}


def _rms(fields, v):
    v = np.asarray(v).reshape(len(fields.dV), -1)
    return float(np.sqrt(np.sum(fields.dV[:, None] * v**2) / np.sum(fields.dV)))


class ForceBalance(Term):
    """(J x B - grad p) / F_0, F_0 = rms |grad(|B|^2)| / 2 mu0 of DESC (or ``norm``)."""

    name = "force"

    def __init__(self, fields, weight=1.0, norm=None, name=None):
        super().__init__(fields, weight, name)
        self.norm = norm or _rms(fields, _desc_at_points(fields, ["|grad(|B|^2)|/2mu0"])
                                 ["|grad(|B|^2)|/2mu0"])

    def residual(self, f, d):
        return d["sw"][:, None] * (jnp.cross(f["J"], f["B"]) - f["grad_p"]) / self.norm


class _Distance(Term):
    key, desc_key = None, None

    def __init__(self, fields, weight=1.0, target=None, norm=None, name=None):
        super().__init__(fields, weight, name)
        if target is None:
            target = _desc_at_points(fields, [self.desc_key])[self.desc_key]
        target = np.asarray(target, dtype=float).reshape(len(fields.dV), -1)
        self.norm = norm or _rms(fields, target)
        self._data["target"] = jnp.asarray(target)

    def residual(self, f, d):
        v = f[self.key].reshape(d["target"].shape)
        return d["sw"][:, None] * (v - d["target"]) / self.norm


class FieldDistance(_Distance):
    """(B - B_target) / rms |B_target|; target default DESC's B."""
    name, key, desc_key = "field", "B", "B"


class CurrentDistance(_Distance):
    """(J - J_target) / rms |J_target|; target default DESC's J."""
    name, key, desc_key = "current", "J", "J"


class PressureDistance(_Distance):
    """(p - p_target) / rms p_target; target default DESC's p."""
    name, key, desc_key = "pressure", "p", "p"


def _rows(tree, i0, i1):
    return jax.tree_util.tree_map(lambda a: a[i0:i1], tree)


class ObjectiveFunction:
    """sum_i w_i ||r_i(x)||^2 over the terms; residual and Jacobian in JAX.

    Parameters
    ----------
    fields : Fields
        With D_B built.
    terms : list of Term
    jac_chunk : int
        Columns per forward-mode batch of the Jacobian (memory).
    """

    def __init__(self, fields, terms, jac_chunk=64):
        if fields.D_B is None:
            raise RuntimeError("the objective needs D_B (Fields(build_D_B=True))")
        self.fields, self.terms, self.jac_chunk = fields, list(terms), int(jac_chunk)
        self.n_points, self.n = len(fields.dV), fields.n
        self._D = {"fields": fields.data(), "terms": [t.data() for t in self.terms]}
        self._r = jax.jit(self._residual)
        self._jvp = jax.jit(jax.vmap(lambda x, v, D, w: jax.jvp(
            lambda y: self._residual(y, D, w), (x,), (v,))[1], in_axes=(None, 0, None, None)))

    @property
    def weights(self):
        return jnp.asarray([t.weight for t in self.terms])

    def set_weights(self, weights):
        for t, w in zip(self.terms, weights):
            t.weight = float(w)

    def _residual(self, x, D, w):
        f = self.fields.evaluate(x, D["fields"])
        return jnp.concatenate([jnp.sqrt(w[i]) * t.residual(f, D["terms"][i]).ravel()
                                for i, t in enumerate(self.terms)])

    def residual(self, x):
        """Weighted residual, all terms stacked: f(x) = ||residual(x)||^2."""
        return self._r(jnp.asarray(x), self._D, self.weights)

    def compute_scalar(self, x):
        return float(jnp.sum(self.residual(x) ** 2))

    def _jacobian(self, x, D, n_rows):
        x = jnp.asarray(x)
        cols = []
        for j0 in range(0, self.n, self.jac_chunk):
            j1 = min(self.n, j0 + self.jac_chunk)
            E = jnp.zeros((j1 - j0, self.n)).at[jnp.arange(j1 - j0), jnp.arange(j0, j1)].set(1.0)
            cols.append(self._jvp(x, E, D, self.weights).T)
        return jnp.concatenate(cols, axis=1)

    def jacobian(self, x):
        """Dense Jacobian of ``residual``, (m, n)."""
        return self._jacobian(x, self._D, None)

    def blocks(self, x, points_per_block):
        """(r, J) of the rows of blocks of points: every term at points i0..i1 together.

        The row order differs from ``residual`` (by block, then by term), which is
        immaterial for J^T J, J^T r and |r|."""
        for i0 in range(0, self.n_points, points_per_block):
            i1 = min(self.n_points, i0 + points_per_block)
            D = _rows(self._D, i0, i1)
            yield self._r(jnp.asarray(x), D, self.weights), self._jacobian(x, D, None)

    def summary(self, x):
        """Unweighted ||r_i|| of every term, and the weighted total f."""
        f = self.fields.evaluate(jnp.asarray(x), self._D["fields"])
        out = {t.name: float(jnp.linalg.norm(t.residual(f, self._D["terms"][i])))
               for i, t in enumerate(self.terms)}
        out["f"] = self.compute_scalar(x)
        return out
