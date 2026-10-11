"""Step 8 of NEST: objectives built from the fields J, B, p, grad p of ``fields.Fields``.

An objective is a list of terms; term i is a residual r_i(x) at the collocation
points and a weight w_i, and the optimizer minimizes

    f(x) = sum_i w_i ||r_i(x)||^2.

Every term carries the volume weights sqrt(dV / V), so ||r_i||^2 is a volume average
over the plasma. Terms given here (targets default to DESC's fields at the points,
``ForceResidual.desc_fields``):

    ForceBalance      (J x B - grad p) / F_0,  F_0 = <|grad(|B|^2)| / 2 mu0>_vol of DESC
                      (``ForceResidual.norm``), so ||r|| = rms |F| / F_0
    FieldDistance     (B - B_target) / rms |B_target|
    CurrentDistance   (J - J_target) / rms |J_target|
    PressureDistance  (p - p_target) / rms p_target

e.g. the combined objective ||r_D||^2 + lambda^-1 ||r_F||^2:

    obj = ObjectiveFunction(F, [FieldDistance(F), ForceBalance(F, weight=1 / lam)])

A new term subclasses ``Term`` and defines ``residual(f, d)``: f the fields at the
points (dict of JAX arrays J, B, grad_p (n, 3), p (n,)), d the term's own arrays (each
with the points along the first axis), returning (n, k), and ``uses``, the fields it
depends on. Terms are pointwise in the points.

Jacobian. The fields are affine in x with the constant D's of ``Fields``, and a term
is pointwise, so its Jacobian is

    d r_p / d x = sum_f (d r_p / d f_p) D_f[p]        f in J, B, grad_p, p

with d r_p / d f_p a small (k x 3) matrix per point, from one jvp per field
component (all points at once). Contracting it with the D's costs O(m n) (m rows, n
unknowns), as fftq's structured force Jacobian (for ForceBalance it is D_J x B + J x
D_B - D_gp). The rows split into blocks of points (``blocks``), for the row-chunked
Gauss-Newton step: only one block of the Jacobian exists at a time.
"""

import jax
import jax.numpy as jnp
import numpy as np

# the fields of ``Fields.state``: name, dimension, its D in ``Fields.data()``, and
# whether it depends on the current (u) or the pressure (x_p) part of x
FIELDS = (("J", 3, "D_J", "u"), ("B", 3, "D_B", "u"), ("grad_p", 3, "D_gp", "p"),
          ("p", 1, "D_p", "p"))


class Term:
    """One residual block: weight, name, the fields it uses, and arrays (``data``).

    Parameters
    ----------
    fields : Fields
    weight : float
    name : str
    """

    name = "term"
    uses = ("J", "B", "grad_p", "p")

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


class ForceBalance(Term):
    """(J x B - grad p) / F_0, F_0 = DESC's <|grad(|B|^2)| / 2 mu0>_vol (or ``norm``)."""

    name, uses = "force", ("J", "B", "grad_p")

    def __init__(self, fields, weight=1.0, norm=None, name=None):
        super().__init__(fields, weight, name)
        self.norm = norm or fields.norm

    def residual(self, f, d):
        return d["sw"][:, None] * (jnp.cross(f["J"], f["B"]) - f["grad_p"]) / self.norm


class _Distance(Term):
    desc_key = None

    def __init__(self, fields, weight=1.0, target=None, norm=None, name=None):
        super().__init__(fields, weight, name)
        if target is None:
            target = fields.desc_fields([self.desc_key])[self.desc_key]
        target = np.asarray(target, dtype=float).reshape(len(fields.dV), -1)
        self.norm = norm or float(np.sqrt(fields.average(np.sum(target**2, axis=1))))
        self._data["target"] = jnp.asarray(target)

    def residual(self, f, d):
        v = f[self.uses[0]].reshape(d["target"].shape)
        return d["sw"][:, None] * (v - d["target"]) / self.norm


class FieldDistance(_Distance):
    """(B - B_target) / rms |B_target|; target default DESC's B."""
    name, uses, desc_key = "field", ("B",), "B"


class CurrentDistance(_Distance):
    """(J - J_target) / rms |J_target|; target default DESC's J."""
    name, uses, desc_key = "current", ("J",), "J"


class PressureDistance(_Distance):
    """(p - p_target) / rms p_target; target default DESC's p."""
    name, uses, desc_key = "pressure", ("p",), "p"


def _rows(tree, i0, i1):
    return jax.tree_util.tree_map(lambda a: a[i0:i1], tree)


class ObjectiveFunction:
    """sum_i w_i ||r_i(x)||^2 over the terms; residual and Jacobian in JAX.

    Parameters
    ----------
    fields : Fields
        With D_B built.
    terms : list of Term
    """

    def __init__(self, fields, terms):
        if fields.D_B is None:
            raise RuntimeError("the objective needs D_B (Fields(build_D_B=True))")
        self.fields, self.terms = fields, list(terms)
        self.n_points, self.n_u, self.n = len(fields.dV), fields.n_u, fields.n
        self._D = {"fields": fields.data(), "terms": [t.data() for t in self.terms]}
        self._r = jax.jit(self._residual)
        self._J = jax.jit(self._jacobian)

    @property
    def weights(self):
        return jnp.asarray([t.weight for t in self.terms])

    def set_weights(self, weights):
        for t, w in zip(self.terms, weights):
            t.weight = float(w)

    def _residual(self, x, D, w):
        f = self.fields.state(x, D["fields"])
        return jnp.concatenate([jnp.sqrt(w[i]) * t.residual(f, D["terms"][i]).ravel()
                                for i, t in enumerate(self.terms)])

    def _jacobian(self, x, D, w):
        """Rows of every term at the points of D, (m, n), term by term as in ``_residual``."""
        DF = D["fields"]
        f = self.fields.state(x, DF)
        n_pts = f["J"].shape[0]
        blocks = []
        for i, t in enumerate(self.terms):
            r = lambda g, t=t, d=D["terms"][i]: t.residual(g, d)
            k = r(f).shape[1]
            Ju = jnp.zeros((n_pts, k, self.n_u))
            Jp = jnp.zeros((n_pts, k, self.n - self.n_u))
            for name, dim, key, part in FIELDS:
                if name not in t.uses:
                    continue
                Df = DF[key].reshape(n_pts, dim, -1)  # (n, dim, columns)
                for c in range(dim):
                    tan = {g: jnp.zeros_like(v) for g, v in f.items()}
                    tan[name] = tan[name].reshape(n_pts, dim).at[:, c].set(1.0).reshape(f[name].shape)
                    L = jax.jvp(r, (f,), (tan,))[1]  # (n, k): d r_p / d f_p[c], all points
                    if part == "u":
                        Ju = Ju + L[:, :, None] * Df[:, c][:, None, :]
                    else:
                        Jp = Jp + L[:, :, None] * Df[:, c][:, None, :]
            blocks.append(jnp.sqrt(w[i]) * jnp.concatenate([Ju, Jp], axis=2).reshape(-1, self.n))
        return jnp.concatenate(blocks)

    def residual(self, x):
        """Weighted residual, all terms stacked: f(x) = ||residual(x)||^2."""
        return self._r(jnp.asarray(x), self._D, self.weights)

    def compute_scalar(self, x):
        return float(jnp.sum(self.residual(x) ** 2))

    def jacobian(self, x):
        """Dense Jacobian of ``residual``, (m, n)."""
        return self._J(jnp.asarray(x), self._D, self.weights)

    def blocks(self, x, points_per_block):
        """(r, J) of the rows of blocks of points: every term at points i0..i1 together.

        The row order differs from ``residual`` (by block, then by term), which is
        immaterial for J^T J, J^T r and |r|."""
        x = jnp.asarray(x)
        for i0 in range(0, self.n_points, points_per_block):
            D = _rows(self._D, i0, min(self.n_points, i0 + points_per_block))
            yield self._r(x, D, self.weights), self._J(x, D, self.weights)

    def summary(self, x):
        """Unweighted ||r_i|| of every term, the weighted total f, and DESC's force
        measure <|F|>_vol / F_0 (``ForceResidual.metrics``)."""
        x = jnp.asarray(x)
        f = self.fields.state(x, self._D["fields"])
        out = {t.name: float(jnp.linalg.norm(t.residual(f, self._D["terms"][i])))
               for i, t in enumerate(self.terms)}
        out["f"] = self.compute_scalar(x)
        out["<|F|>/F_0"] = self.fields.report(x, self._D["fields"])["mean"]
        return out
