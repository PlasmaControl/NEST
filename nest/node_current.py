"""J at the annulus nodes evaluated from the basis directly, instead of interpolated from the
flux grid.

``Fields`` (cylindrical path) puts J on the inside nodes by Lagrange interpolation from a
uniform flux grid (4L x 6M); that leaves ~1.5e-6 relative error in J at the nodes, which set
most of the B_plasma floor (~3e-9 of rms B at a/320, growing towards the LCFS:
gs_cyl_floor.py).  ``NodeFields`` evaluates J = curl T of the basis at every inside node's flux
coordinates (``boundary_potential._current``), in chunks of nodes (``lax.map``): exact, linear
in (u, I), differentiable.  The node coordinates are its dynamic arrays, so it passes through
jit and vmap like ``Fields`` (e.g. the dense G build).
"""

import jax
import jax.numpy as jnp
import numpy as np

from boundary_potential import _current
from objective import Fields


class NodeFields(Fields):
    """``Fields`` whose J on the annulus is the basis at the nodes (see the module docstring)."""

    _dynamic = Fields._dynamic + ("node_rtz", "node_src")

    @classmethod
    def from_fields(cls, fields, current, box, chunk=16384):
        if not getattr(fields, "cyl", False):
            raise ValueError("NodeFields: the cylindrical annulus only")
        obj = object.__new__(cls)
        for k, v in vars(fields).items():
            if k != "_aux":
                object.__setattr__(obj, k, v)
        rtz = np.asarray(box.rtz, dtype=float)
        n = len(rtz)
        n_c = -(-n // chunk)
        pad = np.concatenate([rtz, np.repeat(rtz[-1:], n_c * chunk - n, 0)])
        src = np.full(int(np.prod(box.shape)), n, np.int32)  # every inside node; the rest the zero row
        src[np.asarray(box.inside)] = np.arange(n)
        object.__setattr__(obj, "node_rtz", jnp.asarray(pad.reshape(n_c, chunk, 3)))
        object.__setattr__(obj, "node_src", jnp.asarray(src))
        object.__setattr__(obj, "n_nodes", n)
        object.__setattr__(obj, "node_S", current.S)
        object.__setattr__(obj, "node_RZ", current._RZ)
        return obj

    def J_box(self, u, I=None):
        I = self.I if I is None else I
        u = jnp.asarray(u)
        J = jax.lax.map(lambda p: _current(self.node_S, self.node_RZ, u, I, p[:, 0], p[:, 1], p[:, 2]),
                        self.node_rtz)
        J = J.reshape(-1, 3)[: self.n_nodes].astype(self.cyl_dtype)
        J = jnp.pad(J, ((0, 1), (0, 0)))  # the zero row
        return J[self.node_src].reshape(self.shape + (3,))
