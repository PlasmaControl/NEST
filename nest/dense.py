"""Dense form of the force-balance problem, for small (e.g. axisymmetric) cases.

On a fixed collocation grid and box, J and grad p are linear in x = (u, c_p) and
B_plasma is linear in u (the box sampling, the FFT solve and the interpolation to
the points are linear maps), so

    B_plasma(u) = G u + B_I        (B_I: the net-current part, I fixed)

with G a fixed matrix, (3 n_points) x n_u: one box solve per column, once.
``DenseFields`` is a ``Fields`` whose B_plasma is that matrix product: the
objectives, ``levenberg_marquardt`` and the block preconditioner run unchanged on
it, and each evaluation costs a matrix product instead of a box FFT. G is cached
on disk (``cache``), keyed by the equilibrium file, the bases, the collocation
grid, the box and the kernel.

``force_matched_u``: the potential coefficients whose J matches DESC's in the
force on the collocation points,

    min_u  |sw (J(u) - J_DESC) x B_DESC / norm|^2 + eps^2 |sw (J(u) - J_DESC) / rms J_DESC|^2,

sw = sqrt(dV / V), i.e. the force objective's own weights; eps = inf drops the first
term (J fitted to J_DESC alone). With B(x0) = B_DESC
(``optimize_qa.py --bext desc``) and the pressure fit exact, F_NEST(x0) - F_DESC =
(J(u) - J_DESC) x B_DESC on the points, so the first term is exactly the
difference of NEST's starting force residual from DESC's; the second (eps small)
fixes the components of J along B, which F does not see.
"""

import hashlib
import inspect
import os
import time

import jax
import jax.numpy as jnp
import numpy as np

import fit_cache
from flux_interp import TensorCurrent
from objective import Fields, _cartesian

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def _key(fields, current):
    """Hash of what G depends on."""
    f, box = fields.force, fields.force.box
    b = current.basis
    import magnetic_field
    import objective
    src = "".join(inspect.getsource(m) for m in (objective, magnetic_field))
    if getattr(fields, "cyl", False):  # the annulus: its axes, planes and the solver's code
        import annulus_freespace, annulus_poisson, chebyshev, cylindrical_bs, cylindrical_grid
        import single_layer, toroidal_kernel
        src += "".join(inspect.getsource(m) for m in (
            annulus_freespace, annulus_poisson, chebyshev, cylindrical_bs, cylindrical_grid,
            single_layer, toroidal_kernel))
        import single_layer_fast
        src += inspect.getsource(single_layer_fast)
        geo = tuple(round(float(v), 9) for v in (box.r_axis.a, box.r_axis.b, box.z_axis.a,
                                                   box.z_axis.b))
        src += f"S {getattr(getattr(fields, 'solver', None), 'S', np.zeros(0)).dtype}"  # its precision
        if getattr(fields, "node_rtz", None) is not None:  # J at the nodes from the basis (node_current)
            import node_current
            src += "direct J at the nodes" + inspect.getsource(node_current)
        parts = (fit_cache._file_hash(f.inp.path), (b.L, b.M, b.N, current.edge_power, b.sym),
                 f.tgrid.shape, tuple(np.round(np.asarray(f.tgrid.rho), 14)), box.shape, geo,
                 box.NFP, fields.kernel, str(fields.dtype), hashlib.sha256(src.encode()).hexdigest())
        return hashlib.sha256(repr(parts).encode()).hexdigest()[:24]
    parts = (fit_cache._file_hash(f.inp.path), (b.L, b.M, b.N, current.edge_power, b.sym),
             f.tgrid.shape, tuple(np.round(np.asarray(f.tgrid.rho), 14)), box.shape,
             # rounded: the spacing a / frac comes from DESC's a, which differs in the last
             # bits between CPU and GPU
             tuple(round(float(a[0]), 9) for a in box.axes),
             tuple(round(float(a[1] - a[0]), 9) for a in box.axes),
             fields.kernel, str(fields.dtype), hashlib.sha256(src.encode()).hexdigest())
    return hashlib.sha256(repr(parts).encode()).hexdigest()[:24]


def _key_exact(fields, current):
    """Hash of what the exact G depends on: no box."""
    import axisym_field
    import boundary_potential
    f, b = fields.force, current.basis
    src = "".join(inspect.getsource(m) for m in (axisym_field, boundary_potential))
    parts = (fit_cache._file_hash(f.inp.path), (b.L, b.M, b.N, current.edge_power, b.sym, float(current.I)),
             f.tgrid.shape, tuple(np.round(np.asarray(f.tgrid.rho), 14)),
             hashlib.sha256(np.round(np.asarray(fields.phi), 14).tobytes()).hexdigest(), str(fields.dtype),
             hashlib.sha256(src.encode()).hexdigest())
    return hashlib.sha256(repr(parts).encode()).hexdigest()[:24]


class DenseFields(Fields):
    """``Fields`` with B_plasma(u) = G u + B_I (see the module docstring).

    Built from a ``Fields`` (``from_fields``); its box-side caches are dropped
    once G is formed."""

    _dynamic = Fields._dynamic + ("G", "B_I")

    @classmethod
    def from_fields(cls, fields, current, batch=8, cache=True, verbose=True, save_every=300,
                    method="box"):
        """``method``: "box" (one solve of ``fields``' box per column) or "exact" (the
        exact axisymmetric field, axisym_field.plasma_field_matrix: no box; N = 0 only)."""
        obj = object.__new__(cls)
        for k, v in vars(fields).items():
            if k != "_aux":
                object.__setattr__(obj, k, v)
        where = os.environ.get("NEST_CACHE", os.path.join(ROOT, ".cache"))
        path = None
        if cache and where != "off" and getattr(fields.force.inp, "path", None):
            name = f"DenseG_{_key(fields, current)}" if method == "box" else f"DenseGx_{_key_exact(fields, current)}"
            path = os.path.join(where, name + ".npz")
        n_u, t0 = fields.n_u, time.time()
        if method not in ("box", "exact"):
            raise ValueError(f"method {method!r}")
        if path and os.path.exists(path):
            with np.load(path) as d:
                G, B_I = d["G"], d["B_I"]
            how = "from the cache"
        elif method == "exact":
            from axisym_field import plasma_field, plasma_field_matrix
            from cartesian_grid import cylindrical_to_cartesian
            rtz = np.asarray(fields.force.rtz)
            Gc = plasma_field_matrix(current, rtz, verbose=verbose)  # (n_u, n, 3), (R, phi, Z)
            c, s_ = np.cos(np.asarray(fields.phi)), np.sin(np.asarray(fields.phi))
            BR = Gc[..., 0].copy()
            Gc[..., 0] = BR * c - Gc[..., 1] * s_
            Gc[..., 1] = BR * s_ + Gc[..., 1] * c
            del BR
            G = Gc.reshape(n_u, -1).astype(np.dtype(fields.dtype), copy=False)
            B_I = np.asarray(cylindrical_to_cartesian(plasma_field(current, rtz, u=np.zeros(n_u)),
                                                      np.asarray(fields.phi)))
            if path:
                os.makedirs(where, exist_ok=True)
                tmp = path + f".{os.getpid()}.tmp.npz"
                np.savez(tmp, G=G, B_I=B_I)
                os.replace(tmp, path)
            # against plasma_field itself (same quadrature, u at once): a random u, 256 points
            rng = np.random.default_rng(0)
            v, sub = rng.standard_normal(n_u), rng.choice(len(rtz), min(256, len(rtz)), replace=False)
            ref = np.asarray(cylindrical_to_cartesian(plasma_field(current, rtz[sub], u=v),
                                                      np.asarray(fields.phi)[sub]))
            got = (v @ G).reshape(-1, 3)[sub] + B_I[sub]
            how = (f"exact axisymmetric field (against plasma_field, random u: "
                   f"{np.abs(got - ref).max() / np.abs(ref).max():.1e})")
        else:
            solve = jax.jit(jax.vmap(lambda e, fl: fl.B_plasma(e, 0.0), in_axes=(0, None)))
            G = np.empty((n_u, len(fields.phi) * 3), dtype=np.dtype(fields.dtype))
            # columns done so far are saved every ``save_every`` s (``.partial.npz``), so a
            # job that runs out of time leaves them for the next one
            part, start = (path + ".partial.npz" if path else None), 0
            if part and os.path.exists(part):
                with np.load(part) as d:
                    start = int(d["done"])
                    G[:start] = d["G"]
                if verbose:
                    print(f"  dense B_plasma: resuming at column {start} of {n_u}", flush=True)
            eye, t_save = np.eye(n_u), time.time()
            for i in range(start, n_u, batch):
                E = np.zeros((batch, n_u))
                E[: min(batch, n_u - i)] = eye[i : i + batch]  # fixed batch: one compile
                cols = np.asarray(solve(jnp.asarray(E), fields)).reshape(batch, -1)
                G[i : i + batch] = cols[: min(batch, n_u - i)]
                done = min(i + batch, n_u)
                if verbose and i == start + 4 * batch:
                    print(f"  dense B_plasma: {(time.time() - t0) / (done - start) * (n_u - start) / 60:.1f} "
                          f"min estimated for {n_u - start} columns", flush=True)
                if part and time.time() - t_save > save_every and done < n_u:
                    os.makedirs(where, exist_ok=True)
                    tmp = part + f".{os.getpid()}.tmp.npz"
                    np.savez(tmp, G=G[:done], done=done)
                    os.replace(tmp, part)
                    t_save = time.time()
            B_I = np.asarray(fields.B_plasma(jnp.zeros(n_u), fields.I))
            if path:
                os.makedirs(where, exist_ok=True)
                tmp = path + f".{os.getpid()}.tmp.npz"
                np.savez(tmp, G=G, B_I=B_I)
                os.replace(tmp, path)
                if part and os.path.exists(part):
                    os.remove(part)
            how = f"{n_u - start} box solves"
        obj.G, obj.B_I = jnp.asarray(G), jnp.asarray(B_I)
        # the box side is no longer used
        for k in ("J_flux", "to_box", "src", "K_hat", "to_points", "box_phi", "solver"):
            object.__setattr__(obj, k, None)
        if verbose:
            print(f"  dense B_plasma: G {G.shape[1]:,} x {n_u} ({G.nbytes / 1e9:.2f} GB, "
                  f"{G.dtype}), {how}, {time.time() - t0:.0f} s", flush=True)
        return obj

    def B_plasma(self, u, I=None):
        I = self.I if I is None else I
        B = (u.astype(self.G.dtype) @ self.G).reshape(-1, 3)
        return B + (I / self.I) * self.B_I


def j_target(run, fields, x0, inp=None):
    """The w_J term's target J at ``fields``' collocation points (Cartesian), from a run's
    arguments: J(x0) with j_target "start", J of another run's start with j_target_from
    (that run's own basis: a common target for runs with different bases; exact in any
    basis that contains it), else None (J_DESC)."""
    src = run.get("j_target_from")
    if src:
        import json

        from boundary_potential import BoundaryPotential
        from cartesian_grid import cylindrical_to_cartesian
        from desc_input import DESCInput
        sa = json.load(open(src + "_iter.json"))["args"]
        inp = inp or DESCInput.load(sa["eq"], field=sa["coils"])
        ep = sa.get("edge_power", 2)
        cs = fit_cache.fit(BoundaryPotential, inp, factors=tuple(sa["factors"]),
                           **({"edge_power": ep} if ep != 2 else {}))
        us = np.load(src + "_iter.npz")["x0"][: cs.n_free]
        rtz = np.asarray(fields.force.rtz)
        Jc = jax.jit(cs.with_u(us).point_evaluator(inp.eq))(*(jnp.asarray(rtz[:, i]) for i in range(3)))
        return jnp.asarray(cylindrical_to_cartesian(np.asarray(Jc), np.asarray(fields.phi)))
    if run.get("j_target") == "start":
        return fields.compute(x0)["J"]
    return None


def J_columns(force, current):
    """J at the collocation points of ``force`` (a ``ForceResidual``; Cartesian
    components) for each unit u: (n_u, n_points, 3), and the net-current part
    J_I (n_points, 3). No box needed."""
    J_col = TensorCurrent(current.S, current._RZ, force.tgrid, force.geo)
    phi = jnp.asarray(force.phi)
    lin = jax.jit(jax.vmap(lambda e, Jc: _cartesian(Jc(e, 0.0).reshape(-1, 3), phi),
                           in_axes=(0, None)))
    n_u = current.n_free
    A = np.concatenate([np.asarray(lin(jnp.asarray(np.eye(n_u)[i : i + 64]), J_col))
                        for i in range(0, n_u, 64)])
    J_I = np.asarray(_cartesian(J_col(jnp.zeros(n_u), current.I).reshape(-1, 3), phi))
    return A, J_I


def force_matched_u(force, current, eps=1e-2, desc=None, cols=None):
    """u minimizing the force difference from DESC on the points of ``force``
    (module docstring); ``desc``: DESC's fields there (``force.desc_fields()``),
    ``cols``: ``J_columns(force, current)``, to reuse.

    Returns u and, for the given fit (``current.u``) and the force-matched one:
    <|(J - J_DESC) x B_DESC|>_vol / norm (the difference F_NEST - F_DESC at x0)
    and rms |J - J_DESC| / rms J_DESC."""
    A, J_I = J_columns(force, current) if cols is None else cols
    d = force.desc_fields() if desc is None else desc
    J_D, B_D = np.asarray(d["J"]), np.asarray(d["B"])
    w = force.dV / force.dV.sum()
    sw = np.sqrt(w)[:, None]
    norm = force.norm
    rmsJ = float(np.sqrt(np.sum(w * np.sum(J_D**2, 1))))
    n_u = A.shape[0]
    # rows: the force difference, then eps x the current difference (eps = inf: the
    # current alone, a plain fit of J to J_DESC on the points)
    AJ = (sw * A).reshape(n_u, -1).T / rmsJ
    yJ = (sw * (J_D - J_I)).ravel() / rmsJ
    if np.isinf(eps):
        u = np.linalg.lstsq(AJ, yJ, rcond=None)[0]
    else:
        AF = (sw * np.cross(A, B_D[None])).reshape(n_u, -1).T / norm
        yF = (sw * np.cross(J_D - J_I, B_D)).ravel() / norm
        u = np.linalg.lstsq(np.vstack([AF, eps * AJ]), np.concatenate([yF, eps * yJ]), rcond=None)[0]

    def score(v):
        dJ = np.einsum("i,ipc->pc", v, A) + J_I - J_D
        return {"F - F_DESC": float(np.sum(w * np.linalg.norm(np.cross(dJ, B_D), axis=1))) / norm,
                "J - J_DESC": float(np.sqrt(np.sum(w * np.sum(dJ**2, 1)))) / rmsJ}

    return u, {"given fit": score(np.asarray(current.u)), "force fit": score(u)}
