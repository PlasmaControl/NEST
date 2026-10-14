"""Step 6 of NEST: B = B_ext + B_plasma and the force residual F = J x B - grad p.

F is evaluated on collocation points of its own, a Gauss-uniform tensor grid in
the flux coordinates of the fixed equilibrium (Gauss-Legendre in rho, uniform in
theta and zeta over one field period, or half of it with stellarator symmetry),
with the volume weights sqrt(g) w:

    J, grad p  evaluated at the points from their series
    B_ext      the external field (coils) of the ``DESCInput`` at the points
    B_plasma   free-space Biot-Savart of J sampled on the periodic box G3 (FFT,
               ``magnetic_field.b_plasma_free``), interpolated to the points
               spectrally with a type-2 non-uniform FFT (``magnetic_field.interpolate``)
    F          J x (B_ext + B_plasma) - grad p

The box carries only the sampling of J for the FFT; the residual never uses the
box points, so it has no staircase and is an exact quadrature over the volume.
It is measured like DESC's force error,

    f = <|F|>_vol / <|grad |B|^2 / 2 mu0|>_vol,

with DESC's normalization (computed by DESC on its QuadratureGrid, as is DESC's
own force error, the value quoted for the equilibria in desc_input).

``decompose`` splits F against DESC's fields at the same points:

    F = F_desc + (J - J_desc) x B_desc + J x (B - B_desc) - (grad p - grad p_desc)

DESC's own force imbalance, the error of the fitted current, of the field (coils
plus B_plasma against DESC's B) and of the pressure; the identity is exact.
"""

import functools

import numpy as np
from desc.grid import QuadratureGrid

from cartesian_grid import cylindrical_to_cartesian, evaluate_chunked
from flux_interp import TensorGeometry, TensorGrid
from fourier import fft_J
from magnetic_field import b_plasma_free, biot_savart_kernel_hat, interpolate
from separable import Separable


class ForceResidual:
    """F = J x B - grad p on a flux-coordinate collocation grid.

    Parameters
    ----------
    inp : DESCInput
        The equilibrium and its external field (needed for B_ext).
    box : CartesianGrid or None
        The periodic box on which J is sampled for B_plasma (``locate_plasma`` done);
        None when B_plasma is not needed (``B_plasma`` and ``evaluate`` then fail).
    resolution : (L, M, N), optional
        Of the collocation tensor grid: L // 2 + 1 Gauss-Legendre points in rho,
        2 M + 1 uniform in theta and 2 N + 1 uniform in zeta per field period
        (default (2 L, 2 M, 2 N) of the equilibrium, as DESC's QuadratureGrid).
    sym : bool, optional
        Stellarator symmetry (default: the equilibrium's). F and every term of
        the objective are then symmetric, |F|^2 is even, and zeta only covers
        half a field period, (0, pi / NFP), with N + 1 midpoints: half the points.
    eval_chunk : int
        Points per call of the series evaluators.

    The points (``tgrid``, a ``TensorGrid``; per-point arrays in its C order),
    their positions, volume weights, B_ext, the Biot-Savart kernel and DESC's
    normalization do not depend on J or p: computed once, here. The
    normalization <|grad |B|^2 / 2 mu0|>_vol and DESC's own <|F|>_vol are
    DESC's, on its QuadratureGrid(2 L, 2 M, 2 N).
    """

    def __init__(self, inp, box, resolution=None, sym=None, eval_chunk=50000, rho_slice=None):
        if box is not None and box.inside is None:
            raise RuntimeError("locate the plasma on the box first (box.locate_plasma)")
        eq = inp.eq
        self.inp, self.box, self.eval_chunk = inp, box, eval_chunk
        self.sym = bool(eq.sym) if sym is None else sym
        L, M, N = resolution or (2 * eq.L, 2 * eq.M, 2 * eq.N)
        x, w_r = np.polynomial.legendre.leggauss(L // 2 + 1)
        rho, w_r = (x + 1) / 2, w_r / 2
        if rho_slice is not None:  # a part of the radii (e.g. to evaluate in pieces)
            rho, w_r = rho[rho_slice], w_r[rho_slice]
        theta = 2 * np.pi * np.arange(2 * M + 1) / (2 * M + 1)
        if self.sym:
            n_z = N + 1
            zeta = (np.arange(n_z) + 0.5) * np.pi / (eq.NFP * n_z)
            d_z = np.pi / (eq.NFP * n_z)
        else:
            n_z = 2 * N + 1
            zeta = 2 * np.pi * np.arange(n_z) / (eq.NFP * n_z)
            d_z = 2 * np.pi / (eq.NFP * n_z)
        self.tgrid = TensorGrid(rho, theta, zeta, eq.NFP)
        self.rtz = self.tgrid.nodes()
        # positions and sqrt(g) from the tensor evaluation of DESC's R, Z series
        # (phi = zeta), checked against DESC's pointwise compute on a sample
        RZ = Separable([eq.R_basis, eq.Z_basis], [eq.R_lmn, eq.Z_lmn], eq.NFP)
        self.geo = TensorGeometry(RZ, self.tgrid)
        R, Z = np.asarray(self.geo.R).ravel(), np.asarray(self.geo.Z).ravel()
        sqrt_g = np.asarray(self.geo.sqrt_g).ravel()
        self.phi = self.rtz[:, 2].copy()
        sample = np.random.default_rng(0).choice(len(self.rtz), min(2000, len(self.rtz)),
                                                 replace=False)
        d = inp.compute_points(["R", "phi", "Z", "sqrt(g)"], self.rtz[sample])
        err = max(np.abs(d["R"] - R[sample]).max() / np.abs(R).max(),
                  np.abs(d["Z"] - Z[sample]).max() / np.abs(R).max(),
                  np.abs(d["phi"] - self.phi[sample]).max(),
                  np.abs(d["sqrt(g)"] - sqrt_g[sample]).max() / np.abs(sqrt_g).max())
        if err > 1e-9:
            raise RuntimeError(f"tensor geometry differs from DESC's by {err:.1e}")
        self.x = np.column_stack([R * np.cos(self.phi), R * np.sin(self.phi), Z])
        w = (w_r[:, None, None] * (2 * np.pi / len(theta)) * d_z * np.ones(self.tgrid.shape)).ravel()
        self.dV = np.abs(sqrt_g) * w  # volume weights (of the covered part)
        q = QuadratureGrid(L=2 * eq.L, M=2 * eq.M, N=2 * eq.N, NFP=eq.NFP)
        dq = inp.compute(["<|F|>_vol", "<|grad(|B|^2)|/2mu0>_vol"], q)
        self.norm = float(np.ravel(dq["<|grad(|B|^2)|/2mu0>_vol"])[0])
        self.F_desc_vol = float(np.ravel(dq["<|F|>_vol"])[0])  # DESC's own <|F|>
        self.K_hat = None if box is None else biot_savart_kernel_hat(box.axes)

    @functools.cached_property
    def B_ext(self):
        """The external field at the points (on first use: a coil or surface-current
        Biot-Savart sum)."""
        return self.inp.B_ext(self.x)

    def _series(self, evaluator, rtz):
        return evaluate_chunked(evaluator, rtz, self.eval_chunk)

    def B_plasma(self, current):
        """B_plasma at the collocation points: J on the box -> FFT -> interpolation."""
        J_box = cylindrical_to_cartesian(self._series(current.point_evaluator(self.inp.eq),
                                                      self.box.rtz), self.box.phi_inside)
        J_hat = fft_J(self.box.scatter(J_box, np.float32))
        B_box = np.asarray(b_plasma_free(J_hat, self.box.axes, self.K_hat))  # to the host
        del J_box, J_hat  # free the GPU for the interpolation
        return interpolate(B_box, self.box.axes, self.x)

    def evaluate(self, current, pressure):
        """J, B_plasma, B, grad p and F at the collocation points, each (n, 3)."""
        eq = self.inp.eq
        J = cylindrical_to_cartesian(self._series(current.point_evaluator(eq), self.rtz), self.phi)
        pg = self._series(pressure.point_evaluator(eq), self.rtz)
        grad_p = cylindrical_to_cartesian(pg[:, 1:], self.phi)
        B_plasma = self.B_plasma(current)
        B = self.B_ext + B_plasma
        return {"J": J, "B_plasma": B_plasma, "B": B, "grad_p": grad_p, "p": pg[:, 0],
                "F": np.cross(J, B) - grad_p}

    def average(self, q, mask=None):
        """Volume average of q over the collocation points (or a subset)."""
        w = self.dV if mask is None else self.dV * mask
        return float(np.sum(q * w) / np.sum(w))

    def metrics(self, F, rho_bins=(0.0, 0.5, 0.8, 0.9, 0.95, 1.0)):
        """<|F|>_vol / norm over the plasma and in radial bins; rms and max too."""
        f = np.linalg.norm(F, axis=1) / self.norm
        rho = self.rtz[:, 0]
        bins = {f"{a:g}-{b:g}": self.average(f, (rho > a) & (rho <= b))
                for a, b in zip(rho_bins[:-1], rho_bins[1:])}
        return {"mean": self.average(f), "rms": float(np.sqrt(self.average(f**2))),
                "max": float(f.max()), "bins": bins}

    def desc_fields(self):
        """DESC's J, B, grad p (Cartesian components) and p at the collocation points."""
        names, g = ["J", "B", "grad(p)", "p"], self.tgrid
        d = self.inp.compute_tensor(names, g.rho, g.theta, g.zeta)
        # spot check against DESC's pointwise evaluation
        sample = np.random.default_rng(1).choice(len(self.rtz), min(500, len(self.rtz)),
                                                 replace=False)
        ref = self.inp.compute_points(names, self.rtz[sample])
        err = max(np.abs(d[k][sample] - ref[k]).max() / np.abs(ref[k]).max() for k in names)
        if err > 1e-8:
            raise RuntimeError(f"DESC's tensor evaluation differs from its pointwise one by {err:.1e}")
        out = {k: cylindrical_to_cartesian(d[k], self.phi) for k in ("J", "B", "grad(p)")}
        out["p"] = d["p"]
        return out

    def decompose(self, out, desc=None):
        """F = F_desc + (J - J_desc) x B_desc + J x (B - B_desc) - (grad p - grad p_desc)."""
        desc = self.desc_fields() if desc is None else desc
        Jd, Bd, gpd = desc["J"], desc["B"], desc["grad(p)"]
        parts = {
            "DESC force balance": np.cross(Jd, Bd) - gpd,
            "current fit (J - J_desc) x B_desc": np.cross(out["J"] - Jd, Bd),
            "field (B - B_desc), x J": np.cross(out["J"], out["B"] - Bd),
            "pressure fit -(grad p - grad p_desc)": -(out["grad_p"] - gpd),
        }
        residual = out["F"] - sum(parts.values())
        return parts, residual, desc
