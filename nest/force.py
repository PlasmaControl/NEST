"""Step 6 of NEST: B = B_ext + B_plasma and the force residual F = J x B - grad p.

F is evaluated on collocation points of its own, a Gauss-uniform grid in the
flux coordinates of the fixed equilibrium (by default DESC's QuadratureGrid(2 L,
2 M, 2 N): Gauss-Legendre in rho, uniform in theta and zeta over one field
period), with the volume weights sqrt(g) w:

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

with DESC's normalization; with the default grid, DESC's own force error is
exactly the value quoted for the equilibria in desc_input.

``decompose`` splits F against DESC's fields at the same points:

    F = F_desc + (J - J_desc) x B_desc + J x (B - B_desc) - (grad p - grad p_desc)

DESC's own force imbalance, the error of the fitted current, of the field (coils
plus B_plasma against DESC's B) and of the pressure; the identity is exact.
"""

import numpy as np
from desc.grid import QuadratureGrid

from cartesian_grid import cylindrical_to_cartesian, evaluate_chunked
from fourier import fft_J
from magnetic_field import b_plasma_free, biot_savart_kernel_hat, interpolate


class ForceResidual:
    """F = J x B - grad p on a flux-coordinate collocation grid.

    Parameters
    ----------
    inp : DESCInput
        The equilibrium and its external field (needed for B_ext).
    box : CartesianGrid
        The periodic box on which J is sampled for B_plasma (``locate_plasma`` done).
    grid : desc Grid, optional
        The collocation points (nodes in flux coordinates, weights); default
        QuadratureGrid(2 L, 2 M, 2 N) of the equilibrium.
    eval_chunk : int
        Points per call of the series evaluators.

    The points' positions, volume weights, B_ext, the Biot-Savart kernel and
    DESC's normalization do not depend on J or p: computed once, here.
    """

    def __init__(self, inp, box, grid=None, eval_chunk=50000):
        if box.inside is None:
            raise RuntimeError("locate the plasma on the box first (box.locate_plasma)")
        eq = inp.eq
        self.inp, self.box, self.eval_chunk = inp, box, eval_chunk
        self.grid = grid or QuadratureGrid(L=2 * eq.L, M=2 * eq.M, N=2 * eq.N, NFP=eq.NFP)
        self.rtz = np.asarray(self.grid.nodes)
        d = inp.compute(["R", "phi", "Z", "sqrt(g)", "<|F|>_vol", "<|grad(|B|^2)|/2mu0>_vol"],
                        self.grid)
        self.phi = d["phi"]
        self.x = np.column_stack([d["R"] * np.cos(self.phi), d["R"] * np.sin(self.phi), d["Z"]])
        self.dV = np.abs(d["sqrt(g)"]) * np.asarray(self.grid.weights)  # volume weights
        self.norm = float(np.ravel(d["<|grad(|B|^2)|/2mu0>_vol"])[0])
        self.F_desc_vol = float(np.ravel(d["<|F|>_vol"])[0])  # DESC's own <|F|>
        self.B_ext = inp.B_ext(self.x)
        self.K_hat = biot_savart_kernel_hat(box.axes)

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
        """DESC's J, B and grad p at the collocation points (Cartesian components)."""
        d = self.inp.compute(["J", "B", "grad(p)"], self.grid)
        return {k: cylindrical_to_cartesian(d[k], self.phi) for k in ("J", "B", "grad(p)")}

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
