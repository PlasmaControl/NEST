"""Diagnostics shared by the DESC free-boundary scripts that feed NEMO."""

import json
import time
from fractions import Fraction

import numpy as np

from desc.grid import LinearGrid, QuadratureGrid
from desc.objectives import BoundaryError

T0 = time.time()


def log(msg):
    print(f"[{time.time() - T0:8.1f}s] {msg}", flush=True)


def pad(c, profile):
    """Zero-pad leading coefficients ``c`` to the size of ``profile.params``."""
    return np.pad(c, (0, profile.params.size - c.size))


def force_error(eq):
    """Volume-averaged force error normalized by magnetic pressure gradient."""
    grid = QuadratureGrid(L=2 * eq.L, M=2 * eq.M, N=2 * eq.N, NFP=eq.NFP)
    d = eq.compute(["<|F|>_vol", "<|grad(|B|^2)|/2mu0>_vol"], grid=grid)
    return float(d["<|F|>_vol"] / d["<|grad(|B|^2)|/2mu0>_vol"])


def boundary_error(eq, field):
    """Pointwise free-boundary residuals on the BoundaryError evaluation grid.

    Returns B.n/|B| (total field: external + plasma via virtual casing) and the
    relative pressure jump [[B^2 + 2 mu0 p]] / B^2.
    """
    obj = BoundaryError(eq=eq, field=field, field_fixed=True)
    obj.build(verbose=0)
    f = np.asarray(obj.compute(*obj.xs(eq)))
    grid = obj.constants["eval_transforms"]["grid"]
    d = eq.compute(["|e_theta x e_zeta|", "|B|"], grid=grid)
    n = grid.num_nodes
    Bn = f[:n] / (d["|e_theta x e_zeta|"] * d["|B|"])
    dP = f[n : 2 * n] / (d["|e_theta x e_zeta|"] * d["|B|"] ** 2)
    return {
        "Bn/|B| max": float(np.max(np.abs(Bn))),
        "Bn/|B| rms": float(np.sqrt(np.mean(Bn**2))),
        "[[B^2+2mu0p]]/B^2 max": float(np.max(np.abs(dP))),
        "[[B^2+2mu0p]]/B^2 rms": float(np.sqrt(np.mean(dP**2))),
    }


def rational_surfaces(eq, max_m=12):
    """Locate rational surfaces iota = n/m that NFP-symmetric fields can drive.

    With stellarator symmetry and NFP field periods only toroidal harmonics with
    n = k*NFP exist, so these are the surfaces where islands can open in NEMO.
    Returns rho location and magnetic shear s = -(rho/iota) diota/drho there.
    """
    rho = np.linspace(1e-3, 1, 1001)
    g = LinearGrid(rho=rho, M=eq.M_grid, N=eq.N_grid, NFP=eq.NFP)
    d = eq.compute(["iota", "iota_r"], grid=g)
    iota, iota_r = g.compress(d["iota"]), g.compress(d["iota_r"])
    # helicity sign depends on the sign convention of Psi; resonance needs |iota|
    iota, iota_r = np.abs(iota), np.sign(iota) * iota_r
    out = {}
    for m in range(1, max_m + 1):
        for n in range(eq.NFP, 3 * eq.NFP + 1, eq.NFP):
            t = Fraction(n, m)
            if t in out:  # keep the lowest-m harmonic driving this rational
                continue
            for i in np.nonzero(np.diff(np.sign(iota - float(t))))[0]:
                r = rho[i] + (float(t) - iota[i]) * (rho[i + 1] - rho[i]) / (
                    iota[i + 1] - iota[i]
                )
                out.setdefault(t, []).append(
                    {"m": m, "n": n, "iota": float(t), "rho": float(r),
                     "shear": float(-r / float(t) * np.interp(r, rho, iota_r))}
                )
    return [s for v in out.values() for s in v]


def summary(eq, field, label):
    s = {"label": label, "f_normalized": force_error(eq)}
    s.update(boundary_error(eq, field))
    d = eq.compute(["<beta>_vol", "R0", "a"])
    s["<beta>_vol"] = float(d["<beta>_vol"])
    s["R0"] = float(d["R0"])
    s["a"] = float(d["a"])
    rho = np.linspace(0, 1, 11)
    g = LinearGrid(rho=rho, M=eq.M_grid, N=eq.N_grid, NFP=eq.NFP)
    di = eq.compute(["iota", "current"], grid=g)
    s["iota(rho=0..1)"] = g.compress(di["iota"]).tolist()
    s["I_edge [A]"] = float(g.compress(di["current"])[-1])
    s["rational_surfaces"] = rational_surfaces(eq)
    log(f"{label}: " + json.dumps(s, indent=1))
    return s


def free_boundary_stages(
    eq, objective, constraints, k_list, maxiter, inner_ftol, field, diags, tag,
    outdir, diag_file,
):
    """Free-boundary solve releasing boundary modes max(|m|,|n|) <= k stage by stage.

    Checkpoints the equilibrium and the diagnostics after every stage.
    """
    import os

    from desc.objectives import FixBoundaryR, FixBoundaryZ

    for k in k_list:
        modes = eq.surface.R_basis.modes
        fixR = FixBoundaryR(eq=eq, modes=modes[np.max(np.abs(modes), 1) > k])
        modes = eq.surface.Z_basis.modes
        fixZ = FixBoundaryZ(eq=eq, modes=modes[np.max(np.abs(modes), 1) > k])
        log(f"{tag}: releasing boundary modes with max(|m|,|n|) <= {k}")
        eq, out = eq.optimize(
            objective,
            constraints + (fixR, fixZ),
            optimizer="proximal-lsq-exact",
            maxiter=maxiter,
            ftol=1e-8,
            xtol=1e-12,
            gtol=1e-12,
            verbose=2,
            options={"solve_options": {"ftol": inner_ftol, "xtol": 1e-12,
                                       "gtol": 1e-12, "maxiter": 100}},
            copy=False,
        )
        log(f"{tag} k={k}: {out['message']} (nit={out['nit']})")
        diags[f"{tag}_k{k}"] = summary(eq, field, f"{tag} k={k}")
        eq.save(os.path.join(outdir, f"{tag}_k{k}.h5"))
        with open(diag_file, "w") as f:
            json.dump(diags, f, indent=2)
    return eq
