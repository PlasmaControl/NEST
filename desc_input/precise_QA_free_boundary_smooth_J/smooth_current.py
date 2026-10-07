"""precise_QA free-boundary equilibrium with a current density that vanishes smoothly
at the boundary, as an initial configuration for DESC.

Starts from ``precise_QA_free_boundary_L16N14.h5`` (iota prescribed, p ~ (1-rho^2)^2).
There J_perp ~ p' already vanishes at rho = 1, but only linearly, and the parallel
current does not: with iota prescribed, the toroidal current density dI/drho is
finite at the edge. Here

- the toroidal current I(s), s = rho^2, is prescribed instead of iota: a fit of the
  starting equilibrium's I(s) with dI/ds = (1 - s)^2 Q(s), Q a polynomial, so the
  toroidal current density and its radial derivative vanish at the edge;
- p = p0 (1 - rho^2)^3, so p' and p'' vanish at the edge too; p0 is rescaled to keep
  the same <beta>;

hence J -> 0 at the LCFS with zero radial slope. Everything is re-solved on the same
(free-boundary) LCFS, in stages:

A. new pressure, iota still prescribed (iota at finite beta depends on the pressure
   profile, so the current must be fitted after the pressure change);
B. constrained fit of the stage-A current, tight for rho < 0.77 so that iota at the
   iota = 1/2 surface (rho ~ 0.55) is kept;
C. current prescribed, ramped from the stage-A current to the fit;
D. radial resolution raised (default L = 24) to resolve the edge, where J -> 0.

usage: python smooth_current.py EQ.h5 COILS.h5 OUTDIR [JAC_CHUNK_SIZE [L_FINAL]]
"""

import json
import os
import sys

from desc import set_device

set_device("gpu")

import matplotlib  # noqa: E402

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

import desc.io  # noqa: E402
from desc.grid import LinearGrid  # noqa: E402
from desc.integrals.surface_integral import surface_averages  # noqa: E402
from desc.objectives import ForceBalance, ObjectiveFunction  # noqa: E402
from desc.plotting import plot_section  # noqa: E402
from desc.profiles import PowerSeriesProfile  # noqa: E402
from fb_tools import log, pad, summary  # noqa: E402

eq = desc.io.load(sys.argv[1])
coils = desc.io.load(sys.argv[2])
outdir = sys.argv[3]
chunk = int(sys.argv[4]) if len(sys.argv) > 4 else None  # limits GPU memory
L_final = int(sys.argv[5]) if len(sys.argv) > 5 else 24  # radial resolution, step D
os.makedirs(outdir, exist_ok=True)
K = 5  # degree + 1 of Q(s)


def radial_profiles(eq):
    """Flux-surface averages of |J| and J_parallel, and I, iota, p versus rho."""
    rho = np.linspace(0.02, 1.0, 50)
    g = LinearGrid(rho=rho, M=eq.M_grid, N=eq.N_grid, NFP=eq.NFP, sym=False)
    d = eq.compute(["|J|", "J_parallel", "sqrt(g)", "current", "iota", "p"], grid=g)
    avg = lambda q: np.asarray(  # noqa: E731
        surface_averages(g, q, d["sqrt(g)"], expand_out=False)
    )
    return {
        "rho": rho,
        "<|J|>": avg(d["|J|"]),
        "<|J_par|>": avg(np.abs(d["J_parallel"])),
        "I": g.compress(d["current"]),
        "iota": g.compress(d["iota"]),
        "p": g.compress(d["p"]),
    }


def edge_metrics(prof):
    J = prof["<|J|>"]
    return {
        "<|J|>(rho=1) / max <|J|>": float(J[-1] / J.max()),
        "<|J|>(rho=0.98) / max <|J|>": float(J[-2] / J.max()),
        "<|J_par|>(rho=1) / max <|J|>": float(prof["<|J_par|>"][-1] / J.max()),
    }


# ---------------------------------------------------------------------------
# current profile fit with dI/ds = (1 - s)^2 Q(s)
# ---------------------------------------------------------------------------
diags = {"before": summary(eq, coils, "before (iota prescribed)")}
prof0 = radial_profiles(eq)
diags["before"].update(edge_metrics(prof0))
beta0 = diags["before"]["<beta>_vol"]
objective = ObjectiveFunction(ForceBalance(eq=eq), jac_chunk_size=chunk)


def solve(tight=False):
    eq.solve(objective=objective, ftol=1e-10 if tight else 1e-8, xtol=1e-12,
             gtol=1e-12, maxiter=300 if tight else 100, verbose=1)


nj = K + 3
file_C = os.path.join(outdir, "precise_QA_free_boundary_smooth_J_L16.h5")
if os.path.exists(file_C):  # resume after stage C (e.g. step D ran out of memory)
    eq = desc.io.load(file_C)
    c_new = np.asarray(eq.current.params[:nj])
    p0_new = float(eq.pressure.params[0])
    fit_err = float("nan")  # printed in the log of the run that made file_C
    diags["C_smooth_current_L16"] = summary(eq, coils, "C: smooth current, L=16")
    prof_C = radial_profiles(eq)
    diags["C_smooth_current_L16"].update(edge_metrics(prof_C))
    log(f"resumed from {file_C}")
else:
    # ---------------------------------------------------------------------------
    # A. pressure (1-s)^2 -> (1-s)^3 with iota still prescribed (iota at finite beta
    #    depends on the pressure profile, so the current is fitted only after this)
    # ---------------------------------------------------------------------------
    p_old = np.zeros(nj)
    n_p = min(nj, eq.pressure.params.size)
    p_old[:n_p] = eq.pressure.params[:n_p]  # (1 - s)^2: only the first 3 are nonzero
    p0_new = p_old[0] * 4 / 3  # <(1-s)^3> / <(1-s)^2> ~ (1/4) / (1/3), refined below
    p_new = lambda p0: p0 * np.pad([1.0, -3.0, 3.0, -1.0], (0, nj - 4))  # noqa: E731
    for frac in [0.5, 1.0]:
        eq.pressure.params = pad((1 - frac) * p_old + frac * p_new(p0_new), eq.pressure)
        log(f"A: pressure ramp {frac} (iota prescribed)")
        solve()
    p0_new *= beta0 / float(eq.compute("<beta>_vol")["<beta>_vol"])
    eq.pressure.params = pad(p_new(p0_new), eq.pressure)
    solve(tight=True)
    diags["A_new_pressure_iota_prescribed"] = summary(eq, coils, "A: new pressure")
    eq.save(os.path.join(outdir, "stageA_new_pressure_iota_prescribed.h5"))

    # ---------------------------------------------------------------------------
    # B. fit the current of stage A with dI/ds = (1 - s)^2 Q(s)
    # ---------------------------------------------------------------------------
    s_fit = np.linspace(0, 1, 201)
    g = LinearGrid(rho=np.sqrt(s_fit), M=eq.M_grid, N=eq.N_grid, NFP=eq.NFP)
    I_fit = g.compress(eq.compute("current", grid=g)["current"])
    # basis phi_k(s) = int_0^s (1-t)^2 t^k dt, written as coefficients of s^j, j=0..K+2
    basis = np.zeros((K, nj))
    for k in range(K):
        basis[k, k + 1] += 1 / (k + 1)
        basis[k, k + 2] += -2 / (k + 2)
        basis[k, k + 3] += 1 / (k + 3)
    A = np.vander(s_fit, nj, increasing=True) @ basis.T
    # iota on the resonant surface (rho ~ 0.55) is set by the current enclosed inside
    # it: fit tightly for rho < 0.77 and let the edge flatten
    w = np.where(s_fit < 0.6, 1.0, 0.1)
    q, *_ = np.linalg.lstsq(A * w[:, None], I_fit * w, rcond=None)
    c_new = basis.T @ q  # coefficients of s^j for the smooth current
    # unconstrained fit of the stage-A current on the same modes, to ramp from
    c_old, *_ = np.linalg.lstsq(np.vander(s_fit, nj, increasing=True)[:, 1:], I_fit,
                                rcond=None)
    c_old = np.r_[0.0, c_old]
    fit_err = np.max(np.abs(A @ q - I_fit)[s_fit < 0.6]) / np.max(np.abs(I_fit))
    log(f"B: I_edge = {I_fit[-1]:.4e} A; smooth fit I_edge = {(A @ q)[-1]:.4e} A; "
        f"max fit deviation for rho < 0.77 = {fit_err:.2e} of max |I|")

    # ---------------------------------------------------------------------------
    # C. prescribe the smooth current instead of iota
    # ---------------------------------------------------------------------------
    eq.iota = None
    eq.current = PowerSeriesProfile(c_old, modes=2 * np.arange(nj))
    objective = ObjectiveFunction(ForceBalance(eq=eq), jac_chunk_size=chunk)
    for frac in [0.5, 1.0]:
        eq.current.params = pad((1 - frac) * c_old + frac * c_new, eq.current)
        log(f"C: current ramp {frac}")
        solve()
    solve(tight=True)
    diags["C_smooth_current_L16"] = summary(eq, coils, "C: smooth current, L=16")
    prof_C = radial_profiles(eq)
    diags["C_smooth_current_L16"].update(edge_metrics(prof_C))
    eq.save(file_C)

# ---------------------------------------------------------------------------
# D. more radial resolution to resolve the edge, where J -> 0
# ---------------------------------------------------------------------------
eq.change_resolution(L=L_final, L_grid=2 * L_final)
objective = ObjectiveFunction(ForceBalance(eq=eq), jac_chunk_size=chunk)
log(f"D: L = {L_final}")
solve(tight=True)
diags["after"] = summary(eq, coils, f"D: smooth current, L={L_final}")
prof1 = radial_profiles(eq)
diags["after"].update(edge_metrics(prof1))
diags["params"] = {
    "current: s^j coefficients, j=0..": c_new.tolist(),
    "pressure: p0 (1-rho^2)^3, p0 [Pa]": float(p0_new),
    "current fit max deviation (rho < 0.77) / max|I|": float(fit_err),
    "L, M, N": [eq.L, eq.M, eq.N],
    "source": os.path.abspath(sys.argv[1]),
}
eq.save(os.path.join(outdir, "precise_QA_free_boundary_smooth_J.h5"))
with open(os.path.join(outdir, "diagnostics.json"), "w") as f:
    json.dump(diags, f, indent=2)

# ---------------------------------------------------------------------------
# plots
# ---------------------------------------------------------------------------
fig, ax = plt.subplots(2, 2, figsize=(10, 8))
for prof, lab, c in [(prof0, "before: iota prescribed", "b"),
                     (prof_C, "smooth current, L = 16", "g"),
                     (prof1, f"smooth current, L = {L_final}", "r")]:
    ax[0, 0].semilogy(prof["rho"], prof["<|J|>"], c, label=lab)
    ax[0, 0].semilogy(prof["rho"], prof["<|J_par|>"], c + "--")
    ax[0, 1].plot(prof["rho"], prof["I"] / 1e3, c, label=lab)
    ax[1, 0].plot(prof["rho"], prof["iota"], c, label=lab)
    ax[1, 1].plot(prof["rho"], prof["p"] / 1e3, c, label=lab)
ax[0, 0].set_ylabel(r"$\langle|J|\rangle$ (solid), $\langle|J_\parallel|\rangle$ "
                    r"(dashed) [A/m$^2$]")
ax[0, 1].set_ylabel("toroidal current I [kA]")
ax[1, 0].axhline(0.5, color="k", ls=":", lw=0.8)
ax[1, 0].set_ylabel(r"$\iota$ (dotted: 1/2)")
ax[1, 1].set_ylabel("p [kPa]")
for a in ax.flat:
    a.set_xlabel(r"$\rho$")
    a.legend(fontsize=8)
fig.tight_layout()
fig.savefig(os.path.join(outdir, "profiles_before_after.png"), dpi=130)
fig, _ = plot_section(eq, "|J|")
fig.savefig(os.path.join(outdir, "J_section.png"), dpi=130)
fig, _ = plot_section(eq, "|F|_normalized", log=True)
fig.savefig(os.path.join(outdir, "force_error_section.png"), dpi=130)
log("done")
