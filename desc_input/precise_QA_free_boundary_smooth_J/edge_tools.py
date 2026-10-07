"""Edge current-density diagnostics shared by the smooth-J scripts."""

import numpy as np

from desc.grid import Grid, LinearGrid
from desc.integrals.surface_integral import surface_averages

J_REF = 3.9e5  # peak <|J|> of the smooth-J equilibrium [A/m^2]


def edge_current(eq):
    """Edge current density: averages, pointwise max of |J_par|, (4,2) harmonic."""
    rho = np.array([0.5, 0.9, 0.96, 0.98, 1.0])
    g = LinearGrid(rho=rho, M=eq.M_grid, N=eq.N_grid, NFP=eq.NFP, sym=False)
    d = eq.compute(["|J|", "J*B", "|B|", "sqrt(g)"], grid=g)
    jpar = np.abs(d["J*B"] / d["|B|"])
    avg = lambda q: np.asarray(  # noqa: E731
        surface_averages(g, q, d["sqrt(g)"], expand_out=False)
    )
    amp42 = []
    nt = nz = 64
    for r in rho:
        tp, z = np.meshgrid(np.linspace(0, 2 * np.pi, nt, endpoint=False),
                            np.linspace(0, 2 * np.pi, nz, endpoint=False),
                            indexing="ij")
        nodes = eq.map_coordinates(np.c_[np.full(tp.size, r), tp.ravel(), z.ravel()],
                                   inbasis=("rho", "theta_PEST", "zeta"),
                                   outbasis=("rho", "theta", "zeta"))
        dd = eq.compute(["J*B", "|B|"], grid=Grid(np.asarray(nodes), sort=False))
        c = np.fft.fft2(np.asarray(dd["J*B"] / dd["|B|"]).reshape(nt, nz)) / (nt * nz)
        amp42.append(2 * abs(c[4, -2 % nz]))
    return {
        "rho": rho.tolist(),
        "<|J|>/J_ref": (avg(d["|J|"]) / J_REF).tolist(),
        "max |J_par|/J_ref": (g.meshgrid_reshape(jpar, "rtz").max(axis=(1, 2))
                              / J_REF).tolist(),
        "net |<J.B>|/<B^2>^.5/J_ref": (np.abs(avg(d["J*B"]))
                                       / np.sqrt(avg(d["|B|"] ** 2)) / J_REF).tolist(),
        "(4,-2) J_par amplitude [kA/m^2]": (np.array(amp42) / 1e3).tolist(),
    }


def show(label, e, log=print):
    log(label)
    print("  rho   <|J|>      max|J_par|  net <J.B>   (4,2) [kA/m2]", flush=True)
    for i, r in enumerate(e["rho"]):
        print(f"  {r:.2f}  {e['<|J|>/J_ref'][i]:.3e}  {e['max |J_par|/J_ref'][i]:.3e}  "
              f"{e['net |<J.B>|/<B^2>^.5/J_ref'][i]:.3e}  "
              f"{e['(4,-2) J_par amplitude [kA/m^2]'][i]:.1f}", flush=True)


def rho_of_iota(eq, target=0.5):
    """Radii where iota crosses ``target`` (linear interpolation on a fine grid)."""
    rho = np.linspace(0, 1, 1001)
    g = LinearGrid(rho=rho, M=eq.M_grid, N=eq.N_grid, NFP=eq.NFP)
    iota = g.compress(eq.compute("iota", grid=g)["iota"])
    idx = np.nonzero(np.diff(np.sign(iota - target)))[0]
    return [float(rho[i] + (target - iota[i]) * (rho[i + 1] - rho[i])
                  / (iota[i + 1] - iota[i])) for i in idx]
