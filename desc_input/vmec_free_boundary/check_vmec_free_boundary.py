"""Load a VMEC free-boundary wout and its coils in DESC and check that they match.

For each case:
1. the MAKEGRID coil file is read into a DESC coil set (currents in amperes, as the
   VMEC run used them through EXTCUR) and saved as ``<case>_coils.h5``;
2. the wout is fitted by DESC (``VMECIO.load``, current profile) and saved as
   ``<case>_vmec.h5``;
3. the free-boundary residuals of that equilibrium in the coil field are computed
   with ``BoundaryError`` (plasma field from virtual casing): B.n/|B| and the
   pressure-balance jump [[B^2 + 2 mu0 p]]/B^2 on the LCFS, plus beta, net current
   and the DESC force error of the VMEC solution.

Usage: python check_vmec_free_boundary.py [NCSX QH_5beta CTH_like] [--device cpu]
"""

import argparse
import json
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, "..", "precise_QA_free_boundary"))

parser = argparse.ArgumentParser()
parser.add_argument("cases", nargs="*", default=["NCSX", "QH_5beta", "CTH_like"])
parser.add_argument("--device", default="cpu")
args = parser.parse_args()

from desc import set_device  # noqa: E402

set_device(args.device)

import netCDF4  # noqa: E402
import numpy as np  # noqa: E402

from desc.coils import (  # noqa: E402
    FourierPlanarCoil,
    MixedCoilSet,
    SplineXYZCoil,
)
from desc.grid import LinearGrid  # noqa: E402
from desc.vmec import VMECIO  # noqa: E402
from fb_tools import boundary_error, force_error, log  # noqa: E402

CASES = {
    # coil-file currents are already the EXTCUR amperes (MAKEGRID scaled mode)
    "NCSX": {"coils": "coils.c09r00", "wout": "wout_ncsx_c09r00_free.nc"},
    "QH_5beta": {
        "coils": "coils.5Beta_2023_09_28_13_16_38",
        "wout": "wout_LBDQH5_coils.nc",
    },
    # MAKEGRID raw mode: the file holds turns, multiplied by EXTCUR per group
    "CTH_like": {
        "coils": "coils.cth_like",
        "wout": "wout_cth_like_free_bdy.nc",
        "extcur": [4700.0, 1000.0],
    },
}


def read_makegrid_raw(path, extcur):
    """MAKEGRID file with raw currents, including VMEC++ circular filaments.

    A filament with a single line is a circle of radius R at height z about the z
    axis (VMEC++ convention); the others are closed polygons. Currents are
    turns * EXTCUR(group).
    """
    coils, pts = [], []
    with open(path) as f:
        lines = f.read().split("** coils_dot_starts_below **")[-1].splitlines()
    for line in lines:
        w = line.split()
        if len(w) < 4 or w[0] in ("periods", "begin", "mirror"):
            continue
        pts.append([float(v) for v in w[:4]])
        if len(w) >= 6:  # last point of a filament: group index and name
            g = int(w[4]) - 1
            p = np.array(pts)
            pts = []
            if len(p) == 1:
                R, z, turns = np.hypot(p[0, 0], p[0, 1]), p[0, 2], p[0, 3]
                coils.append(
                    FourierPlanarCoil(
                        current=turns * extcur[g], center=[0, 0, z],
                        normal=[0, 0, 1], r_n=R, name=w[5],
                    )
                )
            else:
                # last point repeats the first one with zero current
                coils.append(
                    SplineXYZCoil(
                        current=p[0, 3] * extcur[g], X=p[:-1, 0], Y=p[:-1, 1],
                        Z=p[:-1, 2], name=w[5],
                    )
                )
    return MixedCoilSet(*coils)


def wout_scalars(path):
    d = netCDF4.Dataset(path)
    keys = ["betatotal", "ctor", "fsqr", "fsqz", "fsql", "Rmajor_p", "Aminor_p",
            "volavgB", "ns", "mpol", "ntor", "nfp"]
    return {k: float(np.asarray(d.variables[k][:])) for k in keys}


results = {}
for case in args.cases:
    c = CASES[case]
    folder = os.path.join(HERE, case)
    log(f"{case}: reading coils")
    if "extcur" in c:
        coils = read_makegrid_raw(os.path.join(folder, c["coils"]), c["extcur"])
    else:
        coils = MixedCoilSet.from_makegrid_coilfile(os.path.join(folder, c["coils"]))
    coils.save(os.path.join(folder, f"{case}_coils.h5"))

    wout = os.path.join(folder, c["wout"])
    s = {"wout": wout_scalars(wout)}
    log(f"{case}: fitting the wout in DESC")
    eq = VMECIO.load(wout, profile="current")
    eq.save(os.path.join(folder, f"{case}_vmec.h5"))
    s["DESC L M N"] = [eq.L, eq.M, eq.N]

    d = eq.compute(["<beta>_vol", "R0", "a", "<|B|>_vol"])
    s.update({k: float(d[k]) for k in ["<beta>_vol", "R0", "a", "<|B|>_vol"]})
    g = LinearGrid(rho=np.array([1.0]), M=eq.M_grid, N=eq.N_grid, NFP=eq.NFP)
    s["I_edge [A]"] = float(g.compress(eq.compute("current", grid=g)["current"])[0])
    g = LinearGrid(rho=np.linspace(0, 1, 11), M=eq.M_grid, N=eq.N_grid, NFP=eq.NFP)
    s["iota(rho=0..1)"] = g.compress(eq.compute("iota", grid=g)["iota"]).tolist()
    log(f"{case}: force error")
    s["f_normalized (DESC on the VMEC solution)"] = force_error(eq)
    log(f"{case}: boundary error in the coil field")
    s.update(boundary_error(eq, coils))
    log(f"{case}: " + json.dumps(s, indent=1))
    results[case] = s
    with open(os.path.join(folder, f"{case}_check.json"), "w") as f:
        json.dump(s, f, indent=2)
