"""Pressure and net toroidal current of a VMEC run as even power series in rho.

VMEC defines the profiles in s = rho^2 through the input coefficients (AM, AC,
AC_AUX_*), stored in the wout. Rebuilding them from those definitions keeps
I ~ rho^2 and p - p0 ~ rho^2 at the axis. The spline that ``VMECIO.load`` fits to
the wout's discrete current does not: it wiggles near the axis and iota spikes
there.

The wout does not store PRES_SCALE: it is fitted to the wout pressure (presf)
when the AM profile alone is off by more than 1 %.

- pressure ``power_series``: p = PRES_SCALE sum am_k s^k (exact).
- pressure ``two_power``: p = PRES_SCALE am0 (1 - s^am1)^am2 (exact for integer
  am1, am2).
- current ``power_series``: I'(s) = sum ac_k s^k (exact).
- current ``two_power``: I'(s) = ac0 (1 - s^ac1)^ac2 (exact for integer ac1, ac2).
- current ``cubic_spline_ip``: I'(s) is a cubic spline through
  (AC_AUX_S, AC_AUX_F); I(s) is least-squares fitted with rho^2 .. rho^(2K).
The current is normalized to CURTOR at the edge.
"""

import netCDF4
import numpy as np
from scipy.interpolate import CubicSpline
from scipy.special import comb

from desc.profiles import PowerSeriesProfile


def _str(d, k):
    return b"".join(np.asarray(d[k][:]).ravel()).decode().strip()


def _two_power(c0, c1, c2):
    """Coefficients (power of s -> value) of c0 (1 - s^c1)^c2, integer c1, c2."""
    c1, c2 = int(round(c1)), int(round(c2))
    return {c1 * k: c0 * comb(c2, k, exact=True) * (-1) ** k for k in range(c2 + 1)}


def _series(coef, name):
    """PowerSeriesProfile in rho from {power of s: coefficient}."""
    modes = np.array(sorted(coef)) * 2
    return PowerSeriesProfile(
        params=np.array([coef[k] for k in sorted(coef)]), modes=modes, name=name
    )


def vmec_profiles(wout, K=12):
    """Return (pressure, current, info) from a VMEC wout file."""
    d = netCDF4.Dataset(wout)
    am = np.asarray(d["am"][:], float)
    ac = np.asarray(d["ac"][:], float)
    curtor = float(d["ctor"][:])

    ptype = _str(d, "pmass_type")
    if ptype == "power_series":
        p = {k: a for k, a in enumerate(am) if a != 0}
    elif ptype == "two_power":
        p = _two_power(*am[:3])
    else:
        raise NotImplementedError(ptype)
    presf = np.asarray(d["presf"][:], float)
    s = np.linspace(0, 1, presf.size)
    f = sum(v * s**k for k, v in p.items())
    pres_scale = float(f @ presf / (f @ f))
    if abs(pres_scale - 1) < 1e-2:
        pres_scale = 1.0
    pressure = _series({k: v * pres_scale for k, v in p.items()}, "pressure")

    ctype = _str(d, "pcurr_type")
    info = {"pmass_type": ptype, "pcurr_type": ctype, "pres_scale": pres_scale}
    if ctype in ("power_series", "two_power"):
        # I'(s) = sum c_k s^k  ->  I(s) = sum c_k s^(k+1) / (k+1)
        dI = dict(enumerate(ac)) if ctype == "power_series" else _two_power(*ac[:3])
        c = {k + 1: v / (k + 1) for k, v in dI.items() if v != 0}
    elif ctype == "cubic_spline_ip":
        s_aux = np.asarray(d["ac_aux_s"][:], float)
        n = np.argmax(s_aux[1:] <= s_aux[:-1]) + 1  # aux arrays are padded
        dI = CubicSpline(s_aux[:n], np.asarray(d["ac_aux_f"][:n], float))
        I = dI.antiderivative()
        rho = np.linspace(0, 1, 2001)
        A = rho[:, None] ** (2 * np.arange(1, K + 1))[None, :]
        coef = np.linalg.lstsq(A, I(rho**2), rcond=None)[0]
        c = {k + 1: v for k, v in enumerate(coef)}
        info["fit rms / |I(1)|"] = float(
            np.sqrt(np.mean((A @ coef - I(rho**2)) ** 2)) / abs(I(1.0))
        )
    else:
        raise NotImplementedError(ctype)
    scale = curtor / sum(c.values())
    current = _series({k: v * scale for k, v in c.items()}, "current")
    return pressure, current, info
