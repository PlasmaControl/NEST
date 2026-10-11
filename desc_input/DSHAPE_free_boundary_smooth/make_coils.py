"""Build the external field files from coil_parameters.json (the fitted coil set).

    python make_coils.py      -> DSHAPE_smooth_TF.h5, DSHAPE_smooth_PF_coils.h5

The field is a 1/R toroidal field, B_phi = B0 R0 / R with B0 R0 = <R B_phi> on the
LCFS, plus 24 up-down-symmetric circular poloidal-field rings. The rings sit on the
LCFS offset outward by 1 m, equally spaced in arc length; their currents were fitted
by weighted least squares so that B_coils.n = -B_plasma.n on the LCFS (virtual
casing) of the fixed-boundary start (fit residual B.n/|B| rms 5.1e-6). Each ring is a
FourierPlanarCoil with zero modes padded to |n| <= 64, so that DESC's default
discretization (2 N + 5 points) is 133 points; an unpadded circle gets 11 points (2 %
field error).

The two parts are separate files because DESC cannot save a SumMagneticField
(``SumMagneticField._io_attrs`` should be ``_io_attrs_``, so nothing of it is written;
DESC 0.16 and 0.17). Load the sum with
``DESCInput.load(eq_file, field=[TF_FILE, PF_FILE])`` or ``coil_field(params)`` here.
"""

import json
import os

import numpy as np
from desc.coils import CoilSet, FourierPlanarCoil
from desc.magnetic_fields import SumMagneticField, ToroidalMagneticField

HERE = os.path.dirname(os.path.abspath(__file__))
TF_FILE = os.path.join(HERE, "DSHAPE_smooth_TF.h5")
PF_FILE = os.path.join(HERE, "DSHAPE_smooth_PF_coils.h5")
NPAD = 64


def ring(R, Z, I):
    r_n = np.zeros(2 * NPAD + 1)
    r_n[NPAD] = R
    return FourierPlanarCoil(current=I, center=[0, 0, Z], normal=[0, 0, 1], r_n=r_n,
                             modes=np.arange(-NPAD, NPAD + 1))


def parts(params):
    """(toroidal field, CoilSet of the PF rings)."""
    rings = CoilSet(*[ring(R, Z, I) for R, Z, I in
                      zip(params["ring_R"], params["ring_Z"], params["ring_I_A"])])
    return ToroidalMagneticField(B0=params["B0_TF"], R0=params["R0"]), rings


def coil_field(params):
    """The whole external field, TF + PF rings."""
    return SumMagneticField(list(parts(params)))


if __name__ == "__main__":
    params = json.load(open(os.path.join(HERE, "coil_parameters.json")))
    tf, pf = parts(params)
    tf.save(TF_FILE)
    pf.save(PF_FILE)
