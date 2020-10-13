import gt4py.gtscript as gtscript
from gt4py.gtscript import PARALLEL, computation, interval, parallel, region

import fv3core._config as spec
import fv3core.utils.gt4py_utils as utils
from fv3core.decorators import gtstencil


sd = utils.sd


@gtstencil()
def circulation_cgrid(uc: sd, vc: sd, dxc: sd, dyc: sd, vort_c: sd):
    """Update vort_c.

    Args:
        uc: x-velocity on C-grid (input)
        vc: y-velocity on C-grid (input)
        dxc: grid spacing in x-dir (input)
        dyc: grid spacing in y-dir (input)
        vort_c: C-grid vorticity (output)
    """
    from __splitters__ import i_end, i_start, j_end, j_start

    with computation(PARALLEL), interval(...):
        fx = dxc * uc
        fy = dyc * vc

        vort_c = fx[0, -1, 0] - fx - fy[-1, 0, 0] + fy

        with parallel(region[i_start, j_start], region[i_start, j_end + 1]):
            vort_c += fy[-1, 0, 0]

        with parallel(region[i_end + 1, j_start], region[i_end + 1, j_end + 1]):
            vort_c -= fy[0, 0, 0]
