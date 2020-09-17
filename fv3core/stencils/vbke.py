#!/usr/bin/env python3
import fv3core.utils.gt4py_utils as utils
import gt4py.gtscript as gtscript
import fv3core._config as spec
from gt4py.gtscript import computation, interval, PARALLEL, parallel, region

sd = utils.sd


@utils.stencil()
def main_vb(vc: sd, uc: sd, cosa: sd, rsina: sd, vt: sd, vb: sd, dt4: float, dt5: float):
    from __splitters__ import i_start, i_end, j_start, j_end
    with computation(PARALLEL), interval(...):
        vb[0, 0, 0] = dt5 * (vc[-1, 0, 0] + vc - (uc[0, -1, 0] + uc) * cosa) * rsina
        with parallel(region[i_start, j_start:j_end], region[i_end, j_start:j_end]):
            vb[0, 0, 0] = dt4 * (-vt[-2, 0, 0] + 3.0 * (vt[-1, 0, 0] + vt) - vt[1, 0, 0])
        with parallel(region[i_start:i_end+1, j_start], region[i_start:i_end+1, j_end]):
            vb[0, 0, 0] = dt5 * (vt[-1, 0, 0] + vt)


def compute(uc, vc, vt, vb, dt5, dt4):
    grid = spec.grid
    splitters = {
        "i_start": 0,
        "i_end": grid.ie + 1 - grid.is_,
        "j_start": 0,
        "j_end": grid.je + 1 - grid.js,
    }
    if spec.namelist.grid_type < 3 and not grid.nested:
        main_vb(
            vc,
            uc,
            grid.cosa,
            grid.rsina,
            vt,
            vb,
            dt4=dt4,
            dt5=dt5,
            origin=(grid.is_, grid.js, 0),
            domain=(grid.ie - grid.is_ + 2, grid.je - grid.js + 2, grid.npz),
            splitters=splitters
        )
    else:
        # should be a stencil like vb = dt5 * (vc[-1, 0,0] + vc)
        raise Exception("unimplemented grid_type >= 3 or nested")
