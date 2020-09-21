#!/usr/bin/env python3
import fv3core.utils.gt4py_utils as utils
import gt4py.gtscript as gtscript
import fv3core._config as spec
from gt4py.gtscript import computation, interval, PARALLEL, parallel, region

sd = utils.sd


@utils.stencil()
def vbke(vc: sd, uc: sd, cosa: sd, rsina: sd, vt: sd, vb: sd, dt4: float, dt5: float):
    from __splitters__ import i_start, i_end, j_start, j_end

    with computation(PARALLEL), interval(...):
        vb[0, 0, 0] = dt5 * (vc[-1, 0, 0] + vc - (uc[0, -1, 0] + uc) * cosa) * rsina
        with parallel(region[i_start, :], region[i_end + 1, :]):
            vb[0, 0, 0] = dt4 * (
                -vt[-2, 0, 0] + 3.0 * (vt[-1, 0, 0] + vt) - vt[1, 0, 0]
            )
        with parallel(region[:, j_start], region[:, j_end + 1]):
            vb[0, 0, 0] = dt5 * (vt[-1, 0, 0] + vt)


def compute(uc, vc, vt, vb, dt5, dt4):
    grid = spec.grid
    if spec.namelist.grid_type < 3 and not grid.nested:
        vbke(
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
            splitters=grid.splitters,
        )
    else:
        # should be a stencil like vb = dt5 * (vc[-1, 0,0] + vc)
        raise Exception("unimplemented grid_type >= 3 or nested")
