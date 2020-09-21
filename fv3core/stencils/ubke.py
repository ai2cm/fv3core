#!/usr/bin/env python3
import fv3core.utils.gt4py_utils as utils
import gt4py.gtscript as gtscript
import fv3core._config as spec
from gt4py.gtscript import computation, interval, PARALLEL, parallel, region

sd = utils.sd


# TODO: merge with vbke?
@utils.stencil()
def main_ub(uc: sd, vc: sd, cosa: sd, rsina: sd, ut:sd, ub: sd, *, dt4: float, dt5: float):
    from __splitters__ import i_start, i_end, j_start, j_end
    with computation(PARALLEL), interval(...):
        ub[0, 0, 0] = dt5 * (uc[0, -1, 0] + uc - (vc[-1, 0, 0] + vc) * cosa) * rsina
        with parallel(region[:, j_start], region[:, j_end]):
            ub[0, 0, 0] = dt4 * (-ut[0, -2, 0] + 3.0 * (ut[0, -1, 0] + ut) - ut[0, 1, 0])
        with parallel(region[i_start, :], region[i_end, :]):
            ub[0, 0, 0] = dt5 * (ut[0, -1, 0] + ut)


def compute(uc, vc, ut, ub, dt5, dt4):
    grid = spec.grid
    splitters = {
        "i_start": 0 if grid.west_edge else -2,
        "i_end": grid.ie + 1 - grid.is_ if grid.east_edge else grid.ie + 1 - grid.is_ + 2,
        "j_start": 0 if grid.south_edge else -2,
        "j_end": grid.je + 1 - grid.js if grid.north_edge else grid.je + 1 - grid.js + 2,
    }
    if spec.namelist.grid_type < 3 and not grid.nested:
        main_ub(
            uc,
            vc,
            grid.cosa,
            grid.rsina,
            ut,
            ub,
            dt4=dt4,
            dt5=dt5,
            origin=(grid.is_, grid.js, 0),
            domain=(grid.ie - grid.is_ + 2, grid.je - grid.js + 2, grid.npz),
            splitters=splitters
        )
    else:
        # should be a stencil like vb = dt5 * (vc[-1, 0,0] + vc)
        raise Exception("unimplemented grid_type >= 3 or nested")
