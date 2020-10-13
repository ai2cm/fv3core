import gt4py as gt
import gt4py.gtscript as gtscript
import numpy as np
from gt4py.gtscript import PARALLEL, computation, interval, parallel, region

import fv3core._config as spec
import fv3core.stencils.basic_operations as basic
import fv3core.utils.gt4py_utils as utils
from fv3core.decorators import gtstencil


sd = utils.sd


@gtstencil()
def compute_diverg_d(
    u: sd,
    v: sd,
    ua: sd,
    va: sd,
    dxc: sd,
    dyc: sd,
    sin_sg1: sd,
    sin_sg2: sd,
    sin_sg3: sd,
    sin_sg4: sd,
    cos_sg1: sd,
    cos_sg2: sd,
    cos_sg3: sd,
    cos_sg4: sd,
    rarea_c: sd,
    divg_d: sd,
):
    from __splitters__ import i_end, i_start, j_end, j_start

    with computation(PARALLEL), interval(...):
        uf = (
            (u - 0.25 * (va[0, -1, 0] + va) * (cos_sg4[0, -1, 0] + cos_sg2))
            * dyc
            * 0.5
            * (sin_sg4[0, -1, 0] + sin_sg2)
        )
        with parallel(region[:, j_start], region[:, j_end + 1]):
            uf = u * dyc * 0.5 * (sin_sg4[0, -1, 0] + sin_sg2)

        vf = (
            (v - 0.25 * (ua[-1, 0, 0] + ua) * (cos_sg3[-1, 0, 0] + cos_sg1))
            * dxc
            * 0.5
            * (sin_sg3[-1, 0, 0] + sin_sg1)
        )
        with parallel(region[i_start, :], region[i_end + 1, :]):
            vf = v * dxc * 0.5 * (sin_sg3[-1, 0, 0] + sin_sg1)

        divg_d = vf[0, -1, 0] - vf + uf[-1, 0, 0] - uf
        with parallel(region[i_start, j_start], region[i_end + 1, j_start]):
            divg_d -= vf[0, -1, 0]
        with parallel(region[i_end + 1, j_end + 1], region[i_start, j_end + 1]):
            divg_d += vf
        divg_d *= rarea_c


def compute(u, v, ua, va, divg_d):
    grid = spec.grid

    compute_diverg_d(
        u,
        v,
        ua,
        va,
        grid.dxc,
        grid.dyc,
        grid.sin_sg1,
        grid.sin_sg2,
        grid.sin_sg3,
        grid.sin_sg4,
        grid.cos_sg1,
        grid.cos_sg2,
        grid.cos_sg3,
        grid.cos_sg4,
        grid.rarea_c,
        divg_d,
        origin=grid.compute_origin(),
        domain=(grid.nic + 1, grid.njc + 1, grid.npz),
    )
