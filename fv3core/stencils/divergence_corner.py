import gt4py as gt
import gt4py.gtscript as gtscript
import numpy as np

import fv3core._config as spec
import fv3core.stencils.basic_operations as basic
import fv3core.utils.gt4py_utils as utils
from fv3core.decorators import gtstencil


sd = utils.sd
origin = utils.origin

## Stencil Definitions
@gtstencil()
def compute_uf(
    uf: sd, u: sd, va: sd, cos_sg4: sd, cos_sg2: sd, dyc: sd, sin_sg4: sd, sin_sg2: sd
):
    from __splitters__ import j_end, j_start

    with computation(PARALLEL), interval(...):
        uf = (
            (u - 0.25 * (va[0, -1, 0] + va) * (cos_sg4[0, -1, 0] + cos_sg2))
            * dyc
            * 0.5
            * (sin_sg4[0, -1, 0] + sin_sg2)
        )
        with parallel(
            region[:, j_start : j_start + 1], region[:, j_end + 1 : j_end + 2]
        ):
            uf = u * dyc * 0.5 * (sin_sg4[0, -1, 0] + sin_sg2)


@gtstencil()
def compute_vf(
    vf: sd, v: sd, ua: sd, cos_sg3: sd, cos_sg1: sd, dxc: sd, sin_sg3: sd, sin_sg1: sd
):
    from __splitters__ import i_end, i_start

    with computation(PARALLEL), interval(...):
        vf = (
            (v - 0.25 * (ua[-1, 0, 0] + ua) * (cos_sg3[-1, 0, 0] + cos_sg1))
            * dxc
            * 0.5
            * (sin_sg3[-1, 0, 0] + sin_sg1)
        )
        with parallel(region[i_start : i_start + 1, :], region[i_end : i_end + 1, :]):
            vf = v * dxc * 0.5 * (sin_sg3[-1, 0, 0] + sin_sg1)


@gtstencil()
def compute_diverg_d(div: sd, vf: sd, uf: sd, rarea_c: sd):
    from __splitters__ import i_end, i_start, j_end, j_start

    with computation(PARALLEL), interval(...):
        div = vf[0, -1, 0] - vf + uf[-1, 0, 0] - uf
        with parallel(
            region[i_start : i_start + 1, j_start : j_start + 1, :],
            region[i_end : i_end + 1, j_start : j_start + 1, :],
        ):
            div = div - vf[0, -1, 0]
        with parallel(
            region[i_end : i_end + 1, j_end : j_end + 1],
            region[i_start : i_start + 1, j_end : j_end + 1],
        ):
            div = div + vf
        div = rarea_c * div


def compute(u, v, ua, va, divg_d):
    grid = spec.grid
    is2 = grid.is_
    ie1 = grid.ie + 1
    # Create storage objects for the temporary velocity arrays, uf and vf
    uf = utils.make_storage_from_shape(ua.shape, origin=(grid.is_ - 2, grid.js - 1, 0))
    vf = utils.make_storage_from_shape(va.shape, origin=(grid.is_ - 1, grid.js - 2, 0))

    # Compute values for the temporary velocity arrays
    i_start = grid.global_is - grid.is_ if grid.west_edge else 10000
    i_end = grid.ie - grid.is_ + 1 if grid.east_edge else -10000
    j_start = grid.js - grid.global_js if grid.south_edge else 10000
    j_end = grid.je - grid.js if grid.north_edge else -10000

    compute_uf(
        uf,
        u,
        va,
        grid.cos_sg4,
        grid.cos_sg2,
        grid.dyc,
        grid.sin_sg4,
        grid.sin_sg2,
        origin=(grid.is_ - 1, grid.js, 0),
        domain=(grid.nic + 2, grid.njc + 1, grid.npz),
        splitters={"j_start": j_start, "j_end": j_end},
    )

    compute_vf(
        vf,
        v,
        ua,
        grid.cos_sg3,
        grid.cos_sg1,
        grid.dxc,
        grid.sin_sg3,
        grid.sin_sg1,
        origin=(is2, grid.js - 1, 0),
        domain=(ie1 - is2 + 1, grid.njc + 2, grid.npz),
        splitters={"i_start": i_start, "i_end": i_end},
    )

    # Compute the divergence tensor values
    if grid.ne_corner or grid.nw_corner:
        j_end = grid.je - grid.js + 1
    compute_diverg_d(
        divg_d,
        vf,
        uf,
        grid.rarea_c,
        origin=(grid.is_, grid.js, 0),
        domain=(grid.nic + 1, grid.njc + 1, grid.npz),
        splitters={
            "i_start": i_start,
            "i_end": i_end,
            "j_start": j_start,
            "j_end": j_end,
        },
    )
