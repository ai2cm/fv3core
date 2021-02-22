import gt4py
import gt4py.gtscript as gtscript
from gt4py.gtscript import (
    __INLINED,
    PARALLEL,
    computation,
    horizontal,
    interval,
    region,
)

import fv3core._config as spec
import fv3core.utils.corners as corners
import fv3core.utils.gt4py_utils as utils
from fv3core.decorators import gtstencil
from fv3core.utils import global_config
from fv3core.utils.typing import Float, FloatField, FloatFieldIJ


sd = utils.sd
origin = utils.origin


#
# Flux value
# ----------
@gtscript.function
def compute_zonal_flux(A_in, del_term):
    return del_term * (A_in[-1, 0, 0] - A_in)


@gtscript.function
def compute_meridional_flux(A_in, del_term):
    return del_term * (A_in[0, -1, 0] - A_in)


#
# Q update
# --------
@gtscript.function
def update_q(q, rarea, fx, fy, cd):
    return q + cd * rarea * (fx[0, 0, 0] - fx[1, 0, 0] + fy[0, 0, 0] - fy[0, 1, 0])


@gtscript.function
def corner_fill(q):
    from __externals__ import i_end, i_start, j_end, j_start

    # Fills the same scalar value into three locations in q for each corner

    with horizontal(region[i_start, j_start]):
        q = (q[0, 0, 0] + q[-1, 0, 0] + q[0, -1, 0]) / 3.0
    with horizontal(region[i_start - 1, j_start]):
        q = q[1, 0, 0]
    with horizontal(region[i_start, j_start - 1]):
        q = q[0, 1, 0]

    with horizontal(region[i_end, j_start]):
        q = (q[0, 0, 0] + q[1, 0, 0] + q[0, -1, 0]) / 3.0
    with horizontal(region[i_end + 1, j_start]):
        q = q[-1, 0, 0]
    with horizontal(region[i_end, j_start - 1]):
        q = q[0, 1, 0]

    with horizontal(region[i_end, j_end]):
        q = (q[0, 0, 0] + q[1, 0, 0] + q[0, 1, 0]) / 3.0
    with horizontal(region[i_end + 1, j_end]):
        q = q[-1, 0, 0]
    with horizontal(region[i_end, j_end + 1]):
        q = q[0, -1, 0]

    with horizontal(region[i_start, j_end]):
        q = (q[0, 0, 0] + q[-1, 0, 0] + q[0, 1, 0]) / 3.0
    with horizontal(region[i_start - 1, j_end]):
        q = q[1, 0, 0]
    with horizontal(region[i_start, j_end + 1]):
        q = q[0, -1, 0]

    return q


def _del2cubed_loop(
    qdel: FloatField,
    del6_u: FloatFieldIJ,
    del6_v: FloatFieldIJ,
    rarea: FloatFieldIJ,
    cd: Float,
):
    from __externals__ import nt

    with computation(PARALLEL), interval(...):
        qdel = corner_fill(qdel)

        if __INLINED(nt > 0):
            qdel = corners.copy_corners_x(qdel)
        fx = compute_zonal_flux(qdel, del6_v)

        if __INLINED(nt > 0):
            qdel = corners.copy_corners_y(qdel)
        fy = compute_meridional_flux(qdel, del6_u)

        qdel = update_q(qdel, rarea, fx, fy, cd)


def _make_grid_storage_2d(grid_array: gt4py.storage.Storage, index: int = 0):
    grid = spec.grid
    return gt4py.storage.from_array(
        grid_array[:, :, index],
        backend=global_config.get_backend(),
        default_origin=grid.compute_origin()[:-1],
        shape=grid_array[:, :, index].shape,
        dtype=grid_array.dtype,
        mask=(True, True, False),
    )


def compute(qdel: FloatField, nmax: int, cd: Float, km: int):
    """
    Compute nested Laplacians

    Args:
        qdel: Quantity to compute Laplacian (inout)
        nmax: Number of times to apply the Laplacian operator
        cd: Courant number on D-grid???
        km: Number of vertical levels
    Grid variable inputs:
        del6_u, del6_v, rarea
    """
    grid = spec.grid

    del6_u = _make_grid_storage_2d(grid.del6_u)
    del6_v = _make_grid_storage_2d(grid.del6_v)
    rarea = _make_grid_storage_2d(grid.rarea)

    ntimes = min(3, nmax)
    for n in range(1, ntimes + 1):
        nt = ntimes - n
        stencil = gtstencil(definition=_del2cubed_loop, externals={"nt": nt})

        stencil(
            qdel,
            del6_u,
            del6_v,
            rarea,
            cd,
            origin=(grid.is_ - nt, grid.js - nt, 0),
            domain=(grid.nic + 2 * nt, grid.njc + 2 * nt, km),
        )
