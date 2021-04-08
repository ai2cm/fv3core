from gt4py.gtscript import PARALLEL, computation, interval

import fv3core._config as spec
import fv3core.utils.corners as corners
import fv3core.utils.gt4py_utils as utils
from fv3core.decorators import gtstencil
from fv3core.utils.typing import FloatField, FloatFieldIJ


#
# Flux value stencils
# ---------------------
@gtstencil()
def compute_zonal_flux(flux: FloatField, a_in: FloatField, del_term: FloatFieldIJ):
    with computation(PARALLEL), interval(...):
        flux = del_term * (a_in[-1, 0, 0] - a_in)


@gtstencil()
def compute_meridional_flux(flux: FloatField, a_in: FloatField, del_term: FloatFieldIJ):
    with computation(PARALLEL), interval(...):
        flux = del_term * (a_in[0, -1, 0] - a_in)


#
# Q update stencil
# ------------------
@gtstencil()
def update_q(
    q: FloatField, rarea: FloatFieldIJ, fx: FloatField, fy: FloatField, cd: float
):
    with computation(PARALLEL), interval(...):
        q = q + cd * rarea * (fx - fx[1, 0, 0] + fy - fy[0, 1, 0])


@gtstencil()
def copy_row(a: FloatField):
    with computation(PARALLEL), interval(...):
        a0 = a
        a = a0[1, 0, 0]


@gtstencil()
def copy_column(a: FloatField):
    with computation(PARALLEL), interval(...):
        a0 = a
        a = a0[0, 1, 0]


#
# corner_fill
#
# Subroutine that copies/fills in the appropriate corner values for qdel
# ------------------------------------------------------------------------
def corner_fill(grid, q):
    r3 = 1.0 / 3.0
    utils.device_sync()
    if grid.sw_corner:
        q[grid.is_, grid.js, :] = (
            q[grid.is_, grid.js, :]
            + q[grid.is_ - 1, grid.js, :]
            + q[grid.is_, grid.js - 1, :]
        ) * r3
        q[grid.is_ - 1, grid.js, :] = q[grid.is_, grid.js, :]
        q[grid.is_, grid.js - 1, :] = q[grid.is_, grid.js, :]
    if grid.se_corner:
        q[grid.ie, grid.js, :] = (
            q[grid.ie, grid.js, :]
            + q[grid.ie + 1, grid.js, :]
            + q[grid.ie, grid.js - 1, :]
        ) * r3
        q[grid.ie + 1, grid.js, :] = q[grid.ie, grid.js, :]
        copy_column(q, origin=(grid.ie, grid.js - 1, 0), domain=(1, 1, grid.npz))

    if grid.ne_corner:
        q[grid.ie, grid.je, :] = (
            q[grid.ie, grid.je, :]
            + q[grid.ie + 1, grid.je, :]
            + q[grid.ie, grid.je + 1, :]
        ) * r3
        q[grid.ie + 1, grid.je, :] = q[grid.ie, grid.je, :]
        q[grid.ie, grid.je + 1, :] = q[grid.ie, grid.je, :]

    if grid.nw_corner:
        q[grid.is_, grid.je, :] = (
            q[grid.is_, grid.je, :]
            + q[grid.is_ - 1, grid.je, :]
            + q[grid.is_, grid.je + 1, :]
        ) * r3
        copy_row(q, origin=(grid.is_ - 1, grid.je, 0), domain=(1, 1, grid.npz))
        q[grid.is_, grid.je + 1, :] = q[grid.is_, grid.je, :]

    return q


def compute(qdel: FloatField, nmax: int, cd: float, km: int):
    grid = spec.grid
    origin = (grid.isd, grid.jsd, 0)

    # Construct some necessary temporary storage objects
    fx = utils.make_storage_from_shape(
        qdel.shape, origin=origin, cache_key="del2cubed_fx", init=True
    )
    fy = utils.make_storage_from_shape(
        qdel.shape, origin=origin, cache_key="del2cubed_fy", init=True
    )

    # set up the temporal loop
    ntimes = min(3, nmax)
    for n in range(1, ntimes + 1):
        nt = ntimes - n
        origin = (grid.is_ - nt, grid.js - nt, 0)

        # Fill in appropriate corner values
        qdel = corner_fill(grid, qdel)

        if nt > 0:
            corners.copy_corners_x_stencil(
                qdel, origin=(grid.isd, grid.jsd, 0), domain=(grid.nid, grid.njd, km)
            )

        nx = grid.njc + 2 * nt + 1  # (grid.ie+nt+1) - (grid.is_-nt) + 1
        ny = grid.njc + 2 * nt  # (grid.je+nt) - (grid.js-nt) + 1
        compute_zonal_flux(fx, qdel, grid.del6_v, origin=origin, domain=(nx, ny, km))

        if nt > 0:
            corners.copy_corners_y_stencil(
                qdel, origin=(grid.isd, grid.jsd, 0), domain=(grid.nid, grid.njd, km)
            )

        nx = grid.nic + 2 * nt  # (grid.ie+nt) - (grid.is_-nt) + 1
        ny = grid.njc + 2 * nt + 1  # (grid.je+nt+1) - (grid.js-nt) + 1
        compute_meridional_flux(
            fy, qdel, grid.del6_u, origin=origin, domain=(nx, ny, km)
        )

        # Update q values
        ny = grid.njc + 2 * nt  # (grid.je+nt) - (grid.js-nt) + 1
        update_q(qdel, grid.rarea, fx, fy, cd, origin=origin, domain=(nx, ny, km))
