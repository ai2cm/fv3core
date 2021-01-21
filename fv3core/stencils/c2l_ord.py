from gt4py.gtscript import PARALLEL, computation, horizontal, interval, region

import fv3core._config as spec
import fv3core.utils.gt4py_utils as utils
from fv3core.decorators import gtstencil
from fv3core.utils.typing import FloatField
from fv3gfs.util import CubedSphereCommunicator


sd = utils.sd

C1 = 1.125
C2 = -0.125


@gtstencil()
def c2l_ord2(
    u: FloatField,
    v: FloatField,
    dx: FloatField,
    dy: FloatField,
    a11: FloatField,
    a12: FloatField,
    a21: FloatField,
    a22: FloatField,
    ua: FloatField,
    va: FloatField,
):
    with computation(PARALLEL), interval(...):
        wu = u * dx
        wv = v * dy
        # Co-variant vorticity-conserving interpolation
        u1 = 2.0 * (wu + wu[0, 1, 0]) / (dx + dx[0, 1, 0])
        v1 = 2.0 * (wv + wv[1, 0, 0]) / (dy + dy[1, 0, 0])
        # Cubed (cell center co-variant winds) to lat-lon
        ua = a11 * u1 + a12 * v1
        va = a21 * u1 + a22 * v1


@gtstencil()
def ord4_transform(
    u: FloatField,
    v: FloatField,
    a11: FloatField,
    a12: FloatField,
    a21: FloatField,
    a22: FloatField,
    dx: FloatField,
    dy: FloatField,
    ua: FloatField,
    va: FloatField,
):
    with computation(PARALLEL), interval(...):
        from __externals__ import i_end, i_start, j_end, j_start

        utmp = C2 * (u[0, -1, 0] + u[0, 2, 0]) + C1 * (u + u[0, 1, 0])
        vtmp = C2 * (v[-1, 0, 0] + v[2, 0, 0]) + C1 * (v + v[1, 0, 0])

        # south/north edge
        with horizontal(region[:, j_start], region[:, j_end]):
            vtmp = 2.0 * ((v * dy) + (v[1, 0, 0] * dy[1, 0, 0])) / (dy + dy[1, 0, 0])
            utmp = 2.0 * (u * dx + u[0, 1, 0] * dx[0, 1, 0]) / (dx + dx[0, 1, 0])

        # west/east edge
        with horizontal(region[i_start, :], region[i_end, :]):
            utmp = 2.0 * ((u * dx) + (u[0, 1, 0] * dx[0, 1, 0])) / (dx + dx[0, 1, 0])
            vtmp = 2.0 * ((v * dy) + (v[1, 0, 0] * dy[1, 0, 0])) / (dy + dy[1, 0, 0])

        # Transform local a-grid winds into latitude-longitude coordinates
        ua = a11 * utmp + a12 * vtmp
        va = a21 * utmp + a22 * vtmp


def compute_cubed_to_latlon(
    u: FloatField,
    v: FloatField,
    ua: FloatField,
    va: FloatField,
    comm: CubedSphereCommunicator,
    halo_update_uv: bool = True,
):
    """
    Interpolate D-grid winds to into a-grid winds at latitude-longitude.
    Args:
        u: x-wind on D-grid (in)
        v: y-wind on D-grid (in)
        ua: x-wind on A-grid (out)
        va: y-wind on A-grid (out)
        comm: Communicator in case of halo update (in)
        mode: If True, halo update before transforming to lat/lon
    """
    namelist = spec.namelist
    grid = spec.grid
    if namelist.c2l_ord == 2:
        c2l_ord2(
            u,
            v,
            grid.dx,
            grid.dy,
            grid.a11,
            grid.a12,
            grid.a21,
            grid.a22,
            ua,
            va,
            origin=grid.compute_origin(
                add=(-1, -1, 0) if halo_update_uv else (0, 0, 0)
            ),
            domain=grid.domain_shape_compute(
                add=(2, 2, 0) if halo_update_uv else (0, 0, 0)
            ),
        )
    else:
        if halo_update_uv:
            comm.vector_halo_update(u, v, n_points=utils.halo)
        ord4_transform(
            u.storage,
            v.storage,
            grid.a11,
            grid.a12,
            grid.a21,
            grid.a22,
            grid.dx,
            grid.dy,
            ua,
            va,
            origin=grid.compute_origin(),
            domain=grid.domain_shape_compute(),
        )
