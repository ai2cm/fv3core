from gt4py.gtscript import PARALLEL, computation, interval, parallel, region

import fv3core._config as spec
import fv3core.utils.gt4py_utils as utils
from fv3core.decorators import gtstencil


sd = utils.sd

C1 = 1.125
C2 = -0.125


@gtstencil()
def c2l_ord2(
    u: sd, v: sd, dx: sd, dy: sd, a11: sd, a12: sd, a21: sd, a22: sd, ua: sd, va: sd
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


def compute_ord2(u, v, ua, va, do_halo=False):
    grid = spec.grid
    i1 = grid.is_
    i2 = grid.ie
    j1 = grid.js
    j2 = grid.je
    # Usually used for nesting
    if do_halo:
        i1 -= -1
        i2 += 1
        j1 -= 1
        j2 += 1

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
        origin=(i1, j1, 0),
        domain=(i2 - i1 + 1, j2 - j1 + 1, grid.npz),
    )


@gtstencil()
def ord4_transform(
    u: sd, v: sd, a11: sd, a12: sd, a21: sd, a22: sd, dx: sd, dy: sd, ua: sd, va: sd
):
    with computation(PARALLEL), interval(...):
        from __externals__ import i_end, i_start, j_end, j_start

        utmp = C2 * (u[0, -1, 0] + u[0, 2, 0]) + C1 * (u + u[0, 1, 0])
        vtmp = C2 * (v[-1, 0, 0] + v[2, 0, 0]) + C1 * (v + v[1, 0, 0])

        # south/north edge
        with parallel(region[:, j_start], region[:, j_end]):
            vtmp = 2.0 * ((v * dy) + (v[1, 0, 0] * dy[1, 0, 0])) / (dy + dy[1, 0, 0])
            utmp = 2.0 * (u * dx + u[0, 1, 0] * dx[0, 1, 0]) / (dx + dx[0, 1, 0])

        # west/east edge
        with parallel(region[i_start, :], region[i_end, :]):
            utmp = 2.0 * ((u * dx) + (u[0, 1, 0] * dx[0, 1, 0])) / (dx + dx[0, 1, 0])
            vtmp = 2.0 * ((v * dy) + (v[1, 0, 0] * dy[1, 0, 0])) / (dy + dy[1, 0, 0])

        # Transform local a-grid winds into latitude-longitude coordinates
        ua = a11 * utmp + a12 * vtmp
        va = a21 * utmp + a22 * vtmp


def compute_cubed_to_latlon(u, v, ua, va, comm, mode=True):
    """
    Interpolate D-grid winds to into a-grid winds at latitude-longitude.
    Args:
        u: x-wind on D-grid (in)
        v: y-wind on D-grid (in)
        ua: x-wind on A-grid (out)
        va: y-wind on A-grid (out)
        comm: MPI communicator, in case of a halo update (in)
        mode: If True, halo update before transforming to lat/lon
    """
    if spec.namelist.c2l_ord == 2:
        compute_ord2(u, v, ua, va, False)
    else:
        grid = spec.grid
        if mode:
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
