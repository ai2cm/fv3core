import gt4py as gt
import gt4py.gtscript as gtscript
import numpy as np

import fv3core._config as spec
import fv3core.utils.gt4py_utils as utils
from fv3core.decorators import gtstencil
from fv3core.stencils.a2b_ord4 import a1, a2, lagrange_x_func, lagrange_y_func
from fv3core.utils.corners import fill_4corners_x, fill_4corners_y


sd = utils.sd
c1 = -2.0 / 14.0
c2 = 11.0 / 14.0
c3 = 5.0 / 14.0
OFFSET = 2


@gtscript.function
def contravariant(u, v, cosa, rsin):
    return (u - v * cosa) * rsin


@gtscript.function
def vol_conserv_cubic_interp_func_x(u):
    return c1 * u[-2, 0, 0] + c2 * u[-1, 0, 0] + c3 * u


@gtscript.function
def vol_conserv_cubic_interp_func_x_rev(u):
    return c1 * u[1, 0, 0] + c2 * u + c3 * u[-1, 0, 0]


@gtscript.function
def vol_conserv_cubic_interp_func_y(v):
    return c1 * v[0, -2, 0] + c2 * v[0, -1, 0] + c3 * v


@gtscript.function
def vol_conserv_cubic_interp_func_y_rev(v):
    return c1 * v[0, 1, 0] + c2 * v + c3 * v[0, -1, 0]


@gtstencil()
def vol_conserv_cubic_interp_x(utmp: sd, uc: sd):
    with computation(PARALLEL), interval(...):
        uc = vol_conserv_cubic_interp_func_x(utmp)


@gtstencil()
def vol_conserv_cubic_interp_x_rev(utmp: sd, uc: sd):
    with computation(PARALLEL), interval(...):
        uc = vol_conserv_cubic_interp_func_x_rev(utmp)


@gtstencil()
def vol_conserv_cubic_interp_y(vtmp: sd, vc: sd):
    with computation(PARALLEL), interval(...):
        vc = vol_conserv_cubic_interp_func_y(vtmp)


@gtstencil()
def lagrange_interpolation_x(u: sd, utmp: sd):
    with computation(PARALLEL), interval(...):
        utmp = a2 * (u[0, -1, 0] + u[0, 2, 0]) + a1 * (u + u[0, 1, 0])


@gtstencil()
def lagrange_interpolation_y(v: sd, vtmp: sd):
    with computation(PARALLEL), interval(...):
        vtmp = a2 * (v[-1, 0, 0] + v[2, 0, 0]) + a1 * (v + v[1, 0, 0])


@gtstencil(externals={"HALO": 3})
def d2a2c_stencil1(
    u: sd,
    v: sd,
    cosa_s: sd,
    rsin2: sd,
    utmp: sd,
    vtmp: sd,
    ua: sd,
    va: sd,
):
    # in: u, v, cosa_s, rsin2
    # inout: utmp, vtmp, ua, va
    from __externals__ import HALO, namelist
    from __splitters__ import i_end, i_start, j_end, j_start

    with computation(PARALLEL), interval(...):

        assert __INLINED(namelist.grid_type < 3)

        # The order of these blocks matters, so they cannot be merged into a
        # single block since then the order is not guaranteed
        with parallel(region[:, : j_start + HALO]):
            utmp = 0.5 * (u + u[0, 1, 0])
            vtmp = 0.5 * (v + v[1, 0, 0])
        with parallel(region[:, j_end - HALO + 1 :]):
            utmp = 0.5 * (u + u[0, 1, 0])
            vtmp = 0.5 * (v + v[1, 0, 0])
        with parallel(region[: i_start + HALO, :]):
            utmp = 0.5 * (u + u[0, 1, 0])
            vtmp = 0.5 * (v + v[1, 0, 0])
        with parallel(region[i_end - HALO + 1 :, :]):
            utmp = 0.5 * (u + u[0, 1, 0])
            vtmp = 0.5 * (v + v[1, 0, 0])

        ua = contravariant(utmp, vtmp, cosa_s, rsin2)
        va = contravariant(vtmp, utmp, cosa_s, rsin2)

        # SW corner
        with parallel(region[i_start - 3, j_start - 1]):
            utmp = -vtmp[2, 3, 0]
        with parallel(region[i_start - 2, j_start - 1]):
            utmp = -vtmp[1, 2, 0]
        with parallel(region[i_start - 1, j_start - 1]):
            utmp = -vtmp[0, 1, 0]

        # SE corner
        with parallel(region[i_end + 1, j_start - 1]):
            utmp = vtmp[0, 1, 0]
        with parallel(region[i_end + 2, j_start - 1]):
            utmp = vtmp[-1, 2, 0]
        with parallel(region[i_end + 3, j_start - 1]):
            utmp = vtmp[-2, 3, 0]

        # NE corner
        with parallel(region[i_end + 1, j_end + 1]):
            utmp = -vtmp[0, -1, 0]
        with parallel(region[i_end + 2, j_end + 1]):
            utmp = -vtmp[-1, -2, 0]
        with parallel(region[i_end + 3, j_end + 1]):
            utmp = -vtmp[-2, -3, 0]

        # NW corner
        with parallel(region[i_start - 3, j_end + 1]):
            utmp = vtmp[2, -3, 0]
        with parallel(region[i_start - 2, j_end + 1]):
            utmp = vtmp[1, -2, 0]
        with parallel(region[i_start - 1, j_end + 1]):
            utmp = vtmp[0, -1, 0]

        ua = fill_4corners_x(ua, va, sw_mult=-1, se_mult=1, ne_mult=-1, nw_mult=1)


@gtstencil()
def d2a2c_stencil2(utmp: sd, v: sd, cosa_u: sd, rsin_u: sd, uc: sd, utc: sd):
    # in: utmp, v, cosa_u, rsin_u
    # inout: uc
    # out: utc
    from __splitters__ import i_end, i_start, j_end, j_start

    with computation(PARALLEL), interval(...):
        uc = lagrange_x_func(utmp)
        utc = contravariant(uc, v, cosa_u, rsin_u)


@gtstencil()
def d2a2c_stencil_west(
    utmp: sd,
    ua: sd,
    v: sd,
    cosa_u: sd,
    rsin_u: sd,
    dxa: sd,
    sin_sg1: sd,
    sin_sg3: sd,
    uc: sd,
    utc: sd,
):
    # in: utmp, ua, v, cosa_u, rsin_u, dxa, sin_sg1, sin_sg3
    # inout: uc, utc
    from __splitters__ import i_end, i_start, j_end, j_start

    with computation(PARALLEL), interval(...):
        # West
        with parallel(region[i_start - 1, :]):
            uc = vol_conserv_cubic_interp_func_x(utmp)

        with parallel(region[i_start, :]):
            t1 = dxa[-2, 0, 0] + dxa[-1, 0, 0]
            t2 = dxa[0, 0, 0] + dxa[1, 0, 0]
            n1 = (t1 + dxa[-1, 0, 0]) * ua[-1, 0, 0] - dxa[-1, 0, 0] * ua[-2, 0, 0]
            n2 = (t1 + dxa[0, 0, 0]) * ua[0, 0, 0] - dxa[0, 0, 0] * ua[1, 0, 0]
            utc = 0.5 * (n1 / t1 + n2 / t2)

        with parallel(region[i_start, :]):
            uc = utc * sin_sg3[-1, 0, 0] if utc > 0 else utc * sin_sg1

        with parallel(region[i_start + 1, :]):
            uc = vol_conserv_cubic_interp_func_x_rev(utmp)

        with parallel(region[i_start - 1, :]):
            utc = contravariant(uc, v, cosa_u, rsin_u)

        with parallel(region[i_start + 1, :]):
            utc = contravariant(uc, v, cosa_u, rsin_u)


@gtstencil()
def d2a2c_stencil_east(
    utmp: sd,
    ua: sd,
    v: sd,
    cosa_u: sd,
    rsin_u: sd,
    dxa: sd,
    sin_sg1: sd,
    sin_sg3: sd,
    uc: sd,
    utc: sd,
):
    # in: utmp, ua, v, cosa_u, rsin_u, dxa, sin_sg1, sin_sg3
    # inout: uc, utc
    from __splitters__ import i_end, i_start, j_end, j_start

    with computation(PARALLEL), interval(...):
        # East
        with parallel(region[i_end, :]):
            uc = vol_conserv_cubic_interp_func_x(utmp)

        with parallel(region[i_end + 1, j_start - 1 : j_end + 2]):
            t1 = dxa[-2, 0, 0] + dxa[-1, 0, 0]
            t2 = dxa[0, 0, 0] + dxa[1, 0, 0]
            n1 = (t1 + dxa[-1, 0, 0]) * ua[-1, 0, 0] - dxa[-1, 0, 0] * ua[-2, 0, 0]
            n2 = (t1 + dxa[0, 0, 0]) * ua[0, 0, 0] - dxa[0, 0, 0] * ua[1, 0, 0]
            utc = 0.5 * (n1 / t1 + n2 / t2)

        with parallel(region[i_end + 1, :]):
            uc = utc * sin_sg3[-1, 0, 0] if utc > 0 else utc * sin_sg1

        with parallel(region[i_end + 2, :]):
            uc = vol_conserv_cubic_interp_func_x_rev(utmp)

        with parallel(region[i_end, :]):
            utc = contravariant(uc, v, cosa_u, rsin_u)

        with parallel(region[i_end + 2, :]):
            utc = contravariant(uc, v, cosa_u, rsin_u)


@gtstencil(externals={"HALO": 3})
def d2a2c_stencil3(
    utmp: sd,
    ua: sd,
    va: sd,
    vtmp: sd,
):
    # in: utmp, ua
    # input: va
    # out: vtmp
    from __externals__ import HALO, namelist
    from __splitters__ import i_end, i_start, j_end, j_start

    with computation(PARALLEL), interval(...):

        assert __INLINED(namelist.grid_type < 3)

        # SW corner
        with parallel(region[i_start - 1, j_start - 3]):
            vtmp = -utmp[3, 2, 0]
        with parallel(region[i_start - 1, j_start - 2]):
            vtmp = -utmp[2, 1, 0]
        with parallel(region[i_start - 1, j_start - 1]):
            vtmp = -utmp[1, 0, 0]

        # SE corner
        with parallel(region[i_end + 1, j_start - 3]):
            vtmp = utmp[-3, 2, 0]
        with parallel(region[i_end + 1, j_start - 2]):
            vtmp = utmp[-2, 1, 0]
        with parallel(region[i_end + 1, j_start - 1]):
            vtmp = utmp[-1, 0, 0]

        # NE corner
        with parallel(region[i_end + 1, j_end + 1]):
            vtmp = -utmp[-1, 0, 0]
        with parallel(region[i_end + 1, j_end + 2]):
            vtmp = -utmp[-2, -1, 0]
        with parallel(region[i_end + 1, j_end + 3]):
            vtmp = -utmp[-3, -2, 0]

        # NW corner
        with parallel(region[i_start - 1, j_end + 1]):
            vtmp = utmp[1, 0, 0]
        with parallel(region[i_start - 1, j_end + 2]):
            vtmp = utmp[2, -1, 0]
        with parallel(region[i_start - 1, j_end + 3]):
            vtmp = utmp[3, -2, 0]

        va = fill_4corners_y(va, ua, sw_mult=-1, se_mult=1, ne_mult=-1, nw_mult=1)


@gtstencil()
def d2a2c_stencil_south(
    vtmp: sd,
    va: sd,
    u: sd,
    cosa_v: sd,
    rsin_v: sd,
    dya: sd,
    sin_sg2: sd,
    sin_sg4: sd,
    vc: sd,
    vtc: sd,
):
    # in: vtmp, va, u, cosa_v, rsin_v, dya, sin_sg2, sin_sg4
    # inout: vc, vtc
    from __splitters__ import i_end, i_start, j_end, j_start

    with computation(PARALLEL), interval(...):
        with parallel(region[:, j_start - 1]):
            vc = vol_conserv_cubic_interp_func_y(vtmp)
            vtc = contravariant(vc, u, cosa_v, rsin_v)

        with parallel(region[:, j_start]):
            t1 = dya[0, -2, 0] + dya[0, -1, 0]
            t2 = dya[0, 0, 0] + dya[0, 1, 0]
            n1 = (t1 + dya[0, -1, 0]) * va[0, -1, 0] - dya[0, -1, 0] * va[0, -2, 0]
            n2 = (t1 + dya[0, 0, 0]) * va[0, 0, 0] - dya[0, 0, 0] * va[0, 1, 0]
            vtc = 0.5 * (n1 / t1 + n2 / t2)

        with parallel(region[:, j_start]):
            vc = vtc * sin_sg4[0, -1, 0] if vtc > 0 else vtc * sin_sg2

        with parallel(region[:, j_start + 1]):
            vc = vol_conserv_cubic_interp_func_y_rev(vtmp)
            vtc = contravariant(vc, u, cosa_v, rsin_v)


@gtstencil()
def d2a2c_stencil_north(
    vtmp: sd,
    va: sd,
    u: sd,
    cosa_v: sd,
    rsin_v: sd,
    dya: sd,
    sin_sg2: sd,
    sin_sg4: sd,
    vc: sd,
    vtc: sd,
):
    # in: vtmp, va, u, cosa_v, rsin_v, dya, sin_sg2, sin_sg4
    # inout: vc, vtc
    from __splitters__ import i_end, i_start, j_end, j_start

    with computation(PARALLEL), interval(...):
        # NOTE: vtc can be a new temp here
        with parallel(region[:, j_end]):
            vc = vol_conserv_cubic_interp_func_y(vtmp)
            vtc = contravariant(vc, u, cosa_v, rsin_v)

        with parallel(region[:, j_end + 1]):
            t1 = dya[0, -2, 0] + dya[0, -1, 0]
            t2 = dya[0, 0, 0] + dya[0, 1, 0]
            n1 = (t1 + dya[0, -1, 0]) * va[0, -1, 0] - dya[0, -1, 0] * va[0, -2, 0]
            n2 = (t1 + dya[0, 0, 0]) * va[0, 0, 0] - dya[0, 0, 0] * va[0, 1, 0]
            vtc = 0.5 * (n1 / t1 + n2 / t2)

        with parallel(region[:, j_end + 1]):
            vc = vtc * sin_sg4[0, -1, 0] if vtc > 0 else vtc * sin_sg2

        with parallel(region[:, j_end + 2]):
            vc = vol_conserv_cubic_interp_func_y_rev(vtmp)
            vtc = contravariant(vc, u, cosa_v, rsin_v)


@gtstencil()
def d2a2c_stencil4(
    vtmp: sd,
    u: sd,
    cosa_v: sd,
    rsin_v: sd,
    vc: sd,
    vtc: sd,
):
    # in: vtmp, u, cosa_v, rsin_v
    # inout: vc
    # out: vtc
    with computation(PARALLEL), interval(...):
        vc = lagrange_y_func(vtmp)
        vtc = contravariant(vc, u, cosa_v, rsin_v)


def compute(dord4, uc, vc, u, v, ua, va, utc, vtc):
    grid = spec.grid
    big_number = 1e30  # 1e8 if 32 bit
    nx = grid.ie + 1  # grid.npx + 2
    ny = grid.je + 1  # grid.npy + 2
    i1 = grid.is_ - 1
    j1 = grid.js - 1
    id_ = 1 if dord4 else 0
    npt = 4 if spec.namelist.grid_type < 3 and not grid.nested else 0
    if npt > grid.nic - 1 or npt > grid.njc - 1:
        npt = 0
    utmp = utils.make_storage_from_shape(ua.shape, grid.default_origin())
    vtmp = utils.make_storage_from_shape(va.shape, grid.default_origin())
    utmp[:] = big_number
    vtmp[:] = big_number
    js1 = npt + OFFSET if grid.south_edge else grid.js - 1
    je1 = ny - npt if grid.north_edge else grid.je + 1
    is1 = npt + OFFSET if grid.west_edge else grid.isd
    ie1 = nx - npt if grid.east_edge else grid.ied

    is1 = npt + OFFSET if grid.west_edge else grid.is_ - 1
    ie1 = nx - npt if grid.east_edge else grid.ie + 1
    js1 = npt + OFFSET if grid.south_edge else grid.jsd
    je1 = ny - npt if grid.north_edge else grid.jed

    js2 = npt + OFFSET if grid.south_edge else grid.jsd
    je2 = ny - npt if grid.north_edge else grid.jed
    jdiff = je2 - js2 + 1
    pad = 2 + 2 * id_
    ifirst = grid.is_ + 2 if grid.west_edge else grid.is_ - 1
    ilast = grid.ie - 1 if grid.east_edge else grid.ie + 2
    idiff = ilast - ifirst + 1

    lagrange_interpolation_x(
        u,
        utmp,
        origin=(grid.is_ - 1, grid.js, 0),
        # domain=(ie1 - is1 + 1, je1 - js1 + 1, grid.npz),
    )
    lagrange_interpolation_y(
        v,
        vtmp,
        origin=(grid.is_, grid.js - 1, 0),
        # domain=(ie1 - is1 + 1, je1 - js1 + 1, grid.npz),
    )

    d2a2c_stencil1(
        u,
        v,
        grid.cosa_s,
        grid.rsin2,
        utmp,
        vtmp,
        ua,
        va,
        origin=(grid.is_ - 3, grid.js - 3, 0),
        domain=(grid.nic + 6, grid.njc + 6, grid.npz),
    )

    d2a2c_stencil2(
        utmp,
        v,
        grid.cosa_u,
        grid.rsin_u,
        uc,
        utc,
        origin=(i1, j1, 0),
        domain=(grid.nic + 2, grid.njc + 2, grid.npz),
    )

    d2a2c_stencil_west(
        utmp,
        ua,
        v,
        grid.cosa_u,
        grid.rsin_u,
        grid.dxa,
        grid.sin_sg1,
        grid.sin_sg3,
        uc,
        utc,
        origin=(grid.is_ - 1, grid.js - 1, 0),
        domain=(4, grid.njc + 2, grid.npz),
    )

    d2a2c_stencil_east(
        utmp,
        ua,
        v,
        grid.cosa_u,
        grid.rsin_u,
        grid.dxa,
        grid.sin_sg1,
        grid.sin_sg3,
        uc,
        utc,
        origin=(grid.ie - 1, grid.js - 1, 0),
        domain=(4, grid.njc + 2, grid.npz),
    )

    d2a2c_stencil3(
        utmp,
        ua,
        va,
        vtmp,
        origin=(grid.is_ - 3, grid.js - 3, 0),
        domain=(grid.nic + 6, grid.njc + 6, grid.npz),
    )

    d2a2c_stencil_south(
        vtmp,
        va,
        u,
        grid.cosa_v,
        grid.rsin_v,
        grid.dya,
        grid.sin_sg2,
        grid.sin_sg4,
        vc,
        vtc,
        origin=(grid.is_ - 1, grid.js - 1, 0),
        domain=(grid.nic + 2, 4, grid.npz),
    )

    d2a2c_stencil_north(
        vtmp,
        va,
        u,
        grid.cosa_v,
        grid.rsin_v,
        grid.dya,
        grid.sin_sg2,
        grid.sin_sg4,
        vc,
        vtc,
        origin=(grid.js - 1, grid.ie - 1, 0),
        domain=(grid.nic + 2, 4, grid.npz),
    )

    jfirst = grid.js + 2 if grid.south_edge else grid.js - 1
    jlast = grid.je - 1 if grid.north_edge else grid.je + 2
    jdiff = jlast - jfirst + 1

    d2a2c_stencil4(
        vtmp,
        u,
        grid.cosa_v,
        grid.rsin_v,
        vc,
        vtc,
        origin=(i1, jfirst, 0),
        domain=(grid.nic + 2, jdiff, grid.npz),
    )
