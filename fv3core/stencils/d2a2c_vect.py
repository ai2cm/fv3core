import gt4py.gtscript as gtscript
from gt4py.gtscript import (
    __INLINED,
    PARALLEL,
    computation,
    horizontal,
    interval,
    region,
)

import fv3core.utils.gt4py_utils as utils
from fv3core.decorators import gtstencil
from fv3core.stencils.a2b_ord4 import a1, a2, lagrange_x_func, lagrange_y_func
from fv3core.utils.corners import (
    fill2_4corners_x,
    fill2_4corners_y,
    fill3_4corners_x,
    fill3_4corners_y,
)


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


@gtscript.function
def lagrange_y_func_p1(qx):
    return a2 * (qx[0, -1, 0] + qx[0, 2, 0]) + a1 * (qx + qx[0, 1, 0])


@gtstencil()
def lagrange_interpolation_y_p1(qx: sd, qout: sd):
    with computation(PARALLEL), interval(...):
        qout = lagrange_y_func_p1(qx)


@gtscript.function
def lagrange_x_func_p1(qy):
    return a2 * (qy[-1, 0, 0] + qy[2, 0, 0]) + a1 * (qy + qy[1, 0, 0])


@gtstencil()
def lagrange_interpolation_x_p1(qy: sd, qout: sd):
    with computation(PARALLEL), interval(...):
        qout = lagrange_x_func_p1(qy)


def interp_winds_d_to_a(u, v):
    from __externals__ import (
        HALO,
        i_end,
        i_start,
        j_end,
        j_start,
        local_ie,
        local_is,
        local_je,
        local_js,
    )

    with horizontal(region[:, local_js - 2 : local_je + 2]):
        utmp = lagrange_y_func_p1(u)
    with horizontal(region[local_is - 2 : local_ie + 2, :]):
        vtmp = lagrange_x_func_p1(v)

    with horizontal(
        region[:, : j_start + HALO],
        region[:, j_end - HALO + 1 :],
        region[: i_start + HALO, :],
        region[i_end - HALO + 1 :, :],
    ):
        utmp = 0.5 * (u + u[0, 1, 0])
        vtmp = 0.5 * (v + v[1, 0, 0])

    return utmp, vtmp


def edge_interpolate4_x(ua, dxa):
    t1 = dxa[-2, 0, 0] + dxa[-1, 0, 0]
    t2 = dxa[0, 0, 0] + dxa[1, 0, 0]
    n1 = (t1 + dxa[-1, 0, 0]) * ua[-1, 0, 0] - dxa[-1, 0, 0] * ua[-2, 0, 0]
    n2 = (t1 + dxa[0, 0, 0]) * ua[0, 0, 0] - dxa[0, 0, 0] * ua[1, 0, 0]
    return 0.5 * (n1 / t1 + n2 / t2)


def edge_interpolate4_y(va, dya):
    t1 = dya[0, -2, 0] + dya[0, -1, 0]
    t2 = dya[0, 0, 0] + dya[0, 1, 0]
    n1 = (t1 + dya[0, -1, 0]) * va[0, -1, 0] - dya[0, -1, 0] * va[0, -2, 0]
    n2 = (t1 + dya[0, 0, 0]) * va[0, 0, 0] - dya[0, 0, 0] * va[0, 1, 0]
    return 0.5 * (n1 / t1 + n2 / t2)


@gtstencil(externals={"HALO": 3})
def d2a2c_vect(
    cosa_s: sd,
    cosa_u: sd,
    cosa_v: sd,
    dxa: sd,
    dya: sd,
    rsin2: sd,
    rsin_u: sd,
    rsin_v: sd,
    sin_sg1: sd,
    sin_sg2: sd,
    sin_sg3: sd,
    sin_sg4: sd,
    u: sd,
    ua: sd,
    uc: sd,
    utc: sd,
    v: sd,
    va: sd,
    vc: sd,
    vtc: sd,
):
    from __externals__ import (
        i_end,
        i_start,
        j_end,
        j_start,
        local_ie,
        local_is,
        local_je,
        local_js,
        namelist,
    )

    with computation(PARALLEL), interval(...):

        utmp, vtmp = interp_winds_d_to_a(u, v)

        with horizontal(
            region[local_is - 2 : local_ie + 3, local_js - 2 : local_je + 3]
        ):
            ua = contravariant(utmp, vtmp, cosa_s, rsin2)
            va = contravariant(vtmp, utmp, cosa_s, rsin2)

        # A -> C
        # Fix the edges
        utmp = fill3_4corners_x(
            utmp, vtmp, sw_mult=-1, se_mult=1, ne_mult=-1, nw_mult=1
        )
        ua = fill2_4corners_x(ua, va, sw_mult=-1, se_mult=1, ne_mult=-1, nw_mult=1)

    # X
    with computation(PARALLEL), interval(...):

        with horizontal(
            region[local_is - 1 : local_ie + 3, local_js - 1 : local_je + 2]
        ):
            uc = lagrange_x_func(utmp)
            utc = contravariant(uc, v, cosa_u, rsin_u)

        # West
        with horizontal(region[i_start - 1, local_js - 1 : local_je + 2]):
            uc = vol_conserv_cubic_interp_func_x(utmp)

        with horizontal(region[i_start, local_js - 1 : local_je + 2]):
            utc = edge_interpolate4_x(ua, dxa)
            uc = utc * sin_sg3[-1, 0, 0] if utc > 0 else utc * sin_sg1

        with horizontal(region[i_start + 1, local_js - 1 : local_je + 2]):
            uc = vol_conserv_cubic_interp_func_x_rev(utmp)

        with horizontal(region[i_start - 1, local_js - 1 : local_je + 2]):
            utc = contravariant(uc, v, cosa_u, rsin_u)

        with horizontal(region[i_start + 1, local_js - 1 : local_je + 2]):
            utc = contravariant(uc, v, cosa_u, rsin_u)

        # East
        with horizontal(region[i_end, local_js - 1 : local_je + 2]):
            uc = vol_conserv_cubic_interp_func_x(utmp)

        with horizontal(region[i_end + 1, local_js - 1 : local_je + 2]):
            utc = edge_interpolate4_x(ua, dxa)
            uc = utc * sin_sg3[-1, 0, 0] if utc > 0 else utc * sin_sg1

        with horizontal(region[i_end + 2, local_js - 1 : local_je + 2]):
            uc = vol_conserv_cubic_interp_func_x_rev(utmp)

        with horizontal(region[i_end, local_js - 1 : local_je + 2]):
            utc = contravariant(uc, v, cosa_u, rsin_u)

        with horizontal(region[i_end + 2, local_js - 1 : local_je + 2]):
            utc = contravariant(uc, v, cosa_u, rsin_u)

    # Fill corners for Y
    with computation(PARALLEL), interval(...):

        assert __INLINED(namelist.grid_type < 3)

        vtmp = fill3_4corners_y(
            vtmp, utmp, sw_mult=-1, se_mult=1, ne_mult=-1, nw_mult=1
        )
        va = fill2_4corners_y(va, ua, sw_mult=-1, se_mult=1, ne_mult=-1, nw_mult=1)

    # Y
    with computation(PARALLEL), interval(...):

        with horizontal(
            region[local_is - 1 : local_ie + 2, local_js - 1 : local_je + 3]
        ):
            vc = lagrange_y_func(vtmp)
            vtc = contravariant(vc, u, cosa_v, rsin_v)

        with horizontal(region[local_is - 1 : local_ie + 2, j_start - 1]):
            vc = vol_conserv_cubic_interp_func_y(vtmp)
            vtc = contravariant(vc, u, cosa_v, rsin_v)

        with horizontal(region[local_is - 1 : local_ie + 2, j_start]):
            vtc = edge_interpolate4_y(va, dya)
            vc = vtc * sin_sg4[0, -1, 0] if vtc > 0 else vtc * sin_sg2

        with horizontal(region[local_is - 1 : local_ie + 2, j_start + 1]):
            vc = vol_conserv_cubic_interp_func_y_rev(vtmp)
            vtc = contravariant(vc, u, cosa_v, rsin_v)

        with horizontal(region[local_is - 1 : local_ie + 2, j_end]):
            vc = vol_conserv_cubic_interp_func_y(vtmp)
            vtc = contravariant(vc, u, cosa_v, rsin_v)

        with horizontal(region[local_is - 1 : local_ie + 2, j_end + 1]):
            vtc = edge_interpolate4_y(va, dya)
            vc = vtc * sin_sg4[0, -1, 0] if vtc > 0 else vtc * sin_sg2

        with horizontal(region[local_is - 1 : local_ie + 2, j_end + 2]):
            vc = vol_conserv_cubic_interp_func_y_rev(vtmp)
            vtc = contravariant(vc, u, cosa_v, rsin_v)
