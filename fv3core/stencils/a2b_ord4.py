import gt4py.gtscript as gtscript
from gt4py.gtscript import (
    PARALLEL,
    asin,
    computation,
    cos,
    horizontal,
    interval,
    region,
    sin,
    sqrt,
)

import fv3core._config as spec
import fv3core.utils.gt4py_utils as utils
# import fv3core.utils.gt4py_utils as utils
from fv3core.decorators import gtstencil
from fv3core.stencils.basic_operations import copy_stencil
from fv3core.utils.typing import FloatField, FloatFieldI, FloatFieldIJ


# comact 4-pt cubic interpolation
c1 = 2.0 / 3.0
c2 = -1.0 / 6.0
d1 = 0.375
d2 = -1.0 / 24.0
# PPM volume mean form
b1 = 7.0 / 12.0
b2 = -1.0 / 12.0
# 4-pt Lagrange interpolation
a1 = 9.0 / 16.0
a2 = -1.0 / 16.0


def grid():
    return spec.grid


@gtscript.function
def great_circle_dist(p1a, p1b, p2a, p2b):
    tb = sin((p1b - p2b) / 2.0) ** 2.0
    ta = sin((p1a - p2a) / 2.0) ** 2.0
    return asin(sqrt(tb + cos(p1b) * cos(p2b) * ta)) * 2.0


@gtscript.function
def extrap_corner(
    p0a,
    p0b,
    p1a,
    p1b,
    p2a,
    p2b,
    qa,
    qb,
):
    x1 = great_circle_dist(p1a, p1b, p0a, p0b)
    x2 = great_circle_dist(p2a, p2b, p0a, p0b)
    return qa + x1 / (x2 - x1) * (qa - qb)


@gtstencil()
def _sw_corner(
    qin: FloatField,
    qout: FloatField,
    agrid1: FloatFieldIJ,
    agrid2: FloatFieldIJ,
    bgrid1: FloatFieldIJ,
    bgrid2: FloatFieldIJ,
):

    with computation(PARALLEL), interval(...):
        ec1 = extrap_corner(
            bgrid1[0, 0],
            bgrid2[0, 0],
            agrid1[0, 0],
            agrid2[0, 0],
            agrid1[1, 1],
            agrid2[1, 1],
            qin[0, 0, 0],
            qin[1, 1, 0],
        )
        ec2 = extrap_corner(
            bgrid1[0, 0],
            bgrid2[0, 0],
            agrid1[-1, 0],
            agrid2[-1, 0],
            agrid1[-2, 1],
            agrid2[-2, 1],
            qin[-1, 0, 0],
            qin[-2, 1, 0],
        )
        ec3 = extrap_corner(
            bgrid1[0, 0],
            bgrid2[0, 0],
            agrid1[0, -1],
            agrid2[0, -1],
            agrid1[1, -2],
            agrid2[1, -2],
            qin[0, -1, 0],
            qin[1, -2, 0],
        )

        qout = (ec1 + ec2 + ec3) * (1.0 / 3.0)


@gtstencil()
def _nw_corner(
    qin: FloatField,
    qout: FloatField,
    agrid1: FloatFieldIJ,
    agrid2: FloatFieldIJ,
    bgrid1: FloatFieldIJ,
    bgrid2: FloatFieldIJ,
):
    with computation(PARALLEL), interval(...):
        ec1 = extrap_corner(
            bgrid1[0, 0],
            bgrid2[0, 0],
            agrid1[-1, 0],
            agrid2[-1, 0],
            agrid1[-2, 1],
            agrid2[-2, 1],
            qin[-1, 0, 0],
            qin[-2, 1, 0],
        )
        ec2 = extrap_corner(
            bgrid1[0, 0],
            bgrid2[0, 0],
            agrid1[-1, -1],
            agrid2[-1, -1],
            agrid1[-2, -2],
            agrid2[-2, -2],
            qin[-1, -1, 0],
            qin[-2, -2, 0],
        )
        ec3 = extrap_corner(
            bgrid1[0, 0],
            bgrid2[0, 0],
            agrid1[0, 0],
            agrid2[0, 0],
            agrid1[1, 1],
            agrid2[1, 1],
            qin[0, 0, 0],
            qin[1, 1, 0],
        )
        qout = (ec1 + ec2 + ec3) * (1.0 / 3.0)


@gtstencil()
def _ne_corner(
    qin: FloatField,
    qout: FloatField,
    agrid1: FloatFieldIJ,
    agrid2: FloatFieldIJ,
    bgrid1: FloatFieldIJ,
    bgrid2: FloatFieldIJ,
):
    with computation(PARALLEL), interval(...):
        ec1 = extrap_corner(
            bgrid1[0, 0],
            bgrid2[0, 0],
            agrid1[-1, -1],
            agrid2[-1, -1],
            agrid1[-2, -2],
            agrid2[-2, -2],
            qin[-1, -1, 0],
            qin[-2, -2, 0],
        )
        ec2 = extrap_corner(
            bgrid1[0, 0],
            bgrid2[0, 0],
            agrid1[0, -1],
            agrid2[0, -1],
            agrid1[1, -2],
            agrid2[1, -2],
            qin[0, -1, 0],
            qin[1, -2, 0],
        )
        ec3 = extrap_corner(
            bgrid1[0, 0],
            bgrid2[0, 0],
            agrid1[-1, 0],
            agrid2[-1, 0],
            agrid1[-2, 1],
            agrid2[-2, 1],
            qin[-1, 0, 0],
            qin[-2, 1, 0],
        )
        qout = (ec1 + ec2 + ec3) * (1.0 / 3.0)


@gtstencil()
def _se_corner(
    qin: FloatField,
    qout: FloatField,
    agrid1: FloatFieldIJ,
    agrid2: FloatFieldIJ,
    bgrid1: FloatFieldIJ,
    bgrid2: FloatFieldIJ,
):
    with computation(PARALLEL), interval(...):
        ec1 = extrap_corner(
            bgrid1[0, 0],
            bgrid2[0, 0],
            agrid1[0, -1],
            agrid2[0, -1],
            agrid1[1, -2],
            agrid2[1, -2],
            qin[0, -1, 0],
            qin[1, -2, 0],
        )
        ec2 = extrap_corner(
            bgrid1[0, 0],
            bgrid2[0, 0],
            agrid1[-1, -1],
            agrid2[-1, -1],
            agrid1[-2, -2],
            agrid2[-2, -2],
            qin[-1, -1, 0],
            qin[-2, -2, 0],
        )
        ec3 = extrap_corner(
            bgrid1[0, 0],
            bgrid2[0, 0],
            agrid1[0, 0],
            agrid2[0, 0],
            agrid1[1, 1],
            agrid2[1, 1],
            qin[0, 0, 0],
            qin[1, 1, 0],
        )
        qout = (ec1 + ec2 + ec3) * (1.0 / 3.0)


@gtstencil()
def ppm_volume_mean_x(qin: FloatField, dxa: FloatFieldIJ, qx: FloatField):
    from __externals__ import i_end, i_start

    with computation(PARALLEL), interval(...):
        qx = b2 * (qin[-2, 0, 0] + qin[1, 0, 0]) + b1 * (qin[-1, 0, 0] + qin)
        with horizontal(region[i_start, :]):
            qx = qx_edge_west(qin, dxa)
        with horizontal(region[i_start + 1, :]):
            qx = qx_edge_west2(qin, dxa, qx)
        with horizontal(region[i_end + 1, :]):
            qx = qx_edge_east(qin, dxa)
        with horizontal(region[i_end, :]):
            qx = qx_edge_east2(qin, dxa, qx)


@gtstencil()
def ppm_volume_mean_y(qin: FloatField, dya: FloatFieldIJ, qy: FloatField):
    from __externals__ import j_end, j_start

    with computation(PARALLEL), interval(...):
        qy = b2 * (qin[0, -2, 0] + qin[0, 1, 0]) + b1 * (qin[0, -1, 0] + qin)
        with horizontal(region[:, j_start]):
            qy = qy_edge_south(qin, dya)
        with horizontal(region[:, j_start + 1]):
            qy = qy_edge_south2(qin, dya, qy)
        with horizontal(region[:, j_end + 1]):
            qy = qy_edge_north(qin, dya)
        with horizontal(region[:, j_end]):
            qy = qy_edge_north2(qin, dya, qy)


@gtscript.function
def lagrange_y_func(qx):
    return a2 * (qx[0, -2, 0] + qx[0, 1, 0]) + a1 * (qx[0, -1, 0] + qx)


@gtscript.function
def lagrange_x_func(qy):
    return a2 * (qy[-2, 0, 0] + qy[1, 0, 0]) + a1 * (qy[-1, 0, 0] + qy)


@gtstencil()
def a2b_interpolation(
    qout: FloatField,
    qin: FloatField,
    qx: FloatField,
    qy: FloatField,
    qxx: FloatField,
    qyy: FloatField,
    dxa: FloatFieldIJ,
    dya: FloatFieldIJ,
):
    from __externals__ import i_end, i_start, j_end, j_start

    with computation(PARALLEL), interval(...):
        qx = b2 * (qin[-2, 0, 0] + qin[1, 0, 0]) + b1 * (qin[-1, 0, 0] + qin)
        with horizontal(region[i_start, :]):
            qx = qx_edge_west(qin, dxa)
        with horizontal(region[i_start + 1, :]):
            qx = qx_edge_west2(qin, dxa, qx)
        with horizontal(region[i_end + 1, :]):
            qx = qx_edge_east(qin, dxa)
        with horizontal(region[i_end, :]):
            qx = qx_edge_east2(qin, dxa, qx)
        qy = b2 * (qin[0, -2, 0] + qin[0, 1, 0]) + b1 * (qin[0, -1, 0] + qin)
        with horizontal(region[:, j_start]):
            qy = qy_edge_south(qin, dya)
        with horizontal(region[:, j_start + 1]):
            qy = qy_edge_south2(qin, dya, qy)
        with horizontal(region[:, j_end + 1]):
            qy = qy_edge_north(qin, dya)
        with horizontal(region[:, j_end]):
            qy = qy_edge_north2(qin, dya, qy)

        qxx = a2 * (qx[0, -2, 0] + qx[0, 1, 0]) + a1 * (qx[0, -1, 0] + qx)
        with horizontal(region[:, j_start + 1]):
            qxx = c1 * (qx[0, -1, 0] + qx) + c2 * (qout[0, -1, 0] + qxx[0, 1, 0])
        with horizontal(region[:, j_end]):
            qxx = c1 * (qx[0, -1, 0] + qx) + c2 * (qout[0, 1, 0] + qxx[0, -1, 0])
        qyy = a2 * (qy[-2, 0, 0] + qy[1, 0, 0]) + a1 * (qy[-1, 0, 0] + qy)
        with horizontal(region[i_start + 1, :]):
            qyy = c1 * (qy[-1, 0, 0] + qy) + c2 * (qout[-1, 0, 0] + qyy[1, 0, 0])
        with horizontal(region[i_end, :]):
            qyy = c1 * (qy[-1, 0, 0] + qy) + c2 * (qout[1, 0, 0] + qyy[-1, 0, 0])
        qout = 0.5 * (qxx + qyy)


@gtstencil()
def qout_x_edge(
    qin: FloatField, dxa: FloatFieldIJ, edge_w: FloatFieldIJ, qout: FloatField
):
    with computation(PARALLEL), interval(...):
        q2 = (qin[-1, 0, 0] * dxa + qin * dxa[-1, 0]) / (dxa[-1, 0] + dxa)
        qout[0, 0, 0] = edge_w * q2[0, -1, 0] + (1.0 - edge_w) * q2


@gtstencil()
def qout_y_edge(
    qin: FloatField, dya: FloatFieldIJ, edge_s: FloatFieldI, qout: FloatField
):
    with computation(PARALLEL), interval(...):
        q1 = (qin[0, -1, 0] * dya + qin * dya[0, -1]) / (dya[0, -1] + dya)
        qout[0, 0, 0] = edge_s * q1[-1, 0, 0] + (1.0 - edge_s) * q1


@gtscript.function
def qx_edge_west(qin: FloatField, dxa: FloatFieldIJ):
    g_in = dxa[1, 0] / dxa
    g_ou = dxa[-2, 0] / dxa[-1, 0]
    return 0.5 * (
        ((2.0 + g_in) * qin - qin[1, 0, 0]) / (1.0 + g_in)
        + ((2.0 + g_ou) * qin[-1, 0, 0] - qin[-2, 0, 0]) / (1.0 + g_ou)
    )
    # This does not work, due to access of qx that is changing above

    # qx[1, 0, 0] = (3.0 * (g_in * qin + qin[1, 0, 0])
    #     - (g_in * qx + qx[2, 0, 0])) / (2.0 + 2.0 * g_in)


@gtscript.function
def qx_edge_west2(qin: FloatField, dxa: FloatFieldIJ, qx: FloatField):
    g_in = dxa / dxa[-1, 0]
    return (
        3.0 * (g_in * qin[-1, 0, 0] + qin) - (g_in * qx[-1, 0, 0] + qx[1, 0, 0])
    ) / (2.0 + 2.0 * g_in)


@gtscript.function
def qx_edge_east(qin: FloatField, dxa: FloatFieldIJ):
    g_in = dxa[-2, 0] / dxa[-1, 0]
    g_ou = dxa[1, 0] / dxa
    return 0.5 * (
        ((2.0 + g_in) * qin[-1, 0, 0] - qin[-2, 0, 0]) / (1.0 + g_in)
        + ((2.0 + g_ou) * qin - qin[1, 0, 0]) / (1.0 + g_ou)
    )


@gtscript.function
def qx_edge_east2(qin: FloatField, dxa: FloatFieldIJ, qx: FloatField):
    g_in = dxa[-1, 0] / dxa
    return (
        3.0 * (qin[-1, 0, 0] + g_in * qin) - (g_in * qx[1, 0, 0] + qx[-1, 0, 0])
    ) / (2.0 + 2.0 * g_in)


@gtscript.function
def qy_edge_south(qin: FloatField, dya: FloatFieldIJ):
    g_in = dya[0, 1] / dya
    g_ou = dya[0, -2] / dya[0, -1]
    return 0.5 * (
        ((2.0 + g_in) * qin - qin[0, 1, 0]) / (1.0 + g_in)
        + ((2.0 + g_ou) * qin[0, -1, 0] - qin[0, -2, 0]) / (1.0 + g_ou)
    )


@gtscript.function
def qy_edge_south2(qin: FloatField, dya: FloatFieldIJ, qy: FloatField):
    g_in = dya / dya[0, -1]
    return (
        3.0 * (g_in * qin[0, -1, 0] + qin) - (g_in * qy[0, -1, 0] + qy[0, 1, 0])
    ) / (2.0 + 2.0 * g_in)


@gtscript.function
def qy_edge_north(qin: FloatField, dya: FloatFieldIJ):
    g_in = dya[0, -2] / dya[0, -1]
    g_ou = dya[0, 1] / dya
    return 0.5 * (
        ((2.0 + g_in) * qin[0, -1, 0] - qin[0, -2, 0]) / (1.0 + g_in)
        + ((2.0 + g_ou) * qin - qin[0, 1, 0]) / (1.0 + g_ou)
    )


@gtscript.function
def qy_edge_north2(qin: FloatField, dya: FloatFieldIJ, qy: FloatField):
    g_in = dya[0, -1] / dya
    return (
        3.0 * (qin[0, -1, 0] + g_in * qin) - (g_in * qy[0, 1, 0] + qy[0, -1, 0])
    ) / (2.0 + 2.0 * g_in)


def compute_qout_edges(qin, qout, kstart, nk):

    js2 = grid().js + 1 if grid().south_edge else grid().js
    je1 = grid().je if grid().north_edge else grid().je + 1
    dj2 = je1 - js2 + 1
    if grid().west_edge:
        qout_x_edge(
            qin,
            grid().dxa,
            grid().edge_w,
            qout,
            origin=(grid().is_, js2, kstart),
            domain=(1, dj2, nk),
        )
    if grid().east_edge:
        qout_x_edge(
            qin,
            grid().dxa,
            grid().edge_e,
            qout,
            origin=(grid().ie + 1, js2, kstart),
            domain=(1, dj2, nk),
        )

    is2 = grid().is_ + 1 if grid().west_edge else grid().is_
    ie1 = grid().ie if grid().east_edge else grid().ie + 1
    di2 = ie1 - is2 + 1
    if grid().south_edge:
        qout_y_edge(
            qin,
            grid().dya,
            grid().edge_s,
            qout,
            origin=(is2, grid().js, kstart),
            domain=(di2, 1, nk),
        )
    if grid().north_edge:
        qout_y_edge(
            qin,
            grid().dya,
            grid().edge_n,
            qout,
            origin=(is2, grid().je + 1, kstart),
            domain=(di2, 1, nk),
        )


def compute(qin, qout, kstart=0, nk=None, replace=False):
    if nk is None:
        nk = grid().npz - kstart
    corner_domain = (1, 1, nk)
    _sw_corner(
        qin,
        qout,
        grid().agrid1,
        grid().agrid2,
        grid().bgrid1,
        grid().bgrid2,
        origin=(grid().is_, grid().js, kstart),
        domain=corner_domain,
    )

    _nw_corner(
        qin,
        qout,
        grid().agrid1,
        grid().agrid2,
        grid().bgrid1,
        grid().bgrid2,
        origin=(grid().ie + 1, grid().js, kstart),
        domain=corner_domain,
    )
    _ne_corner(
        qin,
        qout,
        grid().agrid1,
        grid().agrid2,
        grid().bgrid1,
        grid().bgrid2,
        origin=(grid().ie + 1, grid().je + 1, kstart),
        domain=corner_domain,
    )
    _se_corner(
        qin,
        qout,
        grid().agrid1,
        grid().agrid2,
        grid().bgrid1,
        grid().bgrid2,
        origin=(grid().is_, grid().je + 1, kstart),
        domain=corner_domain,
    )
    if spec.namelist.grid_type < 3:

        compute_qout_edges(qin, qout, kstart, nk)

        qx = utils.make_storage_from_shape(
            qin.shape, origin=(grid().is_, grid().jsd, kstart), cache_key="a2b_ord4_qx"
        )
        qy = utils.make_storage_from_shape(
            qin.shape, origin=(grid().isd, grid().js, kstart), cache_key="a2b_ord4_qy"
        )
        qxx = utils.make_storage_from_shape(
            qin.shape, origin=(grid().isd, grid().js, kstart), cache_key="a2b_ord4_qxx"
        )

        qyy = utils.make_storage_from_shape(
            qin.shape, origin=(grid().isd, grid().js, kstart), cache_key="a2b_ord4_qyy"
        )
        js = grid().js + 1 if grid().south_edge else grid().js
        je = grid().je if grid().north_edge else grid().je + 1
        is_ = grid().is_ + 1 if grid().west_edge else grid().is_
        ie = grid().ie if grid().east_edge else grid().ie + 1

        a2b_interpolation(
            qout,
            qin,
            qx,
            qy,
            qxx,
            qyy,
            grid().dxa,
            grid().dya,
            origin=(is_, js, kstart),
            domain=(ie - is_ + 1, je - js + 1, nk),
        )

        if replace:
            copy_stencil(
                qout,
                qin,
                origin=(grid().is_, grid().js, kstart),
                domain=(grid().ie - grid().is_ + 2, grid().je - grid().js + 2, nk),
            )
    else:
        raise Exception("grid_type >= 3 is not implemented")
