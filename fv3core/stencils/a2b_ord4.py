from typing import Optional

import gt4py
import gt4py.gtscript as gtscript
#from gt4py.gtscript import PARALLEL, computation, interval

from gt4py.gtscript import (
    __INLINED,
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
from fv3core.decorators import gtstencil

from fv3core.stencils.basic_operations import copy_stencil
from fv3core.utils.typing import FloatField, FloatFieldI, FloatFieldIJ
from fv3core.utils import global_config
import fv3core.utils.gt4py_utils as utils


# compact 4-pt cubic interpolation
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





@gtscript.function
def ppm_volume_mean_x(qin: FloatField):
    return b2 * (qin[-2, 0, 0] + qin[1, 0, 0]) + b1 * (qin[-1, 0, 0] + qin)


@gtscript.function
def ppm_volume_mean_y(qin: FloatField):
    return b2 * (qin[0, -2, 0] + qin[0, 1, 0]) + b1 * (qin[0, -1, 0] + qin)


@gtscript.function
def lagrange_y(qx: FloatField):
    return a2 * (qx[0, -2, 0] + qx[0, 1, 0]) + a1 * (qx[0, -1, 0] + qx)

@gtscript.function                                                                                                 
def lagrange_x(qy: FloatField):                                                                                    
    return a2 * (qy[-2, 0, 0] + qy[1, 0, 0]) + a1 * (qy[-1, 0, 0] + qy)                                            
                                                                                                                   
 
'''
@gtstencil()
def lagrange_interpolation_y(qx: FloatField, qout: FloatField):
    with computation(PARALLEL), interval(...):
        qout = lagrange_y_func(qx)



@gtstencil()
def lagrange_interpolation_x(qy: FloatField, qout: FloatField):
    with computation(PARALLEL), interval(...):
        qout = lagrange_x_func(qy)


@gtstencil()
def cubic_interpolation_south(qx: FloatField, qout: FloatField, qxx: FloatField):
    with computation(PARALLEL), interval(...):
        qxx0 = qxx
        qxx = c1 * (qx[0, -1, 0] + qx) + c2 * (qout[0, -1, 0] + qxx0[0, 1, 0])


@gtstencil()
def cubic_interpolation_north(qx: FloatField, qout: FloatField, qxx: FloatField):
    with computation(PARALLEL), interval(...):
        qxx0 = qxx
        qxx = c1 * (qx[0, -1, 0] + qx) + c2 * (qout[0, 1, 0] + qxx0[0, -1, 0])


@gtstencil()
def cubic_interpolation_west(qy: FloatField, qout: FloatField, qyy: FloatField):
    with computation(PARALLEL), interval(...):
        qyy0 = qyy
        qyy = c1 * (qy[-1, 0, 0] + qy) + c2 * (qout[-1, 0, 0] + qyy0[1, 0, 0])


@gtstencil()
def cubic_interpolation_east(qy: FloatField, qout: FloatField, qyy: FloatField):
    with computation(PARALLEL), interval(...):
        qyy0 = qyy
        qyy = c1 * (qy[-1, 0, 0] + qy) + c2 * (qout[1, 0, 0] + qyy0[-1, 0, 0])
'''
@gtscript.function
def cubic_interpolation_south(qx: FloatField, qout: FloatField, qxx: FloatField):
    return c1 * (qx[0, -1, 0] + qx) + c2 * (qout[0, -1, 0] + qxx[0, 1, 0])


@gtscript.function
def cubic_interpolation_north(qx: FloatField, qout: FloatField, qxx: FloatField):
    return c1 * (qx[0, -1, 0] + qx) + c2 * (qout[0, 1, 0] + qxx[0, -1, 0])


@gtscript.function
def cubic_interpolation_west(qy: FloatField, qout: FloatField, qyy: FloatField):
    return c1 * (qy[-1, 0, 0] + qy) + c2 * (qout[-1, 0, 0] + qyy[1, 0, 0])



@gtscript.function
def cubic_interpolation_east(qy: FloatField, qout: FloatField, qyy: FloatField):
    return c1 * (qy[-1, 0, 0] + qy) + c2 * (qout[1, 0, 0] + qyy[-1, 0, 0])


#@gtstencil()
#def qout_avg(qxx: FloatField, qyy: FloatField, qout: FloatField):
#    with computation(PARALLEL), interval(...):
#        qout[0, 0, 0] = 0.5 * (qxx + qyy)

@gtscript.function
def qout_x_edge(edge_w: FloatFieldIJ, q2: FloatField):
    return edge_w * q2[0, -1, 0] + (1.0 - edge_w) * q2


#@gtstencil()
#def vort_adjust(qxx: FloatField, qyy: FloatField, qout: FloatField):
#    with computation(PARALLEL), interval(...):
#        qout[0, 0, 0] = 0.5 * (qxx + qyy)

@gtscript.function
def qout_y_edge(edge_s: FloatFieldI, q1: FloatField):
    return edge_s * q1[-1, 0, 0] + (1.0 - edge_s) * q1
#@gtstencil()
#def qout_x_edge(
#    qin: FloatField, dxa: FloatFieldIJ, edge_w: FloatFieldIJ, qout: FloatField
#):
#    with computation(PARALLEL), interval(...):
#        q2 = (qin[-1, 0, 0] * dxa + qin * dxa[-1, 0]) / (dxa[-1, 0] + dxa)
#        qout[0, 0, 0] = edge_w * q2[0, -1, 0] + (1.0 - edge_w) * q2


#@gtstencil()
#def qout_y_edge(
#    qin: FloatField, dya: FloatFieldIJ, edge_s: FloatFieldI, qout: FloatField
#):
#    with computation(PARALLEL), interval(...):
#        q1 = (qin[0, -1, 0] * dya + qin * dya[0, -1]) / (dya[0, -1] + dya)
#        qout[0, 0, 0] = edge_s * q1[-1, 0, 0] + (1.0 - edge_s) * q1


#@gtstencil()
#def qx_edge_west(qin: FloatField, dxa: FloatFieldIJ, qx: FloatField):
#    with computation(PARALLEL), interval(...):
#        g_in = dxa[1, 0] / dxa
#        g_ou = dxa[-2, 0] / dxa[-1, 0]
#        qx[0, 0, 0] = 0.5 * (
#            ((2.0 + g_in) * qin - qin[1, 0, 0]) / (1.0 + g_in)
#            + ((2.0 + g_ou) * qin[-1, 0, 0] - qin[-2, 0, 0]) / (1.0 + g_ou)
#        )
#        # This does not work, due to access of qx that is changing above


@gtscript.function
def qx_edge_west(qin: FloatField, dxa: FloatFieldIJ):
    g_in = dxa[1, 0] / dxa
    g_ou = dxa[-2, 0] / dxa[-1, 0]
    return 0.5 * (
        ((2.0 + g_in) * qin - qin[1, 0, 0]) / (1.0 + g_in)
        + ((2.0 + g_ou) * qin[-1, 0, 0] - qin[-2, 0, 0]) / (1.0 + g_ou)
    )


@gtscript.function
def qx_edge_west2(qin: FloatField, dxa: FloatFieldIJ, qx: FloatFieldIJ):
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


#@gtstencil()
#def qx_edge_west2(qin: FloatField, dxa: FloatFieldIJ, qx: FloatField):
#    with computation(PARALLEL), interval(...):
#        g_in = dxa / dxa[-1, 0]
#        qx0 = qx
#        qx = (
#            3.0 * (g_in * qin[-1, 0, 0] + qin) - (g_in * qx0[-1, 0, 0] + qx0[1, 0, 0])
#        ) / (2.0 + 2.0 * g_in)

@gtscript.function
def qy_edge_south(qin: FloatField, dya: FloatFieldIJ):
    g_in = dya[0, 1] / dya
    g_ou = dya[0, -2] / dya[0, -1]
    return 0.5 * (
        ((2.0 + g_in) * qin - qin[0, 1, 0]) / (1.0 + g_in)
        + ((2.0 + g_ou) * qin[0, -1, 0] - qin[0, -2, 0]) / (1.0 + g_ou)
    )

'''
@gtstencil()
def qx_edge_east(qin: FloatField, dxa: FloatFieldIJ, qx: FloatField):
    with computation(PARALLEL), interval(...):
        g_in = dxa[-2, 0] / dxa[-1, 0]
        g_ou = dxa[1, 0] / dxa
        qx[0, 0, 0] = 0.5 * (
            ((2.0 + g_in) * qin[-1, 0, 0] - qin[-2, 0, 0]) / (1.0 + g_in)
            + ((2.0 + g_ou) * qin - qin[1, 0, 0]) / (1.0 + g_ou)

'''
@gtscript.function
def qy_edge_south2(qin: FloatField, dya: FloatFieldIJ, qy: FloatField):
    g_in = dya / dya[0, -1]
    return (
        3.0 * (g_in * qin[0, -1, 0] + qin) - (g_in * qy[0, -1, 0] + qy[0, 1, 0])
    ) / (2.0 + 2.0 * g_in)

'''
@gtstencil()
def qx_edge_east2(qin: FloatField, dxa: FloatFieldIJ, qx: FloatField):
    with computation(PARALLEL), interval(...):
        g_in = dxa[-1, 0] / dxa
        qx0 = qx
        qx = (
            3.0 * (qin[-1, 0, 0] + g_in * qin) - (g_in * qx0[1, 0, 0] + qx0[-1, 0, 0])
        ) / (2.0 + 2.0 * g_in)
'''


@gtscript.function
def qy_edge_north(qin: FloatField, dya: FloatFieldIJ):
    g_in = dya[0, -2] / dya[0, -1]
    g_ou = dya[0, 1] / dya
    return 0.5 * (
        ((2.0 + g_in) * qin[0, -1, 0] - qin[0, -2, 0]) / (1.0 + g_in)
        + ((2.0 + g_ou) * qin - qin[0, 1, 0]) / (1.0 + g_ou)
    )

'''
@gtstencil()
def qy_edge_south(qin: FloatField, dya: FloatFieldIJ, qy: FloatField):
    with computation(PARALLEL), interval(...):
        g_in = dya[0, 1] / dya
        g_ou = dya[0, -2] / dya[0, -1]
        qy[0, 0, 0] = 0.5 * (
            ((2.0 + g_in) * qin - qin[0, 1, 0]) / (1.0 + g_in)
            + ((2.0 + g_ou) * qin[0, -1, 0] - qin[0, -2, 0]) / (1.0 + g_ou)
        )
'''

@gtscript.function
def qy_edge_north2(qin: FloatField, dya: FloatFieldIJ, qy: FloatField):
    g_in = dya[0, -1] / dya
    return (
        3.0 * (qin[0, -1, 0] + g_in * qin) - (g_in * qy[0, 1, 0] + qy[0, -1, 0])
    ) / (2.0 + 2.0 * g_in)

'''
@gtstencil()
def qy_edge_south2(qin: FloatField, dya: FloatFieldIJ, qy: FloatField):
    with computation(PARALLEL), interval(...):
        g_in = dya / dya[0, -1]
        qy0 = qy
        qy = (
            3.0 * (g_in * qin[0, -1, 0] + qin) - (g_in * qy0[0, -1, 0] + qy0[0, 1, 0])
        ) / (2.0 + 2.0 * g_in)
'''

@gtscript.function
def great_circle_dist_noradius(p1a: FloatField, p1b: FloatField, p2a: FloatField, p2b: FloatField):
    tb = sin((p1b - p2b) / 2.0) ** 2
    ta = sin((p1a - p2a) / 2.0) ** 2
    return asin(sqrt(tb + cos(p1b) * cos(p2b) * ta)) * 2.0

'''
@gtstencil()
def qy_edge_north(qin: FloatField, dya: FloatFieldIJ, qy: FloatField):
    with computation(PARALLEL), interval(...):
        g_in = dya[0, -2] / dya[0, -1]
        g_ou = dya[0, 1] / dya
        qy[0, 0, 0] = 0.5 * (
            ((2.0 + g_in) * qin[0, -1, 0] - qin[0, -2, 0]) / (1.0 + g_in)
            + ((2.0 + g_ou) * qin - qin[0, 1, 0]) / (1.0 + g_ou)
        )
'''

@gtscript.function
def extrap_corner(
    p0a: FloatField,
    p0b: FloatField,
    p1a: FloatField,
    p1b: FloatField,
    p2a: FloatField,
    p2b: FloatField,
    qa: FloatField,
    qb: FloatField,
):
    x1 = great_circle_dist_noradius(p1a, p1b, p0a, p0b)
    x2 = great_circle_dist_noradius(p2a, p2b, p0a, p0b)
    return qa + x1 / (x2 - x1) * (qa - qb)
    

'''
@gtstencil()
def qy_edge_north2(qin: FloatField, dya: FloatFieldIJ, qy: FloatField):
    with computation(PARALLEL), interval(...):
        g_in = dya[0, -1] / dya
        qy0 = qy
        qy = (
            3.0 * (qin[0, -1, 0] + g_in * qin) - (g_in * qy0[0, 1, 0] + qy0[0, -1, 0])
        ) / (2.0 + 2.0 * g_in)


def ec1_offsets(corner):
    i1a, i1b = ec1_offsets_dir(corner, "w")
    j1a, j1b = ec1_offsets_dir(corner, "s")
    return i1a, i1b, j1a, j1b


def ec1_offsets_dir(corner, lower_direction):
    if lower_direction in corner:
        a = 0
        b = 1
    else:
        a = -1
        b = -2
    return (a, b)


def ec2_offsets_dirs(corner, lower_direction, other_direction):
    if lower_direction in corner or other_direction in corner:
        a = -1
        b = -2
    else:
        a = 0
        b = 1
    return a, b


def ec2_offsets(corner):
    i2a, i2b = ec2_offsets_dirs(corner, "s", "w")
    j2a, j2b = ec2_offsets_dirs(corner, "e", "n")
    return i2a, i2b, j2a, j2b


def ec3_offsets_dirs(corner, lower_direction, other_direction):
    if lower_direction in corner or other_direction in corner:
        a = 0
        b = 1
    else:
        a = -1
        b = -2
    return a, b


def ec3_offsets(corner):
    i3a, i3b = ec3_offsets_dirs(corner, "s", "w")
    j3a, j3b = ec3_offsets_dirs(corner, "e", "n")
    return i3a, i3b, j3a, j3b


# TODO: Put into stencil?
def extrapolate_corner_qout(qin, qout, i, j, kstart, nk, corner):
    if not getattr(grid(), corner + "_corner"):
        return
    kslice = slice(kstart, kstart + nk)
    bgrid = utils.stack((grid().bgrid1[:, :], grid().bgrid2[:, :]), axis=2)
    agrid = utils.stack((grid().agrid1[:, :], grid().agrid2[:, :]), axis=2)
    p0 = bgrid[i, j, :]
    # TODO: Please simplify
    i1a, i1b, j1a, j1b = ec1_offsets(corner)
    i2a, i2b, j2a, j2b = ec2_offsets(corner)
    i3a, i3b, j3a, j3b = ec3_offsets(corner)
    ec1 = utils.extrap_corner(
        p0,
        agrid[i + i1a, j + j1a, :],
        agrid[i + i1b, j + j1b, :],
        qin[i + i1a, j + j1a, kslice],
        qin[i + i1b, j + j1b, kslice],
    )
    ec2 = utils.extrap_corner(
        p0,
        agrid[i + i2a, j + j2a, :],
        agrid[i + i2b, j + j2b, :],
        qin[i + i2a, j + j2a, kslice],
        qin[i + i2b, j + j2b, kslice],
    )
    ec3 = utils.extrap_corner(
        p0,
        agrid[i + i3a, j + j3a, :],
        agrid[i + i3b, j + j3b, :],
        qin[i + i3a, j + j3a, kslice],
        qin[i + i3b, j + j3b, kslice],
    )
    qout[i, j, kslice] = (ec1 + ec2 + ec3) * (1.0 / 3.0)


def extrapolate_corners(qin, qout, kstart, nk):
    # qout corners, 3 way extrapolation
    extrapolate_corner_qout(qin, qout, grid().is_, grid().js, kstart, nk, "sw")
    extrapolate_corner_qout(qin, qout, grid().ie + 1, grid().js, kstart, nk, "se")
    extrapolate_corner_qout(qin, qout, grid().ie + 1, grid().je + 1, kstart, nk, "ne")
    extrapolate_corner_qout(qin, qout, grid().is_, grid().je + 1, kstart, nk, "nw")


def compute_qout_edges(qin, qout, kstart, nk):
    compute_qout_x_edges(qin, qout, kstart, nk)
    compute_qout_y_edges(qin, qout, kstart, nk)


def compute_qout_x_edges(qin, qout, kstart, nk):
    # qout bounds
    # avoid running west/east computation on south/north tile edges, since
    # they'll be overwritten.
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


def compute_qout_y_edges(qin, qout, kstart, nk):
    # avoid running south/north computation on west/east tile edges, since
    # they'll be overwritten.
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


def compute_qx(qin, qout, kstart, nk):
    qx = utils.make_storage_from_shape(
        qin.shape, origin=(grid().is_, grid().jsd, kstart), cache_key="a2b_ord4_qx"
    )
    # qx bounds
    # avoid running center-domain computation on tile edges, since they'll be
    # overwritten.
    js = grid().js if grid().south_edge else grid().js - 2
    je = grid().je if grid().north_edge else grid().je + 2
    is_ = grid().is_ + 2 if grid().west_edge else grid().is_
    ie = grid().ie - 1 if grid().east_edge else grid().ie + 1
    dj = je - js + 1
    # qx interior
    ppm_volume_mean_x(qin, qx, origin=(is_, js, kstart), domain=(ie - is_ + 1, dj, nk))

    # qx edges
    if grid().west_edge:
        qx_edge_west(
            qin, grid().dxa, qx, origin=(grid().is_, js, kstart), domain=(1, dj, nk)
        )
        qx_edge_west2(
            qin, grid().dxa, qx, origin=(grid().is_ + 1, js, kstart), domain=(1, dj, nk)
        )
    if grid().east_edge:
        qx_edge_east(
            qin, grid().dxa, qx, origin=(grid().ie + 1, js, kstart), domain=(1, dj, nk)
        )
        qx_edge_east2(
            qin, grid().dxa, qx, origin=(grid().ie, js, kstart), domain=(1, dj, nk)
        )
    return qx


def compute_qy(qin, qout, kstart, nk):
    qy = utils.make_storage_from_shape(
        qin.shape, origin=(grid().isd, grid().js, kstart), cache_key="a2b_ord4_qy"
    )
    # qy bounds
    # avoid running center-domain computation on tile edges, since they'll be
    # overwritten.
    js = grid().js + 2 if grid().south_edge else grid().js
    je = grid().je - 1 if grid().north_edge else grid().je + 1
    is_ = grid().is_ if grid().west_edge else grid().is_ - 2
    ie = grid().ie if grid().east_edge else grid().ie + 2
    di = ie - is_ + 1
    # qy interior
    ppm_volume_mean_y(qin, qy, origin=(is_, js, kstart), domain=(di, je - js + 1, nk))
    # qy edges
    if grid().south_edge:
        qy_edge_south(
            qin, grid().dya, qy, origin=(is_, grid().js, kstart), domain=(di, 1, nk)
        )
        qy_edge_south2(
            qin, grid().dya, qy, origin=(is_, grid().js + 1, kstart), domain=(di, 1, nk)
        )
    if grid().north_edge:
        qy_edge_north(
            qin, grid().dya, qy, origin=(is_, grid().je + 1, kstart), domain=(di, 1, nk)
        )
        qy_edge_north2(
            qin, grid().dya, qy, origin=(is_, grid().je, kstart), domain=(di, 1, nk)
        )
    return qy


def compute_qxx(qx, qout, kstart, nk):
    qxx = utils.make_storage_from_shape(
        qx.shape, origin=grid().full_origin(), cache_key="a2b_ord4_qxx"
    )
    # avoid running center-domain computation on tile edges, since they'll be
    # overwritten.
    js = grid().js + 2 if grid().south_edge else grid().js
    je = grid().je - 1 if grid().north_edge else grid().je + 1
    is_ = grid().is_ + 1 if grid().west_edge else grid().is_
    ie = grid().ie if grid().east_edge else grid().ie + 1
    di = ie - is_ + 1
    lagrange_interpolation_y(
        qx, qxx, origin=(is_, js, kstart), domain=(di, je - js + 1, nk)
    )
    if grid().south_edge:
        cubic_interpolation_south(
            qx, qout, qxx, origin=(is_, grid().js + 1, kstart), domain=(di, 1, nk)
        )
    if grid().north_edge:
        cubic_interpolation_north(
            qx, qout, qxx, origin=(is_, grid().je, kstart), domain=(di, 1, nk)
        )
    return qxx


def compute_qyy(qy, qout, kstart, nk):
    qyy = utils.make_storage_from_shape(
        qy.shape, origin=grid().full_origin(), cache_key="a2b_ord4_qyy"
    )
    # avoid running center-domain computation on tile edges, since they'll be
    # overwritten.
    js = grid().js + 1 if grid().south_edge else grid().js
    je = grid().je if grid().north_edge else grid().je + 1
    is_ = grid().is_ + 2 if grid().west_edge else grid().is_
    ie = grid().ie - 1 if grid().east_edge else grid().ie + 1
    dj = je - js + 1
    lagrange_interpolation_x(
        qy, qyy, origin=(is_, js, kstart), domain=(ie - is_ + 1, dj, nk)
    )
    if grid().west_edge:
        cubic_interpolation_west(
            qy, qout, qyy, origin=(grid().is_ + 1, js, kstart), domain=(1, dj, nk)
        )
    if grid().east_edge:
        cubic_interpolation_east(
            qy, qout, qyy, origin=(grid().ie, js, kstart), domain=(1, dj, nk)
        )
    return qyy


def compute_qout(qxx, qyy, qout, kstart, nk):
    # avoid running center-domain computation on tile edges, since they'll be
    # overwritten.
    js = grid().js + 1 if grid().south_edge else grid().js
    je = grid().je if grid().north_edge else grid().je + 1
    is_ = grid().is_ + 1 if grid().west_edge else grid().is_
    ie = grid().ie if grid().east_edge else grid().ie + 1
    qout_avg(
        qxx, qyy, qout, origin=(is_, js, kstart), domain=(ie - is_ + 1, je - js + 1, nk)
'''
def _a2b_ord4_stencil(
    qin: FloatField,
    qout: FloatField,
    agrid1: FloatFieldIJ,
    agrid2: FloatFieldIJ,
    bgrid1: FloatFieldIJ,
    bgrid2: FloatFieldIJ,
    dxa: FloatFieldIJ,
    dya: FloatFieldIJ,
    edge_n: FloatFieldI,
    edge_s: FloatFieldI,
    edge_e: FloatFieldIJ,
    edge_w: FloatFieldIJ,
):
    from __externals__ import REPLACE, i_end, i_start, j_end, j_start

    with computation(PARALLEL), interval(...):
        # TODO remove the fake_out when the HorizontalIf bug is fixed
        fake_out=0
        with horizontal(region[i_start, j_start]):
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

        with horizontal(region[i_end + 1, j_start]):
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

        with horizontal(region[i_end + 1, j_end + 1]):
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

        with horizontal(region[i_start, j_end + 1]):
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

        #assert __INLINED(namelist.grid_type < 3)

        # {

        #with computation(PARALLEL), interval(...):
        # TODO remove q2 = 0 when that doesn't trigger HorizontalIf bug
        q2 = 0.0
        with horizontal(
            region[i_start - 1 : i_start + 1, :], region[i_end : i_end + 2, :]
        ):
            q2 = (qin[-1, 0, 0] * dxa + qin * dxa[-1, 0]) / (dxa[-1, 0] + dxa)

        #with computation(PARALLEL), interval(...):
        with horizontal(region[i_start, j_start + 1 : j_end + 1]):
            qout = qout_x_edge(edge_w, q2)

        with horizontal(region[i_end + 1, j_start + 1 : j_end + 1]):
            qout = qout_x_edge(edge_e, q2)

        # TODO remove q1 = 0 when that doesn't trigger HorizontalIf bug
        q1 = 0.0
        with horizontal(
            region[:, j_start - 1 : j_start + 1], region[:, j_end : j_end + 2]
        ):
            q1 = (qin[0, -1, 0] * dya + qin * dya[0, -1]) / (dya[0, -1] + dya)

        #with computation(PARALLEL), interval(...):
        with horizontal(region[i_start + 1 : i_end + 1, j_start]):
            qout = qout_y_edge(edge_s, q1)
        with horizontal(region[i_start + 1 : i_end + 1, j_end + 1]):
            qout = qout_y_edge(edge_n, q1)

        # compute_qx
        #with computation(PARALLEL), interval(...):
        #qx=b2 * (qin[-2, 0, 0] + qin[1, 0, 0]) + b1 * (qin[-1, 0, 0] + qin)
        qx = ppm_volume_mean_x(qin)

        with horizontal(region[i_start, :]):
            qx = qx_edge_west(qin, dxa)
        with horizontal(region[i_start + 1, :]):
            qx = qx_edge_west2(qin, dxa, qx)
        with horizontal(region[i_end + 1, :]):
            qx = qx_edge_east(qin, dxa)
        with horizontal(region[i_end, :]):
            qx = qx_edge_east2(qin, dxa, qx)

        # compute_qy
        qy = ppm_volume_mean_y(qin)
        with horizontal(region[:, j_start]):
            qy = qy_edge_south(qin, dya)
        with horizontal(region[:, j_start + 1]):
            qy = qy_edge_south2(qin, dya, qy)

        with horizontal(region[:, j_end + 1]):
            qy = qy_edge_north(qin, dya)
        with horizontal(region[:, j_end]):
            qy = qy_edge_north2(qin, dya, qy)
        #with computation(PARALLEL), interval(...):
        # compute_qxx
        qxx = lagrange_y(qx)
        with horizontal(region[:, j_start + 1]):
            qxx = cubic_interpolation_south(qx, qout, qxx)
        with horizontal(region[:, j_end]):
            qxx = cubic_interpolation_north(qx, qout, qxx)

        # compute_qyy
        qyy = lagrange_x(qy)
        with horizontal(region[i_start + 1, :]):
            qyy = cubic_interpolation_west(qy, qout, qyy)
        with horizontal(region[i_end, :]):
            qyy = cubic_interpolation_east(qy, qout, qyy)

        with horizontal(region[i_start + 1 : i_end + 1, j_start + 1 : j_end + 1]):
            qout = 0.5 * (qxx + qyy)
        # }

        if __INLINED(REPLACE):
            qin = qout
'''
'''

def _make_grid_storage_2d(grid_array: gt4py.storage.Storage, shape3d):#index: int = 0):
    grid = spec.grid
    return utils.make_storage_data(grid_array, shape3d[0:2], (0, 0))
'''
    return gt4py.storage.from_array(
        grid_array.data[:, None],
        backend=global_config.get_backend(),
        default_origin=grid.compute_origin()[:-1],
        shape=grid_array.shape,
        dtype=grid_array.dtype,
        mask=(True, True, False),
    )
'''

def compute(
    qin: FloatField,
    qout: FloatField,
    kstart: int = 0,
    nk: Optional[int] = None,
    replace: bool = False,
):
    """
    Transfers qin from A-grid to B-grid.

    Args:
        qin: Input on A-grid (in)
        qout: Output on B-grid (out)
        kstart: Starting level
        nk: Number of levels
        replace: If True, sets `qout = qin` as the last step
    """
    grid = spec.grid
    if nk is None:
        nk = grid.npz - kstart
    #agrid1 = _make_grid_storage_2d(grid.agrid1)
    #agrid2 = _make_grid_storage_2d(grid.agrid2)
    #bgrid1 = _make_grid_storage_2d(grid.bgrid1)
    #bgrid2 = _make_grid_storage_2d(grid.bgrid2)
    #dxa = _make_grid_storage_2d(grid.dxa)
    #dya = _make_grid_storage_2d(grid.dya)
    shape = qin.shape
    #edge_n = _make_grid_storage_2d(grid.edge_n, shape)
    #edge_s = _make_grid_storage_2d(grid.edge_s, shape)
    edge_e = _make_grid_storage_2d(grid.edge_e, shape)
    edge_w = _make_grid_storage_2d(grid.edge_w, shape)
    #edge_w = utils.make_storage_data(grid.edge_w, qin.shape[0:2], (0, 0))
    stencil = gtstencil(definition=_a2b_ord4_stencil, externals={"REPLACE": replace})

    stencil(
        qin,
        qout,
        grid.agrid1,
        grid.agrid2,
        grid.bgrid1,
        grid.bgrid2,
        grid.dxa,
        grid.dya,
        grid.edge_n,
        grid.edge_s,
        edge_e,
        edge_w,
        origin=(grid.is_, grid.js, kstart),
        domain=(grid.nic + 1, grid.njc + 1, nk),
    )
