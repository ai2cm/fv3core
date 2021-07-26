import gt4py.gtscript as gtscript
from gt4py.gtscript import PARALLEL, asin, computation, cos, interval, sin, sqrt

import fv3core._config as spec
import fv3core.utils.gt4py_utils as utils
from fv3core.decorators import FrozenStencil, computepath_method
from fv3core.stencils.basic_operations import copy_defn
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


@gtscript.function
def lagrange_y_func(qx):
    return a2 * (qx[0, -2, 0] + qx[0, 1, 0]) + a1 * (qx[0, -1, 0] + qx)


@gtscript.function
def lagrange_x_func(qy):
    return a2 * (qy[-2, 0, 0] + qy[1, 0, 0]) + a1 * (qy[-1, 0, 0] + qy)


def ppm_volume_mean_x(
    qin: FloatField,
    qx: FloatField,
    dxa: FloatFieldIJ,
):
    with computation(PARALLEL), interval(...):
        qx = b2 * (qin[-2, 0, 0] + qin[1, 0, 0]) + b1 * (qin[-1, 0, 0] + qin)


def qx_west_edge(
    qin: FloatField,
    qx: FloatField,
    dxa: FloatFieldIJ,
):
    with computation(PARALLEL), interval(...):
        g_in = dxa[1, 0] / dxa
        g_ou = dxa[-2, 0] / dxa[-1, 0]
        qx = 0.5 * (
            ((2.0 + g_in) * qin - qin[1, 0, 0]) / (1.0 + g_in)
            + ((2.0 + g_ou) * qin[-1, 0, 0] - qin[-2, 0, 0]) / (1.0 + g_ou)
        )


def qx_west_edge2(
    qin: FloatField,
    qx: FloatField,
    dxa: FloatFieldIJ,
):
    with computation(PARALLEL), interval(...):
        g_in = dxa / dxa[-1, 0]
        q_tmp = qx
        qx = (
            3.0 * (g_in * qin[-1, 0, 0] + qin)
            - (g_in * q_tmp[-1, 0, 0] + q_tmp[1, 0, 0])
        ) / (2.0 + 2.0 * g_in)


def qx_east_edge(
    qin: FloatField,
    qx: FloatField,
    dxa: FloatFieldIJ,
):
    with computation(PARALLEL), interval(...):
        g_in = dxa[-2, 0] / dxa[-1, 0]
        g_ou = dxa[1, 0] / dxa
        qx = 0.5 * (
            ((2.0 + g_in) * qin[-1, 0, 0] - qin[-2, 0, 0]) / (1.0 + g_in)
            + ((2.0 + g_ou) * qin - qin[1, 0, 0]) / (1.0 + g_ou)
        )


def qx_east_edge2(
    qin: FloatField,
    qx: FloatField,
    dxa: FloatFieldIJ,
):
    with computation(PARALLEL), interval(...):
        g_in = dxa[-1, 0] / dxa
        q_tmp = qx
        qx = (
            3.0 * (qin[-1, 0, 0] + g_in * qin)
            - (g_in * q_tmp[1, 0, 0] + q_tmp[-1, 0, 0])
        ) / (2.0 + 2.0 * g_in)


def ppm_volume_mean_y(
    qin: FloatField,
    qy: FloatField,
    dya: FloatFieldIJ,
):
    with computation(PARALLEL), interval(...):
        qy = b2 * (qin[0, -2, 0] + qin[0, 1, 0]) + b1 * (qin[0, -1, 0] + qin)


def qy_south_edge(
    qin: FloatField,
    qy: FloatField,
    dya: FloatFieldIJ,
):
    with computation(PARALLEL), interval(...):
        g_in = dya[0, 1] / dya
        g_ou = dya[0, -2] / dya[0, -1]
        qy = 0.5 * (
            ((2.0 + g_in) * qin - qin[0, 1, 0]) / (1.0 + g_in)
            + ((2.0 + g_ou) * qin[0, -1, 0] - qin[0, -2, 0]) / (1.0 + g_ou)
        )


def qy_south_edge2(
    qin: FloatField,
    qy: FloatField,
    dya: FloatFieldIJ,
):
    with computation(PARALLEL), interval(...):
        g_in = dya / dya[0, -1]
        q_tmp = qy
        qy = (
            3.0 * (g_in * qin[0, -1, 0] + qin)
            - (g_in * q_tmp[0, -1, 0] + q_tmp[0, 1, 0])
        ) / (2.0 + 2.0 * g_in)


def qy_north_edge(
    qin: FloatField,
    qy: FloatField,
    dya: FloatFieldIJ,
):
    with computation(PARALLEL), interval(...):
        g_in = dya[0, -2] / dya[0, -1]
        g_ou = dya[0, 1] / dya
        qy = 0.5 * (
            ((2.0 + g_in) * qin[0, -1, 0] - qin[0, -2, 0]) / (1.0 + g_in)
            + ((2.0 + g_ou) * qin - qin[0, 1, 0]) / (1.0 + g_ou)
        )


def qy_north_edge2(
    qin: FloatField,
    qy: FloatField,
    dya: FloatFieldIJ,
):
    with computation(PARALLEL), interval(...):
        g_in = dya[0, -1] / dya
        q_tmp = qy
        qy = (
            3.0 * (qin[0, -1, 0] + g_in * qin)
            - (g_in * q_tmp[0, 1, 0] + q_tmp[0, -1, 0])
        ) / (2.0 + 2.0 * g_in)


def a2b_interpolation(
    qx: FloatField,
    qy: FloatField,
    qxx: FloatField,
    qyy: FloatField,
):
    with computation(PARALLEL), interval(...):
        qxx = a2 * (qx[0, -2, 0] + qx[0, 1, 0]) + a1 * (qx[0, -1, 0] + qx)
        qyy = a2 * (qy[-2, 0, 0] + qy[1, 0, 0]) + a1 * (qy[-1, 0, 0] + qy)


def qxx_edge_south(
    qout: FloatField,
    qx: FloatField,
    qxx: FloatField,
):
    with computation(PARALLEL), interval(...):
        q_tmp = qxx
        qxx = c1 * (qx[0, -1, 0] + qx) + c2 * (qout[0, -1, 0] + q_tmp[0, 1, 0])


def qxx_edge_north(
    qout: FloatField,
    qx: FloatField,
    qxx: FloatField,
):
    with computation(PARALLEL), interval(...):
        q_tmp = qxx
        qxx = c1 * (qx[0, -1, 0] + qx) + c2 * (qout[0, 1, 0] + q_tmp[0, -1, 0])


def qyy_edge_west(
    qout: FloatField,
    qy: FloatField,
    qyy: FloatField,
):
    with computation(PARALLEL), interval(...):
        q_tmp = qyy
        qyy = c1 * (qy[-1, 0, 0] + qy) + c2 * (qout[-1, 0, 0] + q_tmp[1, 0, 0])


def qyy_edge_east(
    qout: FloatField,
    qy: FloatField,
    qyy: FloatField,
):
    with computation(PARALLEL), interval(...):
        q_tmp = qyy
        qyy = c1 * (qy[-1, 0, 0] + qy) + c2 * (qout[1, 0, 0] + q_tmp[-1, 0, 0])


def final_qout(
    qxx: FloatField,
    qyy: FloatField,
    qout: FloatField,
):
    with computation(PARALLEL), interval(...):
        qout = 0.5 * (qxx + qyy)


def qout_x_edge(
    qin: FloatField, dxa: FloatFieldIJ, edge_w: FloatFieldIJ, qout: FloatField
):
    with computation(PARALLEL), interval(...):
        q2 = (qin[-1, 0, 0] * dxa + qin * dxa[-1, 0]) / (dxa[-1, 0] + dxa)
        qout[0, 0, 0] = edge_w * q2[0, -1, 0] + (1.0 - edge_w) * q2


def qout_y_edge(
    qin: FloatField, dya: FloatFieldIJ, edge_s: FloatFieldI, qout: FloatField
):
    with computation(PARALLEL), interval(...):
        q1 = (qin[0, -1, 0] * dya + qin * dya[0, -1]) / (dya[0, -1] + dya)
        qout[0, 0, 0] = edge_s * q1[-1, 0, 0] + (1.0 - edge_s) * q1


class AGrid2BGridFourthOrder:
    """
    Fortran name is a2b_ord4, test module is A2B_Ord4
    """

    def __init__(
        self, grid_type, kstart: int = 0, nk: int = None, replace: bool = False
    ):
        """
        Args:
            grid_type: integer representing the type of grid
            kstart: first klevel to compute on
            nk: number of k levels to compute
            replace: boolean, update qin to the B grid as well
        """
        assert grid_type < 3
        self.grid = spec.grid
        self.replace = replace
        shape = self.grid.domain_shape_full(add=(1, 1, 1))
        full_origin = (self.grid.isd, self.grid.jsd, kstart)

        self._tmp_qx = utils.make_storage_from_shape(shape)
        self._tmp_qy = utils.make_storage_from_shape(shape)
        self._tmp_qxx = utils.make_storage_from_shape(shape)
        self._tmp_qyy = utils.make_storage_from_shape(shape)

        if nk is None:
            nk = self.grid.npz - kstart
        corner_domain = (1, 1, nk)

        self._sw_corner_stencil = FrozenStencil(
            _sw_corner,
            origin=(self.grid.is_, self.grid.js, kstart),
            domain=corner_domain,
        )
        self._nw_corner_stencil = FrozenStencil(
            _nw_corner,
            origin=(self.grid.ie + 1, self.grid.js, kstart),
            domain=corner_domain,
        )
        self._ne_corner_stencil = FrozenStencil(
            _ne_corner,
            origin=(self.grid.ie + 1, self.grid.je + 1, kstart),
            domain=corner_domain,
        )
        self._se_corner_stencil = FrozenStencil(
            _se_corner,
            origin=(self.grid.is_, self.grid.je + 1, kstart),
            domain=corner_domain,
        )
        js2 = self.grid.js + 1 if self.grid.south_edge else self.grid.js
        je1 = self.grid.je if self.grid.north_edge else self.grid.je + 1
        dj2 = je1 - js2 + 1

        # edge_w is singleton in the I-dimension to work around gt4py not yet
        # supporting J-fields. As a result, the origin has to be zero for
        # edge_w, anything higher is outside its index range
        self._qout_x_edge_west = FrozenStencil(
            qout_x_edge,
            origin={"_all_": (self.grid.is_, js2, kstart), "edge_w": (0, js2)},
            domain=(1, dj2, nk),
        )
        self._qout_x_edge_east = FrozenStencil(
            qout_x_edge,
            origin={"_all_": (self.grid.ie + 1, js2, kstart), "edge_w": (0, js2)},
            domain=(1, dj2, nk),
        )

        is2 = self.grid.is_ + 1 if self.grid.west_edge else self.grid.is_
        ie1 = self.grid.ie if self.grid.east_edge else self.grid.ie + 1
        di2 = ie1 - is2 + 1
        self._qout_y_edge_south = FrozenStencil(
            qout_y_edge, origin=(is2, self.grid.js, kstart), domain=(di2, 1, nk)
        )
        self._qout_y_edge_north = FrozenStencil(
            qout_y_edge, origin=(is2, self.grid.je + 1, kstart), domain=(di2, 1, nk)
        )
        origin_x = (self.grid.is_, self.grid.js - 2, kstart)
        domain_x = (self.grid.nic + 1, self.grid.njc + 4, nk)

        self._ppm_volume_mean_x_stencil = FrozenStencil(
            ppm_volume_mean_x, origin=origin_x, domain=domain_x
        )
        if self.grid.west_edge:
            self._qx_west_edge_stencil = FrozenStencil(
                qx_west_edge, origin=origin_x, domain=(1, domain_x[1], nk)
            )
            self._qx_west_edge_stencil2 = FrozenStencil(
                qx_west_edge2,
                origin=(self.grid.is_ + 1, origin_x[1], kstart),
                domain=(1, domain_x[1], nk),
            )
        if self.grid.east_edge:
            self._qx_east_edge_stencil = FrozenStencil(
                qx_east_edge,
                origin=(self.grid.ie + 1, origin_x[1], kstart),
                domain=(1, domain_x[1], nk),
            )
            self._qx_east_edge_stencil2 = FrozenStencil(
                qx_east_edge2,
                origin=(self.grid.ie, origin_x[1], kstart),
                domain=(1, domain_x[1], nk),
            )
        origin_y = (self.grid.is_ - 2, self.grid.js, kstart)
        domain_y = (self.grid.nic + 4, self.grid.njc + 1, nk)

        self._ppm_volume_mean_y_stencil = FrozenStencil(
            ppm_volume_mean_y, origin=origin_y, domain=domain_y
        )
        if self.grid.south_edge:
            self._qy_south_edge_stencil = FrozenStencil(
                qy_south_edge, origin=origin_y, domain=(domain_y[0], 1, nk)
            )
            self._qy_south_edge_stencil2 = FrozenStencil(
                qy_south_edge2,
                origin=(origin_y[0], self.grid.js + 1, kstart),
                domain=(domain_y[0], 1, nk),
            )
        if self.grid.north_edge:
            self._qy_north_edge_stencil = FrozenStencil(
                qy_north_edge,
                origin=(origin_y[0], self.grid.je + 1, kstart),
                domain=(domain_y[0], 1, nk),
            )
            self._qy_north_edge_stencil2 = FrozenStencil(
                qy_north_edge2,
                origin=(origin_y[0], self.grid.je, kstart),
                domain=(domain_y[0], 1, nk),
            )
        js = self.grid.js + 1 if self.grid.south_edge else self.grid.js
        je = self.grid.je if self.grid.north_edge else self.grid.je + 1
        is_ = self.grid.is_ + 1 if self.grid.west_edge else self.grid.is_
        ie = self.grid.ie if self.grid.east_edge else self.grid.ie + 1
        origin = (is_, js, kstart)
        domain = (ie - is_ + 1, je - js + 1, nk)

        self._a2b_interpolation_stencil = FrozenStencil(
            a2b_interpolation, origin=origin, domain=domain
        )
        if self.grid.south_edge:
            self._qxx_edge_south_stencil = FrozenStencil(
                qxx_edge_south,
                origin=(origin[0], self.grid.js + 1, origin[2]),
                domain=(domain[0], 1, domain[2]),
            )
        if self.grid.north_edge:
            self._qxx_edge_north_stencil = FrozenStencil(
                qxx_edge_north,
                origin=(origin[0], self.grid.je, origin[2]),
                domain=(domain[0], 1, domain[2]),
            )
        if self.grid.west_edge:
            self._qyy_edge_west_stencil = FrozenStencil(
                qyy_edge_west,
                origin=(self.grid.is_ + 1, origin[1], origin[2]),
                domain=(1, domain[1], domain[2]),
            )
        if self.grid.east_edge:
            self._qyy_edge_east_stencil = FrozenStencil(
                qyy_edge_east,
                origin=(self.grid.ie, origin[1], origin[2]),
                domain=(1, domain[1], domain[2]),
            )
        self._final_qout_stencil = FrozenStencil(
            final_qout, origin=origin, domain=domain
        )
        self._copy_stencil = FrozenStencil(
            copy_defn,
            origin=(self.grid.is_, self.grid.js, kstart),
            domain=(self.grid.nic + 1, self.grid.njc + 1, nk),
        )

    @computepath_method
    def __call__(self, qin, qout):
        """Converts qin from A-grid to B-grid in qout.
        In some cases, qin is also updated to the B grid.
        Args:
        qin: Input on A-grid (inout)
        qout: Output on B-grid (inout)
        """

        self._sw_corner_stencil(
            qin,
            qout,
            self.grid.agrid1,
            self.grid.agrid2,
            self.grid.bgrid1,
            self.grid.bgrid2,
        )

        self._nw_corner_stencil(
            qin,
            qout,
            self.grid.agrid1,
            self.grid.agrid2,
            self.grid.bgrid1,
            self.grid.bgrid2,
        )
        self._ne_corner_stencil(
            qin,
            qout,
            self.grid.agrid1,
            self.grid.agrid2,
            self.grid.bgrid1,
            self.grid.bgrid2,
        )
        self._se_corner_stencil(
            qin,
            qout,
            self.grid.agrid1,
            self.grid.agrid2,
            self.grid.bgrid1,
            self.grid.bgrid2,
        )

        self._compute_qout_edges(qin, qout)
        self._ppm_volume_mean_x_stencil(
            qin,
            self._tmp_qx,
            self.grid.dxa,
        )
        if self.grid.west_edge:
            self._qx_west_edge_stencil(qin, self._tmp_qx, self.grid.dxa)
            self._qx_west_edge_stencil2(qin, self._tmp_qx, self.grid.dxa)
        if self.grid.east_edge:
            self._qx_east_edge_stencil(qin, self._tmp_qx, self.grid.dxa)
            self._qx_east_edge_stencil2(qin, self._tmp_qx, self.grid.dxa)
        self._ppm_volume_mean_y_stencil(
            qin,
            self._tmp_qy,
            self.grid.dya,
        )
        if self.grid.south_edge:
            self._qy_south_edge_stencil(qin, self._tmp_qy, self.grid.dya)
            self._qy_south_edge_stencil2(qin, self._tmp_qy, self.grid.dya)
        if self.grid.north_edge:
            self._qy_north_edge_stencil(qin, self._tmp_qy, self.grid.dya)
            self._qy_north_edge_stencil2(qin, self._tmp_qy, self.grid.dya)
        self._a2b_interpolation_stencil(
            self._tmp_qx,
            self._tmp_qy,
            self._tmp_qxx,
            self._tmp_qyy,
        )
        if self.grid.south_edge:
            self._qxx_edge_south_stencil(qout, self._tmp_qx, self._tmp_qxx)
        if self.grid.north_edge:
            self._qxx_edge_north_stencil(qout, self._tmp_qx, self._tmp_qxx)
        if self.grid.west_edge:
            self._qyy_edge_west_stencil(qout, self._tmp_qy, self._tmp_qyy)
        if self.grid.east_edge:
            self._qyy_edge_east_stencil(qout, self._tmp_qy, self._tmp_qyy)
        self._final_qout_stencil(self._tmp_qxx, self._tmp_qyy, qout)
        if self.replace:
            self._copy_stencil(
                qout,
                qin,
            )

    @computepath_method
    def _compute_qout_edges(self, qin, qout):
        if self.grid.west_edge:
            self._qout_x_edge_west(
                qin,
                self.grid.dxa,
                self.grid.edge_w,
                qout,
            )
        if self.grid.east_edge:
            self._qout_x_edge_east(
                qin,
                self.grid.dxa,
                self.grid.edge_e,
                qout,
            )

        if self.grid.south_edge:
            self._qout_y_edge_south(
                qin,
                self.grid.dya,
                self.grid.edge_s,
                qout,
            )
        if self.grid.north_edge:
            self._qout_y_edge_north(
                qin,
                self.grid.dya,
                self.grid.edge_n,
                qout,
            )
