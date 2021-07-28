import numpy as np

import gt4py.gtscript as gtscript
from gt4py.gtscript import FORWARD, PARALLEL, asin, computation, cos, interval, sin, sqrt

import fv3core._config as spec
import fv3core.utils.global_config as config
import fv3core.utils.gt4py_utils as utils
from fv3core.decorators import FrozenStencil
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
def lagrange_y_func(qx):
    return a2 * (qx[0, -2, 0] + qx[0, 1, 0]) + a1 * (qx[0, -1, 0] + qx)


@gtscript.function
def lagrange_x_func(qy):
    return a2 * (qy[-2, 0, 0] + qy[1, 0, 0]) + a1 * (qy[-1, 0, 0] + qy)

@gtscript.function
def cubic_interpolation_y_fn(qx, qout_offset_y, qxx_offset_y):
    return c1 * (qx[0, -1, 0] + qx) + c2 * (qout_offset_y + qxx_offset_y)


@gtscript.function
def cubic_interpolation_x_fn(qy, qout_offset_x, qyy_offset_x):
    return c1 * (qy[-1, 0, 0] + qy) + c2 * (qout_offset_x + qyy_offset_x)

@gtscript.function
def ppm_volume_mean_x_fn(
    qin: FloatField,
):
    return b2 * (qin[-2, 0, 0] + qin[1, 0, 0]) + b1 * (qin[-1, 0, 0] + qin)

def ppm_volume_mean_x(
    qin: FloatField,
    qx: FloatField,
):
    with computation(PARALLEL), interval(...):
        qx = ppm_volume_mean_x_fn(qin)


@gtscript.function
def ppm_volume_mean_y_fn(
    qin: FloatField,
):
    return b2 * (qin[0, -2, 0] + qin[0, 1, 0]) + b1 * (qin[0, -1, 0] + qin)


def ppm_volume_mean_y(
    qin: FloatField,
    qy: FloatField,
):
    with computation(PARALLEL), interval(...):
        qy = ppm_volume_mean_y_fn(qin)

def a2b_interpolation(
    qx: FloatField,
    qy: FloatField,
    qxx: FloatField,
    qyy: FloatField,
):
    with computation(PARALLEL), interval(...):
        qxx = a2 * (qx[0, -2, 0] + qx[0, 1, 0]) + a1 * (qx[0, -1, 0] + qx)
        qyy = a2 * (qy[-2, 0, 0] + qy[1, 0, 0]) + a1 * (qy[-1, 0, 0] + qy)


def final_qout(
    qxx: FloatField,
    qyy: FloatField,
    qout: FloatField,
):
    with computation(PARALLEL), interval(...):
        qout = 0.5 * (qxx + qyy)


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

        self._tmp_q = utils.make_storage_from_shape(shape)
        self._tmp_qx = utils.make_storage_from_shape(shape)
        self._tmp_qy = utils.make_storage_from_shape(shape)
        self._tmp_qxx = utils.make_storage_from_shape(shape)
        self._tmp_qyy = utils.make_storage_from_shape(shape)

        if nk is None:
            nk = self.grid.npz - kstart


        origin_x = (self.grid.is_, self.grid.js - 2, kstart)
        domain_x = (self.grid.nic + 1, self.grid.njc + 4, nk)

        self._ppm_volume_mean_x_stencil = FrozenStencil(
            ppm_volume_mean_x, origin=origin_x, domain=domain_x
        )
        origin_y = (self.grid.is_ - 2, self.grid.js, kstart)
        domain_y = (self.grid.nic + 4, self.grid.njc + 1, nk)

        self._ppm_volume_mean_y_stencil = FrozenStencil(
            ppm_volume_mean_y, origin=origin_y, domain=domain_y
        )

        js = self.grid.js
        je = self.grid.je+1
        is_ = self.grid.is_ 
        ie = self.grid.ie +1
        origin = (is_, js, kstart)
        domain = (ie - is_ + 1, je - js + 1, nk)

        self._a2b_interpolation_stencil = FrozenStencil(
            a2b_interpolation, origin=origin, domain=domain
        )

        self._final_qout_stencil = FrozenStencil(
            final_qout, origin=origin, domain=domain
        )
        self._copy_stencil = FrozenStencil(
            copy_defn,
            origin=(self.grid.is_, self.grid.js, kstart),
            domain=(self.grid.nic + 1, self.grid.njc + 1, nk),
        )

    def __call__(self, qin: FloatField, qout: FloatField):
        """Converts qin from A-grid to B-grid in qout.
        In some cases, qin is also updated to the B grid.
        Args:
        qin: Input on A-grid (inout)
        qout: Output on B-grid (inout)
        """
        self._ppm_volume_mean_x_stencil(
            qin,
            self._tmp_qx,
        )

        self._ppm_volume_mean_y_stencil(
            qin,
            self._tmp_qy,
        )

        self._a2b_interpolation_stencil(
            self._tmp_qx,
            self._tmp_qy,
            self._tmp_qxx,
            self._tmp_qyy,
        )
        self._final_qout_stencil(self._tmp_qxx, self._tmp_qyy, qout)
        if self.replace:
            self._copy_stencil(
                qout,
                qin,
            )
