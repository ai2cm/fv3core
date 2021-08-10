import numpy as np

import gt4py.gtscript as gtscript
from gt4py.gtscript import FORWARD, PARALLEL, asin, computation, cos, interval, sin, sqrt

import fv3core._config as spec
import fv3core.utils.global_config as config
import fv3core.utils.gt4py_utils as utils
from fv3core.decorators import FrozenStencil
from fv3core.stencils.basic_operations import copy_defn
from fv3core.utils.typing import FloatField, FloatFieldI, FloatFieldIJ


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


def a2b_interpolation(
        qin: FloatField,
        qout: FloatField,
):
    with computation(PARALLEL), interval(...):
        qx = b2 * (qin[-2, 0, 0] + qin[1, 0, 0]) + b1 * (qin[-1, 0, 0] + qin)
        qy = b2 * (qin[0, -2, 0] + qin[0, 1, 0]) + b1 * (qin[0, -1, 0] + qin)
        qxx = a2 * (qx[0, -2, 0] + qx[0, 1, 0]) + a1 * (qx[0, -1, 0] + qx)
        qyy = a2 * (qy[-2, 0, 0] + qy[1, 0, 0]) + a1 * (qy[-1, 0, 0] + qy)
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

        if nk is None:
            nk = self.grid.npz - kstart


        self._a2b_interpolation_stencil = FrozenStencil(
            a2b_interpolation, origin=(self.grid.is_, self.grid.js, kstart), domain=(self.grid.nic + 1, self.grid.njc + 1, nk)
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
        self._a2b_interpolation_stencil(qin, qout)
        if self.replace:
            self._copy_stencil(
                qout,
                qin,
            )
