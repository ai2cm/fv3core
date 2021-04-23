from typing import Optional
import numpy as np

import gt4py.gtscript as gtscript
from gt4py.gtscript import PARALLEL, FORWARD, computation, interval, horizontal, region, __INLINED

import fv3core._config as spec
import fv3core.utils.corners as corners
import fv3core.utils.gt4py_utils as utils
from fv3core.decorators import gtstencil
from fv3core.stencils.basic_operations import copy_stencil
from fv3core.utils.typing import FloatField, FloatFieldIJ, FloatFieldK


def copy_corners_x_nord(q: FloatField):
    from __externals__ import i_end, i_start, j_end, j_start, nord0, nord1, nord2, nord3

    with computation(PARALLEL), interval(0,1):
        if __INLINED(nord0 > 0):
            with horizontal(
                region[i_start - 3, j_start - 3], region[i_end + 3, j_start - 3]
            ):
                q = q[0, 5, 0]
            with horizontal(
                region[i_start - 2, j_start - 3], region[i_end + 3, j_start - 2]
            ):
                q = q[-1, 4, 0]
            with horizontal(
                region[i_start - 1, j_start - 3], region[i_end + 3, j_start - 1]
            ):
                q = q[-2, 3, 0]
            with horizontal(
                region[i_start - 3, j_start - 2], region[i_end + 2, j_start - 3]
            ):
                q = q[1, 4, 0]
            with horizontal(
                region[i_start - 2, j_start - 2], region[i_end + 2, j_start - 2]
            ):
                q = q[0, 3, 0]
            with horizontal(
                region[i_start - 1, j_start - 2], region[i_end + 2, j_start - 1]
            ):
                q = q[-1, 2, 0]
            with horizontal(
                region[i_start - 3, j_start - 1], region[i_end + 1, j_start - 3]
            ):
                q = q[2, 3, 0]
            with horizontal(
                region[i_start - 2, j_start - 1], region[i_end + 1, j_start - 2]
            ):
                q = q[1, 2, 0]
            with horizontal(
                region[i_start - 1, j_start - 1], region[i_end + 1, j_start - 1]
            ):
                q = q[0, 1, 0]
            with horizontal(region[i_start - 3, j_end + 1], region[i_end + 1, j_end + 3]):
                q = q[2, -3, 0]
            with horizontal(region[i_start - 2, j_end + 1], region[i_end + 1, j_end + 2]):
                q = q[1, -2, 0]
            with horizontal(region[i_start - 1, j_end + 1], region[i_end + 1, j_end + 1]):
                q = q[0, -1, 0]
            with horizontal(region[i_start - 3, j_end + 2], region[i_end + 2, j_end + 3]):
                q = q[1, -4, 0]
            with horizontal(region[i_start - 2, j_end + 2], region[i_end + 2, j_end + 2]):
                q = q[0, -3, 0]
            with horizontal(region[i_start - 1, j_end + 2], region[i_end + 2, j_end + 1]):
                q = q[-1, -2, 0]
            with horizontal(region[i_start - 3, j_end + 3], region[i_end + 3, j_end + 3]):
                q = q[0, -5, 0]
            with horizontal(region[i_start - 2, j_end + 3], region[i_end + 3, j_end + 2]):
                q = q[-1, -4, 0]
            with horizontal(region[i_start - 1, j_end + 3], region[i_end + 3, j_end + 1]):
                q = q[-2, -3, 0]
    with computation(PARALLEL), interval(1,2):
        if __INLINED(nord1 > 0):
            with horizontal(
                region[i_start - 3, j_start - 3], region[i_end + 3, j_start - 3]
            ):
                q = q[0, 5, 0]
            with horizontal(
                region[i_start - 2, j_start - 3], region[i_end + 3, j_start - 2]
            ):
                q = q[-1, 4, 0]
            with horizontal(
                region[i_start - 1, j_start - 3], region[i_end + 3, j_start - 1]
            ):
                q = q[-2, 3, 0]
            with horizontal(
                region[i_start - 3, j_start - 2], region[i_end + 2, j_start - 3]
            ):
                q = q[1, 4, 0]
            with horizontal(
                region[i_start - 2, j_start - 2], region[i_end + 2, j_start - 2]
            ):
                q = q[0, 3, 0]
            with horizontal(
                region[i_start - 1, j_start - 2], region[i_end + 2, j_start - 1]
            ):
                q = q[-1, 2, 0]
            with horizontal(
                region[i_start - 3, j_start - 1], region[i_end + 1, j_start - 3]
            ):
                q = q[2, 3, 0]
            with horizontal(
                region[i_start - 2, j_start - 1], region[i_end + 1, j_start - 2]
            ):
                q = q[1, 2, 0]
            with horizontal(
                region[i_start - 1, j_start - 1], region[i_end + 1, j_start - 1]
            ):
                q = q[0, 1, 0]
            with horizontal(region[i_start - 3, j_end + 1], region[i_end + 1, j_end + 3]):
                q = q[2, -3, 0]
            with horizontal(region[i_start - 2, j_end + 1], region[i_end + 1, j_end + 2]):
                q = q[1, -2, 0]
            with horizontal(region[i_start - 1, j_end + 1], region[i_end + 1, j_end + 1]):
                q = q[0, -1, 0]
            with horizontal(region[i_start - 3, j_end + 2], region[i_end + 2, j_end + 3]):
                q = q[1, -4, 0]
            with horizontal(region[i_start - 2, j_end + 2], region[i_end + 2, j_end + 2]):
                q = q[0, -3, 0]
            with horizontal(region[i_start - 1, j_end + 2], region[i_end + 2, j_end + 1]):
                q = q[-1, -2, 0]
            with horizontal(region[i_start - 3, j_end + 3], region[i_end + 3, j_end + 3]):
                q = q[0, -5, 0]
            with horizontal(region[i_start - 2, j_end + 3], region[i_end + 3, j_end + 2]):
                q = q[-1, -4, 0]
            with horizontal(region[i_start - 1, j_end + 3], region[i_end + 3, j_end + 1]):
                q = q[-2, -3, 0]
    with computation(PARALLEL), interval(2,3):
        if __INLINED(nord2 > 0):
            with horizontal(
                region[i_start - 3, j_start - 3], region[i_end + 3, j_start - 3]
            ):
                q = q[0, 5, 0]
            with horizontal(
                region[i_start - 2, j_start - 3], region[i_end + 3, j_start - 2]
            ):
                q = q[-1, 4, 0]
            with horizontal(
                region[i_start - 1, j_start - 3], region[i_end + 3, j_start - 1]
            ):
                q = q[-2, 3, 0]
            with horizontal(
                region[i_start - 3, j_start - 2], region[i_end + 2, j_start - 3]
            ):
                q = q[1, 4, 0]
            with horizontal(
                region[i_start - 2, j_start - 2], region[i_end + 2, j_start - 2]
            ):
                q = q[0, 3, 0]
            with horizontal(
                region[i_start - 1, j_start - 2], region[i_end + 2, j_start - 1]
            ):
                q = q[-1, 2, 0]
            with horizontal(
                region[i_start - 3, j_start - 1], region[i_end + 1, j_start - 3]
            ):
                q = q[2, 3, 0]
            with horizontal(
                region[i_start - 2, j_start - 1], region[i_end + 1, j_start - 2]
            ):
                q = q[1, 2, 0]
            with horizontal(
                region[i_start - 1, j_start - 1], region[i_end + 1, j_start - 1]
            ):
                q = q[0, 1, 0]
            with horizontal(region[i_start - 3, j_end + 1], region[i_end + 1, j_end + 3]):
                q = q[2, -3, 0]
            with horizontal(region[i_start - 2, j_end + 1], region[i_end + 1, j_end + 2]):
                q = q[1, -2, 0]
            with horizontal(region[i_start - 1, j_end + 1], region[i_end + 1, j_end + 1]):
                q = q[0, -1, 0]
            with horizontal(region[i_start - 3, j_end + 2], region[i_end + 2, j_end + 3]):
                q = q[1, -4, 0]
            with horizontal(region[i_start - 2, j_end + 2], region[i_end + 2, j_end + 2]):
                q = q[0, -3, 0]
            with horizontal(region[i_start - 1, j_end + 2], region[i_end + 2, j_end + 1]):
                q = q[-1, -2, 0]
            with horizontal(region[i_start - 3, j_end + 3], region[i_end + 3, j_end + 3]):
                q = q[0, -5, 0]
            with horizontal(region[i_start - 2, j_end + 3], region[i_end + 3, j_end + 2]):
                q = q[-1, -4, 0]
            with horizontal(region[i_start - 1, j_end + 3], region[i_end + 3, j_end + 1]):
                q = q[-2, -3, 0]
    with computation(PARALLEL), interval(3,None):
        if __INLINED(nord3 > 0):
            with horizontal(
                region[i_start - 3, j_start - 3], region[i_end + 3, j_start - 3]
            ):
                q = q[0, 5, 0]
            with horizontal(
                region[i_start - 2, j_start - 3], region[i_end + 3, j_start - 2]
            ):
                q = q[-1, 4, 0]
            with horizontal(
                region[i_start - 1, j_start - 3], region[i_end + 3, j_start - 1]
            ):
                q = q[-2, 3, 0]
            with horizontal(
                region[i_start - 3, j_start - 2], region[i_end + 2, j_start - 3]
            ):
                q = q[1, 4, 0]
            with horizontal(
                region[i_start - 2, j_start - 2], region[i_end + 2, j_start - 2]
            ):
                q = q[0, 3, 0]
            with horizontal(
                region[i_start - 1, j_start - 2], region[i_end + 2, j_start - 1]
            ):
                q = q[-1, 2, 0]
            with horizontal(
                region[i_start - 3, j_start - 1], region[i_end + 1, j_start - 3]
            ):
                q = q[2, 3, 0]
            with horizontal(
                region[i_start - 2, j_start - 1], region[i_end + 1, j_start - 2]
            ):
                q = q[1, 2, 0]
            with horizontal(
                region[i_start - 1, j_start - 1], region[i_end + 1, j_start - 1]
            ):
                q = q[0, 1, 0]
            with horizontal(region[i_start - 3, j_end + 1], region[i_end + 1, j_end + 3]):
                q = q[2, -3, 0]
            with horizontal(region[i_start - 2, j_end + 1], region[i_end + 1, j_end + 2]):
                q = q[1, -2, 0]
            with horizontal(region[i_start - 1, j_end + 1], region[i_end + 1, j_end + 1]):
                q = q[0, -1, 0]
            with horizontal(region[i_start - 3, j_end + 2], region[i_end + 2, j_end + 3]):
                q = q[1, -4, 0]
            with horizontal(region[i_start - 2, j_end + 2], region[i_end + 2, j_end + 2]):
                q = q[0, -3, 0]
            with horizontal(region[i_start - 1, j_end + 2], region[i_end + 2, j_end + 1]):
                q = q[-1, -2, 0]
            with horizontal(region[i_start - 3, j_end + 3], region[i_end + 3, j_end + 3]):
                q = q[0, -5, 0]
            with horizontal(region[i_start - 2, j_end + 3], region[i_end + 3, j_end + 2]):
                q = q[-1, -4, 0]
            with horizontal(region[i_start - 1, j_end + 3], region[i_end + 3, j_end + 1]):
                q = q[-2, -3, 0]


def copy_corners_y_nord(q: FloatField):
    from __externals__ import i_end, i_start, j_end, j_start, nord0, nord1, nord2, nord3

    with computation(PARALLEL), interval(0,1):
        if __INLINED(nord0 > 0):
            with horizontal(
                region[i_start - 3, j_start - 3], region[i_start - 3, j_end + 3]
            ):
                q = q[5, 0, 0]
            with horizontal(
                region[i_start - 2, j_start - 3], region[i_start - 3, j_end + 2]
            ):
                q = q[4, 1, 0]
            with horizontal(
                region[i_start - 1, j_start - 3], region[i_start - 3, j_end + 1]
            ):
                q = q[3, 2, 0]
            with horizontal(
                region[i_start - 3, j_start - 2], region[i_start - 2, j_end + 3]
            ):
                q = q[4, -1, 0]
            with horizontal(
                region[i_start - 2, j_start - 2], region[i_start - 2, j_end + 2]
            ):
                q = q[3, 0, 0]
            with horizontal(
                region[i_start - 1, j_start - 2], region[i_start - 2, j_end + 1]
            ):
                q = q[2, 1, 0]
            with horizontal(
                region[i_start - 3, j_start - 1], region[i_start - 1, j_end + 3]
            ):
                q = q[3, -2, 0]
            with horizontal(
                region[i_start - 2, j_start - 1], region[i_start - 1, j_end + 2]
            ):
                q = q[2, -1, 0]
            with horizontal(
                region[i_start - 1, j_start - 1], region[i_start - 1, j_end + 1]
            ):
                q = q[1, 0, 0]
            with horizontal(region[i_end + 1, j_start - 3], region[i_end + 3, j_end + 1]):
                q = q[-3, 2, 0]
            with horizontal(region[i_end + 2, j_start - 3], region[i_end + 3, j_end + 2]):
                q = q[-4, 1, 0]
            with horizontal(region[i_end + 3, j_start - 3], region[i_end + 3, j_end + 3]):
                q = q[-5, 0, 0]
            with horizontal(region[i_end + 1, j_start - 2], region[i_end + 2, j_end + 1]):
                q = q[-2, 1, 0]
            with horizontal(region[i_end + 2, j_start - 2], region[i_end + 2, j_end + 2]):
                q = q[-3, 0, 0]
            with horizontal(region[i_end + 3, j_start - 2], region[i_end + 2, j_end + 3]):
                q = q[-4, -1, 0]
            with horizontal(region[i_end + 1, j_start - 1], region[i_end + 1, j_end + 1]):
                q = q[-1, 0, 0]
            with horizontal(region[i_end + 2, j_start - 1], region[i_end + 1, j_end + 2]):
                q = q[-2, -1, 0]
            with horizontal(region[i_end + 3, j_start - 1], region[i_end + 1, j_end + 3]):
                q = q[-3, -2, 0]
    with computation(PARALLEL), interval(1,2):
        if __INLINED(nord1 > 0):
            with horizontal(
                region[i_start - 3, j_start - 3], region[i_start - 3, j_end + 3]
            ):
                q = q[5, 0, 0]
            with horizontal(
                region[i_start - 2, j_start - 3], region[i_start - 3, j_end + 2]
            ):
                q = q[4, 1, 0]
            with horizontal(
                region[i_start - 1, j_start - 3], region[i_start - 3, j_end + 1]
            ):
                q = q[3, 2, 0]
            with horizontal(
                region[i_start - 3, j_start - 2], region[i_start - 2, j_end + 3]
            ):
                q = q[4, -1, 0]
            with horizontal(
                region[i_start - 2, j_start - 2], region[i_start - 2, j_end + 2]
            ):
                q = q[3, 0, 0]
            with horizontal(
                region[i_start - 1, j_start - 2], region[i_start - 2, j_end + 1]
            ):
                q = q[2, 1, 0]
            with horizontal(
                region[i_start - 3, j_start - 1], region[i_start - 1, j_end + 3]
            ):
                q = q[3, -2, 0]
            with horizontal(
                region[i_start - 2, j_start - 1], region[i_start - 1, j_end + 2]
            ):
                q = q[2, -1, 0]
            with horizontal(
                region[i_start - 1, j_start - 1], region[i_start - 1, j_end + 1]
            ):
                q = q[1, 0, 0]
            with horizontal(region[i_end + 1, j_start - 3], region[i_end + 3, j_end + 1]):
                q = q[-3, 2, 0]
            with horizontal(region[i_end + 2, j_start - 3], region[i_end + 3, j_end + 2]):
                q = q[-4, 1, 0]
            with horizontal(region[i_end + 3, j_start - 3], region[i_end + 3, j_end + 3]):
                q = q[-5, 0, 0]
            with horizontal(region[i_end + 1, j_start - 2], region[i_end + 2, j_end + 1]):
                q = q[-2, 1, 0]
            with horizontal(region[i_end + 2, j_start - 2], region[i_end + 2, j_end + 2]):
                q = q[-3, 0, 0]
            with horizontal(region[i_end + 3, j_start - 2], region[i_end + 2, j_end + 3]):
                q = q[-4, -1, 0]
            with horizontal(region[i_end + 1, j_start - 1], region[i_end + 1, j_end + 1]):
                q = q[-1, 0, 0]
            with horizontal(region[i_end + 2, j_start - 1], region[i_end + 1, j_end + 2]):
                q = q[-2, -1, 0]
            with horizontal(region[i_end + 3, j_start - 1], region[i_end + 1, j_end + 3]):
                q = q[-3, -2, 0]
    with computation(PARALLEL), interval(2,3):
        if __INLINED(nord2 > 0):
            with horizontal(
                region[i_start - 3, j_start - 3], region[i_start - 3, j_end + 3]
            ):
                q = q[5, 0, 0]
            with horizontal(
                region[i_start - 2, j_start - 3], region[i_start - 3, j_end + 2]
            ):
                q = q[4, 1, 0]
            with horizontal(
                region[i_start - 1, j_start - 3], region[i_start - 3, j_end + 1]
            ):
                q = q[3, 2, 0]
            with horizontal(
                region[i_start - 3, j_start - 2], region[i_start - 2, j_end + 3]
            ):
                q = q[4, -1, 0]
            with horizontal(
                region[i_start - 2, j_start - 2], region[i_start - 2, j_end + 2]
            ):
                q = q[3, 0, 0]
            with horizontal(
                region[i_start - 1, j_start - 2], region[i_start - 2, j_end + 1]
            ):
                q = q[2, 1, 0]
            with horizontal(
                region[i_start - 3, j_start - 1], region[i_start - 1, j_end + 3]
            ):
                q = q[3, -2, 0]
            with horizontal(
                region[i_start - 2, j_start - 1], region[i_start - 1, j_end + 2]
            ):
                q = q[2, -1, 0]
            with horizontal(
                region[i_start - 1, j_start - 1], region[i_start - 1, j_end + 1]
            ):
                q = q[1, 0, 0]
            with horizontal(region[i_end + 1, j_start - 3], region[i_end + 3, j_end + 1]):
                q = q[-3, 2, 0]
            with horizontal(region[i_end + 2, j_start - 3], region[i_end + 3, j_end + 2]):
                q = q[-4, 1, 0]
            with horizontal(region[i_end + 3, j_start - 3], region[i_end + 3, j_end + 3]):
                q = q[-5, 0, 0]
            with horizontal(region[i_end + 1, j_start - 2], region[i_end + 2, j_end + 1]):
                q = q[-2, 1, 0]
            with horizontal(region[i_end + 2, j_start - 2], region[i_end + 2, j_end + 2]):
                q = q[-3, 0, 0]
            with horizontal(region[i_end + 3, j_start - 2], region[i_end + 2, j_end + 3]):
                q = q[-4, -1, 0]
            with horizontal(region[i_end + 1, j_start - 1], region[i_end + 1, j_end + 1]):
                q = q[-1, 0, 0]
            with horizontal(region[i_end + 2, j_start - 1], region[i_end + 1, j_end + 2]):
                q = q[-2, -1, 0]
            with horizontal(region[i_end + 3, j_start - 1], region[i_end + 1, j_end + 3]):
                q = q[-3, -2, 0]
    with computation(PARALLEL), interval(3,None):
        if __INLINED(nord3 > 0):
            with horizontal(
                region[i_start - 3, j_start - 3], region[i_start - 3, j_end + 3]
            ):
                q = q[5, 0, 0]
            with horizontal(
                region[i_start - 2, j_start - 3], region[i_start - 3, j_end + 2]
            ):
                q = q[4, 1, 0]
            with horizontal(
                region[i_start - 1, j_start - 3], region[i_start - 3, j_end + 1]
            ):
                q = q[3, 2, 0]
            with horizontal(
                region[i_start - 3, j_start - 2], region[i_start - 2, j_end + 3]
            ):
                q = q[4, -1, 0]
            with horizontal(
                region[i_start - 2, j_start - 2], region[i_start - 2, j_end + 2]
            ):
                q = q[3, 0, 0]
            with horizontal(
                region[i_start - 1, j_start - 2], region[i_start - 2, j_end + 1]
            ):
                q = q[2, 1, 0]
            with horizontal(
                region[i_start - 3, j_start - 1], region[i_start - 1, j_end + 3]
            ):
                q = q[3, -2, 0]
            with horizontal(
                region[i_start - 2, j_start - 1], region[i_start - 1, j_end + 2]
            ):
                q = q[2, -1, 0]
            with horizontal(
                region[i_start - 1, j_start - 1], region[i_start - 1, j_end + 1]
            ):
                q = q[1, 0, 0]
            with horizontal(region[i_end + 1, j_start - 3], region[i_end + 3, j_end + 1]):
                q = q[-3, 2, 0]
            with horizontal(region[i_end + 2, j_start - 3], region[i_end + 3, j_end + 2]):
                q = q[-4, 1, 0]
            with horizontal(region[i_end + 3, j_start - 3], region[i_end + 3, j_end + 3]):
                q = q[-5, 0, 0]
            with horizontal(region[i_end + 1, j_start - 2], region[i_end + 2, j_end + 1]):
                q = q[-2, 1, 0]
            with horizontal(region[i_end + 2, j_start - 2], region[i_end + 2, j_end + 2]):
                q = q[-3, 0, 0]
            with horizontal(region[i_end + 3, j_start - 2], region[i_end + 2, j_end + 3]):
                q = q[-4, -1, 0]
            with horizontal(region[i_end + 1, j_start - 1], region[i_end + 1, j_end + 1]):
                q = q[-1, 0, 0]
            with horizontal(region[i_end + 2, j_start - 1], region[i_end + 1, j_end + 2]):
                q = q[-2, -1, 0]
            with horizontal(region[i_end + 3, j_start - 1], region[i_end + 1, j_end + 3]):
                q = q[-3, -2, 0]

@gtstencil()
def calc_damp(nord: FloatFieldK, damp_c: FloatFieldK, damp4: FloatFieldK, da_min:float):
    with computation(FORWARD), interval(...):
        damp4 = (damp_c * da_min) ** (nord + 1)

@gtstencil()
def fx_calc_stencil_region(q: FloatField, del6_v: FloatFieldIJ, fx: FloatField, order: int):
    from __externals__ import i_end, i_start, j_end, j_start, nmax, nord0, nord1, nord2, nord3

    with computation(PARALLEL), interval(0,1):
        if __INLINED(nord0 == 0):
            with horizontal(
                region[i_start + nmax: i_end - nmax, j_start + nmax: j_end + nmax]
            ):
                fx = fx_calculation(q, del6_v, order)
        else:
            fx = fx_calculation(q, del6_v, order)
    with computation(PARALLEL), interval(1,2):
        if __INLINED(nord1 == 0):
            with horizontal(
                region[i_start + nmax: i_end - nmax, j_start + nmax: j_end + nmax]
            ):
                fx = fx_calculation(q, del6_v, order)
        else:
            fx = fx_calculation(q, del6_v, order)
    with computation(PARALLEL), interval(2,3):
        if __INLINED(nord2 == 0):
            with horizontal(
                region[i_start + nmax: i_end - nmax, j_start + nmax: j_end + nmax]
            ):
                fx = fx_calculation(q, del6_v, order)
        else:
            fx = fx_calculation(q, del6_v, order)
    with computation(PARALLEL), interval(3,None):
        if __INLINED(nord3 == 0):
            with horizontal(
                region[i_start + nmax: i_end - nmax, j_start + nmax: j_end + nmax]
            ):
                fx = fx_calculation(q, del6_v, order)
        else:
            fx = fx_calculation(q, del6_v, order)

@gtstencil()
def fy_calc_stencil_region(q: FloatField, del6_u: FloatFieldIJ, fy: FloatField, order: int):
    from __externals__ import i_end, i_start, j_end, j_start, nmax, nord0, nord1, nord2, nord3

    with computation(PARALLEL), interval(0,1):
        if __INLINED(nord0 == 0):
            with horizontal(
                region[i_start + nmax: i_end - nmax, j_start + nmax: j_end + nmax]
            ):
                fy = fy_calculation(q, del6_u, order)
        else:
            fy = fy_calculation(q, del6_u, order)
    with computation(PARALLEL), interval(1,2):
        if __INLINED(nord1 == 0):
            with horizontal(
                region[i_start + nmax: i_end - nmax, j_start + nmax: j_end + nmax]
            ):
                fy = fy_calculation(q, del6_u, order)
        else:
            fy = fy_calculation(q, del6_u, order)
    with computation(PARALLEL), interval(2,3):
        if __INLINED(nord2 == 0):
            with horizontal(
                region[i_start + nmax: i_end - nmax, j_start + nmax: j_end + nmax]
            ):
                fy = fy_calculation(q, del6_u, order)
        else:
            fy = fy_calculation(q, del6_u, order)
    with computation(PARALLEL), interval(3,None):
        if __INLINED(nord3 == 0):
            with horizontal(
                region[i_start + nmax: i_end - nmax, j_start + nmax: j_end + nmax]
            ):
                fy = fy_calculation(q, del6_u, order)
        else:
            fy = fy_calculation(q, del6_u, order)
        

@gtstencil()
def fx_calc_stencil_column(q: FloatField, del6_v: FloatFieldIJ, fx: FloatField, nord: FloatFieldK):
    with computation(PARALLEL), interval(...):
        if nord > 0:
            fx = fx_calculation(q, del6_v, nord + 2)

@gtstencil()
def fy_calc_stencil_column(q: FloatField, del6_u: FloatFieldIJ, fy: FloatField, nord: FloatFieldK):
    with computation(PARALLEL), interval(...):
        if nord > 0:
            fy = fy_calculation(q, del6_u, nord + 2)


@gtscript.function
def fx_calculation(q: FloatField, del6_v: FloatField, order: int):
    fx = del6_v * (q[-1, 0, 0] - q)
    fx = -1.0 * fx if order > 1 else fx
    return fx


@gtscript.function
def fy_calculation(q: FloatField, del6_u: FloatField, order: int):
    fy = del6_u * (q[0, -1, 0] - q)
    fy = fy * -1 if order > 1 else fy
    return fy


# WARNING: untested
@gtstencil()
def fx_firstorder_use_sg(
    q: FloatField,
    sin_sg1: FloatField,
    sin_sg3: FloatField,
    dy: FloatField,
    rdxc: FloatField,
    fx: FloatField,
):
    with computation(PARALLEL), interval(...):
        fx[0, 0, 0] = (
            0.5 * (sin_sg3[-1, 0, 0] + sin_sg1) * dy * (q[-1, 0, 0] - q) * rdxc
        )


# WARNING: untested
@gtstencil()
def fy_firstorder_use_sg(
    q: FloatField,
    sin_sg2: FloatField,
    sin_sg4: FloatField,
    dx: FloatField,
    rdyc: FloatField,
    fy: FloatField,
):
    with computation(PARALLEL), interval(...):
        fy[0, 0, 0] = (
            0.5 * (sin_sg4[0, -1, 0] + sin_sg2) * dx * (q[0, -1, 0] - q) * rdyc
        )


def d2_highorder_stencil(
    fx: FloatField, fy: FloatField, rarea: FloatFieldIJ, d2: FloatField
):
    from __externals__ import nord0, nord1, nord2, nord3

    with computation(PARALLEL), interval(0,1):
        if __INLINED(nord0 > 0):
            d2 = d2_highorder(fx, fy, rarea)
    with computation(PARALLEL), interval(1,2):
        if __INLINED(nord1 > 0):
            d2 = d2_highorder(fx, fy, rarea)
    with computation(PARALLEL), interval(2,3):
        if __INLINED(nord2 > 0):
            d2 = d2_highorder(fx, fy, rarea)
    with computation(PARALLEL), interval(3,None):
        if __INLINED(nord3 > 0):
            d2 = d2_highorder(fx, fy, rarea)


@gtscript.function
def d2_highorder(fx: FloatField, fy: FloatField, rarea: FloatField):
    d2 = (fx - fx[1, 0, 0] + fy - fy[0, 1, 0]) * rarea
    return d2


def d2_damp_interval(q: FloatField, d2: FloatField, damp: FloatFieldK):
    from __externals__ import i_end, i_start, j_end, j_start, nmax, nord0, nord1, nord2, nord3

    with computation(PARALLEL), interval(0,1):
        if __INLINED(nord0 == 0):
            with horizontal(
                region[i_start + nmax: i_end - nmax, j_start + nmax: j_end + nmax]
            ):
                d2[0, 0, 0] = damp * q
        else:
            d2[0, 0, 0] = damp * q
    with computation(PARALLEL), interval(1,2):
        if __INLINED(nord1 == 0):
            with horizontal(
                region[i_start + nmax: i_end - nmax, j_start + nmax: j_end + nmax]
            ):
                d2[0, 0, 0] = damp * q
        else:
            d2[0, 0, 0] = damp * q
    with computation(PARALLEL), interval(2,3):
        if __INLINED(nord2 == 0):
            with horizontal(
                region[i_start + nmax: i_end - nmax, j_start + nmax: j_end + nmax]
            ):
                d2[0, 0, 0] = damp * q
        else:
            d2[0, 0, 0] = damp * q
    with computation(PARALLEL), interval(3,None):
        if __INLINED(nord3 == 0):
            with horizontal(
                region[i_start + nmax: i_end - nmax, j_start + nmax: j_end + nmax]
            ):
                d2[0, 0, 0] = damp * q
        else:
            d2[0, 0, 0] = damp * q


def copy_stencil_interval(q_in: FloatField, q_out: FloatField):
    from __externals__ import i_end, i_start, j_end, j_start, nmax, nord0, nord1, nord2, nord3

    with computation(PARALLEL), interval(0,1):
        if __INLINED(nord0 == 0):
            with horizontal(
                region[i_start + nmax: i_end - nmax, j_start + nmax: j_end - nmax]
            ):
                q_out = q_in
        else:
            q_out = q_in
    with computation(PARALLEL), interval(1,2):
        if __INLINED(nord1 == 0):
            with horizontal(
                region[i_start + nmax: i_end - nmax, j_start + nmax: j_end - nmax]
            ):
                q_out = q_in
        else:
            q_out = q_in
    with computation(PARALLEL), interval(2,3):
        if __INLINED(nord2 == 0):
            with horizontal(
                region[i_start + nmax: i_end - nmax, j_start + nmax: j_end - nmax]
            ):
                q_out = q_in
        else:
            q_out = q_in
    with computation(PARALLEL), interval(3,None):
        if __INLINED(nord3 == 0):
            with horizontal(
                region[i_start + nmax: i_end - nmax, j_start + nmax: j_end - nmax]
            ):
                q_out = q_in
        else: q_out = q_in

@gtstencil()
def add_diffusive_component(
    fx: FloatField, fx2: FloatField, fy: FloatField, fy2: FloatField
):
    with computation(PARALLEL), interval(...):
        fx[0, 0, 0] = fx + fx2
        fy[0, 0, 0] = fy + fy2


@gtstencil()
def diffusive_damp(
    fx: FloatField,
    fx2: FloatField,
    fy: FloatField,
    fy2: FloatField,
    mass: FloatField,
    damp: FloatFieldK,
):
    with computation(PARALLEL), interval(...):
        fx[0, 0, 0] = fx + 0.5 * damp * (mass[-1, 0, 0] + mass) * fx2
        fy[0, 0, 0] = fy + 0.5 * damp * (mass[0, -1, 0] + mass) * fy2


def compute_delnflux_no_sg(
    q: FloatField,
    fx: FloatField,
    fy: FloatField,
    nord: int,
    damp_c: float,
    d2: Optional["FloatField"] = None,
    mass: Optional["FloatField"] = None,
):
    """
    Del-n damping for fluxes, where n = 2 * nord + 2
    Args:
        q: Field for which to calculate damped fluxes (in)
        fx: x-flux on A-grid (inout)
        fy: y-flux on A-grid (inout)
        nord: Order of divergence damping (in)
        damp_c: damping coefficient (in)
        d2: A damped copy of the q field (in)
        mass: Mass to weight the diffusive flux by (in)
    """

    grid = spec.grid
    nk = grid.npz
    full_origin = (grid.isd, grid.jsd, 0)
    if d2 is None:
        d2 = utils.make_storage_from_shape(
            q.shape, full_origin, cache_key="delnflux_d2"
        )
    if damp_c <= 1e-4:
        return fx, fy
    
    damp_3d = utils.make_storage_from_shape((1,1, nk)) # fields must be 3d to assign to them
    calc_damp(nord, damp_c, damp_3d, grid.da_min, origin=(0,0,0), domain=(1,1,nk))
    damp = utils.make_storage_data(damp_3d[0,0,:], (nk,), (0,))

    fx2 = utils.make_storage_from_shape(q.shape, full_origin, cache_key="delnflux_fx2")
    fy2 = utils.make_storage_from_shape(q.shape, full_origin, cache_key="delnflux_fy2")
    diffuse_origin = (grid.is_, grid.js, 0)
    extended_domain = (grid.nic + 1, grid.njc + 1, nk)

    compute_no_sg(q, fx2, fy2, nord, damp, d2, mass, conditional_calc=False)

    if mass is None:
        add_diffusive_component(fx, fx2, fy, fy2, origin=diffuse_origin, domain=extended_domain)
    else:
        # TODO: To join these stencils you need to overcompute, making the edges
        # 'wrong', but not actually used, separating now for comparison sanity.

        # diffusive_damp(fx, fx2, fy, fy2, mass, damp, origin=diffuse_origin,
        # domain=(grid.nic + 1, grid.njc + 1, nk))
        diffusive_damp(
            fx, fx2, fy, fy2, mass, damp, origin=diffuse_origin, domain=extended_domain
        )
    return fx, fy


def compute_no_sg(
        q,
        fx2,
        fy2,
        nord,
        damp_c,
        d2,
        mass=None,
        conditional_calc=True,
        column_check=False,
):
    if (conditional_calc==True) and (column_check==False):
        if damp_c[0] <= 1e-5: #dcon_threshold
            raise Exception("damp <= 1e-5 in column_cols is untested")
    if max(nord[:]) > 3:
        raise Exception("nord must be less than 3")
    if not np.all(n in [0,2,3] for n in nord[:]):
        raise NotImplementedError("nord must have values 0, 2, or 3")
    nmax = max(nord[:])
    grid = spec.grid
    # nord = int(nord)
    i1 = grid.is_ - 1 - nmax
    i2 = grid.ie + 1 + nmax
    j1 = grid.js - 1 - nmax
    j2 = grid.je + 1 + nmax
    nk = grid.npz
    origin_d2 = (i1, j1, 0)
    domain_d2 = (i2 - i1 + 1, j2 - j1 + 1, nk)
    f1_ny = grid.je - grid.js + 1 + 2 * nmax
    f1_nx = grid.ie - grid.is_ + 2 + 2 * nmax
    fx_origin = (grid.is_ - nmax, grid.js - nmax, 0)
    if mass is None:
        d2_damp = gtstencil(d2_damp_interval, externals={"nmax": nmax, "nord0":nord[0], "nord1":nord[1], "nord2":nord[2], "nord3":nord[3]})
        d2_damp(q, d2, damp_c, origin=origin_d2, domain=domain_d2)
    else:
        new_copy_stencil = gtstencil(copy_stencil_interval, externals={"nmax": nmax, "nord0":nord[0], "nord1":nord[1], "nord2":nord[2], "nord3":nord[3]})
        new_copy_stencil(q, d2, origin=origin_d2, domain=domain_d2)

    conditional_corner_copy_x = gtstencil(copy_corners_x_nord, externals={"nord0":nord[0], "nord1":nord[1], "nord2":nord[2], "nord3":nord[3]})
    conditional_corner_copy_y = gtstencil(copy_corners_y_nord, externals={"nord0":nord[0], "nord1":nord[1], "nord2":nord[2], "nord3":nord[3]})

    d2_stencil = gtstencil(d2_highorder_stencil, externals={"nord0":nord[0], "nord1":nord[1], "nord2":nord[2], "nord3":nord[3]})

    conditional_corner_copy_x(
        d2, origin=(grid.isd, grid.jsd, 0), domain=(grid.nid, grid.njd, nk)
    )

    fx_calc_stencil = gtstencil(fx_calc_stencil_region, externals={"nmax": nmax, "nord0":nord[0], "nord1":nord[1], "nord2":nord[2], "nord3":nord[3]})
    fx_calc_stencil(
        d2, grid.del6_v, fx2, order=1, origin=fx_origin, domain=(f1_nx, f1_ny, nk)
    )

    conditional_corner_copy_y(
        d2, origin=(grid.isd, grid.jsd, 0), domain=(grid.nid, grid.njd, nk)
    )

    fy_calc_stencil = gtstencil(fy_calc_stencil_region, externals={"nmax": nmax, "nord0":nord[0], "nord1":nord[1], "nord2":nord[2], "nord3":nord[3]})
    fy_calc_stencil(
        d2,
        grid.del6_u,
        fy2,
        order=1,
        origin=fx_origin,
        domain=(f1_nx - 1, f1_ny + 1, nk),
    )

    for n in range(nmax):
        nt = nmax - 1 - n
        nt_origin = (grid.is_ - nt - 1, grid.js - nt - 1, 0)
        nt_ny = grid.je - grid.js + 3 + 2 * nt
        nt_nx = grid.ie - grid.is_ + 3 + 2 * nt
        d2_stencil(
            fx2, fy2, grid.rarea, d2, origin=nt_origin, domain=(nt_nx, nt_ny, nk)
        )
        conditional_corner_copy_x(
            d2, origin=(grid.isd, grid.jsd, 0), domain=(grid.nid, grid.njd, nk)
        )
        nt_origin = (grid.is_ - nt, grid.js - nt, 0)
        fx_calc_stencil_column(
            d2,
            grid.del6_v,
            fx2,
            n,
            origin=nt_origin,
            domain=(nt_nx - 1, nt_ny - 2, nk),
        )
        conditional_corner_copy_y(
            d2, origin=(grid.isd, grid.jsd, 0), domain=(grid.nid, grid.njd, nk)
        )

        fy_calc_stencil_column(
            d2,
            grid.del6_u,
            fy2,
            n,
            origin=nt_origin,
            domain=(nt_nx - 2, nt_ny - 1, nk),
        )