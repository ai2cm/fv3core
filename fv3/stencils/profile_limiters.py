import fv3.utils.gt4py_utils as utils
from fv3.utils.corners import fill2_4corners, fill_4corners
import gt4py.gtscript as gtscript
import fv3._config as spec
from gt4py.gtscript import computation, interval, PARALLEL
import fv3.stencils.copy_stencil as cp
from fv3.stencils.basic_operations import absolute_value

sd = utils.sd


def grid():
    return spec.grid


@utils.stencil()
def posdef_constraint_iv0(a4_1: sd, a4_2: sd, a4_3: sd, a4_4: sd):
    with computation(PARALLEL), interval(...):
        if a4_1 <= 0.0:
            a4_2 = a4_1
            a4_3 = a4_1
            a4_4 = 0.0
        else:
            a32 = a4_3 - a4_2
            abs_32 = absolute_value(a32)
            if abs_32 < -a4_4:
                if (a4_1 + 0.25 * (a4_3 - a4_2) ** 2 / a4_4 + a4_4 * 1.0 / 12.0) < 0.0:
                    if (a4_1 < a4_3) and (a4_1 < a4_2):
                        a4_3 = a4_1
                        a4_2 = a4_1
                        a4_4 = 0.0
                    elif a4_3 > a4_2:
                        a4_4 = 3.0 * (a4_2 - a4_1)
                        a4_3 = a4_2 - a4_4
                    else:
                        a4_4 = 3.0 * (a4_3 - a4_1)
                        a4_2 = a4_3 - a4_4
                else:
                    a4_2 = a4_2
            else:
                a4_2 = a4_2


@utils.stencil()
def posdef_constraint_iv1(a4_1: sd, a4_2: sd, a4_3: sd, a4_4: sd):
    with computation(PARALLEL), interval(...):
        da1 = a4_3 - a4_2
        da2 = da1 ** 2
        a6da = a4_4 * da1
        if ((a4_1 - a4_2) * (a4_1 - a4_3)) >= 0.0:
            a4_2 = a4_1
            a4_3 = a4_1
            a4_4 = 0.0
        else:
            if a6da < -1.0 * da2:
                a4_4 = 3.0 * (a4_2 - a4_1)
                a4_3 = a4_2 - a4_4
                a4_2 = a4_2
            elif a6da > da2:
                a4_4 = 3.0 * (a4_3 - a4_1)
                a4_2 = a4_3 - a4_4
                a4_3 = a4_3
            else:
                a4_2 = a4_2
                a4_3 = a4_3
                a4_4 = a4_4


@utils.stencil()
def ppm_constraint(a4_1: sd, a4_2: sd, a4_3: sd, a4_4: sd, extm: sd):
    with computation(PARALLEL), interval(...):
        da1 = a4_3 - a4_2
        da2 = da1 ** 2
        a6da = a4_4 * da1
        if extm == 1:
            a4_2 = a4_1
            a4_3 = a4_1
            a4_4 = 0.0
        else:
            if a6da < -da2:
                a4_4 = 3.0 * (a4_2 - a4_1)
                a4_3 = a4_2 - a4_4
            elif a6da > da2:
                a4_4 = 3.0 * (a4_3 - a4_1)
                a4_2 = a4_3 - a4_4
            else:
                a4_2 = a4_2


def compute(a4_1, a4_2, a4_3, a4_4, extm, iv, i1, i_extent, kstart, nk, js, j_extent):
    if iv == 0:
        posdef_constraint_iv0(
            a4_1,
            a4_2,
            a4_3,
            a4_4,
            origin=(i1, js, kstart),
            domain=(i_extent, j_extent, nk),
        )
    elif iv == 1:
        posdef_constraint_iv1(
            a4_1,
            a4_2,
            a4_3,
            a4_4,
            origin=(i1, js, kstart),
            domain=(i_extent, j_extent, nk),
        )
    else:
        ppm_constraint(
            a4_1,
            a4_2,
            a4_3,
            a4_4,
            extm,
            origin=(i1, js, kstart),
            domain=(i_extent, j_extent, nk),
        )
    return a4_1, a4_2, a4_3, a4_4
