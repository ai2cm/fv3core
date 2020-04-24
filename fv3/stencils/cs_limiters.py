import fv3.utils.gt4py_utils as utils
from fv3.utils.corners import fill2_4corners, fill_4corners
import gt4py.gtscript as gtscript
import fv3._config as spec
from gt4py.gtscript import computation, interval, PARALLEL
import fv3.stencils.copy_stencil as cp

sd = utils.sd


def grid():
    return spec.grid


@gtscript.function
def absolute_value(in_array):
    abs_value = in_array if in_array > 0 else -in_array
    return abs_value

# @gtscript.function
# def neg_a4(a4_1, a4_2, a4_3, a4_4):
#     a4_2 = a4_1
#     a4_3 = a4_1
#     a4_4 = 0.
#     return a4_2, a4_3, a4_4

# @gtscript.function
# def calc_2_a3(a4_1, a4_2, a4_3, a4_4):
#     a4_4 = 3.*(a4_2-a4_1)
#     a4_3 = a4_2 - a4_4
#     return a4_2, a4_3, a4_4

# @gtscript.function
# def calc_2_a2(a4_1, a4_2, a4_3, a4_4):
#     a4_4 = 3.*(a4_3-a4_1)
#     a4_2 = a4_3 - a4_4
#     return a4_2, a4_3, a4_4

# @gtscript.function
# def calc_a4s(a4_1, a4_2, a4_3, a4_4):
#     a4_2, a4_3, a4_4 = neg_a4(a4_1, a4_2, a4_3, a4_4) if (a4_1 < a4_3) and (a4_1 < a4_2) else calc_2_a3(a4_1, a4_2, a4_3, a4_4) if a4_3 > a4_2 else calc_2_a2(a4_1, a4_2, a4_3, a4_4)
#     return a4_2, a4_3, a4_4


# @gtscript.function
# def semipos_a4(a4_1, a4_2, a4_3, a4_4):
#     a32 = a4_3 - a4_2 
#     abs_32 = absolute_value(a32)
#     a4_2, a4_3, a4_4 = calc_a4s(a4_1, a4_2, a4_3, a4_4) if (abs_32 < -a4_4) and (a4_1 + 0.25*(a4_3-a4_2)**2 / a4_4 + a4_4*1./12.) < 0. else a4_2, a4_3, a4_4
#     return a4_2, a4_3, a4_4

@utils.stencil()
def posdef_constraint_iv0(a4_1: sd, a4_2: sd, a4_3: sd, a4_4: sd):
    with computation(PARALLEL), interval(...):
        # a4_2 = a4_1 if a4_1 <= 0. else semipos_a4(a4_1, a4_2, a4_3, a4_4)
        # a4_3 = a4_1 if a4_1 <= 0. else semipos_a4(a4_1, a4_2, a4_3, a4_4)
        # a4_2, a4_3, a4_4 = neg_a4(a4_1, a4_2, a4_3, a4_4) if a4_1 <= 0. else semipos_a4(a4_1, a4_2, a4_3, a4_4)
        if a4_1 <= 0.:
            a4_2 = a4_1
            a4_3 = a4_1
            a4_4 = 0.
        else:
            a32 = a4_3 - a4_2 
            abs_32 = absolute_value(a32)
            if abs_32 < -a4_4 :
                if (a4_1 + 0.25*(a4_3-a4_2)**2 / a4_4 + a4_4*1./12.) < 0.:
                    if (a4_1 < a4_3) and (a4_1 < a4_2):
                        a4_3 = a4_1
                        a4_2 = a4_1
                        a4_4 = 0.
                    elif a4_3 > a4_2:
                        a4_4 = 3.*(a4_2-a4_1)
                        a4_3 = a4_2 - a4_4
                    else:
                        a4_4 = 3.*(a4_3-a4_1)
                        a4_2 = a4_3 - a4_4
                else:
                     a4_2 = a4_2
            else:
                a4_2 = a4_2

# @gtscript.function
# def calc_iv1(a4_1, a4_2, a4_3, a4_4):
#     da1 = a4_3 - a4_2
#     da2 = da1**2
#     a6da = a4_4*da1
#     a4_2, a4_3, a4_4 = calc_2_a3(a4_1, a4_2, a4_3, a4_4) if a6da < -da2 else calc_2_a2(a4_1, a4_2, a4_3, a4_4) if a6da > da2 else a4_2, a4_3, a4_4
#     return a4_2, a4_3, a4_4


@utils.stencil()
def posdef_constraint_iv1(a4_1: sd, a4_2: sd, a4_3: sd, a4_4: sd):
    with computation(PARALLEL), interval(...):
        # da1 = a4_3 - a4_2
        # da2 = da1**2
        # a6da = a4_4 * da1
        # a4_4 = 0. if (((a4_1 - a4_2) * (a4_1 - a4_3)) > 0.) else 3.*(a4_2-a4_1) if (a6da < -da2) else 3.*(a4_3-a4_1) if a6da > da2 else a4_4
        # a4_2 = a4_1 if (((a4_1 - a4_2) * (a4_1 - a4_3)) > 0.) else a4_3 - a4_4 if (a6da > da2) else a4_2
        # a4_3 = a4_1 if (((a4_1 - a4_2) * (a4_1 - a4_3)) > 0.) else a4_2 - a4_4 if (a6da < -da2) else a4_3
        da1 = a4_3 - a4_2
        da2 = da1**2
        a6da = a4_4 * da1
        if ((a4_1 - a4_2) * (a4_1 - a4_3)) >= 0.:
            a4_2 = a4_1
            a4_3 = a4_1
            a4_4 = 0.
        else:
            # da1 = a4_3 - a4_2
            # da2 = da1**2
            # a6da = a4_4 * da1
            if a6da < -1.*da2:
                a4_4 = 3.*(a4_2-a4_1)
                a4_3 = a4_2 - a4_4
            elif a6da > da2:
                a4_4 = 3.*(a4_3-a4_1)
                a4_2 = a4_3 - a4_4
            else:
                a4_2 = a4_2


@utils.stencil()
def ppm_constraint(a4_1: sd, a4_2: sd, a4_3: sd, a4_4: sd, extm: sd):
    with computation(PARALLEL), interval(...):
        # da1 = a4_3 - a4_2
        # da2 = da1**2
        # a6da = a4_4 * da1
        # a4_4 = 0. if extm == 1 else 3.*(a4_2-a4_1) if (a6da < -da2) else 3.*(a4_3-a4_1) if a6da > da2 else a4_4
        # a4_2 = a4_1 if extm == 1 else a4_3 - a4_4 if (a6da > da2) else a4_2
        # a4_3 = a4_1 if extm == 1 else a4_2 - a4_4 if (a6da < -da2) else a4_3
        da1 = a4_3 - a4_2
        da2 = da1**2
        a6da = a4_4 * da1
        if extm==1:
            a4_2 = a4_1
            a4_3 = a4_1
            a4_4 = 0.
        else:
            # da1 = a4_3 - a4_2
            # da2 = da1**2
            # a6da = a4_4 * da1
            if a6da < -da2:
                a4_4 = 3.*(a4_2-a4_1)
                a4_3 = a4_2 - a4_4
            elif a6da > da2:
                a4_4 = 3.*(a4_3-a4_1)
                a4_2 = a4_3 - a4_4
            else:
                a4_2 = a4_2


def compute(a4_1, a4_2, a4_3, a4_4, extm, iv, i1, i_extent, kstart, nk):

    if iv==0:
        posdef_constraint_iv0(a4_1, a4_2, a4_3, a4_4, origin=(i1, grid().js, kstart), domain=(i_extent, 1, nk))
    elif iv==1:
        posdef_constraint_iv1(a4_1, a4_2, a4_3, a4_4, origin=(i1, 0, kstart), domain=(i_extent, 1, nk))
    else:
        ppm_constraint(a4_1, a4_2, a4_3, a4_4, extm, origin=(i1, 0, kstart), domain=(i_extent, 1, nk))
    return a4_1, a4_2, a4_3, a4_4
