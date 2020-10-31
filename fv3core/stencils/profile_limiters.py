import gt4py.gtscript as gtscript
from gt4py.gtscript import PARALLEL, computation, interval

import fv3core._config as spec
import fv3core.utils.gt4py_utils as utils
from fv3core.decorators import gtstencil


FloatField = utils.FloatField


@gtstencil()
def ppm_constraint(
    a4_1: FloatField,
    a4_2: FloatField,
    a4_3: FloatField,
    a4_4: FloatField,
    extm: FloatField,
    iv: int,
):
    with computation(PARALLEL), interval(...):
        # posdef_constraint_iv0
        if iv == 0:
            if a4_1 <= 0.0:
                a4_2 = a4_1
                a4_3 = a4_1
                a4_4 = 0.0
            else:
                if abs(a4_3 - a4_2) < -a4_4:
                    if (
                        a4_1 + 0.25 * (a4_3 - a4_2) ** 2 / a4_4 + a4_4 * (1.0 / 12.0)
                    ) < 0.0:
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
        if iv == 1:
            # posdef_constraint_iv1
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
                elif a6da > da2:
                    a4_4 = 3.0 * (a4_3 - a4_1)
                    a4_2 = a4_3 - a4_4
        # ppm_constraint
        if iv >= 2:
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


def compute(a4_1, a4_2, a4_3, a4_4, extm, iv, i1, i_extent, kstart, nk, js, j_extent):
    ppm_constraint(
        a4_1,
        a4_2,
        a4_3,
        a4_4,
        extm,
        iv,
        origin=(i1, js, kstart),
        domain=(i_extent, j_extent, nk),
    )
    return a4_1, a4_2, a4_3, a4_4
