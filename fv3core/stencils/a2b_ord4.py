from typing import Tuple

import gt4py
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
from fv3core.decorators import FrozenStencil
from fv3core.stencils.basic_operations import copy_defn
from fv3core.utils import axis_offsets
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


# return qa + (asin(sqrt(sin((p1b - p0b) / 2.0) ** 2.0 +
# cos(p1b) * cos(p0b) *  sin((p1a - p0a) / 2.0) ** 2.0)) * 2.0)
# / (asin(sqrt(sin((p2b - p0b) / 2.0) ** 2.0 + cos(p2b) * cos(p0b) *
# sin((p2a - p0a) / 2.0) ** 2.0)) * 2.0 -asin(sqrt(sin((p1b - p0b) /
# 2.0) ** 2.0 + cos(p1b) * cos(p0b) *  sin((p1a - p0a) / 2.0) ** 2.0))
# * 2.0) * (qa - qb)


@gtscript.function
def lagrange_y_func(qx):
    return a2 * (qx[0, -2, 0] + qx[0, 1, 0]) + a1 * (qx[0, -1, 0] + qx)


@gtscript.function
def lagrange_x_func(qy):
    return a2 * (qy[-2, 0, 0] + qy[1, 0, 0]) + a1 * (qy[-1, 0, 0] + qy)


def a2b_interpolation_qx(
    qin: FloatField,
    qx: FloatField,
    dxa: FloatFieldIJ,
):
    from __externals__ import i_end, i_start

    with computation(PARALLEL), interval(...):
        # ppm_volume_mean_x
        qx = b2 * (qin[-2, 0, 0] + qin[1, 0, 0]) + b1 * (qin[-1, 0, 0] + qin)
        with horizontal(region[i_start, :]):
            qx = 0.5 * (
                ((2.0 + dxa[1, 0] / dxa) * qin - qin[1, 0, 0]) / (1.0 + dxa[1, 0] / dxa)
                + ((2.0 + dxa[-2, 0] / dxa[-1, 0]) * qin[-1, 0, 0] - qin[-2, 0, 0])
                / (1.0 + dxa[-2, 0] / dxa[-1, 0])
            )

        with horizontal(region[i_start + 1, :]):
            qx = (
                3.0 * (dxa / dxa[-1, 0] * qin[-1, 0, 0] + qin)
                - (
                    dxa
                    / dxa[-1, 0]
                    * (
                        0.5
                        * (
                            ((2.0 + dxa[0, 0] / dxa[-1, 0]) * qin[-1, 0, 0] - qin)
                            / (1.0 + dxa[0, 0] / dxa[-1, 0])
                            + (
                                (2.0 + dxa[-3, 0] / dxa[-2, 0]) * qin[-2, 0, 0]
                                - qin[-3, 0, 0]
                            )
                            / (1.0 + dxa[-3, 0] / dxa[-2, 0])
                        )
                    )
                    + (b2 * (qin[-1, 0, 0] + qin[2, 0, 0]) + b1 * (qin + qin[1, 0, 0]))
                )
            ) / (2.0 + 2.0 * dxa / dxa[-1, 0])
        with horizontal(region[i_end + 1, :]):
            qx = 0.5 * (
                ((2.0 + dxa[-2, 0] / dxa[-1, 0]) * qin[-1, 0, 0] - qin[-2, 0, 0])
                / (1.0 + dxa[-2, 0] / dxa[-1, 0])
                + ((2.0 + dxa[1, 0] / dxa) * qin - qin[1, 0, 0])
                / (1.0 + dxa[1, 0] / dxa)
            )
        with horizontal(region[i_end, :]):
            qx = (
                3.0 * (qin[-1, 0, 0] + dxa[-1, 0] / dxa * qin)
                - (
                    dxa[-1, 0]
                    / dxa
                    * (
                        0.5
                        * (
                            ((2.0 + dxa[-1, 0] / dxa) * qin - qin[-1, 0, 0])
                            / (1.0 + dxa[-1, 0] / dxa)
                            + (
                                (2.0 + dxa[2, 0] / dxa[1, 0]) * qin[1, 0, 0]
                                - qin[2, 0, 0]
                            )
                            / (1.0 + dxa[2, 0] / dxa[1, 0])
                        )
                    )
                    + (
                        b2 * (qin[-3, 0, 0] + qin)
                        + b1 * (qin[-2, 0, 0] + qin[-1, 0, 0])
                    )
                )
            ) / (2.0 + 2.0 * dxa[-1, 0] / dxa)


def a2b_interpolation_qy(
    qin: FloatField,
    qy: FloatField,
    dya: FloatFieldIJ,
):
    from __externals__ import j_end, j_start

    with computation(PARALLEL), interval(...):

        # ppm_volume_mean_y
        qy = b2 * (qin[0, -2, 0] + qin[0, 1, 0]) + b1 * (qin[0, -1, 0] + qin)
        with horizontal(region[:, j_start]):
            qy = 0.5 * (
                ((2.0 + dya[0, 1] / dya) * qin - qin[0, 1, 0]) / (1.0 + dya[0, 1] / dya)
                + ((2.0 + dya[0, -2] / dya[0, -1]) * qin[0, -1, 0] - qin[0, -2, 0])
                / (1.0 + dya[0, -2] / dya[0, -1])
            )

        with horizontal(region[:, j_start + 1]):
            qy = (
                3.0 * (dya / dya[0, -1] * qin[0, -1, 0] + qin)
                - (
                    dya
                    / dya[0, -1]
                    * (
                        0.5
                        * (
                            ((2.0 + dya / dya[0, -1]) * qin[0, -1, 0] - qin)
                            / (1.0 + dya / dya[0, -1])
                            + (
                                (2.0 + dya[0, -3] / dya[0, -2]) * qin[0, -2, 0]
                                - qin[0, -3, 0]
                            )
                            / (1.0 + dya[0, -3] / dya[0, -2])
                        )
                    )
                    + (b2 * (qin[0, -1, 0] + qin[0, 2, 0]) + b1 * (qin + qin[0, 1, 0]))
                )
            ) / (2.0 + 2.0 * dya / dya[0, -1])
        with horizontal(region[:, j_end + 1]):
            qy = 0.5 * (
                ((2.0 + dya[0, -2] / dya[0, -1]) * qin[0, -1, 0] - qin[0, -2, 0])
                / (1.0 + dya[0, -2] / dya[0, -1])
                + ((2.0 + dya[0, 1] / dya) * qin - qin[0, 1, 0])
                / (1.0 + dya[0, 1] / dya)
            )
        with horizontal(region[:, j_end]):
            qy = (
                3.0 * (qin[0, -1, 0] + dya[0, -1] / dya * qin)
                - (dya[0, -1] / dya * qy[0, 1, 0] + qy[0, -1, 0])
            ) / (2.0 + 2.0 * dya[0, -1] / dya)


def a2b_interpolation(
    qin: FloatField,
    qout: FloatField,
    qx: FloatField,
    qy: FloatField,
    dxa: FloatFieldIJ,
    dya: FloatFieldIJ,
    edge_w: FloatFieldIJ,
    edge_e: FloatFieldIJ,
    edge_s: FloatFieldI,
    edge_n: FloatFieldI,
    agrid1: FloatFieldIJ,
    agrid2: FloatFieldIJ,
    bgrid1: FloatFieldIJ,
    bgrid2: FloatFieldIJ,
):
    from __externals__ import i_end, i_start, j_end, j_start

    with computation(PARALLEL), interval(...):
        with horizontal(region[i_start, j_start]):
            qout = (
                (
                    qin[0, 0, 0]
                    + (
                        asin(
                            sqrt(
                                sin((agrid2[0, 0] - bgrid2[0, 0]) / 2.0) ** 2.0
                                + cos(agrid2[0, 0])
                                * cos(bgrid2[0, 0])
                                * sin((agrid1[0, 0] - bgrid1[0, 0]) / 2.0) ** 2.0
                            )
                        )
                        * 2.0
                    )
                    / (
                        asin(
                            sqrt(
                                sin((agrid2[1, 1] - bgrid2[0, 0]) / 2.0) ** 2.0
                                + cos(agrid2[1, 1])
                                * cos(bgrid2[0, 0])
                                * sin((agrid1[1, 1] - bgrid1[0, 0]) / 2.0) ** 2.0
                            )
                        )
                        * 2.0
                        - asin(
                            sqrt(
                                sin((agrid2[0, 0] - bgrid2[0, 0]) / 2.0) ** 2.0
                                + cos(agrid2[0, 0])
                                * cos(bgrid2[0, 0])
                                * sin((agrid1[0, 0] - bgrid1[0, 0]) / 2.0) ** 2.0
                            )
                        )
                        * 2.0
                    )
                    * (qin[0, 0, 0] - qin[1, 1, 0])
                )
                + (
                    qin[-1, 0, 0]
                    + (
                        asin(
                            sqrt(
                                sin((agrid2[-1, 0] - bgrid2[0, 0]) / 2.0) ** 2.0
                                + cos(agrid2[-1, 0])
                                * cos(bgrid2[0, 0])
                                * sin((agrid1[-1, 0] - bgrid1[0, 0]) / 2.0) ** 2.0
                            )
                        )
                        * 2.0
                    )
                    / (
                        asin(
                            sqrt(
                                sin((agrid2[-2, 1] - bgrid2[0, 0]) / 2.0) ** 2.0
                                + cos(agrid2[-2, 1])
                                * cos(bgrid2[0, 0])
                                * sin((agrid1[-2, 1] - bgrid1[0, 0]) / 2.0) ** 2.0
                            )
                        )
                        * 2.0
                        - asin(
                            sqrt(
                                sin((agrid2[-1, 0] - bgrid2[0, 0]) / 2.0) ** 2.0
                                + cos(agrid2[-1, 0])
                                * cos(bgrid2[0, 0])
                                * sin((agrid1[-1, 0] - bgrid1[0, 0]) / 2.0) ** 2.0
                            )
                        )
                        * 2.0
                    )
                    * (qin[-1, 0, 0] - qin[-2, 1, 0])
                )
                + (
                    qin[0, -1, 0]
                    + (
                        asin(
                            sqrt(
                                sin((agrid2[0, -1] - bgrid2[0, 0]) / 2.0) ** 2.0
                                + cos(agrid2[0, -1])
                                * cos(bgrid2[0, 0])
                                * sin((agrid1[0, -1] - bgrid1[0, 0]) / 2.0) ** 2.0
                            )
                        )
                        * 2.0
                    )
                    / (
                        asin(
                            sqrt(
                                sin((agrid2[1, -2] - bgrid2[0, 0]) / 2.0) ** 2.0
                                + cos(agrid2[1, -2])
                                * cos(bgrid2[0, 0])
                                * sin((agrid1[1, -2] - bgrid1[0, 0]) / 2.0) ** 2.0
                            )
                        )
                        * 2.0
                        - asin(
                            sqrt(
                                sin((agrid2[0, -1] - bgrid2[0, 0]) / 2.0) ** 2.0
                                + cos(agrid2[0, -1])
                                * cos(bgrid2[0, 0])
                                * sin((agrid1[0, -1] - bgrid1[0, 0]) / 2.0) ** 2.0
                            )
                        )
                        * 2.0
                    )
                    * (qin[0, -1, 0] - qin[1, -2, 0])
                )
            ) * (1.0 / 3.0)
        with horizontal(region[i_end + 1, j_start]):
            qout = (
                (
                    qin[-1, 0, 0]
                    + (
                        asin(
                            sqrt(
                                sin((agrid2[-1, 0] - bgrid2[0, 0]) / 2.0) ** 2.0
                                + cos(agrid2[-1, 0])
                                * cos(bgrid2[0, 0])
                                * sin((agrid1[-1, 0] - bgrid1[0, 0]) / 2.0) ** 2.0
                            )
                        )
                        * 2.0
                    )
                    / (
                        asin(
                            sqrt(
                                sin((agrid2[-2, 1] - bgrid2[0, 0]) / 2.0) ** 2.0
                                + cos(agrid2[-2, 1])
                                * cos(bgrid2[0, 0])
                                * sin((agrid1[-2, 1] - bgrid1[0, 0]) / 2.0) ** 2.0
                            )
                        )
                        * 2.0
                        - asin(
                            sqrt(
                                sin((agrid2[-1, 0] - bgrid2[0, 0]) / 2.0) ** 2.0
                                + cos(agrid2[-1, 0])
                                * cos(bgrid2[0, 0])
                                * sin((agrid1[-1, 0] - bgrid1[0, 0]) / 2.0) ** 2.0
                            )
                        )
                        * 2.0
                    )
                    * (qin[-1, 0, 0] - qin[-2, 1, 0])
                )
                + (
                    qin[-1, -1, 0]
                    + (
                        asin(
                            sqrt(
                                sin((agrid2[-1, -1] - bgrid2[0, 0]) / 2.0) ** 2.0
                                + cos(agrid2[-1, -1])
                                * cos(bgrid2[0, 0])
                                * sin((agrid1[-1, -1] - bgrid1[0, 0]) / 2.0) ** 2.0
                            )
                        )
                        * 2.0
                    )
                    / (
                        asin(
                            sqrt(
                                sin((agrid2[-2, -2] - bgrid2[0, 0]) / 2.0) ** 2.0
                                + cos(agrid2[-2, -2])
                                * cos(bgrid2[0, 0])
                                * sin((agrid1[-2, -2] - bgrid1[0, 0]) / 2.0) ** 2.0
                            )
                        )
                        * 2.0
                        - asin(
                            sqrt(
                                sin((agrid2[-1, -1] - bgrid2[0, 0]) / 2.0) ** 2.0
                                + cos(agrid2[-1, -1])
                                * cos(bgrid2[0, 0])
                                * sin((agrid1[-1, -1] - bgrid1[0, 0]) / 2.0) ** 2.0
                            )
                        )
                        * 2.0
                    )
                    * (qin[-1, -1, 0] - qin[-2, -2, 0])
                )
                + (
                    qin[0, 0, 0]
                    + (
                        asin(
                            sqrt(
                                sin((agrid2[0, 0] - bgrid2[0, 0]) / 2.0) ** 2.0
                                + cos(agrid2[0, 0])
                                * cos(bgrid2[0, 0])
                                * sin((agrid1[0, 0] - bgrid1[0, 0]) / 2.0) ** 2.0
                            )
                        )
                        * 2.0
                    )
                    / (
                        asin(
                            sqrt(
                                sin((agrid2[1, 1] - bgrid2[0, 0]) / 2.0) ** 2.0
                                + cos(agrid2[1, 1])
                                * cos(bgrid2[0, 0])
                                * sin((agrid1[1, 1] - bgrid1[0, 0]) / 2.0) ** 2.0
                            )
                        )
                        * 2.0
                        - asin(
                            sqrt(
                                sin((agrid2[0, 0] - bgrid2[0, 0]) / 2.0) ** 2.0
                                + cos(agrid2[0, 0])
                                * cos(bgrid2[0, 0])
                                * sin((agrid1[0, 0] - bgrid1[0, 0]) / 2.0) ** 2.0
                            )
                        )
                        * 2.0
                    )
                    * (qin[0, 0, 0] - qin[1, 1, 0])
                )
            ) * (1.0 / 3.0)
        with horizontal(region[i_end + 1, j_end + 1]):
            qout = (
                (
                    qin[-1, -1, 0]
                    + (
                        asin(
                            sqrt(
                                sin((agrid2[-1, -1] - bgrid2[0, 0]) / 2.0) ** 2.0
                                + cos(agrid2[-1, -1])
                                * cos(bgrid2[0, 0])
                                * sin((agrid1[-1, -1] - bgrid1[0, 0]) / 2.0) ** 2.0
                            )
                        )
                        * 2.0
                    )
                    / (
                        asin(
                            sqrt(
                                sin((agrid2[-2, -2] - bgrid2[0, 0]) / 2.0) ** 2.0
                                + cos(agrid2[-2, -2])
                                * cos(bgrid2[0, 0])
                                * sin((agrid1[-2, -2] - bgrid1[0, 0]) / 2.0) ** 2.0
                            )
                        )
                        * 2.0
                        - asin(
                            sqrt(
                                sin((agrid2[-1, -1] - bgrid2[0, 0]) / 2.0) ** 2.0
                                + cos(agrid2[-1, -1])
                                * cos(bgrid2[0, 0])
                                * sin((agrid1[-1, -1] - bgrid1[0, 0]) / 2.0) ** 2.0
                            )
                        )
                        * 2.0
                    )
                    * (qin[-1, -1, 0] - qin[-2, -2, 0])
                )
                + (
                    qin[0, -1, 0]
                    + (
                        asin(
                            sqrt(
                                sin((agrid2[0, -1] - bgrid2[0, 0]) / 2.0) ** 2.0
                                + cos(agrid2[0, -1])
                                * cos(bgrid2[0, 0])
                                * sin((agrid1[0, -1] - bgrid1[0, 0]) / 2.0) ** 2.0
                            )
                        )
                        * 2.0
                    )
                    / (
                        asin(
                            sqrt(
                                sin((agrid2[1, -2] - bgrid2[0, 0]) / 2.0) ** 2.0
                                + cos(agrid2[1, -2])
                                * cos(bgrid2[0, 0])
                                * sin((agrid1[1, -2] - bgrid1[0, 0]) / 2.0) ** 2.0
                            )
                        )
                        * 2.0
                        - asin(
                            sqrt(
                                sin((agrid2[0, -1] - bgrid2[0, 0]) / 2.0) ** 2.0
                                + cos(agrid2[0, -1])
                                * cos(bgrid2[0, 0])
                                * sin((agrid1[0, -1] - bgrid1[0, 0]) / 2.0) ** 2.0
                            )
                        )
                        * 2.0
                    )
                    * (qin[0, -1, 0] - qin[1, -2, 0])
                )
                + (
                    qin[-1, 0, 0]
                    + (
                        asin(
                            sqrt(
                                sin((agrid2[-1, 0] - bgrid2[0, 0]) / 2.0) ** 2.0
                                + cos(agrid2[-1, 0])
                                * cos(bgrid2[0, 0])
                                * sin((agrid1[-1, 0] - bgrid1[0, 0]) / 2.0) ** 2.0
                            )
                        )
                        * 2.0
                    )
                    / (
                        asin(
                            sqrt(
                                sin((agrid2[-2, 1] - bgrid2[0, 0]) / 2.0) ** 2.0
                                + cos(agrid2[-2, 1])
                                * cos(bgrid2[0, 0])
                                * sin((agrid1[-2, 1] - bgrid1[0, 0]) / 2.0) ** 2.0
                            )
                        )
                        * 2.0
                        - asin(
                            sqrt(
                                sin((agrid2[-1, 0] - bgrid2[0, 0]) / 2.0) ** 2.0
                                + cos(agrid2[-1, 0])
                                * cos(bgrid2[0, 0])
                                * sin((agrid1[-1, 0] - bgrid1[0, 0]) / 2.0) ** 2.0
                            )
                        )
                        * 2.0
                    )
                    * (qin[-1, 0, 0] - qin[-2, 1, 0])
                )
            ) * (1.0 / 3.0)
        with horizontal(region[i_start, j_end + 1]):
            qout = (
                (
                    qin[0, -1, 0]
                    + (
                        asin(
                            sqrt(
                                sin((agrid2[0, -1] - bgrid2[0, 0]) / 2.0) ** 2.0
                                + cos(agrid2[0, -1])
                                * cos(bgrid2[0, 0])
                                * sin((agrid1[0, -1] - bgrid1[0, 0]) / 2.0) ** 2.0
                            )
                        )
                        * 2.0
                    )
                    / (
                        asin(
                            sqrt(
                                sin((agrid2[1, -2] - bgrid2[0, 0]) / 2.0) ** 2.0
                                + cos(agrid2[1, -2])
                                * cos(bgrid2[0, 0])
                                * sin((agrid1[1, -2] - bgrid1[0, 0]) / 2.0) ** 2.0
                            )
                        )
                        * 2.0
                        - asin(
                            sqrt(
                                sin((agrid2[0, -1] - bgrid2[0, 0]) / 2.0) ** 2.0
                                + cos(agrid2[0, -1])
                                * cos(bgrid2[0, 0])
                                * sin((agrid1[0, -1] - bgrid1[0, 0]) / 2.0) ** 2.0
                            )
                        )
                        * 2.0
                    )
                    * (qin[0, -1, 0] - qin[1, -2, 0])
                )
                + (
                    qin[-1, -1, 0]
                    + (
                        asin(
                            sqrt(
                                sin((agrid2[-1, -1] - bgrid2[0, 0]) / 2.0) ** 2.0
                                + cos(agrid2[-1, -1])
                                * cos(bgrid2[0, 0])
                                * sin((agrid1[-1, -1] - bgrid1[0, 0]) / 2.0) ** 2.0
                            )
                        )
                        * 2.0
                    )
                    / (
                        asin(
                            sqrt(
                                sin((agrid2[-2, -2] - bgrid2[0, 0]) / 2.0) ** 2.0
                                + cos(agrid2[-2, -2])
                                * cos(bgrid2[0, 0])
                                * sin((agrid1[-2, -2] - bgrid1[0, 0]) / 2.0) ** 2.0
                            )
                        )
                        * 2.0
                        - asin(
                            sqrt(
                                sin((agrid2[-1, -1] - bgrid2[0, 0]) / 2.0) ** 2.0
                                + cos(agrid2[-1, -1])
                                * cos(bgrid2[0, 0])
                                * sin((agrid1[-1, -1] - bgrid1[0, 0]) / 2.0) ** 2.0
                            )
                        )
                        * 2.0
                    )
                    * (qin[-1, -1, 0] - qin[-2, -2, 0])
                )
                + (
                    qin[0, 0, 0]
                    + (
                        asin(
                            sqrt(
                                sin((agrid2[0, 0] - bgrid2[0, 0]) / 2.0) ** 2.0
                                + cos(agrid2[0, 0])
                                * cos(bgrid2[0, 0])
                                * sin((agrid1[0, 0] - bgrid1[0, 0]) / 2.0) ** 2.0
                            )
                        )
                        * 2.0
                    )
                    / (
                        asin(
                            sqrt(
                                sin((agrid2[1, 1] - bgrid2[0, 0]) / 2.0) ** 2.0
                                + cos(agrid2[1, 1])
                                * cos(bgrid2[0, 0])
                                * sin((agrid1[1, 1] - bgrid1[0, 0]) / 2.0) ** 2.0
                            )
                        )
                        * 2.0
                        - asin(
                            sqrt(
                                sin((agrid2[0, 0] - bgrid2[0, 0]) / 2.0) ** 2.0
                                + cos(agrid2[0, 0])
                                * cos(bgrid2[0, 0])
                                * sin((agrid1[0, 0] - bgrid1[0, 0]) / 2.0) ** 2.0
                            )
                        )
                        * 2.0
                    )
                    * (qin[0, 0, 0] - qin[1, 1, 0])
                )
            ) * (1.0 / 3.0)
        # qout_edges_x
        with horizontal(region[i_start, j_start + 1 : j_end + 1]):
            qout = edge_w * (
                (qin[-1, -1, 0] * dxa[0, -1] + qin[0, -1, 0] * dxa[-1, -1])
                / (dxa[-1, -1] + dxa[0, -1])
            ) + (1.0 - edge_w) * (
                (qin[-1, 0, 0] * dxa[0, 0] + qin * dxa[-1, 0])
                / (dxa[-1, 0] + dxa[0, 0])
            )
        with horizontal(region[i_end + 1, j_start + 1 : j_end + 1]):
            qout = edge_e * (
                (qin[-1, -1, 0] * dxa[0, -1] + qin[0, -1, 0] * dxa[-1, -1])
                / (dxa[-1, -1] + dxa[0, -1])
            ) + (1.0 - edge_e) * (
                (qin[-1, 0, 0] * dxa + qin * dxa[-1, 0]) / (dxa[-1, 0] + dxa)
            )
        # qout edges y
        with horizontal(region[i_start + 1 : i_end + 1, j_start]):
            qout = edge_s * (
                (qin[-1, -1, 0] * dya[-1, 0] + qin[-1, 0, 0] * dya[-1, -1])
                / (dya[-1, -1] + dya[-1, 0])
            ) + (1.0 - edge_s) * (
                (qin[0, -1, 0] * dya + qin * dya[0, -1]) / (dya[0, -1] + dya)
            )
        with horizontal(region[i_start + 1 : i_end + 1, j_end + 1]):
            qout = edge_n * (
                (qin[-1, -1, 0] * dya[-1, 0] + qin[-1, 0, 0] * dya[-1, -1])
                / (dya[-1, -1] + dya[-1, 0])
            ) + (1.0 - edge_n) * (
                (qin[0, -1, 0] * dya + qin * dya[0, -1]) / (dya[0, -1] + dya)
            )

        # combined (qxx and qyy folded in)
        with horizontal(region[i_start + 1, j_start + 1]):
            qout = 0.5 * (
                (
                    c1 * (qx[0, -1, 0] + qx)
                    + c2
                    * (
                        (
                            edge_s
                            * (
                                (
                                    qin[-1, -2, 0] * dya[-1, -1]
                                    + qin[-1, -1, 0] * dya[-1, -2]
                                )
                                / (dya[-1, -2] + dya[-1, -1])
                            )
                            + (1.0 - edge_s)
                            * (
                                (
                                    qin[0, -2, 0] * dya[0, -1]
                                    + qin[0, -1, 0] * dya[0, -2]
                                )
                                / (dya[0, -2] + dya[0, -1])
                            )
                        )
                        + (a2 * (qx[0, -1, 0] + qx[0, 2, 0]) + a1 * (qx + qx[0, 1, 0]))
                    )
                )
                + (
                    c1 * (qy[-1, 0, 0] + qy)
                    + c2
                    * (
                        (
                            edge_w
                            * (
                                (
                                    qin[-2, -1, 0] * dxa[-1, -1]
                                    + qin[-1, -1, 0] * dxa[-2, -1]
                                )
                                / (dxa[-2, -1] + dxa[-1, -1])
                            )
                            + (1.0 - edge_w)
                            * (
                                (
                                    qin[-2, 0, 0] * dxa[-1, 0]
                                    + qin[-1, 0, 0] * dxa[-2, 0]
                                )
                                / (dxa[-2, 0] + dxa[-1, 0])
                            )
                        )
                        + (a2 * (qy[-1, 0, 0] + qy[2, 0, 0]) + a1 * (qy + qy[1, 0, 0]))
                    )
                )
            )
        with horizontal(region[i_end, j_start + 1]):
            qout = 0.5 * (
                (
                    c1 * (qx[0, -1, 0] + qx)
                    + c2
                    * (
                        (
                            edge_s
                            * (
                                (
                                    qin[-1, -2, 0] * dya[-1, -1]
                                    + qin[-1, -1, 0] * dya[-1, -2]
                                )
                                / (dya[-1, -2] + dya[-1, -1])
                            )
                            + (1.0 - edge_s)
                            * (
                                (
                                    qin[0, -2, 0] * dya[0, -1]
                                    + qin[0, -1, 0] * dya[0, -2]
                                )
                                / (dya[0, -2] + dya[0, -1])
                            )
                        )
                        + (a2 * (qx[0, -1, 0] + qx[0, 2, 0]) + a1 * (qx + qx[0, 1, 0]))
                    )
                )
                + (
                    c1 * (qy[-1, 0, 0] + qy)
                    + c2
                    * (
                        (
                            edge_e
                            * (
                                (
                                    qin[0, -1, 0] * dxa[1, -1]
                                    + qin[1, -1, 0] * dxa[0, -1]
                                )
                                / (dxa[0, -1] + dxa[1, -1])
                            )
                            + (1.0 - edge_e)
                            * (
                                (qin[0, 0, 0] * dxa[1, 0] + qin[1, 0, 0] * dxa[0, 0])
                                / (dxa[0, 0] + dxa[1, 0])
                            )
                        )
                        + (
                            a2 * (qy[-3, 0, 0] + qy)
                            + a1 * (qy[-2, 0, 0] + qy[-1, 0, 0])
                        )
                    )
                )
            )
        with horizontal(region[i_start + 2 : i_end, j_start + 1]):
            qout = 0.5 * (
                (
                    c1 * (qx[0, -1, 0] + qx)
                    + c2
                    * (
                        (
                            edge_s
                            * (
                                (
                                    qin[-1, -2, 0] * dya[-1, -1]
                                    + qin[-1, -1, 0] * dya[-1, -2]
                                )
                                / (dya[-1, -2] + dya[-1, -1])
                            )
                            + (1.0 - edge_s)
                            * (
                                (
                                    qin[0, -2, 0] * dya[0, -1]
                                    + qin[0, -1, 0] * dya[0, -2]
                                )
                                / (dya[0, -2] + dya[0, -1])
                            )
                        )
                        + (a2 * (qx[0, -1, 0] + qx[0, 2, 0]) + a1 * (qx + qx[0, 1, 0]))
                    )
                )
                + (a2 * (qy[-2, 0, 0] + qy[1, 0, 0]) + a1 * (qy[-1, 0, 0] + qy))
            )
        with horizontal(region[i_end, j_start + 2 : j_end]):
            qout = 0.5 * (
                (a2 * (qx[0, -2, 0] + qx[0, 1, 0]) + a1 * (qx[0, -1, 0] + qx))
                + (
                    c1 * (qy[-1, 0, 0] + qy)
                    + c2
                    * (
                        (
                            edge_e
                            * (
                                (
                                    qin[0, -1, 0] * dxa[1, -1]
                                    + qin[1, -1, 0] * dxa[0, -1]
                                )
                                / (dxa[0, -1] + dxa[1, -1])
                            )
                            + (1.0 - edge_e)
                            * (
                                (qin[0, 0, 0] * dxa[1, 0] + qin[1, 0, 0] * dxa[0, 0])
                                / (dxa[0, 0] + dxa[1, 0])
                            )
                        )
                        + (
                            a2 * (qy[-3, 0, 0] + qy)
                            + a1 * (qy[-2, 0, 0] + qy[-1, 0, 0])
                        )
                    )
                )
            )
        with horizontal(region[i_start + 1, j_end]):
            qout = 0.5 * (
                (
                    c1 * (qx[0, -1, 0] + qx)
                    + c2
                    * (
                        (
                            edge_n
                            * (
                                (
                                    qin[-1, 0, 0] * dya[-1, 1]
                                    + qin[-1, 1, 0] * dya[-1, 0]
                                )
                                / (dya[-1, 0] + dya[-1, 1])
                            )
                            + (1.0 - edge_n)
                            * (
                                (qin[0, 0, 0] * dya[0, 1] + qin[0, 1, 0] * dya[0, 0])
                                / (dya[0, 0] + dya[0, 1])
                            )
                        )
                        + (
                            a2 * (qx[0, -3, 0] + qx)
                            + a1 * (qx[0, -2, 0] + qx[0, -1, 0])
                        )
                    )
                )
                + (
                    c1 * (qy[-1, 0, 0] + qy)
                    + c2
                    * (
                        (
                            edge_w
                            * (
                                (
                                    qin[-2, -1, 0] * dxa[-1, -1]
                                    + qin[-1, -1, 0] * dxa[-2, -1]
                                )
                                / (dxa[-2, -1] + dxa[-1, -1])
                            )
                            + (1.0 - edge_w)
                            * (
                                (
                                    qin[-2, 0, 0] * dxa[-1, 0]
                                    + qin[-1, 0, 0] * dxa[-2, 0]
                                )
                                / (dxa[-2, 0] + dxa[-1, 0])
                            )
                        )
                        + (a2 * (qy[-1, 0, 0] + qy[2, 0, 0]) + a1 * (qy + qy[1, 0, 0]))
                    )
                )
            )
        with horizontal(region[i_end, j_end]):
            qout = 0.5 * (
                (
                    c1 * (qx[0, -1, 0] + qx)
                    + c2
                    * (
                        (
                            edge_n
                            * (
                                (
                                    qin[-1, 0, 0] * dya[-1, 1]
                                    + qin[-1, 1, 0] * dya[-1, 0]
                                )
                                / (dya[-1, 0] + dya[-1, 1])
                            )
                            + (1.0 - edge_n)
                            * (
                                (qin[0, 0, 0] * dya[0, 1] + qin[0, 1, 0] * dya[0, 0])
                                / (dya[0, 0] + dya[0, 1])
                            )
                        )
                        + (
                            a2 * (qx[0, -3, 0] + qx)
                            + a1 * (qx[0, -2, 0] + qx[0, -1, 0])
                        )
                    )
                )
                + (
                    c1 * (qy[-1, 0, 0] + qy)
                    + c2
                    * (
                        (
                            edge_e
                            * (
                                (
                                    qin[0, -1, 0] * dxa[1, -1]
                                    + qin[1, -1, 0] * dxa[0, -1]
                                )
                                / (dxa[0, -1] + dxa[1, -1])
                            )
                            + (1.0 - edge_e)
                            * (
                                (qin[0, 0, 0] * dxa[1, 0] + qin[1, 0, 0] * dxa[0, 0])
                                / (dxa[0, 0] + dxa[1, 0])
                            )
                        )
                        + (
                            a2 * (qy[-3, 0, 0] + qy)
                            + a1 * (qy[-2, 0, 0] + qy[-1, 0, 0])
                        )
                    )
                )
            )
        with horizontal(region[i_start + 2 : i_end, j_end]):
            qout = 0.5 * (
                (
                    c1 * (qx[0, -1, 0] + qx)
                    + c2
                    * (
                        (
                            edge_n
                            * (
                                (
                                    qin[-1, 0, 0] * dya[-1, 1]
                                    + qin[-1, 1, 0] * dya[-1, 0]
                                )
                                / (dya[-1, 0] + dya[-1, 1])
                            )
                            + (1.0 - edge_n)
                            * (
                                (qin[0, 0, 0] * dya[0, 1] + qin[0, 1, 0] * dya[0, 0])
                                / (dya[0, 0] + dya[0, 1])
                            )
                        )
                        + (
                            a2 * (qx[0, -3, 0] + qx)
                            + a1 * (qx[0, -2, 0] + qx[0, -1, 0])
                        )
                    )
                )
                + (a2 * (qy[-2, 0, 0] + qy[1, 0, 0]) + a1 * (qy[-1, 0, 0] + qy))
            )
        with horizontal(region[i_start + 1, j_start + 2 : j_end]):
            qout = 0.5 * (
                (a2 * (qx[0, -2, 0] + qx[0, 1, 0]) + a1 * (qx[0, -1, 0] + qx))
                + (
                    c1 * (qy[-1, 0, 0] + qy)
                    + c2
                    * (
                        (
                            edge_w
                            * (
                                (
                                    qin[-2, -1, 0] * dxa[-1, -1]
                                    + qin[-1, -1, 0] * dxa[-2, -1]
                                )
                                / (dxa[-2, -1] + dxa[-1, -1])
                            )
                            + (1.0 - edge_w)
                            * (
                                (
                                    qin[-2, 0, 0] * dxa[-1, 0]
                                    + qin[-1, 0, 0] * dxa[-2, 0]
                                )
                                / (dxa[-2, 0] + dxa[-1, 0])
                            )
                        )
                        + (a2 * (qy[-1, 0, 0] + qy[2, 0, 0]) + a1 * (qy + qy[1, 0, 0]))
                    )
                )
            )
        with horizontal(region[i_start + 2 : i_end, j_start + 2 : j_end]):
            qout = 0.5 * (
                (a2 * (qx[0, -2, 0] + qx[0, 1, 0]) + a1 * (qx[0, -1, 0] + qx))
                + (a2 * (qy[-2, 0, 0] + qy[1, 0, 0]) + a1 * (qy[-1, 0, 0] + qy))
            )


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
        self._replace = replace
        shape = self.grid.domain_shape_full(add=(1, 1, 1))
        if nk is None:
            nk = self.grid.npz - kstart
        self._edge_e = self._j_storage_repeat_over_i(self.grid.edge_e, shape[0:2])
        self._edge_w = self._j_storage_repeat_over_i(self.grid.edge_w, shape[0:2])
        self._tmp_qx = utils.make_storage_from_shape(shape)
        self._tmp_qy = utils.make_storage_from_shape(shape)
        origin = (self.grid.is_, self.grid.js, kstart)
        domain = (self.grid.nic + 1, self.grid.njc + 1, nk)
        ax_offsets = axis_offsets(self.grid, origin, domain)

        origin_prep_x = (self.grid.is_ - 1, self.grid.js - 2, kstart)
        domain_prep_x = (self.grid.nic + 3, self.grid.njc + 4, nk)
        ax_offsets_prep_x = axis_offsets(self.grid, origin_prep_x, domain_prep_x)
        self._a2b_interpolation_qx_stencil = FrozenStencil(
            a2b_interpolation_qx,
            externals=ax_offsets_prep_x,
            origin=origin_prep_x,
            domain=domain_prep_x,
        )
        origin_prep_y = (self.grid.is_ - 2, self.grid.js - 1, kstart)
        domain_prep_y = (self.grid.nic + 4, self.grid.njc + 3, nk)
        ax_offsets_prep_y = axis_offsets(self.grid, origin_prep_y, domain_prep_y)
        self._a2b_interpolation_qy_stencil = FrozenStencil(
            a2b_interpolation_qy,
            externals=ax_offsets_prep_y,
            origin=origin_prep_y,
            domain=domain_prep_y,
        )

        self._a2b_interpolation_stencil = FrozenStencil(
            a2b_interpolation,
            externals=ax_offsets,
            origin=origin,
            domain=domain,
        )
        self._copy_stencil = FrozenStencil(
            copy_defn,
            origin=origin,
            domain=domain,
        )

    # TODO
    # within regions, the edge_w and edge_w variables that are singleton in the
    # I dimension error, workaround is repeating the data, but the longterm
    # fix should happen in regions
    def _j_storage_repeat_over_i(
        self, grid_array: gt4py.storage.Storage, shape: Tuple[int, int]
    ):
        dup = utils.repeat(grid_array, shape[1], axis=0)
        return utils.make_storage_data(dup, shape, (0, 0))

    def __call__(self, qin: FloatField, qout: FloatField):
        """Converts qin from A-grid to B-grid in qout.
        In some cases, qin is also updated to the B grid.
        Args:
        qin: Input on A-grid (inout)
        qout: Output on B-grid (inout)
        """

        self._a2b_interpolation_qx_stencil(
            qin,
            self._tmp_qx,
            self.grid.dxa,
        )

        self._a2b_interpolation_qy_stencil(
            qin,
            self._tmp_qy,
            self.grid.dya,
        )

        self._a2b_interpolation_stencil(
            qin,
            qout,
            self._tmp_qx,
            self._tmp_qy,
            self.grid.dxa,
            self.grid.dya,
            self._edge_w,
            self._edge_e,
            self.grid.edge_s,
            self.grid.edge_n,
            self.grid.agrid1,
            self.grid.agrid2,
            self.grid.bgrid1,
            self.grid.bgrid2,
        )

        if self._replace:
            self._copy_stencil(qout, qin)
