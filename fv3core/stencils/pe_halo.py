from gt4py.gtscript import FORWARD, computation, horizontal, interval, region

import fv3core._config as spec
from fv3core.decorators import gtstencil
from fv3core.utils.typing import FloatField


@gtstencil()
def edge_pe(pe: FloatField, delp: FloatField, ptop: float):
    from __externals__ import i_end, i_start, j_end, j_start

    with computation(FORWARD):
        with interval(0, 1):
            with horizontal(
                region[i_start - 1 : i_start, j_start : j_end + 1],
                region[i_end + 1 : i_end + 2, j_start : j_end + 1],
                region[i_start - 1 : i_end + 1, j_start - 1 : j_start],
                region[i_start - 1 : i_end + 1, j_end + 1 : j_end + 2],
            ):
                pe[0, 0, 0] = ptop
        with interval(1, None):
            with horizontal(
                region[i_start - 1 : i_start, j_start : j_end + 1],
                region[i_end + 1 : i_end + 2, j_start : j_end + 1],
                region[i_start - 1 : i_end + 2, j_start - 1 : j_start],
                region[i_start - 1 : i_end + 2, j_end + 1 : j_end + 2],
            ):
                pe[0, 0, 0] = pe[0, 0, -1] + delp[0, 0, -1]


def compute(pe, delp, ptop):
    grid = spec.grid
    edge_pe(
        pe,
        delp,
        ptop,
        origin=grid.full_origin(),
        domain=grid.domain_shape_full(add=(0, 0, 1)),
    )
