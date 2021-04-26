from gt4py.gtscript import FORWARD, computation, horizontal, interval, region

import fv3core._config as spec
from fv3core.utils import axis_offsets
from fv3core.decorators import gtstencil, StencilWrapper
from fv3core.utils.typing import FloatField


def edge_pe(pe: FloatField, delp: FloatField, ptop: float):
    from __externals__ import local_ie, local_is, local_je, local_js

    with computation(FORWARD):
        with interval(0, 1):
            with horizontal(
                region[local_is - 1, local_js : local_je + 1],
                region[local_ie + 1, local_js : local_je + 1],
                region[local_is - 1 : local_ie + 2, local_js - 1],
                region[local_is - 1 : local_ie + 2, local_je + 1],
            ):
                pe[0, 0, 0] = ptop
        with interval(1, None):
            with horizontal(
                region[local_is - 1, local_js : local_je + 1],
                region[local_ie + 1, local_js : local_je + 1],
                region[local_is - 1 : local_ie + 2, local_js - 1],
                region[local_is - 1 : local_ie + 2, local_je + 1],
            ):
                pe[0, 0, 0] = pe[0, 0, -1] + delp[0, 0, -1]


class PeHalo:
    """
    This corresponds to the pe_halo routine in FV3core
    """

    def __init__(self):
        grid = spec.grid

        ax_offsets = axis_offsets(
            grid, grid.full_origin(), grid.domain_shape_full(add=(0, 0, 1))
        )
        local_axis_offsets = {}
        for axis_offset_name, axis_offset_value in ax_offsets.items():
            if "local" in axis_offset_name:
                local_axis_offsets[axis_offset_name] = axis_offset_value

        self._edge_pe_stencil = StencilWrapper(
            edge_pe,
            origin=grid.full_origin(),
            domain=grid.domain_shape_full(add=(0, 0, 1)),
            externals=local_axis_offsets,
        )

    def __call__(self, pe: FloatField, delp: FloatField, ptop: float):
        """
        Some in depth explanation of what is happening
        Arguments:
            pe: ???
            delp: The pressure difference between grid levels
            ptop: The pressure level at the top of the grid
        """
        self._edge_pe_stencil(pe, delp, ptop)


def compute(pe, delp, ptop):
    grid = spec.grid
    edge_pe(
        pe,
        delp,
        ptop,
        origin=grid.full_origin(),
        domain=grid.domain_shape_full(add=(0, 0, 1)),
    )
