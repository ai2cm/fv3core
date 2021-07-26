import dace
from gt4py.gtscript import FORWARD, computation, interval

import fv3core.utils.gt4py_utils as utils
from fv3core.decorators import FrozenStencil, computepath_method
from fv3core.utils.typing import FloatField, FloatFieldIJ


# TODO merge with pe_halo? reuse partials?
# NOTE: This is different from fv3core.stencils.pe_halo.edge_pe
def edge_pe_update(
    pe: FloatFieldIJ, delp: FloatField, pk3: FloatField, ptop: float, akap: float
):

    with computation(FORWARD):
        with interval(0, 1):
            pe = ptop
        with interval(1, None):
            pe = pe + delp[0, 0, -1]
            pk3 = pe ** akap


class PK3Halo:
    """
    Fortran name is pk3_halo
    """

    def __init__(self, grid):
        shape_2D = grid.domain_shape_full(add=(1, 1, 1))[0:2]
        origin = grid.full_origin()
        domain = grid.domain_shape_full(add=(0, 0, 1))

        self._pe_tmp = utils.make_storage_from_shape(shape_2D, grid.full_origin())
        edge_domain_x = (2, grid.njc, grid.npz + 1)
        self._edge_pe_update_west = FrozenStencil(
            edge_pe_update,
            origin=(grid.is_ - 2, grid.js, 0),
            domain=edge_domain_x,
        )
        self._edge_pe_update_east = FrozenStencil(
            edge_pe_update,
            origin=(grid.ie + 1, grid.js, 0),
            domain=edge_domain_x,
        )
        edge_domain_y = (grid.nic + 4, 2, grid.npz + 1)
        self._edge_pe_update_south = FrozenStencil(
            edge_pe_update,
            origin=(grid.is_ - 2, grid.js - 2, 0),
            domain=edge_domain_y,
        )
        self._edge_pe_update_north = FrozenStencil(
            edge_pe_update,
            origin=(grid.is_ - 2, grid.je + 1, 0),
            domain=edge_domain_y,
        )

    @computepath_method
    def __call__(self, pk3, delp, ptop: float, akap: float):
        """Update pressure (pk3) in halo region

        Args:
            pk3: Interface pressure raised to power of kappa using constant kappa
            delp: Vertical delta in pressure
            ptop: The pressure level at the top of atmosphere
            akap: Poisson constant (KAPPA)
        """
        self._edge_pe_update_west(self._pe_tmp, delp, pk3, ptop, akap)
        self._edge_pe_update_east(self._pe_tmp, delp, pk3, ptop, akap)
        self._edge_pe_update_south(self._pe_tmp, delp, pk3, ptop, akap)
        self._edge_pe_update_north(self._pe_tmp, delp, pk3, ptop, akap)
