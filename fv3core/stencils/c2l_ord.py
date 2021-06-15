from gt4py.gtscript import PARALLEL, computation, interval

import fv3core.utils.global_config as global_config
import fv3core.utils.gt4py_utils as utils
from fv3core.decorators import FrozenStencil
from fv3core.utils.grid import axis_offsets
from fv3core.utils.typing import FloatField, FloatFieldIJ
from fv3gfs.util import CubedSphereCommunicator
from fv3gfs.util.quantity import Quantity


C1 = 1.125
C2 = -0.125


def tmps_main(
    u: FloatField,
    v: FloatField,
    utmp: FloatField,
    vtmp: FloatField,
):
    with computation(PARALLEL), interval(...):
        utmp = C2 * (u[0, -1, 0] + u[0, 2, 0]) + C1 * (u + u[0, 1, 0])
        vtmp = C2 * (v[-1, 0, 0] + v[2, 0, 0]) + C1 * (v + v[1, 0, 0])

def tmps_y_edge(
    u: FloatField,
    v: FloatField,
    dx: FloatFieldIJ,
    dy: FloatFieldIJ,
    utmp: FloatField,
    vtmp: FloatField,
):
    with computation(PARALLEL), interval(...):
        vtmp = 2.0 * ((v * dy) + (v[1, 0, 0] * dy[1, 0])) / (dy + dy[1, 0])
        utmp = 2.0 * (u * dx + u[0, 1, 0] * dx[0, 1]) / (dx + dx[0, 1])

def tmps_x_edge(
    u: FloatField,
    v: FloatField,
    dx: FloatFieldIJ,
    dy: FloatFieldIJ,
    utmp: FloatField,
    vtmp: FloatField,
):
    with computation(PARALLEL), interval(...):
        utmp = 2.0 * ((u * dx) + (u[0, 1, 0] * dx[0, 1])) / (dx + dx[0, 1])
        vtmp = 2.0 * ((v * dy) + (v[1, 0, 0] * dy[1, 0])) / (dy + dy[1, 0])

def ord4_transform(
    utmp: FloatField,
    vtmp: FloatField,
    a11: FloatFieldIJ,
    a12: FloatFieldIJ,
    a21: FloatFieldIJ,
    a22: FloatFieldIJ,
    ua: FloatField,
    va: FloatField,
):
    with computation(PARALLEL), interval(...):
        # Transform local a-grid winds into latitude-longitude coordinates
        ua = a11 * utmp + a12 * vtmp
        va = a21 * utmp + a22 * vtmp


class CubedToLatLon:
    """
    Fortan name is c2l_ord2
    """

    def __init__(self, grid, namelist, do_halo_update=None):
        """
        Initializes stencils to use either 2nd or 4th order of interpolation
        based on namelist setting
        Args:
            grid: fv3core grid object
            namelist:
                c2l_ord: Order of interpolation
            do_halo_update: Optional. If passed, overrides global halo exchange flag
                            and performs a halo update on u and v
        """
        if do_halo_update is not None:
            self._do_halo_update = do_halo_update
        else:
            self._do_halo_update = global_config.get_do_halo_exchange()
        self._do_ord4 = True
        self.grid = grid
        assert namelist.c2l_ord != 2, "not implemented c2l_rd == 2"
            
        origin = self.grid.compute_origin()
        domain = self.grid.domain_shape_compute()
        shape = self.grid.domain_shape_full(add=(1, 1, 1))
        self._utmp = utils.make_storage_from_shape(shape)
        self._vtmp = utils.make_storage_from_shape(shape)
        self._tmps_main = FrozenStencil(
            tmps_main,
            origin=origin,
            domain=domain,
        )
        if self.grid.south_edge:
            self._tmps_south_edge = FrozenStencil(
                tmps_y_edge,
                origin=(origin[0], self.grid.js, origin[2]),
                domain=(domain[0], 1, domain[2]),
            )
        if self.grid.north_edge:
            self._tmps_north_edge = FrozenStencil(
                tmps_y_edge,
                origin=(origin[0], self.grid.je, origin[2]),
                domain=(domain[0], 1, domain[2]),
            )
        if self.grid.west_edge:
            self._tmps_west_edge = FrozenStencil(
                tmps_x_edge,
                origin=(self.grid.is_, origin[1],  origin[2]),
                domain=(1, domain[1], domain[2]),
            )
        if self.grid.east_edge:
            self._tmps_east_edge = FrozenStencil(
                tmps_x_edge,
                origin=(self.grid.ie, origin[1],  origin[2]),
                domain=(1, domain[1], domain[2]),
            )
        self._compute_cubed_to_latlon = FrozenStencil(
            ord4_transform,
            origin=origin,
            domain=domain,
        )

    def __call__(
        self,
        u: Quantity,
        v: Quantity,
        ua: FloatField,
        va: FloatField,
        comm: CubedSphereCommunicator,
    ):
        """
        Interpolate D-grid to A-grid winds at latitude-longitude coordinates.
        Args:
            u: x-wind on D-grid (in)
            v: y-wind on D-grid (in)
            ua: x-wind on A-grid (out)
            va: y-wind on A-grid (out)
            comm: Cubed-sphere communicator
        """
        if self._do_halo_update and self._do_ord4:
            comm.vector_halo_update(u, v, n_points=self.grid.halo)
        self._tmps_main(
            u.storage,
            v.storage,
            self._utmp,
            self._vtmp,
        )
        if self.grid.south_edge:
            self._tmps_south_edge(
                u.storage,
                v.storage,
                self.grid.dx,
                self.grid.dy,
                self._utmp,
                self._vtmp,
            )
        if self.grid.north_edge:
            self._tmps_north_edge(
                u.storage,
                v.storage,
                self.grid.dx,
                self.grid.dy,
                self._utmp,
                self._vtmp,
            )
        if self.grid.west_edge:
            self._tmps_west_edge(
                u.storage,
                v.storage,
                self.grid.dx,
                self.grid.dy,
                self._utmp,
                self._vtmp,
            )
        if self.grid.east_edge:
            self._tmps_east_edge(
                u.storage,
                v.storage,
                self.grid.dx,
                self.grid.dy,
                self._utmp,
                self._vtmp,
            )
        self._compute_cubed_to_latlon(
            self._utmp,
            self._vtmp,
            self.grid.a11,
            self.grid.a12,
            self.grid.a21,
            self.grid.a22,
            ua,
            va,
        )
