from gt4py.gtscript import PARALLEL, computation, horizontal, interval, region

import fv3core.utils.global_config as global_config
import fv3core.utils.gt4py_utils as utils
from fv3core.decorators import FrozenStencil
from fv3core.utils.grid import axis_offsets
from fv3core.utils.typing import FloatField, FloatFieldIJ
from fv3gfs.util import CubedSphereCommunicator
from fv3gfs.util.quantity import Quantity


C1 = 1.125
C2 = -0.125


@utils.mark_untested("This namelist option is not tested")
def c2l_ord2(
    u: FloatField,
    v: FloatField,
    dx: FloatFieldIJ,
    dy: FloatFieldIJ,
    a11: FloatFieldIJ,
    a12: FloatFieldIJ,
    a21: FloatFieldIJ,
    a22: FloatFieldIJ,
    ua: FloatField,
    va: FloatField,
):
    with computation(PARALLEL), interval(...):
        wu = u * dx
        wv = v * dy
        # Co-variant vorticity-conserving interpolation
        u1 = 2.0 * (wu + wu[0, 1, 0]) / (dx + dx[0, 1])
        v1 = 2.0 * (wv + wv[1, 0, 0]) / (dy + dy[1, 0])
        # Cubed (cell center co-variant winds) to lat-lon
        ua = a11 * u1 + a12 * v1
        va = a21 * u1 + a22 * v1


def ord4_transform(
    u: FloatField,
    v: FloatField,
    dx: FloatFieldIJ,
    dy: FloatFieldIJ,
    a11: FloatFieldIJ,
    a12: FloatFieldIJ,
    a21: FloatFieldIJ,
    a22: FloatFieldIJ,
    ua: FloatField,
    va: FloatField,
):
    with computation(PARALLEL), interval(...):
        from __externals__ import i_end, i_start, j_end, j_start

        utmp = C2 * (u[0, -1, 0] + u[0, 2, 0]) + C1 * (u + u[0, 1, 0])
        vtmp = C2 * (v[-1, 0, 0] + v[2, 0, 0]) + C1 * (v + v[1, 0, 0])

        # south/north edge
        with horizontal(region[:, j_start], region[:, j_end]):
            vtmp = 2.0 * ((v * dy) + (v[1, 0, 0] * dy[1, 0])) / (dy + dy[1, 0])
            utmp = 2.0 * (u * dx + u[0, 1, 0] * dx[0, 1]) / (dx + dx[0, 1])

        # west/east edge
        with horizontal(region[i_start, :], region[i_end, :]):
            utmp = 2.0 * ((u * dx) + (u[0, 1, 0] * dx[0, 1])) / (dx + dx[0, 1])
            vtmp = 2.0 * ((v * dy) + (v[1, 0, 0] * dy[1, 0])) / (dy + dy[1, 0])

        # Transform local a-grid winds into latitude-longitude coordinates
        ua = a11 * utmp + a12 * vtmp
        va = a21 * utmp + a22 * vtmp


class CubedToLatLon:
    """
    Fortan name is c2l_ord2
    """

    def __init__(
        self,
        grid,
        namelist,
        comm: CubedSphereCommunicator,
        u: Quantity,
        v: Quantity,
        do_halo_update=None,
    ):
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
        if namelist.c2l_ord == 2:
            self._do_ord4 = False
            self._compute_cubed_to_latlon = FrozenStencil(
                func=c2l_ord2,
                origin=self.grid.compute_origin(
                    add=(-1, -1, 0) if self._do_halo_update else (0, 0, 0)
                ),
                domain=self.grid.domain_shape_compute(
                    add=(2, 2, 0) if self._do_halo_update else (0, 0, 0)
                ),
            )
        else:
            origin = self.grid.compute_origin()
            domain = self.grid.domain_shape_compute()
            ax_offsets = axis_offsets(self.grid, origin, domain)
            self._compute_cubed_to_latlon = FrozenStencil(
                func=ord4_transform,
                externals={
                    **ax_offsets,
                },
                origin=origin,
                domain=domain,
            )
        self._u__v_halo_updater = comm.get_vector_halo_updater([u], [v], self.grid.halo)

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
            self._u__v_halo_updater.blocking_exchange()
        self._compute_cubed_to_latlon(
            u.storage,
            v.storage,
            self.grid.dx,
            self.grid.dy,
            self.grid.a11,
            self.grid.a12,
            self.grid.a21,
            self.grid.a22,
            ua,
            va,
        )
