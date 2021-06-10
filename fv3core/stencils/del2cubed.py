from gt4py.gtscript import PARALLEL, computation, interval

import fv3core._config as spec
import fv3core.utils.corners as corners
import fv3core.utils.gt4py_utils as utils
from fv3core.decorators import FrozenStencil, get_stencils_with_varied_bounds
from fv3core.utils.grid import axis_offsets
from fv3core.utils.typing import FloatField, FloatFieldIJ


#
# Flux value stencils
# ---------------------
def compute_zonal_flux(flux: FloatField, a_in: FloatField, del_term: FloatFieldIJ):
    with computation(PARALLEL), interval(...):
        flux = del_term * (a_in[-1, 0, 0] - a_in)


def compute_meridional_flux(flux: FloatField, a_in: FloatField, del_term: FloatFieldIJ):
    with computation(PARALLEL), interval(...):
        flux = del_term * (a_in[0, -1, 0] - a_in)


#
# Q update stencil
# ------------------
def update_q(
    q: FloatField, rarea: FloatFieldIJ, fx: FloatField, fy: FloatField, cd: float
):
    with computation(PARALLEL), interval(...):
        q = q + cd * rarea * (fx - fx[1, 0, 0] + fy - fy[0, 1, 0])


#
# corner_fill
#
# Stencil that copies/fills in the appropriate corner values for qdel
# ------------------------------------------------------------------------
def corner_fill(grid, q):
    r3 = 1.0 / 3.0
    if grid.sw_corner:
        q[grid.is_, grid.js, :] = (
            q[grid.is_, grid.js, :]
            + q[grid.is_ - 1, grid.js, :]
            + q[grid.is_, grid.js - 1, :]
        ) * r3
        q[grid.is_ - 1, grid.js, :] = q[grid.is_, grid.js, :]
        q[grid.is_, grid.js - 1, :] = q[grid.is_, grid.js, :]
    if grid.se_corner:
        q[grid.ie, grid.js, :] = (
            q[grid.ie, grid.js, :]
            + q[grid.ie + 1, grid.js, :]
            + q[grid.ie, grid.js - 1, :]
        ) * r3
        q[grid.ie + 1, grid.js, :] = q[grid.ie, grid.js, :]
        for k in range(grid.npz):
            q[grid.ie, grid.js - 1, k] = q[grid.ie, grid.js, k]

    if grid.ne_corner:
        q[grid.ie, grid.je, :] = (
            q[grid.ie, grid.je, :]
            + q[grid.ie + 1, grid.je, :]
            + q[grid.ie, grid.je + 1, :]
        ) * r3
        q[grid.ie + 1, grid.je, :] = q[grid.ie, grid.je, :]
        q[grid.ie, grid.je + 1, :] = q[grid.ie, grid.je, :]

    if grid.nw_corner:
        q[grid.is_, grid.je, :] = (
            q[grid.is_, grid.je, :]
            + q[grid.is_ - 1, grid.je, :]
            + q[grid.is_, grid.je + 1, :]
        ) * r3
        for k in range(grid.npz):
            q[grid.is_ - 1, grid.je, k] = q[grid.is_, grid.je, k]

        q[grid.is_, grid.je + 1, :] = q[grid.is_, grid.je, :]


class HyperdiffusionDamping:
    """
    Fortran name is del2_cubed
    """

    def __init__(self, grid, nmax: int):
        """
        Args:
            grid: fv3core grid object
        """
        self.grid = spec.grid
        origin = self.grid.full_origin()
        domain = self.grid.domain_shape_full()
        ax_offsets = axis_offsets(spec.grid, origin, domain)
        self._fx = utils.make_storage_from_shape(
            self.grid.domain_shape_full(add=(1, 1, 1)), origin=origin
        )
        self._fy = utils.make_storage_from_shape(
            self.grid.domain_shape_full(add=(1, 1, 1)), origin=origin
        )
        self._ntimes = min(3, nmax)
        origins = []
        domains_x = []
        domains_y = []
        domains = []
        for n in range(1, self._ntimes + 1):
            nt = self._ntimes - n
            origins.append((self.grid.is_ - nt, self.grid.js - nt, 0))
            nx = self.grid.nic + 2 * nt
            ny = self.grid.njc + 2 * nt
            domains_x.append((nx + 1, ny, self.grid.npz))
            domains_y.append((nx, ny + 1, self.grid.npz))
            domains.append((nx, ny, self.grid.npz))
        self._compute_zonal_flux = get_stencils_with_varied_bounds(
            compute_zonal_flux,
            origins,
            domains_x,
        )
        self._compute_meridional_flux = get_stencils_with_varied_bounds(
            compute_meridional_flux,
            origins,
            domains_y,
        )
        self._update_q = get_stencils_with_varied_bounds(
            update_q,
            origins,
            domains,
        )

        self._copy_corners_x: corners.CopyCorners = corners.CopyCorners("x")
        """Stencil responsible for doing corners updates in x-direction."""
        self._copy_corners_y: corners.CopyCorners = corners.CopyCorners("y")
        """Stencil responsible for doing corners updates in y-direction."""

    def __call__(self, qdel: FloatField, cd: float):
        """
        Perform hyperdiffusion damping/filtering

        Args:
            qdel (inout): Variable to be filterd
            nmax: Number of times to apply filtering
            cd: Damping coeffcient
        """

        for n in range(self._ntimes):
            nt = self._ntimes - (n + 1)
            # Fill in appropriate corner values
            corner_fill(self.grid, qdel)

            if nt > 0:
                self._copy_corners_x(qdel)

            self._compute_zonal_flux[n](
                self._fx,
                qdel,
                self.grid.del6_v,
            )

            if nt > 0:
                self._copy_corners_y(qdel)

            self._compute_meridional_flux[n](
                self._fy,
                qdel,
                self.grid.del6_u,
            )

            # Update q values
            self._update_q[n](
                qdel,
                self.grid.rarea,
                self._fx,
                self._fy,
                cd,
            )
