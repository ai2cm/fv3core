from gt4py.gtscript import PARALLEL, computation, horizontal, interval, region

import fv3core._config as spec
import fv3core.utils.corners as corners
import fv3core.utils.gt4py_utils as utils
from fv3core.decorators import StencilWrapper
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
def corner_fill(q: FloatField):
    from __externals__ import i_end, i_start, j_end, j_start

    # Fills the same scalar value into three locations in q for each corner
    with computation(PARALLEL), interval(...):
        with horizontal(region[i_start, j_start]):
            q = (q[0, 0, 0] + q[-1, 0, 0] + q[0, -1, 0]) * (1.0 / 3.0)
        with horizontal(region[i_start - 1, j_start]):
            q = q[1, 0, 0]
        with horizontal(region[i_start, j_start - 1]):
            q = q[0, 1, 0]

        with horizontal(region[i_end, j_start]):
            q = (q[0, 0, 0] + q[1, 0, 0] + q[0, -1, 0]) * (1.0 / 3.0)
        with horizontal(region[i_end + 1, j_start]):
            q = q[-1, 0, 0]
        with horizontal(region[i_end, j_start - 1]):
            q = q[0, 1, 0]

        with horizontal(region[i_end, j_end]):
            q = (q[0, 0, 0] + q[1, 0, 0] + q[0, 1, 0]) * (1.0 / 3.0)
        with horizontal(region[i_end + 1, j_end]):
            q = q[-1, 0, 0]
        with horizontal(region[i_end, j_end + 1]):
            q = q[0, -1, 0]

        with horizontal(region[i_start, j_end]):
            q = (q[0, 0, 0] + q[-1, 0, 0] + q[0, 1, 0]) * (1.0 / 3.0)
        with horizontal(region[i_start - 1, j_end]):
            q = q[1, 0, 0]
        with horizontal(region[i_start, j_end + 1]):
            q = q[0, -1, 0]


class HyperdiffusionDamping:
    """
    Fortran name is del2cubed
    """

    def __init__(self, grid, qdel: FloatField, km: int):
        """
        Args:
            grid: fv3core grid object
            qdel (inout): variable to be filterd
            km: vertical levels (grid.npz)
        """
        self.grid = spec.grid
        self.origin = (self.grid.isd, self.grid.jsd, 0)
        self.domain = (self.grid.nid, self.grid.njd, km)
        ax_offsets = axis_offsets(spec.grid, self.origin, self.domain)
        self._fx = utils.make_storage_from_shape(
            qdel.shape, origin=self.origin, cache_key="del2cubed_fx"
        )
        self._fy = utils.make_storage_from_shape(
            qdel.shape, origin=self.origin, cache_key="del2cubed_fy"
        )

        self._corner_fill = StencilWrapper(
            func=corner_fill,
            externals={
                **ax_offsets,
            },
            origin=self.origin,
            domain=self.domain,
        )
        self._compute_zonal_flux = StencilWrapper(func=compute_zonal_flux)
        self._compute_meridional_flux = StencilWrapper(func=compute_meridional_flux)
        self._update_q = StencilWrapper(func=update_q)

    def __call__(self, qdel: FloatField, nmax: int, cd: float, km: int):
        """
        Perform hyperdiffusion damping

        Args:
            qdel (inout): variable to be filterd
            nmax: number of times to damping
            cd: damping coeffcient
            km: vertical levels (grid.npz)
        """
        ntimes = min(3, nmax)
        for n in range(1, ntimes + 1):
            nt = ntimes - n
            origin = (self.grid.is_ - nt, self.grid.js - nt, 0)

            # Fill in appropriate corner values
            self._corner_fill(qdel)

            if nt > 0:
                corners.copy_corners_x_stencil(
                    qdel,
                    origin=(self.grid.isd, self.grid.jsd, 0),
                    domain=(self.grid.nid, self.grid.njd, km),
                )
            nx = self.grid.nic + 2 * nt + 1  # (grid.ie+nt+1) - (grid.is_-nt) + 1
            ny = self.grid.njc + 2 * nt  # (grid.je+nt) - (grid.js-nt) + 1
            self._compute_zonal_flux(
                self._fx, qdel, self.grid.del6_v, origin=origin, domain=(nx, ny, km)
            )

            if nt > 0:
                corners.copy_corners_y_stencil(
                    qdel,
                    origin=(self.grid.isd, self.grid.jsd, 0),
                    domain=(self.grid.nid, self.grid.njd, km),
                )
            nx = self.grid.nic + 2 * nt  # (grid.ie+nt) - (grid.is_-nt) + 1
            ny = self.grid.njc + 2 * nt + 1  # (grid.je+nt+1) - (grid.js-nt) + 1
            self._compute_meridional_flux(
                self._fy, qdel, self.grid.del6_u, origin=origin, domain=(nx, ny, km)
            )

            # Update q values
            ny = self.grid.njc + 2 * nt  # (grid.je+nt) - (grid.js-nt) + 1
            self._update_q(
                qdel,
                self.grid.rarea,
                self._fx,
                self._fy,
                cd,
                origin=origin,
                domain=(nx, ny, km),
            )


def compute(qdel: FloatField, nmax: int, cd: float, km: int):
    hyperdiffusion = utils.cached_stencil_class(HyperdiffusionDamping)(
        spec.grid, qdel, km
    )
    hyperdiffusion(qdel, nmax, cd, km)
