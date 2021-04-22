import gt4py.gtscript as gtscript
from gt4py.gtscript import BACKWARD, PARALLEL, computation, interval

import fv3core.utils.global_constants as constants
from fv3core.decorators import StencilWrapper
from fv3core.utils import corners
from fv3core.utils.grid import axis_offsets
from fv3core.utils.typing import FloatField, FloatFieldIJ, FloatFieldK


DZ_MIN = constants.DZ_MIN


@gtscript.function
def p_weighted_average_top(vel, dp0):
    # TODO: ratio is a constant, where should this be placed?
    ratio = dp0 / (dp0 + dp0[1])
    return vel + (vel - vel[0, 0, 1]) * ratio


@gtscript.function
def p_weighted_average_bottom(vel, dp0):
    ratio = dp0[-1] / (dp0[-2] + dp0[-1])
    return vel[0, 0, -1] + (vel[0, 0, -1] - vel[0, 0, -2]) * ratio


@gtscript.function
def p_weighted_average_domain(vel, dp0):
    int_ratio = 1.0 / (dp0[-1] + dp0)
    return (dp0 * vel[0, 0, -1] + dp0[-1] * vel) * int_ratio


def update_dz_c(
    area: FloatFieldIJ,
    dp_ref: FloatFieldK,
    gz_surface: FloatFieldIJ,
    ut: FloatField,
    vt: FloatField,
    gz: FloatField,
    ws: FloatFieldIJ,
    dt2: float,
):
    """Update the model heights from the C-grid wind flux

    After the model runs c_sw and advances the c-grid variables half a timestep,
    the grid deforms with the flow in the Lagrangian coordinate system. This
    module updates the model heights based ontime averaged air mass fluxes. The
    update is based on finite volulme Lagrangin control-flow discretization,
    where Lagrangian surfaces are considered bounding material of the control
    volumes (Lin 2004).

    Args:
         dp_ref: vertical difference in column reference pressure (in)
         gz_surface: surface geopotential height (m) (in)
         ut: x-velocity on the C-grid, contravariant of the D-grid winds (in)
         vt: y-velocity on the C-grid, contravariant of the D-grid winds (in)
         gz: geopotential height of the model grid cells (m) (inout)
         ws: rate of change in the surface geopotential height
             from the C-grid wind. A geopotential vertical velocity estimate
             at the surface.  (inout)
         dt2: timestep of the C-grid update in seconds (in)
    Grid variable inputs:
         area
    """
    with computation(PARALLEL):
        with interval(0, 1):
            u_average = p_weighted_average_top(ut, dp_ref)
            v_average = p_weighted_average_top(vt, dp_ref)
        with interval(1, -1):
            u_average = p_weighted_average_domain(ut, dp_ref)
            v_average = p_weighted_average_domain(vt, dp_ref)
        with interval(-1, None):
            u_average = p_weighted_average_bottom(ut, dp_ref)
            v_average = p_weighted_average_bottom(vt, dp_ref)
    with computation(PARALLEL), interval(...):
        gz_tmp = corners.fill_corners_2cells_mult_x(gz, gz, 1.0, 1.0, 1.0, 1.0)
        fx = u_average * (gz_tmp[-1, 0, 0] if u_average > 0.0 else gz_tmp)
        gz_tmp = corners.fill_corners_2cells_mult_y(gz_tmp, gz_tmp, 1.0, 1.0, 1.0, 1.0)
        fy = v_average * (gz_tmp[0, -1, 0] if v_average > 0.0 else gz_tmp)
        gz = (gz_tmp * area + fx - fx[1, 0, 0] + fy - fy[0, 1, 0]) / (
            area + u_average - u_average[1, 0, 0] + v_average - v_average[0, 1, 0]
        )
    with computation(BACKWARD):
        with interval(-1, None):
            rdt = 1.0 / dt2
            gz_surface = gz_surface
            ws = (gz_surface - gz) * rdt
        with interval(0, -1):
            gz_min = gz[0, 0, 1] + DZ_MIN
            gz = gz if gz > gz_min else gz_min


class UpdateGeopotentialHeightOnCGrid:
    def __init__(self, grid):
        self.grid = grid
        origin = self.grid.compute_origin(add=(-1, -1, 0))
        domain = self.grid.domain_shape_compute(add=(2, 2, 1))
        ax_offsets = axis_offsets(self.grid, origin, domain)
        self._update_dz_c = StencilWrapper(
            update_dz_c, origin=origin, domain=domain, externals=ax_offsets
        )

    def __call__(
        self,
        dp_ref: FloatFieldK,
        gz_surface: FloatFieldIJ,
        ut: FloatField,
        vt: FloatField,
        gz: FloatField,
        ws: FloatFieldIJ,
        dt: float,
    ):
        """
        Args:
            dp_ref: layer thickness in Pa
            gz_surface: surface height in m
            ut: horizontal wind (TODO: covariant or contravariant?)
            vt: horizontal wind (TODO: covariant or contravariant?)
            gz: geopotential height (TODO: on cell mid levels or interfaces?)
            ws: surface vertical wind implied by horizontal motion over topography
            dt: timestep over which to evolve the geopotential height
        """
        return self._update_dz_c(self.grid.area, dp_ref, gz_surface, ut, vt, gz, ws, dt)
