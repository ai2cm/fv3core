import gt4py.gtscript as gtscript
from gt4py.gtscript import BACKWARD, PARALLEL, computation, horizontal, interval, region

import fv3core.utils.global_constants as constants
from fv3core.decorators import gtstencil
from fv3core.utils import corners
from fv3core.utils.typing import FloatField


DZ_MIN = constants.DZ_MIN


@gtscript.function
def p_weighted_average_top(vel, dp0):
    ratio = dp0 / (dp0 + dp0[0, 0, 1])
    return vel + (vel - vel[0, 0, 1]) * ratio


@gtscript.function
def p_weighted_average_bottom(vel, dp0):
    ratio = dp0[0, 0, -1] / (dp0[0, 0, -2] + dp0[0, 0, -1])
    return vel[0, 0, -1] + (vel[0, 0, -1] - vel[0, 0, -2]) * ratio


@gtscript.function
def p_weighted_average_domain(vel, dp0):
    int_ratio = 1.0 / (dp0[0, 0, -1] + dp0)
    return (dp0 * vel[0, 0, -1] + dp0[0, 0, -1] * vel) * int_ratio


@gtscript.function
def xy_flux(gz_x, gz_y, xfx, yfx):
    fx = xfx * (gz_x[-1, 0, 0] if xfx > 0.0 else gz_x)
    fy = yfx * (gz_y[0, -1, 0] if yfx > 0.0 else gz_y)
    return fx, fy


@gtstencil()
def update_dz_c_stencil(
    area: FloatField,
    dp_ref: FloatField,
    zs: FloatField,
    ut: FloatField,
    vt: FloatField,
    gz: FloatField,
    ws: FloatField,
    dt2: float,
):
    """Update the model heights from the C-grid wind flux

    After the model runs c_sw and advances the c-grid variables half a timestep,
    the grid deforms with the flow in the Lagrangian coordinate system. This
    module updates the model heights based on the flux, and the rate of change of
    the lowest model height compared to the fixed surface height.

    Args:
         dp_ref: vertical delta in column reference pressure(in)
         zs: surface height (m) (in)
         ut: x-velocity on the C-grid, contravariantof the D-grid winds(in)
         vt: y-velocity on the C-grid, contravariantof the D-grid winds(in)
         gz: height of the model grid cells (m) (inout)
         ws: change in the height of the lowest model layer this C-grid timestep(inout)
         dt2: half a model timestep (for C-grid update) in seconds (in)
    Grid variable inputs:
         area
    """
    from __externals__ import local_ie, local_is, local_je, local_js

    with computation(PARALLEL):
        with interval(0, 1):
            xfx = p_weighted_average_top(ut, dp_ref)
            yfx = p_weighted_average_top(vt, dp_ref)
        with interval(1, -1):
            xfx = p_weighted_average_domain(ut, dp_ref)
            yfx = p_weighted_average_domain(vt, dp_ref)
        with interval(-1, None):
            xfx = p_weighted_average_bottom(ut, dp_ref)
            yfx = p_weighted_average_bottom(vt, dp_ref)
    with computation(PARALLEL), interval(...):
        gz_x = gz
        gz_x = corners.fill_corners_2cells_mult_x(gz_x, gz_x, 1.0, 1.0, 1.0, 1.0)
        gz_y = gz_x
        gz_y = corners.fill_corners_2cells_mult_y(gz_y, gz_y, 1.0, 1.0, 1.0, 1.0)
        fx, fy = xy_flux(gz_x, gz_y, xfx, yfx)
        # TODO: region for local validation only
        with horizontal(
            region[local_is - 1 : local_ie + 2, local_js - 1 : local_je + 2]
        ):
            gz = (gz_y * area + fx - fx[1, 0, 0] + fy - fy[0, 1, 0]) / (
                area + xfx - xfx[1, 0, 0] + yfx - yfx[0, 1, 0]
            )
    with computation(PARALLEL), interval(-1, None):
        rdt = 1.0 / dt2
        # TODO: region for local validation only
        with horizontal(
            region[local_is - 1 : local_ie + 2, local_js - 1 : local_je + 2]
        ):
            ws = (zs - gz) * rdt
    with computation(BACKWARD), interval(0, -1):
        # TODO region for local validation only
        with horizontal(
            region[local_is - 1 : local_ie + 2, local_js - 1 : local_je + 2]
        ):
            gz_kp1 = gz[0, 0, 1] + DZ_MIN
            gz = gz if gz > gz_kp1 else gz_kp1
