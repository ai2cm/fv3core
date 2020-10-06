import gt4py.gtscript as gtscript
from gt4py.gtscript import __INLINED, PARALLEL, computation, interval

import fv3core._config as spec
import fv3core.utils.gt4py_utils as utils
from fv3core.utils.corners import fill_4corners_x_func, fill_4corners_y_func


sd = utils.sd


@gtscript.function
def nonhydro_x_fluxes(delp: sd, pt: sd, w: sd, utc: sd):
    fx = pt[-1, 0, 0] if utc > 0.0 else pt
    fx1 = delp[-1, 0, 0] if utc > 0.0 else delp
    fx2 = w[-1, 0, 0] if utc > 0.0 else w

    fx = fx1 * fx
    fx1 = utc * fx1
    fx2 = fx1 * fx2

    return fx, fx1, fx2


@gtscript.function
def nonhydro_y_fluxes(delp: sd, pt: sd, w: sd, vtc: sd):
    fy = pt[0, -1, 0] if vtc > 0.0 else pt
    fy1 = delp[0, -1, 0] if vtc > 0.0 else delp
    fy2 = w[0, -1, 0] if vtc > 0.0 else w

    fy = fy1 * fy
    fy1 = vtc * fy1
    fy2 = fy1 * fy2

    return fy, fy1, fy2


@utils.stencil
def transportdelp(
    delp: sd, pt: sd, utc: sd, vtc: sd, wc: sd, rarea: sd, delpc: sd, ptc: sd
):
    """In the words of the Fortran code: 'Transport delp'

    Args:
        delp: What is transported (input)
        pt: Pressure (input)
        utc: x-velocity on C-grid (input)
        vtc: y-velocity on C-grid (input)
        wc: z-velocity on C-grid (input, output)
        rarea: Inverse areas (input) -- IJ field

    TODO: Remove these when a function
        delpc: Updated delp
        ptc: Updated pt
    """

    with computation(PARALLEL), interval(...):
        if __INLINED(spec.namelist.grid_type < 3):
            fill_4corners_x_func(delp)
            fill_4corners_x_func(pt)

        fx, fx1, fx2 = nonhydro_x_fluxes(delp, pt, wc, utc)

        if __INLINED(spec.namelist.grid_type < 3):
            fill_4corners_y_func(delp)
            fill_4corners_y_func(pt)

        fy, fy1, fy2 = nonhydro_x_fluxes(delp, pt, wc, vtc)

        # Compute outputs
        delpc = delp + (fx1 - fx1[1, 0, 0] + fy1 - fy1[0, 1, 0]) * rarea
        ptc = (pt * delp + (fx - fx[1, 0, 0] + fy - fy[0, 1, 0]) * rarea) / delpc
        wc = (wc * delp + (fx2 - fx2[1, 0, 0] + fy2 - fy2[0, 1, 0]) * rarea) / delpc
