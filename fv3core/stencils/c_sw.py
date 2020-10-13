#!/usr/bin/env python3
import gt4py.gtscript as gtscript
from gt4py.gtscript import PARALLEL, computation, interval

import fv3core._config as spec
import fv3core.stencils.circulation_cgrid as circulation_cgrid
import fv3core.stencils.d2a2c_vect as d2a2c
import fv3core.stencils.divergence_corner as divergence_corner
import fv3core.stencils.ke_c_sw as ke_c_sw
import fv3core.stencils.vorticitytransport_cgrid as vorticity_transport
import fv3core.utils.gt4py_utils as utils
from fv3core.decorators import gtstencil
from fv3core.utils.corners import fill_4corners_x, fill_4corners_y


sd = utils.sd


@gtstencil()
def geoadjust_ut(ut: sd, dy: sd, sin_sg3: sd, sin_sg1: sd, dt2: float):
    with computation(PARALLEL), interval(...):
        ut[0, 0, 0] = (
            dt2 * ut * dy * sin_sg3[-1, 0, 0] if ut > 0 else dt2 * ut * dy * sin_sg1
        )


@gtstencil()
def geoadjust_vt(vt: sd, dx: sd, sin_sg4: sd, sin_sg2: sd, dt2: float):
    with computation(PARALLEL), interval(...):
        vt[0, 0, 0] = (
            dt2 * vt * dx * sin_sg4[0, -1, 0] if vt > 0 else dt2 * vt * dx * sin_sg2
        )


@gtstencil()
def absolute_vorticity(vort: sd, fC: sd, rarea_c: sd):
    with computation(PARALLEL), interval(...):
        vort[0, 0, 0] = fC + rarea_c * vort


@gtscript.function
def nonhydro_x_fluxes(delp: sd, pt: sd, w: sd, utc: sd):
    fx1 = delp[-1, 0, 0] if utc > 0.0 else delp
    fx = pt[-1, 0, 0] if utc > 0.0 else pt
    fx2 = w[-1, 0, 0] if utc > 0.0 else w
    fx1 = utc * fx1
    fx = fx1 * fx
    fx2 = fx1 * fx2
    return fx, fx1, fx2


@gtscript.function
def nonhydro_y_fluxes(delp: sd, pt: sd, w: sd, vtc: sd):
    fy1 = delp[0, -1, 0] if vtc > 0.0 else delp
    fy = pt[0, -1, 0] if vtc > 0.0 else pt
    fy2 = w[0, -1, 0] if vtc > 0.0 else w
    fy1 = vtc * fy1
    fy = fy1 * fy
    fy2 = fy1 * fy2
    return fy, fy1, fy2


@gtstencil()
def transportdelp(
    delp: sd, pt: sd, utc: sd, vtc: sd, w: sd, rarea: sd, delpc: sd, ptc: sd, wc: sd
):
    """Transport delp.

    Args:
        delp: What is transported (input)
        pt: Pressure (input)
        utc: x-velocity on C-grid (input)
        vtc: y-velocity on C-grid (input)
        w: z-velocity on C-grid (input)
        rarea: Inverse areas (input) -- IJ field
        delpc: Updated delp (output)
        ptc: Updated pt (output)
        wc: Updated w (output)
    """

    with computation(PARALLEL), interval(...):
        if __INLINED(spec.namelist.grid_type < 3):
            # additional assumption (not grid.nested)
            delp = fill_4corners_x(delp)
            pt = fill_4corners_x(pt)
            w = fill_4corners_x(w)

        fx, fx1, fx2 = nonhydro_x_fluxes(delp, pt, w, utc)

        if __INLINED(spec.namelist.grid_type < 3):
            # additional assumption (not grid.nested)
            delp = fill_4corners_y(delp)
            pt = fill_4corners_y(pt)
            w = fill_4corners_y(w)

        fy, fy1, fy2 = nonhydro_y_fluxes(delp, pt, w, vtc)

        delpc = delp + (fx1 - fx1[1, 0, 0] + fy1 - fy1[0, 1, 0]) * rarea
        ptc = (pt * delp + (fx - fx[1, 0, 0] + fy - fy[0, 1, 0]) * rarea) / delpc
        wc = (w * delp + (fx2 - fx2[1, 0, 0] + fy2 - fy2[0, 1, 0]) * rarea) / delpc


def compute(delp, pt, u, v, w, uc, vc, ua, va, ut, vt, divgd, omga, dt2):
    grid = spec.grid
    dord4 = True
    origin_halo1 = (grid.is_ - 1, grid.js - 1, 0)
    fx = utils.make_storage_from_shape(delp.shape, origin_halo1)
    fy = utils.make_storage_from_shape(delp.shape, origin_halo1)
    delpc = utils.make_storage_from_shape(delp.shape, origin=origin_halo1)
    ptc = utils.make_storage_from_shape(pt.shape, origin=origin_halo1)
    d2a2c.compute(dord4, uc, vc, u, v, ua, va, ut, vt)
    if spec.namelist.nord > 0:
        divergence_corner.compute(u, v, ua, va, divgd)
    geo_origin = (grid.is_ - 1, grid.js - 1, 0)
    geoadjust_ut(
        ut,
        grid.dy,
        grid.sin_sg3,
        grid.sin_sg1,
        dt2,
        origin=geo_origin,
        domain=(grid.nic + 3, grid.njc + 2, grid.npz),
    )
    geoadjust_vt(
        vt,
        grid.dx,
        grid.sin_sg4,
        grid.sin_sg2,
        dt2,
        origin=geo_origin,
        domain=(grid.nic + 2, grid.njc + 3, grid.npz),
    )
    transportdelp(
        delp,
        pt,
        ut,
        vt,
        w,
        grid.rarea,
        delpc,
        ptc,
        omga,
        origin=geo_origin,
        domain=(grid.nic + 2, grid.njc + 2, grid.npz),
    )
    ke, vort = ke_c_sw.compute(uc, vc, u, v, ua, va, dt2)
    circulation_cgrid.compute(uc, vc, vort)
    absolute_vorticity(
        vort,
        grid.fC,
        grid.rarea_c,
        origin=grid.compute_origin(),
        domain=(grid.nic + 1, grid.njc + 1, grid.npz),
    )
    vorticity_transport.compute(uc, vc, vort, ke, v, u, fx, fy, dt2)
    return delpc, ptc
