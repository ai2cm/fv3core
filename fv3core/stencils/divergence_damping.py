import gt4py.gtscript as gtscript
from gt4py.gtscript import PARALLEL, computation, interval

import fv3core._config as spec
import fv3core.stencils.a2b_ord4 as a2b_ord4
import fv3core.stencils.basic_operations as basic
import fv3core.utils.corners as corners
import fv3core.utils.gt4py_utils as utils
from fv3core.decorators import gtstencil
from fv3core.stencils.basic_operations import copy_stencil


sd = utils.sd


@gtstencil()
def ptc_main(u: sd, v: sd, ua: sd, va: sd, uc: sd, vc: sd, cosa_u: sd, cosa_v: sd, sina_u: sd, sina_v: sd, dxc: sd, dyc: sd, sin_sg1: sd, sin_sg2:sd, sin_sg3: sd, sin_sg4: sd, rarea_c: sd, ptc: sd, vort: sd, delpc: sd, ke: sd, da_min_c: float, d2_bg: float, dt: float):
    from __externals__ import namelist, i_start, i_end, j_start, j_end, local_is, local_ie, local_js, local_je
    with computation(PARALLEL), interval(...):
        ptc = (u - 0.5 * (va[0, -1, 0] + va) * cosa_v) * dyc * sina_v
        with horizontal(region[:, j_start], region[:, j_end + 1]):
            ptc = u * dyc * sin_sg4[0, -1, 0] if vc > 0 else u * dyc * sin_sg2
        vort_copy = vort
        with horizontal(region[local_is:local_ie + 2, :]):
            vort = (v - 0.5 * (ua[-1, 0, 0] + ua) * cosa_u) * dxc * sina_u
        with horizontal(region[i_start,:], region[i_end + 1,:]):
            vort = vort_copy
        with horizontal(region[i_start, :], region[i_end + 1, :]):
            vort = v * dxc * sin_sg3[-1, 0, 0] if uc > 0 else v * dxc * sin_sg1
        with horizontal(region[local_is: local_ie + 2, local_js: local_je + 2]):
            delpc = vort[0, -1, 0] - vort + ptc[-1, 0, 0] - ptc
        with horizontal(region[i_start, j_start], region[i_end + 1, j_start]):
            delpc = remove_extra_term_south_corner(vort, delpc)
        with horizontal(region[i_start, j_end + 1], region[i_end + 1, j_end + 1]):
            delpc= remove_extra_term_north_corner(vort, delpc)
        with horizontal(region[local_is: local_ie + 2, local_js: local_je + 2]):
            delpc = rarea_c * delpc
            delpcdt = delpc * dt
            absdelpcdt = delpcdt if delpcdt >= 0 else -delpcdt
            damp = damp_tmp(absdelpcdt, da_min_c, d2_bg, namelist.dddmp)
            vort = damp * delpc
            ke = ke + vort

@gtscript.function
def remove_extra_term_south_corner(extra: sd, field: sd):
    # from __externals__ import i_start, i_end,  j_start
    # TODO: why does this not work?
    # with horizontal(region[i_start, j_start], region[i_end + 1, j_start]):
    #    field = field - extra[0, -1, 0]
    # return field
    return field - extra[0, -1, 0]

@gtscript.function
def remove_extra_term_north_corner(extra: sd, field: sd):
    # TODO: why does this not work?
    # from __externals__ import i_start, i_end, j_end
    # with horizontal(region[i_start, j_end + 1], region[i_end + 1, j_end + 1]):
    #     field = field + extra
    # return field
    return field + extra

@gtscript.function
def damp_tmp(q, da_min_c, d2_bg, dddmp):
    tmpddd = dddmp * q
    mintmp = 0.2 if 0.2 < tmpddd else tmpddd
    maxd2 = d2_bg if d2_bg > mintmp else mintmp
    damp = da_min_c * maxd2
    return damp


@gtstencil()
def damping_nord0_stencil(
    rarea_c: sd,
    delpc: sd,
    vort: sd,
    ke: sd,
    da_min_c: float,
    d2_bg: float,
    dddmp: float,
    dt: float,
):
    with computation(PARALLEL), interval(...):
        delpc[0, 0, 0] = rarea_c * delpc
        delpcdt = delpc * dt
        absdelpcdt = delpcdt if delpcdt >= 0 else -delpcdt
        damp = damp_tmp(absdelpcdt, da_min_c, d2_bg, dddmp)
        vort[0, 0, 0] = damp * delpc
        ke[0, 0, 0] = ke + vort


@gtstencil()
def damping_nord_highorder_stencil(
    vort: sd,
    ke: sd,
    delpc: sd,
    divg_d: sd,
    da_min_c: float,
    d2_bg: float,
    dddmp: float,
    dd8: float,
):
    with computation(PARALLEL), interval(...):
        damp = damp_tmp(vort, da_min_c, d2_bg, dddmp)
        vort = damp * delpc + dd8 * divg_d
        ke = ke + vort


@gtscript.function
def vc_from_divg(divg_d: sd, divg_u: sd):
    return (divg_d[1, 0, 0] - divg_d) * divg_u

@gtscript.function
def uc_from_divg(divg_d: sd, divg_v: sd):
    return (divg_d[0, 1, 0] - divg_d) * divg_v


@gtscript.function
def redo_divg_d(uc: sd, vc: sd):
    return uc[0, -1, 0] - uc + vc[-1, 0, 0] - vc


@gtstencil()
def smagorinksy_diffusion_approx(delpc: sd, vort: sd, absdt: float):
    with computation(PARALLEL), interval(...):
        vort = absdt * (delpc ** 2.0 + vort ** 2.0) ** 0.5


def vorticity_calc(wk, vort, delpc, dt, nord, kstart, nk):
    if nord != 0:
        if spec.namelist.dddmp < 1e-5:
            vort[:, :, kstart : kstart + nk] = 0
        else:
            if spec.namelist.grid_type < 3:
                a2b_ord4.compute(wk, vort, kstart, nk, False)
                smagorinksy_diffusion_approx(
                    delpc,
                    vort,
                    abs(dt),
                    origin=(spec.grid.is_, spec.grid.js, kstart),
                    domain=(spec.grid.nic + 1, spec.grid.njc + 1, nk),
                )
            else:
                raise Exception("Not implemented, smag_corner")

@gtscript.function
def damping_nt2(rarea_c: sd, divg_u: sd, divg_v: sd, divg_d: sd, uc: sd, vc: sd):
    from __externals__ import local_is, local_ie, local_js, local_je, i_start, i_end, j_start, j_end
    divg_d = corners.fill_corners_2d_bgrid_x(divg_d)
    with horizontal(region[local_is - 3:local_ie + 4 , local_js - 2:local_je + 4  ]):
        vc = vc_from_divg(divg_d, divg_u)
    divg_d = corners.fill_corners_2d_bgrid_y(divg_d)
    with horizontal(region[local_is - 2:local_ie + 4 , local_js - 3:local_je + 4  ]):
        uc = uc_from_divg(divg_d, divg_v) 
    vc, uc = corners.fill_corners_dgrid_fn(vc, uc, -1.0)
    with horizontal(region[local_is - 2:local_ie + 4 , local_js - 2:local_je + 4  ]):
        divg_d = redo_divg_d(uc, vc) 
    with horizontal(region[i_start, j_start], region[i_end + 1, j_start]):
        divg_d = remove_extra_term_south_corner(uc, divg_d)
    with horizontal(region[i_start, j_end + 1], region[i_end + 1, j_end + 1]):
        divg_d= remove_extra_term_north_corner(uc, divg_d)
    # ASSUMED not grid.stretched_grid
    with horizontal(region[local_is - 2:local_ie + 4 , local_js - 2:local_je + 4] ):
        divg_d = basic.adjustmentfactor(rarea_c, divg_d)
    return divg_d, uc, vc

@gtscript.function
def damping_nt1(rarea_c: sd, divg_u: sd, divg_v: sd, divg_d: sd, uc: sd, vc: sd):
    from __externals__ import local_is, local_ie, local_js, local_je, i_start, i_end, j_start, j_end
    divg_d = corners.fill_corners_2d_bgrid_x(divg_d)
    with horizontal(region[local_is - 2:local_ie + 3 , local_js - 1:local_je + 3  ]):
        vc = vc_from_divg(divg_d, divg_u)
    divg_d = corners.fill_corners_2d_bgrid_y(divg_d)
    with horizontal(region[local_is - 1:local_ie + 3 , local_js - 2:local_je + 3  ]):
        uc = uc_from_divg(divg_d, divg_v)
    vc, uc = corners.fill_corners_dgrid_fn(vc, uc, -1.0)
    with horizontal(region[local_is - 1:local_ie + 3 , local_js - 1:local_je + 3  ]):
        divg_d = redo_divg_d(uc, vc)
    with horizontal(region[i_start, j_start], region[i_end + 1, j_start]):
        divg_d = remove_extra_term_south_corner(uc, divg_d)
    with horizontal(region[i_start, j_end + 1], region[i_end + 1, j_end + 1]):
        divg_d= remove_extra_term_north_corner(uc, divg_d)
    # ASSUMED not grid.stretched_grid                                              
    with horizontal(region[local_is - 1:local_ie + 3, local_js - 1:local_je + 3] ):
        divg_d = basic.adjustmentfactor(rarea_c, divg_d)
    return divg_d, uc, vc

@gtscript.function
def damping_nt0(rarea_c: sd, divg_u: sd, divg_v: sd, divg_d: sd, uc: sd, vc: sd):
    from __externals__ import local_is, local_ie, local_js, local_je, i_start, i_end, j_start, j_end
    
    with horizontal(region[local_is - 1:local_ie + 2 , local_js:local_je + 2  ]):
        vc = vc_from_divg(divg_d, divg_u)
    
    with horizontal(region[local_is:local_ie + 2 , local_js - 1:local_je + 2  ]):
        uc = uc_from_divg(divg_d, divg_v)
    
    with horizontal(region[local_is:local_ie + 2 , local_js:local_je + 2  ]):
        divg_d = redo_divg_d(uc, vc)
    with horizontal(region[i_start, j_start], region[i_end + 1, j_start]):
        divg_d = remove_extra_term_south_corner(uc, divg_d)
    with horizontal(region[i_start, j_end + 1], region[i_end + 1, j_end + 1]):
        divg_d= remove_extra_term_north_corner(uc, divg_d)
    # ASSUMED not grid.stretched_grid 
    with horizontal(region[local_is:local_ie + 2 , local_js:local_je + 2] ):
        divg_d = basic.adjustmentfactor(rarea_c, divg_d)
    return divg_d, uc, vc

@gtstencil(externals={})
def damping_nonzero_nord(rarea_c: sd, divg_u: sd, divg_v: sd, divg_d: sd, uc: sd, vc: sd, delpc: sd):
    from __externals__ import local_is, local_ie, local_js, local_je
    with computation(PARALLEL), interval(...):
        # TODO: needed for validation of DivergenceDamping, D_SW, but not DynCore
        with horizontal(region[local_is: local_ie + 2, local_js:local_je + 2]):
            delpc = divg_d
        # TODO, can we call the same function 3 times and let gt4py do the extent analysis?
        # currently does not work because corner calculations need entire array,
        # and vc/uc need offsets
        divg_d, uc, vc = damping_nt2(rarea_c, divg_u, divg_v, divg_d, uc, vc)
        divg_d, uc, vc = damping_nt1(rarea_c, divg_u, divg_v, divg_d, uc, vc)
        divg_d, uc, vc = damping_nt0(rarea_c, divg_u, divg_v, divg_d, uc, vc)
def compute(
    u,
    v,
    va,
    ptc,
    vort,
    ua,
    divg_d,
    vc,
    uc,
    delpc,
    ke,
    wk,
    d2_bg,
    dt,
    nord,
    kstart=0,
    nk=None,
):
    grid = spec.grid
    if nk is None:
        nk = grid.npz - kstart
    # Avoid running center-domain computation on tile edges, since they'll be
    # overwritten.
    is2 = grid.is_ + 1 if grid.west_edge else grid.is_
    ie1 = grid.ie if grid.east_edge else grid.ie + 1
    nord = int(nord)
    if nord == 0:
        damping_zero_order(
            u, v, va, ptc, vort, ua, vc, uc, delpc, ke, d2_bg, dt, is2, ie1, kstart, nk
        )
    else:
        
        damping_nonzero_nord(grid.rarea_c, grid.divg_u, grid.divg_v, divg_d, uc, vc, delpc, origin=(grid.isd, grid.jsd, kstart), domain=(grid.nid + 1, grid.njd + 1, nk))
            

        vorticity_calc(wk, vort, delpc, dt, nord, kstart, nk)
        # TODO put this inside of stencil when grid variables an be externals
        if grid.stretched_grid:
            dd8 = grid.da_min * spec.namelist.d4_bg ** (nord + 1)
        else:
            dd8 = (grid.da_min_c * spec.namelist.d4_bg) ** (nord + 1)
        damping_nord_highorder_stencil(
            vort,
            ke,
            delpc,
            divg_d,
            grid.da_min_c,
            d2_bg,
            spec.namelist.dddmp,
            dd8,
            origin=(grid.is_, grid.js, kstart),
            domain=(grid.nic + 1, grid.njc + 1, nk),
        )

    return vort, ke, delpc


def damping_zero_order(
    u, v, va, ptc, vort, ua, vc, uc, delpc, ke, d2_bg, dt, is2, ie1, kstart, nk
):
    grid = spec.grid
    compute_origin = (grid.is_, grid.js, kstart)
    compute_domain = (grid.nic + 1, grid.njc + 1, nk)
    is2 = grid.is_ + 1 if grid.west_edge else grid.is_
    ie1 = grid.ie if grid.east_edge else grid.ie + 1
    if grid.nested:
        raise Exception("nested not implemented")

    ptc_main(
        u, v,
        ua, va, uc, vc,
        grid.cosa_u, grid.cosa_v,
        grid.sina_u, grid.sina_v,
        grid.dxc, grid.dyc, grid.sin_sg1, grid.sin_sg2,grid.sin_sg3, grid.sin_sg4,grid.rarea_c, 
        ptc,vort, delpc, ke, grid.da_min_c, d2_bg, dt,
        origin=(grid.is_ - 1, grid.js - 1, kstart),
        domain=(grid.nic + 2, grid.njc + 2, nk),
    )
    #damping_nord0_stencil(
    #    grid.rarea_c,
    #    delpc,
    #    vort,
    #    ke,
    #    grid.da_min_c,
    #    d2_bg,
    #    spec.namelist.dddmp,
    #    dt,
    #    origin=compute_origin,
    #    domain=compute_domain,
    #)
