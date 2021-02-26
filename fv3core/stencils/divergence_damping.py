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
def ptc_main(u: sd, va: sd, vc: sd, cosa_v: sd, sina_v: sd, dyc: sd, sin_sg4: sd, sin_sg2: sd, ptc: sd):
    from __externals__ import i_start, i_end, j_start, j_end
    with computation(PARALLEL), interval(...):
        ptc[0, 0, 0] = (u - 0.5 * (va[0, -1, 0] + va) * cosa_v) * dyc * sina_v
        with horizontal(region[:, j_start], region[:, j_end + 1]):
            ptc[0, 0, 0] = u * dyc * sin_sg4[0, -1, 0] if vc > 0 else u * dyc * sin_sg2
            
@gtstencil()
def ptc_y_edge(u: sd, vc: sd, dyc: sd, sin_sg4: sd, sin_sg2: sd, ptc: sd):
    with computation(PARALLEL), interval(...):
        ptc[0, 0, 0] = u * dyc * sin_sg4[0, -1, 0] if vc > 0 else u * dyc * sin_sg2


@gtstencil()
def vorticity_main(v: sd, ua: sd, cosa_u: sd, sina_u: sd, dxc: sd, vort: sd):
    with computation(PARALLEL), interval(...):
        vort[0, 0, 0] = (v - 0.5 * (ua[-1, 0, 0] + ua) * cosa_u) * dxc * sina_u


@gtstencil()
def vorticity_x_edge(v: sd, uc: sd, dxc: sd, sin_sg3: sd, sin_sg1: sd, vort: sd):
    with computation(PARALLEL), interval(...):
        vort[0, 0, 0] = v * dxc * sin_sg3[-1, 0, 0] if uc > 0 else v * dxc * sin_sg1


@gtstencil()
def delpc_main(vort: sd, ptc: sd, delpc: sd):
    with computation(PARALLEL), interval(...):
        delpc[0, 0, 0] = vort[0, -1, 0] - vort + ptc[-1, 0, 0] - ptc

@gtscript.function
def remove_extra_term_south_corner(extra: sd, field: sd):
    return field - extra[0, -1, 0]

@gtstencil()
def corner_south_remove_extra_term(vort: sd, delpc: sd):
    with computation(PARALLEL), interval(...):
        delpc = remove_extra_term_south_corner(vort, delpc)

@gtscript.function
def remove_extra_term_north_corner(extra: sd, field: sd):
    return field + extra

@gtstencil()
def corner_north_remove_extra_term(vort: sd, delpc: sd):
    with computation(PARALLEL), interval(...):
        delpc = remove_extra_term_north_corner(vort, delpc)


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
    
    with horizontal(region[local_is - 0:local_ie + 2 , local_js:local_je + 2  ]):
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
    if not grid.nested:
        # TODO: ptc and vort are equivalent, but x vs y, consolidate if possible.
        ptc_main(
            u,
            va, vc,
            grid.cosa_v,
            grid.sina_v,
            grid.dyc, grid.sin_sg4, grid.sin_sg2,
            ptc,
            origin=(grid.is_ - 1, grid.js, kstart),
            domain=(grid.nic + 2, grid.njc + 1, nk),
        )
        #y_edge_domain = (grid.nic + 2, 1, nk)
        #if grid.south_edge:
        #    ptc_y_edge(
        #        u,
        #        vc,
        #        grid.dyc,
        #        grid.sin_sg4,
        #        grid.sin_sg2,
        #        ptc,
        #        origin=(grid.is_ - 1, grid.js, kstart),
        #        domain=y_edge_domain,
        #    )
        #if grid.north_edge:
        #    ptc_y_edge(
        #        u,
        #        vc,
        #        grid.dyc,
        #        grid.sin_sg4,
        #        grid.sin_sg2,
        #        ptc,
        #        origin=(grid.is_ - 1, grid.je + 1, kstart),
        #        domain=y_edge_domain,
        #    )

        vorticity_main(
            v,
            ua,
            grid.cosa_u,
            grid.sina_u,
            grid.dxc,
            vort,
            origin=(is2, grid.js - 1, kstart),
            domain=(ie1 - is2 + 1, grid.njc + 2, nk),
        )
        x_edge_domain = (1, grid.njc + 2, nk)
        if grid.west_edge:
            vorticity_x_edge(
                v,
                uc,
                grid.dxc,
                grid.sin_sg3,
                grid.sin_sg1,
                vort,
                origin=(grid.is_, grid.js - 1, kstart),
                domain=x_edge_domain,
            )
        if grid.east_edge:
            vorticity_x_edge(
                v,
                uc,
                grid.dxc,
                grid.sin_sg3,
                grid.sin_sg1,
                vort,
                origin=(grid.ie + 1, grid.js - 1, kstart),
                domain=x_edge_domain,
            )
    else:
        raise Exception("nested not implemented")
    #compute_origin = (grid.is_, grid.js, kstart)
    #compute_domain = (grid.nic + 1, grid.njc + 1, nk)
    delpc_main(vort, ptc, delpc, origin=compute_origin, domain=compute_domain)
    corner_domain = (1, 1, nk)
    if grid.sw_corner:
        corner_south_remove_extra_term(
            vort, delpc, origin=(grid.is_, grid.js, kstart), domain=corner_domain
        )
    if grid.se_corner:
        corner_south_remove_extra_term(
            vort, delpc, origin=(grid.ie + 1, grid.js, kstart), domain=corner_domain
        )
    if grid.ne_corner:
        corner_north_remove_extra_term(
            vort, delpc, origin=(grid.ie + 1, grid.je + 1, kstart), domain=corner_domain
        )
    if grid.nw_corner:
        corner_north_remove_extra_term(
            vort, delpc, origin=(grid.is_, grid.je + 1, kstart), domain=corner_domain
        )

    damping_nord0_stencil(
        grid.rarea_c,
        delpc,
        vort,
        ke,
        grid.da_min_c,
        d2_bg,
        spec.namelist.dddmp,
        dt,
        origin=compute_origin,
        domain=compute_domain,
    )
