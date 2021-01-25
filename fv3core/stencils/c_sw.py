from gt4py.gtscript import (
    __INLINED,
    PARALLEL,
    computation,
    horizontal,
    interval,
    region,
)

import fv3core._config as spec
import fv3core.utils.gt4py_utils as utils
from fv3core.decorators import gtstencil
from fv3core.stencils.d2a2c_vect import d2a2c_vect
from fv3core.utils.corners import fill2_4corners_x, fill2_4corners_y


sd = utils.sd


def nonhydro_x_fluxes(delp: sd, pt: sd, w: sd, utc: sd):
    fx1 = delp[-1, 0, 0] if utc > 0.0 else delp
    fx = pt[-1, 0, 0] if utc > 0.0 else pt
    fx2 = w[-1, 0, 0] if utc > 0.0 else w
    fx1 = utc * fx1
    fx = fx1 * fx
    fx2 = fx1 * fx2
    return fx, fx1, fx2


def nonhydro_y_fluxes(delp: sd, pt: sd, w: sd, vtc: sd):
    fy1 = delp[0, -1, 0] if vtc > 0.0 else delp
    fy = pt[0, -1, 0] if vtc > 0.0 else pt
    fy2 = w[0, -1, 0] if vtc > 0.0 else w
    fy1 = vtc * fy1
    fy = fy1 * fy
    fy2 = fy1 * fy2
    return fy, fy1, fy2


def transportdelp(delp: sd, pt: sd, utc: sd, vtc: sd, w: sd, rarea: sd):
    """Transport delp.

    Args:
        delp: What is transported (input)
        pt: Pressure (input)
        utc: x-velocity on C-grid (input)
        vtc: y-velocity on C-grid (input)
        w: z-velocity on C-grid (input)
        rarea: Inverse areas (input) -- IJ field

    Returns:
        delpc: Updated delp (output)
        ptc: Updated pt (output)
        wc: Updated w (output)
    """

    from __externals__ import namelist

    assert __INLINED(namelist.grid_type < 3)
    # additional assumption (not grid.nested)

    delp = fill2_4corners_x(delp, delp, 1, 1, 1, 1)
    pt = fill2_4corners_x(pt, pt, 1, 1, 1, 1)
    w = fill2_4corners_x(w, w, 1, 1, 1, 1)

    fx, fx1, fx2 = nonhydro_x_fluxes(delp, pt, w, utc)

    delp = fill2_4corners_y(delp, delp, 1, 1, 1, 1)
    pt = fill2_4corners_y(pt, pt, 1, 1, 1, 1)
    w = fill2_4corners_y(w, w, 1, 1, 1, 1)

    fy, fy1, fy2 = nonhydro_y_fluxes(delp, pt, w, vtc)

    delpc = delp + (fx1 - fx1[1, 0, 0] + fy1 - fy1[0, 1, 0]) * rarea
    ptc = (pt * delp + (fx - fx[1, 0, 0] + fy - fy[0, 1, 0]) * rarea) / delpc
    wc = (w * delp + (fx2 - fx2[1, 0, 0] + fy2 - fy2[0, 1, 0]) * rarea) / delpc

    return delpc, ptc, wc, delp, pt, w


def divergence_corner(
    u: sd,
    v: sd,
    ua: sd,
    va: sd,
    dxc: sd,
    dyc: sd,
    sin_sg1: sd,
    sin_sg2: sd,
    sin_sg3: sd,
    sin_sg4: sd,
    cos_sg1: sd,
    cos_sg2: sd,
    cos_sg3: sd,
    cos_sg4: sd,
    rarea_c: sd,
):
    """Calculate divg on d-grid.

    Args:
        u: x-velocity (input)
        v: y-velocity (input)
        ua: x-velocity on a (input)
        va: y-velocity on a (input)
        dxc: grid spacing in x-direction (input)
        dyc: grid spacing in y-direction (input)
        sin_sg1: grid sin(sg1) (input)
        sin_sg2: grid sin(sg2) (input)
        sin_sg3: grid sin(sg3) (input)
        sin_sg4: grid sin(sg4) (input)
        cos_sg1: grid cos(sg1) (input)
        cos_sg2: grid cos(sg2) (input)
        cos_sg3: grid cos(sg3) (input)
        cos_sg4: grid cos(sg4) (input)
        rarea_c: inverse cell areas on c-grid (input)

    Returns:
        divg_d: divergence on d-grid (output)
    """
    from __externals__ import i_end, i_start, j_end, j_start, namelist

    if __INLINED(namelist.nord > 0):
        uf = (
            (u - 0.25 * (va[0, -1, 0] + va) * (cos_sg4[0, -1, 0] + cos_sg2))
            * dyc
            * 0.5
            * (sin_sg4[0, -1, 0] + sin_sg2)
        )
        with horizontal(region[:, j_start], region[:, j_end + 1]):
            uf = u * dyc * 0.5 * (sin_sg4[0, -1, 0] + sin_sg2)

        vf = (
            (v - 0.25 * (ua[-1, 0, 0] + ua) * (cos_sg3[-1, 0, 0] + cos_sg1))
            * dxc
            * 0.5
            * (sin_sg3[-1, 0, 0] + sin_sg1)
        )
        with horizontal(region[i_start, :], region[i_end + 1, :]):
            vf = v * dxc * 0.5 * (sin_sg3[-1, 0, 0] + sin_sg1)

        divg_d = vf[0, -1, 0] - vf + uf[-1, 0, 0] - uf
        with horizontal(region[i_start, j_start], region[i_end + 1, j_start]):
            divg_d -= vf[0, -1, 0]
        with horizontal(region[i_end + 1, j_end + 1], region[i_start, j_end + 1]):
            divg_d += vf
        divg_d *= rarea_c

    else:
        divg_d = 0.0

    return divg_d


def circulation_cgrid(uc: sd, vc: sd, dxc: sd, dyc: sd):
    """Update vort_c.

    Args:
        uc: x-velocity on C-grid (input)
        vc: y-velocity on C-grid (input)
        dxc: grid spacing in x-dir (input)
        dyc: grid spacing in y-dir (input)

    Returns:
        vort_c: C-grid vorticity (output)
    """
    from __externals__ import i_end, i_start, j_end, j_start

    fx = dxc * uc
    fy = dyc * vc

    vort_c = fx[0, -1, 0] - fx - fy[-1, 0, 0] + fy

    with horizontal(region[i_start, j_start], region[i_start, j_end + 1]):
        vort_c += fy[-1, 0, 0]

    with horizontal(region[i_end + 1, j_start], region[i_end + 1, j_end + 1]):
        vort_c -= fy[0, 0, 0]

    return vort_c


def update_vorticity_and_kinetic_energy(
    ua: sd,
    va: sd,
    uc: sd,
    vc: sd,
    u: sd,
    v: sd,
    sin_sg1: sd,
    cos_sg1: sd,
    sin_sg2: sd,
    cos_sg2: sd,
    sin_sg3: sd,
    cos_sg3: sd,
    sin_sg4: sd,
    cos_sg4: sd,
    dt2: float,
):
    from __externals__ import i_end, i_start, j_end, j_start, namelist

    assert __INLINED(namelist.grid_type < 3)

    ke = uc if ua > 0.0 else uc[1, 0, 0]
    vort = vc if va > 0.0 else vc[0, 1, 0]

    with horizontal(region[:, j_start - 1], region[:, j_end]):
        vort = vort * sin_sg4 + u[0, 1, 0] * cos_sg4 if va <= 0.0 else vort
    with horizontal(region[:, j_start], region[:, j_end + 1]):
        vort = vort * sin_sg2 + u * cos_sg2 if va > 0.0 else vort

    with horizontal(region[i_end, :], region[i_start - 1, :]):
        ke = ke * sin_sg3 + v[1, 0, 0] * cos_sg3 if ua <= 0.0 else ke
    with horizontal(region[i_end + 1, :], region[i_start, :]):
        ke = ke * sin_sg1 + v * cos_sg1 if ua > 0.0 else ke

    ke = 0.5 * dt2 * (ua * ke + va * vort)

    return ke, vort


def vorticitytransport(
    vort: sd,
    ke: sd,
    u: sd,
    v: sd,
    uc: sd,
    vc: sd,
    cosa_u: sd,
    sina_u: sd,
    cosa_v: sd,
    sina_v: sd,
    rdxc: sd,
    rdyc: sd,
    dt2: float,
):
    from __externals__ import (
        i_end,
        i_start,
        j_end,
        j_start,
        local_ie,
        local_is,
        local_je,
        local_js,
        namelist,
    )

    assert __INLINED(namelist.grid_type < 3)
    # additional assumption: not __INLINED(spec.grid.nested)

    tmp_flux_zonal = dt2 * (v - uc * cosa_u) / sina_u
    with horizontal(region[i_start, :], region[i_end + 1, :]):
        tmp_flux_zonal = dt2 * v

    with horizontal(region[local_is : local_ie + 2, local_js : local_je + 1]):
        flux = vort[0, 0, 0] if tmp_flux_zonal > 0.0 else vort[0, 1, 0]
        uc = uc + tmp_flux_zonal * flux + rdxc * (ke[-1, 0, 0] - ke)

    tmp_flux_merid = dt2 * (u - vc * cosa_v) / sina_v
    with horizontal(region[:, j_start], region[:, j_end + 1]):
        tmp_flux_merid = dt2 * u

    with horizontal(region[local_is : local_ie + 1, local_js : local_je + 2]):
        flux = vort[0, 0, 0] if tmp_flux_merid > 0.0 else vort[1, 0, 0]
        vc = vc - tmp_flux_merid * flux + rdyc * (ke[0, -1, 0] - ke)

    return uc, vc


def vorticitytransport_cgrid(
    uc: sd, vc: sd, vort_c: sd, ke_c: sd, v: sd, u: sd, dt2: float
):
    """Update the C-Grid zonal and meridional velocity fields.

    Args:
        uc: x-velocity on C-grid (input, output)
        vc: y-velocity on C-grid (input, output)
        vort_c: Vorticity on C-grid (input)
        ke_c: kinetic energy on C-grid (input)
        v: y-velocity on D-grid (input)
        u: x-velocity on D-grid (input)
        dt2: timestep (input)
    """

    grid = spec.grid

    def stencil(
        vort: sd,
        ke: sd,
        u: sd,
        v: sd,
        uc: sd,
        vc: sd,
        cosa_u: sd,
        sina_u: sd,
        cosa_v: sd,
        sina_v: sd,
        rdxc: sd,
        rdyc: sd,
        dt2: float,
    ):
        with computation(PARALLEL), interval(...):
            uc, vc = vorticitytransport(
                vort, ke, u, v, uc, vc, cosa_u, sina_u, cosa_v, sina_v, rdxc, rdyc, dt2
            )

    stencil = gtstencil(definition=stencil)
    stencil(
        vort_c,
        ke_c,
        u,
        v,
        uc,
        vc,
        grid.cosa_u,
        grid.sina_u,
        grid.cosa_v,
        grid.sina_v,
        grid.rdxc,
        grid.rdyc,
        dt2,
        origin=grid.compute_origin(),
        domain=grid.domain_shape_compute_buffer_2d(add=(1, 1, 0)),
    )


@gtstencil(externals={"HALO": 3})
def csw_stencil(
    delpc: sd,
    ptc: sd,
    delp: sd,
    pt: sd,
    divgd: sd,
    cosa_s: sd,
    cosa_u: sd,
    cosa_v: sd,
    sina_u: sd,
    sina_v: sd,
    dx: sd,
    dy: sd,
    dxa: sd,
    dya: sd,
    dxc: sd,
    dyc: sd,
    rdxc: sd,
    rdyc: sd,
    rsin2: sd,
    rsin_u: sd,
    rsin_v: sd,
    sin_sg1: sd,
    sin_sg2: sd,
    sin_sg3: sd,
    sin_sg4: sd,
    cos_sg1: sd,
    cos_sg2: sd,
    cos_sg3: sd,
    cos_sg4: sd,
    rarea: sd,
    rarea_c: sd,
    fC: sd,
    u: sd,
    ua: sd,
    uc: sd,
    ut: sd,
    v: sd,
    va: sd,
    vc: sd,
    vt: sd,
    w: sd,
    omga: sd,
    dt2: float,
):
    from __externals__ import local_ie, local_is, local_je, local_js

    with computation(PARALLEL), interval(...):
        uc, vc, ua, va, ut, vt = d2a2c_vect(
            cosa_s,
            cosa_u,
            cosa_v,
            dxa,
            dya,
            rsin2,
            rsin_u,
            rsin_v,
            sin_sg1,
            sin_sg2,
            sin_sg3,
            sin_sg4,
            u,
            ua,
            uc,
            ut,
            v,
            va,
            vc,
            vt,
        )

        divgd_t = divergence_corner(
            u,
            v,
            ua,
            va,
            dxc,
            dyc,
            sin_sg1,
            sin_sg2,
            sin_sg3,
            sin_sg4,
            cos_sg1,
            cos_sg2,
            cos_sg3,
            cos_sg4,
            rarea_c,
        )

        # Extra
        # {
        with horizontal(region[local_is : local_ie + 2, local_js : local_je + 2]):
            divgd = divgd_t
        # }

        ut = dt2 * ut * dy * sin_sg3[-1, 0, 0] if ut > 0 else dt2 * ut * dy * sin_sg1
        vt = dt2 * vt * dx * sin_sg4[0, -1, 0] if vt > 0 else dt2 * vt * dx * sin_sg2

        delpc_t, ptc_t, omga_t, delp, pt, w = transportdelp(delp, pt, ut, vt, w, rarea)

        # Extra
        # {
        with horizontal(
            region[local_is - 1 : local_ie + 2, local_js - 1 : local_je + 2]
        ):
            delpc = delpc_t
            ptc = ptc_t
            omga = omga_t
        # }

        ke, vort = update_vorticity_and_kinetic_energy(
            ua,
            va,
            uc,
            vc,
            u,
            v,
            sin_sg1,
            cos_sg1,
            sin_sg2,
            cos_sg2,
            sin_sg3,
            cos_sg3,
            sin_sg4,
            cos_sg4,
            dt2,
        )

        vort = fC + rarea_c * circulation_cgrid(uc, vc, dxc, dyc)

        uc, vc = vorticitytransport(
            vort, ke, u, v, uc, vc, cosa_u, sina_u, cosa_v, sina_v, rdxc, rdyc, dt2
        )


def compute(delp, pt, u, v, w, uc, vc, ua, va, ut, vt, divgd, omga, dt2):
    grid = spec.grid
    dord4 = True
    origin_halo1 = (grid.is_ - 1, grid.js - 1, 0)
    delpc = utils.make_storage_from_shape(delp.shape, origin=origin_halo1)
    ptc = utils.make_storage_from_shape(pt.shape, origin=origin_halo1)

    csw_stencil(
        delpc,
        ptc,
        delp,
        pt,
        divgd,
        grid.cosa_s,
        grid.cosa_u,
        grid.cosa_v,
        grid.sina_u,
        grid.sina_v,
        grid.dx,
        grid.dy,
        grid.dxa,
        grid.dya,
        grid.dxc,
        grid.dyc,
        grid.rdxc,
        grid.rdyc,
        grid.rsin2,
        grid.rsin_u,
        grid.rsin_v,
        grid.sin_sg1,
        grid.sin_sg2,
        grid.sin_sg3,
        grid.sin_sg4,
        grid.cos_sg1,
        grid.cos_sg2,
        grid.cos_sg3,
        grid.cos_sg4,
        grid.rarea,
        grid.rarea_c,
        grid.fC,
        u,
        ua,
        uc,
        ut,
        v,
        va,
        vc,
        vt,
        w,
        omga,
        dt2,
        origin=grid.compute_origin(add=(-1, -1, 0)),
        domain=grid.domain_shape_compute(add=(3, 3, 0)),
    )

    return delpc, ptc
