import logging

import gt4py.gtscript as gtscript
from gt4py.gtscript import (
    __INLINED,
    PARALLEL,
    computation,
    horizontal,
    interval,
    region,
)

import fv3core._config as spec
import fv3core.stencils.basic_operations as basic
import fv3core.stencils.delnflux as delnflux
import fv3core.stencils.divergence_damping as divdamp
import fv3core.stencils.flux_capacitor as fluxcap
import fv3core.stencils.fvtp2d as fvtp2d
import fv3core.stencils.fxadv as fxadv
import fv3core.stencils.xtp_u as xtp_u
import fv3core.stencils.ytp_v as ytp_v
import fv3core.utils.corners as corners
import fv3core.utils.global_constants as constants
import fv3core.utils.gt4py_utils as utils
from fv3core.decorators import gtstencil
from fv3core.utils.typing import FloatField


dcon_threshold = 1e-5
sd = utils.sd

logger = logging.getLogger("fv3ser")


def grid():
    return spec.grid


def k_bounds():
    return [[0, 1], [1, 1], [2, 1], [3, grid().npz - 3]]


@gtscript.function
def flux_component(gx, gy, rarea):
    return (gx - gx[1, 0, 0] + gy - gy[0, 1, 0]) * rarea


@gtscript.function
def flux_integral(w, delp, gx, gy, rarea):
    return w * delp + flux_component(gx, gy, rarea)


@gtstencil()
def flux_adjust(w: sd, delp: sd, gx: sd, gy: sd, rarea: sd):
    with computation(PARALLEL), interval(...):
        w = flux_integral(w, delp, gx, gy, rarea)


@gtstencil()
def horizontal_relative_vorticity_from_winds(
    u: FloatField,
    v: FloatField,
    ut: FloatField,
    vt: FloatField,
    dx: FloatField,
    dy: FloatField,
    rarea: FloatField,
    vorticity: FloatField,
):
    """
    Compute the area mean relative vorticity in the z-direction from the D-grid winds.

    Args:
        u (in): x-direction wind on D grid
        v (in): y-direction wind on D grid
        ut (out): u * dx
        vt (out): v * dy
        dx (in): gridcell width in x-direction
        dy (in): gridcell width in y-direction
        rarea (in): inverse of area
        vorticity (out): area mean horizontal relative vorticity
    """
    with computation(PARALLEL), interval(...):
        vt = u * dx
        ut = v * dy
        vorticity[0, 0, 0] = rarea * (vt - vt[0, 1, 0] - ut + ut[1, 0, 0])


@gtstencil()
def not_inlineq_pressure(gx: sd, gy: sd, rarea: sd, fx: sd, fy: sd, pt: sd, delp: sd):
    with computation(PARALLEL), interval(...):
        pt = flux_integral(
            pt, delp, gx, gy, rarea
        )  # TODO: Put [0, 0, 0] on left when gt4py bug is fixed
        delp = delp + flux_component(
            fx, fy, rarea
        )  # TODO: Put [0, 0, 0] on left when gt4py bug is fixed
        pt[0, 0, 0] = pt / delp


@gtstencil()
def ke_from_bwind(ke: sd, ub: sd, vb: sd):
    with computation(PARALLEL), interval(...):
        ke[0, 0, 0] = 0.5 * (ke + ub * vb)


@gtstencil()
def ub_from_vort(vort: sd, ub: sd):
    with computation(PARALLEL), interval(...):
        ub[0, 0, 0] = vort - vort[1, 0, 0]


@gtstencil()
def vb_from_vort(vort: sd, vb: sd):
    with computation(PARALLEL), interval(...):
        vb[0, 0, 0] = vort - vort[0, 1, 0]


@gtstencil()
def u_from_ke(ke: sd, vt: sd, fy: sd, u: sd):
    with computation(PARALLEL), interval(...):
        u[0, 0, 0] = vt + ke - ke[1, 0, 0] + fy


@gtstencil()
def v_from_ke(ke: sd, ut: sd, fx: sd, v: sd):
    with computation(PARALLEL), interval(...):
        v[0, 0, 0] = ut + ke - ke[0, 1, 0] - fx


# TODO: This is untested and the radius may be incorrect
@gtstencil(externals={"radius": constants.RADIUS})
def coriolis_force_correction(zh: sd, z_rat: sd):
    from __externals__ import radius

    with computation(PARALLEL), interval(...):
        z_rat[0, 0, 0] = 1.0 + (zh + zh[0, 0, 1]) / radius


@gtstencil()
def zrat_vorticity(wk: sd, f0: sd, z_rat: sd, vort: sd):
    with computation(PARALLEL), interval(...):
        vort[0, 0, 0] = wk + f0 * z_rat


@gtscript.function
def add_dw(w, dw, damp_w):
    w = w + dw if damp_w > 1e-5 else w
    return w


@gtstencil()
def adjust_w_and_qcon(w: sd, delp: sd, dw: sd, q_con: sd, damp_w: float):
    with computation(PARALLEL), interval(...):
        w = w / delp
        w = add_dw(w, dw, damp_w)
        # USE_COND
        q_con = q_con / delp


@gtscript.function
def heat_damping_term(ub, vb, gx, gy, rsin2, cosa_s, u2, v2, du2, dv2):
    return rsin2 * (
        (ub * ub + ub[0, 1, 0] * ub[0, 1, 0] + vb * vb + vb[1, 0, 0] * vb[1, 0, 0])
        + 2.0 * (gy + gy[0, 1, 0] + gx + gx[1, 0, 0])
        - cosa_s * (u2 * dv2 + v2 * du2 + du2 * dv2)
    )


@gtstencil()
def heat_diss(
    fx2: FloatField,
    fy2: FloatField,
    w: FloatField,
    rarea: FloatField,
    heat_source: FloatField,
    diss_est: FloatField,
    dw: FloatField,
    dd8: float,
):
    with computation(PARALLEL), interval(...):
        dw[0, 0, 0] = (fx2 - fx2[1, 0, 0] + fy2 - fy2[0, 1, 0]) * rarea
        heat_source[0, 0, 0] = dd8 - dw * (w + 0.5 * dw)
        diss_est[0, 0, 0] = heat_source


@gtstencil()
def heat_source_from_vorticity_damping(
    ub: FloatField,
    vb: FloatField,
    ut: FloatField,
    vt: FloatField,
    u: FloatField,
    v: FloatField,
    delp: FloatField,
    rsin2: FloatField,
    cosa_s: FloatField,
    rdx: FloatField,
    rdy: FloatField,
    heat_source: FloatField,
    dissipation_estimate: FloatField,
    kinetic_energy_fraction_to_damp: float,
    calculate_dissipation_estimate: int,
):
    """
    Calculates heat source from vorticity damping implied by energy conservation.

    Args:
        ub (in)
        vb (in)
        ut (in)
        vt (in)
        u (in)
        v (in)
        delp (in)
        rsin2 (in)
        cosa_s (in)
        rdx (in): radius of Earth multiplied by x-direction gridcell width
        rdy (in): radius of Earth multiplied by y-direction gridcell width
        heat_source (out): heat source from vorticity damping
            implied by energy conservation
        diss_est (out): dissipation estimate, only calculated if
            calculate_dissipation_estimate is 1
        kinetic_energy_fraction_to_damp (in): according to its comment in fv_arrays,
            the fraction of kinetic energy to explicitly damp and convert into heat.
            TODO: confirm this description is accurate, why is it multiplied
            by 0.25 below?
        calculate_dissipation_estimate (in): If 1, calculate dissipation estimate.
            Equivalent in Fortran model is do_skeb
    """
    with computation(PARALLEL), interval(...):
        ubt = (ub + vt) * rdx
        fy = u * rdx
        gy = fy * ubt
        vbt = (vb - ut) * rdy
        fx = v * rdy
        gx = fx * vbt
        u2 = fy + fy[0, 1, 0]
        du2 = ubt + ubt[0, 1, 0]
        v2 = fx + fx[1, 0, 0]
        dv2 = vbt + vbt[1, 0, 0]
        dampterm = heat_damping_term(ubt, vbt, gx, gy, rsin2, cosa_s, u2, v2, du2, dv2)
        heat_source[0, 0, 0] = delp * (
            heat_source - 0.25 * kinetic_energy_fraction_to_damp * dampterm
        )
        dissipation_estimate[0, 0, 0] = (
            -dampterm if calculate_dissipation_estimate == 1 else dissipation_estimate
        )


def initialize_heat_source(heat_source, diss_est):
    heat_source[grid().compute_interface()] = 0
    diss_est[grid().compute_interface()] = 0


def heat_from_damping(
    ub,
    vb,
    ut,
    vt,
    u,
    v,
    delp,
    fx,
    fy,
    heat_source,
    diss_est,
    kinetic_energy_fraction_to_damp,
    kstart,
    nk,
):
    heat_source_from_vorticity_damping(
        ub,
        vb,
        ut,
        vt,
        u,
        v,
        delp,
        grid().rsin2,
        grid().cosa_s,
        grid().rdx,
        grid().rdy,
        heat_source,
        diss_est,
        kinetic_energy_fraction_to_damp,
        int(spec.namelist.do_skeb),
        origin=(grid().is_, grid().js, kstart),
        domain=(grid().nic, grid().njc, nk),
    )


def set_low_kvals(col):
    for name in ["nord", "nord_w", "d_con"]:
        col[name] = 0
    col["damp_w"] = col["d2_divg"]


def vort_damp_option(col):
    if spec.namelist.do_vort_damp:
        col["nord_v"] = 0
        col["damp_vt"] = 0.5 * col["d2_divg"]


def lowest_kvals(col):
    set_low_kvals(col)
    vort_damp_option(col)


def max_d2_bg0():
    return max(0.01, spec.namelist.d2_bg, spec.namelist.d2_bg_k1)


def max_d2_bg1():
    return max(spec.namelist.d2_bg, spec.namelist.d2_bg_k2)


def get_column_namelist():
    ks = [k[0] for k in k_bounds()]
    col = {}
    for ki in ks:
        col[ki] = column_namelist_options(ki)
    return col


def get_single_column(key):
    col = []
    for k in range(0, grid().npz):
        col.append(column_namelist_options(k)[key])
    col.append(0.0)
    return col


def column_namelist_options(k):
    direct_namelist = ["ke_bg", "d_con", "nord"]
    col = {}
    for name in direct_namelist:
        col[name] = getattr(spec.namelist, name)
    col["d2_divg"] = min(0.2, spec.namelist.d2_bg)
    col["nord_v"] = min(2, col["nord"])
    col["nord_w"] = col["nord_v"]
    col["nord_t"] = col["nord_v"]
    if spec.namelist.do_vort_damp:
        col["damp_vt"] = spec.namelist.vtdm4
    else:
        col["damp_vt"] = 0
    col["damp_w"] = col["damp_vt"]
    col["damp_t"] = col["damp_vt"]
    if grid().npz == 1 or spec.namelist.n_sponge < 0:
        pass
    # commenting because unused, never gets set into col
    #     d2_divg = spec.namelist.d2_bg
    else:
        if k == 0:
            col["d2_divg"] = max_d2_bg0()
            lowest_kvals(col)
        if k == 1 and spec.namelist.d2_bg_k2 > 0.01:
            col["d2_divg"] = max_d2_bg1()
            lowest_kvals(col)
        if k == 2 and spec.namelist.d2_bg_k2 > 0.05:
            col["d2_divg"] = max(spec.namelist.d2_bg, 0.2 * spec.namelist.d2_bg_k2)
            set_low_kvals(col)
    return col


def compute(
    delpc,
    delp,
    ptc,
    pt,
    u,
    v,
    w,
    uc,
    vc,
    ua,
    va,
    divgd,
    mfx,
    mfy,
    cx,
    cy,
    crx,
    cry,
    xfx,
    yfx,
    q_con,
    zh,
    heat_source,
    diss_est,
    dt,
):

    # TODO: Remove paired with removal of #d_sw belos
    # column_namelist = column_namelist_options(0)
    column_namelist = get_column_namelist()
    heat_s = utils.make_storage_from_shape(heat_source.shape, grid().compute_origin())
    diss_e = utils.make_storage_from_shape(heat_source.shape, grid().compute_origin())
    z_rat = utils.make_storage_from_shape(heat_source.shape, grid().full_origin())
    # TODO: If namelist['hydrostatic' and not namelist['use_old_omega'] and last_step.
    if spec.namelist.d_ext > 0:
        raise Exception(
            "untested d_ext > 0. need to call a2b_ord2, not yet implemented"
        )
    if spec.namelist.do_f3d and not spec.namelist.hydrostatic:
        coriolis_force_correction(
            zh,
            z_rat,
            origin=grid().full_origin(),
            domain=grid().domain_shape_full(),
        )

    d_sw(
        delpc,
        delp,
        ptc,
        pt,
        u,
        v,
        w,
        uc,
        vc,
        ua,
        va,
        divgd,
        mfx,
        mfy,
        cx,
        cy,
        crx,
        cry,
        xfx,
        yfx,
        q_con,
        z_rat,
        heat_s,
        diss_e,
        dt,
        column_namelist,
    )

    # TODO: If namelist['hydrostatic' and not namelist['use_old_omega'] and last_step.

    # TODO: If namelist['d_ext'] > 0

    if spec.namelist.d_con > dcon_threshold or spec.namelist.do_skeb:
        basic.add_term_two_vars(
            heat_s,
            heat_source,
            diss_e,
            diss_est,
            origin=grid().compute_origin(),
            domain=grid().domain_shape_compute(),
        )
    nord_v = get_single_column("nord_v")
    damp_vt = get_single_column("damp_vt")
    return nord_v, damp_vt


def damp_vertical_wind(w, heat_s, diss_e, dt, column_namelist, kstart, nk):
    dw = utils.make_storage_from_shape(w.shape, grid().compute_origin())
    wk = utils.make_storage_from_shape(w.shape, grid().full_origin())
    fx2 = utils.make_storage_from_shape(w.shape, grid().full_origin())
    fy2 = utils.make_storage_from_shape(w.shape, grid().full_origin())
    if column_namelist[kstart]["damp_w"] > 1e-5:
        dd8 = column_namelist[kstart]["ke_bg"] * abs(dt)
        damp4 = (column_namelist[kstart]["damp_w"] * grid().da_min_c) ** (
            column_namelist[kstart]["nord_w"] + 1
        )
        delnflux.compute_no_sg(
            w,
            fx2,
            fy2,
            column_namelist[kstart]["nord_w"],
            damp4,
            wk,
            kstart=kstart,
            nk=nk,
        )
        heat_diss(
            fx2,
            fy2,
            w,
            grid().rarea,
            heat_s,
            diss_e,
            dw,
            dd8,
            origin=(grid().is_, grid().js, kstart),
            domain=(grid().nic, grid().njc, nk),
        )
    return dw, wk


@gtstencil()
def ubke(uc: sd, vc: sd, cosa: sd, rsina: sd, ut: sd, ub: sd, dt4: float, dt5: float):
    from __externals__ import i_end, i_start, j_end, j_start

    with computation(PARALLEL), interval(...):
        ub = dt5 * (uc[0, -1, 0] + uc - (vc[-1, 0, 0] + vc) * cosa) * rsina
        if __INLINED(spec.namelist.grid_type < 3):
            with horizontal(region[:, j_start], region[:, j_end + 1]):
                ub = dt4 * (-ut[0, -2, 0] + 3.0 * (ut[0, -1, 0] + ut) - ut[0, 1, 0])
            with horizontal(region[i_start, :], region[i_end + 1, :]):
                ub = dt5 * (ut[0, -1, 0] + ut)


@gtstencil()
def vbke(vc: sd, uc: sd, cosa: sd, rsina: sd, vt: sd, vb: sd, dt4: float, dt5: float):
    from __externals__ import i_end, i_start, j_end, j_start

    with computation(PARALLEL), interval(...):
        vb = dt5 * (vc[-1, 0, 0] + vc - (uc[0, -1, 0] + uc) * cosa) * rsina
        if __INLINED(spec.namelist.grid_type < 3):
            with horizontal(region[i_start, :], region[i_end + 1, :]):
                vb = dt4 * (-vt[-2, 0, 0] + 3.0 * (vt[-1, 0, 0] + vt) - vt[1, 0, 0])
            with horizontal(region[:, j_start], region[:, j_end + 1]):
                vb = dt5 * (vt[-1, 0, 0] + vt)


def d_sw(
    delpc,
    delp,
    ptc,
    pt,
    u,
    v,
    w,
    uc,
    vc,
    ua,
    va,
    divgd,
    xflux,
    yflux,
    cx,
    cy,
    crx,
    cry,
    xfx,
    yfx,
    q_con,
    z_rat,
    heat_s,
    diss_e,
    dt,
    column_namelist,
):

    logger.debug("Parameters that vary with k: {}".format(column_namelist))
    shape = heat_s.shape
    ub = utils.make_storage_from_shape(shape, grid().compute_origin())
    vb = utils.make_storage_from_shape(shape, grid().compute_origin())
    ke = utils.make_storage_from_shape(shape, grid().full_origin())
    vort = utils.make_storage_from_shape(shape, grid().full_origin())
    ut = utils.make_storage_from_shape(shape, grid().full_origin())
    vt = utils.make_storage_from_shape(shape, grid().full_origin())
    fx = utils.make_storage_from_shape(shape, grid().compute_origin())
    fy = utils.make_storage_from_shape(shape, grid().compute_origin())
    gx = utils.make_storage_from_shape(shape, grid().compute_origin())
    gy = utils.make_storage_from_shape(shape, grid().compute_origin())
    ra_x, ra_y = fxadv.compute(uc, vc, ut, vt, xfx, yfx, crx, cry, dt)
    for kstart, nk in k_bounds():
        fvtp2d.compute_no_sg(
            delp,
            crx,
            cry,
            spec.namelist.hord_dp,
            xfx,
            yfx,
            ra_x,
            ra_y,
            fx,
            fy,
            kstart=kstart,
            nk=nk,
            nord=column_namelist[kstart]["nord_v"],
            damp_c=column_namelist[kstart]["damp_vt"],
        )

    fluxcap.compute(cx, cy, xflux, yflux, crx, cry, fx, fy)
    initialize_heat_source(heat_s, diss_e)

    if not spec.namelist.hydrostatic:
        for kstart, nk in k_bounds():
            dw, wk = damp_vertical_wind(
                w, heat_s, diss_e, dt, column_namelist, kstart, nk
            )
            fvtp2d.compute_no_sg(
                w,
                crx,
                cry,
                spec.namelist.hord_vt,
                xfx,
                yfx,
                ra_x,
                ra_y,
                gx,
                gy,
                kstart=kstart,
                nk=nk,
                nord=column_namelist[kstart]["nord_v"],
                damp_c=column_namelist[kstart]["damp_vt"],
                mfx=fx,
                mfy=fy,
            )

        flux_adjust(
            w,
            delp,
            gx,
            gy,
            grid().rarea,
            origin=grid().compute_origin(),
            domain=grid().domain_shape_compute(),
        )
    # USE_COND
    for kstart, nk in k_bounds():
        fvtp2d.compute_no_sg(
            q_con,
            crx,
            cry,
            spec.namelist.hord_dp,
            xfx,
            yfx,
            ra_x,
            ra_y,
            gx,
            gy,
            kstart=kstart,
            nk=nk,
            nord=column_namelist[kstart]["nord_t"],
            damp_c=column_namelist[kstart]["damp_t"],
            mass=delp,
            mfx=fx,
            mfy=fy,
        )

    flux_adjust(
        q_con,
        delp,
        gx,
        gy,
        grid().rarea,
        origin=grid().compute_origin(),
        domain=grid().domain_shape_compute(),
    )

    # END USE_COND
    for kstart, nk in k_bounds():
        fvtp2d.compute_no_sg(
            pt,
            crx,
            cry,
            spec.namelist.hord_tm,
            xfx,
            yfx,
            ra_x,
            ra_y,
            gx,
            gy,
            kstart=kstart,
            nk=nk,
            nord=column_namelist[kstart]["nord_v"],
            damp_c=column_namelist[kstart]["damp_vt"],
            mass=delp,
            mfx=fx,
            mfy=fy,
        )

    if spec.namelist.inline_q:
        raise Exception("inline_q not yet implemented")
    else:
        not_inlineq_pressure(
            gx,
            gy,
            grid().rarea,
            fx,
            fy,
            pt,
            delp,
            origin=grid().compute_origin(),
            domain=grid().domain_shape_compute(),
        )

    dt5 = 0.5 * dt
    dt4 = 0.25 * dt

    vbke(
        vc,
        uc,
        grid().cosa,
        grid().rsina,
        vt,
        vb,
        dt4,
        dt5,
        origin=grid().compute_origin(),
        domain=grid().domain_shape_compute(add=(1, 1, 0)),
    )

    ytp_v.compute(vb, u, v, ub)

    basic.multiply_stencil(
        vb,
        ub,
        ke,
        origin=grid().compute_origin(),
        domain=grid().domain_shape_compute(add=(1, 1, 0)),
    )

    ubke(
        uc,
        vc,
        grid().cosa,
        grid().rsina,
        ut,
        ub,
        dt4,
        dt5,
        origin=grid().compute_origin(),
        domain=grid().domain_shape_compute(add=(1, 1, 0)),
    )

    xtp_u.compute(ub, u, v, vb)

    ke_from_bwind(
        ke,
        ub,
        vb,
        origin=grid().compute_origin(),
        domain=grid().domain_shape_compute(add=(1, 1, 0)),
    )

    if not grid().nested:
        corners.fix_corner_ke(ke, u, v, ut, vt, dt, grid())

    horizontal_relative_vorticity_from_winds(
        u,
        v,
        ut,
        vt,
        spec.grid.dx,
        spec.grid.dy,
        spec.grid.rarea,
        wk,
        origin=(0, 0, 0),
        domain=spec.grid.domain_shape_full(),
    )

    # TODO if spec.namelist.d_f3d and ROT3 unimplemeneted
    for kstart, nk in k_bounds():
        adjust_w_and_qcon(
            w,
            delp,
            dw,
            q_con,
            column_namelist[kstart]["damp_w"],
            origin=(grid().is_, grid().js, kstart),
            domain=(grid().nic, grid().njc, nk),
        )

        divdamp.compute(
            u,
            v,
            va,
            ptc,
            vort,
            ua,
            divgd,
            vc,
            uc,
            delpc,
            ke,
            wk,
            column_namelist[kstart]["d2_divg"],
            dt,
            column_namelist[kstart]["nord"],
            kstart=kstart,
            nk=nk,
        )

        if column_namelist[kstart]["d_con"] > dcon_threshold:
            ub_from_vort(
                vort,
                ub,
                # origin=grid().compute_origin(),
                # domain=grid().domain_shape_compute(add=(0, 1, 0)),
                origin=(grid().is_, grid().js, kstart),
                domain=(grid().nic, grid().njc + 1, nk),
            )
            vb_from_vort(
                vort,
                vb,
                origin=(grid().is_, grid().js, kstart),
                domain=(grid().nic + 1, grid().njc, nk),
                # origin=grid().compute_origin(),
                # domain=grid().domain_shape_compute(add=(1, 0, 0)),
            )

    # Vorticity transport
    if spec.namelist.do_f3d and not spec.namelist.hydrostatic:
        zrat_vorticity(
            wk,
            grid().f0,
            z_rat,
            vort,
            orgin=grid().full_origin(),
            domain=grid().domain_shape_full(),
        )
    else:
        basic.addition_stencil(
            wk,
            grid().f0,
            vort,
            origin=grid().full_origin(),
            domain=grid().domain_shape_full(),
        )

    fvtp2d.compute_no_sg(
        vort, crx, cry, spec.namelist.hord_vt, xfx, yfx, ra_x, ra_y, fx, fy
    )

    u_from_ke(
        ke,
        vt,
        fy,
        u,
        origin=grid().compute_origin(),
        domain=grid().domain_shape_compute(add=(0, 1, 0)),
    )

    v_from_ke(
        ke,
        ut,
        fx,
        v,
        origin=grid().compute_origin(),
        domain=grid().domain_shape_compute(add=(1, 0, 0)),
    )
    for kstart, nk in k_bounds():
        if column_namelist[kstart]["damp_vt"] > dcon_threshold:
            damp4 = (column_namelist[kstart]["damp_vt"] * grid().da_min_c) ** (
                column_namelist[kstart]["nord_v"] + 1
            )
            delnflux.compute_no_sg(
                wk,
                ut,
                vt,
                column_namelist[kstart]["nord_v"],
                damp4,
                vort,
                kstart=kstart,
                nk=nk,
            )

            if (
                column_namelist[kstart]["d_con"] > dcon_threshold
                or spec.namelist.do_skeb
            ):
                heat_from_damping(
                    ub,
                    vb,
                    ut,
                    vt,
                    u,
                    v,
                    delp,
                    fx,
                    fy,
                    heat_s,
                    diss_e,
                    column_namelist[kstart]["d_con"],
                    kstart,
                    nk,
                )
        if column_namelist[kstart]["damp_vt"] > 1e-5:
            basic.add_term_stencil(
                vt,
                u,
                origin=(grid().is_, grid().js, kstart),
                domain=(grid().nic, grid().njc + 1, nk),
                # origin=grid().compute_origin(),
                # domain=grid().domain_shape_compute(add=(0, 1, 0)),
            )
            basic.subtract_term_stencil(
                ut,
                v,
                origin=(grid().is_, grid().js, kstart),
                domain=(grid().nic + 1, grid().njc, nk),
                # origin=grid().compute_origin(),
                # domain=grid().domain_shape_compute(add=(1, 0, 0)),
            )
