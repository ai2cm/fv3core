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
import fv3core.stencils.heatdiss as heatdiss
import fv3core.stencils.xtp_u as xtp_u
import fv3core.stencils.ytp_v as ytp_v
import fv3core.utils.corners as corners
import fv3core.utils.global_constants as constants
import fv3core.utils.gt4py_utils as utils
from fv3core.decorators import gtstencil
from fv3core.utils.typing import FloatField


dcon_threshold = 1e-5

logger = logging.getLogger("fv3ser")


def grid():
    return spec.grid


def k_indices():
    return [[0, 1], [1, 2], [2, 3], list(range(3, grid().npz + 1))]


def d_sw_ksplit(func, data, splitvars_values, outputs, grid, allz=False):
    utils.k_split_run_dataslice(
        func, data, k_indices(), splitvars_values, outputs, grid, allz
    )


@gtscript.function
def flux_component(gx, gy, rarea):
    return (gx - gx[1, 0, 0] + gy - gy[0, 1, 0]) * rarea


@gtscript.function
def flux_integral(w, delp, gx, gy, rarea):
    return w * delp + flux_component(gx, gy, rarea)


@gtstencil()
def flux_adjust(
    w: FloatField, delp: FloatField, gx: FloatField, gy: FloatField, rarea: FloatField
):
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


@gtscript.function
def add_dw(w, dw, damp_w):
    w = w + dw if damp_w > 1e-5 else w
    return w


@gtscript.function
def ke_from_bwind(ke, ub, vb):
    return 0.5 * (ke + ub * vb)


@gtscript.function
def adjust_w_and_qcon(w, delp, dw, damp_w, q_con):
    w = w / delp
    w = add_dw(w, dw, damp_w)
    q_con = q_con / delp

    return w, q_con


@gtstencil()
def ke_horizontal_vorticity_w_qcon_adjust(
    ke: FloatField,
    u: FloatField,
    v: FloatField,
    ub: FloatField,
    vb: FloatField,
    ut: FloatField,
    vt: FloatField,
    dx: FloatField,
    dy: FloatField,
    rarea: FloatField,
    vorticity: FloatField,
    w: FloatField,
    delp: FloatField,
    dw: FloatField,
    q_con: FloatField,
    dt: float,
    nested: bool,
    damp_w: float,
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
    )

    with computation(PARALLEL), interval(...):
        ke = ke_from_bwind(ke, ub, vb)

        with horizontal(region[i_start, j_start]):
            if not nested:
                ke = corners.corner_ke(ke, u, v, ut, vt, dt, 0, 0, -1, 1)
        with horizontal(region[i_end + 1, j_start]):
            if not nested:
                ke = corners.corner_ke(ke, u, v, ut, vt, dt, -1, 0, 0, -1)
        with horizontal(region[i_end + 1, j_end + 1]):
            if not nested:
                ke = corners.corner_ke(ke, u, v, ut, vt, dt, -1, -1, 0, 1)
        with horizontal(region[i_start, j_end + 1]):
            if not nested:
                ke = corners.corner_ke(ke, u, v, ut, vt, dt, 0, -1, -1, -1)

        vt = u * dx
        ut = v * dy
        vorticity[0, 0, 0] = rarea * (vt - vt[0, 1, 0] - ut + ut[1, 0, 0])

        with horizontal(region[local_is : local_ie + 1, local_js : local_je + 1]):
            w, q_con = adjust_w_and_qcon(w, delp, dw, damp_w, q_con)


@gtstencil()
def not_inlineq_pressure(
    gx: FloatField,
    gy: FloatField,
    rarea: FloatField,
    fx: FloatField,
    fy: FloatField,
    pt: FloatField,
    delp: FloatField,
):
    with computation(PARALLEL), interval(...):
        pt = flux_integral(
            pt, delp, gx, gy, rarea
        )  # TODO: Put [0, 0, 0] on left when gt4py bug is fixed
        delp = delp + flux_component(
            fx, fy, rarea
        )  # TODO: Put [0, 0, 0] on left when gt4py bug is fixed
        pt[0, 0, 0] = pt / delp


@gtscript.function
def ub_from_vort(vort, ub):
    return vort - vort[1, 0, 0]


@gtscript.function
def vb_from_vort(vort, vb):
    return vort - vort[0, 1, 0]


@gtstencil()
def ub_and_vb_from_vort(vort: FloatField, ub: FloatField, vb: FloatField):
    with computation(PARALLEL), interval(...):
        ub = ub_from_vort(vort, ub)
        vb = vb_from_vort(vort, vb)


@gtscript.function
def u_from_ke(ke, vt, fy):
    return vt + ke - ke[1, 0, 0] + fy


@gtscript.function
def v_from_ke(ke, ut, fx):
    return ut + ke - ke[0, 1, 0] - fx


@gtstencil()
def u_and_v_from_ke(
    ke: FloatField,
    ut: FloatField,
    vt: FloatField,
    fx: FloatField,
    fy: FloatField,
    u: FloatField,
    v: FloatField,
):
    from __externals__ import local_ie, local_is, local_je, local_js

    with computation(PARALLEL), interval(...):
        with horizontal(region[local_is : local_ie + 1, local_js : local_je + 2]):
            u = u_from_ke(ke, vt, fy)
        with horizontal(region[local_is : local_ie + 2, local_js : local_je + 1]):
            v = v_from_ke(ke, ut, fx)


# TODO: This is untested and the radius may be incorrect
@gtstencil(externals={"radius": constants.RADIUS})
def coriolis_force_correction(zh: FloatField, z_rat: FloatField):
    from __externals__ import radius

    with computation(PARALLEL), interval(...):
        z_rat[0, 0, 0] = 1.0 + (zh + zh[0, 0, 1]) / radius


@gtscript.function
def zrat_vorticity(wk: FloatField, f0: FloatField, z_rat: FloatField):
    return wk + f0 * z_rat


@gtstencil()
def zrat_vort_or_addition(
    wk: FloatField,
    f0: FloatField,
    z_rat: FloatField,
    vort: FloatField,
    do_f3d: bool,
    hydrostatic: bool,
):
    with computation(PARALLEL), interval(...):
        if do_f3d and not hydrostatic:
            vort = zrat_vorticity(wk, f0, z_rat)
        else:
            vort = wk[0, 0, 0] + f0[0, 0, 0]


@gtscript.function
def heat_damping_term(ub, vb, gx, gy, rsin2, cosa_s, u2, v2, du2, dv2):
    return rsin2 * (
        (ub * ub + ub[0, 1, 0] * ub[0, 1, 0] + vb * vb + vb[1, 0, 0] * vb[1, 0, 0])
        + 2.0 * (gy + gy[0, 1, 0] + gx + gx[1, 0, 0])
        - cosa_s * (u2 * dv2 + v2 * du2 + du2 * dv2)
    )


@gtscript.function
def heat_source_from_vorticity_damping_fxn(
    ub,
    vb,
    ut,
    vt,
    u,
    v,
    delp,
    rsin2,
    cosa_s,
    rdx,
    rdy,
    heat_source,
    dissipation_estimate,
    kinetic_energy_fraction_to_damp,
    calculate_dissipation_estimate,
):
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
    heat_source = delp * (
        heat_source - 0.25 * kinetic_energy_fraction_to_damp * dampterm
    )
    dissipation_estimate = (
        -dampterm if calculate_dissipation_estimate == 1 else dissipation_estimate
    )

    return heat_source, dissipation_estimate


@gtstencil()
def heat_from_damping_and_add_sub(
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
    dcon_threshold: float,
    calculate_dissipation_estimate: int,
    do_skeb: bool,
    damp_vt: float,
):
    from __externals__ import local_ie, local_is, local_je, local_js

    with computation(PARALLEL), interval(...):
        if kinetic_energy_fraction_to_damp > dcon_threshold or do_skeb:
            heat_source, dissipation_estimate = heat_source_from_vorticity_damping_fxn(
                ub,
                vb,
                ut,
                vt,
                u,
                v,
                delp,
                rsin2,
                cosa_s,
                rdx,
                rdy,
                heat_source,
                dissipation_estimate,
                kinetic_energy_fraction_to_damp,
                calculate_dissipation_estimate,
            )

        with horizontal(region[local_is : local_ie + 1, local_js : local_je + 2]):
            if damp_vt > 1e-5:
                # basic.add_term_stencil()
                u = u + vt
        with horizontal(region[local_is : local_ie + 2, local_js : local_je + 1]):
            if damp_vt > 1e-5:
                # basic.subtract_term_stencil()
                v = v - ut


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
    ks = [k[0] for k in k_indices()]
    col = {"column_namelist": []}
    for ki in ks:
        col["column_namelist"].append(column_namelist_options(ki))
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
    # TODO: This seems a little redundant, revisit the k column split mechanism
    # and/or the argument passing method
    in_only_vars = ["z_rat", "dt"]
    xflux = mfx
    yflux = mfy
    inout_vars = [
        "delpc",
        "delp",
        "ptc",
        "pt",
        "u",
        "v",
        "w",
        "uc",
        "vc",
        "ua",
        "va",
        "divgd",
        "xflux",
        "yflux",
        "cx",
        "cy",
        "crx",
        "cry",
        "xfx",
        "yfx",
        "q_con",
        "heat_s",
        "diss_e",
    ]
    data = {}
    for varname in inout_vars + in_only_vars:
        data[varname] = locals()[varname]
    outputs = {}
    for iv in inout_vars:
        outputs[iv] = data[iv]
    d_sw_ksplit(d_sw, data, column_namelist, outputs, grid())
    # TODO: Remove when it has been decided how to handle the parameter
    # arguments that change in the vertical. Helpful for debugging.

    # d_sw(delpc, delp, ptc, pt, u, v, w, uc, vc,  ua, va, divgd, mfx, mfy, cx,
    # cy,  crx, cry, xfx, yfx, q_con, z_rat, heat_s, diss_e, dt,column_namelist)

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


def damp_vertical_wind(w, heat_s, diss_e, dt, column_namelist):
    dw = utils.make_storage_from_shape(w.shape, grid().compute_origin())
    wk = utils.make_storage_from_shape(w.shape, grid().full_origin())
    fx2 = utils.make_storage_from_shape(w.shape, grid().full_origin())
    fy2 = utils.make_storage_from_shape(w.shape, grid().full_origin())
    if column_namelist["damp_w"] > 1e-5:
        dd8 = column_namelist["ke_bg"] * abs(dt)
        damp4 = (column_namelist["damp_w"] * grid().da_min_c) ** (
            column_namelist["nord_w"] + 1
        )

        delnflux.compute_no_sg(w, fx2, fy2, column_namelist["nord_w"], damp4, wk)
        heatdiss.compute(fx2, fy2, w, dd8, dw, heat_s, diss_e)
    return dw, wk


@gtstencil()
def ubke(
    uc: FloatField,
    vc: FloatField,
    cosa: FloatField,
    rsina: FloatField,
    ut: FloatField,
    ub: FloatField,
    dt4: float,
    dt5: float,
):
    from __externals__ import i_end, i_start, j_end, j_start

    with computation(PARALLEL), interval(...):
        ub = dt5 * (uc[0, -1, 0] + uc - (vc[-1, 0, 0] + vc) * cosa) * rsina
        if __INLINED(spec.namelist.grid_type < 3):
            with horizontal(region[:, j_start], region[:, j_end + 1]):
                ub = dt4 * (-ut[0, -2, 0] + 3.0 * (ut[0, -1, 0] + ut) - ut[0, 1, 0])
            with horizontal(region[i_start, :], region[i_end + 1, :]):
                ub = dt5 * (ut[0, -1, 0] + ut)


@gtstencil()
def vbke(
    vc: FloatField,
    uc: FloatField,
    cosa: FloatField,
    rsina: FloatField,
    vt: FloatField,
    vb: FloatField,
    dt4: float,
    dt5: float,
):
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
    ra_x = utils.make_storage_from_shape(shape, grid().compute_origin())
    ra_y = utils.make_storage_from_shape(shape, grid().compute_origin())
    fxadv.fxadv_stencil(
        grid().cosa_u,
        grid().cosa_v,
        grid().rsin_u,
        grid().rsin_v,
        grid().sin_sg1,
        grid().sin_sg2,
        grid().sin_sg3,
        grid().sin_sg4,
        grid().rdxa,
        grid().rdya,
        grid().area,
        grid().dy,
        grid().dx,
        uc,
        vc,
        crx,
        cry,
        xfx,
        yfx,
        ut,
        vt,
        ra_x,
        ra_y,
        dt,
        origin=grid().full_origin(),
        domain=grid().domain_shape_full(),
    )
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
        nord=column_namelist["nord_v"],
        damp_c=column_namelist["damp_vt"],
    )

    fluxcap.compute(cx, cy, xflux, yflux, crx, cry, fx, fy)

    if not spec.namelist.hydrostatic:
        dw, wk = damp_vertical_wind(w, heat_s, diss_e, dt, column_namelist)
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
            nord=column_namelist["nord_v"],
            damp_c=column_namelist["damp_vt"],
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
        nord=column_namelist["nord_t"],
        damp_c=column_namelist["damp_t"],
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
        nord=column_namelist["nord_v"],
        damp_c=column_namelist["damp_vt"],
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

    ytp_v.compute(vb, v, ub)

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

    xtp_u.compute(ub, u, vb)

    ke_horizontal_vorticity_w_qcon_adjust(
        ke,
        u,
        v,
        ub,
        vb,
        ut,
        vt,
        spec.grid.dx,
        spec.grid.dy,
        spec.grid.rarea,
        wk,
        w,
        delp,
        dw,
        q_con,
        dt,
        grid().nested,
        column_namelist["damp_w"],
        origin=(0, 0, 0),
        domain=spec.grid.domain_shape_full(),
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
        column_namelist["d2_divg"],
        dt,
        column_namelist["nord"],
    )

    kinetic_energy_fraction_to_damp = column_namelist["d_con"]

    if kinetic_energy_fraction_to_damp > dcon_threshold:
        ub_and_vb_from_vort(
            vort,
            ub,
            vb,
            origin=grid().compute_origin(),
            domain=grid().domain_shape_compute(add=(1, 1, 0)),
        )

    # Vorticity transport
    zrat_vort_or_addition(
        wk,
        grid().f0,
        z_rat,
        vort,
        spec.namelist.do_f3d,
        spec.namelist.hydrostatic,
        origin=grid().full_origin(),
        domain=grid().domain_shape_full(),
    )

    fvtp2d.compute_no_sg(
        vort, crx, cry, spec.namelist.hord_vt, xfx, yfx, ra_x, ra_y, fx, fy
    )

    u_and_v_from_ke(
        ke,
        ut,
        vt,
        fx,
        fy,
        u,
        v,
        origin=grid().compute_origin(),
        domain=grid().domain_shape_compute(add=(1, 1, 0)),
    )

    if column_namelist["damp_vt"] > dcon_threshold:
        damp4 = (column_namelist["damp_vt"] * grid().da_min_c) ** (
            column_namelist["nord_v"] + 1
        )
        delnflux.compute_no_sg(wk, ut, vt, column_namelist["nord_v"], damp4, vort)

    heat_from_damping_and_add_sub(
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
        heat_s,
        diss_e,
        float(
            kinetic_energy_fraction_to_damp
        ),  # GT4Py seems to only see this as an 'int64' regardless of
        # the type declaration used in the stencil definition.
        # Casting it as a float fixes the problem
        dcon_threshold,
        int(spec.namelist.do_skeb),
        spec.namelist.do_skeb,
        column_namelist["damp_vt"],
        origin=grid().compute_origin(),
        domain=grid().domain_shape_compute(add=(1, 1, 0)),
    )
