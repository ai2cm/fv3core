#!/usr/bin/env python3
import fv3.utils.gt4py_utils as utils
import gt4py.gtscript as gtscript
import fv3._config as spec
from gt4py.gtscript import computation, interval, PARALLEL
from fv3.utils.global_constants import (
    CP_AIR,
    CV_AIR,
    CP_VAP,
    CV_VAP,
    C_LIQ,
    C_ICE,
    GRAV,
    ZVIR,
    RDGAS,
)
from fv3.stencils.basic_operations import dim, multiply_constant_inout
import fv3.stencils.copy_stencil as cp
from ..decorators import ArgSpec, state_inputs
import numpy as np

sd = utils.sd
RK = CP_AIR / RDGAS + 1.0
G2 = 0.5 * GRAV
T1_MIN = 160.0
T2_MIN = 165.0
T2_MAX = 315.0
T3_MAX = 325.0
USTAR2 = 1.0e-4
RI_MAX = 1.0
RI_MIN = 0.25


@gtscript.function
def standard_cm(cpm, cvm, q0_vapor, q0_liquid, q0_rain, q0_ice, q0_snow, q0_graupel):
    q_liq = q0_liquid + q0_rain
    q_sol = q0_ice + q0_snow + q0_graupel
    cpm = (
        (1.0 - (q0_vapor + q_liq + q_sol)) * CP_AIR
        + q0_vapor * CP_VAP
        + q_liq * C_LIQ
        + q_sol * C_ICE
    )
    cvm = (
        (1.0 - (q0_vapor + q_liq + q_sol)) * CV_AIR
        + q0_vapor * CV_VAP
        + q_liq * C_LIQ
        + q_sol * C_ICE
    )
    return cpm, cvm


@gtscript.function
def tvol(gz, u0, v0, w0):
    return gz + 0.5 * (u0 ** 2 + v0 ** 2 + w0 ** 2)


@utils.stencil()
def init(
    den: sd,
    gz: sd,
    gzh: sd,
    t0: sd,
    pm: sd,
    u0: sd,
    v0: sd,
    w0: sd,
    hd: sd,
    cvm: sd,
    cpm: sd,
    te: sd,
    ua: sd,
    va: sd,
    w: sd,
    ta: sd,
    peln: sd,
    delp: sd,
    delz: sd,
    q0_vapor: sd,
    q0_liquid: sd,
    q0_rain: sd,
    q0_ice: sd,
    q0_snow: sd,
    q0_graupel: sd,
    xvir: float,
):
    with computation(PARALLEL), interval(...):
        t0 = ta
        # tvm = t0 * (1. + xvir*q0_vapor)  # this only gets used in hydrostatic mode
        u0 = ua
        v0 = va
        pm = delp / (peln[0, 0, 1] - peln)
    with computation(BACKWARD), interval(...):
        # note only for nwat = 6
        cpm, cvm = standard_cm(
            cpm, cvm, q0_vapor, q0_liquid, q0_rain, q0_ice, q0_snow, q0_graupel
        )
        den = -delp / (GRAV * delz)
        w0 = w
        gz = gzh[0, 0, 1] - G2 * delz
        tmp = tvol(gz, u0, v0, w0)
        hd = cpm * t0 + tmp
        te = cvm * t0 + tmp
        gzh = gzh[0, 0, 1] - GRAV * delz


#
@gtscript.function
def qcon_func(qcon, q0_liquid, q0_ice, q0_snow, q0_rain, q0_graupel):
    return q0_liquid + q0_ice + q0_snow + q0_rain + q0_graupel


@utils.stencil()
def compute_qcon(
    qcon: sd, q0_liquid: sd, q0_ice: sd, q0_snow: sd, q0_rain: sd, q0_graupel: sd
):
    with computation(PARALLEL), interval(...):
        qcon = qcon_func(qcon, q0_liquid, q0_ice, q0_snow, q0_rain, q0_graupel)


@utils.stencil()
def recompute_qcon(
    ri: sd,
    ri_ref: sd,
    qcon: sd,
    q0_liquid: sd,
    q0_rain: sd,
    q0_ice: sd,
    q0_snow: sd,
    q0_graupel: sd,
):
    with computation(BACKWARD), interval(...):
        if ri[0, 0, 1] < ri_ref[0, 0, 1]:
            qcon = qcon_func(qcon, q0_liquid, q0_ice, q0_snow, q0_rain, q0_graupel)


@utils.stencil()
def m_loop(
    ri: sd,
    ri_ref: sd,
    pm: sd,
    u0: sd,
    v0: sd,
    w0: sd,
    t0: sd,
    hd: sd,
    gz: sd,
    qcon: sd,
    delp: sd,
    pkz: sd,
    q0_vapor: sd,
    pt1: sd,
    pt2: sd,
    tv2: sd,
    t_min: float,
    t_max: float,
    ratio: float,
    xvir: float,
):
    with computation(BACKWARD), interval(
        ...
    ):  # interval(1, None): -- from doing the full stencil
        tv1 = t0[0, 0, -1] * (1.0 + xvir * q0_vapor[0, 0, -1] - qcon[0, 0, -1])
        tv2 = t0 * (1.0 + xvir * q0_vapor - qcon)
        pt1 = tv1 / pkz[0, 0, -1]
        pt2 = tv2 / pkz
        ri = (
            (gz[0, 0, -1] - gz)
            * (pt1 - pt2)
            / (
                0.5
                * (pt1 + pt2)
                * ((u0[0, 0, -1] - u0) ** 2 + (v0[0, 0, -1] - v0) ** 2 + USTAR2)
            )
        )
        if tv1 > t_max and tv1 > tv2:
            ri = 0
        elif tv2 < t_min:
            ri = ri if ri < 0.1 else 0.1
        # Adjustment for K-H instability:
        # Compute equivalent mass flux: mc
        # Add moist 2-dz instability consideration:
        ri_ref = RI_MIN + (RI_MAX - RI_MIN) * dim(400.0e2, pm) / 200.0e2
        if RI_MAX < ri_ref:
            ri_ref = RI_MAX


"""
    with computation(BACKWARD):
        with interval(1, 2):
            ri_ref = 4. * ri_ref
        with interval(2, 3):
            ri_ref = 2. * ri_ref
        #with interval(3, 4):  # This crashes gt4py, invalid interval, offset_limit default is 2 in make_axis_interval function
        #    ri_ref = 1.5 * ri_ref
    with computation(BACKWARD),interval(1, None):
        max_ri_ratio = ri / ri_ref
        if max_ri_ratio < 0.:
            max_ri_ratio = 0.
        if ri < ri_ref:
            mc = ratio * delp[0, 0, -1] * delp / (delp[0, 0, -1] + delp) * (1. - max_ri_ratio)**2.

@utils.stencil()
def m_loop_hack_interval_3_4(ri: sd, ri_ref: sd, mc: sd, delp: sd, ratio: float):
    with computation(BACKWARD), interval(2, 3):
        ri_ref = 1.5 * ri_ref
        max_ri_ratio = ri / ri_ref
        if max_ri_ratio < 0.:
            max_ri_ratio = 0.
        if ri < ri_ref:
            mc = ratio * delp[0, 0, -1] * delp / (delp[0, 0, -1] + delp) * (1. - max_ri_ratio)**2
"""


@utils.stencil()
def equivalent_mass_flux(ri: sd, ri_ref: sd, mc: sd, delp: sd, ratio: float):
    with computation(PARALLEL), interval(...):
        max_ri_ratio = ri / ri_ref
        if max_ri_ratio < 0.0:
            max_ri_ratio = 0.0
        if ri < ri_ref:
            mc = (
                ratio
                * delp[0, 0, -1]
                * delp
                / (delp[0, 0, -1] + delp)
                * (1.0 - max_ri_ratio) ** 2
            )


# 3d version, doesn't work due to this k-1 value needing to be updated before calculating variables in the k - 1 case
"""
@utils.stencil()
def KH_instability_adjustment(ri: sd, ri_ref: sd, mc: sd, q0: sd, delp: sd):
    with computation(BACKWARD):
        with interval(-1, None):
            h0 = 0.
            if ri < ri_ref:
                h0 = mc * (q0 - q0[0, 0, -1])
                q0 = q0 - h0 / delp
        with interval(1, -1):
            h0 = 0.
            if ri[0, 0, 1] < ri_ref[0, 0, 1]:
                q0 = q0 + h0[0, 0, 1] / delp
            if ri < ri_ref:
                h0 = mc * (q0 - q0[0, 0, -1])
                q0 = q0 - h0 / delp
        with interval(0, 1):
            if ri[0, 0, 1] < ri_ref[0, 0, 1]:
                q0 = q0 + h0[0, 0, 1] / delp
"""


@utils.stencil()
def KH_instability_adjustment_bottom(
    ri: sd, ri_ref: sd, mc: sd, q0: sd, delp: sd, h0: sd
):
    with computation(BACKWARD), interval(...):
        if ri < ri_ref:
            h0 = mc * (q0 - q0[0, 0, -1])
            q0 = q0 - h0 / delp


@utils.stencil()
def KH_instability_adjustment_top(ri: sd, ri_ref: sd, mc: sd, q0: sd, delp: sd, h0: sd):
    with computation(BACKWARD), interval(...):
        if ri[0, 0, 1] < ri_ref[0, 0, 1]:
            q0 = q0 + h0[0, 0, 1] / delp


def KH_instability_adjustment(ri, ri_ref, mc, q0, delp, h0, origin=None, domain=None):
    KH_instability_adjustment_bottom(
        ri, ri_ref, mc, q0, delp, h0, origin=origin, domain=domain
    )
    KH_instability_adjustment_top(
        ri,
        ri_ref,
        mc,
        q0,
        delp,
        h0,
        origin=(origin[0], origin[1], origin[2] - 1),
        domain=domain,
    )


# special case for total energy that may be a fortran bug
def KH_instability_adjustment_te(
    ri, ri_ref, mc, q0, delp, h0, hd, origin=None, domain=None
):
    KH_instability_adjustment_bottom_te(
        ri, ri_ref, mc, q0, delp, h0, hd, origin=origin, domain=domain
    )
    KH_instability_adjustment_top(
        ri,
        ri_ref,
        mc,
        q0,
        delp,
        h0,
        origin=(origin[0], origin[1], origin[2] - 1),
        domain=domain,
    )


@utils.stencil()
def KH_instability_adjustment_bottom_te(
    ri: sd, ri_ref: sd, mc: sd, q0: sd, delp: sd, h0: sd, hd: sd
):
    with computation(BACKWARD), interval(...):
        if ri < ri_ref:
            h0 = mc * (hd - hd[0, 0, -1])
            q0 = q0 - h0 / delp


@utils.stencil()
def double_adjust_cvm(
    cvm: sd,
    cpm: sd,
    gz: sd,
    u0: sd,
    v0: sd,
    w0: sd,
    hd: sd,
    t0: sd,
    te: sd,
    q0_liquid: sd,
    q0_vapor: sd,
    q0_ice: sd,
    q0_snow: sd,
    q0_rain: sd,
    q0_graupel: sd,
):
    with computation(BACKWARD), interval(...):
        cpm, cvm = standard_cm(
            cpm, cvm, q0_vapor, q0_liquid, q0_rain, q0_ice, q0_snow, q0_graupel
        )
        tv = tvol(gz, u0, v0, w0)
        t0 = (te - tv) / cvm
        hd = cpm * t0 + tv


@gtscript.function
def readjust_by_frac(a0, a, fra):
    return a + (a0 - a) * fra


@utils.stencil()
def fraction_adjust(
    t0: sd,
    ta: sd,
    u0: sd,
    ua: sd,
    v0: sd,
    va: sd,
    w0: sd,
    w: sd,
    fra: float,
    hydrostatic: bool,
):
    with computation(PARALLEL), interval(...):
        t0 = readjust_by_frac(t0, ta, fra)
        u0 = readjust_by_frac(u0, ua, fra)
        v0 = readjust_by_frac(v0, va, fra)
        if not hydrostatic:
            w0 = readjust_by_frac(w0, w, fra)


@utils.stencil()
def fraction_adjust_tracer(q0: sd, q: sd, fra: float):
    with computation(PARALLEL), interval(...):
        q0 = readjust_by_frac(q0, q, fra)


@utils.stencil()
def finalize(
    u0: sd,
    v0: sd,
    w0: sd,
    t0: sd,
    ua: sd,
    va: sd,
    ta: sd,
    w: sd,
    u_dt: sd,
    v_dt: sd,
    rdt: float,
):
    with computation(PARALLEL), interval(...):
        u_dt = rdt * (u0 - ua)
        v_dt = rdt * (v0 - va)
        ta = t0
        ua = u0
        va = v0
        w = w0


# TODO replace with something from fv3config probably, using the field_table. When finalize reperesentation of tracers, adjust this
def tracers_dict(state):
    tracers = {}
    for tracername in utils.tracer_variables:
        tracers[tracername] = state.__dict__[tracername]
    state.tracers = tracers


@state_inputs(
    ArgSpec("delp", "pressure_thickness_of_atmospheric_layer", "Pa", intent="inout"),
    ArgSpec("delz", "vertical_thickness_of_atmospheric_layer", "m", intent="inout"),
    ArgSpec("pe", "interface_pressure", "Pa", intent="inout"),
    ArgSpec(
        "pkz",
        "finite_volume_mean_pressure_raised_to_power_of_kappa",
        "unknown",
        intent="inout",
    ),
    ArgSpec("peln", "logarithm_of_interface_pressure", "ln(Pa)", intent="inout"),
    ArgSpec("pt", "air_temperature", "degK", intent="inout"),
    ArgSpec("ua", "x_wind_on_a_grid", "m/s", intent="inout"),
    ArgSpec("va", "y_wind_on_a_grid", "m/s", intent="inout"),
    ArgSpec("w", "vertical_wind", "m/s", intent="inout"),
    ArgSpec("qvapor", "specific_humidity", "kg/kg", intent="inout"),
    ArgSpec("qliquid", "cloud_water_mixing_ratio", "kg/kg", intent="inout"),
    ArgSpec("qrain", "rain_mixing_ratio", "kg/kg", intent="inout"),
    ArgSpec("qsnow", "snow_mixing_ratio", "kg/kg", intent="inout"),
    ArgSpec("qice", "ice_mixing_ratio", "kg/kg", intent="inout"),
    ArgSpec("qgraupel", "graupel_mixing_ratio", "kg/kg", intent="inout"),
    ArgSpec("qo3mr", "ozone_mixing_ratio", "kg/kg", intent="inout"),
    ArgSpec("qsgs_tke", "turbulent_kinetic_energy", "m**2/s**2", intent="inout"),
    ArgSpec("qcld", "cloud_fraction", "", intent="inout"),
    ArgSpec("u_dt", "x_wind_tendency", "m/s**2", intent="inout"),
    ArgSpec("v_dt", "y_wind_tendency", "m/s**2", intent="inout"),
    ArgSpec("te_1", "te_1", "", intent="in"),
    ArgSpec("gzh_1", "gzh_1", "", intent="in"),
    ArgSpec("gz_1", "gz_1", "", intent="in"),
    ArgSpec("cvm_1", "cvm_1", "", intent="in"),
    ArgSpec("u0_2", "u0_2", "", intent="in"),
)
def compute(state, nq, dt):
    tracers_dict(state)  # TODO get rid of this when finalize representation of tracers

    grid = spec.grid
    rdt = 1.0 / dt
    k_bot = spec.namelist["n_sponge"]
    if k_bot is not None:
        if k_bot < 3:
            return
    else:
        k_bot = grid.npz
    if k_bot < min(grid.npz, 24):
        t_max = T2_MAX
    else:
        t_max = T3_MAX
    if state.pe[grid.is_, grid.js, 0] < 2.0:
        t_min = T1_MIN
    else:
        t_min = T2_MIN

    if spec.namelist["nwat"] == 0:
        xvir = 0.0
        # rz = 0 # hydrostatic only
    else:
        xvir = ZVIR
        # rz = constants.RV_GAS - constants.RDGAS # hydrostatic only
    m = 3
    fra = dt / float(spec.namelist["fv_sg_adj"])
    if spec.namelist["hydrostatic"]:
        raise Exception("Hydrostatic not supported for fv_subgridz")
    q0 = {}
    for tracername in utils.tracer_variables:
        q0[tracername] = cp.copy(state.__dict__[tracername], (0, 0, 0))
    origin = grid.compute_origin()
    shape = state.delp.shape
    u0 = utils.make_storage_from_shape(shape, origin)
    v0 = utils.make_storage_from_shape(shape, origin)
    w0 = utils.make_storage_from_shape(shape, origin)
    gzh = utils.make_storage_from_shape(shape, origin)
    gz = utils.make_storage_from_shape(shape, origin)
    t0 = utils.make_storage_from_shape(shape, origin)
    pm = utils.make_storage_from_shape(shape, origin)
    hd = utils.make_storage_from_shape(shape, origin)
    te = utils.make_storage_from_shape(shape, origin)
    den = utils.make_storage_from_shape(shape, origin)
    qcon = utils.make_storage_from_shape(shape, origin)
    cvm = utils.make_storage_from_shape(shape, origin)
    cpm = utils.make_storage_from_shape(shape, origin)

    kbot_domain = (grid.nic, grid.njc, k_bot)
    origin = grid.compute_origin()
    init(
        den,
        gz,
        gzh,
        t0,
        pm,
        u0,
        v0,
        w0,
        hd,
        cvm,
        cpm,
        te,
        state.ua,
        state.va,
        state.w,
        state.pt,
        state.peln,
        state.delp,
        state.delz,
        q0["qvapor"],
        q0["qliquid"],
        q0["qrain"],
        q0["qice"],
        q0["qsnow"],
        q0["qgraupel"],
        xvir,
        origin=origin,
        domain=kbot_domain,
    )
    """
    print(np.where(state.gzh_1[:, 3, 0] != gzh[:, 3, 0]))
    print(np.any(gzh == (state.gzh_1[3, 3, 0])))
    print(np.where(gzh == (state.gzh_1[3, 3, 0])))
    print(gzh[3, 3, 0] == state.gzh_1[3, 3, 0])
    print(gzh[:, 3,0])
    print('ref',state.gzh_1[:, 3, 0])
    for i in range(grid.is_, grid.ie + 1):
        ref = state.gzh_1[i, 3, 0]
        comp = gzh[i, 3, 0]
        if ref != comp:
            print(i, comp, ref, ref - comp)
    
    print('CVM COMPARE')
    print(np.where(state.cvm_1[:, 3, 0] != cvm[:, 3, 0]))
    print(np.any(cvm == (state.cvm_1[3, 3, 0])))
    print(np.where(cvm == (state.cvm_1[3, 3, 0])))
    print(cvm[3, 3, 0] == state.cvm_1[3, 3, 0])
    print(cvm[:, 3,0])
    print('ref',state.cvm_1[:, 3, 0])
    for i in range(grid.is_, grid.ie + 1):
        ref = state.cvm_1[i, 3, 0]
        comp = cvm[i, 3, 0]
        if ref != comp:
            print(i, comp, ref, ref - comp)
    print('COMPARE omg te')
    print(np.where(state.te_1[:, 0, 0] != te[:, 3, 0]))
    print(np.any(te == (state.te_1[3, 0, 0])), state.te_1.shape)
    print(np.where(te == (state.te_1[3, 0, 0])))
    print(te[3, 3, 0] == state.te_1[3, 0, 0])
    print('computed', te[:, 3,0])
    print('ref',state.te_1[:, 0, 0])
    for i in range(grid.is_, grid.ie + 1):
        ref = state.te_1[i, 0, 0]
        comp = te[i, 3, 0]
        if ref != comp:
            print(i, comp, ref, ref - comp)
    print('COMPARE omg GZ')
    print(np.where(state.gz_1[:, 0, 0] != gz[:, 3, 0]))
    print(np.any(gz == (state.gz_1[3, 0, 0])), state.gz_1.shape)
    print(np.where(gz == (state.gz_1[3, 0, 0])))
    print(gz[3, 3, 0] == state.gz_1[3, 0, 0])
    print('computed', gz[:, 3,0])
    print('ref',state.gz_1[:, 0, 0])
    
    for i in range(grid.is_, grid.ie + 1):
        ref = state.gz_1[i, 0, 0]
        comp = gz[i, 3, 0]
        if ref != comp:
            print(i, comp, ref, ref - comp)
    """
    ri = utils.make_storage_from_shape(shape, origin)
    ri_ref = utils.make_storage_from_shape(shape, origin)
    mc = utils.make_storage_from_shape(shape, origin)
    h0 = utils.make_storage_from_shape(shape, origin)
    pt1 = utils.make_storage_from_shape(shape, origin)
    pt2 = utils.make_storage_from_shape(shape, origin)
    tv2 = utils.make_storage_from_shape(shape, origin)
    ratios = {0: 0.25, 1: 0.5, 2: 0.999}
    j2d = 26
    k2d = 27
    i2d = 5

    for n in range(m):
        ratio = ratios[n]
        compute_qcon(
            qcon,
            q0["qliquid"],
            q0["qrain"],
            q0["qice"],
            q0["qsnow"],
            q0["qgraupel"],
            origin=origin,
            domain=kbot_domain,
        )
        for k in range(k_bot - 1, 0, -1):
            korigin = (grid.is_, grid.js, k)
            korigin_m1 = (grid.is_, grid.js, k - 1)
            kdomain = (grid.nic, grid.njc, 1)
            kdomain_m1 = (grid.nic, grid.njc, 2)
            if k == k2d:
                print("input t0", qcon[i2d, j2d, k2d])
            m_loop(
                ri,
                ri_ref,
                pm,
                u0,
                v0,
                w0,
                t0,
                hd,
                gz,
                qcon,
                state.delp,
                state.pkz,
                q0["qvapor"],
                pt1,
                pt2,
                tv2,
                t_min,
                t_max,
                ratio,
                xvir,
                origin=korigin,
                domain=kdomain,
            )

            if k == 1:
                multiply_constant_inout(ri_ref, 4.0, origin=korigin, domain=kdomain)
            if k == 2:
                multiply_constant_inout(ri_ref, 2.0, origin=korigin, domain=kdomain)
            if k == 3:
                multiply_constant_inout(ri_ref, 1.5, origin=korigin, domain=kdomain)

            # work around that gt4py will not accept interval(3, 4), no longer used, mc calc per k
            # m_loop_hack_interval_3_4(ri, ri_ref, mc, state.delp, ratio, origin=(grid.is_, grid.js, 1), domain=(grid.nic, grid.njc, k_bot - 1))
            equivalent_mass_flux(
                ri, ri_ref, mc, state.delp, ratio, origin=korigin, domain=kdomain
            )
            for tracername in utils.tracer_variables:
                KH_instability_adjustment(
                    ri,
                    ri_ref,
                    mc,
                    q0[tracername],
                    state.delp,
                    h0,
                    origin=korigin,
                    domain=kdomain,
                )
            if k == k2d + 1:
                print(
                    "input tracers",
                    qcon[i2d, j2d, k2d],
                    q0["qliquid"][i2d, j2d, k2d],
                    q0["qrain"][i2d, j2d, k2d],
                    q0["qice"][i2d, j2d, k2d],
                    q0["qsnow"][i2d, j2d, k2d],
                    q0["qgraupel"][i2d, j2d, k2d],
                )

            recompute_qcon(
                ri,
                ri_ref,
                qcon,
                q0["qliquid"],
                q0["qrain"],
                q0["qice"],
                q0["qsnow"],
                q0["qgraupel"],
                origin=korigin_m1,
                domain=kdomain,
            )
            if k == k2d + 1:
                print(
                    "after input tracers",
                    qcon[i2d, j2d, k2d],
                    q0["qliquid"][i2d, j2d, k2d],
                    q0["qrain"][i2d, j2d, k2d],
                    q0["qice"][i2d, j2d, k2d],
                    q0["qsnow"][i2d, j2d, k2d],
                    q0["qgraupel"][i2d, j2d, k2d],
                )

            # input tracers 7.4409851198972536e-09 0.0 0.0 6.980017722948923e-09 0.0 0.0
            # after input tracers 7.4409851198972536e-09 0.0 0.0 6.980017722948923e-09 0.0 0.0
            KH_instability_adjustment(
                ri, ri_ref, mc, u0, state.delp, h0, origin=korigin, domain=kdomain
            )

            if n == 0 and k == k2d:

                print("COMPARE u0")
                print(np.where(state.u0_2[:, 0, k2d] != u0[:, j2d, k2d]))
                print(np.any(u0 == (state.u0_2[i2d, 0, k2d])), state.u0_2.shape)
                print(np.where(u0 == (state.u0_2[i2d, 0, k2d])))
                print(u0[3, j2d, k2d] == state.u0_2[i2d, 0, k2d])
                print("computed", u0[:, j2d, k2d])
                print("ref", state.u0_2[:, 0, k2d])
                for ki in range(k, k_bot):
                    ref = state.u0_2[i2d, 0, ki]
                    comp = u0[i2d, j2d, ki]
                    if ref != comp:
                        print(
                            "BROKE",
                            ki,
                            ri[i2d, j2d, ki] < ri_ref[i2d, j2d, ki],
                            ri[i2d, j2d, ki],
                            ri_ref[i2d, j2d, ki],
                            "h0",
                            h0[i2d, j2d, ki],
                            state.delp[i2d, j2d, ki],
                            "mc",
                            mc[i2d, j2d, ki],
                            comp,
                            ref,
                            ref - comp,
                        )
                        print(
                            pt1[i2d, j2d, ki],
                            pt2[i2d, j2d, ki],
                            tv2[i2d, j2d, ki],
                            t0[i2d, j2d, ki],
                        )
                    else:
                        print(
                            ki,
                            ri[i2d, j2d, ki] < ri_ref[i2d, j2d, ki],
                            ri[i2d, j2d, ki],
                            ri_ref[i2d, j2d, ki],
                            "h0",
                            h0[i2d, j2d, ki],
                            state.delp[i2d, j2d, ki],
                            "mc",
                            mc[i2d, j2d, ki],
                            ref,
                        )
                """
                for i in range(grid.is_, grid.ie + 1):
                ref = state.u0_2[i, 0, k2d]
                comp = u0[i, j2d, k2d]
                if ref != comp:
                print(i, ri[i, j2d, k2d]< ri_ref[i, j2d, k2d], ri[i, j2d,  k2d], ri_ref[i, j2d, k2d], comp, ref, ref- comp)
                else:
                print(i, ri[i, j2d, k2d]< ri_ref[i, j2d, k2d], ri[i, j2d,  k2d], ri_ref[i, j2d, k2d], ref)
                """
                # BROKE 23 True -1.7486147469239424 1.0 h0 -230.22997640362954 1272.893593285895 mc 155.7299509952383 13.178566660886279 12.99769531577305 -0.18087134511322844
                # 13.059474639156168 13.078025261274892 211.20182034576501
                # BAD RIS   u0=12.702255173037734  ri=0.88078096720304144  gz1=73925.800432355230  gz=69658.794003553558  u01=9.8646853339093035  u0=12.702255173037734  v01=1.3318337031843186 v0=5.8722371633355577
                # t01=221.41877166853371 t0=224.34975868858552  qcon1=1.8288245669175295E-008   qcon=6.9800177229489227E-009 pkz1=17.170058033070525    pkz=17.500624285911066
                # pt1=12.895976358714460  pt2=12.819891091362800   tv1=221.42466247223280    tv2=224.35609737623872
                # pt1=12.89597635871446   pt2=12.801212283662595                             tv2=224.35609727282076
                # t0=224.34975868858552
                #    224.34975868858552
                #  t0 * (1. + xvir * q0_vapor - qcon)
                #  qcon=6.9800177229489227E-009
                #       7.4409851198972536e-09

            KH_instability_adjustment(
                ri, ri_ref, mc, v0, state.delp, h0, origin=korigin, domain=kdomain
            )
            KH_instability_adjustment(
                ri, ri_ref, mc, w0, state.delp, h0, origin=korigin, domain=kdomain
            )
            KH_instability_adjustment_te(
                ri, ri_ref, mc, te, state.delp, h0, hd, origin=korigin, domain=kdomain
            )
            if k == k2d + 1:
                print("double trouble", t0[i2d, j2d, k2d + 1], t0[i2d, j2d, k2d])
            double_adjust_cvm(
                cvm,
                cpm,
                gz,
                u0,
                v0,
                w0,
                hd,
                t0,
                te,
                q0["qliquid"],
                q0["qvapor"],
                q0["qice"],
                q0["qsnow"],
                q0["qrain"],
                q0["qgraupel"],
                origin=korigin_m1,
                domain=kdomain_m1,
            )
            if k == k2d + 1:
                print("after double trouble", t0[i2d, j2d, k2d + 1], t0[i2d, j2d, k2d])
    if fra < 1.0:
        fraction_adjust(
            t0,
            state.pt,
            u0,
            state.ua,
            v0,
            state.va,
            w0,
            state.w,
            fra,
            spec.namelist["hydrostatic"],
            origin=origin,
            domain=kbot_domain,
        )
        for tracername in utils.tracer_variables:
            fraction_adjust_tracer(
                q0[tracername],
                state.tracers[tracername],
                fra,
                origin=origin,
                domain=kbot_domain,
            )
    for tracername in utils.tracer_variables:
        cp.copy_stencil(
            q0[tracername], state.tracers[tracername], origin=origin, domain=kbot_domain
        )
    finalize(
        u0,
        v0,
        w0,
        t0,
        state.ua,
        state.va,
        state.pt,
        state.w,
        state.u_dt,
        state.v_dt,
        rdt,
        origin=origin,
        domain=kbot_domain,
    )
