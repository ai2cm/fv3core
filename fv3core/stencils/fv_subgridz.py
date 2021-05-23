import gt4py.gtscript as gtscript
from gt4py.gtscript import BACKWARD, PARALLEL, computation, interval

import fv3core._config as spec
import fv3core.utils.gt4py_utils as utils
from fv3core.decorators import ArgSpec, gtstencil, state_inputs
from fv3core.stencils.basic_operations import copy, copy_stencil, dim
from fv3core.utils.global_constants import (
    C_ICE,
    C_LIQ,
    CP_AIR,
    CP_VAP,
    CV_AIR,
    CV_VAP,
    GRAV,
    RDGAS,
    ZVIR,
)


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


@gtstencil
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


@gtscript.function
def qcon_func(qcon, q0_liquid, q0_ice, q0_snow, q0_rain, q0_graupel):
    return q0_liquid + q0_ice + q0_snow + q0_rain + q0_graupel


@gtstencil
def compute_qcon(
    qcon: sd, q0_liquid: sd, q0_ice: sd, q0_snow: sd, q0_rain: sd, q0_graupel: sd
):
    with computation(PARALLEL), interval(...):
        qcon = qcon_func(qcon, q0_liquid, q0_ice, q0_snow, q0_rain, q0_graupel)


@gtstencil
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

@gtscript.function
def adjust_cvm( cpm, cvm, q0_vapor, q0_liquid, q0_rain, q0_ice, q0_snow, q0_graupel, gz, u0, v0, w0, t0, te, hd):
    cpm, cvm = standard_cm(
        cpm, cvm, q0_vapor, q0_liquid, q0_rain, q0_ice, q0_snow, q0_graupel
    )
    tv = tvol(gz, u0, v0, w0)
    t0 = (te - tv) / cvm
    hd = cpm * t0 + tv
    return cpm, cvm, t0, hd
@gtscript.function
def compute_ri(t0, q0_vapor, qcon, pkz, pm, gz, u0, v0,xvir,t_max, t_min):
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
    return ri, ri_ref
@gtscript.function
def compute_mass_flux(ri, ri_ref, delp, mc, ratio):
    max_ri_ratio = ri / ri_ref
    if max_ri_ratio < 0.0:
        max_ri_ratio = 0.0
    if ri < ri_ref:
        mc = (
            ratio
            * delp[0, 0, -1]
            * delp
            / (delp[0, 0, -1] + delp)
            * (1.0 - max_ri_ratio) ** 2.0
        )
    return mc

@gtscript.function
def KH_instability_adjustment_bottom(ri, ri_ref, mc, delp, h0, q0):
    if ri < ri_ref:
        h0 = mc * (q0 - q0[0, 0, -1])
        q0 = q0 - h0 / delp
    return q0, h0

@gtscript.function
def KH_instability_adjustment_top(ri, ri_ref, delp, h0, q0):
    if ri[0, 0, 1] < ri_ref[0, 0, 1]:
        q0 = q0 + h0[0, 0, 1] / delp
    return q0

@gtscript.function
def KH_instability_adjustment_bottom_te(ri, ri_ref, mc, delp, h0, q0, hd):
    if ri < ri_ref:
        h0 = mc * (hd - hd[0, 0, -1])
        q0 = q0 - h0 / delp
    else:
        h0 = h0
        q0 = q0

    return q0, h0


@gtstencil
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
    #pt1: sd,
    #pt2: sd,
    #tv2: sd,
    te: sd,
    mc: sd,cpm: sd, cvm:sd, 
    q0_liquid: sd,
    q0_rain: sd,
    q0_ice: sd,
    q0_snow: sd,
    q0_graupel: sd,
    q0_o3mr: sd,
    q0_sgs_tke: sd,
    q0_cld: sd,
    t_min: float,
    t_max: float,
    ratio: float,
    xvir: float,
):
    with computation(PARALLEL), interval(...):
        h0_vapor = 0.0
        h0_liquid = 0.0
        h0_rain = 0.0
        h0_ice = 0.0
        h0_snow = 0.0
        h0_graupel = 0.0
        h0_o3mr = 0.0
        h0_sgs_tke = 0.0
        h0_cld = 0.0
        h0_u = 0.0
        h0_v = 0.0
        h0_w = 0.0
        h0_te = 0.0
    with computation(BACKWARD):
        with interval(-1, None):
            ri, ri_ref = compute_ri(t0, q0_vapor, qcon, pkz, pm, gz, u0, v0,xvir, t_max, t_min)
            mc = compute_mass_flux(ri, ri_ref, delp, mc, ratio)
            q0_vapor, h0_vapor = KH_instability_adjustment_bottom(ri, ri_ref, mc, delp, h0_vapor, q0_vapor)
            q0_liquid, h0_liquid = KH_instability_adjustment_bottom(ri, ri_ref, mc, delp, h0_liquid, q0_liquid)
            q0_rain, h0_rain = KH_instability_adjustment_bottom(ri, ri_ref, mc, delp, h0_rain, q0_rain)
            q0_ice, h0_ice = KH_instability_adjustment_bottom(ri, ri_ref, mc, delp, h0_ice, q0_ice)
            q0_snow, h0_snow = KH_instability_adjustment_bottom(ri, ri_ref, mc, delp, h0_snow, q0_snow)
            q0_graupel, h0_graupel = KH_instability_adjustment_bottom(ri, ri_ref, mc, delp, h0_graupel, q0_graupel)
            q0_o3mr, h0_o3mr = KH_instability_adjustment_bottom(ri, ri_ref, mc, delp, h0_o3mr, q0_o3mr)
            q0_sgs_tke, h0_sgs_tke = KH_instability_adjustment_bottom(ri, ri_ref, mc, delp, h0_sgs_tke, q0_sgs_tke)
            q0_cld, h0_cld = KH_instability_adjustment_bottom(ri, ri_ref, mc, delp, h0_cld, q0_cld)
            u0, h0_u = KH_instability_adjustment_bottom(ri, ri_ref, mc, delp, h0_u, u0)
            v0, h0_v = KH_instability_adjustment_bottom(ri, ri_ref, mc, delp, h0_v, v0)
            w0, h0_w = KH_instability_adjustment_bottom(ri, ri_ref, mc, delp, h0_w, w0)
            te, h0_te = KH_instability_adjustment_bottom_te(ri, ri_ref, mc, delp, h0_te, te, hd)
            cpm, cvm, t0, hd = adjust_cvm( cpm, cvm, q0_vapor, q0_liquid, q0_rain, q0_ice, q0_snow, q0_graupel, gz, u0, v0, w0, t0, te,hd)
        with interval(4, -1):
            q0_vapor  = KH_instability_adjustment_top(ri, ri_ref, delp, h0_vapor,q0_vapor)
            q0_liquid= KH_instability_adjustment_top(ri, ri_ref,  delp, h0_liquid, q0_liquid)
            q0_rain = KH_instability_adjustment_top(ri, ri_ref,  delp, h0_rain, q0_rain)
            q0_ice = KH_instability_adjustment_top(ri, ri_ref,  delp, h0_ice, q0_ice)
            q0_snow = KH_instability_adjustment_top(ri, ri_ref,  delp, h0_snow, q0_snow)
            q0_graupel = KH_instability_adjustment_top(ri, ri_ref,  delp, h0_graupel, q0_graupel)
            q0_o3mr = KH_instability_adjustment_top(ri, ri_ref,  delp, h0_o3mr, q0_o3mr)
            q0_sgs_tke = KH_instability_adjustment_top(ri, ri_ref,  delp, h0_o3mr, q0_sgs_tke)
            q0_cld = KH_instability_adjustment_top(ri, ri_ref,  delp, h0_cld, q0_cld)
            if ri[0, 0, 1] < ri_ref[0, 0, 1]:
                qcon = qcon_func(qcon, q0_liquid, q0_ice, q0_snow, q0_rain, q0_graupel)
            u0 = KH_instability_adjustment_top(ri, ri_ref, delp, h0_u, u0)
            v0 = KH_instability_adjustment_top(ri, ri_ref, delp, h0_v, v0)
            w0 = KH_instability_adjustment_top(ri, ri_ref, delp, h0_w, w0)
            te = KH_instability_adjustment_top(ri, ri_ref, delp, h0_te, te)
            cpm, cvm, t0, hd = adjust_cvm( cpm, cvm, q0_vapor, q0_liquid, q0_rain, q0_ice, q0_snow, q0_graupel, gz, u0, v0, w0, t0, te,hd)
            ri, ri_ref = compute_ri(t0, q0_vapor, qcon, pkz, pm, gz, u0, v0,xvir, t_max, t_min)
            # with computation(PARALLEL):
            #    with interval(1, 2):
            #        ri_ref = ri_ref_copy * 3.0
            #    with interval(2, 3):
            #        ri_ref = ri_ref_copy * 2.0
            #    with interval(3, 4):
            #        ri_ref = ri_ref_copy * 1.5
            #with computation(PARALLEL), interval(1, None):
            mc = compute_mass_flux(ri, ri_ref, delp, mc, ratio)
        
            q0_vapor, h0_vapor =KH_instability_adjustment_bottom(ri, ri_ref, mc, delp, h0_vapor, q0_vapor)
            q0_liquid, h0_liquid = KH_instability_adjustment_bottom(ri, ri_ref, mc, delp, h0_liquid, q0_liquid)
            q0_rain, h0_rain = KH_instability_adjustment_bottom(ri, ri_ref, mc, delp, h0_rain, q0_rain)
            q0_ice, h0_ice = KH_instability_adjustment_bottom(ri, ri_ref, mc, delp, h0_ice, q0_ice)
            q0_snow, h0_snow = KH_instability_adjustment_bottom(ri, ri_ref, mc, delp, h0_snow, q0_snow)
            q0_graupel, h0_graupel = KH_instability_adjustment_bottom(ri, ri_ref, mc, delp, h0_graupel, q0_graupel)
            q0_o3mr, h0_o3mr = KH_instability_adjustment_bottom(ri, ri_ref, mc, delp, h0_o3mr, q0_o3mr)
            q0_sgs_tke, h0_sgs_tke = KH_instability_adjustment_bottom(ri, ri_ref, mc, delp, h0_sgs_tke, q0_sgs_tke)
            q0_cld, h0_cld = KH_instability_adjustment_bottom(ri, ri_ref, mc, delp, h0_cld, q0_cld)
            u0, h0_u = KH_instability_adjustment_bottom(ri, ri_ref, mc, delp, h0_u, u0)
            v0, h0_v = KH_instability_adjustment_bottom(ri, ri_ref, mc, delp, h0_v, v0)
            w0, h0_w = KH_instability_adjustment_bottom(ri, ri_ref, mc, delp, h0_w, w0)
            te, h0_te = KH_instability_adjustment_bottom_te(ri, ri_ref, mc, delp, h0_te, te, hd)
           
            cpm, cvm, t0, hd = adjust_cvm( cpm, cvm, q0_vapor, q0_liquid, q0_rain, q0_ice, q0_snow, q0_graupel, gz, u0, v0, w0, t0, te, hd)
        #with interval(3, 4)
"""
 # kh_bottom tracers
            # with computation(BACKWARD), interval(...):
            #if ri < ri_ref:
            #    h0 = mc * (q0 - q0[0, 0, -1])
            #    q0 = q0 - h0 / delp
            # kh _top tracers k - 1
            #with computation(BACKWARD), interval(...):
            #    if ri[0, 0, 1] < ri_ref[0, 0, 1]:
            #        q0 = q0 + h0[0, 0, 1] / delp
            # recompute qcon
            #with computation(BACKWARD), interval(...):
            #    if ri[0, 0, 1] < ri_ref[0, 0, 1]:
            #        qcon = qcon_func(qcon, q0_liquid, q0_ice, q0_snow, q0_rain, q0_graupel)
            # KH on u0, v0, w0, te
            # double adjust
            #with computation(BACKWARD), interval(...):
"""
@gtstencil
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


# 3d version, doesn't work due to this k-1 value needing to be updated before
# calculating variables in the k - 1 case.
# @gtstencil
# def KH_instability_adjustment(ri: sd, ri_ref: sd, mc: sd, q0: sd, delp: sd):
#     with computation(BACKWARD):
#         with interval(-1, None):
#             h0 = 0.
#             if ri < ri_ref:
#                 h0 = mc * (q0 - q0[0, 0, -1])
#                 q0 = q0 - h0 / delp
#         with interval(1, -1):
#             h0 = 0.
#             if ri[0, 0, 1] < ri_ref[0, 0, 1]:
#                 q0 = q0 + h0[0, 0, 1] / delp
#             if ri < ri_ref:
#                 h0 = mc * (q0 - q0[0, 0, -1])
#                 q0 = q0 - h0 / delp
#         with interval(0, 1):
#             if ri[0, 0, 1] < ri_ref[0, 0, 1]:
#                 q0 = q0 + h0[0, 0, 1] / delp


"""

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


@gtstencil
def KH_instability_adjustment_bottom_te(
    ri: sd, ri_ref: sd, mc: sd, q0: sd, delp: sd, h0: sd, hd: sd
):
    with computation(BACKWARD), interval(...):
        if ri < ri_ref:
            h0 = mc * (hd - hd[0, 0, -1])
            q0 = q0 - h0 / delp


@gtstencil
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

"""
@gtscript.function
def readjust_by_frac(a0, a, fra):
    return a + (a0 - a) * fra


@gtstencil
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


@gtstencil
def fraction_adjust_tracer(q0: sd, q: sd, fra: float):
    with computation(PARALLEL), interval(...):
        q0 = readjust_by_frac(q0, q, fra)


@gtstencil
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


# TODO: Replace with something from fv3core.onfig probably, using the
# field_table. When finalize reperesentation of tracers, adjust this.
def tracers_dict(state):
    tracers = {}
    for tracername in utils.tracer_variables:
        tracers[tracername] = state.__dict__[tracername]
    state.tracers = tracers


@state_inputs(
    ArgSpec("delp", "pressure_thickness_of_atmospheric_layer", "Pa", intent="in"),
    ArgSpec("delz", "vertical_thickness_of_atmospheric_layer", "m", intent="in"),
    ArgSpec("pe", "interface_pressure", "Pa", intent="in"),
    ArgSpec(
        "pkz", "layer_mean_pressure_raised_to_power_of_kappa", "unknown", intent="in"
    ),
    ArgSpec("peln", "logarithm_of_interface_pressure", "ln(Pa)", intent="in"),
    ArgSpec("pt", "air_temperature", "degK", intent="inout"),
    ArgSpec("ua", "eastward_wind", "m/s", intent="inout"),
    ArgSpec("va", "northward_wind", "m/s", intent="inout"),
    ArgSpec("w", "vertical_wind", "m/s", intent="inout"),
    ArgSpec("qvapor", "specific_humidity", "kg/kg", intent="inout"),
    ArgSpec("qliquid", "cloud_water_mixing_ratio", "kg/kg", intent="inout"),
    ArgSpec("qrain", "rain_mixing_ratio", "kg/kg", intent="inout"),
    ArgSpec("qsnow", "snow_mixing_ratio", "kg/kg", intent="inout"),
    ArgSpec("qice", "cloud_ice_mixing_ratio", "kg/kg", intent="inout"),
    ArgSpec("qgraupel", "graupel_mixing_ratio", "kg/kg", intent="inout"),
    ArgSpec("qo3mr", "ozone_mixing_ratio", "kg/kg", intent="inout"),
    ArgSpec("qsgs_tke", "turbulent_kinetic_energy", "m**2/s**2", intent="inout"),
    ArgSpec("qcld", "cloud_fraction", "", intent="inout"),
    ArgSpec("u_dt", "eastward_wind_tendency_due_to_physics", "m/s**2", intent="inout"),
    ArgSpec("v_dt", "northward_wind_tendency_due_to_physics", "m/s**2", intent="inout"),
)
def compute(state, nq, dt):
    tracers_dict(state)  # TODO get rid of this when finalize representation of tracers

    grid = spec.grid
    rdt = 1.0 / dt
    k_bot = spec.namelist.n_sponge
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

    if spec.namelist.nwat == 0:
        xvir = 0.0
        # rz = 0 # hydrostatic only
    else:
        xvir = ZVIR
        # rz = constants.RV_GAS - constants.RDGAS # hydrostatic only
    m = 3
    fra = dt / float(spec.namelist.fv_sg_adj)
    if spec.namelist.hydrostatic:
        raise Exception("Hydrostatic not supported for fv_subgridz")
    q0 = {}
    for tracername in utils.tracer_variables:
        q0[tracername] = copy(
            state.__dict__[tracername], cache_key="fv_subgridz_" + tracername
        )
    origin = grid.compute_origin()
    shape = state.delp.shape
    # not 100% sure which of these require init=True,
    # if you figure it out please remove unnecessary ones and this comment
    u0 = utils.make_storage_from_shape(
        shape, origin, init=True, cache_key="fv_subgridz_u0"
    )
    v0 = utils.make_storage_from_shape(
        shape, origin, init=True, cache_key="fv_subgridz_v0"
    )
    w0 = utils.make_storage_from_shape(
        shape, origin, init=True, cache_key="fv_subgridz_w0"
    )
    gzh = utils.make_storage_from_shape(
        shape, origin, init=True, cache_key="fv_subgridz_gzh"
    )
    gz = utils.make_storage_from_shape(
        shape, origin, init=True, cache_key="fv_subgridz_gz"
    )
    t0 = utils.make_storage_from_shape(
        shape, origin, init=True, cache_key="fv_subgridz_t0"
    )
    pm = utils.make_storage_from_shape(
        shape, origin, init=True, cache_key="fv_subgridz_pm"
    )
    hd = utils.make_storage_from_shape(shape, origin, cache_key="fv_subgridz_hd")
    te = utils.make_storage_from_shape(shape, origin, cache_key="fv_subgridz_te")
    den = utils.make_storage_from_shape(shape, origin, cache_key="fv_subgridz_den")
    qcon = utils.make_storage_from_shape(
        shape, origin, init=True, cache_key="fv_subgridz_qcon"
    )
    cvm = utils.make_storage_from_shape(shape, origin, cache_key="fv_subgridz_cvm")
    cpm = utils.make_storage_from_shape(shape, origin, cache_key="fv_subgridz_cpm")

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

    ri = utils.make_storage_from_shape(shape, origin, cache_key="fv_subgridz_ri")
    ri_ref = utils.make_storage_from_shape(
        shape, origin, cache_key="fv_subgridz_ri_ref"
    )
    mc = utils.make_storage_from_shape(shape, origin, cache_key="fv_subgridz_mc")
    h0 = utils.make_storage_from_shape(shape, origin, cache_key="fv_subgridz_h0")
    #pt1 = utils.make_storage_from_shape(shape, origin, cache_key="fv_subgridz_pt1")
    #pt2 = utils.make_storage_from_shape(shape, origin, cache_key="fv_subgridz_pt2")
    #tv2 = utils.make_storage_from_shape(shape, origin, cache_key="fv_subgridz_tv2")
    ratios = {0: 0.25, 1: 0.5, 2: 0.999}

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
            #pt1,
            #pt2,
            #tv2,
            te,
            mc,cpm, cvm, q0["qliquid"], q0["qrain"], q0["qice"], q0["qsnow"], q0["qgraupel"], q0["qo3mr"], q0["qsgs_tke"], q0["qcld"],
            t_min,
            t_max,
            ratio,
            xvir,
            origin=grid.compute_origin(),
            domain=kbot_domain,
        )
        if n ==	0:
            print(t_min)
            for	z in range(40,50):
                print('---')
                print(z, 'ri',  ri[10, 11,z], ri_ref[10, 11, z])
                print(z, 't0',  t0[10, 11,z])
                print(z, 'qv',  q0["qvapor"][10, 11,z])
                print(z, 'qt',  qcon[10, 11,z])
                print(z, 'u0',  u0[10, 11,z])
                print(z, 'v0',  v0[10, 11,z])
                print(z, 'mc', mc[10,11,z])
                print(z, 'cv', cvm[10,11,z])
                print(z, 'cp', cpm[10,11,z])
                print(z, 'hd', hd[10,11,z])
                print(z, 'te', te[10,11,z])
                print(z, 'w0', w0[10,11,z])
            #for z in range(ri.shape[2]):
            #    print('pm at ',z,  pm[10, 11,z])
        #if k == 1:
        #    ri_ref *= 4.0
        #if k == 2:
        #    ri_ref *= 2.0
        #if k == 3:
        #    ri_ref *= 1.5
            
        #equivalent_mass_flux(
        #    ri, ri_ref, mc, state.delp, ratio, origin=korigin, domain=kdomain
        #
        """
        for k in range(k_bot - 1, 0, -1):
            korigin = (grid.is_, grid.js, k)
            korigin_m1 = (grid.is_, grid.js, k - 1)
            kdomain = (grid.nic, grid.njc, 1)
            kdomain_m1 = (grid.nic, grid.njc, 2)

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

            KH_instability_adjustment(
                ri, ri_ref, mc, u0, state.delp, h0, origin=korigin, domain=kdomain
            )

            KH_instability_adjustment(
                ri, ri_ref, mc, v0, state.delp, h0, origin=korigin, domain=kdomain
            )
            KH_instability_adjustment(
                ri, ri_ref, mc, w0, state.delp, h0, origin=korigin, domain=kdomain
            )
            KH_instability_adjustment_te(
                ri, ri_ref, mc, te, state.delp, h0, hd, origin=korigin, domain=kdomain
            )

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
        """
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
            spec.namelist.hydrostatic,
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
        copy_stencil(
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
