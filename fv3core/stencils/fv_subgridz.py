import gt4py.gtscript as gtscript
from gt4py.gtscript import BACKWARD, PARALLEL, computation, interval

import fv3core._config as spec
import fv3core.utils.gt4py_utils as utils
from fv3core.decorators import ArgSpec, FrozenStencil, gtstencil, state_inputs
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
from fv3core.utils.typing import FloatField

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
    gz: FloatField,
    t0: FloatField,
    u0: FloatField,
    v0: FloatField,
    w0: FloatField,
    hd: FloatField,
    cvm: FloatField,
    cpm: FloatField,
    te: FloatField,
    ua: FloatField,
    va: FloatField,
    w: FloatField,
    ta: FloatField,
    delz: FloatField,
    q0_vapor: FloatField,
    q0_liquid: FloatField,
    q0_rain: FloatField,
    q0_ice: FloatField,
    q0_snow: FloatField,
    q0_graupel: FloatField,
    q0_o3mr: FloatField,
    q0_sgs_tke: FloatField,
    q0_cld: FloatField,
    qvapor: FloatField,
    qliquid: FloatField,
    qrain: FloatField,
    qice: FloatField,
    qsnow: FloatField,
    qgraupel: FloatField,
    qo3mr: FloatField,
    qsgs_tke: FloatField,
    qcld: FloatField,    
    xvir: float,
):
    with computation(PARALLEL), interval(...):
        t0 = ta
        u0 = ua
        v0 = va
        w0 = w
        # TODO: in a loop over tracers
        q0_vapor = qvapor
        q0_liquid = qliquid
        q0_rain = qrain
        q0_ice = qice
        q0_snow = qsnow
        q0_graupel = qgraupel
        q0_o3mr = qo3mr
        q0_sgs_tke = qsgs_tke
        q0_cld = qcld
        gzh = 0.0
    with computation(BACKWARD), interval(0, -1):
        # note only for nwat = 6
        cpm, cvm = standard_cm(
            cpm, cvm, q0_vapor, q0_liquid, q0_rain, q0_ice, q0_snow, q0_graupel
        )
        gz = gzh[0, 0, 1] - G2 * delz
        tmp = tvol(gz, u0, v0, w0)
        hd = cpm * t0 + tmp
        te = cvm * t0 + tmp
        gzh = gzh[0, 0, 1] - GRAV * delz


@gtscript.function
def qcon_func(qcon, q0_liquid, q0_ice, q0_snow, q0_rain, q0_graupel):
    return q0_liquid + q0_ice + q0_snow + q0_rain + q0_graupel



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
def compute_richardson_number(t0, q0_vapor, qcon, pkz, delp, peln, gz, u0, v0,xvir,t_max, t_min):
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
    ri_ref = RI_MIN + (RI_MAX - RI_MIN) * dim(400.0e2, delp / (peln[0, 0, 1] - peln)) / 200.0e2
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
def kh_adjustment(mc, q0):
    return mc * (q0 - q0[0, 0, -1])

@gtscript.function
def adjust_down(delp, h0, q0):
    return  q0 - h0 / delp

@gtscript.function
def adjust_up(delp, h0, q0):
    return  q0 + h0[0, 0, 1] / delp


@gtstencil
def m_loop(
    u0: FloatField,
    v0: FloatField,
    w0: FloatField,
    t0: FloatField,
    hd: FloatField,
    gz: FloatField,
    delp: FloatField,peln: FloatField, 
    pkz: FloatField,
    q0_vapor: FloatField,
    q0_liquid: FloatField,
    q0_rain: FloatField,
    q0_ice: FloatField,
    q0_snow: FloatField,
    q0_graupel: FloatField,
    q0_o3mr: FloatField,
    q0_sgs_tke: FloatField,
    q0_cld: FloatField,
    te: FloatField,
    cpm: FloatField, cvm:FloatField, 
    t_min: float,
    t_max: float,
    ratio: float,
    xvir: float,n: int,
):
    with computation(PARALLEL), interval(...):
        qcon = qcon_func(qcon, q0_liquid, q0_ice, q0_snow, q0_rain, q0_graupel)
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
        mc = 0.0
        ri = 0.0
        ref = 0.0
    with computation(BACKWARD):
        with interval(-1, None):
            # Adjustment for K-H instability:
            # Compute equivalent mass flux: mc
            # Add moist 2-dz instability consideration:
            ri, ri_ref = compute_richardson_number(t0, q0_vapor, qcon, pkz,delp, peln, gz, u0, v0,xvir, t_max, t_min)
            mc = compute_mass_flux(ri, ri_ref, delp, mc, ratio)
            if ri < ri_ref:
                # TODO: loop over tracers not hardcoded
                # Note combining into functions breaks
                # validation
                h0_vapor = kh_adjustment(mc, q0_vapor)
                q0_vapor = adjust_down(delp, h0_vapor, q0_vapor)
                h0_liquid = kh_adjustment(mc, q0_liquid)
                q0_liquid = adjust_down(delp, h0_liquid, q0_liquid)
                h0_rain = kh_adjustment(mc, q0_rain)
                q0_rain = adjust_down(delp, h0_rain, q0_rain)
                h0_ice = kh_adjustment(mc, q0_ice)
                q0_ice = adjust_down(delp, h0_ice, q0_ice)
                h0_snow = kh_adjustment(mc, q0_snow)
                q0_snow = adjust_down(delp, h0_snow, q0_snow)
                h0_graupel = kh_adjustment(mc, q0_graupel)
                q0_graupel = adjust_down(delp, h0_graupel, q0_graupel)
                h0_o3mr = kh_adjustment(mc, q0_o3mr)
                q0_o3mr = adjust_down(delp, h0_o3mr, q0_o3mr)
                h0_sgs_tke = kh_adjustment(mc, q0_sgs_tke)
                q0_sgs_tke = adjust_down(delp, h0_sgs_tke, q0_sgs_tke)
                h0_cld = kh_adjustment(mc, q0_cld)
                q0_cld = adjust_down(delp, h0_cld, q0_cld)
                h0_u = kh_adjustment(mc, u0)
                u0 = adjust_down(delp, h0_u, u0)
                h0_v = kh_adjustment(mc, v0)
                v0 = adjust_down(delp, h0_v, v0)
                h0_w = kh_adjustment(mc, w0)
                w0 = adjust_down(delp, h0_w, w0)
                h0_te = kh_adjustment(mc, hd)
                te = adjust_down(delp, h0_te, te)
            cpm, cvm, t0, hd = adjust_cvm( cpm, cvm, q0_vapor, q0_liquid, q0_rain, q0_ice, q0_snow, q0_graupel, gz, u0, v0, w0, t0, te,hd)
        with interval(4, -1):
            if ri[0, 0, 1] < ri_ref[0, 0, 1]:
                q0_vapor = adjust_up(delp, h0_vapor, q0_vapor)
                q0_liquid = adjust_up(delp, h0_liquid, q0_liquid)
                q0_rain = adjust_up(delp, h0_rain, q0_rain)
                q0_ice = adjust_up(delp, h0_ice, q0_ice)
                q0_snow = adjust_up(delp, h0_snow, q0_snow)
                q0_graupel = adjust_up(delp, h0_graupel, q0_graupel)
                q0_o3mr = adjust_up(delp, h0_o3mr, q0_o3mr)
                q0_sgs_tke = adjust_up(delp, h0_sgs_tke, q0_sgs_tke)
                q0_cld = adjust_up(delp, h0_cld, q0_cld)
                # recompute qcon
                qcon = qcon_func(qcon, q0_liquid, q0_ice, q0_snow, q0_rain, q0_graupel)
                u0 = adjust_up(delp, h0_u, u0)
                v0 = adjust_up(delp, h0_v, v0)
                w0 = adjust_up(delp, h0_w, w0)
                te = adjust_up(delp, h0_te, te)
            cpm, cvm, t0, hd = adjust_cvm( cpm, cvm, q0_vapor, q0_liquid, q0_rain, q0_ice, q0_snow, q0_graupel, gz, u0, v0, w0, t0, te,hd)
            ri, ri_ref = compute_richardson_number(t0, q0_vapor, qcon, pkz,delp, peln, gz, u0, v0,xvir, t_max, t_min)

            mc = compute_mass_flux(ri, ri_ref, delp, mc, ratio)
            if ri < ri_ref:
                h0_vapor = kh_adjustment(mc, q0_vapor)
                q0_vapor = adjust_down(delp, h0_vapor, q0_vapor)
                h0_liquid = kh_adjustment(mc, q0_liquid)
                q0_liquid = adjust_down(delp, h0_liquid, q0_liquid)
                h0_rain = kh_adjustment(mc, q0_rain)
                q0_rain = adjust_down(delp, h0_rain, q0_rain)
                h0_ice = kh_adjustment(mc, q0_ice)
                q0_ice = adjust_down(delp, h0_ice, q0_ice)
                h0_snow = kh_adjustment(mc, q0_snow)
                q0_snow = adjust_down(delp, h0_snow, q0_snow)
                h0_graupel = kh_adjustment(mc, q0_graupel)
                q0_graupel = adjust_down(delp, h0_graupel, q0_graupel)
                h0_o3mr = kh_adjustment(mc, q0_o3mr)
                q0_o3mr = adjust_down(delp, h0_o3mr, q0_o3mr)
                h0_sgs_tke = kh_adjustment(mc, q0_sgs_tke)
                q0_sgs_tke = adjust_down(delp, h0_sgs_tke, q0_sgs_tke)
                h0_cld = kh_adjustment(mc, q0_cld)
                q0_cld = adjust_down(delp, h0_cld, q0_cld)
               
                h0_u = kh_adjustment(mc, u0)
                u0 = adjust_down(delp, h0_u, u0)
                h0_v = kh_adjustment(mc, v0)
                v0 = adjust_down(delp, h0_v, v0)
                h0_w = kh_adjustment(mc, w0)
                w0 = adjust_down(delp, h0_w, w0)
                h0_te = kh_adjustment(mc, hd)
                te = adjust_down(delp, h0_te, te)
        
            cpm, cvm, t0, hd = adjust_cvm( cpm, cvm, q0_vapor, q0_liquid, q0_rain, q0_ice, q0_snow, q0_graupel, gz, u0, v0, w0, t0, te, hd)
        with interval(3, 4):
            # TODO: this is repetitive, but using functions did not work as
            # expected. spend some more time here so not so much needs
            # to be repeated just to multiply ri_ref by a constant
            if ri[0, 0, 1] < ri_ref[0, 0, 1]:
                q0_vapor = adjust_up(delp, h0_vapor, q0_vapor)
                q0_liquid = adjust_up(delp, h0_liquid, q0_liquid)
                q0_rain = adjust_up(delp, h0_rain, q0_rain)
                q0_ice = adjust_up(delp, h0_ice, q0_ice)
                q0_snow = adjust_up(delp, h0_snow, q0_snow)
                q0_graupel = adjust_up(delp, h0_graupel, q0_graupel)
                q0_o3mr = adjust_up(delp, h0_o3mr, q0_o3mr)
                q0_sgs_tke = adjust_up(delp, h0_sgs_tke, q0_sgs_tke)
                q0_cld = adjust_up(delp, h0_cld, q0_cld)
                # recompute qcon
                qcon = qcon_func(qcon, q0_liquid, q0_ice, q0_snow, q0_rain, q0_graupel)
                u0 = adjust_up(delp, h0_u, u0)
                v0 = adjust_up(delp, h0_v, v0)
                w0 = adjust_up(delp, h0_w, w0)
                te = adjust_up(delp, h0_te, te)

            cpm, cvm, t0, hd = adjust_cvm( cpm, cvm, q0_vapor, q0_liquid, q0_rain, q0_ice, q0_snow, q0_graupel, gz, u0, v0, w0, t0, te,hd)
            ri, ri_ref = compute_richardson_number(t0, q0_vapor, qcon, pkz,delp, peln, gz, u0, v0,xvir, t_max, t_min)
            # TODO, can we just check if index(K) == 3?
            ri_ref = ri_ref * 1.5
            mc = compute_mass_flux(ri, ri_ref, delp, mc, ratio)
            if ri < ri_ref:
                h0_vapor = kh_adjustment(mc, q0_vapor)
                q0_vapor = adjust_down(delp, h0_vapor, q0_vapor)
                h0_liquid = kh_adjustment(mc, q0_liquid)
                q0_liquid = adjust_down(delp, h0_liquid, q0_liquid)
                h0_rain = kh_adjustment(mc, q0_rain)
                q0_rain = adjust_down(delp, h0_rain, q0_rain)
                h0_ice = kh_adjustment(mc, q0_ice)
                q0_ice = adjust_down(delp, h0_ice, q0_ice)
                h0_snow = kh_adjustment(mc, q0_snow)
                q0_snow = adjust_down(delp, h0_snow, q0_snow)
                h0_graupel = kh_adjustment(mc, q0_graupel)
                q0_graupel = adjust_down(delp, h0_graupel, q0_graupel)
                h0_o3mr = kh_adjustment(mc, q0_o3mr)
                q0_o3mr = adjust_down(delp, h0_o3mr, q0_o3mr)
                h0_sgs_tke = kh_adjustment(mc, q0_sgs_tke)
                q0_sgs_tke = adjust_down(delp, h0_sgs_tke, q0_sgs_tke)
                h0_cld = kh_adjustment(mc, q0_cld)
                q0_cld = adjust_down(delp, h0_cld, q0_cld)
               
                h0_u = kh_adjustment(mc, u0)
                u0 = adjust_down(delp, h0_u, u0)
                h0_v = kh_adjustment(mc, v0)
                v0 = adjust_down(delp, h0_v, v0)
                h0_w = kh_adjustment(mc, w0)
                w0 = adjust_down(delp, h0_w, w0)
                h0_te = kh_adjustment(mc, hd)
                te = adjust_down(delp, h0_te, te)
        
            cpm, cvm, t0, hd = adjust_cvm( cpm, cvm, q0_vapor, q0_liquid, q0_rain, q0_ice, q0_snow, q0_graupel, gz, u0, v0, w0, t0, te, hd)
        with interval(2, 3):
            if ri[0, 0, 1] < ri_ref[0, 0, 1]:
                q0_vapor = adjust_up(delp, h0_vapor, q0_vapor)
                q0_liquid = adjust_up(delp, h0_liquid, q0_liquid)
                q0_rain = adjust_up(delp, h0_rain, q0_rain)
                q0_ice = adjust_up(delp, h0_ice, q0_ice)
                q0_snow = adjust_up(delp, h0_snow, q0_snow)
                q0_graupel = adjust_up(delp, h0_graupel, q0_graupel)
                q0_o3mr = adjust_up(delp, h0_o3mr, q0_o3mr)
                q0_sgs_tke = adjust_up(delp, h0_sgs_tke, q0_sgs_tke)
                q0_cld = adjust_up(delp, h0_cld, q0_cld)
                # recompute qcon
                qcon = qcon_func(qcon, q0_liquid, q0_ice, q0_snow, q0_rain, q0_graupel)
                u0 = adjust_up(delp, h0_u, u0)
                v0 = adjust_up(delp, h0_v, v0)
                w0 = adjust_up(delp, h0_w, w0)
                te = adjust_up(delp, h0_te, te)

            cpm, cvm, t0, hd = adjust_cvm( cpm, cvm, q0_vapor, q0_liquid, q0_rain, q0_ice, q0_snow, q0_graupel, gz, u0, v0, w0, t0, te,hd)
            ri, ri_ref = compute_richardson_number(t0, q0_vapor, qcon, pkz,delp, peln, gz, u0, v0,xvir, t_max, t_min)
            ri_ref = ri_ref * 2.0
            mc = compute_mass_flux(ri, ri_ref, delp, mc, ratio)
            if ri < ri_ref:
                h0_vapor = kh_adjustment(mc, q0_vapor)
                q0_vapor = adjust_down(delp, h0_vapor, q0_vapor)
                h0_liquid = kh_adjustment(mc, q0_liquid)
                q0_liquid = adjust_down(delp, h0_liquid, q0_liquid)
                h0_rain = kh_adjustment(mc, q0_rain)
                q0_rain = adjust_down(delp, h0_rain, q0_rain)
                h0_ice = kh_adjustment(mc, q0_ice)
                q0_ice = adjust_down(delp, h0_ice, q0_ice)
                h0_snow = kh_adjustment(mc, q0_snow)
                q0_snow = adjust_down(delp, h0_snow, q0_snow)
                h0_graupel = kh_adjustment(mc, q0_graupel)
                q0_graupel = adjust_down(delp, h0_graupel, q0_graupel)
                h0_o3mr = kh_adjustment(mc, q0_o3mr)
                q0_o3mr = adjust_down(delp, h0_o3mr, q0_o3mr)
                h0_sgs_tke = kh_adjustment(mc, q0_sgs_tke)
                q0_sgs_tke = adjust_down(delp, h0_sgs_tke, q0_sgs_tke)
                h0_cld = kh_adjustment(mc, q0_cld)
                q0_cld = adjust_down(delp, h0_cld, q0_cld)
               
                h0_u = kh_adjustment(mc, u0)
                u0 = adjust_down(delp, h0_u, u0)
                h0_v = kh_adjustment(mc, v0)
                v0 = adjust_down(delp, h0_v, v0)
                h0_w = kh_adjustment(mc, w0)
                w0 = adjust_down(delp, h0_w, w0)
                h0_te = kh_adjustment(mc, hd)
                te = adjust_down(delp, h0_te, te)
        
            cpm, cvm, t0, hd = adjust_cvm( cpm, cvm, q0_vapor, q0_liquid, q0_rain, q0_ice, q0_snow, q0_graupel, gz, u0, v0, w0, t0, te, hd)
        with interval(1, 2):
            if ri[0, 0, 1] < ri_ref[0, 0, 1]:
                q0_vapor = adjust_up(delp, h0_vapor, q0_vapor)
                q0_liquid = adjust_up(delp, h0_liquid, q0_liquid)
                q0_rain = adjust_up(delp, h0_rain, q0_rain)
                q0_ice = adjust_up(delp, h0_ice, q0_ice)
                q0_snow = adjust_up(delp, h0_snow, q0_snow)
                q0_graupel = adjust_up(delp, h0_graupel, q0_graupel)
                q0_o3mr = adjust_up(delp, h0_o3mr, q0_o3mr)
                q0_sgs_tke = adjust_up(delp, h0_sgs_tke, q0_sgs_tke)
                q0_cld = adjust_up(delp, h0_cld, q0_cld)
                # recompute qcon
                qcon = qcon_func(qcon, q0_liquid, q0_ice, q0_snow, q0_rain, q0_graupel)
                u0 = adjust_up(delp, h0_u, u0)
                v0 = adjust_up(delp, h0_v, v0)
                w0 = adjust_up(delp, h0_w, w0)
                te = adjust_up(delp, h0_te, te)

            cpm, cvm, t0, hd = adjust_cvm( cpm, cvm, q0_vapor, q0_liquid, q0_rain, q0_ice, q0_snow, q0_graupel, gz, u0, v0, w0, t0, te,hd)
            ri, ri_ref = compute_richardson_number(t0, q0_vapor, qcon, pkz,delp, peln, gz, u0, v0,xvir, t_max, t_min)
            ri_ref = ri_ref * 4.0
            mc = compute_mass_flux(ri, ri_ref, delp, mc, ratio)
            if ri < ri_ref:
                h0_vapor = kh_adjustment(mc, q0_vapor)
                q0_vapor = adjust_down(delp, h0_vapor, q0_vapor)
                h0_liquid = kh_adjustment(mc, q0_liquid)
                q0_liquid = adjust_down(delp, h0_liquid, q0_liquid)
                h0_rain = kh_adjustment(mc, q0_rain)
                q0_rain = adjust_down(delp, h0_rain, q0_rain)
                h0_ice = kh_adjustment(mc, q0_ice)
                q0_ice = adjust_down(delp, h0_ice, q0_ice)
                h0_snow = kh_adjustment(mc, q0_snow)
                q0_snow = adjust_down(delp, h0_snow, q0_snow)
                h0_graupel = kh_adjustment(mc, q0_graupel)
                q0_graupel = adjust_down(delp, h0_graupel, q0_graupel)
                h0_o3mr = kh_adjustment(mc, q0_o3mr)
                q0_o3mr = adjust_down(delp, h0_o3mr, q0_o3mr)
                h0_sgs_tke = kh_adjustment(mc, q0_sgs_tke)
                q0_sgs_tke = adjust_down(delp, h0_sgs_tke, q0_sgs_tke)
                h0_cld = kh_adjustment(mc, q0_cld)
                q0_cld = adjust_down(delp, h0_cld, q0_cld)
               
                h0_u = kh_adjustment(mc, u0)
                u0 = adjust_down(delp, h0_u, u0)
                h0_v = kh_adjustment(mc, v0)
                v0 = adjust_down(delp, h0_v, v0)
                h0_w = kh_adjustment(mc, w0)
                w0 = adjust_down(delp, h0_w, w0)
                h0_te = kh_adjustment(mc, hd)
                te = adjust_down(delp, h0_te, te)
        
            cpm, cvm, t0, hd = adjust_cvm( cpm, cvm, q0_vapor, q0_liquid, q0_rain, q0_ice, q0_snow, q0_graupel, gz, u0, v0, w0, t0, te, hd)

@gtscript.function
def readjust_by_frac(a0, a, fra):
    return a + (a0 - a) * fra


@gtstencil
def fraction_adjust(
    t0: FloatField,
    ta: FloatField,
    u0: FloatField,
    ua: FloatField,
    v0: FloatField,
    va: FloatField,
    w0: FloatField,
    w: FloatField,
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
def finalize_tracer(q0: FloatField, q: FloatField, fra: float):
    with computation(PARALLEL), interval(...):
        if fra < 1.0:
            q0 = readjust_by_frac(q0, q, fra)
        q = q0

@gtstencil
def finalize(
    u0: FloatField,
    v0: FloatField,
    w0: FloatField,
    t0: FloatField,
    ua: FloatField,
    va: FloatField,
    ta: FloatField,
    w: FloatField,
    u_dt: FloatField,
    v_dt: FloatField,
    rdt: float, fra: float, hydrostatic: bool
):
    with computation(PARALLEL), interval(...):
        if fra < 1.0:
            t0 = readjust_by_frac(t0, ta, fra)
            u0 = readjust_by_frac(u0, ua, fra)
            v0 = readjust_by_frac(v0, va, fra)
            if not hydrostatic:
                w0 = readjust_by_frac(w0, w, fra)
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
    origin = grid.compute_origin()
    shape = state.delp.shape
    q0 = {}
    for tracername in utils.tracer_variables:
        q0[tracername] =utils.make_storage_from_shape(shape)
        #copy(
        #    state.__dict__[tracername], cache_key="fv_subgridz_" + tracername
        #)

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
    gz = utils.make_storage_from_shape(
        shape, origin, init=True, cache_key="fv_subgridz_gz"
    )
    t0 = utils.make_storage_from_shape(
        shape, origin, init=True, cache_key="fv_subgridz_t0"
    )

    hd = utils.make_storage_from_shape(shape, origin, cache_key="fv_subgridz_hd")
    te = utils.make_storage_from_shape(shape, origin, cache_key="fv_subgridz_te")

    cvm = utils.make_storage_from_shape(shape, origin, cache_key="fv_subgridz_cvm")
    cpm = utils.make_storage_from_shape(shape, origin, cache_key="fv_subgridz_cpm")

    kbot_domain = (grid.nic, grid.njc, k_bot)
    origin = grid.compute_origin()
    init(
        gz,
        t0,
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
        state.delz,
        q0["qvapor"],
        q0["qliquid"],
        q0["qrain"],
        q0["qice"],
        q0["qsnow"],
        q0["qgraupel"],
        q0["qo3mr"], q0["qsgs_tke"], q0["qcld"],
        state.qvapor,state.qliquid, state.qrain, state.qice, state.qsnow, state.qgraupel, state.qo3mr, state.qsgs_tke, state.qcld, 
        xvir,
        origin=origin,
        domain=(grid.nic, grid.njc, k_bot+1),
    )

    ratios = {0: 0.25, 1: 0.5, 2: 0.999}

    for n in range(m):
        ratio = ratios[n]
       
        m_loop(
            u0,
            v0,
            w0,
            t0,
            hd,
            gz,
            state.delp,state.peln, 
            state.pkz,
            q0["qvapor"],q0["qliquid"], q0["qrain"], q0["qice"], q0["qsnow"], q0["qgraupel"], q0["qo3mr"], q0["qsgs_tke"], q0["qcld"],
            te,
            cpm,
            cvm, 
            t_min,
            t_max,
            ratio,
            xvir,n,
            origin=grid.compute_origin(),
            domain=kbot_domain,
        )
       
   
    for tracername in utils.tracer_variables:
        finalize_tracer(
            q0[tracername],
            state.tracers[tracername],
            fra,
            origin=origin,
            domain=kbot_domain,
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
        rdt,fra, spec.namelist.hydrostatic, 
        origin=origin,
        domain=kbot_domain,
    )
