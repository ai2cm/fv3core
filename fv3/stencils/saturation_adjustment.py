#!/usr/bin/env python3
import fv3.utils.gt4py_utils as utils
import gt4py.gtscript as gtscript
import fv3._config as spec
from gt4py.gtscript import computation, interval, PARALLEL
import fv3.utils.global_constants as constants
import math
import numpy as np


sd = utils.sd
DC_VAP = constants.CP_VAP - constants.C_LIQ  # - 2339.5, isobaric heating / cooling
DC_ICE = constants.C_LIQ - constants.C_ICE  # 2213.5, isobaric heating / cooling
LV0 = constants.HLV - DC_VAP * constants.TICE  # 3.13905782e6, evaporation latent heat coefficient at 0 deg k
LI00 = constants.HLF - DC_ICE * constants.TICE  # -2.7105966e5, fusion latent heat coefficient at 0 deg k
LI2 = LV0 + LI00  # 2.86799816e6, sublimation latent heat coefficient at 0 deg k
D2ICE = DC_VAP + DC_ICE  # - 126, isobaric heating / cooling
E00 = 611.21  # saturation vapor pressure at 0 deg c
TMIN = constants.TICE - 160.0
DELT = 0.1
satmix = {'table': None, 'table2': None, 'tablew': None, 'des2': None, 'desw': None}
TICE = 273.16  #constants.TICE
T_WFR = TICE - 40. #homogeneous freezing temperature
TICE0 = TICE - 0.01
# melting of cloud ice to cloud water and rain
# TODO, when if blocks are possible,, only compute when 'melting'

def qs_init():
    length = 2621
    for tablename in satmix.keys():
        if satmix[tablename] is None:
            satmix[tablename] = np.zeros(length)
        else:
            return  # already initialized any table means initializec them all
    qs_table(length)
    qs_table2(length)
    qs_tablew(length)
    for i in range(length - 1):
        satmix['des2'][i] = max(0., satmix['table2'][i + 1] - satmix['table2'][i])
        satmix['desw'][i] = max(0., satmix['tablew'][i + 1] - satmix['tablew'][i])
    satmix['des2'][length - 1] = satmix['des2'][length - 2]
    satmix['desw'][length - 1] = satmix['desw'][length - 2]

def get_fac0(tem):
    return (tem - TICE) / (tem * TICE)
    
def get_fac2(tem, fac1, d):
    return (d * math.log(tem / TICE) + fac1) / constants.RVGAS
# TODO refactor into streamlined array calcs 
def qs_table(n):
    esupc = np.zeros(200)
    # compute es over ice between - 160 deg c and 0 deg c.
    for i in range(1600):
        tem = TMIN + DELT * i  # (i - 1)
        # fac0 = get_fac0(tem)
        # fac1 = fac0 * LI2
        fac2 = get_fac2(tem, get_fac0(tem) * LI2, D2ICE)
        satmix['table'][i] = E00 * math.exp(fac2)
    
    # compute es over water between - 20 deg c and 102 deg c.
    for i in range(1221):
        tem = 253.16 + DELT * i  # real (i - 1)
        # fac0 = get_fac0(tem)
        # fac1 = fac0 * LV0
        fac2 = get_fac2(tem, get_fac0(tem) * LV0, DC_VAP)
        esh20 = E00 * math.exp(fac2)
        if (i < 200):
            esupc[i] = esh20
        else:
            satmix['table'][i + 1400] = esh20  # TODO
    
    #  derive blended es over ice and supercooled water between - 20 deg c and 0 deg c
    for i in range(200):
        tem = 253.16 + DELT * i  # real (i - 1)
        wice = 0.05 * (TICE - tem)
        wh2o = 0.05 * (tem - 253.16)
        satmix['table'][i + 1400] = wice * satmix['table'][i + 1400] + wh2o * esupc[i]

def qs_table2(n):
   
    for i in range(n):
        tem0 = TMIN + DELT * i
        fac0 = get_fac0(tem0)
        if (i < 1600):
            # compute es over ice between - 160 deg c and 0 deg c.
            fac2 = get_fac2(tem0, fac0 * LI2, D2ICE)
        else:
            # compute es over water between 0 deg c and 102 deg c.
            fac2 = get_fac2(tem0, fac0 * LV0, DC_VAP)
        satmix['table2'][i] = E00 * math.exp(fac2)
   
    # smoother around 0 deg c
    smooth_indices = {1599: None, 1600: None}
    for i in smooth_indices.keys():
        smooth_indices[i] = 0.25 * (satmix['table2'][i - 1] + 2.0 * satmix['table'][i] + satmix['table2'][i + 1])
    for i, v in smooth_indices.items():
        satmix['table2'][i] = v
  
def qs_tablew(n):
    for i in range(n):
        tem = TMIN + DELT * i
        #fac0 = get_fac0(tem)
        #fac1 = fac0 * LV0
        fac2 =  get_fac2(tem,  get_fac0(tem) * LV0, DC_VAP)
        satmix['tablew'][i] = E00 * math.exp(fac2)
        
@gtscript.function
def compute_cvm(mc_air, qv, c_vap, q_liq, c_liq, q_sol, c_ice):
     return mc_air + qv * c_vap + q_liq * c_liq + q_sol * c_ice

 
@gtscript.function
def dim(a, b):
    diff = a - b if a - b > 0 else 0
    return diff
 
@gtscript.function
def melt_cloud_ice(qv, qi, ql, q_liq, q_sol, pt1, icp2, fac_imlt, mc_air, c_vap, c_liq, c_ice, lhi, cvm):
    factmp = 0.0  # TODO, if temporaries inside of if-statements become supported, remove this
    sink = 0.0  # TODO ditto factmp
    if ((qi > 1.0e-8) and (pt1 > TICE)):
        factmp = fac_imlt * (pt1 - TICE) / icp2
        sink = qi if qi < factmp else factmp
        qi = qi - sink
        ql = ql + sink
        q_liq = q_liq + sink
        q_sol = q_sol - sink
        cvm = compute_cvm(mc_air, qv, c_vap, q_liq, c_liq, q_sol, c_ice)
        pt1 = pt1 - sink * lhi / cvm
    else:  # TODO when else blocks are not required, remove this
        qi = qi
    return qi, ql, q_liq, q_sol, cvm, pt1

@gtscript.function
def minmax_tmp_h20(qa, qb):
    tmpmax = qb if qb > 0 else 0
    tmp = -qa if -qa < tmpmax else tmpmax
    return tmp

# fix negative snow with graupel or graupel with available snow
# TODO fix so only compute tmp when it is needed
@gtscript.function
def fix_negative_snow(qs, qg):
    tmp = minmax_tmp_h20(qg, qs)
    if qs < 0.0:
        qg = qg + qs
        qs = 0.0
    elif qg < 0.0:
        qg = qg + tmp
        qs = qs - tmp
    else:  # TODO remove when possible
        qg = qg
        qs = qs
    return qs, qg

# fix negative cloud water with rain or rain with available cloud water
# TODO fix so only compute tmp when it is needed
@gtscript.function
def fix_negative_cloud_water(ql, qr):
    tmpl = minmax_tmp_h20(ql, qr)
    tmpr = minmax_tmp_h20(qr, ql)
    if ql < 0.0:
        ql = ql + tmpl
        qr = qr - tmpl
    elif qr < 0.0:
        ql = ql - tmpr
        qr = qr + tmpr
    else:  # TODO remove when possible
        ql = ql
        qr = qr
    return ql, qr

 
# enforce complete freezing of cloud water to cloud ice below - 48 c
@gtscript.function
def complete_freezing(qv, ql, qi, q_liq, q_sol, pt1, cvm, icp2, mc_air, lhi, c_vap, c_ice, c_liq):
    dtmp = TICE - 48. - pt1
    sink = 0.0
    if (ql > 0. and dtmp > 0.):
        sink = ql if ql < dtmp / icp2 else dtmp / icp2
        ql = ql - sink
        qi = qi + sink
        q_liq = q_liq - sink
        q_sol = q_sol + sink
        cvm = compute_cvm(mc_air, qv, c_vap, q_liq, c_liq, q_sol, c_ice)
        pt1 = pt1 + sink * lhi / cvm
    else:  # TODO remove when possible
        qi = qi
    return ql, qi, q_liq, q_sol, cvm, pt1

@gtscript.function
def homogenous_freezing(qv, ql, qi, q_liq, q_sol, pt1, cvm, icp2, mc_air, lhi, c_vap, c_ice, c_liq):
    dtmp = T_WFR -  pt1 # [ - 40, - 48]
    sink = 0.
    if (ql > 0. and dtmp > 0.):
        sink = ql if ql < dtmp / icp2 else dtmp / icp2
        sink = sink if sink < ql * dtmp * 0.125 else ql * dtmp * 0.125  # min (ql, ql * dtmp * 0.125, dtmp / icp2)
        ql = ql - sink
        qi = qi + sink
        q_liq = q_liq - sink
        q_sol = q_sol + sink
        cvm = compute_cvm(mc_air, qv, c_vap, q_liq, c_liq, q_sol, c_ice)
        pt1 = pt1 + sink * lhi / cvm
    else:
        qi = qi
    return ql, qi, q_liq, q_sol, cvm, pt1

# bigg mechanism for heterogeneous freezing
@gtscript.function
def heterogeneous_freezing(exptc, pt1, cvm, ql, qi, q_liq, q_sol, den, icp2, dt_bigg, mc_air, lhi, qv, c_vap, c_liq, c_ice):
    tc = TICE0 - pt1
    sink = 0.
    if (ql > 0. and tc > 0.):
        sink = 3.3333e-10 * dt_bigg * (exptc - 1.) * den * ql** 2
        sink = ql if ql < sink else sink
        sink = ql if ql < tc / icp2 else tc / icp2
        ql = ql - sink
        qi = qi + sink
        q_liq = q_liq - sink
        q_sol = q_sol + sink
        cvm = compute_cvm(mc_air, qv, c_vap, q_liq, c_liq, q_sol, c_ice)
        pt1 = pt1 + sink * lhi / cvm
    else:
        sink = 0.
    return ql, qi, q_liq, q_sol, cvm, pt1

@gtscript.function
def make_graupel(pt1, cvm, fac_r2g, qr, qg, q_liq, q_sol, lhi, icp2, mc_air, qv, c_vap, c_liq, c_ice):
    dtmp = (TICE - 0.1) - pt1
    tmp = 0.
    sinktmp = 0.
    rainfac = 0.
    sink = 0
    if (qr > 1e-7 and dtmp > 0.):
        rainfac = (dtmp * 0.025) ** 2
        tmp = qr if 1.0 < rainfac else rainfac * qr  #  no limit on freezing below - 40 deg c
        sinktmp = fac_r2g * dtmp / icp2
        sink = tmp if tmp < sinktmp else sinktmp
        qr = qr - sink
        qg = qg + sink
        q_liq = q_liq - sink
        q_sol = q_sol + sink
        cvm = compute_cvm(mc_air, qv, c_vap, q_liq, c_liq, q_sol, c_ice)
        pt1 = pt1 + sink * lhi / cvm
    else:
        sink=sink
    return qr, qg , q_liq, q_sol, cvm, pt1


@gtscript.function
def melt_snow(pt1, cvm, fac_smlt, qs, ql, qr, q_liq, q_sol, lhi, icp2, mc_air, qv, c_vap, c_liq, c_ice, qs_mlt):
    dtmp = pt1 - (TICE + 0.1)
    tmp = 0.
    sink = 0.
    snowfac = 0.
    sinktmp = 0.
    dimqs = dim(qs_mlt, ql)
    if (qs > 1e-7 and dtmp > 0.):
        snowfac = (dtmp * 0.1)**2
        tmp = qs if 1. < snowfac else snowfac * qs  # no limiter on melting above 10 deg c
        sinktmp = fac_smlt * dtmp / icp2
        sink = tmp if tmp < sinktmp else sinktmp
        tmp = sink if sink < dimqs else dimqs
        qs = qs - sink
        ql = ql + tmp
        qr = qr - sink - tmp
        q_liq = q_liq + sink
        q_sol = q_sol - sink
        cvm = compute_cvm(mc_air, qv, c_vap, q_liq, c_liq, q_sol, c_ice)
        pt1 = pt1 + sink * lhi / cvm
    else:
        sink=sink
    return qs, ql, qr, q_liq, q_sol, cvm, pt1

@gtscript.function
def autoconversion_cloud_to_rain(ql, qr, fac_l2r, ql0_max):
    sink = 0.
    if (ql > ql0_max):
        sink = fac_l2r * (ql - ql0_max)
        qr = qr + sink
        ql = ql - sink
    else:
        sink=sink
    return ql, qr
@gtscript.function
def update_latent_heat_coefficient_i(pt1, cvm):
    lhi = LI00 + DC_ICE * pt1
    icp2 = lhi / cvm
    return lhi, icp2

@gtscript.function
def update_latent_heat_coefficient(pt1, cvm, lv00, d0_vap):
    lhl = lv00 + d0_vap * pt1
    lhi = LI00 + DC_ICE * pt1
    lcp2 = lhl / cvm
    icp2 = lhi / cvm
    #lhi, icp2 = update_latent_heat_coefficient_i(pt1, cvm)
    return lhl, lhi, lcp2, icp2

@gtscript.function
def adjust_pt1(pt1, src, lhl, cvm):
     return pt1 + src * lhl / cvm

@gtscript.function
def compute_dq0(qv, wqsat, dq2dt, tcp3):
    return (qv - wqsat) / (1.0 + tcp3 * dq2dt)
 
@gtscript.function
def get_factor(wqsat, qv, fac_l2v):
    factor = fac_l2v * 10. * (1. - qv / wqsat)
    factor = -1 if 1 < factor else -factor  # min_fn(1, factor) * -1
    return factor
 
@gtscript.function
def get_src(ql, factor, dq0):
    src = -ql if ql < factor * dq0 else -factor*dq0  # min_func(ql, factor * dq0) * -1
    return src

@gtscript.function
def ql_evaporation(wqsat, qv, ql, dq0,fac_l2v):
    factor =  get_factor(wqsat, qv, fac_l2v)
    src = get_src(ql, factor, dq0)
    return factor, src

@gtscript.function
def wqsat_correct(src, pt1, lhl, qv, ql, q_liq, q_sol, mc_air, c_vap, c_liq, c_ice):
    qv = qv - src
    ql = ql + src
    q_liq = q_liq + src
    cvm = compute_cvm(mc_air, qv, c_vap, q_liq, c_liq, q_sol, c_ice)
    pt1 = adjust_pt1(pt1, src, lhl, cvm) # pt1 + src * lhl / cvm
    return qv, ql, q_liq, cvm, pt1




@gtscript.function
def min_fn(a, b):
    return a if a < b else b


@gtscript.function
def max_fn(a, b):
    return a if a > b else a


def numpy_dim(a, b):
    diff = a - b
    diff[diff < 0] = 0
    return diff


# it's not clear yet how this can be in a stencil
#  gradient of saturated specific humidity for table ii.
# TODO oh so bad, how to put in gt4py
def wqs2_vect(ta, den):
    ap1 = 10.0 * numpy_dim(ta.data, TMIN) + 1.0
    min_index = np.zeros(ap1.shape) + 2620
    ap1 = np.minimum(ap1, min_index)
    it = ap1
    es = np.zeros(ap1.shape)
    for i in range(es.shape[0]):
        for j in range(es.shape[1]):
            for k in range(es.shape[2]):
                it = int(ap1[i, j, k])
                es[i, j, k] = satmix['tablew'][it] + (ap1[i, j, k] - it) * satmix['desw'][it]
    wqsat = es / (constants.RVGAS * ta.data * den.data)
    
    # finite diff, del_t = 0.1:
    dqdt = np.zeros(ap1.shape)
    for i in range(es.shape[0]):
        for j in range(es.shape[1]):
            for k in range(es.shape[2]):
                it = int(ap1[i, j, k] - 0.5)
                dqdt[i, j, k] = 10.0 * (satmix['desw'][it] + (ap1[i, j, k] - it) * (satmix['desw'][it + 1] - satmix['desw'][it]))
    dqdt = dqdt / (constants.RVGAS * ta.data * den.data)
    
    return wqsat, dqdt


@utils.stencil(externals={'cv_air': constants.CV_AIR, 'cv_vap': constants.CV_VAP, 'c_liq': constants.C_LIQ, 'c_ice': constants.C_ICE, 'rdgas': constants.RDGAS, 'rvgas': constants.RVGAS, 'grav': constants.GRAV, 'tice':constants.TICE})
def satadjust_part1(dpln: sd, den: sd, pt1: sd, cvm:sd, mc_air: sd, peln: sd, qv:sd, ql: sd, q_liquid:sd, qi: sd, qr: sd, qs: sd, q_sol:sd, qg: sd,
              pt: sd, dp: sd, delz: sd, te0: sd, zvir: float, hydrostatic: bool, consv_te: bool, c_air: float, c_vap: float, fac_imlt: float, d0_vap: float, lv00: float):
    from __externals__ import rdgas, rvgas, grav, c_liq, c_ice
    with computation(PARALLEL), interval(...):
        dpln = peln[0, 0, 1] - peln
        q_liq = ql + qr
        q_sol = qi + qs + qg
        qpz = q_liq + q_sol
        pt1 = pt / ((1 + zvir * qv) * (1 - qpz))
        t0 = pt1  # true temperature
        qpz = qpz + qv  # total_wat conserved in this routine
        # define air density based on hydrostatical property
        den = dp / (dpln * rdgas * pt) if hydrostatic else -dp / (grav * delz)
        # define heat capacity and latend heat coefficient
        mc_air = (1. - qpz) * c_air
        cvm = compute_cvm(mc_air, qv, c_vap, q_liq, c_liq, q_sol, c_ice)
        lhi = LI00 + DC_ICE * pt1
        icp2 = lhi / cvm
        #  fix energy conservation
        # TODO could really use if blocks here
        te0_consv = -c_air * t0 if hydrostatic else -cvm * t0
        te0 = te0_consv if consv_te else te0
        # fix negative cloud ice with snow
        qs = qs + qi if qi < 0 else qs
        qi = 0. if qi < 0 else qi
        #  melting of cloud ice to cloud water and rain
        qi, ql, q_liq, q_sol, cvm, pt1 = melt_cloud_ice(qv, qi, ql, q_liq, q_sol, pt1, icp2, fac_imlt, mc_air, c_vap, c_liq, c_ice, lhi, cvm)
        # update latend heat coefficient
        lhi = LI00 + DC_ICE * pt1
        icp2 = lhi / cvm
        # fix negative snow with graupel or graupel with available snow
        qs, qg = fix_negative_snow(qs, qg)
        # after this point cloud ice & snow are positive definite
        # fix negative cloud water with rain or rain with available cloud water
        ql, qr = fix_negative_cloud_water(ql, qr)
        # enforce complete freezing of cloud water to cloud ice below - 48 c
        ql, qi, q_liq, q_sol, cvm, pt1 = complete_freezing(qv, ql, qi, q_liq, q_sol, pt1, cvm, icp2, mc_air, lhi, c_vap, c_ice, c_liq)
        ## update latent heat coefficient
        #lhl, lhi, lcp2, icp2 = update_latent_heat_coefficient(pt1, cvm, lv00, d0_vap)
        #diff_ice = dim(TICE, pt1) / 48.0
        #dimmin = 1.0 if 1.0 < diff_ice else diff_ice
        #tcp3 = lcp2 + icp2 * dimmin
        # condensation / evaporation between water vapor and cloud water
        # wqsat, dqdt = wqs2_vect(pt1, den, rvgas)
       

@utils.stencil(externals={'cv_air': constants.CV_AIR, 'cv_vap': constants.CV_VAP, 'c_liq': constants.C_LIQ, 'c_ice': constants.C_ICE,
                          'rdgas': constants.RDGAS, 'rvgas': constants.RVGAS, 'grav': constants.GRAV, 'tice':constants.TICE, 'sat_adj0': constants.sat_adj0,
                          'ql_gen': constants.ql_gen
})
def satadjust_part2(wqsat:sd, dq2dt:sd, pt1:sd, cvm: sd, mc_air:sd, tcp3: sd, lhl:sd, lhi:sd, lcp2:sd, icp2:sd, qv: sd, ql:sd, q_liq:sd, q_sol:sd, fac_v2l:float, fac_l2v: float, lv00:float, d0_vap: float, c_vap: float):
    from __externals__ import rdgas, rvgas, grav, c_liq, c_ice, sat_adj0, ql_gen
    with computation(PARALLEL), interval(...):
        # update latent heat coefficient
        lhl, lhi, lcp2, icp2 = update_latent_heat_coefficient(pt1, cvm, lv00, d0_vap)
      
        diff_ice = dim(TICE, pt1) / 48.0
        dimmin = min_fn(1.0, diff_ice)
        tcp3 = lcp2 + icp2 * dimmin
        adj_fac = sat_adj0
        dq0 = compute_dq0(qv, wqsat, dq2dt, tcp3)  #(qv - wqsat) / (1.0 + tcp3 * dq2dt)
        # TODO might be able to get rid of these temporary allocations when not used? 
        tmpmax = 0.
        src = 0.
        factor = 0.
        a=0.
        b=0.
        if (dq0 > 0):  # whole grid - box saturated
            a = ql_gen - ql
            b = fac_v2l * dq0
            tmpmax = a if a > b else b # max_fn(a, b)
            src = adj_fac * dq0 if adj_fac * dq0 <  tmpmax else tmpmax # min_func(adj_fac * dq0, tmpmax)
        else:
            # TODO -- we'd like to use this abstraction rather than duplicate code, but inside the if conditional complains 'not implemented'
            #factor, src = ql_evaporation(wqsat, qv, ql, dq0,fac_l2v)
            factor = fac_l2v * 10. * (1. - qv / wqsat)
            factor = -1 if 1 < factor else -factor  # min_fn(1, factor) * -1
            src = -ql if ql < factor * dq0 else -factor*dq0  # min_func(ql, factor * dq0) * -1
        qv, ql, q_liq, cvm, pt1 = wqsat_correct(src, pt1, lhl, qv, ql, q_liq, q_sol, mc_air, c_vap, c_liq, c_ice)
        #qv = qv - src
        #ql = ql + src
        #q_liq = q_liq + src
        #cvm = compute_cvm(mc_air, qv, c_vap, q_liq, c_liq, q_sol, c_ice)
        #pt1 = adjust_pt1(pt1, src, lhl, cvm) # pt1 + src * lhl / cvm
        # update latent heat coefficient
        lhl, lhi, lcp2, icp2 = update_latent_heat_coefficient(pt1, cvm, lv00, d0_vap)
        # TODO uncomment when exp and log are supported
        # pkz = exp(cappa * log(rrg * delp / delz * pt)) #rrg = constants.RDG

@utils.stencil(externals={'cv_air': constants.CV_AIR, 'cv_vap': constants.CV_VAP, 'c_liq': constants.C_LIQ, 'c_ice': constants.C_ICE,
                          'rdgas': constants.RDGAS, 'rvgas': constants.RVGAS, 'grav': constants.GRAV, 'tice':constants.TICE, 'sat_adj0': constants.sat_adj0,
                          'ql_gen': constants.ql_gen
})
def satadjust_part3(wqsat: sd, dq2dt: sd, pt1: sd, cvm: sd, mc_air: sd, tcp3: sd, lhl:sd, lhi:sd, lcp2:sd, icp2:sd, last_step:bool, qv: sd, ql:sd, q_liq:sd, qi:sd, q_sol: sd, fac_v2l: float,
                    fac_l2v: float, lv00: float, d0_vap: float, c_vap: float):
    from __externals__ import rdgas, rvgas, grav, c_liq, c_ice, sat_adj0, ql_gen
    with computation(PARALLEL), interval(...):
        dq0 = 0.
        src = 0.
        factor = 0.
        adj_fac = sat_adj0   
        if last_step:
            dq0 = compute_dq0(qv, wqsat, dq2dt, tcp3)
            if dq0 > 0:
                src = dq0
            else:
                # TODO -- we'd like to use this abstraction rather than duplicate code, but inside the if conditional complains 'not implemented'
                #factor, src = ql_evaporation(wqsat, qv, ql, dq0,fac_l2v)
                factor = fac_l2v * 10. * (1. - qv / wqsat)
                factor = -1 if 1 < factor else -factor  # min_fn(1, factor) * -1
                src = -ql if ql < factor * dq0 else -factor*dq0  # min_func(ql, factor * dq0) * -1
            adj_fac = 1.0
            # TOD this is all repeated, but not implemented error 
            #qv, ql, q_liq, cvm, pt1 = wqsat_correct(src, pt1, lhl, qv, ql, q_liq, q_sol, mc_air, c_vap, c_liq, c_ice)
            #lhl, lhi, lcp2, icp2 = update_latent_heat_coefficient(pt1, cvm, lv00, d0_vap)
            qv = qv - src
            ql = ql + src
            q_liq = q_liq + src
            # TODO WHY does cvm not work but adjust_pt1 does? new = vs function of self?
            #cvm = compute_cvm(mc_air, qv, c_vap, q_liq, c_liq, q_sol, c_ice)
            cvm = mc_air + qv * c_vap + q_liq * c_liq + q_sol * c_ice
            pt1 = adjust_pt1(pt1, src, lhl, cvm) # pt1 + src * lhl / cvm
            #lhl, lhi, lcp2, icp2 = update_latent_heat_coefficient(pt1, cvm, lv00, d0_vap)
            lhl = lv00 + d0_vap * pt1
            lhi = LI00 + DC_ICE * pt1
            lcp2 = lhl / cvm
            icp2 = lhi / cvm
        else:
            adj_fac = sat_adj0
        # homogeneous freezing of cloud water to cloud ice
        ql, qi, q_liq, q_sol, cvm, pt1 = homogenous_freezing(qv, ql, qi, q_liq, q_sol, pt1, cvm, icp2, mc_air, lhi, c_vap, c_ice, c_liq)
        # update some of the latent heat coefficients
    
        lhi, icp2 = update_latent_heat_coefficient_i(pt1, cvm)
        
@utils.stencil(externals={'cv_air': constants.CV_AIR, 'cv_vap': constants.CV_VAP, 'c_liq': constants.C_LIQ, 'c_ice': constants.C_ICE,
                          'rdgas': constants.RDGAS, 'rvgas': constants.RVGAS, 'grav': constants.GRAV, 'tice':constants.TICE, 'sat_adj0': constants.sat_adj0,
                          'ql_gen': constants.ql_gen, 'qs_mlt': constants.qs_mlt, 'ql0_max': constants.ql0_max
})
def satadjust_part4(wqsat: sd, dq2dt: sd, den:sd, pt1: sd, cvm: sd, mc_air: sd, tcp3: sd, lhl:sd, lhi:sd, lcp2:sd, icp2:sd, exptc:sd, last_step:bool, qv: sd, ql:sd, q_liq:sd, qi:sd, q_sol:sd, qr:sd, qg:sd, qs:sd, fac_v2l:float, fac_l2v: float, lv00:float, d0_vap: float, c_vap: float, mdt:float, fac_r2g: float, fac_smlt: float, fac_l2r:float):
    from __externals__ import rdgas, rvgas, grav, c_liq, c_ice, sat_adj0, ql_gen, qs_mlt, ql0_max
    with computation(PARALLEL), interval(...):
        # bigg mechanism (heterogeneous freezing of cloud water to cloud ice)
        ql, qi, q_liq, q_sol, cvm, pt1 = heterogeneous_freezing(exptc, pt1, cvm, ql, qi, q_liq, q_sol, den, icp2, mdt, mc_air, lhi, qv, c_vap, c_liq, c_ice)
        lhi, icp2 = update_latent_heat_coefficient_i(pt1, cvm)
        # freezing of rain to graupel
        qr, qg , q_liq, q_sol, cvm, pt1 = make_graupel(pt1, cvm, fac_r2g, qr, qg, q_liq, q_sol, lhi, icp2,  mc_air, qv, c_vap, c_liq, c_ice)
        lhi, icp2 = update_latent_heat_coefficient_i(pt1, cvm)
        # melting of snow to rain or cloud water
        qs, ql, qr, q_liq, q_sol, cvm, pt1= melt_snow(pt1, cvm, fac_smlt, qs, ql, qr, q_liq, q_sol, lhi, icp2, mc_air, qv, c_vap, c_liq, c_ice, qs_mlt)
        #  autoconversion from cloud water to rain
        ql, qr = autoconversion_cloud_to_rain(ql, qr, fac_l2r, ql0_max)
        lhl, lhi, lcp2, icp2 = update_latent_heat_coefficient(pt1, cvm, lv00, d0_vap)
        tcp2 = lcp2 + icp2
        # sublimation / deposition between water vapor and cloud ice
        #qv, qi, q_sol, cvm, pt1 = sublimation_deposition_vapor_ice(pt1, cvm, qv, den, dqsdt, tcp2, )

def compute(dpln, te, qvapor, qliquid, qice, qrain, qsnow, qgraupel, qcld, hs, peln, delp, delz, q_con, pt, pkz, cappa, r_vir, mdt, fast_mp_consv, out_dt, last_step, akap, kmp):
    grid = spec.grid
    qs_init()
    hydrostatic = spec.namelist['hydrostatic']
    sdt = 0.5 * mdt # half remapping time step
    dt_bigg = mdt # bigg mechinism time step
    tice0 = TICE - 0.01 # 273.15, standard freezing temperature
    # define conversion scalar / factor
    #delnadjust()
    fac_i2s = 1. - math.exp(-mdt / constants.tau_i2s)
    fac_v2l = 1. - math.exp(-sdt / constants.tau_v2l)
    fac_r2g = 1. - math.exp(-mdt / constants.tau_r2g)
    fac_l2r = 1. - math.exp(-mdt / constants.tau_l2r)
    
    fac_l2v = 1. - math.exp(-sdt / constants.tau_l2v)
    fac_l2v = min(constants.sat_adj0, fac_l2v)
    
    fac_imlt = 1. - math.exp(-sdt / constants.tau_imlt)
    fac_smlt = 1. - math.exp(-mdt / constants.tau_smlt)
    
    # define heat capacity of dry air and water vapor based on hydrostatical property
  
    if (hydrostatic):
        c_air = constants.CP_AIR
        c_vap = constants.CP_VAP
    else:
        c_air = constants.CV_AIR
        c_vap = constants.CV_VAP
    
    d0_vap = c_vap - constants.C_LIQ
    lv00 = constants.HLV - d0_vap * TICE
    # temporaries needed for wqs2_vect, and then for passing info before and after this call
    den = utils.make_storage_from_shape(peln.shape, utils.origin)
    pt1 = utils.make_storage_from_shape(peln.shape, utils.origin)
    cvm = utils.make_storage_from_shape(peln.shape, utils.origin)
    q_liq = utils.make_storage_from_shape(peln.shape, utils.origin)
    mc_air = utils.make_storage_from_shape(peln.shape, utils.origin)
    q_sol = utils.make_storage_from_shape(peln.shape, utils.origin)
    tcp3 = utils.make_storage_from_shape(peln.shape, utils.origin)
    lhl = utils.make_storage_from_shape(peln.shape, utils.origin)
    lhi = utils.make_storage_from_shape(peln.shape, utils.origin)
    lcp2 = utils.make_storage_from_shape(peln.shape, utils.origin)
    icp2 = utils.make_storage_from_shape(peln.shape, utils.origin)
    #lhi:sd, icp2:sd, lcp2:sd,
    satadjust_part1(dpln, den, pt1, cvm, mc_air, peln, qvapor, qliquid, q_liq, qice, qrain, qsnow, q_sol, qgraupel,
              pt, delp, delz, te, r_vir, hydrostatic, fast_mp_consv, c_air, c_vap, fac_imlt, d0_vap, lv00,
              origin=grid.compute_origin(), domain=grid.domain_shape_compute()
    )
    
    wqsat, dq2dt = wqs2_vect(pt1, den)
    wqsat = utils.make_storage_data(wqsat, peln.shape, origin=utils.origin)
    dq2dt = utils.make_storage_data(dq2dt, peln.shape, origin=utils.origin)
    satadjust_part2(wqsat, dq2dt, pt1, cvm, mc_air, tcp3, lhl, lhi, lcp2, icp2, qvapor, qliquid, q_liq, q_sol, fac_v2l, fac_l2v, lv00, d0_vap, c_vap, origin=grid.compute_origin(), domain=grid.domain_shape_compute())
    if last_step:
        #condensation / evaporation between water vapor and cloud water, last time step
        #  enforce upper (no super_sat) & lower (critical rh) bounds
        # final iteration:
        wqsat, dq2dt = wqs2_vect(pt1, den)
        wqsat = utils.make_storage_data(wqsat, peln.shape, origin=utils.origin)
        dq2dt = utils.make_storage_data(dq2dt, peln.shape, origin=utils.origin)
    satadjust_part3(wqsat, dq2dt, pt1, cvm, mc_air, tcp3, lhl,lhi, lcp2, icp2,last_step, qvapor, qliquid, q_liq, qice, q_sol, fac_v2l,
                    fac_l2v, lv00, d0_vap, c_vap, origin=grid.compute_origin(), domain=grid.domain_shape_compute())
    exptc = np.exp(0.66 *( TICE0 - pt1))
    satadjust_part4(wqsat, dq2dt, den, pt1, cvm, mc_air, tcp3, lhl,  lhi, lcp2, icp2, exptc, last_step, qvapor, qliquid, q_liq, qice, q_sol, qrain, qgraupel, qsnow, fac_v2l, fac_l2v, lv00, d0_vap, c_vap, mdt, fac_r2g, fac_smlt, fac_l2r, origin=grid.compute_origin(), domain=grid.domain_shape_compute())
    # TODO put into stencil when exp allowed inside stencil
    tmpslice = (slice(grid.is_, grid.ie + 1), slice(grid.js, grid.je+1), slice(kmp, grid.npz))
    pkz[tmpslice] = np.exp(cappa[tmpslice]*np.log(constants.RDG*delp[tmpslice]/delz[tmpslice]*pt[tmpslice]))
