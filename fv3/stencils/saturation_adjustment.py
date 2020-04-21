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
TICE = constants.TICE
E8 = 1.0e-8
# melting of cloud ice to cloud water and rain
# TODO, when if blocks are possible,, only compute when 'melting'

@gtscript.function
def sidestep_and(cond1, cond2, not_both_true, both_true):
    tmp = both_true if cond1 == 1 else not_both_true
    result = tmp if cond2 == 1 else not_both_true
    return result
@gtscript.function
def melt_cloud_ice(qv, qi, ql, q_liq, q_sol, pt1, icp2, fac_imlt, mc_air, c_vap, c_liq, c_ice, lhi, cvm):
    factmp = fac_imlt * (pt1 - TICE) / icp2
    sink = qi if qi < factmp else factmp
    qim = qi - sink
    qlm = ql + sink
    q_liqm = q_liq + sink
    q_solm = q_sol - sink
    cvmm = mc_air + qv * c_vap + q_liq * c_liq + q_sol * c_ice
    pt1m = pt1 - sink * lhi / cvm
    #return qim, qlm, q_liqm, q_solm, cvmm, pt1m
    #melting = ((qi > E8)  and (pt1 > TICE))
    #qi = qim if melting else qi
    #ql = qlm if melting else ql
    #q_liq = q_liqm if melting else q_liq
    #q_sol = q_solm if melting else q_sol
    #cvm = cvmm if melting else cvm
    #pt1 = pt1m if melting else pt1
    cond1 = (qi > 1.e-8)
    cond2 = (pt1 > TICE)
    qi = sidestep_and(cond1, cond2, qi, qim)
    ql = sidestep_and(cond1, cond2, ql, qlm)
    q_liq = sidestep_and(cond1, cond2, q_liq, q_liqm)
    q_sol = sidestep_and(cond1, cond2, q_sol, q_solm)
    cvm = sidestep_and(cond1, cond2, cvm, cvmm)
    pt1 = sidestep_and(cond1, cond2, pt1, pt1m)
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
    tmp =  minmax_tmp_h20(qg, qs)
    qg_neg_qg = qg + tmp if qg < 0 else qg
    qs_neg_qg = qs - tmp if qg < 0 else qs
    qg = qg + qs if qs < 0 else qg_neg_qg
    qs = 0 if qs < 0 else qs_neg_qg
    return qs, qg

# fix negative cloud water with rain or rain with available cloud water
# TODO fix so only compute tmp when it is needed
@gtscript.function
def fix_negative_cloud_water(ql, qr):
    tmpl = minmax_tmp_h20(ql, qr)
    tmpr = minmax_tmp_h20(qr, ql)
    ql_neg_qr = ql - tmpr if qr < 0 else ql
    qr_neg_qr = qr + tmpr if qr < 0 else qr
    ql = ql + tmpl if ql < 0 else ql_neg_qr
    qr = qr - tmpl if ql < 0 else qr_neg_qr
    return ql, qr

# enforce complete freezing of cloud water to cloud ice below - 48 c
@gtscript.function
def complete_freezing(qv, ql, qi, q_liq, q_sol, pt1, cvm, icp2, mc_air, lhi, c_vap, c_ice, c_liq):
    dtmp = TICE - 48. - pt1
    #needed = (ql > 0. and dtmp > 0.)
    cond1 = (ql > 0.)
    cond2 = (dtmp > 0.)
    sink = ql if ql < dtmp / icp2 else dtmp / icp2
    ql = sidestep_and(cond1, cond2, ql, ql - sink)  # ql - sink if needed else ql
    qi = sidestep_and(cond1, cond2, qi, qi + sink)  # qi + sink if needed else qi
    q_liq = sidestep_and(cond1, cond2, q_liq, q_liq - sink)  # q_liq - sink if needed else q_liq
    q_sol = sidestep_and(cond1, cond2, q_sol, q_sol + sink)  # q_sol + sink if needed else q_sol 
    cvm = sidestep_and(cond1, cond2, cvm, mc_air + qv * c_vap + q_liq * c_liq + q_sol * c_ice)  # mc_air + qv * c_vap + q_liq * c_liq + q_sol * c_ice if needed else cvm
    pt1 = sidestep_and(cond1, cond2, pt1, pt1 + sink * lhi / cvm )  #  pt1 + sink * lhi / cvm if needed else pt1
    return ql, qi, q_liq, q_sol, cvm, pt1

#
@gtscript.function
def dim(a, b):
    diff = a - b if a - b > 0 else 0
    return diff

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
def satadjust_part1(dpln: sd, den: sd, pt1: sd, peln: sd, qv:sd, ql: sd, qi: sd, qr: sd, qs: sd, qg: sd,
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
        cvm = mc_air + qv * c_vap + q_liq * c_liq + q_sol * c_ice
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
        # update latend heat coefficient
        lhl = lv00 + d0_vap * pt1
        lhi = LI00 + DC_ICE * pt1
        lcp2 = lhl / cvm
        icp2 = lhi / cvm
        diff_ice = dim(TICE, pt1) / 48.0
        dimmin = 1.0 if 1.0 < diff_ice else diff_ice
        tcp3 = lcp2 + icp2 * dimmin
        # condensation / evaporation between water vapor and cloud water
        # wqsat, dqdt = wqs2_vect(pt1, den, rvgas)
        # TODO uncomment when exp and log are supported
        # pkz = exp(cappa * log(rrg * delp / delz * pt)) #rrg = constants.RDG

#@utils.stencil(externals={'cv_air': constants.CV_AIR, 'cv_vap': constants.CV_VAP, 'c_liq': constants.C_LIQ, 'c_ice': constants.C_ICE, 'rdgas': constants.RDGAS, 'rvgas'#: constants.RVGAS, 'grav': constants.GRAV, 'tice':constants.TICE})
#def satadjust_part2(wqsat:sd, dqdt:sd, dpln: sd, den: sd, pt1: sd, peln: sd, qv:sd, ql: sd, qi: sd, qr: sd, qs: sd, qg: sd,
#              pt: sd, dp: sd, delz: sd, te0: sd, zvir: float, hydrostatic: bool, consv_te: bool, c_air: float, c_vap: float, fac_imlt: float, d0_vap: float, lv00: float):
#    from __externals__ import rdgas, rvgas, grav, c_    
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
    # temporaries needed for wqs2_vect
    den = utils.make_storage_from_shape(peln.shape, utils.origin)
    pt1 = utils.make_storage_from_shape(peln.shape, utils.origin)
    satadjust_part1(dpln, den, pt1, peln, qvapor, qliquid, qice, qrain, qsnow, qgraupel,
              pt, delp, delz, te, r_vir, hydrostatic, fast_mp_consv, c_air, c_vap, fac_imlt, d0_vap, lv00,
              origin=grid.compute_origin(), domain=grid.domain_shape_compute()
    )
    
    wqsat, dqdt = wqs2_vect(pt1, den)
    wqsat = utils.make_storage_data(wqsat, peln.shape, origin=utils.origin)
    dqdt = utils.make_storage_data(dqdt, peln.shape, origin=utils.origin)
    # TODO put into stencil when exp allowed inside stencil
    tmpslice = (slice(grid.is_, grid.ie + 1), slice(grid.js, grid.je+1), slice(kmp, grid.npz))
    pkz[tmpslice] = np.exp(cappa[tmpslice]*np.log(constants.RDG*delp[tmpslice]/delz[tmpslice]*pt[tmpslice]))
