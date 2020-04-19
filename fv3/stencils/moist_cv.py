import fv3.utils.gt4py_utils as utils
from fv3.utils.corners import fill2_4corners, fill_4corners
import gt4py.gtscript as gtscript
import fv3._config as spec
from gt4py.gtscript import computation, interval, PARALLEL
import fv3.stencils.copy_stencil as cp
import fv3.stencils.cs_profile as cs_profile
import fv3.utils.global_constants as constants
# import fv3.stencils.ppm_profile as ppm_profile
import numpy as np

sd = utils.sd

@gtscript.function
def set_cappa(qvapor, cvm, r_vir, rdgas):
    cappa = rdgas / (rdgas + cvm / (1.0 + r_vir * qvapor))
    return cappa
@gtscript.function
def moist_cvm(qvapor, gz, ql, qs, cv_air, cv_vap, c_liq, c_ice):
    cvm = (1.0 - (qvapor + gz)) * cv_air + qvapor * cv_vap + ql * c_liq + qs * c_ice
    return cvm

@gtscript.function
def moist_cv_nwat6_fn(qvapor, qliquid, qrain, qsnow, qice, qgraupel, cv_air, cv_vap, c_liq, c_ice):
    ql = qliquid + qrain
    qs = qice + qsnow + qgraupel
    gz = ql + qs
    cvm = moist_cvm(qvapor, gz, ql, qs, cv_air, cv_vap, c_liq, c_ice)
    return cvm, gz

# TODO : note untested
@gtscript.function
def moist_cv_nwat5_fn(qvapor, qliquid, qrain, qsnow, qice, cv_air, cv_vap, c_liq, c_ice):
    ql = qliquid + qrain
    qs = qice + qsnow
    gz = ql + qs
    cvm = moist_cvm(qvapor, gz, ql, qs, cv_air, cv_vap, c_liq, c_ice)
    return cvm, gz

# TODO : note untested
@gtscript.function
def moist_cv_nwat4_fn(qvapor, qliquid, qrain, cv_air, cv_vap, c_liq):
    gz = qliquid + qrain
    cvm = moist_cvm(qvapor, gz, gz, gz, cv_air, cv_vap, c_liq, 0)
    return cvm, gz

# TODO : note untested
@gtscript.function
def moist_cv_nwat3_fn(qvapor, qliquid, qice, cv_air, cv_vap, c_liq, c_ice):
    gz = qliquid + qice
    cvm = moist_cvm(qvapor, gz, qliquid, qice, cv_air, cv_vap, c_liq, c_ice)
    return cvm, gz

# TODO : note untested
@gtscript.function
def moist_cv_nwat2_fn(qvapor, qliquid, cv_air, cv_vap):
    qv = qvapor if qvapor > 0 else 0.0
    qs = qliquid if qliquid > 0 else 0.0
    gz = qs
    cvm = (1.0 - qv) * cv_air + qv * cv_vap
    return cvm, gz

# TODO : note untested
@gtscript.function
def moist_cv_nwat2_gfs_fn(qvapor, qliquid, cv_air, cv_vap, c_liq, c_ice, t1, tice): #note constants.TICE
    gz = qliquid if qliquid > 0 else 0.0
    qtmp = gz if t1 < tice - 15.0 else gz * (tice - t1) / 15.0
    qs = 0 if t1 > tice else qtmp
    ql = gz - qs
    qv = qvapor if qvapor > 0 else 0.0
    cvm = moist_cvm(qv, gz, ql, qs, cv_air, cv_vap, c_liq, c_ice)
    return cvm, gz

# TODO : note untested
@gtscript.function
def moist_cv_default_fn(cv_air):
    gz = 0
    cvm = cv_air
    return cvm, gz

@utils.stencil(externals={'cv_air': constants.CV_AIR, 'cv_vap': constants.CV_VAP, 'c_liq': constants.C_LIQ, 'c_ice': constants.C_ICE})
def moist_cv_nwat6(qvapor: sd, qliquid: sd, qrain: sd, qsnow: sd,qice: sd,  qgraupel: sd, cvm: sd):
    from __externals__ import cv_air, cv_vap, c_liq, c_ice
    with computation(PARALLEL), interval(...):
        cvm = moist_cv_nwat6_fn(qvapor, qliquid, qrain, qsnow, qice, qgraupel, cv_air, cv_vap, c_liq, c_ice)
       
@utils.stencil(externals={'cv_air': constants.CV_AIR, 'cv_vap': constants.CV_VAP, 'c_liq': constants.C_LIQ, 'c_ice': constants.C_ICE})
def moist_te_2d(qvapor: sd, qliquid: sd, qrain: sd, qsnow: sd, qice: sd, qgraupel: sd, q_con: sd, gz: sd, cvm: sd, te_2d: sd, delp: sd, pt: sd, phis: sd, u: sd, v: sd, w: sd, rsin2: sd, cosa_s: sd, r_vir: float, nwat: int):
    from __externals__ import cv_air, cv_vap, c_liq, c_ice
    with computation(FORWARD), interval(...):
        cvm, gz = moist_cv_nwat6_fn(qvapor, qliquid, qrain, qsnow, qice, qgraupel, cv_air, cv_vap, c_liq, c_ice) #if (nwat == 6) else moist_cv_default_fn(cv_air)
        q_con[0, 0, 0] = gz
        te_2d[0, 0, 0] = te_2d[0, 0, -1] + delp * (cvm * pt / ((1.0 + r_vir * qvapor) * (1.0 - gz)) +
                                     0.5 * (phis + phis[0, 0, 1] + w**2 + 0.5 * rsin2 *
                                            (u**2 + u[0, 1, 0]**2 + v**2 + v[1, 0, 0]**2 -
                                             (u + u[0, 1, 0]) * (v + v[1, 0, 0]) * cosa_s)))

@utils.stencil(externals={'cv_air': constants.CV_AIR, 'cv_vap': constants.CV_VAP, 'c_liq': constants.C_LIQ, 'c_ice': constants.C_ICE, 'rdgas': constants.RDGAS, 'rrg': constants.RDG})
def moist_pt(qvapor: sd, qliquid: sd, qrain: sd, qsnow: sd,qice: sd, qgraupel: sd, q_con: sd, gz: sd, cvm: sd, pt: sd, cappa: sd, delp: sd, delz: sd, r_vir: float, nwat: int):
    from __externals__ import cv_air, cv_vap, c_liq, c_ice, rdgas, rrg
    with computation(FORWARD), interval(...):
        cvm, gz = moist_cv_nwat6_fn(qvapor, qliquid, qrain, qsnow, qice, qgraupel, cv_air, cv_vap, c_liq, c_ice) #if (nwat == 6) else moist_cv_default_fn(cv_air)
        q_con[0, 0, 0] = gz
        cappa = set_cappa(qvapor, cvm, r_vir, rdgas)
        #pt[0, 0, 0] = pt * exp(cappa / (1.0 - cappa) * log(rrg * delp / delz * pt))

@utils.stencil(externals={'cv_air': constants.CV_AIR, 'cv_vap': constants.CV_VAP, 'c_liq': constants.C_LIQ, 'c_ice': constants.C_ICE, 'rdgas': constants.RDGAS, 'rrg': constants.RDG})
def moist_pkz(qvapor: sd, qliquid: sd, qrain: sd, qsnow: sd,qice: sd, qgraupel: sd, q_con: sd, gz: sd, cvm: sd, pkz: sd, pt:sd, cappa: sd, delp: sd, delz: sd, r_vir: float, nwat: int):
    from __externals__ import cv_air, cv_vap, c_liq, c_ice, rdgas, rrg
    with computation(FORWARD), interval(...):
        cvm, gz = moist_cv_nwat6_fn(qvapor, qliquid, qrain, qsnow, qice, qgraupel, cv_air, cv_vap, c_liq, c_ice) #if (nwat == 6) else moist_cv_default_fn(cv_air)
        q_con[0, 0, 0] = gz
        cappa = set_cappa(qvapor, cvm, r_vir, rdgas)
        #pkz[0, 0, 0] = exp(cappa * log(rrg * delp /delz * pt))
# 
# Computes the FV3-consistent moist heat capacity under constant volume,
# including the heating capacity of water vapor and condensates.
# See emanuel1994atmospheric for information on variable heat capacities.
   
# assumes 3d variables are indexed to j
def compute_te(qvapor_js, qliquid_js, qice_js, qrain_js, qsnow_js, qgraupel_js, te_2d, gz, cvm, delp, q_con, pt, phis, w, u, v, r_vir, j_2d):
    grid = spec.grid
    
    nwat = spec.namelist['nwat']
    if nwat != 6: #TODO -- to do this cleanly, we probably need if blocks working inside stencils
        raise Exception('We still need to implement other nwats for moist_cv')
    moist_te_2d(qvapor_js, qliquid_js, qrain_js, qsnow_js, qice_js, qgraupel_js,
                q_con, gz, cvm, te_2d, delp, pt, phis, u, v, w,
                grid.rsin2, grid.cosa_s, r_vir, spec.namelist['nwat'],
                origin=(grid.is_, j_2d, 0), domain=(grid.nic, 1, grid.npz)
    )

def compute_pt(qvapor_js, qliquid_js, qice_js, qrain_js, qsnow_js, qgraupel_js, q_con, gz, cvm, pt, cappa, delp, delz, r_vir, j_2d):
    grid = spec.grid
    moist_pt(qvapor_js, qliquid_js, qrain_js, qsnow_js, qice_js, qgraupel_js,
             q_con, gz, cvm, pt, cappa, delp, delz, r_vir, spec.namelist['nwat'],
             origin=(grid.is_, j_2d, 0), domain=(grid.nic, 1, grid.npz)
    )
    # TODO push theis inside stencil one we can do exp and log there
    tmpslice = (slice(grid.is_, grid.ie + 1), slice(j_2d, j_2d + 1), slice(0, grid.npz))
    pt[tmpslice] = pt[tmpslice] * np.exp(cappa[tmpslice] / (1.0 - cappa[tmpslice]) * np.log(constants.RDG * delp[tmpslice] / delz[tmpslice] * pt[tmpslice]))
   
def compute_pkz(qvapor_js, qliquid_js, qice_js, qrain_js, qsnow_js, qgraupel_js, q_con, gz, cvm, pkz, pt, cappa, delp, delz, r_vir, j_2d):
    grid = spec.grid
    moist_pkz(qvapor_js, qliquid_js, qrain_js, qsnow_js, qice_js, qgraupel_js,
              q_con, gz, cvm, pkz, pt, cappa, delp, delz, r_vir, spec.namelist['nwat'],
              origin=(grid.is_, j_2d, 0), domain=(grid.nic, 1, grid.npz)
    )
    # TODO push theis inside stencil one we can do exp and log there
    tmpslice = (slice(grid.is_, grid.ie + 1), slice(j_2d, j_2d + 1), slice(0, grid.npz))
    pkz[tmpslice] = np.exp(cappa[tmpslice] * np.log(constants.RDG * delp[tmpslice] / delz[tmpslice] * pt[tmpslice]))
