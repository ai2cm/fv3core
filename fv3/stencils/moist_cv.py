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


def grid():
    return spec.grid


@utils.stencil()
def set_locals(dp1: sd, q4_1: sd, pe1: sd, q1: sd):
    with computation(PARALLEL), interval(...):
        dp1 = pe1[0, 0, 1] - pe1
        q4_1 = q1

@gtscript.function
def moist_cv_nwat6_fn(qvapor, qliquid, qrain, qsnow, qice, qgraupel, cv_air, cv_vap, c_liq, c_ice):
    ql = qliquid + qrain
    qs = qice + qsnow + qgraupel
    qd = ql + qs
    cvm = (1.0 - (qvapor + qd)) * cv_air + qvapor * cv_vap + ql * c_liq + qs * c_ice
    return cvm

@utils.stencil(externals={'cv_air': constants.CV_AIR, 'cv_vap': constants.CV_VAP, 'c_liq': constants.C_LIQ, 'c_ice': constants.C_ICE})
def moist_cv_nwat6(qvapor: sd, qliquid: sd, qrain: sd, qsnow: sd,qice: sd,  qgraupel: sd, cvm: sd):
    from __externals__ import cv_air, cv_vap, c_liq, c_ice
    with computation(PARALLEL), interval(...):
        cvm = moist_cv_nwat6_fn(qvapor, qliquid, qrain, qsnow, qice, qgraupel, cv_air, cv_vap, c_liq, c_ice)
        #ql = qliquid + qrain
        #qs = qice + qsnow + qgraupel
        #qd = ql + qs
        #cvm[0, 0, 0] = (1.0 - (qvapor + qd)) * cv_air + qvapor * cv_vap + ql * c_liq + qs * c_ice

@utils.stencil(externals={'cv_air': constants.CV_AIR, 'cv_vap': constants.CV_VAP, 'c_liq': constants.C_LIQ, 'c_ice': constants.C_ICE})
def cvm_adjust(qvapor: sd, qliquid: sd, qrain: sd, qsnow: sd,qice: sd,  qgraupel: sd, q_con: sd, gz: sd, cvm: sd, te_2d: sd, delp: sd, pt: sd, phis: sd, u: sd, v: sd, w: sd, rsin2: sd, cosa_s: sd, r_vir: float):
    from __externals__ import cv_air, cv_vap, c_liq, c_ice
    with computation(PARALLEL), interval(...):
        cvm = moist_cv_nwat6_fn(qvapor, qliquid, qrain, qsnow, qice, qgraupel, cv_air, cv_vap, c_liq, c_ice)
        q_con[0, 0, 0] = gz
        te_2d[0, 0, 0] = te_2d + delp * (cvm * pt / ((1.0 + r_vir * qvapor) * (1.0 - gz)) +
                                     0.5 * (phis + phis[0, 0, 1] + w**2 + 0.5 * rsin2 *
                                            (u**2 + u[0, 1, 0]**2 + v**2 + v[1, 0, 0]**2 -
                                             (u + u[0, 1, 0]) * (v + v[1, 0, 0]) * cosa_s )))
# The subroutine 'moist_cv' computes the FV3-consistent moist heat capacity under constant volume,
# including the heating capacity of water vapor and condensates.
# See emanuel1994atmospheric for information on variable heat capacities.
def moist_cv():
     #j, k, nwat, sphum, liq_wat, rainwat,ice_wat, snowwat, graupel, q, gz, cvm
    nwat = spec.namelist['nwat']
    if nwat != 6:
        raise Exception('We still need to implement other nwats for moist_cv')
# assumes 3d variables are indexed to j
def compute(qvapor_js, qliquid_js, qice_js, qrain_js, qsnow_js, qgraupel_js, qcld_js, te_2d, gz, cvm, delp, q_con, pt, phis, w, u, v, r_vir, j_2d):
    grid = spec.grid
    print(gz.shape, qsnow_js.shape, te_2d.shape, phis.shape, u.shape)
    cosa_s = utils.make_storage_data(grid.cosa_s.data[:, j_2d, :], phis.shape, dummy=[1])
    rsin2 = utils.make_storage_data(grid.cosa_s.data[:, j_2d, :], phis.shape, dummy=[1])
   
    cvm_adjust(qvapor_js, qliquid_js, qrain_js, qsnow_js, qice_js, qgraupel_js,
               q_con, gz, cvm, te_2d, delp, pt, phis, u, v, w,
               rsin2, cosa_s, r_vir,
               origin=(grid.is_, 0, 0), domain=(grid.nic, 1, grid.npz)
    )
