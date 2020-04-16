import fv3.utils.gt4py_utils as utils
from fv3.utils.corners import fill2_4corners, fill_4corners
import gt4py.gtscript as gtscript
import fv3._config as spec
from gt4py.gtscript import computation, interval, PARALLEL
import fv3.stencils.copy_stencil as cp
import fv3.stencils.cs_profile as cs_profile
#import fv3.stencils.ppm_profile as ppm_profile
import numpy as np

sd = utils.sd


def grid():
    return spec.grid

@utils.stencil()
def set_locals(dp1: sd, q4_1: sd, pe1: sd, q1: sd):
    with computation(PARALLEL), interval(...):
        dp1 = pe1[0, 0, 1] - pe1
        q4_1 = q1

def compute(q1, pe1, pe2, qs, iv, jj, kord):
    grid = spec.grid
    i1=grid.is_
    i_extent = grid.nid
    i2= i1 + i_extent - 1
    km = grid.npz - 1
    kn = grid.npz - 1
    r3 = 1./3.
    r23 = 2./3.
    orig = (grid.is_, grid.js, 0)
    dp1 = utils.make_storage_from_shape(pe1.shape, origin=orig)
    q4_1 = utils.make_storage_from_shape(q1.shape, origin=orig)
    q4_2 = cp.copy(q4_1, origin=orig)
    q4_3 = cp.copy(q4_1, origin=orig)
    q4_4 = cp.copy(q4_1, origin=orig)

    q2 = utils.make_storage_from_shape(q1.shape, origin=orig)

    set_locals(dp1, q4_1, pe1, q1, origin=(i1, grid.js,0), domain=(i_extent, 1, km))

    if kord > 7:
        q4_1, q4_2, q4_3,q4_4 = cs_profile.compute(qs, q4_1, q4_2, q4_3, q4_4, dp1, km, i1, i2, iv, kord)
    # else:
    #     ppm_profile.compute(q4_1, q4_2, q4_3, q4_4, dp1, km, i1, i2, iv, kord)

    i_vals = np.arange(i1,i2+1)
    for ii in i_vals:
        k0 = 0
        for k2 in np.arange(kn):#loop over new, remapped ks
            for k1 in np.arange(k0, km):#loop over old ks
                #find the top edge: pe2[ii, k2]
                if (pe2[ii,k2] >= pe1[ii,k1] and pe2[ii,k2] <= pe1[ii,k1+1]):
                    pl = (pe2[ii,k2]-pe1[ii,k1]) / dp1[ii,k1]
                    if pe2[ii,k2+1] <= pe1[ii,k1+1]: #then the new grid layer is entirely within the old one
                        pr = (pe2[ii,k2+1]-pe1[ii,k1]) / dp1[ii,k1]
                        q2[ii, jj, k2] = q4_2[ii,1,k1] + 0.5*(q4_4[ii, 1, k1] + q4_3[ii,1,k1] - q4_2[ii,1,k1])*(pr+pl)-q4_4[ii,1,k1]*r3*(pr*(pr+pl)*pl**2)
                        k0=k1
                        continue
                    else: #we have some fractional coverage
                        qsum = (pe1[ii, k1+1] - pe2[ii,k2])*(q4_2[ii,1,k1]+ 0.5*(q4_4[ii,k1] + q4_3[ii,1,k1] - q4_2[ii,1,k1])*(1.+pl) - q4_4[ii,1,k1]*(r3*(1.+pl*(1.+pl))))
                        for mm in np.arange(k1+1,km):#find the bottom edge
                            if pe2[ii,k2+2] > pe1[ii,mm+1]:#add the whole layer
                                qsum += dp1[ii,mm] * q4_1[ii,1,mm]
                            else:
                                dp = pe2[ii,k2+1] - pe1[ii,mm]
                                esl = dp/dp1[ii,mm]
                                qsum += dp*(q4_2[ii,1,mm]+0.5*esl*(q4_3[ii,1,mm]-q4_2[ii,1,mm]+q4_4[ii,1,mm]*(1.-r23*esl)))
                                k0=mm
                                q2[ii,jj,k2] = qsum/(pe2[ii,k2+1]-pe2[ii,k2])
                        q2[ii,jj,k2] = qsum/(pe2[ii,k2+1]-pe2[ii,k2])
    return q2