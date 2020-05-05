import fv3.utils.gt4py_utils as utils
from fv3.utils.corners import fill2_4corners, fill_4corners
import gt4py.gtscript as gtscript
import fv3._config as spec
import math as math
from gt4py.gtscript import computation, interval, PARALLEL
import fv3.stencils.copy_stencil as cp
import fv3.stencils.cs_profile as cs_profile

# import fv3.stencils.ppm_profile as ppm_profile
import numpy as np

sd = utils.sd


def grid():
    return spec.grid

@utils.stencil()
def set_dp(dp1: sd, pe1: sd):
    with computation(PARALLEL), interval(...):
        dp1 = pe1[0, 0, 1] - pe1


def compute(q1, pe1, pe2, qs, j_2d, i1, i2, mode, kord):
    grid = spec.grid
    iv = mode
    i_extent = i2 - i1 + 1
    km = grid.npz
    j_2d -=1
    j_2d += grid.is_
    kn = grid.npz
    r3 = 1.0 / 3.0
    r23 = 2.0 / 3.0
    orig = (grid.is_, grid.js, 0)
    q_2d = utils.make_storage_data(q1[:,j_2d:j_2d+1,:], (q1.shape[0],1,q1.shape[2]))
    dp1 = utils.make_storage_from_shape(pe1.shape, origin=orig)

    print(pe1.shape)
    print(iv,kord)
    print(i1, i2, i_extent)

    q4_1 = cp.copy(q_2d, origin=(0, 0, 0))
    q4_2 = utils.make_storage_from_shape(q4_1.shape, origin=(grid.is_, 0, 0))
    q4_3 = utils.make_storage_from_shape(q4_1.shape, origin=(grid.is_, 0, 0))
    q4_4 = utils.make_storage_from_shape(q4_1.shape, origin=(grid.is_, 0, 0))

    q2 = cp.copy(q1, origin = (0,0,0))

    set_dp(dp1, pe1, origin=(i1, 0, 0), domain=(i_extent, 1, km))

    if kord > 7:
        padded_qs = utils.make_storage_from_shape(q_2d.shape, origin=(grid.is_, 0, 0))
        padded_qs[:,:,-1] = qs[:,:,0]
        q4_1, q4_2, q4_3, q4_4 = cs_profile.compute(
            padded_qs, q4_1, q4_2, q4_3, q4_4, dp1, km, i1, i2, iv, kord
        )
    # else:
    #     ppm_profile.compute(q4_1, q4_2, q4_3, q4_4, dp1, km, i1, i2, iv, kord)

    #Pythonized
    i_vals = np.arange(i1, i2 + 1)
    klevs = np.arange(km+1)
    print(klevs.size)
    for ii in i_vals:
        for k2 in np.arange(kn):  # loop over new, remapped ks]
            top1 = pe2[ii, 0, k2] >= pe1[ii, 0,:]
            k1 = klevs[top1][-1]
            pl = (pe2[ii, 0, k2] - pe1[ii, 0, k1]) / dp1[ii, 0, k1]
            if pe2[ii, 0, k2+1] <= pe1[ii, 0, k1+1]:
                #The new grid is contained within the old one
                pr = (pe2[ii, 0, k2 + 1] - pe1[ii, 0, k1]) / dp1[ii, 0, k1]
                q2[ii, j_2d, k2] = (
                    q4_2[ii, 0, k1]
                    + 0.5
                    * (q4_4[ii, 0, k1] + q4_3[ii, 0, k1] - q4_2[ii, 0, k1])
                    * (pr + pl)
                    - q4_4[ii, 0, k1] * r3 * (pr * (pr + pl) + pl ** 2)
                )
                # continue
            else:
                # new grid layer extends into more old grid layers
                qsum = (pe1[ii, 0, k1 + 1] - pe2[ii, 0, k2]) * (
                            q4_2[ii, 0, k1]
                            + 0.5
                            * (q4_4[ii, 0, k1] + q4_3[ii, 0, k1] - q4_2[ii, 0, k1])
                            * (1.0 + pl)
                            - q4_4[ii, 0, k1] * (r3 * (1.0 + pl * (1.0 + pl)))
                )
                bottom_2 = pe2[ii, 0, k2+1] > pe1[ii, 0, k1+1:]
                mm = klevs[k1+1:][bottom_2][-1]
                qsum = qsum + np.sum(dp1[ii, 0, k1+1:mm] * q4_1[ii, 0, k1+1:mm])
                if not bottom_2.all():
                    dp = pe2[ii, 0, k2 + 1] - pe1[ii, 0, mm]
                    esl = dp / dp1[ii, 0, mm]
                    qsum = qsum + dp * (
                        q4_2[ii,0,mm]
                        + 0.5
                        * esl
                        * (
                            q4_3[ii, 0, mm]
                            - q4_2[ii, 0, mm]
                            + q4_4[ii, 0, mm] * (1.0 - r23 * esl)
                        )
                    )
                else:
                q2[ii, j_2d, k2] = qsum / (pe2[ii, 0, k2 + 1] - pe2[ii, 0, k2])

    # print(n_contained)
    # print(contained_index)

                # bottom_1 = pe2[ii, 0, k2+1] > pe1[ii, 0, k1+1:]
                # if klevs[k1+1:][bottom_1].size!=0:
                #     mm = klevs[k1+1:][bottom_1][-1]
                #     qsum += np.sum(dp1[ii, 0, k1+1:mm+1] * q4_1[ii, 0, k1+1:mm+1])
                #     dp = pe2[ii, 0, k2 + 1] - pe1[ii, 0, mm+1]
                #     esl = dp / dp1[ii, 0, mm+1]
                #     qsum += dp * (
                #         + 0.5
                #         * esl
                #         * (
                #             q4_3[ii, 0, mm+1]
                #             - q4_2[ii, 0, mm+1]
                #             + q4_4[ii, 0, mm+1] * (1.0 - r23 * esl)
                #         )
                #     )
                #     k0 = mm
                #     q2[ii, j_2d, k2] = qsum / (pe2[ii, 0, k2 + 1] - pe2[ii, 0, k2])
                # else:
                #     qsum += np.sum(dp1[ii, 0, k1+1:kn] * q4_1[ii, 0, k1+1:kn])
                #     q2[ii, j_2d, k2] = qsum / (pe2[ii, 0, k2 + 1] - pe2[ii, 0, k2])


    # # transliterated fortran
    # i_vals = np.arange(i1, i2 + 1)
    # for ii in i_vals:
    #     k0 = 0
    #     for k2 in np.arange(kn):  # loop over new, remapped ks]
    #         for k1 in np.arange(k0, km):  # loop over old ks
    #             # find the top edge of new grid: pe2[ii, k2]
    #             if pe2[ii, 0, k2] >= pe1[ii, 0, k1] and pe2[ii, 0, k2] <= pe1[ii, 0, k1 + 1]:
    #                 pl = (pe2[ii, 0, k2] - pe1[ii, 0, k1]) / dp1[ii, 0, k1]
    #                 if (
    #                     pe2[ii, 0, k2 + 1] <= pe1[ii, 0, k1 + 1]
    #                 ):  # then the new grid layer is entirely within the old one
    #                     pr = (pe2[ii, 0, k2 + 1] - pe1[ii, 0, k1]) / dp1[ii, 0, k1]
    #                     q2[ii, j_2d, k2] = (
    #                         q4_2[ii, 0, k1]
    #                         + 0.5
    #                         * (q4_4[ii, 0, k1] + q4_3[ii, 0, k1] - q4_2[ii, 0, k1])
    #                         * (pr + pl)
    #                         - q4_4[ii, 0, k1] * r3 * (pr * (pr + pl) + pl ** 2)
    #                     )
    #                     k0 = k1
    #                     break
    #                 else:  # new grid layer extends into more old grid layers
    #                     qsum = (pe1[ii, 0, k1 + 1] - pe2[ii, 0, k2]) * (
    #                         q4_2[ii, 0, k1]
    #                         + 0.5
    #                         * (q4_4[ii, 0, k1] + q4_3[ii, 0, k1] - q4_2[ii, 0, k1])
    #                         * (1.0 + pl)
    #                         - q4_4[ii, 0, k1] * (r3 * (1.0 + pl * (1.0 + pl)))
    #                     )
    #                     for mm in np.arange(k1 + 1, km):  # find the bottom edge
    #                         flag = 0
    #                         if pe2[ii, 0, k2 + 1] > pe1[ii, 0, mm + 1]:  #Not there yet; add the whole layer
    #                             qsum = qsum + dp1[ii, 0, mm] * q4_1[ii, 0, mm]
    #                         else:
    #                             dp = pe2[ii, 0, k2 + 1] - pe1[ii, 0, mm]
    #                             esl = dp / dp1[ii, 0, mm]
    #                             qsum = qsum + dp * (
    #                                 q4_2[ii, 0, mm]
    #                                 + 0.5
    #                                 * esl
    #                                 * (
    #                                     q4_3[ii, 0, mm]
    #                                     - q4_2[ii, 0, mm]
    #                                     + q4_4[ii, 0, mm] * (1.0 - r23 * esl)
    #                                 )
    #                             )
    #                             k0 = mm
    #                             q2[ii, j_2d, k2] = qsum / (
    #                                 pe2[ii, 0, k2 + 1] - pe2[ii, 0, k2]
    #                             )
    #                             flag = 1
    #                             break
    #                     if flag == 0: #if we get to the bottom of the column then we just take everything
    #                         q2[ii, j_2d, k2] = qsum / (pe2[ii, 0, k2 + 1] - pe2[ii, 0, k2])
    

    return q2
