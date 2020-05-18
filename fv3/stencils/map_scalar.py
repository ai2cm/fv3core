import fv3.utils.gt4py_utils as utils
from fv3.utils.corners import fill2_4corners, fill_4corners
import gt4py.gtscript as gtscript
import fv3._config as spec
import math as math
from gt4py.gtscript import computation, interval, PARALLEL
import fv3.stencils.copy_stencil as cp
import fv3.stencils.scalar_profile as scalar_profile

# import fv3.stencils.ppm_profile as ppm_profile
import numpy as np

sd = utils.sd


def grid():
    return spec.grid


@utils.stencil()
def set_dp(dp1: sd, pe1: sd):
    with computation(PARALLEL), interval(...):
        dp1 = pe1[0, 0, 1] - pe1


@utils.stencil()
def lagrangian_contributions(
    pe1: sd,
    ptop: sd,
    pbot: sd,
    q4_1: sd,
    q4_2: sd,
    q4_3: sd,
    q4_4: sd,
    dp1: sd,
    q2_adds: sd,
    r3: float,
    r23: float,
):
    with computation(PARALLEL), interval(...):
        pl = pe1
        pr = pe1
        dp = pe1
        esl = pe1
        if pe1 < pbot and pe1[0, 0, 1] > ptop:
            # We are in the right pressure range to contribute to the Eulerian cell
            if pe1 < ptop:
                # we are in the first Lagrangian level that conributes
                pl = (ptop - pe1) / dp1
                if pbot <= pe1[0, 0, 1]:
                    # eulerian grid element is contained in the Lagrangian element
                    pr = (pbot - pe1) / dp1
                    q2_adds = (
                        q4_2
                        + 0.5 * (q4_4 + q4_3 - q4_2) * (pr + pl)
                        - q4_4 * r3 * (pr * (pr + pl) + pl ** 2)
                    )
                else:
                    # Eulerian element encompasses multiple Lagrangian elements and this is just the first one
                    q2_adds = (
                        (pe1[0, 0, 1] - ptop)
                        * (
                            q4_2
                            + 0.5 * (q4_4 + q4_3 - q4_2) * (1.0 + pl)
                            - q4_4 * r3 * (1.0 + pl * (1.0 + pl))
                        )
                        / (pbot - ptop)
                    )
            else:
                # we are in a farther-down level
                if pbot > pe1[0, 0, 1]:
                    # add the whole level to the Eulerian cell
                    q2_adds = dp1 * q4_1 / (pbot - ptop)
                else:
                    # this is the bottom layer that contributes
                    dp = pbot - pe1
                    esl = dp / dp1
                    q2_adds = (
                        dp
                        * (q4_2 + 0.5 * esl * (q4_3 - q4_2 + q4_4 * (1.0 - r23 * esl)))
                        / (pbot - ptop)
                    )
        else:
            q2_adds = 0


def compute(q1, pe1, pe2, qs, j_2d, mode):
    grid = spec.grid
    kord = abs(spec.namelist['kord_tm'])
    qmin = 184.
    i1 = grid.is_
    i2 = grid.ie
    iv = mode
    i_extent = i2 - i1 + 1
    km = grid.npz
    j_2d -= 1
    j_2d += grid.is_
    orig = (grid.is_, grid.js, 0)
    r3 = 1.0 / 3.0
    r23 = 2.0 / 3.0
    q_2d = utils.make_storage_data(
        q1[:, j_2d : j_2d + 1, :], (q1.shape[0], 1, q1.shape[2])
    )
    dp1 = utils.make_storage_from_shape(pe1.shape, origin=orig)

    q4_1 = cp.copy(q_2d, origin=(0, 0, 0))
    q4_2 = utils.make_storage_from_shape(q4_1.shape, origin=(grid.is_, 0, 0))
    q4_3 = utils.make_storage_from_shape(q4_1.shape, origin=(grid.is_, 0, 0))
    q4_4 = utils.make_storage_from_shape(q4_1.shape, origin=(grid.is_, 0, 0))

    q2 = cp.copy(q1, origin=(0, 0, 0))

    set_dp(dp1, pe1, origin=(i1, 0, 0), domain=(i_extent, 1, km))

    if kord > 7:
        q4_1, q4_2, q4_3, q4_4 = scalar_profile.compute(
            qs, q4_1, q4_2, q4_3, q4_4, dp1, km, i1, i2, iv, kord, qmin
        )

    # else:
    #     ppm_profile.compute(q4_1, q4_2, q4_3, q4_4, dp1, km, i1, i2, iv, kord)

    # Trying a stencil with a loop over k2:
    klevs = np.arange(km)
    ptop = utils.make_storage_from_shape(pe2.shape, origin=orig)
    pbot = utils.make_storage_from_shape(pe2.shape, origin=orig)
    q2_adds = utils.make_storage_from_shape(q4_1.shape, origin=orig)
    for k_eul in klevs:
        eulerian_top_pressure = pe2.data[:, :, k_eul]
        eulerian_bottom_pressure = pe2.data[:, :, k_eul + 1]
        top_p = np.repeat(eulerian_top_pressure[:, :, np.newaxis], km, axis=2)
        bot_p = np.repeat(eulerian_bottom_pressure[:, :, np.newaxis], km, axis=2)
        ptop = utils.make_storage_data(top_p, pe1.shape)
        pbot = utils.make_storage_data(bot_p, pe1.shape)
        lagrangian_contributions(
            pe1,
            ptop,
            pbot,
            q4_1,
            q4_2,
            q4_3,
            q4_4,
            dp1,
            q2_adds,
            r3,
            r23,
            origin=(i1, 0, 0),
            domain=(i_extent, 1, km),
        )

        q2[i1 : i2 + 1, j_2d, k_eul] = np.sum(q2_adds.data[i1 : i2 + 1, 0, :], axis=1)

    # #Pythonized
    # n_cont = 0
    # n_ext = 0
    # n_bot = 0
    # kn = grid.npz
    # i_vals = np.arange(i1, i2 + 1)
    # klevs = np.arange(km+1)
    # for ii in i_vals:
    #     for k2 in np.arange(kn):  # loop over new, remapped ks]
    #         top1 = pe2[ii, 0, k2] >= pe1[ii, 0,:]
    #         k1 = klevs[top1][-1]
    #         pl = (pe2[ii, 0, k2] - pe1[ii, 0, k1]) / dp1[ii, 0, k1]
    #         if pe2[ii, 0, k2+1] <= pe1[ii, 0, k1+1]:
    #             #The new grid is contained within the old one
    #             pr = (pe2[ii, 0, k2 + 1] - pe1[ii, 0, k1]) / dp1[ii, 0, k1]
    #             q2[ii, j_2d, k2] = (
    #                 q4_2[ii, 0, k1]
    #                 + 0.5
    #                 * (q4_4[ii, 0, k1] + q4_3[ii, 0, k1] - q4_2[ii, 0, k1])
    #                 * (pr + pl)
    #                 - q4_4[ii, 0, k1] * r3 * (pr * (pr + pl) + pl ** 2)
    #             )
    #             n_cont+=1
    #             # continue
    #         else:
    #             # new grid layer extends into more old grid layers
    #             qsum = (pe1[ii, 0, k1 + 1] - pe2[ii, 0, k2]) * (
    #                         q4_2[ii, 0, k1]
    #                         + 0.5
    #                         * (q4_4[ii, 0, k1] + q4_3[ii, 0, k1] - q4_2[ii, 0, k1])
    #                         * (1.0 + pl)
    #                         - q4_4[ii, 0, k1] * (r3 * (1.0 + pl * (1.0 + pl)))
    #             )
    #             bottom_2 = pe2[ii, 0, k2+1] > pe1[ii, 0, k1+1:]
    #             mm = klevs[k1+1:][bottom_2][-1]
    #             qsum = qsum + np.sum(dp1[ii, 0, k1+1:mm] * q4_1[ii, 0, k1+1:mm])
    #             if not bottom_2.all():
    #                 dp = pe2[ii, 0, k2 + 1] - pe1[ii, 0, mm]
    #                 esl = dp / dp1[ii, 0, mm]
    #                 qsum = qsum + dp * (
    #                     q4_2[ii,0,mm]
    #                     + 0.5
    #                     * esl
    #                     * (
    #                         q4_3[ii, 0, mm]
    #                         - q4_2[ii, 0, mm]
    #                         + q4_4[ii, 0, mm] * (1.0 - r23 * esl)
    #                     )
    #                 )
    #                 n_ext+=1
    #             else:
    #                 n_bot+=1
    #             q2[ii, j_2d, k2] = qsum / (pe2[ii, 0, k2 + 1] - pe2[ii, 0, k2])

    # # transliterated fortran
    # n_cont = 0
    # n_ext = 0
    # n_bot = 0
    # i_vals = np.arange(i1, i2 + 1)
    # kn = grid.npz
    # elems = np.ones((i_extent,kn))
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
    #                     n_cont +=1
    #                     elems[ii-i1,k2]=0
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
    #                             flag = 1
    #                             n_ext+=1
    #                             elems[ii-i1,k2]=0
    #                             break
    #                     if flag == 0:
    #                         print("Huh")
    #                         n_bot+=1
    #                     #Add everything up and divide by the pressure difference
    #                     q2[ii, j_2d, k2] = qsum / (pe2[ii, 0, k2 + 1] - pe2[ii, 0, k2])
    #                     break

    # print(n_cont, n_ext, n_bot)
    # print(n_cont+ n_ext+ n_bot)
    # print(kn * (i_extent))

    return q2
