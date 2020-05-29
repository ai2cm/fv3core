import fv3.utils.gt4py_utils as utils
from fv3.utils.corners import fill2_4corners, fill_4corners
import gt4py.gtscript as gtscript
import fv3._config as spec
import math as math
from gt4py.gtscript import computation, interval, PARALLEL
import fv3.stencils.copy_stencil as cp
import fv3.stencils.remap_profile as remap_profile
from fv3.stencils.map_scalar import region_mode # TODO import more of this, very similar code
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

# TODO: this is VERY similar to map_scalar -- once matches, consolidate code
def compute(q1, pe1, pe2, qs, i1, i2, mode, kord, j_2d=None):
    i= 3
    if j_2d is None:
        j = 3
    else:
        j = 0
    k = 0
     
    print('inputs', q1[i, j, k], pe1[i, j, k], pe1[i, j, k+1], pe2[i, j, k], pe2[i, j, k+1], qs[i, j, k], qs[i, 0, k], i1, i2, mode, kord, j_2d)
    # 105.60133456459396 64.247 138.13883301684726 64.247 137.79 1.4297856590351063e-05 1.4297856590351063e-05 3 50 -1 10 3
    # 105.60133456459396 64.247 138.13883301684726  64.247 137.79 3.713293436259343e-10           3 50 -1 10 None
   
    grid = spec.grid
    iv = mode
    i_extent = i2 - i1 + 1
    km = grid.npz
    origin, domain, jslice, j_extent = region_mode(j_2d, i1, i_extent, grid)
    orig = (grid.is_, grid.js, 0)
    r3 = 1.0 / 3.0
    r23 = 2.0 / 3.0
    q_2d = utils.make_storage_data(
        q1[:, jslice, :], (q1.shape[0], j_extent, q1.shape[2])
    )
    
    dp1 = utils.make_storage_from_shape(q_2d.shape, origin=orig)
    if j_2d is None: #TODO fix this, not needed for map_scalar, so why here
        qs = utils.make_storage_data(qs.data[:, jslice, :], q_2d.shape)
        pe1 = utils.make_storage_data(
        pe1[:, jslice, :], (pe1.shape[0], j_extent, pe1.shape[2])
    )
    #print('qs', qs.shape, qs[:, 0, 0])
    '''
   
    '''
    q4_1 = cp.copy(q_2d, origin=(0, 0, 0))
    q4_2 = utils.make_storage_from_shape(q4_1.shape, origin=(grid.is_, 0, 0))
    q4_3 = utils.make_storage_from_shape(q4_1.shape, origin=(grid.is_, 0, 0))
    q4_4 = utils.make_storage_from_shape(q4_1.shape, origin=(grid.is_, 0, 0))

    set_dp(dp1, pe1, origin=origin, domain=domain)

    if kord > 7:
        q4_1, q4_2, q4_3, q4_4 = remap_profile.compute(
            qs, q4_1, q4_2, q4_3, q4_4, dp1, km, i1, i2, iv, kord, 0, j_extent
        )
    print('intermediate',q4_1[i, j, k], q4_2[i, j, k], q4_3[i, j, k], q4_4[i, j, k])
    # intermediate
    # 6.060130102114798 6.1176493656838735 5.973314290257339 0.08788964486515205
    # 83.42993456855324 88.8666808590653 79.66487265215093 -5.015053122329277
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
        if j_2d is None:
            ptop = utils.make_storage_data(top_p[:, jslice, :], q_2d.shape)
            pbot = utils.make_storage_data(bot_p[:, jslice, :], q_2d.shape)
        else:
            ptop = utils.make_storage_data(top_p, q_2d.shape)
            pbot = utils.make_storage_data(bot_p, q_2d.shape)
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
            origin=origin,
            domain=domain,
        )

        q1[i1 : i2 + 1, jslice, k_eul] = np.sum(q2_adds.data[i1 : i2 + 1, 0:j_extent, :], axis=2)
    print(q1[3, 3, :])
    print(q1[3, 3, 51])
    '''
inputs 86.52342338850544 64.247 138.20207481477163 64.247 137.79 1.4297856590351063e-05 1.4297856590351063e-05 3 51 -1 10 3
3 51 (55, 1, 64)
intermediate -20.37243262362584 -23.05758635314084 -19.02985575886834 4.0277305942725015
[-20.37995509 -19.02985596 -24.15638621 -12.2514984   -8.92980672
  -4.07206834   1.00064649   3.4520335    4.87389884   6.35842119
   7.49236152   7.69326849   7.16900382   6.46629715   5.7701242
   4.73544988   4.38305839   6.04567937   6.15292413   3.50550053
   7.35628758   0.8309838   -3.65821602  -1.02251797  -0.89881566
  -3.10214317  -6.8589867  -11.62723785 -16.36269201 -19.09314993
 -21.3000881  -21.70904305 -19.87562634 -19.07179523 -18.31776893
 -17.11577856 -15.22981292 -13.1667294  -11.94451632 -12.25698949
 -13.18514777 -13.35411292 -12.07386197 -10.10070185  -8.36112904
  -6.61221976  -5.65217396  -4.76998937  -4.32957009  -3.54788708
  -2.98637778  -2.74968384  -2.60831395  -2.35984275  -1.90027146
  -1.17958948  -0.36522793   0.41368324   1.02794341   1.37976958
   1.41567149   1.2451241    1.29068878   0.        ]
-2.749683835956979



    '''
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

    return q1
