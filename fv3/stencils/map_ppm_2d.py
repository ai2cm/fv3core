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
    j = 0
    k = 51
    #print(qs.shape, qs[3,:,0])
    #print(qs[0,:,0], np.any(qs ==  -2.7386315404751397e-35))
    #for i in range(55):
    #    for j in range(55):
    #        if qs[i, j, 0] ==  -2.7386315404751397e-35:
    #            print('found it', i, j)
   
    #print('inputs', q1[i, j, k], pe1[i, j, k], pe1[i, j, k+1], qs[i, j, k], qs[i, 0, k], i1, i2, mode, kord, j_2d)
    #  0.007027400809571855 64.247 138.24733238480297 -2.7386315404751397e-35 3 50 -2 10 3
    #  0.007027400809571855 64.247 138.24733238480297 0.000259479861638933 3 50 -2 10 None
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
    print('qs', qs.shape, qs[:, 0, 0])
    '''
   [ 0.00000000e+00  0.00000000e+00  0.00000000e+00  2.59479862e-04
  1.73193719e-07  1.89966046e-06 -2.73863154e-35  0.00000000e+00
  0.00000000e+00  0.00000000e+00  0.00000000e+00  0.00000000e+00
  0.00000000e+00  0.00000000e+00  0.00000000e+00  0.00000000e+00
  0.00000000e+00  0.00000000e+00  0.00000000e+00  0.00000000e+00
  0.00000000e+00  0.00000000e+00  0.00000000e+00  0.00000000e+00
  0.00000000e+00 -8.19446808e-10 -7.78272176e-09 -3.17952861e-07
 -7.69738112e-09  2.04340486e-09  0.00000000e+00  0.00000000e+00
  0.00000000e+00  0.00000000e+00  0.00000000e+00  0.00000000e+00
  0.00000000e+00  0.00000000e+00  0.00000000e+00  0.00000000e+00
  0.00000000e+00 -1.31685854e-36 -2.26789832e-35 -5.56935880e-35
 -2.00463172e-34 -2.16413166e-33 -2.85677942e-33 -1.25167614e-34
  0.00000000e+00  0.00000000e+00  0.00000000e+00  0.00000000e+00
  0.00000000e+00  0.00000000e+00  0.00000000e+00]
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
    # intermediate 0.023853538110226953 0.023853538110226953 0.023853538110226953 0.0
    #              0.023853538110226953 0.023853538110226953 0.023853538110226953 0.0
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
[ 0.00702984 -0.00801616 -0.06602874 -0.08471319 -0.07124542 -0.06901264
 -0.0607066  -0.05432758 -0.05004505 -0.04735575 -0.0458133  -0.04462703
 -0.04162227 -0.03511939 -0.02744192 -0.023686   -0.02288945 -0.02356103
 -0.0207161  -0.00530568 -0.00824895 -0.03491334 -0.04393568 -0.03728279
 -0.03406701 -0.02385777 -0.00504872  0.02510467  0.02937964  0.02208558
  0.03615616  0.05122582  0.06445273  0.07320621  0.07929228  0.0850266
  0.09162092  0.10379068  0.12236691  0.13435491  0.13506218  0.12673853
  0.10929293  0.08705788  0.06614519  0.04959823  0.03676787  0.0281662
  0.02399111  0.02312374  0.02383426  0.02378537  0.02262924  0.02058514
  0.01786852  0.01493874  0.01214819  0.00951903  0.00716697  0.00512713
  0.00333111  0.00176921  0.00066804  0.        ]

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
