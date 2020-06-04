import fv3.utils.gt4py_utils as utils
from fv3.utils.corners import fill2_4corners, fill_4corners
import gt4py.gtscript as gtscript
import fv3._config as spec
import math as math
from gt4py.gtscript import computation, interval, PARALLEL
import fv3.stencils.copy_stencil as cp
import fv3.stencils.remap_profile as remap_profile
import fv3.stencils.fillz as fillz

import numpy as np

sd = utils.sd


def grid():
    return spec.grid


@utils.stencil()
def set_dp(dp: sd, pe1: sd):
    with computation(PARALLEL), interval(...):
        dp = pe1[0, 0, 1] - pe1



@utils.stencil()
def lagrangian_tracer_contributions(
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
        fac1 = pe1
        fac2 = pe1
        dp = pe1
        esl = pe1
        if pe1 < pbot and pe1[0, 0, 1] > ptop:
            # We are in the right pressure range to contribute to the Eulerian cell
            if pe1 <= ptop:
                # we are in the first Lagrangian level that conributes
                pl = (ptop - pe1) / dp1
                if pbot <= pe1[0, 0, 1]:
                    # eulerian grid element is contained in the Lagrangian element
                    pr = (pbot - pe1) / dp1
                    fac1 = pr + pl
                    fac2 = r3 * (pr*fac1 + pl*pl)
                    fac1 = 0.5*fac1
                    q2_adds = q4_2 + (q4_4 + q4_3 - q4_2)*fac1 - q4_4*fac2
                else:
                    # Eulerian element encompasses multiple Lagrangian elements and this is just the first one
                    dp = pe1[0,0,1] - ptop
                    fac1 = 1.+pl
                    fac2 = r3*(1.+pl*fac1)
                    fac1 = 0.5*fac1
                    q2_adds = dp * (q4_2 + (q4_4 + q4_3 - q4_2)*fac1 - q4_4*fac2) / (pbot - ptop)
            else:
                # we are in a farther-down level
                if pbot > pe1[0, 0, 1]:
                    # add the whole level to the Eulerian cell
                    q2_adds = dp1 * q4_1 / (pbot - ptop)
                else:
                    # this is the bottom layer that contributes
                    dp = pbot - pe1
                    esl = dp / dp1
                    fac1 = 0.5*esl
                    fac2 = 1.-r23*esl
                    q2_adds = (
                        dp
                        * (q4_2 + fac1*(q4_3-q4_2+q4_4*fac2))
                        / (pbot - ptop)
                    )
        else:
            q2_adds = 0


def compute(pe1, pe2, dp2, qvapor, qliquid, qice, qrain, qsnow, qgraupel, qcld, j_2d, nq, q_min):
    grid = spec.grid
    kord = abs(spec.namelist["kord_tr"])
    fill = spec.namelist["fill"]
    print(qvapor.shape)
    i1 = grid.is_
    i2 = grid.ie
    i_extent = i2 - i1 + 1
    km = grid.npz
    orig = (grid.is_, grid.js, 0)
    r3 = 1.0 / 3.0
    r23 = 2.0 / 3.0

    klevs = np.arange(km)
    ptop = utils.make_storage_from_shape(pe2.shape, origin=orig)
    pbot = utils.make_storage_from_shape(pe2.shape, origin=orig)

    dp1 = utils.make_storage_from_shape(pe1.shape, origin=orig)
    set_dp(dp1, pe1, origin=(i1, 0, 0), domain=(i_extent, 1, km))
    
    q4_1 = utils.make_storage_data(qvapor, qvapor.shape)
    q4_2 = utils.make_storage_from_shape(q4_1.shape, origin=(grid.is_, 0, 0))
    q4_3 = utils.make_storage_from_shape(q4_1.shape, origin=(grid.is_, 0, 0))
    q4_4 = utils.make_storage_from_shape(q4_1.shape, origin=(grid.is_, 0, 0))
    q2_adds = utils.make_storage_from_shape(q4_1.shape, origin=orig)

    print(qice[33,0,15:20])

    tracers = ["qvapor", "qliquid", "qice", "qrain", "qsnow", "qgraupel", "qcld"]
    tracer_qs = {"qvapor":qvapor, "qliquid":qliquid, "qice":qice, "qrain":qrain, "qsnow":qsnow, "qgraupel":qgraupel, "qcld":qcld}
    assert len(tracer_qs)==nq

    # for q in tracers:
    #     print(q)
    #     print((tracer_qs[q]>0.).all())
    #     #reset fields
    #     q4_1.data[:] = tracer_qs[q].data[:]
    #     q4_2.data[:] = np.zeros(q4_1.shape)
    #     q4_3.data[:] = np.zeros(q4_1.shape)
    #     q4_4.data[:] = np.zeros(q4_1.shape)
    #     q2_adds.data[:] = np.zeros(q4_1.shape)

    #     q4_1, q4_2, q4_3, q4_4 = remap_profile.compute_tracer(q4_1, q4_2, q4_3, q4_4, dp1, km, i1, i2, kord, q_min)

    #     # if q=="qice":
    #     #     print(pe1[33,0,:])
    #     #     print(pe2[33,0,:])

    #     # Trying a stencil with a loop over k2:
    #     for k_eul in klevs:
    #         eulerian_top_pressure = pe2.data[:, :, k_eul]
    #         eulerian_bottom_pressure = pe2.data[:, :, k_eul + 1]
    #         top_p = np.repeat(eulerian_top_pressure[:, :, np.newaxis], km, axis=2)
    #         bot_p = np.repeat(eulerian_bottom_pressure[:, :, np.newaxis], km, axis=2)
    #         ptop = utils.make_storage_data(top_p, pe2.shape)
    #         pbot = utils.make_storage_data(bot_p, pe2.shape)
    #         lagrangian_tracer_contributions(
    #             pe1,
    #             ptop,
    #             pbot,
    #             q4_1,
    #             q4_2,
    #             q4_3,
    #             q4_4,
    #             dp1,
    #             q2_adds,
    #             r3,
    #             r23,
    #             origin=(i1, 0, 0),
    #             domain=(i_extent, 1, km),
    #         )

    #         # if (q=="qliquid") and (k_eul>0) and (k_eul < 4):
    #         #     print(q2_adds.data[33,0,:])

    #         tracer_qs[q][i1 : i2 + 1, 0, k_eul] = np.sum(q2_adds.data[i1 : i2 + 1, 0, :], axis=1)
    #     if q=="qice":
    #         print(tracer_qs[q][38,0,10:17])
    #         print(np.sum(tracer_qs[q][33,0,:]))
    #     # if fill:
    #     #     tracer_qs[q] = fillz.compute(tracer_qs[q], dp2, i1, i2, km)
    # if fill:
    #     qvapor, qliquid, qice, qrain, qsnow, qgraupel, qcld = fillz.compute_test(dp2, qvapor, qliquid, qice, qrain, qsnow, qgraupel, qcld, i_extent, km, nq)

    # return qvapor, qliquid, qice, qrain, qsnow, qgraupel, qcld


    # transliterated fortran
    for q in tracers:

        q4_1.data[:] = tracer_qs[q].data[:]
        q4_2.data[:] = np.zeros(q4_1.shape)
        q4_3.data[:] = np.zeros(q4_1.shape)
        q4_4.data[:] = np.zeros(q4_1.shape)

        q4_1, q4_2, q4_3, q4_4 = remap_profile.compute_tracer(q4_1, q4_2, q4_3, q4_4, dp1, km, i1, i2, kord, q_min, 0, 1)

        i_vals = np.arange(i1, i2 + 1)
        kn = grid.npz
        elems = np.ones((i_extent,kn))
        for ii in i_vals:
            k0 = 0
            for k2 in np.arange(kn):  # loop over new, remapped ks]
                for k1 in np.arange(k0, km):  # loop over old ks
                    # find the top edge of new grid: pe2[ii, k2]
                    if pe2[ii, 0, k2] >= pe1[ii, 0, k1] and pe2[ii, 0, k2] <= pe1[ii, 0, k1 + 1]:
                        pl = (pe2[ii, 0, k2] - pe1[ii, 0, k1]) / dp1[ii, 0, k1]
                        if (
                            pe2[ii, 0, k2 + 1] <= pe1[ii, 0, k1 + 1]
                        ):  # then the new grid layer is entirely within the old one
                            pr = (pe2[ii, 0, k2 + 1] - pe1[ii, 0, k1]) / dp1[ii, 0, k1]
                            fac1 = pr+pl
                            fac2 = r3*(pr*fac1 + pl*pl)
                            fac1 = 0.5*fac1
                            tracer_qs[q][ii, 0, k2] = (
                                q4_2[ii, 0, k1] + (q4_4[ii, 0, k1] + q4_3[ii, 0, k1] - q4_2[ii, 0, k1]) * fac1
                                - q4_4[ii, 0, k1] * fac2
                            )
                            k0 = k1
                            elems[ii-i1,k2]=0
                            break
                        else:  # new grid layer extends into more old grid layers
                            dp = pe1[ii, 0, k1+1] - pe2[ii, 0, k2]
                            fac1 = 1.+pl
                            fac2 = r3*(1.+pl*fac1)
                            fac1 = 0.5*fac1
                            qsum = dp * (
                                q4_2[ii, 0, k1] + (q4_4[ii, 0, k1] + q4_3[ii, 0, k1] - q4_2[ii, 0, k1])
                                * fac1 - q4_4[ii, 0, k1] * fac2
                            )

                            for mm in np.arange(k1 + 1, km):  # find the bottom edge
                                if pe2[ii, 0, k2 + 1] > pe1[ii, 0, mm + 1]:  #Not there yet; add the whole layer
                                    qsum = qsum + dp1[ii, 0, mm] * q4_1[ii, 0, mm]
                                else:
                                    dp = pe2[ii, 0, k2 + 1] - pe1[ii, 0, mm]
                                    esl = dp / dp1[ii, 0, mm]
                                    fac1 = 0.5*esl
                                    fac2 = 1.-r23*esl
                                    qsum = qsum + dp * (
                                        q4_2[ii, 0, mm]
                                        + fac1 * (
                                            q4_3[ii, 0, mm]
                                            - q4_2[ii, 0, mm]
                                            + q4_4[ii, 0, mm] * fac2
                                        )
                                    )
                                    k0 = mm
                                    elems[ii-i1,k2]=0
                                    break
                            #Add everything up and divide by the pressure difference
                            tracer_qs[q][ii, 0, k2] = qsum / dp2[ii, 0, k2]
                            break

    #     if fill:
    #         tracer_qs[q] = fillz.compute(tracer_qs[q], dp2, i1, i2, km)
    if fill:
        qvapor, qliquid, qice, qrain, qsnow, qgraupel, qcld = fillz.compute_test(dp2, qvapor, qliquid, qice, qrain, qsnow, qgraupel, qcld, i_extent, km, nq)

    return qvapor, qliquid, qice, qrain, qsnow, qgraupel, qcld

    # return [tracer_qs[tracer] for tracer in tracers]
