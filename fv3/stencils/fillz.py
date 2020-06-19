import fv3.utils.gt4py_utils as utils
from fv3.utils.corners import fill2_4corners, fill_4corners
import gt4py.gtscript as gtscript
import fv3._config as spec
import math as math
from gt4py.gtscript import computation, interval, PARALLEL, FORWARD
import fv3.stencils.copy_stencil as cp
import fv3.stencils.remap_profile as remap_profile
import numpy as np

import numpy as np

sd = utils.sd


def grid():
    return spec.grid


@utils.stencil()
def fix_top(q: sd, dp: sd, dm: sd):
    with computation(PARALLEL), interval(1, 2):
        if q[0, 0, -1] < 0.0:
            q = (
                q + q[0, 0, -1] * dp[0, 0, -1] / dp
            )  # move enough mass up so that the top layer isn't negative
    with computation(PARALLEL), interval(0, 1):
        if q < 0:
            q = 0
        dm = q * dp


@utils.stencil()
def fix_interior(
    q: sd, dp: sd, zfix: sd, upper_fix: sd, lower_fix: sd, dm: sd, dm_pos: sd
):
    with computation(FORWARD), interval(
        1, -1
    ):  # if a higher layer borrowed from this one, account for that here
        if lower_fix[0, 0, -1] != 0.0:
            q = q - (lower_fix[0, 0, -1] / dp)
        dq = q * dp
        if q < 0.0:
            zfix = 1.0
            if q[0, 0, -1] > 0.0:
                # Borrow from the layer above
                dq = (
                    q[0, 0, -1] * dp[0, 0, -1]
                    if q[0, 0, -1] * dp[0, 0, -1] < -(q * dp)
                    else -(q * dp)
                )
                q = q + dq / dp
                upper_fix = dq
            if (q < 0.0) and (q[0, 0, 1] > 0.0):
                # borrow from the layer below
                dq = (
                    q[0, 0, 1] * dp[0, 0, 1]
                    if q[0, 0, 1] * dp[0, 0, 1] < -(q * dp)
                    else -(q * dp)
                )
                q = q + dq / dp
                lower_fix = dq
    with computation(PARALLEL), interval(...):
        if upper_fix[0, 0, 1] != 0.0:
            # If a lower layer borrowed from this one, account for that here
            q = q - upper_fix[0, 0, 1] / dp
        dm = q * dp
        dm_pos = dm if dm > 0.0 else 0.0


@utils.stencil()
def fix_bottom(
    q: sd, dp: sd, zfix: sd, upper_fix: sd, lower_fix: sd, dm: sd, dm_pos: sd
):
    with computation(PARALLEL), interval(1, 2):
        if (
            lower_fix[0, 0, -1] != 0.0
        ):  # the 2nd-to-last layer borrowed from this one, account for that here
            q = q - (lower_fix[0, 0, -1] / dp)
        qup = q[0, 0, -1] * dp[0, 0, -1]
        qly = -q * dp
        dup = qup if qup < qly else qly
        if (q < 0.0) and (q[0, 0, -1] > 0.0):
            zfix = 1.0
            q = q + (dup / dp)
            upper_fix = dup
        dm = q * dp
        dm_pos = dm if dm > 0.0 else 0.0
    with computation(PARALLEL), interval(0, 1):
        if (
            upper_fix[0, 0, 1] != 0.0
        ):  # if the bottom layer borrowed from this one, adjust
            q = q - (upper_fix[0, 0, 1] / dp)
            dm = q * dp
            dm_pos = dm if dm > 0.0 else 0.0  # now we gotta update these too


@utils.stencil()
def final_check(q: sd, dp: sd, dm: sd, zfix: sd, fac: sd):
    with computation(PARALLEL), interval(...):
        if zfix > 0:
            if fac > 0:
                q = fac * dm / dp if fac * dm / dp > 0.0 else 0.0


"""
def compute(q, dp, i1, i2, km, js, j_extent):
    #Run on one tracer
    i_extent = i2-i1+1
    orig = (i1,js,0)
    zfix = utils.make_storage_from_shape(q.shape, origin=(0,0,0))
    upper_fix = utils.make_storage_from_shape(q.shape, origin=(0,0,0))
    lower_fix = utils.make_storage_from_shape(q.shape, origin=(0,0,0))
    dm = utils.make_storage_from_shape(q.shape, origin=(0,0,0))
    dm_pos = utils.make_storage_from_shape(q.shape, origin=(0,0,0))
    #TODO: implement dev_gfs_physics ifdef when we implement compiler defs

    # x=q[38,0,14]*dp[38,0,14]
    # print(q[38,0,14]-(x/dp[38,0,14]))
    # print(q[38,0,14]-(q[38,0,14]*dp[38,0,14]/dp[38,0,14]))
    # print(q[38,0,14], dp[38,0,14])

    # print(q[12,0,35:43])
    # x=q[12,0,40]*dp[12,0,40]
    # print(q[12,0,40]-(x/dp[12,0,40]))
    # print(q[12,0,40]-(q[12,0,40]*dp[12,0,40]/dp[12,0,40]))
    # print(q[12,0,40], dp[12,0,40])

    #print(q[49,0,3:10])
    #x=q[49,0,6]*dp[49,0,6]
    #print(q[49,0,6]-(x/dp[49,0,6]))
    #print(q[49,0,6]-(q[49,0,6]*dp[49,0,6]/dp[49,0,6]))
    #print(q[49,0,6], dp[49,0,6])
   
    fix_top(q, dp, dm, origin=orig, domain=(i_extent, j_extent, 2))
    fix_interior(q, dp, zfix, upper_fix, lower_fix, dm, dm_pos, origin=(i1, 0, 0), domain=(i_extent, j_extent, km))
    fix_bottom(q, dp, zfix, upper_fix, lower_fix, dm, dm_pos, origin=(i1, 0, km-2), domain=(i_extent, j_extent, 2))

    # print(q[33,0,14:20])
    # print(q[12,0,35:43])
    #print(q[49,0,3:10])

    #print(np.sum(q[49,0,:]))

    fix_cols = np.sum(zfix.data, axis=2)
    zfix.data[:]=np.repeat(fix_cols[:,:,np.newaxis], km+1, axis=2)
    sum0 = np.sum(dm[:,:,1:], axis=2)
    sum1 = np.sum(dm_pos[:,:,1:], axis=2)
    adj_factor = np.zeros(sum0.shape)
    adj_factor[sum0 > 0] = sum0[sum0 > 0]/sum1[sum0 > 0]
    fac = utils.make_storage_data(np.repeat(adj_factor[:,:,np.newaxis], km+1, axis=2), q.shape)
    final_check(q, dp, dm, zfix, fac, origin=(i1, js, 1), domain=(i_extent, j_extent, km-1))
   
    return q
"""


def compute_test(
    dp2, qvapor, qliquid, qice, qrain, qsnow, qgraupel, qcld, im, km, nq, js, j_extent
):
    # Same as above, but with multiple tracer fields
    i1 = grid().is_
    orig = (i1, js, 0)
    zfix = utils.make_storage_from_shape(qvapor.shape, origin=(0, 0, 0))
    upper_fix = utils.make_storage_from_shape(qvapor.shape, origin=(0, 0, 0))
    lower_fix = utils.make_storage_from_shape(qvapor.shape, origin=(0, 0, 0))
    dm = utils.make_storage_from_shape(qvapor.shape, origin=(0, 0, 0))
    dm_pos = utils.make_storage_from_shape(qvapor.shape, origin=(0, 0, 0))
    fac = utils.make_storage_from_shape(qvapor.shape, origin=(0, 0, 0))
    # TODO: implement dev_gfs_physics ifdef when we implement compiler defs

    tracers = ["qvapor", "qliquid", "qice", "qrain", "qsnow", "qgraupel", "qcld"]
    tracer_qs = {
        "qvapor": qvapor,
        "qliquid": qliquid,
        "qice": qice,
        "qrain": qrain,
        "qsnow": qsnow,
        "qgraupel": qgraupel,
        "qcld": qcld,
    }

    for q in tracer_qs:
        # reset fields
        zfix.data[:] = np.zeros(qvapor.shape)
        fac.data[:] = np.zeros(qvapor.shape)
        lower_fix.data[:] = np.zeros(qvapor.shape)
        upper_fix.data[:] = np.zeros(qvapor.shape)
        fix_top(tracer_qs[q], dp2, dm, origin=orig, domain=(im, j_extent, 2))
        fix_interior(
            tracer_qs[q],
            dp2,
            zfix,
            upper_fix,
            lower_fix,
            dm,
            dm_pos,
            origin=(i1, js, 0),
            domain=(im, j_extent, km),
        )
        fix_bottom(
            tracer_qs[q],
            dp2,
            zfix,
            upper_fix,
            lower_fix,
            dm,
            dm_pos,
            origin=(i1, js, km - 2),
            domain=(im, j_extent, 2),
        )
        fix_cols = np.sum(zfix.data[:], axis=2)
        zfix.data[:] = np.repeat(fix_cols[:, :, np.newaxis], km + 1, axis=2)
        sum0 = np.sum(dm[:, :, 1:], axis=2)
        sum1 = np.sum(dm_pos[:, :, 1:], axis=2)
        adj_factor = np.zeros(sum0.shape)
        adj_factor[sum0 > 0] = sum0[sum0 > 0] / sum1[sum0 > 0]
        fac.data[:] = np.repeat(adj_factor[:, :, np.newaxis], km + 1, axis=2)
        final_check(
            tracer_qs[q],
            dp2,
            dm,
            zfix,
            fac,
            origin=(i1, js, 1),
            domain=(im, j_extent, km - 1),
        )
    return [tracer_qs[tracer] for tracer in tracers]
