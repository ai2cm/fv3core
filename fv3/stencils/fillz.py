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
def fix_top(q:sd, dp:sd, dm:sd):
    with computation(PARALLEL), interval(1,2):
        if q[0,0,-1] < 0.:
            q = q + q[0,0,-1] * dp[0,0,-1]/dp
    with computation(PARALLEL), interval(0,1):
        if q < 0:
            q = 0
        dm = q*dp

@utils.stencil()
def fix_interior(q:sd, dp:sd, zfix:sd, upper_fix:sd, lower_fix:sd, dm:sd, dm_pos:sd):
    with computation(FORWARD), interval(...):
        dq = q * dp
        dm = dq
        dm_pos = dq if dq > 0. else 0.
        if q < 0.:
            zfix = 1.
            if q[0,0,-1] > 0.:
                #Borrow from above
                dq = dm[0,0,-1] if dm[0,0,-1] < -dm else -dm
                q = q + dq/dp
                upper_fix = dq
            if (q < 0) and (q[0,0,1] > 0.):
                #borrow from below
                dq = q[0,0,1]*dp[0,0,1] if q[0,0,1]*dp[0,0,1] < -dm else -dm
                q = q + dq/dp
                lower_fix = dq
    with computation(PARALLEL), interval(...):
        #logically, if we're moving down we'd be pulled from above and then from below, so we should hit lower_fix then upper fix
        if lower_fix[0,0,-1] != 0:
            #Previous layer borrowed from this one
            q  = q - lower_fix[0,0,-1]/dp
        if upper_fix[0,0,1] != 0.:
            #next layer borrowed from this one
            q = q - upper_fix[0,0,1]/dp

@utils.stencil()
def fix_bottom(q:sd, dp:sd, zfix:sd, upper_fix:sd, dm:sd, dm_pos:sd):
    with computation(PARALLEL), interval(1,2):
        dm = q*dp
        dm_pos = dm if dm > 0. else 0.
        qup = q[0,0,-1]*dp[0,0,-1]
        qly = -q*dp
        dup = qup if qup < qly else qly
        if (q < 0.) and ( q[0,0,-1] > 0.):
            zfix = 1.
            q = q+(dup/dp)
            upper_fix = dup
    with computation(PARALLEL), interval(0,1):
        if zfix[0,0,1] == 1.:
            q = q - (dup/dp)

@utils.stencil()
def final_check(q:sd, dp:sd, dm:sd, zfix:sd, fac:sd):
    with computation(PARALLEL), interval(...):
        if zfix > 0:
            if fac > 0:
                q = fac *dm/dp if fac *dm/dp > 0. else 0.


def compute(q, dp, i1, i2, km):
    i_extent = i2-i1+1
    orig = (i1,0,0)
    zfix = utils.make_storage_from_shape(q.shape, origin=(0,0,0))
    upper_fix = utils.make_storage_from_shape(q.shape, origin=(0,0,0))
    lower_fix = utils.make_storage_from_shape(q.shape, origin=(0,0,0))
    dm = utils.make_storage_from_shape(q.shape, origin=(0,0,0))
    dm_pos = utils.make_storage_from_shape(q.shape, origin=(0,0,0))
    #TODO: implement dev_gfs_physics ifdef when we implement compiler defs

    fix_top(q, dp, dm, origin=orig, domain=(i_extent, 1, 2))
    fix_interior(q, dp, zfix, upper_fix, lower_fix, dm, dm_pos, origin=(i1, 0, 1), domain=(i_extent, 1, km-2))
    fix_bottom(q, dp, zfix, upper_fix, dm, dm_pos, origin=(i1, 0, km-1), domain=(i_extent, 1, 2))

    fix_cols = np.sum(zfix.data, axis=2)
    zfix.data[:]=np.repeat(fix_cols[:,:,np.newaxis], km+1, axis=2)
    sum0 = np.sum(dm.data[:,:,1:], axis=2)
    sum1 = np.sum(dm_pos.data, axis=2)
    adj_factor = sum0
    adj_factor[sum0 > 0] = sum0[sum0 > 0]/sum1[sum0 > 0]
    fac = utils.make_storage_data(np.repeat(fix_cols[:,:,np.newaxis], km+1, axis=2), q.shape)
    final_check(q, dp, dm, zfix, fac, origin=(i1, 0, 1), domain=(i_extent, 1, km-1))
    return q


def compute_test(dp2, qvapor, qliquid, qice, qrain, qsnow, qgraupel, qcld, im, km, nq):
    print(qvapor.shape)
    print(im)
    i1=grid().is_
    orig = (i1,0,0)
    zfix = utils.make_storage_from_shape(qvapor.shape, origin=(0,0,0))
    upper_fix = utils.make_storage_from_shape(qvapor.shape, origin=(0,0,0))
    lower_fix = utils.make_storage_from_shape(qvapor.shape, origin=(0,0,0))
    dm = utils.make_storage_from_shape(qvapor.shape, origin=(0,0,0))
    dm_pos = utils.make_storage_from_shape(qvapor.shape, origin=(0,0,0))
    fac = utils.make_storage_from_shape(qvapor.shape, origin=(0,0,0))
    #TODO: implement dev_gfs_physics ifdef when we implement compiler defs

    tracers = ["qvapor", "qliquid", "qice", "qrain", "qsnow", "qgraupel", "qcld"]
    tracer_qs = {"qvapor":qvapor, "qliquid":qliquid, "qice":qice, "qrain":qrain, "qsnow":qsnow, "qgraupel":qgraupel, "qcld":qcld}

    for q in tracer_qs:
        zfix.data[:] = np.zeros(qvapor.shape)
        fix_top(tracer_qs[q], dp2, dm, origin=orig, domain=(im, 1, 2))
        fix_interior(tracer_qs[q], dp2, zfix, upper_fix, lower_fix, dm, dm_pos, origin=(i1, 0, 1), domain=(im, 1, km-2))
        fix_bottom(tracer_qs[q], dp2, zfix, upper_fix, dm, dm_pos, origin=(i1, 0, km-1), domain=(im, 1, 2))

        fix_cols = np.sum(zfix.data, axis=2)
        zfix.data[:]=np.repeat(fix_cols[:,:,np.newaxis], km+1, axis=2)
        sum0 = np.sum(dm.data[:,:,1:], axis=2)
        sum1 = np.sum(dm_pos.data, axis=2)
        adj_factor = sum0
        adj_factor[sum0 > 0] = sum0[sum0 > 0]/sum1[sum0 > 0]
        fac.data[:] = np.repeat(fix_cols[:,:,np.newaxis], km+1, axis=2)
        final_check(tracer_qs[q], dp2, dm, zfix, fac, origin=(i1, 0, 1), domain=(im, 1, km-1))

    return [tracer_qs[tracer] for tracer in tracers]
