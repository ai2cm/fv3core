#!/usr/bin/env python3
import fv3.utils.gt4py_utils as utils
import gt4py.gtscript as gtscript
import fv3._config as spec
from gt4py.gtscript import computation, interval, PARALLEL
from fv3.stencils.updatedzd import ra_x_stencil, ra_y_stencil
import fv3.stencils.copy_stencil as cp
import fv3.stencils.fvtp2d as fvtp2d

# from mpi4py import MPI
import numpy as np

sd = utils.sd
from mpi4py import MPI


@utils.stencil()
def flux_x(cx: sd, dxa: sd, dy: sd, sin_sg3: sd, sin_sg1: sd, xfx: sd):
    with computation(PARALLEL), interval(...):
        xfx[0, 0, 0] = (
            cx * dxa[-1, 0, 0] * dy * sin_sg3[-1, 0, 0]
            if cx > 0
            else cx * dxa * dy * sin_sg1
        )


@utils.stencil()
def flux_y(cy: sd, dya: sd, dx: sd, sin_sg4: sd, sin_sg2: sd, yfx: sd):
    with computation(PARALLEL), interval(...):
        yfx[0, 0, 0] = (
            cy * dya[0, -1, 0] * dx * sin_sg4[0, -1, 0]
            if cy > 0
            else cy * dya * dx * sin_sg2
        )


@utils.stencil()
def cmax_split(var: sd, nsplt: sd):
    with computation(PARALLEL), interval(...):
        frac = 1.0
        if nsplt > 1.0:
            frac = 1.0 / nsplt
            var = var * frac


@utils.stencil()
def cmax_stencil1(cx: sd, cy: sd, cmax: sd):
    with computation(PARALLEL), interval(...):
        abscx = cx if cx > 0 else -cx
        abscy = cy if cy > 0 else cy
        cmax = abscx if abscx > abscy else abscy


@utils.stencil()
def cmax_stencil2(cx: sd, cy: sd, sin_sg5: sd, cmax: sd):
    with computation(PARALLEL), interval(...):
        abscx = cx if cx > 0 else -cx
        abscy = cy if cy > 0 else cy
        tmpmax = abscx if abscx > abscy else abscy
        cmax = tmpmax + 1.0 - sin_sg5


@utils.stencil()
def dp_fluxadjustment(dp1: sd, mfx: sd, mfy: sd, rarea: sd, dp2: sd):
    with computation(PARALLEL), interval(...):
        dp2 = dp1 + (mfx - mfx[1, 0, 0] + mfy - mfy[0, 1, 0]) * rarea


@gtscript.function
def adjustment(q, dp1, fx, fy, rarea, dp2):
    return (q * dp1 + (fx - fx[1, 0, 0] + fy - fy[0, 1, 0]) * rarea) / dp2


@utils.stencil()
def q_adjust(q: sd, dp1: sd, fx: sd, fy: sd, rarea: sd, dp2: sd):
    with computation(PARALLEL), interval(...):
        q = adjustment(q, dp1, fx, fy, rarea, dp2)


@utils.stencil()
def q_other_adjust(q: sd, qset: sd, dp1: sd, fx: sd, fy: sd, rarea: sd, dp2: sd):
    with computation(PARALLEL), interval(...):
        qset = adjustment(q, dp1, fx, fy, rarea, dp2)


def compute(
    qvapor,
    qliquid,
    qice,
    qrain,
    qsnow,
    qgraupel,
    qcld,
    dp1,
    mfxd,
    mfyd,
    cxd,
    cyd,
    mdt,
    nq,
    comm,
):
    grid = spec.grid

    # start HALO update on q (in dyn_core in fortran -- just has started when this function is called...)
    xfx = utils.make_storage_from_shape(
        qvapor.data.shape, origin=grid.compute_x_origin()
    )
    yfx = utils.make_storage_from_shape(
        qvapor.data.shape, origin=grid.compute_y_origin()
    )
    fx = utils.make_storage_from_shape(qvapor.data.shape, origin=grid.compute_origin())
    fy = utils.make_storage_from_shape(qvapor.data.shape, origin=grid.compute_origin())
    ra_x = utils.make_storage_from_shape(
        qvapor.data.shape, origin=grid.compute_x_origin()
    )
    ra_y = utils.make_storage_from_shape(
        qvapor.data.shape, origin=grid.compute_y_origin()
    )
    cmax = utils.make_storage_from_shape(
        qvapor.data.shape, origin=grid.compute_origin()
    )
    dp2 = utils.make_storage_from_shape(qvapor.data.shape, origin=grid.compute_origin())
    flux_x(
        cxd,
        grid.dxa,
        grid.dy,
        grid.sin_sg3,
        grid.sin_sg1,
        xfx,
        origin=grid.compute_x_origin(),
        domain=grid.domain_y_compute_xbuffer(),
    )
    flux_y(
        cyd,
        grid.dya,
        grid.dx,
        grid.sin_sg4,
        grid.sin_sg2,
        yfx,
        origin=grid.compute_y_origin(),
        domain=grid.domain_x_compute_ybuffer(),
    )
    split = int(grid.npz / 6)
    cmax_stencil1(
        cxd, cyd, cmax, origin=grid.compute_origin(), domain=(grid.nic, grid.njc, split)
    )
    cmax_stencil2(
        cxd,
        cyd,
        grid.sin_sg5,
        cmax,
        origin=(grid.is_, grid.js, split),
        domain=(grid.nic, grid.njc, grid.npz - split + 1),
    )
    cmax_flat = np.amax(cmax, axis=(0, 1))
    # cmax_flat is a gt4py storage still, but of dimension [npz+1]...

    cmax_max_all_ranks = cmax_flat.data
    # comm.Allreduce(cmax_flat, cmax_max_all_ranks, op=MPI.MAX)

    nsplt = np.floor(1.0 + cmax_max_all_ranks)

    # for nsplit > 1
    nsplt3d = utils.make_storage_data(nsplt, cmax.shape, origin=grid.compute_origin())
    cmax_split(
        cxd,
        nsplt3d,
        origin=grid.compute_x_origin(),
        domain=grid.domain_y_compute_xbuffer(),
    )
    cmax_split(
        xfx,
        nsplt3d,
        origin=grid.compute_x_origin(),
        domain=grid.domain_y_compute_xbuffer(),
    )
    cmax_split(
        mfxd,
        nsplt3d,
        origin=grid.compute_origin(),
        domain=grid.domain_shape_compute_x(),
    )
    cmax_split(
        cyd,
        nsplt3d,
        origin=grid.compute_y_origin(),
        domain=grid.domain_x_compute_ybuffer(),
    )
    cmax_split(
        yfx,
        nsplt3d,
        origin=grid.compute_y_origin(),
        domain=grid.domain_x_compute_ybuffer(),
    )
    cmax_split(
        mfyd,
        nsplt3d,
        origin=grid.compute_origin(),
        domain=grid.domain_shape_compute_y(),
    )
    ns = 1  # TODO remove
    # complete HALO update on q
    for q in [qvapor, qliquid, qice, qrain, qsnow, qgraupel]:
        comm.halo_update(q, n_points=utils.halo)

    ra_x_stencil(
        grid.area,
        xfx,
        ra_x,
        origin=grid.compute_x_origin(),
        domain=grid.domain_y_compute_x(),
    )
    ra_y_stencil(
        grid.area,
        yfx,
        ra_y,
        origin=grid.compute_y_origin(),
        domain=grid.domain_x_compute_y(),
    )
    for it in range(int(ns)):
        dp_fluxadjustment(
            dp1,
            mfxd,
            mfyd,
            grid.rarea,
            dp2,
            origin=grid.compute_origin(),
            domain=grid.domain_shape_compute(),
        )

    for q in [qvapor, qliquid, qice, qrain, qsnow, qgraupel]:
        for it in range(int(ns)):
            if ns != 1:
                raise Exception("untested")
                if it == 0:
                    # TODO 1d
                    qn2 = cp.copy(q.data)
                else:
                    qn2 = utils.make_storage_from_shape(
                        q.data.shape, origin=grid.compute_origin()
                    )
                fvtp2d.compute_no_sg(
                    qn2,
                    cxd,
                    cyd,
                    spec.namelist["hord_tr"],
                    xfx,
                    yfx,
                    ra_x,
                    ra_y,
                    fx,
                    fy,
                    mfx=mfxd,
                    mfy=mfyd,
                )
                if it < ns - 1:
                    q_adjust(
                        qn2,
                        dp1,
                        fx,
                        fy,
                        grid.rarea,
                        dp2,
                        origin=grid.compute_origin(),
                        domain=grid.domain_shape_compute(),
                    )
                else:
                    q_other_adjust(
                        qn2,
                        q,
                        dp1,
                        fx,
                        fy,
                        grid.rarea,
                        dp2,
                        origin=grid.compute_origin(),
                        domain=grid.domain_shape_compute(),
                    )
            else:
                fvtp2d.compute_no_sg(
                    q.data,
                    cxd,
                    cyd,
                    spec.namelist["hord_tr"],
                    xfx,
                    yfx,
                    ra_x,
                    ra_y,
                    fx,
                    fy,
                    mfx=mfxd,
                    mfy=mfyd,
                )
                q_adjust(
                    q.data,
                    dp1,
                    fx,
                    fy,
                    grid.rarea,
                    dp2,
                    origin=grid.compute_origin(),
                    domain=grid.domain_shape_compute(),
                )

            if it < ns - 1:
                raise Exception("untested")
                dp1 = cp.copy(
                    dp2, origin=grid.compute_origin, domain=grid.domain_shape_compute()
                )
                # HALO UPDATE qn2
                comm.halo_update(comm, n_points=utils.halo)
