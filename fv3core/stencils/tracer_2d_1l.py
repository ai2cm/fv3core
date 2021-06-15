import math

import gt4py.gtscript as gtscript
from gt4py.gtscript import PARALLEL, computation, interval

import fv3core._config as spec
import fv3core.stencils.fxadv
import fv3core.utils
import fv3core.utils.global_config as global_config
import fv3core.utils.gt4py_utils as utils
import fv3gfs.util
from fv3core.decorators import FrozenStencil
from fv3core.stencils.fvtp2d import FiniteVolumeTransport
from fv3core.utils.typing import FloatField, FloatFieldIJ


def flux_x_compute(
    cx: FloatField,
    dxa: FloatFieldIJ,
    dy: FloatFieldIJ,
    sin_sg1: FloatFieldIJ,
    sin_sg3: FloatFieldIJ,
    xfx: FloatField,
):
    with computation(PARALLEL), interval(...):
        xfx = cx * dxa[-1, 0] * dy * sin_sg3[-1, 0] if cx > 0 else cx * dxa * dy * sin_sg1

def flux_y_compute(
    cy: FloatField,
    dya: FloatFieldIJ,
    dx: FloatFieldIJ,
    sin_sg2: FloatFieldIJ,
    sin_sg4: FloatFieldIJ,
    yfx: FloatField,
):
    with computation(PARALLEL), interval(...):
        yfx = cy * dya[0, -1] * dx * sin_sg4[0, -1] if cy > 0 else cy * dya * dx * sin_sg2


def cmax_multiply_by_frac(
    cxd: FloatField,
    xfx: FloatField,
    mfxd: FloatField,
    cyd: FloatField,
    yfx: FloatField,
    mfyd: FloatField,
    n_split: int,
):
    """multiply all other inputs in-place by frac."""
    with computation(PARALLEL), interval(...):
        frac = 1.0 / n_split
        cxd = cxd * frac
        xfx = xfx * frac
        mfxd = mfxd * frac
        cyd = cyd * frac
        yfx = yfx * frac
        mfyd = mfyd * frac


def cmax_stencil1(cx: FloatField, cy: FloatField, cmax: FloatField):
    with computation(PARALLEL), interval(...):
        cmax = max(abs(cx), abs(cy))


def cmax_stencil2(
    cx: FloatField, cy: FloatField, sin_sg5: FloatField, cmax: FloatField
):
    with computation(PARALLEL), interval(...):
        cmax = max(abs(cx), abs(cy)) + 1.0 - sin_sg5


def dp_fluxadjustment(
    dp1: FloatField,
    mfx: FloatField,
    mfy: FloatField,
    rarea: FloatFieldIJ,
    dp2: FloatField,
):
    with computation(PARALLEL), interval(...):
        dp2 = dp1 + (mfx - mfx[1, 0, 0] + mfy - mfy[0, 1, 0]) * rarea


@gtscript.function
def adjustment(q, dp1, fx, fy, rarea, dp2):
    return (q * dp1 + (fx - fx[1, 0, 0] + fy - fy[0, 1, 0]) * rarea) / dp2


def q_adjust(
    q: FloatField,
    dp1: FloatField,
    fx: FloatField,
    fy: FloatField,
    rarea: FloatFieldIJ,
    dp2: FloatField,
):
    with computation(PARALLEL), interval(...):
        q = adjustment(q, dp1, fx, fy, rarea, dp2)


class TracerAdvection:
    """
    Performs horizontal advection on tracers.

    Corresponds to tracer_2D_1L in the Fortran code.
    """

    def __init__(self, comm: fv3gfs.util.CubedSphereCommunicator, namelist):
        self.comm = comm
        self.grid = spec.grid
        self._do_halo_exchange = global_config.get_do_halo_exchange()
        shape = self.grid.domain_shape_full(add=(1, 1, 1))
        origin = self.grid.compute_origin()
        self._tmp_xfx = utils.make_storage_from_shape(shape, origin)
        self._tmp_yfx = utils.make_storage_from_shape(shape, origin)
        self._tmp_fx = utils.make_storage_from_shape(shape, origin)
        self._tmp_fy = utils.make_storage_from_shape(shape, origin)
        self._tmp_dp = utils.make_storage_from_shape(shape, origin)
        self._tmp_qn2 = self.grid.quantity_wrap(
            utils.make_storage_from_shape(shape, origin),
            units="kg/m^2",
        )

        self._flux_x_compute = FrozenStencil(
            flux_x_compute,
            origin=self.grid.full_origin(add=(3, 0, 0)),
            domain=self.grid.domain_shape_full(add=(-3, 1, 0)),
        )
        self._flux_y_compute = FrozenStencil(
            flux_y_compute,
            origin=self.grid.full_origin(add=(0, 3, 0)),
            domain=self.grid.domain_shape_full(add=(1, -3, 0)),
        )
        self._cmax_multiply_by_frac = FrozenStencil(
            cmax_multiply_by_frac,
            origin=self.grid.full_origin(),
            domain=self.grid.domain_shape_full(add=(1, 1, 0)),
        )
        self._dp_fluxadjustment = FrozenStencil(
            dp_fluxadjustment,
            origin=self.grid.compute_origin(),
            domain=self.grid.domain_shape_compute(),
        )
        self._q_adjust = FrozenStencil(
            q_adjust,
            origin=self.grid.compute_origin(),
            domain=self.grid.domain_shape_compute(),
        )
        self.finite_volume_transport = FiniteVolumeTransport(namelist, namelist.hord_tr)
        # If use AllReduce, will need something like this:
        # self._tmp_cmax = utils.make_storage_from_shape(shape, origin)
        # self._cmax_1 = FrozenStencil(cmax_stencil1)
        # self._cmax_2 = FrozenStencil(cmax_stencil2)

    def __call__(self, tracers, dp1, mfxd, mfyd, cxd, cyd, mdt):
        # start HALO update on q (in dyn_core in fortran -- just has started when
        # this function is called...)
        self._flux_x_compute(
            cxd,
            self.grid.dxa,
            self.grid.dy,
            self.grid.sin_sg1,
            self.grid.sin_sg3,
            self._tmp_xfx,
        )
        self._flux_y_compute(
            cyd,
            self.grid.dya,
            self.grid.dx,
            self.grid.sin_sg2,
            self.grid.sin_sg4,
            self._tmp_yfx,
        )
        # # TODO for if we end up using the Allreduce and compute cmax globally
        # (or locally). For now, hardcoded.
        # split = int(self.grid.npz / 6)
        # self._cmax_1(
        #     cxd, cyd, self._tmp_cmax, origin=self.grid.compute_origin(),
        #     domain=(self.grid.nic, self.grid.njc, split)
        # )
        # self._cmax_2(
        #     cxd,
        #     cyd,
        #     self.grid.sin_sg5,
        #     self._tmp_cmax,
        #     origin=(self.grid.is_, self.grid.js, split),
        #     domain=(self.grid.nic, self.grid.njc, self.grid.npz - split + 1),
        # )
        # cmax_flat = np.amax(self._tmp_cmax, axis=(0, 1))
        # # cmax_flat is a gt4py storage still, but of dimension [npz+1]...

        # cmax_max_all_ranks = cmax_flat.data
        # # TODO mpi allreduce...
        # # comm.Allreduce(cmax_flat, cmax_max_all_ranks, op=MPI.MAX)

        cmax_max_all_ranks = 2.0
        n_split = math.floor(1.0 + cmax_max_all_ranks)
        # NOTE: cmax is not usually a single value, it varies with k, if return to
        # that, make n_split a column as well

        if n_split > 1.0:
            self._cmax_multiply_by_frac(
                cxd,
                self._tmp_xfx,
                mfxd,
                cyd,
                self._tmp_yfx,
                mfyd,
                n_split,
            )

        reqs = []
        if self._do_halo_exchange:
            reqs.clear()
            for q in tracers.values():
                reqs.append(self.comm.start_halo_update(q, n_points=utils.halo))
            for req in reqs:
                req.wait()

        dp2 = self._tmp_dp

        for it in range(int(n_split)):
            last_call = it == n_split - 1
            self._dp_fluxadjustment(
                dp1,
                mfxd,
                mfyd,
                self.grid.rarea,
                dp2,
            )
            for qname, q in tracers.items():
                self.finite_volume_transport(
                    q.storage,
                    cxd,
                    cyd,
                    self._tmp_xfx,
                    self._tmp_yfx,
                    self._tmp_fx,
                    self._tmp_fy,
                    mfx=mfxd,
                    mfy=mfyd,
                )
                self._q_adjust(
                    q.storage,
                    dp1,
                    self._tmp_fx,
                    self._tmp_fy,
                    self.grid.rarea,
                    dp2,
                )
            if not last_call:
                if self._do_halo_exchange:
                    reqs.clear()
                    for q in tracers.values():
                        reqs.append(self.comm.start_halo_update(q, n_points=utils.halo))
                    for req in reqs:
                        req.wait()

                # use variable assignment to avoid a data copy
                dp1, dp2 = dp2, dp1
