import math

import gt4py.gtscript as gtscript
from gt4py.gtscript import PARALLEL, computation, horizontal, interval, region

import fv3core._config as spec
import fv3core.stencils.fxadv
import fv3core.utils
import fv3core.utils.global_config as global_config
import fv3core.utils.gt4py_utils as utils
import fv3gfs.util
from fv3core.decorators import FrozenStencil
from fv3core.stencils.fvtp2d import FiniteVolumeTransport
from fv3core.utils.typing import FloatField, FloatFieldIJ


@gtscript.function
def flux_x(cx, dxa, dy, sin_sg3, sin_sg1, xfx):
    from __externals__ import local_ie, local_is, local_je, local_js

    with horizontal(region[local_is : local_ie + 2, local_js - 3 : local_je + 4]):
        xfx = (
            cx * dxa[-1, 0] * dy * sin_sg3[-1, 0] if cx > 0 else cx * dxa * dy * sin_sg1
        )
    return xfx


@gtscript.function
def flux_y(cy, dya, dx, sin_sg4, sin_sg2, yfx):
    from __externals__ import local_ie, local_is, local_je, local_js

    with horizontal(region[local_is - 3 : local_ie + 4, local_js : local_je + 2]):
        yfx = (
            cy * dya[0, -1] * dx * sin_sg4[0, -1] if cy > 0 else cy * dya * dx * sin_sg2
        )
    return yfx


def flux_compute(
    cx: FloatField,
    cy: FloatField,
    dxa: FloatFieldIJ,
    dya: FloatFieldIJ,
    dx: FloatFieldIJ,
    dy: FloatFieldIJ,
    sin_sg1: FloatFieldIJ,
    sin_sg2: FloatFieldIJ,
    sin_sg3: FloatFieldIJ,
    sin_sg4: FloatFieldIJ,
    xfx: FloatField,
    yfx: FloatField,
):
    with computation(PARALLEL), interval(...):
        xfx = flux_x(cx, dxa, dy, sin_sg3, sin_sg1, xfx)
        yfx = flux_y(cy, dya, dx, sin_sg4, sin_sg2, yfx)


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


def loop_temporaries_copy(
    tmp_dp1_orig: FloatField,
    q: FloatField,
    dp1: FloatField,
    tmp_qn2: FloatField,
):
    with computation(PARALLEL), interval(...):
        dp1 = tmp_dp1_orig
        tmp_qn2 = q


@gtscript.function
def adjustment(q, dp1, fx, fy, rarea, dp2):
    return (q * dp1 + (fx - fx[1, 0, 0] + fy - fy[0, 1, 0]) * rarea) / dp2


def q_adjust(
    q_in: FloatField,
    q_out: FloatField,
    dp1: FloatField,
    fx: FloatField,
    fy: FloatField,
    rarea: FloatFieldIJ,
    dp2: FloatField,
):
    with computation(PARALLEL), interval(...):
        q_out = adjustment(q_in, dp1, fx, fy, rarea, dp2)


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

        ax_offsets = fv3core.utils.axis_offsets(
            self.grid, self.grid.full_origin(), self.grid.domain_shape_full()
        )
        local_axis_offsets = {}
        for axis_offset_name, axis_offset_value in ax_offsets.items():
            if "local" in axis_offset_name:
                local_axis_offsets[axis_offset_name] = axis_offset_value

        self._flux_compute = FrozenStencil(
            flux_compute,
            origin=self.grid.full_origin(),
            domain=self.grid.domain_shape_full(add=(1, 1, 0)),
            externals=local_axis_offsets,
        )
        self._cmax_multiply_by_frac = FrozenStencil(
            cmax_multiply_by_frac,
            origin=self.grid.full_origin(),
            domain=self.grid.domain_shape_full(add=(1, 1, 0)),
            externals=local_axis_offsets,
        )
        self._loop_temporaries_copy = FrozenStencil(
            loop_temporaries_copy,
            origin=self.grid.full_origin(),
            domain=self.grid.domain_shape_full(),
            externals=local_axis_offsets,
        )
        self._dp_fluxadjustment = FrozenStencil(
            dp_fluxadjustment,
            origin=self.grid.compute_origin(),
            domain=self.grid.domain_shape_compute(),
            externals=local_axis_offsets,
        )
        self._q_adjust = FrozenStencil(
            q_adjust,
            origin=self.grid.compute_origin(),
            domain=self.grid.domain_shape_compute(),
            externals=local_axis_offsets,
        )
        self.finite_volume_transport = FiniteVolumeTransport(namelist, namelist.hord_tr)
        # If use AllReduce, will need something like this:
        # self._tmp_cmax = utils.make_storage_from_shape(shape, origin)
        # self._cmax_1 = FrozenStencil(cmax_stencil1)
        # self._cmax_2 = FrozenStencil(cmax_stencil2)

    def __call__(self, tracers, dp1, mfxd, mfyd, cxd, cyd, mdt):
        # start HALO update on q (in dyn_core in fortran -- just has started when
        # this function is called...)
        self._flux_compute(
            cxd,
            cyd,
            self.grid.dxa,
            self.grid.dya,
            self.grid.dx,
            self.grid.dy,
            self.grid.sin_sg1,
            self.grid.sin_sg2,
            self.grid.sin_sg3,
            self.grid.sin_sg4,
            self._tmp_xfx,
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

        if self._do_halo_exchange:
            reqs = {}
            for qname, q in tracers.items():
                reqs[qname] = self.comm.start_halo_update(q, n_points=utils.halo)

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
                if self._do_halo_exchange:
                    reqs[qname].wait()
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
                    q.storage,
                    dp1,
                    self._tmp_fx,
                    self._tmp_fy,
                    self.grid.rarea,
                    dp2,
                )
                if not last_call and self._do_halo_exchange:
                    utils.device_sync()
                    reqs[qname] = self.comm.start_halo_update(q, n_points=utils.halo)

            if not last_call:
                # use variable assignment to avoid a data copy
                dp1, dp2 = dp2, dp1
