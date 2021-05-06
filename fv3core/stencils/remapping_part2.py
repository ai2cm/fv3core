from gt4py.gtscript import BACKWARD, FORWARD, PARALLEL, computation, interval

import fv3core._config as spec
import fv3core.stencils.basic_operations as basic
import fv3core.stencils.moist_cv as moist_cv
import fv3core.utils.global_constants as constants
import fv3core.utils.gt4py_utils as utils
from fv3core.decorators import FrozenStencil, gtstencil
from fv3core.stencils.saturation_adjustment import SatAdjust3d
from fv3core.utils.typing import FloatField, FloatFieldIJ, FloatFieldK


@gtstencil
def copy_from_below(a: FloatField, b: FloatField):
    with computation(PARALLEL), interval(1, None):
        b = a[0, 0, -1]


@gtstencil
def init_phis(hs: FloatField, delz: FloatField, phis: FloatField, te_2d: FloatFieldIJ):
    with computation(BACKWARD):
        with interval(-1, None):
            te_2d = 0.0
            phis = hs
        with interval(0, -1):
            te_2d = 0.0
            phis = phis[0, 0, 1] - constants.GRAV * delz


def sum_te(te: FloatField, te0_2d: FloatField):
    with computation(FORWARD):
        with interval(0, None):
            te0_2d = te0_2d[0, 0, -1] + te


class Remapping_Part2:
    def __init__(self):
        self.grid = spec.grid
        self.namelist = spec.namelist
        self._saturation_adjustment = utils.cached_stencil_class(SatAdjust3d)(
            cache_key="satadjust3d"
        )
        self._copy_from_below_stencil = FrozenStencil(
            copy_from_below,
            origin=self.grid.compute_origin(),
            domain=self.grid.domain_shape_compute(),
        )

        self._moist_cv_last_step_stencil = FrozenStencil(
            moist_cv.moist_pt_last_step,
            origin=(self.grid.is_, self.grid.js, 0),
            domain=(self.grid.nic, self.grid.njc, self.grid.npz + 1),
        )

        self._basic_adjust_divide_stencil = FrozenStencil(
            basic.adjust_divide_stencil,
            origin=self.grid.compute_origin(),
            domain=self.grid.domain_shape_compute(),
        )

        self.do_sat_adjust = self.namelist.do_sat_adj

        # TODO : When remapping becomes an object, these stencils can be declared in
        #        __init__ when (I think) pFull can be passed into __init__
        # self._saturation_adjustment_stencil = FrozenStencil(SatAdjust3d,...)
        # self._sum_te_stencil = FrozenStencil(sum_te,...)

        # TODO : When state.pt from fv_dynamics can be passed into __init__, these
        #        storages can be declared in __init__
        # phis = utils.make_storage_from_shape(
        #     pt.shape, grid.compute_origin(), cache_key="remapping_part2_phis"
        # )
        # te_2d = utils.make_storage_from_shape(
        #     pt.shape[0:2], grid.compute_origin(), cache_key="remapping_part2_te_2d"
        # )
        # zsum1 = utils.make_storage_from_shape(
        #     pt.shape[0:2], grid.compute_origin(), cache_key="remapping_part2_zsum1"
        # )

    def __call__(
        self,
        qvapor: FloatField,
        qliquid: FloatField,
        qice: FloatField,
        qrain: FloatField,
        qsnow: FloatField,
        qgraupel: FloatField,
        qcld: FloatField,
        pt: FloatField,
        delp: FloatField,
        delz: FloatField,
        peln: FloatField,
        u: FloatField,
        v: FloatField,
        w: FloatField,
        ua: FloatField,
        cappa: FloatField,
        q_con: FloatField,
        gz: FloatField,
        pkz: FloatField,
        pk: FloatField,
        pe: FloatField,
        hs: FloatFieldIJ,
        te0_2d: FloatFieldIJ,
        te: FloatField,
        cvm: FloatField,
        pfull: FloatFieldK,
        ptop: float,
        akap: float,
        r_vir: float,
        last_step: bool,
        pdt: float,
        mdt: float,
        consv: float,
        do_adiabatic_init: bool,
    ):
        saturation_adjustment = utils.cached_stencil_class(SatAdjust3d)(
            cache_key="satadjust3d"
        )

        sum_te_stencil = gtstencil(func=sum_te)

        self._copy_from_below_stencil(ua, pe)
        dtmp = 0.0
        phis = utils.make_storage_from_shape(
            pt.shape, self.grid.compute_origin(), cache_key="remapping_part2_phis"
        )
        te_2d = utils.make_storage_from_shape(
            pt.shape[0:2], self.grid.compute_origin(), cache_key="remapping_part2_te_2d"
        )
        zsum1 = utils.make_storage_from_shape(
            pt.shape[0:2], self.grid.compute_origin(), cache_key="remapping_part2_zsum1"
        )
        if self.do_sat_adjust:
            fast_mp_consv = not do_adiabatic_init and consv > constants.CONSV_MIN
            # TODO pfull is a 1d var
            kmp = self.grid.npz - 1
            for k in range(pfull.shape[0]):
                if pfull[k] > 10.0e2:
                    kmp = k
                    break
        if last_step and not do_adiabatic_init:
            if consv > constants.CONSV_MIN:
                raise NotImplementedError(
                    "We do not support consv_te > 0.001 "
                    "because that would trigger an allReduce"
                )
            elif consv < -constants.CONSV_MIN:
                raise Exception(
                    "Unimplemented/untested case consv("
                    + str(consv)
                    + ")  < -CONSV_MIN("
                    + str(-constants.CONSV_MIN)
                    + ")"
                )

        if self.do_sat_adjust:

            kmp_origin = (self.grid.is_, self.grid.js, kmp)
            kmp_domain = (self.grid.nic, self.grid.njc, self.grid.npz - kmp)
            saturation_adjustment(
                te,
                qvapor,
                qliquid,
                qice,
                qrain,
                qsnow,
                qgraupel,
                qcld,
                hs,
                peln,
                delp,
                delz,
                q_con,
                pt,
                pkz,
                cappa,
                r_vir,
                mdt,
                fast_mp_consv,
                last_step,
                akap,
                kmp,
            )
            if fast_mp_consv:
                sum_te_stencil(te, te0_2d, origin=kmp_origin, domain=kmp_domain)
        if last_step:
            self._moist_cv_last_step_stencil(
                qvapor,
                qliquid,
                qrain,
                qsnow,
                qice,
                qgraupel,
                gz,
                pt,
                pkz,
                dtmp,
                r_vir,
            )
        else:
            self._basic_adjust_divide_stencil(pkz, pt)
