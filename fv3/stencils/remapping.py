#!/usr/bin/env python3
import fv3.utils.gt4py_utils as utils
import gt4py.gtscript as gtscript
import fv3._config as spec
from gt4py.gtscript import computation, interval, PARALLEL
import fv3.utils.global_constants as constants
import fv3.stencils.moist_cv as moist_cv
import fv3.stencils.saturation_adjustment as saturation_adjustment
import fv3.stencils.remapping_part1 as remap_part1
import fv3.stencils.remapping_part2 as remap_part2
import numpy as np
import fv3.stencils.copy_stencil as cp
sd = utils.sd


def compute(qvapor, qliquid, qrain, qsnow, qice, qgraupel, qcld, pt, delp, delz, peln, u, v, w, ua, va, cappa, q_con, pkz, pk, pe, hs, te0_2d, ps, wsd, omga, ak, bk, pfull, dp1, ptop, akap, zvir, last_step, consv_te, mdt, bdt, kord_tracer, do_adiabatic_init):
    grid = spec.grid
        
    if spec.namelist['do_sat_adj']:
        fast_mp_consv = not do_adiabatic_init and consv_te > constants.CONSV_MIN # TODO pass when change serialiazation data
        kmp = np.where(pfull[0, 0, :] > 10.e2)[0]
        kmp = kmp[0] if len(kmp) > 0 else grid.npz
        # TODO USE KMP WHEN YOU FIX SERIALIZATION DATA
        # Fortran does a qs_init here, but it does not need to happen here
    gz = utils.make_storage_from_shape(pt.shape, grid.compute_origin())
    cvm = utils.make_storage_from_shape(pt.shape, grid.compute_origin())
    te = utils.make_storage_from_shape(pt.shape, grid.default_origin())
    te_2d = utils.make_storage_from_shape(pt.shape, grid.compute_origin())
    zsum1 = utils.make_storage_from_shape(pt.shape, grid.compute_origin())
    remap_part1.compute(qvapor, qliquid, qrain, qsnow, qice, qgraupel, qcld, pt, delp, delz, peln, u, v, w, ua, cappa, q_con, pkz, pk, pe, hs, te, ps, wsd, omga, ak, bk, gz, cvm, ptop, akap, zvir)
    remap_part2.compute(qvapor, qliquid, qrain, qsnow, qice, qgraupel, qcld, pt, delp, delz, peln, u, v, w, ua, cappa, q_con, gz, pkz, pk, pe, hs, te_2d, te0_2d, te, cvm, zsum1, ptop, akap, zvir, last_step, bdt, mdt, consv_te, kmp, fast_mp_consv, do_adiabatic_init)
