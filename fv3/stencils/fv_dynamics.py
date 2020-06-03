#!/usr/bin/env python3
import fv3.utils.gt4py_utils as utils
import gt4py.gtscript as gtscript
import fv3._config as spec
import fv3.utils.global_constants as constants
from gt4py.gtscript import computation, interval, PARALLEL
import fv3.stencils.moist_cv as moist_cv
import fv3.stencils.rayleigh_super as rayleigh_super
import fv3.stencils.dyn_core as dyn_core
import fv3.stencils.copy_stencil as cp
import fv3.stencils.tracer_2d_1l as tracer_2d_1l
import fv3.stencils.remapping as lagrangian_to_eulerian
import fv3.stencils.del2cubed as del2cubed
import fv3.stencils.neg_adj3 as neg_adj3
from fv3.stencils.c2l_ord import compute_cubed_to_latlon
import fv3util
import numpy as np
from types import SimpleNamespace
sd = utils.sd


@utils.stencil()
def init_ph_columns(ak: sd, bk: sd, pfull: sd, ph1: sd, ph2: sd, p_ref: float):
    with computation(PARALLEL), interval(...):
        ph1 = ak + bk * p_ref
        ph2 = ak[0, 0, 1] + bk[0, 0, 1] * p_ref
        #pfull = (ph2 - ph1) / log(ph2 / ph1)

@utils.stencil()
def pt_adjust(pkz: sd, dp1: sd, q_con: sd, pt: sd):
    with computation(PARALLEL), interval(...):
        pt = pt * (1. + dp1) * (1. - q_con / pkz)

@utils.stencil()
def set_omega(delp: sd, delz: sd, w: sd, omga: sd):
    with computation(PARALLEL), interval(...):
        omga = delp / delz * w



def fvdyn_temporaries(shape):
    grid = spec.grid
    tmps = {}
    halo_vars = ["cappa"]
    storage_vars = ['te_2d', 'dp1', 'ph1', 'ph2', 'dp1', 'wsd']
    column_vars = ["pfull", "gz", "cvm"]
    plane_vars = ["te_2d", "te0_2d"]
    utils.storage_dict(tmps, halo_vars + storage_vars + column_vars + plane_vars, shape, grid.default_origin())
    for q in halo_vars:
        tmps[q + "_quantity"] = grid.quantity_wrap(tmps[q])
    return tmps

def compute(state, comm):
    grid = spec.grid
    state.update(fvdyn_temporaries(state['u'].shape))
    state = SimpleNamespace(**state)
    agrav = 1.0 / constants.GRAV
    state.rdg = - constants.RDGAS / agrav
    state.akap = constants.KAPPA
    state.dt2 = 0.5 * state.bdt
    nq = state.nq_tot - spec.namelist['dnats']
    init_ph_columns(state.ak, state.bk, state.pfull, state.ph1, state.ph2, spec.namelist['p_ref'], origin=grid.compute_origin(), domain=grid.domain_shape_compute()) # TODO put pfull calc below into stencil
    tmpslice = grid.compute_interface()
    state.pfull[tmpslice] = (state.ph2[tmpslice] - state.ph1[tmpslice]) / np.log(state.ph2[tmpslice] / state.ph1[tmpslice])
    if spec.namelist['hydrostatic']:
        raise Exception('Hydrostatic is not implemented')
    
    moist_cv.fv_setup(state.pt, state.pkz, state.delz, state.delp, state.cappa, state.q_con,
                      state.zvir, state.qvapor, state.qliquid, state.qice, state.qrain, state.qsnow, state.qgraupel,
                      state.cvm, state.dp1)
    
    if state.consv_te > 0 and not state.do_adiabatic_init:
        # NOTE untested
        moist_cv.compute_total_energy(state.u, state.v, state.w, state.delz, state.pt, state.delp, state.qc,
                                      state.pe, state.peln, state.phis, state.zvir, state.te_2d,
                                      state.qvapor, state.qliquid, state.qice, state.qrain, state.qsnow, state.qgraupel)
      
    if (not spec.namelist['RF_fast']) and spec.namelist['tau'] != 0:
        if grid.grid_type < 4:
            rayleigh_super.compute(state.u, state.v, state.w, state.ua, state.va, state.pt, state.delz, state.phis, state.bdt, state.ptop, state.pfull, comm)
    
    if spec.namelist['adiabatic'] and spec.namelist['kord_tm'] > 0:
        raise Exception('unimplemented namelist options adiabatic with positive kord_tm')
    else:
        pt_adjust(state.pkz, state.dp1, state.q_con, state.pt, origin=grid.compute_origin(), domain=grid.domain_shape_compute())
    
    last_step = False
    k_split = spec.namelist['k_split']
    state.mdt = state.bdt / k_split
    
    for n_map in range(k_split):
        state.n_map_step = n_map + 1
        if n_map == k_split - 1:
            last_step = True
        cp.copy_stencil(state.delp, state.dp1, origin=grid.default_origin(), domain=grid.domain_shape_standard())
        dyn_core.compute(vars(state), comm)
    
        if not spec.namelist['inline_q'] and nq != 0:
            if spec.namelist['z_tracer']:
                tracer_2d_1l.compute(state.qvapor_quantity, state.qliquid_quantity, state.qice_quantity, state.qrain_quantity, state.qsnow_quantity,
                                     state.qgraupel_quantity, state.qcld_quantity, state.dp1, state.mfxd, state.mfyd,
                                     state.cxd, state.cyd, state.mdt, nq, comm)
            else:
                raise Exception('tracer_2d no =t implemented, turn on z_tracer')
    '''
        if grid.npz > 4:
            kord_tracer = np.ones(nq) * spec.namelist['kord_tr']
            kord_tracer[6] = 9
            do_omega = spec.namelist['hydrostatic'] and last_step
            lagrangian_to_eulerian.compute(state.qvapor, state.qliquid, state.qrain, state.qsnow,
                                           state.qice, state.qgraupel, state.qcld, state.pt, state.delp,
                                           state.delz, state.peln, state.u, state.v, state.w, state.ua,
                                           state.va, state.cappa, state.q_con, state.pkz, state.pk,
                                           state.pe, state.phis, state.te0_2d, state.ps, state.wsd,
                                           state.omga, state.ak, state.bk, state.pfull, state.dp1,
                                           state.ptop, state.akap, state.zvir, last_step,
                                           state.consv_te, state.mdt, state.bdt, kord_tracer, state.do_adiabatic_init,)
            if last_step and not spec.namelist['hydrostatic']:
                set_omega(state.delp, state.delz, state.w, state.omga, origin=grid.compute_origin(), domain=grid.domain_shape_compute())
    
            if spec.namelist['nf_omega'] > 0:
                del2cubed.compute(state.omga, spec.namelist['nf_omega'], 0.18 * grid.da_min, grid.npz)
    if nq == 7:
        neg_adj3.compute(state.qvapor, state.qliquid, state.qrain, state.qsnow, state.qice, state.qgraupel, state.qcld, state.pt, state.delp, state.delz, state.peln)
    else:
        raise Exception('Unimplemented, anything but 7 water species')
    compute_cubed_to_latlon(state.u, state.v, state.ua, state.va)
    
    '''
