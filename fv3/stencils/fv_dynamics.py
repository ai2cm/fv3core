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
sd = utils.sd


@utils.stencil()
def init_ph_columns(ak:sd, bk: sd, pfull: sd, ph1: sd, ph2: sd, p_ref: float):
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
    with computation(PARALLEL), interval(..):
        omga = delp / delz * w
    
def compute(state, comm):
    grid = spec.grid
    agrav = 1.0 / constants.GRAV
    dt2 = 0.5 * bdt
    rdg = - constants.RDGAS / agrav
    for tmpvar in ['te_2d', 'dp1', 'cappa', 'ph1', 'ph2']:
        state[tmpvar] = utils.make_storage_from_shape(state['u'].shape, grid.compute_origin())
   
    akap = constants.KAPPA
    init_ph_columns(state.ak, state.bk, state.pfull, state.ph1, state.ph2, spec.namelist['p_ref'], origin=grid.compute_origin(), domain=grid.domain_shape_compute()) # TODO put pfull into stencil
    state.pfull = (state.ph2 - state.ph1) / np.log(state.ph2 / state.ph1)
    if spec.namelist['hydrostatic']:
        raise Exception('Hydrostatic is not implemented')
    moist_cv.fv_setup(pt, pkz, delz, delp, cappa, q_con,
                      zvir, qvapor, qliquid, qice, qrain, qsnow, qgraupel,
                      cvm, dp1)
    if consv_te > 0 and not do_adiabatic_init:
        moist_cv.compute_total_energy(u, v, w, delz, pt, delp, qc,
                             pe, peln, hs, zvir, te_2d,
                             qvapor, qliquid, qice, qrain, qsnow, qgraupel)
        
    if (not spec.namelist['RF_fast']) and spec.namelist['tau'] != 0:
        if grid.grid_type < 4:
            rayleigh_super.compute(u, v, w, ua, va, pt, delz, phis, bdt, ptop, pfull, comm)
    if spec.namelist['adiabatic'] and spec.namelist['kord_tm'] > 0:
        raise Exception('unimplemented namelist options adiabatic with positive kord_tm')
    else:
        pt_adjust(pkz, dp1, q_con, pt, origin=grid.compute_origin(), domain=grid.domain_shape_compute())

    last_step = False
    k_split = spec.namelist['k_split']
    mdt = bdt / ksplit

    for n_map in range(k_split):
        if n_map == k_split - 1:
            last_step = True
        dp1 = cp.copy(delp, origin=grid.default_origin(), domain=grid.domain_shape_standard())
        dyn_core.compute(state, comm)
    
        if not spec.namelist['inline_q']: # TODO and nq !=0
            if spec.namelist['z_tracer']:
                tracer_2d_1l.compute(state, comm)
            else:
                raise Exception('tracer_2d no =t implemented, turn on z_tracer')
        if grid.npz > 4:
            # TODO kord_tracer set to namelist except kord_tr, handle lower down?
            do_omega = spec.namelist['hydrostatic'] and last_step
            remapping.compute()
            if last_step and not spec.namelist['hydrostatic']:
                set_omega(delp, delz, w, omga, origin=grid.compute_origin(), domain=grid.domain_shape_compute())
            
            if spec.namelist['nf_omega'] > 0:
                del2_cubed.compute()
    neg_adj3.compute()
    cubed_to_latlon.compute()
