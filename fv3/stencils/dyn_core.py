#!/usr/bin/env python3
import fv3.utils.gt4py_utils as utils

import gt4py.gtscript as gtscript
import fv3._config as spec
from types import SimpleNamespace 
from gt4py.gtscript import computation, interval, PARALLEL
import fv3.stencils.d2a2c_vect as d2a2c
import math
import fv3.utils.global_constants as constants
import fv3.stencils.basic_operations as basic
import fv3.stencils.c_sw as c_sw
import fv3.stencils.copy_stencil as cp
import fv3.stencils.updatedzc as updatedzc
import fv3.stencils.updatedzd as updatedzd
import fv3.stencils.riem_solver_c as riem_solver_c
import fv3.stencils.riem_solver3 as riem_solver3
import fv3.stencils.pgradc as pgradc
import fv3.stencils.d_sw as d_sw
import fv3.stencils.pe_halo as pe_halo
import fv3.stencils.pk3_halo as pk3_halo
import fv3.stencils.nh_p_grad as nh_p_grad
import fv3.stencils.del2cubed as del2cubed
import fv3.stencils.temperature_adjust as temperature_adjust
import fv3util
import copy
sd = utils.sd

HUGE_R = 1.0e40

# NOTE in Fortran these are columns
@utils.stencil()
def dp_ref_compute(ak: sd, bk: sd, dp_ref: sd):
    with computation(PARALLEL), interval(0, -1):
        dp_ref = ak[0, 0, 1] - ak + (bk[0, 0, 1] - bk) * 1.0e5


@utils.stencil()
def set_gz(zs: sd, delz:sd, gz: sd):
    with computation(BACKWARD):
        with interval(-1, None):
            gz[0, 0, 0] = zs
        with interval(0, -1):
            gz[0, 0, 0] = gz[0, 0, 1] - delz
     

@utils.stencil()
def set_pem(delp: sd, pem:sd, ptop:float):
    with computation(FORWARD):
        with interval(0, 1):
            pem[0, 0, 0] = ptop
        with interval(1, None):
            pem[0, 0, 0] = pem[0, 0, -1] + delp

@utils.stencil()
def heatadjust_temperature_lowlevel(pt: sd, heat_source: sd, delp: sd, pkz: sd, cp_air):
    with computation(PARALLEL), interval(...):
        pt[0, 0, 0] = pt + heat_source / (cp_air * delp * pkz)

def get_n_con():
    if spec.namelist['convert_ke'] or spec.namelist['vtdm4'] > 1.0e-4:
        n_con = spec.grid.npz
    else:
        if spec.namelist['d2_bg_k1'] < 1.0e-3:
            n_con = 0
        else:
            if spec.namelist['d2_bg_k2'] < 1.0e-3:
                n_con = 1
            else:
                n_con = 2
    return n_con


def dyncore_temporaries(shape):
    grid = spec.grid
    tmps = {}
    utils.storage_dict(tmps, ['ut', 'vt', 'gz', 'zh', 'pem', 'ws3', 'pkc', 'pk3', 'heat_source', 'divgd'], shape, grid.default_origin())
    utils.storage_dict(tmps, ['crx', 'xfx'], shape, grid.compute_x_origin())
    utils.storage_dict(tmps, ['cry', 'yfx'], shape, grid.compute_y_origin())
    tmps['heat_source_quantity'] = grid.quantity_wrap(tmps['heat_source'], dims=[fv3util.X_DIM, fv3util.Y_DIM, fv3util.Z_DIM])
    for q in ['gz', 'pkc', 'zh']:
        tmps[q + '_quantity'] = grid.quantity_wrap(tmps[q], dims=[fv3util.X_DIM, fv3util.Y_DIM, fv3util.Z_INTERFACE_DIM])
    tmps['divgd_quantity'] = grid.quantity_wrap(tmps['divgd'], dims=[fv3util.X_INTERFACE_DIM, fv3util.Y_INTERFACE_DIM, fv3util.Z_DIM])
    return tmps


def compute(data, comm):
    #u, v, w, delz, delp, pt, pe, pk, phis, wsd, omga, ua, va, uc, vc, mfxd, mfyd, cxd, cyd, pkz, peln, q_con, ak, bk, diss_estd, cappa, mdt, n_split, akap, ptop, pfull, n_map_step, comm):
    grid = spec.grid
    init_step = data['n_map_step'] == 1
    end_step = data['n_map_step'] == spec.namelist['k_split']
    akap = data['akap']
    # peln1 = math.log(ptop)
    # ptk = ptop**akap
    dt = data['mdt'] / data['n_split']
    dt2 = 0.5 * dt
    hydrostatic = spec.namelist['hydrostatic']
    rgrav = 1.0 / constants.GRAV
    n_split = data['n_split']
    # TODO -- put defaults into code
    # m_split = 1. + abs(dt_atmos)/real(k_split*n_split*abs(p_split))
    # n_split = nint( real(n0split)/real(k_split*abs(p_split)) * stretch_fac + 0.5 )
    ms = max(1, spec.namelist['m_split'] / 2.0)
    shape = data['delz'].shape
  
    # NOTE in Fortran model the halo update starts happens in fv_dynamics, not here
  
    reqs = {}
    for halovar in ['q_con_quantity', 'cappa_quantity','delp_quantity', 'pt_quantity']:
        reqs[halovar] = comm.start_halo_update(data[halovar], n_points=utils.halo)
    reqs_vector = comm.start_vector_halo_update(data['u_quantity'], data['v_quantity'], n_points=utils.halo)
    reqs['q_con_quantity'].wait()
    reqs['cappa_quantity'].wait()
 
    data.update(dyncore_temporaries(shape))
    if init_step:
        data['gz'][:-1, :-1, :] = HUGE_R
        data['diss_estd'][grid.slice_dict(grid.compute_dict())] = 0.0
        if not hydrostatic:
            data['pk3'][:-1, :-1, :] = HUGE_R
    data['mfxd'][grid.slice_dict(grid.x3d_compute_dict())] = 0.0
    data['mfyd'][grid.slice_dict(grid.y3d_compute_dict())] = 0.0
    data['cxd'][grid.slice_dict(grid.x3d_compute_domain_y_dict())] = 0.0
    data['cyd'][grid.slice_dict(grid.y3d_compute_domain_x_dict())] = 0.0
    if not hydrostatic:
        # k1k = akap / (1.0 - akap)
        # TODO -- is really just a column... when different shapes are supported perhaps change this
        data['dp_ref'] = utils.make_storage_from_shape(data['ak'].shape, grid.default_origin())
        dp_ref_compute(data['ak'], data['bk'], data['dp_ref'])
        data['zs'] = data['phis'] * rgrav
    n_con = get_n_con()
    
    for it in range(n_split):
        remap_step = False
        if spec.namelist['breed_vortex_inline'] or (it == n_split - 1):
            remap_step = True
        
        if not hydrostatic:
            reqs['w_quantity'] = comm.start_halo_update(data['w_quantity'], n_points=utils.halo)
            if it == 0:
                set_gz(data['zs'], data['delz'], data['gz'], origin=grid.compute_origin(), domain=(grid.nic, grid.njc, grid.npz+1))
                reqs['gz_quantity'] = comm.start_halo_update(data['gz_quantity'], n_points=utils.halo)
        if it == 0:
            reqs['delp_quantity'].wait()
            reqs['pt_quantity'].wait()
            beta_d = 0
        else:
            beta_d = spec.namelist['beta']
        last_step = False
        if it == n_split - 1 and end_step:
            last_step = True
       
        if it == n_split - 1 and end_step:
            if spec.namelist['use_old_omega']: # apparently true
                set_pem(data['delp'], data['pem'], data['ptop'],
                        origin=(grid.is_ - 1, grid.js - 1, 0), domain=(grid.nic + 2, grid.njc + 2, grid.npz))
       
        reqs_vector.wait()
        
        if not hydrostatic:
            reqs['w_quantity'].wait()
       
        data['delpc'], data['ptc'], data['omga'] = c_sw.compute(data['delp'], data['pt'], data['u'], data['v'], data['w'],
                                                                data['uc'], data['vc'], data['ua'], data['va'], data['ut'],
                                                                data['vt'], data['divgd'], dt2)
        
        if spec.namelist['nord'] > 0:
            reqs['divgd_quantity'] = comm.start_halo_update(data['divgd_quantity'], n_points=utils.halo)
        if not hydrostatic:
            if it == 0:
                reqs['gz_quantity'].wait()
                cp.copy_stencil(data['gz'], data['zh'], origin=grid.default_origin(), domain=grid.domain_shape_buffer_k())
            else:
                cp.copy_stencil(data['gz'], data['zh'], origin=grid.default_origin(), domain=grid.domain_shape_buffer_k())
        if not hydrostatic:
            data['gz'], data['ws3'] = updatedzc.compute(data['dp_ref'], data['zs'], data['ut'], data['vt'],
                                                        data['gz'], data['ws3'], dt2)
            #TODO this is really a 2d field.
            data['ws3'] = utils.make_storage_data_from_2d(data['ws3'][:, :, -1], shape, origin=(0,0,0))
            riem_solver_c.compute(ms, dt2, akap, data['cappa'], data['ptop'], data['phis'], data['omga'],
                                  data['ptc'], data['q_con'], data['delpc'], data['gz'], data['pkc'], data['ws3'])

        pgradc.compute(data['uc'], data['vc'], data['delpc'], data['pkc'], data['gz'], dt2)
        reqc_vector = comm.start_vector_halo_update(data['uc_quantity'], data['vc_quantity'], n_points=utils.halo)
        if spec.namelist['nord'] > 0:
            reqs['divgd_quantity'].wait()
        reqc_vector.wait()
       
        data['nord_v'], data['damp_vt'] = d_sw.compute(data['vt'], data['delp'], data['ptc'], data['pt'], data['u'], data['v'],
                                                       data['w'], data['uc'], data['vc'], data['ua'], data['va'], data['divgd'],
                                                       data['mfxd'], data['mfyd'], data['cxd'], data['cyd'], data['crx'], data['cry'],
                                                       data['xfx'], data['yfx'], data['q_con'], data['zh'], data['heat_source'],
                                                       data['diss_estd'], dt)
           
        for halovar in ['delp_quantity', 'pt_quantity', 'q_con_quantity']:
            comm.halo_update(data[halovar], n_points=utils.halo)
   
        # Not used unless we implement other betas and alternatives to nh_p_grad
        # if spec.namelist['d_ext'] > 0:
        #    raise 'Unimplemented namelist option d_ext > 0'
        #else:
        #    divg2 = utils.make_storage_from_shape(delz.shape, grid.compute_origin())
       
        if not hydrostatic:
            updatedzd.compute(data['nord_v'], data['damp_vt'], data['dp_ref'], data['zs'], data['zh'], data['crx'], data['cry'], data['xfx'], data['yfx'], data['wsd'], dt)
                
            #TODO this is really a 2d field.
            data['wsd'] = utils.make_storage_data_from_2d(data['wsd'][:, :, -1], shape, origin=grid.compute_origin())
            riem_solver3.compute(remap_step, dt, akap, data['cappa'], data['ptop'], data['zs'], data['w'], data['delz'], data['q_con'], data['delp'],
                                 data['pt'], data['zh'], data['pe'], data['pkc'], data['pk3'], data['pk'], data['peln'], data['wsd'])
            
            reqs['zh_quantity'] = comm.start_halo_update(data['zh_quantity'], n_points=utils.halo)
            if grid.npx == grid.npy:
                reqs['pkc_quantity'] = comm.start_halo_update(data['pkc_quantity'], n_points=2)
            else:
                reqs['pkc_quantity'] = comm.start_halo_update(data['pkc_quantity'], n_points=utils.halo)
            if remap_step:
                pe_halo.compute(data['pe'], data['delp'], data['ptop'])
            if spec.namelist['use_logp']:
                raise Exception('unimplemented namelist option use_logp=True')
            else:
                pk3_halo.compute(data['pk3'], data['delp'], data['ptop'], akap)
        if not hydrostatic:
            reqs['zh_quantity'].wait()
            if grid.npx != grid.npy:
                reqs['pkc_quantity'].wait()
        if not hydrostatic:
            basic.multiply_constant(data['zh'], constants.GRAV, data['gz'],
                                    origin=(grid.is_ - 2, grid.js - 2, 0),
                                    domain=(grid.nic + 4, grid.njc + 4, grid.npz + 1))
            if grid.npx == grid.npy:
                reqs['pkc_quantity'].wait()
            if spec.namelist['beta'] != 0:
                raise Exception('Unimplemented namelist option -- we only support beta=0')
        if not hydrostatic:
            nh_p_grad.compute(data['u'], data['v'], data['pkc'], data['gz'], data['pk3'], data['delp'], dt, data['ptop'], akap)
    
        if spec.namelist['RF_fast']:
            raise Exception('Unimplemented namelist option -- we do not support RF_fast')
       
        if it != n_split - 1:
            reqs_vector = comm.start_vector_halo_update(data['u_quantity'], data['v_quantity'], n_points=utils.halo)
    if n_con != 0 and spec.namelist['d_con'] > 1.0e-5:
        nf_ke = min(3, spec.namelist['nord'] + 1)
       
        comm.halo_update(data['heat_source_quantity'], n_points=utils.halo)
        cd = constants.CNST_0P20 * grid.da_min
        del2cubed.compute(data['heat_source'], nf_ke, cd, grid.npz)
        if not hydrostatic:
            temperature_adjust.compute(data['pt'], data['pkz'], data['heat_source'], data['delz'], data['delp'], data['cappa'], n_con, dt)
   
