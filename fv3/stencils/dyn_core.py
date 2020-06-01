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



def check_vars(varlist,serstring, rank_grid, data, denoter='',print_errors=False, krange=None):
    
    if rank_grid.rank !=0:
        return
    print('checking', denoter, rank_grid.rank)
    for var in varlist:
        servar = var + serstring
        count = 0
        if krange is None:
            krange = data[var].shape[2]
        #print('checking', var)
        for i in range(data[var].shape[0]):
            for j in range(data[var].shape[1]):
                for k in range(krange):
                    comp = data[var][i, j, k]
                    res = data[servar][i, j, k]
                    if abs((comp - res) / res) > 1e-18:
                        count +=1
                        if print_errors:
                            print('mismatch',var, 'rank',rank_grid.rank, i, j, k, comp, res, comp-res)
        if count > 0:
            print('BROKEN', count, denoter, 'bad for rank', rank_grid.rank, var) 
    
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
    rdt = 1.0 / dt
    hydrostatic = spec.namelist['hydrostatic']
    rgrav = 1.0 / constants.GRAV
    n_split = data['n_split']
    # TODO -- put defaults into code
    # m_split = 1. + abs(dt_atmos)/real(k_split*n_split*abs(p_split))
    # n_split = nint( real(n0split)/real(k_split*abs(p_split)) * stretch_fac + 0.5 )
    ms = max(1, spec.namelist['m_split'] / 2.0)
    shape = data['delz'].shape
  
    # NOTE in Fortran model the halo update starts happens in fv_dynamics, not here
    #print('TRACE u 1', u[10, 32, 49])
    #for data, communicator, rank_grid in zipped:
    #    data['q_con_quantity'] = HALO_utils.make_quantity(data['q_con'], grid=rank_grid)
    #    HALO_utils.start_halo_update_comm(communicator, data['q_con_quantity'])
    reqs = {}
    for halovar in ['q_con_quantity', 'cappa_quantity','delp_quantity', 'pt_quantity']:
        reqs[halovar] = comm.start_halo_update(data[halovar], n_points=utils.halo)
    reqs_vector = comm.start_vector_halo_update(data['u_quantity'], data['v_quantity'], n_points=utils.halo)
    #    data['cappa_quantity'] = HALO_utils.make_quantity(data['cappa'], grid=rank_grid)
    #    HALO_utils.start_halo_update_comm(communicator, data['cappa_quantity'])
    #    data['delp_quantity'] = HALO_utils.make_quantity(data['delp'], grid=rank_grid)
    #    data['pt_quantity'] = HALO_utils.make_quantity(data['pt'], grid=rank_grid)
    #    HALO_utils.start_halo_update_comm(communicator, data['delp_quantity'])
    #    HALO_utils.start_halo_update_comm(communicator, data['pt_quantity'])
    #    data['u_quantity'], data['v_quantity'] = get_vector_quantities(data['u'], data['v'], rank_grid)
    #    HALO_utils.start_vector_halo_update_comm(communicator, data['u_quantity'], data['v_quantity'])
    #    data['w_quantity'] = HALO_utils.make_quantity(data['w'], units="m/s", grid=rank_grid)
    #for data, communicator, rank_grid in zipped:
    #    HALO_utils.finish_halo_update_comm(communicator, data['q_con_quantity'])
    #    HALO_utils.finish_halo_update_comm(communicator, data['cappa_quantity'])
    reqs['q_con_quantity'].wait()
    reqs['cappa_quantity'].wait()
    #    #data['cappa'] =  data['cappa_quantity'].data
    tmps = {}
    utils.storage_dict(tmps, ['ut', 'vt', 'gz', 'zh', 'pem', 'ws3', 'pkc', 'pk3', 'heat_source', 'divgd'], shape, grid.default_origin())
    utils.storage_dict(tmps, ['crx', 'xfx'], shape, grid.compute_x_origin())
    utils.storage_dict(tmps, ['cry', 'yfx'], shape, grid.compute_y_origin())
    #    #utils.storage_dict(tmps, ['peln'], shape, rank_grid.compute_origin())
    #data['heat_source'] = grid.make_quantity.empty([fv3util.X_DIM, fv3util.Y_DIM, fv3util.Z_DIM], "tmp", dtype=data['ut'].dtype)
    #for q in ['gz', 'pkc', 'zh']:
    #    data[q] = grid.quantity_factory.empty([fv3util.X_DIM, fv3util.Y_DIM, fv3util.Z_INTERFACE_DIM], "tmp", dtype=data['ut'].dtype)
    # data['divgd'] = grid.quantity_factory.empty([fv3util.X_INTERFACE_DIM, fv3util.Y_INTERFACE_DIM, fv3util.Z_DIM], "tmp", dtype=data['ut'].dtype)
    data['heat_source_quantity'] = grid.make_quantity(tmps['heat_source'], dims=[fv3util.X_DIM, fv3util.Y_DIM, fv3util.Z_DIM] )
    for q in ['gz', 'pkc','zh']:
        data[q + '_quantity'] = grid.make_quantity(tmps[q], dims=[fv3util.X_DIM, fv3util.Y_DIM, fv3util.Z_INTERFACE_DIM], extent=(grid.nic, grid.njc, grid.npz + 1))
    data['divgd_quantity'] = grid.make_quantity(tmps['divgd'], dims=[fv3util.X_INTERFACE_DIM, fv3util.Y_INTERFACE_DIM, fv3util.Z_DIM], extent=(grid.nic + 1, grid.njc + 1, grid.npz))
    #    #tmp = SimpleNamespace(**tmps)
    data.update(tmps)
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
        #for data, communicator, rank_grid in zipped:
        if not hydrostatic:
            #HALO_utils.start_halo_update_comm(communicator, data['w_quantity'])
            reqs['w_quantity'] = comm.start_halo_update(data['w_quantity'], n_points=utils.halo)
            if it == 0:
                #check_vars(['gz'],'d_n', rank_grid, data, denoter='-----INIT GZ', print_errors=False)
                set_gz(data['zs'], data['delz'], data['gz'], origin=grid.compute_origin(), domain=(grid.nic, grid.njc, grid.npz+1))
                #check_vars(['gz'],'d_h', grid, data, denoter='-----SET GZ', print_errors=False)
                
                #HALO_utils.start_halo_update_comm(communicator, data['gz_quantity'])
                reqs['gz_quantity'] = comm.start_halo_update(data['gz_quantity'], n_points=utils.halo)
        #for data, communicator, grid in zipped:
        if it == 0:
            #HALO_utils.finish_halo_update_comm(communicator, data['delp_quantity'])
            #HALO_utils.finish_halo_update_comm(communicator, data['pt_quantity'])
            reqs['delp_quantity'].wait()
            reqs['pt_quantity'].wait()
            beta_d = 0
        else:
            beta_d = spec.namelist['beta']
        last_step = False
        if it == n_split - 1 and end_step:
            last_step = True
        #for data, communicator, grid in zipped:
        if it == n_split - 1 and end_step:
            if spec.namelist['use_old_omega']: # apparently true
                set_pem(data['delp'], data['pem'], data['ptop'],
                        origin=(grid.is_ - 1, grid.js - 1, 0), domain=(grid.nic + 2, grid.njc + 2, grid.npz))
        #HALO_utils.finish_vector_halo_update_comm(communicator, data['u_quantity'], data['v_quantity'])
        reqs_vector.wait()
        if not hydrostatic:
            #HALO_utils.finish_halo_update_comm(communicator, data['w_quantity'])
            reqs['w_quantity'].wait()
        #for data, communicator, grid in zipped:
        #    spec.set_grid(grid)
        if grid.rank == 0:
            print('VTRACE 1', data['v'][8, 42, 1])
        data['delpc'], data['ptc'], data['omga'] = c_sw.compute(data['delp'], data['pt'], data['u'], data['v'], data['w'],
                                                                data['uc'], data['vc'], data['ua'], data['va'], data['ut'],
                                                                data['vt'], data['divgd'], dt2)
        #check_vars(['uc', 'vc', 'divgd', 'delp', 'delpc', 'ptc'],'d_a', grid, data, denoter='c_sw',print_errors=False)
        if grid.rank == 0:
            print('VTRACE 1', data['v'][8, 42, 1])
        if spec.namelist['nord'] > 0:
            #HALO_utils.start_halo_update_comm(communicator, data['divgd_quantity'])
            reqs['divgd_quantity'] = comm.start_halo_update(data['divgd_quantity'], n_points=utils.halo)
        if not hydrostatic:
            if it == 0:
                #HALO_utils.finish_halo_update_comm(communicator, data['gz_quantity'])
                reqs['gz_quantity'].wait()
                #check_vars(['gz'],'d_4', grid, data, denoter='++++++++++++++++HALO UPDATE GZ', print_errors=False)
                cp.copy_stencil(data['gz'], data['zh'], origin=grid.default_origin(), domain=grid.domain_shape_buffer_k())
            else:
                cp.copy_stencil(data['gz'], data['zh'], origin=grid.default_origin(), domain=grid.domain_shape_buffer_k())
        #for data, communicator, grid in zipped:
        #    spec.set_grid(grid)
        if not hydrostatic:
            #check_vars(['ws3','gz', 'dp_ref', 'ut', 'vt', 'zs'],'d_u', grid, data, denoter='IN UpdateDzC', print_errors=False)
            data['gz'], data['ws3'] = updatedzc.compute(data['dp_ref'], data['zs'], data['ut'], data['vt'],
                                                        data['gz'], data['ws3'], dt2)
            #TODO this is really a 2d field.
            data['ws3'] = utils.make_storage_data_from_2d(data['ws3'][:, :, -1], shape, origin=(0,0,0))
            #check_vars(['ws3','gz'],'d_r', grid, data, denoter='OUT UpdateDzC', print_errors=False)
            riem_solver_c.compute(ms, dt2, akap, data['cappa'], data['ptop'], data['phis'], data['omga'],
                                  data['ptc'], data['q_con'], data['delpc'], data['gz'], data['pkc'], data['ws3'])
        #check_vars(['uc', 'vc', 'pkc','gz'],'d_3', grid, data, denoter='PRE pregradc', print_errors=False)

        pgradc.compute(data['uc'], data['vc'], data['delpc'], data['pkc'], data['gz'], dt2)
        #check_vars(['uc', 'vc'],'d_2', grid, data, denoter='POST pregradc', print_errors=False)
          
        #data['uc_quantity'], data['vc_quantity'] = get_vector_quantities_cgrid(data['uc'], data['vc'], grid)
        #HALO_utils.start_vector_halo_update_comm(communicator, data['uc_quantity'], data['vc_quantity'])
        reqc_vector = comm.start_vector_halo_update(data['uc_quantity'], data['vc_quantity'], n_points=utils.halo)
        if spec.namelist['nord'] > 0:
            #HALO_utils.finish_halo_update_comm(communicator, data['divgd_quantity'])
            reqs['divgd_quantity'].wait()
        #for data, communicator, grid in zipped:
        #    HALO_utils.finish_vector_halo_update_comm(communicator, data['uc_quantity'], data['vc_quantity'])
        reqc_vector.wait()
        #for data, communicator, grid in zipped:
        #    check_vars(['uc', 'vc', 'divgd', 'omga', 'gz', 'delpc', 'pkc', 'cappa', 'phis', 'ptc', 'q_con', 'ws3', 'dp_ref', 'zs', 'ut', 'vt', 'zh'],'d_b', grid, data, denoter='After uc/vc halo update ', print_errors=False)
                
        
        
        #for data, communicator, grid in zipped:
        #    spec.set_grid(grid)
        #check_vars(['vt', 'delp', 'ptc', 'pt', 'u', 'v', 'w', 'uc', 'vc', 'ua', 'va', 'divgd', 'mfxd', 'mfyd', 'cxd', 'cyd', 'crx', 'cry', 'xfx', 'yfx', 'q_con', 'zh'], 'd_lol', grid, data, denoter='INPUT D_SW ', print_errors=False)
        #return
        if grid.rank == 0:
            print('VTRACE d', data['v'][8, 42, 1])
        if grid.rank == 4:
            print('MFXTRACE d', data['mfxd'][40, 19, 18])
        data['nord_v'], data['damp_vt'] = d_sw.compute(data['vt'], data['delp'], data['ptc'], data['pt'], data['u'], data['v'],
                                                       data['w'], data['uc'], data['vc'], data['ua'], data['va'], data['divgd'],
                                                       data['mfxd'], data['mfyd'], data['cxd'], data['cyd'], data['crx'], data['cry'],
                                                       data['xfx'], data['yfx'], data['q_con'], data['zh'], data['heat_source'],
                                                       data['diss_estd'], dt)
           
        if grid.rank == 0:
            print('VTRACE d', data['v'][8, 42, 1])
        if grid.rank == 4:
            print('MFXTRACE d', data['mfxd'][40, 19, 18])
        #check_vars(['vt','delp','ptc','pt', 'u', 'v', 'w','uc', 'vc','ua','va','divgd', 'mfxd', 'mfyd', 'cxd','cyd','crx','cry','q_con', 'heat_source', 'diss_estd'],'d_d', grid, data, denoter='D_SW ', print_errors=False)
        #check_vars(['xfx', 'yfx'],'d_d', grid, data, denoter='D_SW xfx', print_errors=False)
        for halovar in ['delp_quantity', 'pt_quantity', 'q_con_quantity']:
            comm.halo_update(data[halovar], n_points=utils.halo)
        #HALO_utils.start_halo_update_comm(communicator, data['delp_quantity'])
        #HALO_utils.start_halo_update_comm(communicator, data['pt_quantity'])
        #HALO_utils.start_halo_update_comm(communicator, data['q_con_quantity'])
        # Not used unless we implement other betas and alternatives to nh_p_grad
        # if spec.namelist['d_ext'] > 0:
        #    raise 'Unimplemented namelist option d_ext > 0'
        #else:
        #    divg2 = utils.make_storage_from_shape(delz.shape, grid.compute_origin())
        #for data, communicator, grid in zipped:
        #    HALO_utils.finish_halo_update_comm(communicator, data['delp_quantity'])
        #    HALO_utils.finish_halo_update_comm(communicator, data['pt_quantity'])
        #    HALO_utils.finish_halo_update_comm(communicator, data['q_con_quantity'])
        #    check_vars(['delp', 'pt', 'q_con'],'d_h2', grid, data, denoter='HALO update delp, pt, q_con', print_errors=False)
        
        #for data, communicator, grid in zipped:
        #    spec.set_grid(grid)
        if not hydrostatic:
            #data['ws'] = data['wsd'] # TODO remove
            updatedzd.compute(data['nord_v'], data['damp_vt'], data['dp_ref'], data['zs'], data['zh'], data['crx'], data['cry'], data['xfx'], data['yfx'], data['wsd'], dt)
                
            #TODO this is really a 2d field.
            data['wsd'] = utils.make_storage_data_from_2d(data['wsd'][:, :, -1], shape, origin=grid.compute_origin())
            #data['ws'] =  data['wsd']
            
            #check_vars(['zh', 'crx', 'cry', 'xfx', 'yfx', 'delz'],'d_z', grid, data, denoter='UPDAT_DZ_D', print_errors=False)
            #check_vars([ 'wsd'],'d_z', grid, data, denoter='UPDAT_DZ_D WS', print_errors=False)
            #print('peln pre riem', data['peln'][49, 49, 10])
            #check_vars(['zs', 'w', 'delz', 'q_con', 'delp', 'pt', 'zh', 'pe', 'pkc', 'pk3', 'pk', 'peln', 'wsd'],'d_9', grid, data, denoter='INPUT RIEM3', print_errors=True)
            #if grid.rank == 0:
            #    print('TRACE IN', data['peln'][50, 50, 46],  data['pelnd_9'][50, 50, 46],data['pe'][50, 50, 46],data['ped_9'][50, 50, 46], data['wsd'][32, 15, 16], data['wsdd_9'][32,15,16])
            riem_solver3.compute(remap_step, dt, akap, data['cappa'], data['ptop'], data['zs'], data['w'], data['delz'], data['q_con'], data['delp'],
                                 data['pt'], data['zh'], data['pe'], data['pkc'], data['pk3'], data['pk'], data['peln'], data['wsd'])
            #print('peln post riem', data['peln'][49, 49, 10],  data['pelnd_s'][49, 49, 10])
            # w rank 0 32 15 16 -7.070962721855006e-08 -7.070963637095112e-08 9.152401059253634e-15
            #                           -0.0019708975809760512-->               -7.070963637095112e-08
            #if grid.rank == 0:
            #    print("RIEM TRACE W", data['w'][32, 15, 16], data['wd_s'][32, 15, 16])
            # w 12 42 59 0.00016506574972249797 0.00016506574969373883 2.875914025153581e-14
            # pkc rank 0 48 20 57 -0.0033814144843803354 -0.0033814144839934045 -3.869309386783648e-13
            #check_vars(['w','delz','zh','pe','pkc','pk3', 'pk', 'peln', 'wsd'],'d_s', grid, data, denoter='RIEM', print_errors=False)
            #HALO_utils.start_halo_update_comm(communicator, data['zh_quantity'])
            reqs['zh_quantity'] = comm.start_halo_update(data['zh_quantity'], n_points=utils.halo)
            if grid.npx == grid.npy:
                #HALO_utils.start_halo_update_comm(communicator, data['pkc_quantity'], n_points=2)
                reqs['pkc_quantity'] = comm.start_halo_update(data['pkc_quantity'], n_points=2)
            else:
                #HALO_utils.start_halo_update_comm(communicator, data['pkc_quantity'])
                reqs['pkc_quantity'] = comm.start_halo_update(data['pkc_quantity'], n_points=utils.halo)
            if remap_step:
                pe_halo.compute(data['pe'], data['delp'], data['ptop'])
            if spec.namelist['use_logp']:
                raise Exception('unimplemented namelist option use_logp=True')
            else:
                pk3_halo.compute(data['pk3'], data['delp'], data['ptop'], akap)
            #check_vars(['pk3', 'pe'],'d_h3', grid, data, denoter='PK3 halo', print_errors=False)
        #for data, communicator, grid in zipped:
        if not hydrostatic:
            #HALO_utils.finish_halo_update_comm(communicator, data['zh_quantity'])
            reqs['zh_quantity'].wait()
            #check_vars(['zh'],'d_h4', grid, data, denoter='PK3 halo', print_errors=False)
            if grid.npx != grid.npy:
                #HALO_utils.finish_halo_update_comm(communicator, data['pkc_quantity'])
                reqs['pkc_quantity'].wait()
        #for data, communicator, grid in zipped:
        if not hydrostatic:
            basic.multiply_constant(data['zh'], constants.GRAV, data['gz'],
                                    origin=(grid.is_ - 2, grid.js - 2, 0),
                                    domain=(grid.nic + 4, grid.njc + 4, grid.npz + 1))
            if grid.npx == grid.npy:
                reqs['pkc_quantity'].wait()
                #HALO_utils.finish_halo_update_comm(communicator, data['pkc_quantity'], n_points=2)
            if spec.namelist['beta'] != 0:
                raise Exception('Unimplemented namelist option -- we only support beta=0')
        #for data, communicator, grid in zipped:
        #    spec.set_grid(grid)
        if not hydrostatic:
            if grid.rank == 0:
                print('VTRACE nh', data['v'][8, 42, 1])
            nh_p_grad.compute(data['u'], data['v'], data['pkc'], data['gz'], data['pk3'], data['delp'], dt, data['ptop'], akap)
            if grid.rank == 0:
                print('VTRACE nh', data['v'][8, 42, 1])
            #check_vars(['u', 'v', 'ppd_nh=pkc gzd_nh=gz pk3d_nh=pk3 delpd_nh=delp,'d_nh', grid, data, denoter='PK3 halo', print_errors=False)
        if spec.namelist['RF_fast']:
            raise Exception('Unimplemented namelist option -- we do not support RF_fast')
        #if it == n_split - 1:
        #    # mpp get boundary, u[is:ie, je+1,:] = nbuffer[i - is + 1, :], v[ie+1,js:je,:] = ebuffer[j - js + 1, :]
        if it != n_split - 1:
            #for data, communicator, grid in zipped:
            #HALO_utils.start_vector_halo_update_comm(communicator, data['u_quantity'], data['v_quantity'])
            reqs_vector = comm.start_vector_halo_update(data['u_quantity'], data['v_quantity'], n_points=utils.halo)
    if n_con != 0 and spec.namelist['d_con'] > 1.0e-5:
        nf_ke = min(3, spec.namelist['nord'] + 1)
       
        #for data, communicator, grid in zipped:
        #    HALO_utils.start_halo_update_comm(communicator, data['heat_source_quantity'])
        #for data, communicator, grid in zipped:
        #    HALO_utils.finish_halo_update_comm(communicator, data['heat_source_quantity'])
        comm.halo_update(data['heat_source_quantity'], n_points=utils.halo)
        #for data, communicator, grid in zipped:
        #    spec.set_grid(grid)
        cd = constants.CNST_0P20 * grid.da_min
        del2cubed.compute(data['heat_source'], nf_ke, cd, grid.npz)
        if not hydrostatic:
            temperature_adjust.compute(data['pt'], data['pkz'], data['heat_source'], data['delz'], data['delp'], data['cappa'], n_con, dt)
   
