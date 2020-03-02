#!/usr/bin/env python3
import fv3.utils.gt4py_utils as utils
import gt4py as gt
import gt4py.gtscript as gtscript
import fv3._config as spec
import fv3.utils.global_constants as constants
import numpy as np
import fv3.stencils.copy_stencil as cp
sd = utils.sd


# TODO: merge with vbke?
@gtscript.stencil(backend=utils.exec_backend, rebuild=utils.rebuild)
def main_pe(gm2: sd, pe:sd):
    with computation(PARALLEL), interval(...):
        pe = gm2 * 2.0
        
# TODO: implement MOIST_CAPPA=false
def solve(dt, gama, gm2, cp2, akap, pe2, dm, pm2, pem, w2, dz2, ptr, wsr, p_fac):
    is_origin = (spec.grid.is_, 0, 0)
    islice = slice(spec.grid.is_, spec.grid.ie + 1)
    islicet= slice(0,spec.grid.nic)
    tmpshape = (spec.grid.nic, spec.grid.npz, 1)
    tmpshape_p1 = (spec.grid.nic, spec.grid.npz+1, 1)
 
    t1g = 2.0 * dt * dt
    npz = spec.grid.npz
    rdt = 1.0 / dt
    cappa1 = akap - 1.0
    print(pe2.shape, gm2.shape)
    slice_m = (islice, slice(0, npz - 1), 0)
    slice_m2 = (islice, slice(1, npz+1), 0)
    slice_n = (islice, slice(0, npz), 0)
    slice_mt = (islicet, slice(0, npz - 1), 0)
    slice_mt2 = (islicet, slice(1, npz+1), 0)
    slice_nt = (islicet, slice(0, npz), 0)
    
    pe2[slice_m] = np.exp(gm2[slice_m] * np.log(-dm[slice_m] / dz2[slice_m] * constants.RDGAS * ptr[slice_m])) - pm2[slice_m]
  
    g_rat =  utils.make_storage_from_shape(tmpshape, (0,0,0), backend=utils.data_backend)
    bb =  utils.make_storage_from_shape(tmpshape, (0,0,0), backend=utils.data_backend)
    aa =  utils.make_storage_from_shape(tmpshape, (0,0,0), backend=utils.data_backend)
    dd =  utils.make_storage_from_shape(tmpshape, (0,0,0), backend=utils.data_backend)
    gam =  utils.make_storage_from_shape(tmpshape, (0,0,0), backend=utils.data_backend)
    w1 =  utils.make_storage_from_shape(tmpshape, (0,0,0), backend=utils.data_backend)
    pp =  utils.make_storage_from_shape(tmpshape_p1, (0,0,0), backend=utils.data_backend)
    w1a = cp.copy(w2, is_origin)
    w1[:,:, 0] = w1a[islice, 0:spec.grid.npz, 0]
    g_rat[:,:,0] = dm[slice_n] / dm[slice_m2]
 
    bb[slice_nt] = 2.0 * (1.0 + g_rat[slice_nt])
    dd[slice_nt] = 3.0 * (pe2[slice_n] + g_rat[slice_nt] * pe2[slice_m2])
    bet = np.squeeze(np.copy(bb)[:,0])
    p1 = np.squeeze(np.copy(bb)[:,0])
    ppa = cp.copy(pem, is_origin)
    pp[:, :, 0] = ppa[islice, :,0]
    pp[:, 0, 0] = 0.0
    pp[:, 1, 0] = dd[:, 0, 0] / bet
    bb[:, npz - 1, 0] = 2.0
    dd[:, npz - 1, 0] = 3.0* pe2[islice, npz - 1, 0]
   
    for k in range(1, npz):
        for i in range(spec.grid.nic):
            gam[i, k] = g_rat[i,k - 1] / bet[i]
            bet[i] = bb[i, k] - gam[i, k]
            pp[i, k+1] = (dd[i, k] - pp[i, k]) / bet[i]
   
    for k in range(npz - 1, 0, -1):
        pp[:, k + 1, 0] = (dd[:, k, 0] - pp[:, k, 0]) / bet
    # w solver
    zslice = slice(0, npz - 1)
    zslice2 = slice(1, npz)
    aa[:, zslice2, 0] = t1g * 0.5 * (gm2[islice, zslice, 0] + gm2[islice, zslice2, 0])\
        / (dz2[islice, zslice, 0] + dz2[islice, zslice2, 0]) * (pem[islice, zslice2, 0] + pp[:, zslice2, 0])
    bet[:] = dm[islice, 0, 0] - aa[:, 1, 0]
    w2[islice, 0, 0] = (dm[islice, 0, 0] * w1[:, 0, 0] + dt * pp[:, 1, 0]) / bet
    for k in range(1, npz - 1):
        for i in range(spec.grid.nic):
            gi = i + spec.grid.is_
            gam[i, k, 0] = aa[i, k, 0] / bet[i]
            bet[i] = dm[gi, k, 0] - (aa[i, k, 0] + aa[i, k + 1, 0] + aa[i, k, 0] * gam[i, k, 0])
            w2[i, k, 0] = (dm[gi, k, 0] * w1[i, k, 0] +
                           dt * (pp[i, k+1, 0] - pp[i, k, 0]) - aa[i, k, 0] * w2[gi, k-1, 0]) / bet[i]
    for i in range(spec.grid.nic):
        gi = i + spec.grid.is_
        p1[i] = t1g * gm2[gi,npz - 1, 0]/dz2[gi,npz - 1, 0]*(pem[gi,npz - 1+1, 0] + pp[i,npz - 1+1, 0])
        gam[i,npz - 1, 0] = aa[i,npz - 1, 0] / bet[i]
        bet[i] =  dm[gi, npz - 1, 0] - (aa[i, npz - 1, 0] + p1[i] + aa[i, npz - 1, 0] * gam[i, npz - 1, 0])
        w2[gi, npz - 1, 0] = (dm[gi, npz - 1, 0] * w1[i,npz - 1, 0] + dt*(pp[i, npz - 1+1, 0] - pp[i, npz - 1, 0]) - p1[i] * wsr[gi, 0, 0]  -aa[i, npz - 1, 0] * w2[gi,npz - 1 -1, 0])/bet[i]

    for k in range(npz - 2, 0, -1):
        w2[islice, k, 0] = w2[islice, k, 0 ] - gam[i, k+1, 0] * w2[islice, k+1, 0]

    pe2[islice, 0, 0] = 0.0
    for k in range(npz):
        pe2[islice, k+1, 0] = pe2[islice, k, 0] + dm[islice, k, 0] * (w2[islice, k, 0] - w1[:, k, 0]) / dt

    p1[:] = (pe2[islice, npz - 1, 0] + 2.0 * pe2[islice, npz, 0]) * 1.0 / 3.0
  
    for i in range(spec.grid.is_, spec.grid.ie + 1):
        dz2[i, npz - 1, 0] = -dm[i, npz - 1, 0] * constants.RDGAS * ptr[i, npz - 1, 0] * np.exp((cp2[i, npz - 1, 0] - 1.0) * np.log(max(p_fac * pm2[i, npz - 1, 0], p1[i - spec.grid.is_] + pm2[i, npz - 1, 0])))
                
    for k in range(npz - 2, 0, -1):
        for i in range(spec.grid.is_, spec.grid.ie + 1):
            li = i - spec.grid.is_
            p1[li] = (pe2[i, k, 0] + bb[li, k, 0] * pe2[i, k+1, 0] + g_rat[li,k, 0] * pe2[i,k+2, 0]) * 1.0 / 3.0 - g_rat[li,k, 0] * p1[li]
            dz2[i, k, 0] = -dm[i, k, 0] * constants.RDGAS * ptr[i, k, 0] * np.exp((cp2[i, k, 0] - 1.0) * np.log(max(p_fac * pm2[i, k, 0], p1[li] + pm2[i, k, 0])))
