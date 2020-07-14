#!/usr/bin/env python3
import fv3.utils.gt4py_utils as utils
import gt4py as gt
import gt4py.gtscript as gtscript
import fv3._config as spec
import fv3.utils.global_constants as constants
import numpy as np
import fv3.stencils.copy_stencil as cp
from gt4py.gtscript import computation, interval, PARALLEL

sd = utils.sd
@utils.stencil()
def initial(w2: sd, dm: sd, gm2: sd, dz2: sd, ptr: sd, pm2:sd, pe2: sd, g_rat: sd, bb: sd, dd: sd, w1: sd):
    with computation(PARALLEL), interval(...):
        w1 = w2
        pe2 = (-dm / dz2 * constants.RDGAS * ptr)**gm2 - pm2
    with computation(PARALLEL):
        with interval(0, -1):
            g_rat = dm / dm[0, 0, 1]
            bb = 2.0 * (1.0 + g_rat)
            dd = 3.0 * (pe2 + g_rat * pe2[0, 0, 1])
        with interval(-1, None):
            bb = 2.0
            dd = 3.0 * pe2


@utils.stencil()
def w_solver(aa: sd, bet: sd, g_rat: sd, gam: sd, pp: sd, dd: sd, gm2: sd, dz2: sd, pem: sd, dm: sd, pe2: sd, bb: sd, t1g: float):
    with computation(PARALLEL):
        with interval(0, 1):
            pp = 0.0
        with interval(1, 2):
            pp = dd[0, 0, -1] / bet
    with computation(FORWARD), interval(1, -1):
        gam = g_rat[0, 0, -1] / bet[0, 0, -1]
        bet = bb - gam
    with computation(FORWARD), interval(2, None):
        pp = (dd[0, 0, -1] - pp[0, 0, -1]) / bet[0, 0, -1]
    with computation(BACKWARD), interval(1, -1):
        pp = pp - gam * pp[0, 0, 1]
        # w solver
        aa = t1g * 0.5 * (gm2[0, 0, -1] + gm2) / (dz2[0, 0, -1] + dz2) * (pem + pp)

@utils.stencil()
def w2_pe2_dz2_compute(dm: sd, w1: sd, pp: sd, aa: sd, gm2: sd, dz2: sd, pem: sd, wsr_top: sd, bb: sd, g_rat: sd, bet: sd, gam: sd, p1: sd, pe2: sd, w2: sd,  pm: sd, ptr: sd, cp: sd, dt: float, t1g: float, rdt: float, p_fac: float):
    with computation(FORWARD):
        with interval(0, 1):
            w2 = (dm * w1 + dt * pp[0, 0, 1]) / bet
        with interval(1, -2):
            gam = aa / bet[0, 0, -1]
            bet = dm - (aa + aa[0, 0, 1] + aa * gam)
            w2 = (dm * w1 + dt * (pp[0, 0, 1] - pp) - aa * w2[0, 0, -1]) / bet
        with interval(-2, -1):
            p1 = t1g * gm2 / dz2 * (pem[0, 0, 1] + pp[0, 0, 1])
            gam = aa / bet[0, 0, -1]
            bet = dm - (aa + p1 + aa * gam)
            w2 = (dm * w1 + dt * (pp[0, 0, 1] - pp) - p1 * wsr_top - aa * w2[0, 0, -1]) / bet
    with computation(BACKWARD), interval(0, -1):
        w2 = w2 - gam[0, 0, 1] * w2[0, 0, 1]
    with computation(FORWARD):
        with interval(0, 1):
            pe2 = 0.0
        with interval(1, None):
            pe2 = pe2[0, 0, -1] + dm[0, 0, -1] * (w2[0, 0, -1] - w1[0, 0, -1]) * rdt
    with computation(BACKWARD):
        with interval(-2, -1):
            p1 = (pe2 + 2.0 * pe2[0, 0, 1]) * 1.0 / 3.0
        with interval(0, -2):
            p1 = (pe2 + bb * pe2[0, 0, 1] + g_rat * pe2[0, 0, 2]) * 1.0 / 3.0 - g_rat * p1[0, 0, 1]
    with computation(PARALLEL), interval(0, -1):
        maxp = p_fac * pm if p_fac * dm >  p1 + pm else  p1 + pm
        dz2 = -dm * constants.RDGAS * ptr * maxp**(cp - 1.0)
                    

# TODO: implement MOIST_CAPPA=false
def solve(is_, ie, dt, gm2, cp2, pe2, dm, pm2, pem, w2, dz2, ptr, wsr):
    grid = spec.grid
    nic = ie - is_ + 1
    km = grid.npz - 1
    npz = grid.npz
    simshape = pe2.shape
    simorigin=(is_, grid.js - 1, 0)
    simdomain=(nic, grid.njc + 2, grid.npz)
    simdomainplus=(nic, grid.njc + 2, grid.npz+1)
    t1g = 2.0 * dt * dt
    rdt = 1.0 / dt
    tmpslice = (slice(is_, ie+1), slice(grid.js - 1, grid.je+2), slice(0, km + 1))
    g_rat = utils.make_storage_from_shape(simshape, simorigin)
    bb = utils.make_storage_from_shape(simshape, simorigin)
    aa = utils.make_storage_from_shape(simshape, simorigin)
    dd = utils.make_storage_from_shape(simshape, simorigin)
    gam = utils.make_storage_from_shape(simshape, simorigin)
    w1 = utils.make_storage_from_shape(simshape, simorigin)
    pp = utils.make_storage_from_shape(simshape, simorigin)
    p1 = utils.make_storage_from_shape(simshape, simorigin)
    pp = utils.make_storage_from_shape(simshape, simorigin)
    # TODO put into a stencil
    #pe2[tmpslice] = np.exp(gm2[tmpslice] * np.log(-dm[tmpslice] / dz2[tmpslice] * constants.RDGAS * ptr[tmpslice])) - pm2[tmpslice]
    
    initial(w2, dm, gm2, dz2, ptr, pm2, pe2, g_rat, bb, dd, w1, origin=simorigin, domain=simdomain)
    bet = utils.make_storage_data(bb.data[:,:,0], simshape)
    w_solver(aa, bet, g_rat, gam, pp, dd, gm2, dz2, pem, dm, pe2, bb, t1g, origin=simorigin, domain=simdomainplus)
    # reset bet column to the new value. TODO reuse the same storage
    bet = utils.make_storage_data(dm.data[:,:,0] - aa.data[:, :, 1], simshape)
    wsr_top = utils.make_storage_data(wsr.data[:,:,0], simshape)

    w2_pe2_dz2_compute(dm, w1, pp, aa, gm2, dz2, pem, wsr_top, bb, g_rat, bet, gam, p1, pe2, w2, pm2, ptr, cp2, dt, t1g, rdt,  spec.namelist['p_fac'], origin=simorigin, domain=simdomainplus)
     
