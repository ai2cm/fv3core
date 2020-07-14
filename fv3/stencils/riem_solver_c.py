#!/usr/bin/env python3
import fv3.utils.gt4py_utils as utils
import fv3._config as spec
import fv3.utils.global_constants as constants
import numpy as np
import fv3.stencils.sim1_solver as sim1_solver
import fv3.stencils.copy_stencil as cp

sd = utils.sd

@utils.stencil()
def precompute(cp3: sd, gz: sd, dm: sd, q_con: sd, pem: sd, peg: sd, dz: sd, gm: sd, pef: sd, ptop: float):
    with computation(FORWARD):
        with interval(0, 1):
            pem = ptop
            peg = ptop
            pef = ptop
        with interval(1, None):
            pem = pem[0, 0, -1] + dm[0, 0, -1]
            peg = peg[0, 0, -1] + dm[0, 0, -1] * (1.0 - q_con[0, 0, -1])
            pef = ptop
    with computation(PARALLEL), interval(0, -1):
        dz = gz[0, 0, 1] - gz
    with computation(PARALLEL), interval(...):
        gm = 1.0 / (1.0 - cp3)
        dm = dm / constants.GRAV

@utils.stencil()
def finalize(pe2: sd, pem: sd, hs_0: sd, dz: sd, pef: sd, gz: sd):
    with computation(PARALLEL), interval(1, None):
        pef = pe2 + pem
    with computation(BACKWARD):
        with interval(-1, None):
            gz = hs_0
        with interval(0, -1):
            gz = gz[0, 0, 1] - dz * constants.GRAV
            
# TODO: this is totally inefficient, can we use stencils?
def compute(ms, dt2, akap, cappa, ptop, hs, w3, ptc, q_con, delpc, gz, pef, ws):
    grid = spec.grid
    is1 = grid.is_ - 1
    ie1 = grid.ie + 1
    km = spec.grid.npz - 1
    islice = slice(is1, ie1 + 1)
    kslice = slice(0, km + 1)
    kslice_shift = slice(1, km + 2)
    shape = w3.shape
    domain = (spec.grid.nic + 2, grid.njc + 2, km + 2)
    riemorigin=(is1, grid.js - 1, 0)
    dm = cp.copy(delpc, (0, 0, 0))
    cp3 = cp.copy(cappa, (0, 0, 0))
    w = cp.copy(w3, (0, 0, 0))
    #pef[islice, spec.grid.js - 1 : spec.grid.je + 2, 0] = ptop
    pem = utils.make_storage_from_shape(shape, riemorigin) #np.zeros(shape1)
    peg = utils.make_storage_from_shape(shape, riemorigin) #np.zeros(shape1)
    pe = utils.make_storage_from_shape(shape, riemorigin) #np.zeros(shape1)
    gm = utils.make_storage_from_shape(shape, riemorigin)
    dz = utils.make_storage_from_shape(shape, riemorigin)
    pm = utils.make_storage_from_shape(shape, riemorigin)
    precompute(cp3, gz, dm, q_con, pem, peg, dz, gm, pef, ptop, origin=riemorigin, domain=domain)
    print(ptop, np.where(dz[:, grid.js - 1, 0] == 0.0))
    #TODO add to stencil when we have math functions
    jslice=slice(grid.js - 1, grid.je + 2)
    tmpslice_shift = (islice, jslice, kslice_shift)
    tmpslice = (islice, jslice, kslice)
    pm[tmpslice] = (peg[tmpslice_shift] - peg[tmpslice]) / np.log(peg[tmpslice_shift] / peg[tmpslice])
    print(np.any(np.isnan(peg)))
    #for j in range(shape[1]):
    #    print(j, np.where(np.isnan(pm[:, j, :])))
    print('pre solve', np.any(np.isnan(dz.data)))
    print(np.any(np.isnan(gm.data)))
    print(np.any(np.isnan(cp3.data)))
    print(np.any(np.isnan(pe.data)))
    print(np.any(np.isnan(dm.data)),  np.any(np.isnan(pm.data)),  np.any(np.isnan(pem.data)),  np.any(np.isnan(w.data)),  np.any(np.isnan(ptc.data)),  np.any(np.isnan(ws.data)))
    sim1_solver.solve(
        is1, ie1, dt2, gm, cp3, pe, dm, pm, pem, w, dz, ptc, ws
    )
    print('post solve', np.any(np.isnan(dz)))
    hs_0 = utils.make_storage_data(hs[:, :, 0].data, shape)
    finalize(pe, pem, hs_0, dz, pef, gz, origin=riemorigin, domain=domain)
    '''
    for j in range(spec.grid.js - 1, spec.grid.je + 2):
        dm2 = np.squeeze(dm.data[islice, j, kslice])
        cp2 = np.squeeze(cp3[islice, j, kslice])
        ptr = ptc.data[islice, j, kslice]
        wsr = ws[islice, j, :]
        pem[:, 0] = ptop
        peg[:, 0] = ptop
        for k in range(1, km + 2):
            pem[:, k] = pem[:, k - 1] + dm2[:, k - 1]
            peg[:, k] = peg[:, k - 1] + dm2[:, k - 1] * (1.0 - q_con[islice, j, k - 1])
        dz2 = gz[islice, j, kslice_shift] - gz[islice, j, kslice]
        pm2 = (peg[:, kslice_shift] - peg[:, kslice]) / np.log(
            peg[:, kslice_shift] / peg[:, kslice]
        )
        gm2 = 1.0 / (1 - cp2)
        dm2 = dm2 / constants.GRAV
        w2 = np.copy(w3[islice, j, kslice])

        sim1_solver.solve(
            is1, ie1, dt2, gm2, cp2, pe2, dm2, pm2, pem, w2, dz2, ptr, wsr
        )

        pef[islice, j, kslice_shift] = pe2[:, kslice_shift] + pem[:, kslice_shift]
        gz[islice, j, km + 1] = hs[islice, j, 0]
        for k in range(km, -1, -1):
            gz[islice, j, k] = gz[islice, j, k + 1] - dz2[:, k] * constants.GRAV
    '''
