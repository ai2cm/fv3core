import fv3.utils.gt4py_utils as utils
from fv3.utils.corners import fill2_4corners, fill_4corners
import gt4py.gtscript as gtscript
import fv3._config as spec
from gt4py.gtscript import computation, interval, PARALLEL
import fv3.stencils.copy_stencil as cp
import fv3.stencils.a2b_ord4 as a2b_ord4
from math import log

sd = utils.sd

def grid():
    return spec.grid

@utils.stencil()
def applya2b(arg1: sd, arg2: sd):
    with computation(PARALLEL), interval(...):
        a2b_ord4.compute(arg1, arg2, replace=True)

def compute(u, v, pp, gz, pk3, delp, dt):
    '''
    u=u v=v pp=pkc gz=gz pk3=pk3 delp=delp dt=dt
    '''
    grid = spec.grid
    orig = (grid.is_, grid.js, 0)
    top_value = peln1 if spec.namelist["use_logp"] else ptk

    wk1 = utils.make_storage_from_shape(pp.shape, origin=orig)
    # do j=js,je+1
    # do i=is,ie+1
    pp[0, 0, ks_] = 0.
    pk3[0, 0, ks_] = top_value

    applya2b(pp, wk1, origin=(grid.is_, grid.js, 1), domain=(grid.nic, grid.njc, grid.npz-1))
    applya2b(pk3, wk1, origin=(grid.is_, grid.js, 1), domain=(grid.nic, grid.njc, grid.npz-1))
    applya2b(gz, wk1, origin=orig, domain=(grid.nic, grid.njc, grid.npz))
    applya2b(delp, wk1, origin=orig, domain=(grid.nic, grid.njc, grid.npz))
    