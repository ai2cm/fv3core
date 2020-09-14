import fv3core.utils.gt4py_utils as utils
import gt4py as gt
import gt4py.gtscript as gtscript
import fv3core._config as spec
import numpy as np

sd = utils.sd
origin = utils.origin

##
## Stencil Definitions
##

# Flux field computations
@utils.stencil()
def compute_tmp_flux1( flux: sd, vorticity: sd, tmp_flux: sd, velocity: sd, velocity_c: sd, cosa: sd, sina: sd, dt2: float ):
    from __splitters__ import j_start, j_end

    with computation(PARALLEL), interval(...):
        tmp_flux = dt2 * (velocity - velocity_c * cosa) / sina
    with computation(PARALLEL), interval(...):
        with parallel(region[:,j_start:j_start+1]):
            tmp_flux = dt2 * velocity
        with parallel(region[:,j_end+1:j_end+2]):
            tmp_flux = dt2 * velocity
    with computation(PARALLEL), interval(...):
        flux = vorticity if tmp_flux > 0.0 else vorticity[1, 0, 0]

@utils.stencil()
def compute_tmp_flux2( flux: sd, vorticity: sd, tmp_flux: sd, velocity: sd, velocity_c: sd, cosa: sd, sina: sd, dt2: float ):
    from __splitters__ import i_start, i_end

    with computation(PARALLEL), interval(...):
        tmp_flux = dt2 * (velocity - velocity_c * cosa) / sina
    with computation(PARALLEL), interval(...):
        with parallel(region[i_start:i_start+1,:]):
            tmp_flux = dt2 * velocity
        with parallel(region[i_end+1:i_end+2,:]):
            tmp_flux = dt2 * velocity
    with computation(PARALLEL), interval(...):
        flux = vorticity if tmp_flux > 0.0 else vorticity[0, 1, 0]


# Wind speed updates
@utils.stencil()
def update_uc(uc: sd, fy1: sd, fy: sd, rdxc: sd, ke: sd):
    with computation(PARALLEL), interval(...):
        uc = uc + fy1 * fy + rdxc * (ke[-1, 0, 0] - ke)


@utils.stencil()
def update_vc(vc: sd, fx1: sd, fx: sd, rdyc: sd, ke: sd):
    with computation(PARALLEL), interval(...):
        vc = vc - fx1 * fx + rdyc * (ke[0, -1, 0] - ke)


def compute(uc, vc, vort_c, ke_c, v, u, fxv, fyv, dt2):
    grid = spec.grid
    co = grid.compute_origin()
    zonal_domain = (grid.nic + 1, grid.njc, grid.npz)
    meridional_domain = (grid.nic, grid.njc + 1, grid.npz)
    splitters={"i_start": grid.is_-grid.is_,
               "i_end": grid.ie-grid.is_,
               "j_start": grid.js-grid.js,
               "j_end": grid.je-grid.js}

    # Create storage objects for the temporary flux fields
    fx1 = utils.make_storage_from_shape(uc.shape, origin=co)
    fy1 = utils.make_storage_from_shape(vc.shape, origin=co)

    # Compute the flux values in the zonal coordinate direction
    compute_tmp_flux1( fxv, vort_c, fx1, u, vc, grid.cosa_v, grid.sina_v, dt2, 
                      origin=co, 
                      domain=meridional_domain,
                      splitters=splitters )

    # Compute the flux values in the meridional coordinate direction
    compute_tmp_flux2( fyv, vort_c, fy1, v, uc, grid.cosa_u, grid.sina_u, dt2, 
                       origin=co, 
                       domain=zonal_domain,
                       splitters=splitters )

    # Update time-centered winds on C-grid
    update_uc(uc, fy1, fyv, grid.rdxc, ke_c, origin=co, domain=zonal_domain)
    update_vc(vc, fx1, fxv, grid.rdyc, ke_c, origin=co, domain=meridional_domain)
