import fv3core.utils.gt4py_utils as utils
import fv3core._config as spec

sd = utils.sd
origin = utils.origin

@utils.stencil()
def update_zonal_velocity( vorticity: sd, ke: sd, velocity: sd, velocity_c: sd, cosa: sd, sina: sd, rdxc: sd, dt2: float ):
    from __splitters__ import i_start, i_end

    with computation(PARALLEL), interval(...):
        tmp_flux = dt2 * (velocity - velocity_c * cosa) / sina
        with parallel(region[i_start:i_start+1,:],region[i_end+1:i_end+2,:]):
            tmp_flux = dt2 * velocity
        flux = vorticity if tmp_flux > 0.0 else vorticity[0, 1, 0]
        velocity_c = velocity_c + tmp_flux * flux + rdxc * (ke[-1, 0, 0] - ke)

@utils.stencil()
def update_meridional_velocity( vorticity: sd, ke: sd, velocity: sd, velocity_c: sd, cosa: sd, sina: sd, rdyc:sd, dt2: float ):
    from __splitters__ import j_start, j_end

    with computation(PARALLEL), interval(...):
        tmp_flux = dt2 * (velocity - velocity_c * cosa) / sina
        with parallel(region[:,j_start:j_start+1],region[:,j_end+1:j_end+2]):
            tmp_flux = dt2 * velocity
        flux = vorticity if tmp_flux > 0.0 else vorticity[1, 0, 0]
        velocity_c = velocity_c - tmp_flux * flux + rdyc * (ke[0, -1, 0] - ke)

# Update the C-Grid zonal and meridional velocity fields
def compute(uc, vc, vort_c, ke_c, v, u, fxv, fyv, dt2):
    grid = spec.grid
    update_meridional_velocity( vort_c, ke_c, u, vc, grid.cosa_v, grid.sina_v, grid.rdyc, dt2,
                                origin=grid.compute_origin(), 
                                domain=(grid.nic, grid.njc + 1, grid.npz),
                                splitters={"j_start": grid.js-grid.js, "j_end": grid.je-grid.js} )

    update_zonal_velocity( vort_c, ke_c, v, uc, grid.cosa_u, grid.sina_u, grid.rdxc, dt2, 
                           origin=grid.compute_origin(), 
                           domain=(grid.nic + 1, grid.njc, grid.npz),
                           splitters={"i_start": grid.is_-grid.is_, "i_end": grid.ie-grid.is_} )