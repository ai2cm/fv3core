import gt4py as gt
import gt4py.gtscript as gtscript
import numpy as np

import fv3core._config as spec
import fv3core.utils.gt4py_utils as utils


sd = utils.sd
origin = utils.origin


@utils.stencil()
def update_zonal_velocity(
    vorticity: sd,
    ke: sd,
    velocity: sd,
    velocity_c: sd,
    cosa: sd,
    sina: sd,
    rdxc: sd,
    dt2: float,
):
    from __splitters__ import i_end, i_start

    with computation(PARALLEL), interval(...):
        tmp_flux = dt2 * (velocity - velocity_c * cosa) / sina
        if __INLINED(spec.namelist.grid_type < 3):
            # additional assumption: not __INLINED(spec.grid.nested)
            with parallel(region[i_start, :], region[i_end + 1, :]):
                tmp_flux = dt2 * velocity
        flux = vorticity[0, 0, 0] if tmp_flux > 0.0 else vorticity[0, 1, 0]
        velocity_c = velocity_c + tmp_flux * flux + rdxc * (ke[-1, 0, 0] - ke)


@utils.stencil()
def update_meridional_velocity(
    vorticity: sd,
    ke: sd,
    velocity: sd,
    velocity_c: sd,
    cosa: sd,
    sina: sd,
    rdyc: sd,
    dt2: float,
):
    from __splitters__ import j_end, j_start

    with computation(PARALLEL), interval(...):
        tmp_flux = dt2 * (velocity - velocity_c * cosa) / sina
        if __INLINED(spec.namelist.grid_type < 3):
            # additional assumption: not __INLINED(spec.grid.nested)
            with parallel(region[:, j_start], region[:, j_end + 1]):
                tmp_flux = dt2 * velocity
        flux = vorticity[0, 0, 0] if tmp_flux > 0.0 else vorticity[1, 0, 0]
        velocity_c = velocity_c - tmp_flux * flux + rdyc * (ke[0, -1, 0] - ke)


def compute(uc: sd, vc: sd, vort_c: sd, ke_c: sd, v: sd, u: sd, dt2: float):
    """Update the C-Grid zonal and meridional velocity fields.

    Args:
        uc (input, output): x-velocity on C-grid
        vc (input, output): y-velocity on C-grid
        vort_c (input): Vorticity on C-grid
        ke_c (input): kinetic energy on C-grid
        v (input): y-velocit on D-grid
        u (input): x-velocity on D-grid
        dt2 (input): timestep
    """
    grid = spec.grid
    update_meridional_velocity(
        vort_c,
        ke_c,
        u,
        vc,
        grid.cosa_v,
        grid.sina_v,
        grid.rdyc,
        dt2,
        origin=grid.compute_origin(),
        domain=grid.domain_shape_compute_buffer_2d(add=(0, 1, 0)),
        splitters=grid.splitters,
    )
    update_zonal_velocity(
        vort_c,
        ke_c,
        v,
        uc,
        grid.cosa_u,
        grid.sina_u,
        grid.rdxc,
        dt2,
        origin=grid.compute_origin(),
        domain=grid.domain_shape_compute_buffer_2d(add=(1, 0, 0)),
        # domain=(grid.nic + 1, grid.njc, grid.npz),
        splitters=grid.splitters,
    )
