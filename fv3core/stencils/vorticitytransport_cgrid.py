from gt4py.gtscript import __INLINED, PARALLEL, computation, interval, parallel, region

import fv3core._config as spec
import fv3core.utils.gt4py_utils as utils
from fv3core.decorators import gtstencil


FloatField = utils.FloatField


@gtstencil()
def update_velocity(
    vort: FloatField,
    ke: FloatField,
    u: FloatField,
    vc: FloatField,
    v: FloatField,
    uc: FloatField,
    cosa_u: FloatField,
    sina_u: FloatField,
    cosa_v: FloatField,
    sina_v: FloatField,
    rdxc: FloatField,
    rdyc: FloatField,
    dt2: float,
):
    from __externals__ import i_end, i_start, j_end, j_start, namelist

    with computation(PARALLEL), interval(...):
        assert __INLINED(namelist.grid_type < 3)
        # additional assumption: not __INLINED(spec.grid.nested)

        # update_meridional_velocity
        with parallel(region[:i_end + 1, :]):
            tmp_flux = dt2 * (u - vc * cosa_v) / sina_v
        with parallel(region[:i_end + 1, j_start], region[:i_end + 1, j_end + 1]):
            tmp_flux = dt2 * u
        with parallel(region[:i_end + 1, :]):
            flux = vort if tmp_flux > 0.0 else vort[1, 0, 0]
            vc = vc - tmp_flux * flux + rdyc * (ke[0, -1, 0] - ke)

        # update_zonal_velocity
        with parallel(region[:, :j_end + 1]):
            tmp_flux = dt2 * (v - uc * cosa_u) / sina_u
        with parallel(region[i_start, :j_end + 1], region[i_end + 1, :j_end + 1]):
            tmp_flux = dt2 * v
        with parallel(region[:, :j_end + 1]):
            flux = vort if tmp_flux > 0.0 else vort[0, 1, 0]
            uc = uc + tmp_flux * flux + rdxc * (ke[-1, 0, 0] - ke)


def compute(uc: FloatField, vc: FloatField, vort_c: FloatField, ke_c: FloatField, v: FloatField, u: FloatField, dt2: float):
    """Update the C-Grid zonal and meridional velocity fields.

    Args:
        uc: x-velocity on C-grid (input, output)
        vc: y-velocity on C-grid (input, output)
        vort_c: Vorticity on C-grid (input)
        ke_c: kinetic energy on C-grid (input)
        v: y-velocit on D-grid (input)
        u: x-velocity on D-grid (input)
        dt2: timestep (input)
    """
    grid = spec.grid
    update_velocity(
        vort_c,
        ke_c,
        u,
        vc,
        v,
        uc,
        grid.cosa_u,
        grid.sina_u,
        grid.cosa_v,
        grid.sina_v,
        grid.rdxc,
        grid.rdyc,
        dt2,
        origin=grid.compute_origin(),
        domain=grid.domain_shape_compute_buffer_2d(add=(1, 1, 0)),
    )
