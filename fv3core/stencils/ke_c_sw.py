from gt4py.gtscript import __INLINED, PARALLEL, computation, interval, parallel, region

import fv3core._config as spec
import fv3core.utils.gt4py_utils as utils
from fv3core.decorators import gtstencil


FloatField = utils.FloatField


@gtstencil()
def update_vorticity_and_kinetic_energy(
    ke: FloatField,
    vort: FloatField,
    ua: FloatField,
    va: FloatField,
    uc: FloatField,
    vc: FloatField,
    u: FloatField,
    v: FloatField,
    sin_sg1: FloatField,
    cos_sg1: FloatField,
    sin_sg2: FloatField,
    cos_sg2: FloatField,
    sin_sg3: FloatField,
    cos_sg3: FloatField,
    sin_sg4: FloatField,
    cos_sg4: FloatField,
    dt2: float,
):
    from __externals__ import i_end, i_start, j_end, j_start, namelist

    with computation(PARALLEL), interval(...):
        assert __INLINED(namelist.grid_type < 3)

        ke = uc if ua > 0.0 else uc[1, 0, 0]
        vort = vc if va > 0.0 else vc[0, 1, 0]

        with parallel(region[:, j_start - 1], region[:, j_end]):
            if va <= 0.0:
                vort = vort * sin_sg4 + u[0, 1, 0] * cos_sg4
        with parallel(region[:, j_start], region[:, j_end + 1]):
            if va > 0.0:
                vort = vort * sin_sg2 + u * cos_sg2

        with parallel(region[i_end, :], region[i_start - 1, :]):
            if ua <= 0.0:
                ke = ke * sin_sg3 + v[1, 0, 0] * cos_sg3
        with parallel(region[i_end + 1, :], region[i_start, :]):
            if ua > 0.0:
                ke = ke * sin_sg1 + v * cos_sg1

        ke = 0.5 * dt2 * (ua * ke + va * vort)


def compute(
    uc: FloatField,
    vc: FloatField,
    u: FloatField,
    v: FloatField,
    ua: FloatField,
    va: FloatField,
    dt2: float,
):
    grid = spec.grid
    origin = (grid.is_ - 1, grid.js - 1, 0)

    # Create storage objects to hold the new vorticity and kinetic energy values
    ke_c = utils.make_storage_from_shape(uc.shape, origin=origin)
    vort_c = utils.make_storage_from_shape(vc.shape, origin=origin)

    # Set vorticity and kinetic energy values
    update_vorticity_and_kinetic_energy(
        ke_c,
        vort_c,
        ua,
        va,
        uc,
        vc,
        u,
        v,
        grid.sin_sg1,
        grid.cos_sg1,
        grid.sin_sg2,
        grid.cos_sg2,
        grid.sin_sg3,
        grid.cos_sg3,
        grid.sin_sg4,
        grid.cos_sg4,
        dt2,
        origin=origin,
        domain=(grid.nic + 2, grid.njc + 2, grid.npz),
    )
    return ke_c, vort_c
