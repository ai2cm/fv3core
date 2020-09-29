import fv3core.utils.gt4py_utils as utils
import gt4py.gtscript as gtscript
import fv3core._config as spec
from gt4py.gtscript import computation, interval, PARALLEL

sd = utils.sd
origin = utils.origin


@utils.stencil()
def copy_values(ke: sd, uc: sd, ua: sd, vort: sd, vc: sd, va: sd):
    with computation(PARALLEL), interval(...):
        ke[0, 0, 0] = uc if ua > 0.0 else uc[1, 0, 0]
        vort[0, 0, 0] = vc if va > 0.0 else vc[0, 1, 0]


# @utils.stencil()
# def copy_uc_values(ke: sd, uc: sd, ua: sd):
#     with computation(PARALLEL), interval(...):
#         ke[0, 0, 0] = uc if ua > 0.0 else uc[1, 0, 0]


# @utils.stencil()
# def copy_vc_values(vort: sd, vc: sd, va: sd):
#     with computation(PARALLEL), interval(...):
#         vort[0, 0, 0] = vc if va > 0.0 else vc[0, 1, 0]


# Vorticity field computation


@utils.stencil()
def update_vorticity(
    vort: sd, va: sd, u: sd, sin_sg2: sd, cos_sg2: sd, sin_sg4: sd, cos_sg4: sd
):
    from __splitters__ import j_start, j_end

    with computation(PARALLEL):
        with interval(...):
            # update_vorticity_outer_edge_values (if grid.south_edge)
            vort = vort * sin_sg4 + u[0, 1, 0] * cos_sg4 if va <= 0.0 else vort

            # update_vorticity_edge_values (if grid.south_edge)
            with parallel(region[:, j_start + 1 :]):
                vort = vort * sin_sg2 + u * cos_sg2 if va > 0.0 else vort

            # update_vorticity_outer_edge_values (if grid.north_edge)
            with parallel(region[:, j_end:]):
                vort = vort * sin_sg4 + u[0, 1, 0] * cos_sg4 if va <= 0.0 else vort

            # update_vorticity_edge_values (if grid.north_edge)
            with parallel(region[:, j_start + 1 :]):
                vort = vort * sin_sg2 + u * cos_sg2 if va > 0.0 else vort


# @utils.stencil()
# def update_vorticity_edge_values(vort: sd, va: sd, u: sd, sin: sd, cos: sd):
#     with computation(PARALLEL), interval(...):
#         vort[0, 0, 0] = vort * sin + u * cos if va > 0.0 else vort


# @utils.stencil()
# def update_vorticity_outer_edge_values(vort: sd, va: sd, u: sd, sin: sd, cos: sd):
#     with computation(PARALLEL), interval(...):
#         vort[0, 0, 0] = vort * sin + u[0, 1, 0] * cos if va <= 0.0 else vort


# Kinetic energy field computations


@utils.stencil()
def update_kinetic_energy(ke: sd, vort: sd, ua: sd, va: sd, dt2: float):
    with computation(PARALLEL), interval(...):
        ke[0, 0, 0] = 0.5 * dt2 * (ua * ke + va * vort)


@utils.stencil()
def update_ke_edges(
    ke: sd, ua: sd, v: sd, sin_sg1: sd, cos_sg1: sd, sin_sg3: sd, cos_sg3: sd
):
    from __splitters__ import i_start, i_end

    # update_ke_outer_edge_values: if grid.east_edge
    with computation(PARALLEL), interval(...):
        with parallel(region[i_end:, :]):
            ke = ke * sin_sg3 + v[1, 0, 0] * cos_sg3 if ua <= 0.0 else ke
    # update_ke_edge_values: if grid.east_edge
    with computation(PARALLEL), interval(...):
        with parallel(region[i_end + 1 :, :]):
            ke = ke * sin_sg1 + v * cos_sg1 if ua > 0.0 else ke
    # update_ke_outer_edge_values: if grid.west_edge
    with computation(PARALLEL), interval(...):
        ke = ke * sin_sg3 + v[1, 0, 0] * cos_sg3 if ua <= 0.0 else ke
    # update_ke_edge_values: if grid.west_edge
    with computation(PARALLEL), interval(...):
        with parallel(region[i_start:, :]):
            ke = ke * sin_sg1 + v * cos_sg1 if ua > 0.0 else ke


# @utils.stencil()
# def update_ke_edge_values(ke: sd, ua: sd, v: sd, sin: sd, cos: sd):
#     with computation(PARALLEL), interval(...):
#         ke[0, 0, 0] = ke * sin + v * cos if ua > 0.0 else ke


# @utils.stencil()
# def update_ke_outer_edge_values(ke: sd, ua: sd, v: sd, sin: sd, cos: sd):
#     with computation(PARALLEL), interval(...):
#         ke[0, 0, 0] = ke * sin + v[1, 0, 0] * cos if ua <= 0.0 else ke


def compute(uc, vc, u, v, ua, va, dt2):
    grid = spec.grid
    origin = (grid.is_ - 1, grid.js - 1, 0)

    splitters = {
        "i_start": grid.is_ - grid.is_,
        "i_end": grid.ie - grid.is_,
        "j_start": grid.js - grid.js,
        "j_end": grid.je - grid.js,
    }

    # Create storage objects to hold the new vorticity and kinetic energy values
    ke_c = utils.make_storage_from_shape(uc.shape, origin=origin)
    vort_c = utils.make_storage_from_shape(vc.shape, origin=origin)

    # Set vorticity and kinetic energy values (ignoring edge values)
    copy_domain = (grid.nic + 2, grid.njc + 2, grid.npz)
    # copy_uc_values(ke_c, uc, ua, origin=origin, domain=copy_domain)
    # copy_vc_values(vort_c, vc, va, origin=origin, domain=copy_domain)
    copy_values(ke_c, uc, ua, vort_c, vc, va, origin=origin, domain=copy_domain)

    vort_domain = (grid.ie + 1, 1, grid.npz)
    update_vorticity(
        vort_c,
        va,
        u,
        grid.sin_sg2,
        grid.cos_sg2,
        grid.sin_sg,
        grid.cos_sg4,
        origin=origin,
        domain=vort_domain,
    )

    # If we are NOT using a nested grid configuration, then edge values need to be evaluated separately
    # if spec.namelist.grid_type < 3 and not grid.nested:
    #     if grid.south_edge:
    #         update_vorticity_outer_edge_values(
    #             vort_c,
    #             va,
    #             u,
    #             grid.sin_sg4,
    #             grid.cos_sg4,
    #             origin=origin,
    #             domain=vort_domain,
    #         )
    #         update_vorticity_edge_values(
    #             vort_c,
    #             va,
    #             u,
    #             grid.sin_sg2,
    #             grid.cos_sg2,
    #             origin=(grid.is_ - 1, grid.js, 0),
    #             domain=vort_domain,
    #         )

    #     if grid.north_edge:
    #         update_vorticity_outer_edge_values(
    #             vort_c,
    #             va,
    #             u,
    #             grid.sin_sg4,
    #             grid.cos_sg4,
    #             origin=(grid.is_ - 1, grid.je, 0),
    #             domain=vort_domain,
    #         )
    #         update_vorticity_edge_values(
    #             vort_c,
    #             va,
    #             u,
    #             grid.sin_sg2,
    #             grid.cos_sg2,
    #             origin=(grid.is_ - 1, grid.je + 1, 0),
    #             domain=vort_domain,
    #         )

    ke_domain = (1, grid.je + 2, grid.npz)
    update_ke_edges(
        ke_c,
        ua,
        v,
        grid.sin_sg1,
        grid.cos_sg1,
        grid.sin_sg3,
        grid.cos_sg3,
        origin=origin,
        domain=ke_domain,
    )

    # if grid.east_edge:
    #     update_ke_outer_edge_values(
    #         ke_c,
    #         ua,
    #         v,
    #         grid.sin_sg3,
    #         grid.cos_sg3,
    #         origin=(grid.ie, grid.js - 1, 0),
    #         domain=ke_domain,
    #     )
    #     update_ke_edge_values(
    #         ke_c,
    #         ua,
    #         v,
    #         grid.sin_sg1,
    #         grid.cos_sg1,
    #         origin=(grid.ie + 1, grid.js - 1, 0),
    #         domain=ke_domain,
    #     )

    # if grid.west_edge:
    #     update_ke_outer_edge_values(
    #         ke_c,
    #         ua,
    #         v,
    #         grid.sin_sg3,
    #         grid.cos_sg3,
    #         origin=(grid.is_ - 1, grid.js - 1, 0),
    #         domain=ke_domain,
    #     )
    #     update_ke_edge_values(
    #         ke_c,
    #         ua,
    #         v,
    #         grid.sin_sg1,
    #         grid.cos_sg1,
    #         origin=(grid.is_, grid.js - 1, 0),
    #         domain=ke_domain,
    #     )

    # Update kinetic energy field using computed vorticity
    update_kinetic_energy(
        ke_c,
        vort_c,
        ua,
        va,
        dt2,
        origin=origin,
        domain=(grid.nic + 2, grid.njc + 2, grid.npz),
    )
    return ke_c, vort_c
