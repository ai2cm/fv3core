import fv3.utils.gt4py_utils as utils
import gt4py.gtscript as gtscript
import fv3._config as spec
from gt4py.gtscript import computation, interval, PARALLEL

sd = utils.sd
origin = utils.origin

# Vorticity field computation
@gtscript.stencil(backend=utils.backend, rebuild=True)
def copy_vc_values(vort: sd, vc: sd, va: sd):
    with computation(PARALLEL), interval(...):
        vort[0, 0, 0] = vc if va > 0.0 else vc[0, 1, 0]


@gtscript.stencil(backend=utils.backend, rebuild=True)
def update_vorticity_edge_values(vort: sd, va: sd, u: sd, sin: sd, cos: sd):
    with computation(PARALLEL), interval(...):
        vort[0, 0, 0] = vort * sin + u * cos if va > 0.0 else vort


@gtscript.stencil(backend=utils.backend, rebuild=True)
def update_vorticity_outer_edge_values(vort: sd, va: sd, u: sd, sin: sd, cos: sd):
    with computation(PARALLEL), interval(...):
        vort[0, 0, 0] = vort * sin + u[0, 1, 0] * cos if va <= 0.0 else vort


# Kinetic energy field computations
@gtscript.stencil(backend=utils.backend, rebuild=True)
def copy_uc_values(ke: sd, uc: sd, ua: sd):
    with computation(PARALLEL), interval(...):
        ke[0, 0, 0] = uc if ua > 0.0 else uc[1, 0, 0]


@gtscript.stencil(backend=utils.backend, rebuild=True)
def update_kinetic_energy(ke: sd, vort: sd, ua: sd, va: sd, dt2: float):
    with computation(PARALLEL), interval(...):
        ke[0, 0, 0] = 0.5 * dt2 * (ua * ke + va * vort)


@gtscript.stencil(backend=utils.backend, rebuild=True)
def update_ke_edge_values(ke: sd, ua: sd, v: sd, sin: sd, cos: sd):
    with computation(PARALLEL), interval(...):
        ke[0, 0, 0] = ke * sin + v * cos if ua > 0.0 else ke


@gtscript.stencil(backend=utils.backend, rebuild=True)
def update_ke_outer_edge_values(ke: sd, ua: sd, v: sd, sin: sd, cos: sd):
    with computation(PARALLEL), interval(...):
        ke[0, 0, 0] = ke * sin + v[1, 0, 0] * cos if ua <= 0.0 else ke


def compute(uc, vc, u, v, ua, va, dt2):
    grid = spec.grid
    # co = grid.compute_origin()
    origin = (grid.is_ - 1, grid.js - 1, 0)

    # Create storage objects to hold the new vorticity and kinetic energy values
    ke_c = utils.make_storage_from_shape(uc.shape, origin=origin)
    vort_c = utils.make_storage_from_shape(vc.shape, origin=origin)

    # Set vorticity and kinetic energy values (ignoring edge values)
    copy_domain = (grid.nic + 2, grid.njc + 2, grid.npz)
    copy_uc_values(ke_c, uc, ua, origin=origin, domain=copy_domain)
    copy_vc_values(vort_c, vc, va, origin=origin, domain=copy_domain)

    # If we are NOT using a nested grid configuration, then edge values need to be evaluated separately
    if spec.namelist["grid_type"] < 3 and not grid.nested:
        vort_domain = (grid.npx + 2, 1, grid.npz)
        if grid.south_edge:
            update_vorticity_outer_edge_values(
                vort_c,
                va,
                u,
                grid.sin_sg4,
                grid.cos_sg4,
                origin=origin,
                domain=vort_domain,
            )
            update_vorticity_edge_values(
                vort_c,
                va,
                u,
                grid.sin_sg2,
                grid.cos_sg2,
                origin=(grid.is_ - 1, grid.js, 0),
                domain=vort_domain,
            )

        if grid.north_edge:
            update_vorticity_outer_edge_values(
                vort_c,
                va,
                u,
                grid.sin_sg4,
                grid.cos_sg4,
                origin=(grid.is_ - 1, grid.npy + 1, 0),
                domain=vort_domain,
            )
            update_vorticity_edge_values(
                vort_c,
                va,
                u,
                grid.sin_sg2,
                grid.cos_sg2,
                origin=(grid.is_ - 1, grid.npy + 2, 0),
                domain=vort_domain,
            )

        ke_domain = (1, grid.npy + 3, grid.npz)
        if grid.east_edge:
            update_ke_outer_edge_values(
                ke_c,
                ua,
                v,
                grid.sin_sg3,
                grid.cos_sg3,
                origin=(grid.npx + 1, grid.js - 1, 0),
                domain=ke_domain,
            )
            update_ke_edge_values(
                ke_c,
                ua,
                v,
                grid.sin_sg1,
                grid.cos_sg1,
                origin=(grid.npx + 2, grid.js - 1, 0),
                domain=ke_domain,
            )

        if grid.west_edge:
            update_ke_outer_edge_values(
                ke_c,
                ua,
                v,
                grid.sin_sg3,
                grid.cos_sg3,
                origin=(grid.is_ - 1, grid.js - 1, 0),
                domain=ke_domain,
            )
            update_ke_edge_values(
                ke_c,
                ua,
                v,
                grid.sin_sg1,
                grid.cos_sg1,
                origin=(grid.is_, grid.js - 1, 0),
                domain=ke_domain,
            )

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
