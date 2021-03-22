from gt4py.gtscript import PARALLEL, computation, interval

import fv3core._config as spec
import fv3core.stencils.delnflux as delnflux
import fv3core.stencils.xppm as xppm
import fv3core.stencils.yppm as yppm
import fv3core.utils.corners as corners
import fv3core.utils.gt4py_utils as utils
from fv3core.decorators import gtstencil

origin = (0, 0, 0)
sd = utils.sd


@gtstencil()
def q_i_stencil(q: sd, area: sd, yfx: sd, fy2: sd, ra_y: sd, q_i: sd):
    with computation(PARALLEL), interval(...):
        fyy = yfx * fy2
        q_i[0, 0, 0] = (q * area + fyy - fyy[0, 1, 0]) / ra_y


@gtstencil()
def q_j_stencil(q: sd, area: sd, xfx: sd, fx2: sd, ra_x: sd, q_j: sd):
    with computation(PARALLEL), interval(...):
        fx1 = xfx * fx2
        q_j[0, 0, 0] = (q * area + fx1 - fx1[1, 0, 0]) / ra_x


@gtstencil()
def transport_flux(f: sd, f2: sd, mf: sd):
    with computation(PARALLEL), interval(...):
        f = 0.5 * (f + f2) * mf


def compute(data, nord_column):
    for optional_arg in ["mass", "mfx", "mfy"]:
        if optional_arg not in data:
            data[optional_arg] = None
    # utils.compute_column_split(
    #     compute_no_sg, data, nord_column, "nord", ["q", "fx", "fy"], grid
    # )
    raise NotImplementedError()


def compute_no_sg(
    q,
    crx,
    cry,
    hord,
    xfx,
    yfx,
    ra_x,
    ra_y,
    fx,
    fy,
    kstart=0,
    nk=None,
    nord=None,
    damp_c=None,
    mass=None,
    mfx=None,
    mfy=None,
):
    grid = spec.grid
    if nk is None:
        nk = grid.npz - kstart
    kslice = slice(kstart, kstart + nk)
    compute_origin = (grid.is_, grid.js, kstart)
    q_i = utils.make_storage_from_shape(q.shape, (grid.isd, grid.js, kstart))
    q_j = utils.make_storage_from_shape(q.shape, (grid.is_, grid.jsd, kstart))
    fy2 = utils.make_storage_from_shape(q.shape, compute_origin)
    fx2 = utils.make_storage_from_shape(q.shape, compute_origin)
    if hord == 10:
        ord_in = 8
    else:
        ord_in = hord
    ord_ou = hord

    xppm_object_in = xppm.XPPM(spec.namelist, ord_in)
    yppm_object_in = yppm.YPPM(spec.namelist, ord_in)
    xppm_object_ou = xppm.XPPM(spec.namelist, ord_ou)
    yppm_object_ou = yppm.YPPM(spec.namelist, ord_ou)

    corners.copy_corners_y_stencil(
        q, origin=(grid.isd, grid.jsd, kstart), domain=(grid.nid, grid.njd, nk)
    )
    yppm_object_in(q, cry, fy2, grid.isd, grid.ied, kstart=kstart, nk=nk)
    q_i_stencil(
        q,
        grid.area,
        yfx,
        fy2,
        ra_y,
        q_i,
        origin=(grid.isd, grid.js, kstart),
        domain=(grid.nid, grid.njc + 1, nk),
    )

    xppm_object_ou(q_i, crx, fx, grid.js, grid.je, kstart=kstart, nk=nk)
    corners.copy_corners_x_stencil(
        q, origin=(grid.isd, grid.jsd, kstart), domain=(grid.nid, grid.njd, nk)
    )
    xppm_object_in(q, crx, fx2, grid.jsd, grid.jed, kstart=kstart, nk=nk)
    q_j_stencil(
        q,
        grid.area,
        xfx,
        fx2,
        ra_x,
        q_j,
        origin=(grid.is_, grid.jsd, kstart),
        domain=(grid.nic + 1, grid.njd, nk),
    )
    yppm_object_ou(q_j, cry, fy, grid.is_, grid.ie, kstart=kstart, nk=nk)

    if mfx is not None and mfy is not None:
        transport_flux(
            fx, fx2, mfx, origin=compute_origin, domain=(grid.nic + 1, grid.njc, nk)
        )
        transport_flux(
            fy, fy2, mfy, origin=compute_origin, domain=(grid.nic, grid.njc + 1, nk)
        )
        if (mass is not None) and (nord is not None) and (damp_c is not None):
            delnflux.compute_delnflux_no_sg(
                q, fx, fy, nord, damp_c, kstart, nk, mass=mass
            )
    else:

        transport_flux(
            fx, fx2, xfx, origin=compute_origin, domain=(grid.nic + 1, grid.njc, nk)
        )
        transport_flux(
            fy, fy2, yfx, origin=compute_origin, domain=(grid.nic, grid.njc + 1, nk)
        )
        if (nord is not None) and (damp_c is not None):
            delnflux.compute_delnflux_no_sg(q, fx, fy, nord, damp_c, kstart, nk)
