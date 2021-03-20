import gt4py.gtscript as gtscript
from gt4py.gtscript import PARALLEL, computation, horizontal, interval, region

import fv3core._config as spec
import fv3core.stencils.d_sw as d_sw
import fv3core.stencils.delnflux as delnflux
import fv3core.stencils.xppm as xppm
import fv3core.stencils.yppm as yppm
import fv3core.utils.corners as corners
import fv3core.utils.gt4py_utils as utils
from fv3core.decorators import gtstencil
from fv3core.utils.typing import FloatField


@gtstencil()
def q_i_stencil(
    q: FloatField,
    area: FloatField,
    yfx: FloatField,
    fy2: FloatField,
    ra_y: FloatField,
    q_i: FloatField,
):
    with computation(PARALLEL), interval(...):
        fyy = yfx * fy2
        q_i[0, 0, 0] = (q * area + fyy - fyy[0, 1, 0]) / ra_y


@gtstencil()
def q_j_stencil(
    q: FloatField,
    area: FloatField,
    xfx: FloatField,
    fx2: FloatField,
    ra_x: FloatField,
    q_j: FloatField,
):
    with computation(PARALLEL), interval(...):
        fx1 = xfx * fx2
        q_j[0, 0, 0] = (q * area + fx1 - fx1[1, 0, 0]) / ra_x


@gtscript.function
def transport_flux(f, f2, mf):
    return 0.5 * (f + f2) * mf


@gtstencil()
def transport_flux_xy(
    fx: FloatField,
    fx2: FloatField,
    fy: FloatField,
    fy2: FloatField,
    mfx: FloatField,
    mfy: FloatField,
):
    with computation(PARALLEL), interval(...):
        with horizontal(region[:, :-1]):
            fx = transport_flux(fx, fx2, mfx)
        with horizontal(region[:-1, :]):
            fy = transport_flux(fy, fy2, mfy)


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
    nord=None,
    damp_c=None,
    mass=None,
    mfx=None,
    mfy=None,
):
    grid = spec.grid
    q_i = utils.make_storage_from_shape(q.shape, (grid.isd, grid.js, 0))
    q_j = utils.make_storage_from_shape(q.shape, (grid.is_, grid.jsd, 0))
    fy2 = utils.make_storage_from_shape(q.shape, grid.compute_origin())
    fx2 = utils.make_storage_from_shape(q.shape, grid.compute_origin())
    if hord == 10:
        ord_in = 8
    else:
        ord_in = hord
    ord_ou = hord
    corners.copy_corners_y_stencil(
        q, origin=grid.full_origin(), domain=grid.domain_shape_full()
    )
    yppm.compute_flux(q, cry, fy2, ord_in, grid.isd, grid.ied)
    q_i_stencil(
        q,
        grid.area,
        yfx,
        fy2,
        ra_y,
        q_i,
        origin=(grid.isd, grid.js, 0),
        domain=(grid.nid, grid.njc + 1, grid.npz),
    )

    xppm.compute_flux(q_i, crx, fx, ord_ou, grid.js, grid.je)
    corners.copy_corners_x_stencil(
        q, origin=grid.full_origin(), domain=grid.domain_shape_full()
    )
    xppm.compute_flux(q, crx, fx2, ord_in, grid.jsd, grid.jed)
    q_j_stencil(
        q,
        grid.area,
        xfx,
        fx2,
        ra_x,
        q_j,
        origin=(grid.is_, grid.jsd, 0),
        domain=(grid.nic + 1, grid.njd, grid.npz),
    )
    yppm.compute_flux(q_j, cry, fy, ord_ou, grid.is_, grid.ie)

    if mfx is not None and mfy is not None:
        transport_flux_xy(
            fx,
            fx2,
            fy,
            fy2,
            mfx,
            mfy,
            origin=grid.compute_origin(),
            domain=grid.domain_shape_compute(add=(1, 1, 0)),
        )

        if (mass is not None) and (nord is not None) and (damp_c is not None):
            for kstart, nk in d_sw.k_bounds():
                delnflux.compute_delnflux_no_sg(
                    q, fx, fy, nord[kstart], damp_c[kstart], kstart, nk, mass=mass
                )
    else:
        transport_flux_xy(
            fx,
            fx2,
            fy,
            fy2,
            xfx,
            yfx,
            origin=grid.compute_origin(),
            domain=grid.domain_shape_compute(add=(1, 1, 0)),
        )

        if (nord is not None) and (damp_c is not None):
            for kstart, nk in d_sw.k_bounds():
                delnflux.compute_delnflux_no_sg(
                    q, fx, fy, nord[kstart], damp_c[kstart], kstart, nk
                )
