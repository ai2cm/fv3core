import gt4py.gtscript as gtscript
from gt4py.gtscript import PARALLEL, computation, interval

import fv3core._config as spec
import fv3core.stencils.delnflux as delnflux
import fv3core.stencils.xppm as xppm
import fv3core.stencils.yppm as yppm
import fv3core.utils.corners as corners
import fv3core.utils.global_config as global_config
import fv3core.utils.gt4py_utils as utils
from fv3core.utils.typing import FloatField


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


def transport_flux(f: FloatField, f2: FloatField, mf: FloatField):
    with computation(PARALLEL), interval(...):
        f = 0.5 * (f + f2) * mf


@utils.cache_stencil_class
class FvTp2d:
    """
    ONLY USE_SG=False compiler flag implementes
    """

    def __init__(self, namelist, hord, cache_key=""):
        shape = spec.grid.domain_shape_full(add=(1, 1, 1))
        origin = spec.grid.compute_origin()
        self.tmp_q_i = utils.make_storage_from_shape(shape, origin)
        self.tmp_q_j = utils.make_storage_from_shape(shape, origin)
        self.tmp_fx2 = utils.make_storage_from_shape(shape, origin)
        self.tmp_fy2 = utils.make_storage_from_shape(shape, origin)
        self.ord_ou = hord
        self.ord_in = 8 if hord == 10 else hord
        stencil_kwargs = {
            "backend": global_config.get_backend(),
            "rebuild": global_config.get_rebuild(),
        }
        stencil_wrapper = gtscript.stencil(**stencil_kwargs)
        self.stencil_q_i = stencil_wrapper(q_i_stencil)
        self.stencil_q_j = stencil_wrapper(q_j_stencil)
        self.stencil_transport_flux = stencil_wrapper(transport_flux)
        # self.xppm_ord_ou = XPPM(namelist, iord=ord_in)
        # self.xppm_ord_in = XPPM(namelist, iord=ord_ou)
        # self.yppm_ord_in = YPPM(namelist, iord=ord_in)
        # self.yppm_ord_ou = YPPM(namelist, iord=ord_ou)

    def __call__(
        self,
        q,
        crx,
        cry,
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

        corners.copy_corners_y_stencil(
            q, origin=(grid.isd, grid.jsd, kstart), domain=(grid.nid, grid.njd, nk)
        )
        yppm.compute_flux(
            q, cry, self.tmp_fy2, self.ord_in, grid.isd, grid.ied, kstart=kstart, nk=nk
        )
        self.stencil_q_i(
            q,
            grid.area,
            yfx,
            self.tmp_fy2,
            ra_y,
            self.tmp_q_i,
            origin=(grid.isd, grid.js, kstart),
            domain=(grid.nid, grid.njc + 1, nk),
        )

        xppm.compute_flux(
            self.tmp_q_i, crx, fx, self.ord_ou, grid.js, grid.je, kstart=kstart, nk=nk
        )
        # self.xppm_ord_ou.__call(self.tmp_q_i, crx, fx, grid.js,
        # grid.je, kstart=kstart, nk=nk)
        corners.copy_corners_x_stencil(
            q, origin=(grid.isd, grid.jsd, kstart), domain=(grid.nid, grid.njd, nk)
        )
        xppm.compute_flux(
            q, crx, self.tmp_fx2, self.ord_in, grid.jsd, grid.jed, kstart=kstart, nk=nk
        )
        # self.xppm_ord_in.__call(self.tmp_q_i, crx, fx, grid.js,
        # grid.je, kstart=kstart,nk=nk)
        self.stencil_q_j(
            q,
            grid.area,
            xfx,
            self.tmp_fx2,
            ra_x,
            self.tmp_q_j,
            origin=(grid.is_, grid.jsd, kstart),
            domain=(grid.nic + 1, grid.njd, nk),
        )
        yppm.compute_flux(
            self.tmp_q_j, cry, fy, self.ord_ou, grid.is_, grid.ie, kstart=kstart, nk=nk
        )

        if mfx is not None and mfy is not None:
            self.stencil_transport_flux(
                fx,
                self.tmp_fx2,
                mfx,
                origin=compute_origin,
                domain=(grid.nic + 1, grid.njc, nk),
            )
            self.stencil_transport_flux(
                fy,
                self.tmp_fy2,
                mfy,
                origin=compute_origin,
                domain=(grid.nic, grid.njc + 1, nk),
            )
            if (mass is not None) and (nord is not None) and (damp_c is not None):
                delnflux.compute_delnflux_no_sg(
                    q, fx, fy, nord, damp_c, kstart, nk, mass=mass
                )
        else:

            self.stencil_transport_flux(
                fx,
                self.tmp_fx2,
                xfx,
                origin=compute_origin,
                domain=(grid.nic + 1, grid.njc, nk),
            )
            self.stencil_transport_flux(
                fy,
                self.tmp_fy2,
                yfx,
                origin=compute_origin,
                domain=(grid.nic, grid.njc + 1, nk),
            )
            if (nord is not None) and (damp_c is not None):
                delnflux.compute_delnflux_no_sg(q, fx, fy, nord, damp_c, kstart, nk)
