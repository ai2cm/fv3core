from gt4py.gtscript import PARALLEL, computation, interval
import gt4py.gtscript as gtscript
import fv3core._config as spec
import fv3core.stencils.delnflux as delnflux
import fv3core.stencils.xppm as xppm
import fv3core.stencils.yppm as yppm
import fv3core.utils.corners as corners
import fv3core.utils.gt4py_utils as utils
from fv3core.decorators import gtstencil
import fv3core.utils.global_config as global_config


origin = (0, 0, 0)
sd = utils.sd


def q_i_stencil(q: sd, area: sd, yfx: sd, fy2: sd, ra_y: sd, q_i: sd):
    with computation(PARALLEL), interval(...):
        fyy = yfx * fy2
        q_i[0, 0, 0] = (q * area + fyy - fyy[0, 1, 0]) / ra_y


def q_j_stencil(q: sd, area: sd, xfx: sd, fx2: sd, ra_x: sd, q_j: sd):
    with computation(PARALLEL), interval(...):
        fx1 = xfx * fx2
        q_j[0, 0, 0] = (q * area + fx1 - fx1[1, 0, 0]) / ra_x


def transport_flux(f: sd, f2: sd, mf: sd):
    with computation(PARALLEL), interval(...):
        f = 0.5 * (f + f2) * mf

class FvTp2d:
    """
    ONLY USE_SG=False compiler flag implementes
    """
    def __init__(self, namelist, hord):
        shape = spec.grid.domain_shape_full(add=(1, 1, 1))
        origin = spec.grid.compute_origin()
        self.q_i = utils.make_storage_from_shape(shape, origin)
        self.q_j = utils.make_storage_from_shape(shape, origin)
        self.fx2 = utils.make_storage_from_shape(shape, origin)
        self.fy2 = utils.make_storage_from_shape(shape, origin)
        self.ord_ou = hord
        self.ord_in = 8 if hord == 10 else hord
        stencil_kwargs = {'backend':global_config.get_backend(), 'rebuild': global_config.get_rebuild()}
        stencil_wrapper = gtscript.stencil(**stencil_kwargs)
        self.q_i_stencil = stencil_wrapper(q_i_stencil)
        self.q_j_stencil = stencil_wrapper(q_j_stencil)
        self.transport_flux_stencil = stencil_wrapper(transport_flux)
        #self.xppm_ordin = XPPM(namelist, iord=self.ord_in)
        #self.xppm_ordin = XPPM(namelist, iord=self.ord_ou)
        #self.yppm = YPPM()
        
        # hord_tm, hord_tr, hord_dp, hord_vt
    def __call__(self,  q, crx, cry, xfx, yfx, ra_x, ra_y, fx, fy, kstart=0,
                 nk=None,nord=None,damp_c=None,mass=None,mfx=None,mfy=None,):
        grid = spec.grid
        if nk is None:
            nk = grid.npz - kstart
        kslice = slice(kstart, kstart + nk)
        compute_origin = (grid.is_, grid.js, kstart)

        corners.copy_corners_y_stencil(
            q, origin=(grid.isd, grid.jsd, kstart), domain=(grid.nid, grid.njd, nk)
        )
        yppm.compute_flux(q, cry, self.fy2, self.ord_in, grid.isd, grid.ied, kstart=kstart, nk=nk)
        self.q_i_stencil(
            q,
            grid.area,
            yfx,
            self.fy2,
            ra_y,
            self.q_i,
            origin=(grid.isd, grid.js, kstart),
            domain=(grid.nid, grid.njc + 1, nk),
        )

        xppm.compute_flux(self.q_i, crx, fx, self.ord_ou, grid.js, grid.je, kstart=kstart, nk=nk)
        #self.xppm_ord_ou.__call(self.q_i, crx, fx, grid.js, grid.je, kstart=kstart, nk=nk)
        corners.copy_corners_x_stencil(
            q, origin=(grid.isd, grid.jsd, kstart), domain=(grid.nid, grid.njd, nk)
        )
        xppm.compute_flux(q, crx, self.fx2, self.ord_in, grid.jsd, grid.jed, kstart=kstart, nk=nk)
        #self.xppm_ord_in.__call(self.q_i, crx, fx, grid.js, grid.je, kstart=kstart, nk=nk)   
        self.q_j_stencil(
            q,
            grid.area,
            xfx,
            self.fx2,
            ra_x,
            self.q_j,
            origin=(grid.is_, grid.jsd, kstart),
            domain=(grid.nic + 1, grid.njd, nk),
        )
        yppm.compute_flux(self.q_j, cry, fy, self.ord_ou, grid.is_, grid.ie, kstart=kstart, nk=nk)

        if mfx is not None and mfy is not None:
            self.transport_flux_stencil(
                fx, self.fx2, mfx, origin=compute_origin, domain=(grid.nic + 1, grid.njc, nk)
            )
            self.transport_flux_stencil(
                fy, self.fy2, mfy, origin=compute_origin, domain=(grid.nic, grid.njc + 1, nk)
            )
            if (mass is not None) and (nord is not None) and (damp_c is not None):
                delnflux.compute_delnflux_no_sg(
                    q, fx, fy, nord, damp_c, kstart, nk, mass=mass
                )
        else:

            self.transport_flux_stencil(
                fx, self.fx2, xfx, origin=compute_origin, domain=(grid.nic + 1, grid.njc, nk)
            )
            self.transport_flux_stencil(
                fy, self.fy2, yfx, origin=compute_origin, domain=(grid.nic, grid.njc + 1, nk)
            )
            if (nord is not None) and (damp_c is not None):
                delnflux.compute_delnflux_no_sg(q, fx, fy, nord, damp_c, kstart, nk)
                
