
from gt4py.gtscript import PARALLEL, computation, interval

import fv3core._config as spec
import fv3core.utils.gt4py_utils as utils
from fv3core.decorators import FrozenStencil
from fv3core.stencils.basic_operations import copy_defn
from fv3core.utils.grid import axis_offsets
from fv3core.utils.typing import FloatField, FloatFieldIJ


def main_ut(
    uc: FloatField,
    vc: FloatField,
    cosa_u: FloatFieldIJ,
    rsin_u: FloatFieldIJ,
    ut: FloatField,
):

    with computation(PARALLEL), interval(...):
        ut = (
            uc - 0.25 * cosa_u * (vc[-1, 0, 0] + vc + vc[-1, 1, 0] + vc[0, 1, 0])
        ) * rsin_u


# TODO: the mix of local and global regions is strange here
# it's a workaround to specify DON'T do this calculation if on the tile edge
def main_vt(
    uc: FloatField,
    vc: FloatField,
    cosa_v: FloatFieldIJ,
    rsin_v: FloatFieldIJ,
    vt: FloatField,
):

    with computation(PARALLEL), interval(...):
        vt = (
            vc - 0.25 * cosa_v * (uc[0, -1, 0] + uc[1, -1, 0] + uc + uc[1, 0, 0])
        ) * rsin_v


def fxadv_x_fluxes(
    uc: FloatField,
    vc: FloatField,
    cosa_u: FloatFieldIJ,
    rsin_u: FloatFieldIJ,
    sin_sg1: FloatFieldIJ,
    sin_sg3: FloatFieldIJ,
    rdxa: FloatFieldIJ,
    dy: FloatFieldIJ,
    crx: FloatField,
    x_area_flux: FloatField,
    ut: FloatField,
    dt: float,
):

    with computation(PARALLEL), interval(...):
        ut = (
            uc - 0.25 * cosa_u * (vc[-1, 0, 0] + vc + vc[-1, 1, 0] + vc[0, 1, 0])
        ) * rsin_u
        
        prod = dt * ut
        if prod > 0:
            crx = prod * rdxa[-1, 0]
            x_area_flux = dy * prod * sin_sg3[-1, 0]
        else:
            crx = prod * rdxa
            x_area_flux = dy * prod * sin_sg1


def fxadv_y_fluxes(
    uc: FloatField,
    vc: FloatField,
    cosa_v: FloatFieldIJ,
    rsin_v: FloatFieldIJ,
    sin_sg2: FloatFieldIJ,
    sin_sg4: FloatFieldIJ,
    rdya: FloatFieldIJ,
    dx: FloatFieldIJ,
    cry: FloatField,
    y_area_flux: FloatField,
    vt: FloatField,
    dt: float,
):
    with computation(PARALLEL), interval(...):
        vt = (
            vc - 0.25 * cosa_v * (uc[0, -1, 0] + uc[1, -1, 0] + uc + uc[1, 0, 0])
        ) * rsin_v
        prod = dt * vt
        if prod > 0:
            cry = prod * rdya[0, -1]
            y_area_flux = dx * prod * sin_sg4[0, -1]
        else:
            cry = prod * rdya
            y_area_flux = dx * prod * sin_sg2


class FiniteVolumeFluxPrep:
    """
    A large section of code near the beginning of Fortran's d_sw subroutinw
    Known in this repo as FxAdv,
    """

    def __init__(self):
        self.grid = spec.grid
        origin = self.grid.full_origin()
        domain = self.grid.domain_shape_full()
        ax_offsets = axis_offsets(self.grid, origin, domain)
        kwargs = {"externals": ax_offsets, "origin": origin, "domain": domain}

        shape = self.grid.domain_shape_full(add=(1, 1, 1))
        self._copy_in_stencil = FrozenStencil(
            copy_defn,
            origin=self.grid.full_origin(),
            domain=self.grid.domain_shape_full(add=(1, 1, 1)),
        )
        # with horizontal(region[local_is - 1 : local_ie + 3, :]):

        #self._main_ut_stencil = FrozenStencil(
        #    main_ut,
        #    origin=(self.grid.is_ - 1, self.grid.jsd, 0),
        #    domain=(self.grid.nic + 3, self.grid.njd, self.grid.npz),
        #)
        # with horizontal(region[:, local_js - 1 : local_je + 3]):
        #self._main_vt_stencil = FrozenStencil(
        #    main_vt,
        #    origin=(self.grid.isd, self.grid.js - 1, 0),
        #    domain=(self.grid.nid, self.grid.njc + 3, self.grid.npz),
        #)


        self._fxadv_x_fluxes_stencil = FrozenStencil(
            fxadv_x_fluxes,
            origin=(self.grid.is_, self.grid.jsd, 0),
            domain=(self.grid.nic + 1, self.grid.njd, self.grid.npz),
        )
        self._fxadv_y_fluxes_stencil = FrozenStencil(
            fxadv_y_fluxes,
            origin=(self.grid.isd, self.grid.js, 0),
            domain=(self.grid.nid, self.grid.njc + 1, self.grid.npz),
        )


    def __call__(
        self,
        uc,
        vc,
        crx,
        cry,
        x_area_flux,
        y_area_flux,
        ut,
        vt,
        dt,
    ):
        """
        Updates flux operators and courant numbers for fvtp2d
        To start off D_SW after the C-grid winds have been advanced half a timestep,
        and and compute finite volume transport on the D-grid (e.g.Putman and Lin 2007),
        this module prepares terms such as parts of equations 7 and 13 in Putnam and
        Lin, 2007, that get consumed by fvtp2d and ppm methods.

        Args:
            uc: x-velocity on the C-grid (in)
            vc: y-velocity on the C-grid (in)
            crx: Courant number, x direction(inout)
            cry: Courant number, y direction(inout)
            x_area_flux: flux of area in x-direction, in units of m^2 (inout)
            y_area_flux: flux of area in y-direction, in units of m^2 (inout)
            ut: temporary x-velocity transformed from C-grid to D-grid equiv(?) (inout)
            vt: temporary y-velocity transformed from C-grid to D-grid equiv(?) (inout)
            dt: acoustic timestep in seconds

        Grid variable inputs:
            cosa_u, cosa_v, rsin_u, rsin_v, sin_sg1,sin_sg2, sin_sg3, sin_sg4, dx, dy
        """

        #self._main_ut_stencil(
        #    uc,
        #    vc,
        #    self.grid.cosa_u,
        #    self.grid.rsin_u,
        #    ut,
        #)

        #self._main_vt_stencil(
        #    uc,
        #    vc,
        #    self.grid.cosa_v,
        #    self.grid.rsin_v,
        #    vt,
        #)

        self._fxadv_x_fluxes_stencil(
            uc,
            vc,
            self.grid.cosa_u,
            self.grid.rsin_u,
            self.grid.sin_sg1,
            self.grid.sin_sg3,
            self.grid.rdxa,
            self.grid.dy,
            crx,
            x_area_flux,
            ut,
            dt,
        )
        self._fxadv_y_fluxes_stencil(
            uc,
            vc,
            self.grid.cosa_v,
            self.grid.rsin_v,
            self.grid.sin_sg2,
            self.grid.sin_sg4,
            self.grid.rdya,
            self.grid.dx,
            cry,
            y_area_flux,
            vt,
            dt,
        )

# -------------------- REGIONS CORNERS-----------------

# def ut_corners(
#     cosa_u: FloatFieldIJ,
#     cosa_v: FloatFieldIJ,
#     uc: FloatField,
#     vc: FloatField,
#     ut: FloatField,
#     vt: FloatField,
# ):

#     """
#     The following code (and vt_corners) solves a 2x2 system to
#     get the interior parallel-to-edge uc,vc values near the corners
#     (ex: for the sw corner ut(2,1) and vt(1,2) are solved for simultaneously).
#     It then computes the halo uc, vc values so as to be consistent with the
#     computations on the facing panel.
#     The system solved is:
#        ut(2,1) = uc(2,1) - avg(vt)*cosa_u(2,1)
#        vt(1,2) = vc(1,2) - avg(ut)*cosa_v(1,2)
#        in which avg(vt) includes vt(1,2) and avg(ut) includes ut(2,1)

#     """
#     from __externals__ import i_end, i_start, j_end, j_start

#     with computation(PARALLEL), interval(...):
#         damp = 1.0 / (1.0 - 0.0625 * cosa_u * cosa_v[-1, 0])
#         with horizontal(region[i_start + 1, j_start - 1], region[i_start + 1, j_end]):
#             ut = (
#                 uc
#                 - 0.25
#                 * cosa_u
#                 * (
#                     vt[-1, 1, 0]
#                     + vt[0, 1, 0]
#                     + vt
#                     + vc[-1, 0, 0]
#                     - 0.25
#                     * cosa_v[-1, 0]
#                     * (ut[-1, 0, 0] + ut[-1, -1, 0] + ut[0, -1, 0])
#                 )
#             ) * damp
#         damp = 1.0 / (1.0 - 0.0625 * cosa_u * cosa_v[-1, 1])
#         with horizontal(region[i_start + 1, j_start], region[i_start + 1, j_end + 1]):
#             damp = 1.0 / (1.0 - 0.0625 * cosa_u * cosa_v[-1, 1])
#             ut = (
#                 uc
#                 - 0.25
#                 * cosa_u
#                 * (
#                     vt[-1, 0, 0]
#                     + vt
#                     + vt[0, 1, 0]
#                     + vc[-1, 1, 0]
#                     - 0.25 * cosa_v[-1, 1] *
#                     (ut[-1, 0, 0] + ut[-1, 1, 0] + ut[0, 1, 0])
#                 )
#             ) * damp
#         damp = 1.0 / (1.0 - 0.0625 * cosa_u * cosa_v)
#         with horizontal(region[i_end, j_start - 1], region[i_end, j_end]):
#             ut = (
#                 uc
#                 - 0.25
#                 * cosa_u
#                 * (
#                     vt[0, 1, 0]
#                     + vt[-1, 1, 0]
#                     + vt[-1, 0, 0]
#                     + vc
#                     - 0.25 * cosa_v * (ut[1, 0, 0] + ut[1, -1, 0] + ut[0, -1, 0])
#                 )
#             ) * damp
#         damp = 1.0 / (1.0 - 0.0625 * cosa_u * cosa_v[0, 1])
#         with horizontal(region[i_end, j_start], region[i_end, j_end + 1]):
#             ut = (
#                 uc
#                 - 0.25
#                 * cosa_u
#                 * (
#                     vt
#                     + vt[-1, 0, 0]
#                     + vt[-1, 1, 0]
#                     + vc[0, 1, 0]
#                     - 0.25 * cosa_v[0, 1] * (ut[1, 0, 0] + ut[1, 1, 0] + ut[0, 1, 0])
#                 )
#             ) * damp


# def vt_corners(
#     cosa_u: FloatFieldIJ,
#     cosa_v: FloatFieldIJ,
#     uc: FloatField,
#     vc: FloatField,
#     ut: FloatField,
#     vt: FloatField,
# ):
#     from __externals__ import i_end, i_start, j_end, j_start

#     with computation(PARALLEL), interval(...):
#         damp = 1.0 / (1.0 - 0.0625 * cosa_u[0, -1] * cosa_v)
#         with horizontal(region[i_start - 1, j_start + 1], region[i_end, j_start + 1]):
#             vt = (
#                 vc
#                 - 0.25
#                 * cosa_v
#                 * (
#                     ut[1, -1, 0]
#                     + ut[1, 0, 0]
#                     + ut
#                     + uc[0, -1, 0]
#                     - 0.25
#                     * cosa_u[0, -1]
#                     * (vt[0, -1, 0] + vt[-1, -1, 0] + vt[-1, 0, 0])
#                 )
#             ) * damp
#         damp = 1.0 / (1.0 - 0.0625 * cosa_u[1, -1] * cosa_v)
#         with horizontal(region[i_start, j_start + 1], region[i_end + 1, j_start + 1]):
#             vt = (
#                 vc
#                 - 0.25
#                 * cosa_v
#                 * (
#                     ut[0, -1, 0]
#                     + ut
#                     + ut[1, 0, 0]
#                     + uc[1, -1, 0]
#                     - 0.25 * cosa_u[1, -1] *
#                     (vt[0, -1, 0] + vt[1, -1, 0] + vt[1, 0, 0])
#                 )
#             ) * damp
#         damp = 1.0 / (1.0 - 0.0625 * cosa_u[1, 0] * cosa_v)
#         with horizontal(region[i_end + 1, j_end], region[i_start, j_end]):
#             vt = (
#                 vc
#                 - 0.25
#                 * cosa_v
#                 * (
#                     ut
#                     + ut[0, -1, 0]
#                     + ut[1, -1, 0]
#                     + uc[1, 0, 0]
#                     - 0.25 * cosa_u[1, 0] * (vt[0, 1, 0] + vt[1, 1, 0] + vt[1, 0, 0])
#                 )
#             ) * damp
#         damp = 1.0 / (1.0 - 0.0625 * cosa_u * cosa_v)
#         with horizontal(region[i_end, j_end], region[i_start - 1, j_end]):
#             vt = (
#                 vc
#                 - 0.25
#                 * cosa_v
#                 * (
#                     ut[1, 0, 0]
#                     + ut[1, -1, 0]
#                     + ut[0, -1, 0]
#                     + uc
#                     - 0.25 * cosa_u * (vt[0, 1, 0] + vt[-1, 1, 0] + vt[-1, 0, 0])
#                 )
#             ) * damp
