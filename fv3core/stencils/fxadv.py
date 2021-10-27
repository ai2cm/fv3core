from gt4py.gtscript import PARALLEL, computation, interval

import fv3core._config as spec
import fv3core.utils.gt4py_utils as utils
from fv3core.decorators import FrozenStencil, computepath_method
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


def ut_y_edge(
    uc: FloatField,
    sin_sg1: FloatFieldIJ,
    sin_sg3: FloatFieldIJ,
    ut: FloatField,
    dt: float,
):

    with computation(PARALLEL), interval(...):
        ut = (uc / sin_sg3[-1, 0]) if (uc * dt > 0) else (uc / sin_sg1)


def ut_x_edge(uc: FloatField, cosa_u: FloatFieldIJ, vt: FloatField, ut: FloatField):

    with computation(PARALLEL), interval(...):
        ut = uc - 0.25 * cosa_u * (vt[-1, 0, 0] + vt + vt[-1, 1, 0] + vt[0, 1, 0])


def vt_y_edge(vc: FloatField, cosa_v: FloatFieldIJ, ut: FloatField, vt: FloatField):
    with computation(PARALLEL), interval(...):
        vt = vc - 0.25 * cosa_v * (ut[0, -1, 0] + ut[1, -1, 0] + ut + ut[1, 0, 0])


def vt_x_edge(
    vc: FloatField,
    sin_sg2: FloatFieldIJ,
    sin_sg4: FloatFieldIJ,
    vt: FloatField,
    dt: float,
):
    with computation(PARALLEL), interval(...):
        vt = (vc / sin_sg4[0, -1]) if (vc * dt > 0) else (vc / sin_sg2)


def fxadv_x_fluxes(
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
        prod = dt * ut
        if prod > 0:
            crx = prod * rdxa[-1, 0]
            x_area_flux = dy * prod * sin_sg3[-1, 0]
        else:
            crx = prod * rdxa
            x_area_flux = dy * prod * sin_sg1


def fxadv_y_fluxes(
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
        prod = dt * vt
        if prod > 0:
            cry = prod * rdya[0, -1]
            y_area_flux = dx * prod * sin_sg4[0, -1]
        else:
            cry = prod * rdya
            y_area_flux = dx * prod * sin_sg2


def corner_ut(
    uc: FloatField,
    vc: FloatField,
    ut: FloatField,
    vt: FloatField,
    cosa_u: FloatFieldIJ,
    cosa_v: FloatFieldIJ,
):
    from __externals__ import ux, uy, vi, vj, vx, vy

    with computation(PARALLEL), interval(...):
        ut_tmp = ut
        ut = (
            (
                uc
                - 0.25
                * cosa_u
                * (
                    vt[vi, vy, 0]
                    + vt[vx, vy, 0]
                    + vt[vx, vj, 0]
                    + vc[vi, vj, 0]
                    - 0.25
                    * cosa_v[vi, vj]
                    * (ut_tmp[ux, 0, 0] + ut_tmp[ux, uy, 0] + ut_tmp[0, uy, 0])
                )
            )
            * 1.0
            / (1.0 - 0.0625 * cosa_u * cosa_v[vi, vj])
        )


def index_offset(lower, u, south=True):
    if lower == u:
        offset = 1
    else:
        offset = -1
    if south:
        offset *= -1
    return offset


def corner_ut_stencil_init(
    ui,
    uj,
    vi,
    vj,
    west,
    lower,
    south=True,
    vswitch=False,
):
    if vswitch:
        lowerfactor = 1 if lower else -1
    else:
        lowerfactor = 1
    vx = vi + index_offset(west, False, south) * lowerfactor
    ux = ui + index_offset(west, True, south) * lowerfactor
    vy = vj + index_offset(lower, False, south) * lowerfactor
    uy = uj + index_offset(lower, True, south) * lowerfactor
    corner_stencil = FrozenStencil(
        corner_ut,
        externals={
            "vi": vi - ui,
            "vj": vj - uj,
            "ux": ux - ui,
            "uy": uy - uj,
            "vx": vx - ui,
            "vy": vy - uj,
        },
        origin=(ui, uj, 0),
        domain=(1, 1, spec.grid.npz),
    )
    return corner_stencil


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
        origin_corners = self.grid.full_origin(add=(1, 1, 0))
        domain_corners = self.grid.domain_shape_full(add=(-1, -1, 0))
        corner_offsets = axis_offsets(self.grid, origin_corners, domain_corners)
        kwargs_corners = {
            "externals": corner_offsets,
            "origin": origin_corners,
            "domain": domain_corners,
        }
        shape = self.grid.domain_shape_full(add=(1, 1, 1))
        self._utmp = utils.make_storage_from_shape(shape)
        self._vtmp = utils.make_storage_from_shape(shape)
        self._copy_in_stencil = FrozenStencil(
            copy_defn,
            origin=self.grid.full_origin(),
            domain=self.grid.domain_shape_full(add=(1, 1, 1)),
        )
        # with horizontal(region[local_is - 1 : local_ie + 3, :]):
        js = self.grid.js + 1 if self.grid.south_edge else self.grid.jsd
        je = self.grid.je if self.grid.north_edge else self.grid.jed
        self._main_ut_stencil = FrozenStencil(
            main_ut,
            origin=(self.grid.is_ - 1, self.grid.jsd, 0),
            domain=(self.grid.nic + 3, self.grid.njd, self.grid.npz),
        )
        # region[:, j_start - 1 : j_start + 1], region[:, j_end : j_end + 2]

        self._copy_ut_south = FrozenStencil(
            copy_defn,
            origin=(origin_corners[0], self.grid.js - 1, 0),
            domain=(domain_corners[0], 2, domain_corners[2]),
        )
        self._copy_ut_north = FrozenStencil(
            copy_defn,
            origin=(origin_corners[0], self.grid.je, 0),
            domain=(domain_corners[0], 2, domain_corners[2]),
        )
        # with horizontal(region[:, local_js - 1 : local_je + 3]):
        self._main_vt_stencil = FrozenStencil(
            main_vt,
            origin=(self.grid.isd, self.grid.js - 1, 0),
            domain=(self.grid.nid, self.grid.njc + 3, self.grid.npz),
        )
        self._copy_vt_south = FrozenStencil(
            copy_defn,
            origin=(origin_corners[0], self.grid.js, 0),
            domain=(domain_corners[0], 1, domain_corners[2]),
        )
        self._copy_vt_north = FrozenStencil(
            copy_defn,
            origin=(origin_corners[0], self.grid.je + 1, 0),
            domain=(domain_corners[0], 1, domain_corners[2]),
        )
        #   with horizontal(region[i_start, :], region[i_end + 1, :]):
        if self.grid.west_edge:
            self._ut_west_stencil = FrozenStencil(
                ut_y_edge,
                origin=(self.grid.is_, self.grid.jsd, 0),
                domain=(1, shape[1], shape[2]),
            )
            self._vt_west_stencil = FrozenStencil(
                vt_y_edge,
                origin=(self.grid.is_ - 1, self.grid.js, 0),
                domain=(2, self.grid.njc + 1, shape[2]),
            )
        if self.grid.east_edge:
            self._ut_east_stencil = FrozenStencil(
                ut_y_edge,
                origin=(self.grid.ie + 1, self.grid.jsd, 0),
                domain=(1, shape[1], shape[2]),
            )
            self._vt_east_stencil = FrozenStencil(
                vt_y_edge,
                origin=(self.grid.ie, self.grid.js, 0),
                domain=(2, self.grid.njc + 1, shape[2]),
            )
        i1 = self.grid.is_ + 2 if self.grid.west_edge else self.grid.is_
        i2 = self.grid.ie - 1 if self.grid.east_edge else self.grid.ie + 1
        if self.grid.south_edge:
            self._vt_south_stencil = FrozenStencil(
                vt_x_edge,
                origin=(self.grid.isd, self.grid.js, 0),
                domain=(shape[0], 1, shape[2]),
            )
            self._ut_south_stencil = FrozenStencil(
                ut_x_edge,
                origin=(i1, self.grid.js - 1, 0),
                domain=(i2 - i1 + 1, 2, shape[2]),
            )
        if self.grid.north_edge:
            self._vt_north_stencil = FrozenStencil(
                vt_x_edge,
                origin=(self.grid.isd, self.grid.je + 1, 0),
                domain=(shape[0], 1, shape[2]),
            )
            self._ut_north_stencil = FrozenStencil(
                ut_x_edge,
                origin=(i1, self.grid.je, 0),
                domain=(i2 - i1 + 1, 2, shape[2]),
            )

        self._sw_corners()
        self._se_corners()
        self._ne_corners()
        self._nw_corners()

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

    def _sw_corners(self):
        t = self.grid.is_ + 1
        n = self.grid.is_
        z = self.grid.is_ - 1
        self._sw_corner_ut_stencil1 = corner_ut_stencil_init(
            t, z, n, z, west=True, lower=True
        )
        self._sw_corner_vt_stencil1 = corner_ut_stencil_init(
            z, t, z, n, west=True, lower=True, vswitch=True
        )
        self._sw_corner_ut_stencil2 = corner_ut_stencil_init(
            t, n, n, t, west=True, lower=False
        )
        self._sw_corner_vt_stencil2 = corner_ut_stencil_init(
            n, t, t, n, west=True, lower=False, vswitch=True
        )

    def _se_corners(self):
        t = self.grid.js + 1
        n = self.grid.js
        z = self.grid.js - 1
        self._se_corner_ut_stencil1 = corner_ut_stencil_init(
            self.grid.ie,
            z,
            self.grid.ie,
            z,
            west=False,
            lower=True,
        )
        self._se_corner_vt_stencil1 = corner_ut_stencil_init(
            self.grid.ie + 1,
            t,
            self.grid.ie + 2,
            n,
            west=False,
            lower=True,
            vswitch=True,
        )
        self._se_corner_ut_stencil2 = corner_ut_stencil_init(
            self.grid.ie,
            n,
            self.grid.ie,
            t,
            west=False,
            lower=False,
        )
        self._se_corner_vt_stencil2 = corner_ut_stencil_init(
            self.grid.ie,
            t,
            self.grid.ie,
            n,
            west=False,
            lower=False,
            vswitch=True,
        )

    def _ne_corners(self):
        self._ne_corner_ut_stencil1 = corner_ut_stencil_init(
            self.grid.ie,
            self.grid.je + 1,
            self.grid.ie,
            self.grid.je + 2,
            west=False,
            lower=False,
        )
        self._ne_corner_vt_stencil1 = corner_ut_stencil_init(
            self.grid.ie + 1,
            self.grid.je,
            self.grid.ie + 2,
            self.grid.je,
            west=False,
            lower=False,
            south=False,
            vswitch=True,
        )
        self._ne_corner_ut_stencil2 = corner_ut_stencil_init(
            self.grid.ie,
            self.grid.je,
            self.grid.ie,
            self.grid.je,
            west=False,
            lower=True,
        )
        self._ne_corner_vt_stencil2 = corner_ut_stencil_init(
            self.grid.ie,
            self.grid.je,
            self.grid.ie,
            self.grid.je,
            west=False,
            lower=True,
            south=False,
            vswitch=True,
        )

    def _nw_corners(self):
        t = self.grid.js + 1
        n = self.grid.js
        z = self.grid.js - 1
        self._nw_corner_ut_stencil1 = corner_ut_stencil_init(
            t,
            self.grid.je + 1,
            n,
            self.grid.je + 2,
            west=True,
            lower=False,
        )
        self._nw_corner_vt_stencil1 = corner_ut_stencil_init(
            z,
            self.grid.je,
            z,
            self.grid.je,
            west=True,
            lower=False,
            south=False,
            vswitch=True,
        )
        self._nw_corner_ut_stencil2 = corner_ut_stencil_init(
            t,
            self.grid.je,
            n,
            self.grid.je,
            west=True,
            lower=True,
        )
        self._nw_corner_vt_stencil2 = corner_ut_stencil_init(
            n,
            self.grid.je,
            t,
            self.grid.je,
            west=True,
            lower=True,
            south=False,
            vswitch=True,
        )

    @computepath_method
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
        self._copy_in_stencil(ut, self._utmp)
        self._copy_in_stencil(vt, self._vtmp)
        self._main_ut_stencil(
            uc,
            vc,
            self.grid.cosa_u,
            self.grid.rsin_u,
            ut,
        )
        if self.grid.south_edge:
            self._copy_ut_south(self._utmp, ut)
        if self.grid.north_edge:
            self._copy_ut_north(self._utmp, ut)
        if self.grid.west_edge:
            self._ut_west_stencil(
                uc,
                self.grid.sin_sg1,
                self.grid.sin_sg3,
                ut,
                dt,
            )
        if self.grid.east_edge:
            self._ut_east_stencil(
                uc,
                self.grid.sin_sg1,
                self.grid.sin_sg3,
                ut,
                dt,
            )
        self._main_vt_stencil(
            uc,
            vc,
            self.grid.cosa_v,
            self.grid.rsin_v,
            vt,
        )
        if self.grid.south_edge:
            self._copy_vt_south(self._vtmp, vt)
        if self.grid.north_edge:
            self._copy_vt_north(self._vtmp, vt)
        if self.grid.west_edge:
            self._vt_west_stencil(
                vc,
                self.grid.cosa_v,
                ut,
                vt,
            )
        if self.grid.east_edge:
            self._vt_east_stencil(
                vc,
                self.grid.cosa_v,
                ut,
                vt,
            )
        if self.grid.south_edge:
            self._vt_south_stencil(
                vc,
                self.grid.sin_sg2,
                self.grid.sin_sg4,
                vt,
                dt,
            )
        if self.grid.north_edge:
            self._vt_north_stencil(
                vc,
                self.grid.sin_sg2,
                self.grid.sin_sg4,
                vt,
                dt,
            )

        if self.grid.south_edge:
            self._ut_south_stencil(
                uc,
                self.grid.cosa_u,
                vt,
                ut,
            )
        if self.grid.north_edge:
            self._ut_north_stencil(
                uc,
                self.grid.cosa_u,
                vt,
                ut,
            )
        if self.grid.sw_corner:
            self._sw_corner_ut_stencil1(
                uc,
                vc,
                ut,
                vt,
                self.grid.cosa_u,
                self.grid.cosa_v,
            )
            self._sw_corner_vt_stencil1(
                vc,
                uc,
                vt,
                ut,
                self.grid.cosa_v,
                self.grid.cosa_u,
            )
            self._sw_corner_ut_stencil2(
                uc,
                vc,
                ut,
                vt,
                self.grid.cosa_u,
                self.grid.cosa_v,
            )
            self._sw_corner_vt_stencil2(
                vc,
                uc,
                vt,
                ut,
                self.grid.cosa_v,
                self.grid.cosa_u,
            )
        if self.grid.se_corner:
            self._se_corner_ut_stencil1(
                uc,
                vc,
                ut,
                vt,
                self.grid.cosa_u,
                self.grid.cosa_v,
            )
            self._se_corner_vt_stencil1(
                vc,
                uc,
                vt,
                ut,
                self.grid.cosa_v,
                self.grid.cosa_u,
            )
            self._se_corner_ut_stencil2(
                uc,
                vc,
                ut,
                vt,
                self.grid.cosa_u,
                self.grid.cosa_v,
            )
            self._se_corner_vt_stencil2(
                vc,
                uc,
                vt,
                ut,
                self.grid.cosa_v,
                self.grid.cosa_u,
            )
        if self.grid.ne_corner:
            self._ne_corner_ut_stencil1(
                uc,
                vc,
                ut,
                vt,
                self.grid.cosa_u,
                self.grid.cosa_v,
            )
            self._ne_corner_vt_stencil1(
                vc,
                uc,
                vt,
                ut,
                self.grid.cosa_v,
                self.grid.cosa_u,
            )
            self._ne_corner_ut_stencil2(
                uc,
                vc,
                ut,
                vt,
                self.grid.cosa_u,
                self.grid.cosa_v,
            )
            self._ne_corner_vt_stencil2(
                vc,
                uc,
                vt,
                ut,
                self.grid.cosa_v,
                self.grid.cosa_u,
            )
        if self.grid.nw_corner:
            self._nw_corner_ut_stencil1(
                uc,
                vc,
                ut,
                vt,
                self.grid.cosa_u,
                self.grid.cosa_v,
            )
            self._nw_corner_vt_stencil1(
                vc,
                uc,
                vt,
                ut,
                self.grid.cosa_v,
                self.grid.cosa_u,
            )
            self._nw_corner_ut_stencil2(
                uc,
                vc,
                ut,
                vt,
                self.grid.cosa_u,
                self.grid.cosa_v,
            )
            self._nw_corner_vt_stencil2(
                vc,
                uc,
                vt,
                ut,
                self.grid.cosa_v,
                self.grid.cosa_u,
            )

        self._fxadv_x_fluxes_stencil(
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
