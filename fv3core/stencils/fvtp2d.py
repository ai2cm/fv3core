import gt4py.gtscript as gtscript
from gt4py.gtscript import PARALLEL, computation, interval

import fv3core._config as spec
import fv3core.utils.corners as corners
import fv3core.utils.gt4py_utils as utils
from fv3core.decorators import FrozenStencil, computepath_method
from fv3core.stencils.delnflux import DelnFlux
from fv3core.stencils.xppm import XPiecewiseParabolic
from fv3core.stencils.yppm import YPiecewiseParabolic
from fv3core.utils.typing import FloatField, FloatFieldIJ


@gtscript.function
def apply_x_flux_divergence(q: FloatField, q_x_flux: FloatField) -> FloatField:
    """
    Update a scalar q according to its flux in the x direction.
    """
    return q + q_x_flux - q_x_flux[1, 0, 0]


@gtscript.function
def apply_y_flux_divergence(q: FloatField, q_y_flux: FloatField) -> FloatField:
    """
    Update a scalar q according to its flux in the x direction.
    """
    return q + q_y_flux - q_y_flux[0, 1, 0]


def q_i_stencil(
    q: FloatField,
    area: FloatFieldIJ,
    y_area_flux: FloatField,
    fy2: FloatField,
    q_i: FloatField,
):
    with computation(PARALLEL), interval(...):
        fyy = y_area_flux * fy2
        area_with_y_flux = apply_y_flux_divergence(area, y_area_flux)
        q_i = (q * area + fyy - fyy[0, 1, 0]) / area_with_y_flux


def q_j_stencil(
    q: FloatField,
    area: FloatFieldIJ,
    x_area_flux: FloatField,
    fx2: FloatField,
    q_j: FloatField,
):
    with computation(PARALLEL), interval(...):
        fx1 = x_area_flux * fx2
        area_with_x_flux = apply_x_flux_divergence(area, x_area_flux)
        q_j = (q * area + fx1 - fx1[1, 0, 0]) / area_with_x_flux


def transport_flux(f: FloatField, f2: FloatField, mf: FloatField):
    with computation(PARALLEL), interval(...):
        ftmp = 0.5 * (f + f2) * mf
        f = ftmp


class FiniteVolumeTransport:
    """
    Equivalent of Fortran FV3 subroutine fv_tp_2d, done in 3 dimensions.
    Tested on serialized data with FvTp2d
    ONLY USE_SG=False compiler flag implements
    """

    def __init__(self, namelist, hord, nord=None, damp_c=None):
        self.grid = spec.grid
        shape = self.grid.domain_shape_full(add=(1, 1, 1))
        origin = self.grid.compute_origin()
        self._tmp_q_i = utils.make_storage_from_shape(shape, origin)
        self._tmp_q_j = utils.make_storage_from_shape(shape, origin)
        self._tmp_fx2 = utils.make_storage_from_shape(shape, origin)
        self._tmp_fy2 = utils.make_storage_from_shape(shape, origin)
        self._corner_tmp = utils.make_storage_from_shape(
            self.grid.domain_shape_full(add=(1, 1, 1)), origin=self.grid.full_origin()
        )
        """Temporary field to use for corner computation in both x and y direction"""
        self._nord = nord
        self._damp_c = damp_c
        ord_outer = hord
        ord_inner = 8 if hord == 10 else hord
        self.stencil_q_i = FrozenStencil(
            q_i_stencil,
            origin=self.grid.full_origin(add=(0, 3, 0)),
            domain=self.grid.domain_shape_full(add=(0, -3, 1)),
        )
        self.stencil_q_j = FrozenStencil(
            q_j_stencil,
            origin=self.grid.full_origin(add=(3, 0, 0)),
            domain=self.grid.domain_shape_full(add=(-3, 0, 1)),
        )
        self.stencil_transport_flux_x = FrozenStencil(
            transport_flux,
            origin=self.grid.compute_origin(),
            domain=self.grid.domain_shape_compute(add=(1, 0, 1)),
        )
        self.stencil_transport_flux_y = FrozenStencil(
            transport_flux,
            origin=self.grid.compute_origin(),
            domain=self.grid.domain_shape_compute(add=(0, 1, 1)),
        )
        if (self._nord is not None) and (self._damp_c is not None):
            self.delnflux = DelnFlux(self._nord, self._damp_c)

        self.x_piecewise_parabolic_inner = XPiecewiseParabolic(
            namelist, ord_inner, self.grid.jsd, self.grid.jed
        )
        self.y_piecewise_parabolic_inner = YPiecewiseParabolic(
            namelist, ord_inner, self.grid.isd, self.grid.ied
        )
        self.x_piecewise_parabolic_outer = XPiecewiseParabolic(
            namelist, ord_outer, self.grid.js, self.grid.je
        )
        self.y_piecewise_parabolic_outer = YPiecewiseParabolic(
            namelist, ord_outer, self.grid.is_, self.grid.ie
        )

        self._copy_corners_x: corners.CopyCorners = corners.CopyCorners(
            "x", self._corner_tmp
        )
        """Stencil responsible for doing corners updates in x-direction."""

        self._copy_corners_y: corners.CopyCorners = corners.CopyCorners(
            "y", self._corner_tmp
        )
        """Stencil responsible for doing corners updates in y-direction."""
        self.damp_c_is_none = self._damp_c is None
        self.nord_is_none = self._nord is None
        import numpy as np
        self.goelse = np.empty((1,), dtype=bool)

    @computepath_method
    def __call__(
        self,
        q,
        crx,
        cry,
        x_area_flux,
        y_area_flux,
        fx,
        fy,
        mass,
        mfx,
        mfy,
    ):
        """
        Calculate fluxes for horizontal finite volume transport.

        Args:
            q: scalar to be transported (in)
            crx: Courant number in x-direction
            cry: Courant number in y-direction
            x_area_flux: flux of area in x-direction, in units of m^2 (in)
            y_area_flux: flux of area in y-direction, in units of m^2 (in)
            fx: transport flux of q in x-direction (out)
            fy: transport flux of q in y-direction (out)
            mass: ???
            mfx: ???
            mfy: ???
        """

        self._copy_corners_y.__call__(q)

        self.y_piecewise_parabolic_inner.__call__(q, cry, self._tmp_fy2)
        self.stencil_q_i(
            q,
            self.grid.area,
            y_area_flux,
            self._tmp_fy2,
            self._tmp_q_i,
        )
        self.x_piecewise_parabolic_outer.__call__(self._tmp_q_i, crx, fx)

        self._copy_corners_x.__call__(q)

        self.x_piecewise_parabolic_inner.__call__(q, crx, self._tmp_fx2)
        self.stencil_q_j(
            q,
            self.grid.area,
            x_area_flux,
            self._tmp_fx2,
            self._tmp_q_j,
        )
        self.y_piecewise_parabolic_outer.__call__(self._tmp_q_j, cry, fy)

        self.goelse[0] = True
        if mfx is not None:
            if mfy is not None:
                self.goelse[0] = False
                self.stencil_transport_flux_x(
                    fx,
                    self._tmp_fx2,
                    mfx,
                )
                self.stencil_transport_flux_y(
                    fy,
                    self._tmp_fy2,
                    mfy,
                )
                if mass is not None:
                    if (not self.nord_is_none) and (not self.damp_c_is_none):
                        self.delnflux.__call__(q, fx, fy, None, mass)
        if self.goelse[0]:
            self.stencil_transport_flux_x(
                fx,
                self._tmp_fx2,
                x_area_flux,
            )
            self.stencil_transport_flux_y(
                fy,
                self._tmp_fy2,
                y_area_flux,
            )
            if (not self.nord_is_none) and (not self.damp_c_is_none):
                self.delnflux.__call__(q, fx, fy, None, None)
