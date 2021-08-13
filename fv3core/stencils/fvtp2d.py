import gt4py.gtscript as gtscript
from gt4py.gtscript import PARALLEL, computation, interval

import fv3core._config as spec
import fv3core.utils.gt4py_utils as utils
from fv3core.decorators import FrozenStencil
from fv3core.stencils.delnflux import DelnFlux
import fv3core.stencils.xppm as xppm
import fv3core.stencils.yppm as yppm
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

@gtscript.function
def transport_flux(f: FloatField, f2: FloatField, mf: FloatField):
    return 0.5 * (f + f2) * mf

@gtscript.function
def compute_q_i(q, area, y_area_flux, fy2):
    fyy = y_area_flux * fy2
    area_with_y_flux = apply_y_flux_divergence(area, y_area_flux)
    return (q * area + fyy - fyy[0, 1, 0]) / area_with_y_flux

@gtscript.function
def compute_q_j(q, area, x_area_flux, fx2):
    fx1 = x_area_flux * fx2
    area_with_x_flux = apply_x_flux_divergence(area, x_area_flux)
    return (q * area + fx1 - fx1[1, 0, 0]) / area_with_x_flux

def combined(q: FloatField, crx: FloatField, cry: FloatField, x_area_flux: FloatField,y_area_flux: FloatField, mfx: FloatField, mfy: FloatField, fx:FloatField, fy:FloatField, area: FloatFieldIJ):
    from __externals__ import mord
    with computation(PARALLEL), interval(...):

        # self.y_piecewise_parabolic_inner(q, cry, self._tmp_fy2)
        fy2 = yppm.y_flux(q, cry)
        # q_i_stencil
        #q_i = compute_q_i(q, area, y_area_flux, fy2)
        fyy = y_area_flux * fy2
        # apply y flux divergence 
        area_with_y_flux = area + y_area_flux - y_area_flux[0, 1, 0]
        q_i = (q * area + fyy - fyy[0, 1, 0]) / area_with_y_flux
        #self.x_piecewise_parabolic_outer(self._tmp_q_i, crx,fx)
        fxt = xppm.x_flux(q_i, crx)
        #self.x_piecewise_parabolic_inner(q, crx,  self._tmp_fx2)
        fx2 = xppm.x_flux(q, crx)
        # q_j
        # q_j =  compute_q_j(q, area, x_area_flux, fx2)
        fx1 = x_area_flux * fx2
        # apply x flux divergence
        area_with_x_flux =  area + x_area_flux - x_area_flux[1, 0, 0]
        q_j = (q * area + fx1 - fx1[1, 0, 0]) / area_with_x_flux
        # self.y_piecewise_parabolic_outer(self._tmp_q_j, cry,  fy)
        fyt = yppm.y_flux(q_j, cry)
        fx =  transport_flux(fxt, fx2, mfx)
        fy =  transport_flux(fyt, fy2, mfy)
        
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

        self._nord = nord
        self._damp_c = damp_c
        ord_outer = hord
        # TODO, not accounted for:
        ord_inner = 8 if hord == 10 else hord
        if (self._nord is not None) and (self._damp_c is not None):
            self.delnflux = DelnFlux(self._nord, self._damp_c)

        self.stencil_combined = FrozenStencil(
            combined,
            externals={
                "mord": abs(hord), # TODO if hord == 10, inner vs outer
            },
            origin=self.grid.compute_origin(),
            domain=self.grid.domain_shape_compute(add=(1,1,1)),
        )

    def __call__(
        self,
        q,
        crx,
        cry,
        x_area_flux,
        y_area_flux,
        fx,
        fy,
        mass=None,
        mfx=None,
        mfy=None,
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

        if mfx is None:
            mfx = x_area_flux
            use_mass = False
        else:
            use_mass = True
        if mfy is None:
            mfy = y_area_flux
        self.stencil_combined(q, crx, cry,x_area_flux,y_area_flux,  mfx, mfy,fx, fy, self.grid.area)
      
        if (self._nord is not None) and (self._damp_c is not None):
            if (use_mass and mass is not None):
                self.delnflux(q, fx, fy, mass=mass)
            elif not use_mass:
                self.delnflux(q, fx, fy)
