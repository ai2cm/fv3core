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
        #fy2 = yppm.y_flux(q, cry)
        
        bl = (yppm.p1 * (q[0, -1, 0] + q) + yppm.p2 * (q[0, -2, 0] + q[0, 1, 0])) - q
        bll = (yppm.p1 * (q[0, -2, 0] + q[0, -1, 0]) + yppm.p2 * (q[0, -3, 0] + q[0, 0, 0])) - q[0, -1, 0]
        blu = (yppm.p1 * (q[0, 0, 0] + q[0, 1, 0]) + yppm.p2 * (q[0, -1, 0] + q[0, 2, 0])) - q[0, 1, 0]
        br = (yppm.p1 * (q + q[0, 1, 0]) + yppm.p2 * (q[0, -1, 0] + q[0, 2, 0])) - q
        brl = (yppm.p1 * (q[0, -1, 0] + q[0, 0, 0]) + yppm.p2 * (q[0, -2, 0] + q[0, 1, 0])) - q[0, -1, 0]
        bru = (yppm.p1 * (q[0, 1, 0] + q[0, 2, 0]) + yppm.p2 * (q[0, 0, 0] + q[0, 3, 0])) - q[0, 1, 0]
        b0 = bl + br
        b0l = bll + brl
        b0u = blu + bru
        smt5 = (3.0 * abs(b0)) < abs(bl - br)
        smt5l =  (3.0 * abs(b0l)) < abs(bll - brl)
        smt5u = (3.0 * abs(b0u)) < abs(blu - bru)
        if cry > 0.0:
            if smt5l or smt5:
                fy2 = q[0, -1, 0] + ((1.0 - cry) * (brl - cry * b0l))
            else:
                fy2 = q[0, -1, 0]
        else:
            if smt5l or smt5:
                fy2 = q + ((1.0 + cry) * (bl + cry * b0))
            else:
                fy2 = q
        if cry[0, 1, 0] > 0.0:
            if smt5 or smt5u:
                fy2u = q[0, 0, 0] + ((1.0 - cry[0, 1, 0]) * (br[0, 0, 0] - cry[0, 1, 0] * b0[0, 0, 0]))
            else:
                fy2u = q[0, 0, 0]
        else:
            if smt5 or smt5u:
                fy2u = q[0, 1, 0] + ((1.0 + cry[0, 1, 0]) * (blu + cry[0, 1, 0] * b0u))
            else:
                fy2u = q[0, 1, 0]

        #q_i = compute_q_i(q, area, y_area_flux, fy2)                                                                                                                                                 

        q_i = (q * area + (y_area_flux * fy2) - (y_area_flux[0, 1, 0] * fy2u)) / (area + y_area_flux - y_area_flux[0, 1, 0])

        #fxt = xppm.x_flux(q_i, crx)

        bl3 = (yppm.p1 * (q_i[-1, 0, 0] + q_i) + yppm.p2 * (q_i[-2, 0, 0] + q_i[1, 0, 0])) - q_i
        bll3 = (yppm.p1 * (q_i[-2, 0, 0] + q_i[-1, 0, 0]) + yppm.p2 * (q_i[-3, 0, 0] + q_i[0, 0, 0])) - q_i[-1, 0, 0]
        blr3 = (yppm.p1 * (q_i[0, 0, 0] + q_i[1, 0, 0]) + yppm.p2 * (q_i[-1, 0, 0] + q_i[2, 0, 0])) - q_i[1, 0, 0]
        br3 = (yppm.p1 * (q_i + q_i[1, 0, 0]) + yppm.p2 * (q_i[-1, 0, 0] + q_i[2, 0, 0])) - q_i
        brl3 = (yppm.p1 * (q_i[-1, 0, 0] + q_i[0, 0, 0]) + yppm.p2 * (q_i[-2, 0, 0] + q_i[1, 0, 0])) - q_i[-1, 0, 0]
        brr3 = (yppm.p1 * (q_i[1, 0, 0] + q_i[2, 0, 0]) + yppm.p2 * (q_i[0, 0, 0] + q_i[3, 0, 0])) - q_i[1, 0, 0]
        b03 = bl3 + br3
        b0l3 = bll3 + brl3
        b0r3 = blr3 + brr3
        smt53 = (3.0 * abs(b03)) < abs(bl3 - br3)
        smt5l3 = (3.0 * abs(b0l3)) < abs(bll3 - brl3)
        smt5r3 = (3.0 * abs(b0r3)) < abs(blr3 - brr3)
        if crx > 0.0:
            if smt5l3 or smt53:
                fxt =  q_i[-1, 0, 0] + ((1.0 - crx) * (brl3 - crx * b0l3))
            else:
                fxt =  q_i[-1, 0, 0]
        else:
            if smt5l3 or smt53:
                fxt = q_i + ((1.0 + crx) * (bl3 + crx * b03))
            else:
                fxt = q_i


        #fx2 = xppm.x_flux(q, crx)
        bl2 = (yppm.p1 * (q[-1, 0, 0] + q) + yppm.p2 * (q[-2, 0, 0] + q[1, 0, 0])) - q
        bll2 = (yppm.p1 * (q[-2, 0, 0] + q[-1, 0, 0]) + yppm.p2 * (q[-3, 0, 0] + q[0, 0, 0])) - q[-1, 0, 0]
        blr2 = (yppm.p1 * (q_i[0, 0, 0] + q[1, 0, 0]) + yppm.p2 * (q[-1, 0, 0] + q[2, 0, 0])) - q[1, 0, 0]
        br2 = (yppm.p1 * (q + q[1, 0, 0]) + yppm.p2 * (q[-1, 0, 0] + q[2, 0, 0])) - q
        brl2 = (yppm.p1 * (q[-1, 0, 0] + q[0, 0, 0]) + yppm.p2 * (q[-2, 0, 0] + q[1, 0, 0])) - q[-1, 0, 0]
        brr2 = (yppm.p1 * (q[1, 0, 0] + q[2, 0, 0]) + yppm.p2 * (q[0, 0, 0] + q[3, 0, 0])) - q[1, 0, 0]

        b02 = bl2 + br2
        b0l2 = bll2 + brl2
        b0r2 = blr2 + brr2
        smt52 = (3.0 * abs(b02)) < abs(bl2 - br2)
        smt5l2 = (3.0 * abs(b0l2)) < abs(bll2 - brl2)
        smt5r2 = (3.0 * abs(b0r2)) < abs(blr2 - brr2)
        if crx > 0.0:
            if smt5l2 or smt52:
                fx2 =  q[-1, 0, 0] + ((1.0 - crx) * (brl2 - crx * b0l2))
            else:
                fx2 =  q[-1, 0, 0]
        else:
            if smt5l2 or smt52:
                fx2 = q + ((1.0 + crx) * (bl2 + crx * b02))
            else:
                fx2 = q
        
        #if crx[1, 0, 0] > 0.0:
        #    if smt52 or smt5r2:
        #        fx2r =  q[0, 0, 0] + ((1.0 - crx[1, 0, 0]) * (br2 - crx[1, 0, 0] * b02))
        #    else:
        #        fx2r =  q[0, 0, 0]
        #else:
        #    if smt52 or smt5r2:
        #        fx2r = q[1, 0, 0] + ((1.0 + crx[1, 0, 0]) * (blr2 + crx[1, 0, 0] * b0r2))
        #    else:
        #        fx2r = q[1, 0, 0]
        

        q_j =  compute_q_j(q, area, x_area_flux, fx2)
 
        #q_j = (q * area + (x_area_flux * fx2) - (x_area_flux[1, 0, 0] * fx2r)) / (area + x_area_flux - x_area_flux[1, 0, 0])
        #q_jl1 = (q * area + (x_area_flux * fx2) - (x_area_flux[1, 0, 0] * fx2r)) / (area + x_area_flux - x_area_flux[1, 0, 0])
        #q_jl2 = (q * area + (x_area_flux * fx2) - (x_area_flux[1, 0, 0] * fx2r)) / (area + x_area_flux - x_area_flux[1, 0, 0])
        #q_ju1
        #fyt = yppm.y_flux(q_j, cry)

        bl4 = (yppm.p1 * (q_j[0, -1, 0] + q_j) + yppm.p2 * (q_j[0, -2, 0] + q_j[0, 1, 0])) - q_j
        bll4 = (yppm.p1 * (q_j[0, -2, 0] + q_j[0, -1, 0]) + yppm.p2 * (q_j[0, -3, 0] + q_j[0, 0, 0])) - q_j[0, -1, 0]
        blu4 = (yppm.p1 * (q_j[0, 0, 0] + q_j[0, 1, 0]) + yppm.p2 * (q_j[0, -1, 0] + q_j[0, 2, 0])) - q_j[0, 1, 0]
        br4 = (yppm.p1 * (q_j + q_j[0, 1, 0]) + yppm.p2 * (q_j[0, -1, 0] + q_j[0, 2, 0])) - q_j
        brl4 = (yppm.p1 * (q_j[0, -1, 0] + q_j[0, 0, 0]) + yppm.p2 * (q_j[0, -2, 0] + q_j[0, 1, 0])) - q_j[0, -1, 0]
        bru4 = (yppm.p1 * (q_j[0, 1, 0] + q_j[0, 2, 0]) + yppm.p2 * (q_j[0, 0, 0] + q_j[0, 3, 0])) - q_j[0, 1, 0]

        b04 = bl4 + br4
        b0l4 = bll4 + brl4
        b0u4 = blu4 + bru4
        smt54 = (3.0 * abs(b04)) < abs(bl4 - br4)
        smt5l4 =  (3.0 * abs(b0l4)) < abs(bll4 - brl4)
        smt5u4 = (3.0 * abs(b0u4)) < abs(blu4 - bru4)
        if cry > 0.0:
            if smt5l4 or smt54:
                fyt = q_j[0, -1, 0] + ((1.0 - cry) * (brl4 - cry * b0l4))
            else:
                fyt = q_j[0, -1, 0]
        else:
            if smt5l4 or smt54:
                fyt = q_j + ((1.0 + cry) * (bl4 + cry * b04))
            else:
                fyt = q_j

        
        fx =  0.5 * (fxt + fx2) * mfx 
        fy =  0.5 * (fyt + fy2) * mfy 
       
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
