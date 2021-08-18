from typing import Optional

import gt4py.gtscript as gtscript
import numpy as np
from gt4py.gtscript import FORWARD, PARALLEL, computation, interval

import fv3core._config as spec
import fv3core.utils.gt4py_utils as utils
from fv3core.decorators import FrozenStencil
from fv3core.stencils.basic_operations import copy_defn
from fv3core.utils.typing import FloatField, FloatFieldIJ, FloatFieldK


def calc_damp(damp4: FloatField, nord: FloatFieldK, damp_c: FloatFieldK, da_min: float):
    with computation(FORWARD), interval(...):
        damp4 = (damp_c * da_min) ** (nord + 1)


def fx_calc_stencil(q: FloatField, del6_v: FloatFieldIJ, fx: FloatField):

    with computation(PARALLEL), interval(...):
        fx = fx_calculation(q, del6_v)


def fy_calc_stencil(q: FloatField, del6_u: FloatFieldIJ, fy: FloatField):
    with computation(PARALLEL), interval(...):
        fy = fy_calculation(q, del6_u)


def fx_calc_stencil_column(
    q: FloatField, del6_v: FloatFieldIJ, fx: FloatField, nord: FloatFieldK
):
    with computation(PARALLEL), interval(...):
        if nord > 0:
            fx = fx_calculation_neg(q, del6_v)


def fy_calc_stencil_column(
    q: FloatField, del6_u: FloatFieldIJ, fy: FloatField, nord: FloatFieldK
):
    with computation(PARALLEL), interval(...):
        if nord > 0:
            fy = fy_calculation_neg(q, del6_u)


@gtscript.function
def fx_calculation(q: FloatField, del6_v: FloatField):
    return del6_v * (q[-1, 0, 0] - q)


@gtscript.function
def fx_calculation_neg(q: FloatField, del6_v: FloatField):
    return -del6_v * (q[-1, 0, 0] - q)


@gtscript.function
def fy_calculation(q: FloatField, del6_u: FloatField):
    return del6_u * (q[0, -1, 0] - q)


@gtscript.function
def fy_calculation_neg(q: FloatField, del6_u: FloatField):
    return -del6_u * (q[0, -1, 0] - q)


def d2_highorder_stencil(
    fx: FloatField,
    fy: FloatField,
    rarea: FloatFieldIJ,
    d2: FloatField,
    nord: FloatFieldK,
):
    with computation(PARALLEL), interval(...):
        if nord > 0:
            d2 = d2_highorder(fx, fy, rarea)


@gtscript.function
def d2_highorder(fx: FloatField, fy: FloatField, rarea: FloatField):
    d2 = (fx - fx[1, 0, 0] + fy - fy[0, 1, 0]) * rarea
    return d2


def d2_damp_interval(q: FloatField, d2: FloatField, damp: FloatFieldK):
    with computation(PARALLEL), interval(...):
        d2 = damp * q


def add_diffusive_component(
    fx: FloatField, fx2: FloatField, fy: FloatField, fy2: FloatField
):
    with computation(PARALLEL), interval(...):
        fx = fx + fx2
        fy = fy + fy2


def diffusive_damp(
    fx: FloatField,
    fx2: FloatField,
    fy: FloatField,
    fy2: FloatField,
    mass: FloatField,
    damp: FloatFieldK,
):
    with computation(PARALLEL), interval(...):
        fx = fx + 0.5 * damp * (mass[-1, 0, 0] + mass) * fx2
        fy = fy + 0.5 * damp * (mass[0, -1, 0] + mass) * fy2


def delnflux_combined(
    q: FloatField,
    del6_u: FloatFieldIJ,
    del6_v: FloatFieldIJ,
    rarea: FloatFieldIJ,
    d2: FloatField,
    fx: FloatField,
    fy: FloatField,
    nord: FloatFieldK,
    damp: FloatFieldK,
    do_damp: bool
):
    with computation(PARALLEL), interval(...):
        if do_damp:
            d2 = damp * q
        else:
            d2 = q
        d2copy = d2
        fx1 = del6_v * (d2copy[-1, 0, 0] - d2copy)
        fy1 = del6_u * (d2copy[0, -1, 0] - d2copy)
        if nord > 0:
            d2_2 = (fx1 - fx1[1, 0, 0] + fy1 - fy1[0, 1, 0]) * rarea
            fx1 = -del6_v * (d2_2[-1, 0, 0] - d2_2)
            fy1 = -del6_u * (d2_2[0, -1, 0] - d2_2)
            d2 = (fx1 - fx1[1, 0, 0] + fy1 - fy1[0, 1, 0]) * rarea
            fx = -del6_v * (d2[-1, 0, 0] - d2)
            fy = -del6_u * (d2[0, -1, 0] - d2)
        else:
            fx = fx1
            fy = fy1
class DelnFlux:
    """
    Fortran name is deln_flux
    The test class is DelnFlux
    """

    def __init__(self, nord: FloatFieldK, damp_c: FloatFieldK):
        """
        nord sets the order of damping to apply:
        nord = 0:   del-2
        nord = 1:   del-4
        nord = 2:   del-6

        nord and damp_c define the damping coefficient used in DelnFluxNoSG
        """

        self._no_compute = False
        if (damp_c <= 1e-4).all():
            self._no_compute = True
        elif (damp_c[:-1] <= 1e-4).any():
            raise NotImplementedError(
                "damp_c currently must be always greater than 10^-4 for delnflux"
            )

        grid = spec.grid
        nk = grid.npz
        self._origin = (grid.isd, grid.jsd, 0)

        shape = grid.domain_shape_full(add=(1, 1, 1))
        k_shape = (1, 1, nk)

        self._damp_3d = utils.make_storage_from_shape(k_shape)
        # fields must be 3d to assign to them
        self._fx2 = utils.make_storage_from_shape(shape)
        self._fy2 = utils.make_storage_from_shape(shape)
        self._d2 = utils.make_storage_from_shape(grid.domain_shape_full())

        diffuse_origin = (grid.is_, grid.js, 0)
        extended_domain = (grid.nic + 1, grid.njc + 1, nk)

        self._damping_factor_calculation = FrozenStencil(
            calc_damp, origin=(0, 0, 0), domain=k_shape
        )
        self._add_diffusive_stencil = FrozenStencil(
            add_diffusive_component, origin=diffuse_origin, domain=extended_domain
        )
        self._diffusive_damp_stencil = FrozenStencil(
            diffusive_damp, origin=diffuse_origin, domain=extended_domain
        )

        self._damping_factor_calculation(self._damp_3d, nord, damp_c, grid.da_min)
        self._damp = utils.make_storage_data(self._damp_3d[0, 0, :], (nk,), (0,))

        self.delnflux_nosg = DelnFluxNoSG(nord, nk=nk)

    def __call__(
        self,
        q: FloatField,
        fx: FloatField,
        fy: FloatField,
        d2: Optional["FloatField"] = None,
        mass: Optional["FloatField"] = None,
    ):
        """
        Del-n damping for fluxes, where n = 2 * nord + 2
        Args:
            q: Field for which to calculate damped fluxes (in)
            fx: x-flux on A-grid (inout)
            fy: y-flux on A-grid (inout)
            d2: A damped copy of the q field (in)
            mass: Mass to weight the diffusive flux by (in)
        """
        if self._no_compute is True:
            return fx, fy

        if d2 is None:
            d2 = self._d2

        self.delnflux_nosg(q, self._fx2, self._fy2, self._damp, d2, mass)

        if mass is None:
            self._add_diffusive_stencil(fx, self._fx2, fy, self._fy2)
        else:
            # TODO: To join these stencils you need to overcompute, making the edges
            # 'wrong', but not actually used, separating now for comparison sanity.

            # diffusive_damp(fx, fx2, fy, fy2, mass, damp, origin=diffuse_origin,
            # domain=(grid.nic + 1, grid.njc + 1, nk))
            self._diffusive_damp_stencil(fx, self._fx2, fy, self._fy2, mass, self._damp)

        return fx, fy


class DelnFluxNoSG:
    """
    This contains the mechanics of del6_vt and some of deln_flux from
    the Fortran code, since they are very similar routines. The test class
    is Del6VtFlux
    """

    def __init__(self, nord, nk: Optional[int] = None):
        """
        nord sets the order of damping to apply:
        nord = 0:   del-2
        nord = 1:   del-4
        nord = 2:   del-6
        """
        if max(nord[:]) > 3:
            raise ValueError("nord must be less than 3")
        if not np.all(n in [0, 2, 3] for n in nord[:]):
            raise NotImplementedError("nord must have values 0, 2, or 3")

        self._nmax = int(max(nord[:]))
        self._grid = spec.grid
        if nk is None:
            nk = self._grid.npz
        self._nk = nk
        self._nord = nord

        if self._nk <= 3:
            raise Exception("nk must be more than 3 for DelnFluxNoSG")

        # nmax for this namelist is always 2
        # for n in range(self._nmax):
        # n = 0
        #nt = self._nmax - 1 - 0
        # loop n = 1
        #nt = self._nmax - 1 - 1
        #nt_ny = self._grid.je - self._grid.js + 3 + 2 * nt
        #nt_nx = self._grid.ie - self._grid.is_ + 3 + 2 * nt
       
        self._d2_stencil1 = FrozenStencil(
            delnflux_combined,
            origin=self._grid.full_origin(add=(1, 1, 0)),
            domain=self._grid.domain_shape_full(add=(-2, -2, self._nk - self._grid.npz))
            
        )

    def __call__(self, q, fx2, fy2, damp_c, d2, mass=None):
        """
        Applies del-n damping to fluxes, where n is set by nord.

        Args:
            q: Field for which to calculate damped fluxes (in)
            fx2: diffusive x-flux on A grid (in/out)
            fy2: diffusive y-flux on A grid (in/out)
            damp_c: damping coefficient for q (in)
            d2: A damped copy of the q field (in)
            mass: Mass to weight the diffusive flux by (in)
        """
        do_damp = mass is None
        self._d2_stencil1(q, self._grid.del6_u, self._grid.del6_v, self._grid.rarea, d2, fx2, fy2, self._nord, damp_c, do_damp)
