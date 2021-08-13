from gt4py import gtscript
from gt4py.gtscript import __INLINED, PARALLEL, computation, interval

import fv3core._config as spec
import fv3core.utils.gt4py_utils as utils
from fv3core.decorators import FrozenStencil
from fv3core.stencils import yppm
from fv3core.utils.typing import FloatField, FloatFieldIJ
from fv3core.stencils.basic_operations import sign

@gtscript.function
def final_flux(courant, q, fx1, tmp):
    return q[-1, 0, 0] + fx1 * tmp if courant > 0.0 else q + fx1 * tmp


@gtscript.function
def fx1_fn(courant, br, b0, bl):
    if courant > 0.0:
        ret = (1.0 - courant) * (br[-1, 0, 0] - courant * b0[-1, 0, 0])
    else:
        ret = (1.0 + courant) * (bl + courant * b0)
    return ret


@gtscript.function
def get_tmp(bl, b0, br):
    from __externals__ import mord

    if __INLINED(mord == 5):
        smt5 = bl * br < 0
    else:
        smt5 = (3.0 * abs(b0)) < abs(bl - br)

    if smt5[-1, 0, 0] or smt5[0, 0, 0]:
        tmp = 1.0
    else:
        tmp = 0.0

    return tmp


@gtscript.function
def get_flux(q: FloatField, courant: FloatField, al: FloatField):
    bl = al[0, 0, 0] - q[0, 0, 0]
    br = al[1, 0, 0] - q[0, 0, 0]
    b0 = bl + br

    tmp = get_tmp(bl, b0, br)
    fx1 = fx1_fn(courant, br, b0, bl)
    return final_flux(courant, q, fx1, tmp)  # noqa
@gtscript.function
def main_al(q: FloatField):
    return yppm.p1 * (q[-1, 0, 0] + q) + yppm.p2 * (q[-2, 0, 0] + q[1, 0, 0])

@gtscript.function
def x_flux(q, courant):
    al = main_al(q)
    return get_flux(q, courant, al)
def compute_x_flux(
    q: FloatField,
    courant: FloatField,
    xflux: FloatField,
):
    with computation(PARALLEL), interval(...):
        xflux = x_flux(q, courant)


def finalflux_ord8plus(q: FloatField, c: FloatField, bl: FloatField, br: FloatField, flux: FloatField):
    with computation(PARALLEL), interval(...):
        b0 = yppm.get_b0(bl, br)
        fx1 = fx1_fn(c, br, b0, bl)
        flux = q[-1, 0, 0] + fx1 if c > 0.0 else q + fx1
def dm_iord8plus(q: FloatField, al: FloatField, dm: FloatField):
    with computation(PARALLEL), interval(...):
        xt = 0.25 * (q[1, 0, 0] - q[-1, 0, 0])
        dqr = max(max(q, q[-1, 0, 0]), q[1, 0, 0]) - q
        dql = q - min(min(q, q[-1, 0, 0]), q[1, 0, 0])
        dm = sign(min(min(abs(xt), dqr), dql), xt)

def al_iord8plus(q: FloatField, al: FloatField, dm: FloatField, r3: float):
    with computation(PARALLEL), interval(...):
        al = 0.5 * (q[-1, 0, 0] + q) + r3 * (dm[-1, 0, 0] - dm)

def blbr_iord8(q: FloatField, al: FloatField, bl: FloatField, br: FloatField, dm: FloatField):
    with computation(PARALLEL), interval(...):
        # al, dm = al_iord8plus_fn(q, al, dm, r3)
        xt = 2.0 * dm
        bl = -1.0 * sign(min(abs(xt), abs(al - q)), xt)
        br = sign(min(abs(xt), abs(al[1, 0, 0] - q)), xt)



class XPiecewiseParabolic:
    """
    Fortran name is xppm
    """

    def __init__(
        self,
        namelist,
        iord,
        jfirst,
        jlast,
        dxa=None,
    ):
        self.grid = spec.grid
        assert namelist.grid_type < 3
        self._mord = abs(iord)

        
        flux_origin = (self.grid.is_, jfirst, 0)
        flux_domain = (self.grid.nic + 1, jlast - jfirst + 1, self.grid.npz + 1)

        is1 = self.grid.is_ - 1
        ie3 = self.grid.ie + 2
        if dxa is None:
            self._dxa = self.grid.dxa
        else:
            self._dxa = dxa
        shape = self.grid.domain_shape_full(add=(1, 1, 1))
        self._al = utils.make_storage_from_shape(shape)

        if self._mord < 8:
           # self._main_al_stencil = FrozenStencil(
           #     main_al,
           #     origin=(is1, jfirst, 0),
           #     domain=(ie3 - is1 + 1, jlast - jfirst + 1, self.grid.npz + 1),
           # )

            self._compute_flux_stencil = FrozenStencil(
                func=compute_x_flux,
                externals={
                    "mord": self._mord,
                },
                origin=flux_origin,
                domain=flux_domain,
            )
        else:
            assert iord == 8
            ie1 =  self.grid.ie + 1
            self._bl = utils.make_storage_from_shape(shape)
            self._br = utils.make_storage_from_shape(shape)
            self._dm = utils.make_storage_from_shape(shape)
            dj = jlast - jfirst + 1
            self._dm_iord8plus_stencil = FrozenStencil(
                dm_iord8plus,
                origin=(self.grid.is_ - 2, jfirst, 0), domain=(self.grid.njc + 4, dj,self.grid.npz + 1)
    
            )
            self._al_iord8plus_stencil = FrozenStencil(
                 al_iord8plus, origin=(is1, jfirst, 0), domain=(ie1 - is1 + 2, dj, self.grid.npz + 1)
          
            )
            self._blbr_iord8_stencil = FrozenStencil(
                blbr_iord8,
                origin=(is1, jfirst, 0),
                domain=(ie1 - is1 + 1, dj, self.grid.npz + 1),
            )

            self._do_xt_minmax = True
                
            self._finalflux_ord8plus_stencil =  FrozenStencil(
                finalflux_ord8plus,
                origin=flux_origin,
                domain=flux_domain,
            )

    #def compute_al(self, q):
    #    self._main_al_stencil(q, self._al)

    def _compute_blbr_ord8plus(self, q):
        r3 = 1.0 / 3.0
       
        self._dm_iord8plus_stencil(
            q, self._al, self._dm,
        )
        self._al_iord8plus_stencil(
            q, self._al, self._dm, r3,
        )
        self._blbr_iord8_stencil(
            q,
            self._al,
            self._bl,
            self._br,
            self._dm,
        )


    def __call__(self, q: FloatField, c: FloatField, xflux: FloatField):
        """
        Compute x-flux using the PPM method.

        Args:
            q (in): Transported scalar
            c (in): Courant number
            xflux (out): Flux
            jfirst: Starting index of the J-dir compute domain
            jlast: Final index of the J-dir compute domain
        """
        if self._mord < 8:
            #self.compute_al(q)
            self._compute_flux_stencil(q, c,  xflux)
        else:
            self._compute_blbr_ord8plus(q)
            self._finalflux_ord8plus_stencil(q, c, self._bl, self._br, xflux)
