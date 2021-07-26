import dace
from gt4py import gtscript
from gt4py.gtscript import __INLINED, PARALLEL, computation, interval

import fv3core._config as spec
import fv3core.utils.gt4py_utils as utils
from fv3core.decorators import FrozenStencil, computepath_method
from fv3core.stencils import yppm
from fv3core.utils.typing import FloatField, FloatFieldIJ


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


def main_al(q: FloatField, al: FloatField):
    with computation(PARALLEL), interval(...):
        al = yppm.p1 * (q[-1, 0, 0] + q) + yppm.p2 * (q[-2, 0, 0] + q[1, 0, 0])


def al_y_edge_0(q: FloatField, al: FloatField):
    with computation(PARALLEL), interval(0, None):
        al = yppm.c1 * q[-2, 0, 0] + yppm.c2 * q[-1, 0, 0] + yppm.c3 * q


def al_y_edge_1(q: FloatField, dxa: FloatFieldIJ, al: FloatField):
    with computation(PARALLEL), interval(0, None):
        al = 0.5 * (
            ((2.0 * dxa[-1, 0] + dxa[-2, 0]) * q[-1, 0, 0] - dxa[-1, 0] * q[-2, 0, 0])
            / (dxa[-2, 0] + dxa[-1, 0])
            + ((2.0 * dxa[0, 0] + dxa[1, 0]) * q[0, 0, 0] - dxa[0, 0] * q[1, 0, 0])
            / (dxa[0, 0] + dxa[1, 0])
        )


def al_y_edge_2(q: FloatField, dxa: FloatFieldIJ, al: FloatField):
    with computation(PARALLEL), interval(0, None):
        al = yppm.c3 * q[-1, 0, 0] + yppm.c2 * q[0, 0, 0] + yppm.c1 * q[1, 0, 0]


def compute_x_flux(
    q: FloatField,
    courant: FloatField,
    al: FloatField,
    dxa: FloatFieldIJ,
    xflux: FloatField,
):
    with computation(PARALLEL), interval(...):
        xflux = get_flux(q, courant, al)


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
        assert iord < 8
        flux_origin = (self.grid.is_, jfirst, 0)
        flux_domain = (self.grid.nic + 1, jlast - jfirst + 1, self.grid.npz + 1)

        is1 = self.grid.is_ + 2 if self.grid.west_edge else self.grid.is_ - 1
        ie3 = self.grid.ie - 1 if self.grid.east_edge else self.grid.ie + 2
        if dxa is None:
            self._dxa = self.grid.dxa
        else:
            self._dxa = dxa
        shape = self.grid.domain_shape_full(add=(1, 1, 1))
        edge_domain = (1, shape[1], shape[2])
        self._al = utils.make_storage_from_shape(shape)
        self._main_al_stencil = FrozenStencil(
            main_al,
            origin=(is1, jfirst, 0),
            domain=(ie3 - is1 + 1, jlast - jfirst + 1, self.grid.npz + 1),
        )
        if self.grid.west_edge:
            self._al_west_0_stencil = FrozenStencil(
                al_y_edge_0, origin=(self.grid.is_ - 1, 0, 0), domain=edge_domain
            )
            self._al_west_1_stencil = FrozenStencil(
                al_y_edge_1, origin=(self.grid.is_, 0, 0), domain=edge_domain
            )
            self._al_west_2_stencil = FrozenStencil(
                al_y_edge_2, origin=(self.grid.is_ + 1, 0, 0), domain=edge_domain
            )
        if self.grid.east_edge:
            self._al_east_0_stencil = FrozenStencil(
                al_y_edge_0, origin=(self.grid.ie, 0, 0), domain=edge_domain
            )
            self._al_east_1_stencil = FrozenStencil(
                al_y_edge_1, origin=(self.grid.ie + 1, 0, 0), domain=edge_domain
            )
            self._al_east_2_stencil = FrozenStencil(
                al_y_edge_2, origin=(self.grid.ie + 2, 0, 0), domain=edge_domain
            )

        self._compute_flux_stencil = FrozenStencil(
            func=compute_x_flux,
            externals={
                "mord": abs(iord),
            },
            origin=flux_origin,
            domain=flux_domain,
        )

    @computepath_method
    def compute_al(self, q):
        self._main_al_stencil(q, self._al)
        if self.grid.west_edge:
            self._al_west_0_stencil(q, self._al)
            self._al_west_1_stencil(q, self._dxa, self._al)
            self._al_west_2_stencil(q, self._dxa, self._al)

        if self.grid.east_edge:
            self._al_east_0_stencil(q, self._al)
            self._al_east_1_stencil(q, self._dxa, self._al)
            self._al_east_2_stencil(q, self._dxa, self._al)

    @computepath_method
    def __call__(self, q, c, xflux):
        """
        Compute x-flux using the PPM method.

        Args:
            q (in): Transported scalar
            c (in): Courant number
            xflux (out): Flux
            jfirst: Starting index of the J-dir compute domain
            jlast: Final index of the J-dir compute domain
        """
        self.compute_al(q)
        self._compute_flux_stencil(q, c, self._al, self._dxa, xflux)
