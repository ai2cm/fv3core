import dace
from gt4py import gtscript
from gt4py.gtscript import PARALLEL, computation, interval

import fv3core._config as spec
import fv3core.utils.gt4py_utils as utils
from fv3core.decorators import FrozenStencil, computepath_method
from fv3core.utils.typing import FloatField, FloatFieldIJ


input_vars = ["q", "c"]
inputs_params = ["jord", "ifirst", "ilast"]
output_vars = ["flux"]

# volume-conserving cubic with 2nd drv=0 at end point:
# non-monotonic
c1 = -2.0 / 14.0
c2 = 11.0 / 14.0
c3 = 5.0 / 14.0

# PPM volume mean form
p1 = 7.0 / 12.0
p2 = -1.0 / 12.0

s11 = 11.0 / 14.0
s14 = 4.0 / 7.0
s15 = 3.0 / 14.0


@gtscript.function
def final_flux(courant, q, fx1, tmp):
    return q[0, -1, 0] + fx1 * tmp if courant > 0.0 else q + fx1 * tmp


@gtscript.function
def fx1_fn(courant, br, b0, bl):
    if courant > 0.0:
        ret = (1.0 - courant) * (br[0, -1, 0] - courant * b0[0, -1, 0])
    else:
        ret = (1.0 + courant) * (bl + courant * b0)
    return ret


@gtscript.function
def get_tmp(bl, b0, br):
    from __externals__ import mord

    if mord == 5:
        smt5 = bl * br < 0
    else:
        smt5 = (3.0 * abs(b0)) < abs(bl - br)

    if smt5[0, -1, 0] or smt5[0, 0, 0]:
        tmp = 1.0
    else:
        tmp = 0.0

    return tmp


@gtscript.function
def get_flux(q: FloatField, courant: FloatField, al: FloatField):
    bl = al[0, 0, 0] - q[0, 0, 0]
    br = al[0, 1, 0] - q[0, 0, 0]
    b0 = bl + br

    tmp = get_tmp(bl, b0, br)
    fx1 = fx1_fn(courant, br, b0, bl)
    return final_flux(courant, q, fx1, tmp)


def main_al(q: FloatField, al: FloatField):
    with computation(PARALLEL), interval(...):
        al = p1 * (q[0, -1, 0] + q) + p2 * (q[0, -2, 0] + q[0, 1, 0])


def al_x_edge_0(q: FloatField, al: FloatField):
    with computation(PARALLEL), interval(0, None):
        al = c1 * q[0, -2, 0] + c2 * q[0, -1, 0] + c3 * q


def al_x_edge_1(q: FloatField, dya: FloatFieldIJ, al: FloatField):
    with computation(PARALLEL), interval(0, None):
        al = 0.5 * (
            ((2.0 * dya[0, -1] + dya[0, -2]) * q[0, -1, 0] - dya[0, -1] * q[0, -2, 0])
            / (dya[0, -2] + dya[0, -1])
            + ((2.0 * dya[0, 0] + dya[0, 1]) * q[0, 0, 0] - dya[0, 0] * q[0, 1, 0])
            / (dya[0, 0] + dya[0, 1])
        )


def al_x_edge_2(q: FloatField, dya: FloatFieldIJ, al: FloatField):
    with computation(PARALLEL), interval(0, None):
        al = c3 * q[0, -1, 0] + c2 * q[0, 0, 0] + c1 * q[0, 1, 0]


def compute_y_flux(
    q: FloatField,
    courant: FloatField,
    al: FloatField,
    dya: FloatFieldIJ,
    yflux: FloatField,
):
    with computation(PARALLEL), interval(...):
        yflux = get_flux(q, courant, al)


class YPiecewiseParabolic:
    """
    Fortran name is yppm
    """

    def __init__(self, namelist, jord, ifirst, ilast, dya=None, js1=None, je3=None):
        self.grid = spec.grid
        assert namelist.grid_type < 3
        if abs(jord) not in [5, 6, 7, 8]:
            raise NotImplementedError(
                f"Unimplemented hord value, {jord}. "
                "Currently only support hord={5, 6, 7, 8}"
            )
        flux_origin = (ifirst, self.grid.js, 0)
        flux_domain = (ilast - ifirst + 1, self.grid.njc + 1, self.grid.npz + 1)
        if js1 is None:
            js1 = self.grid.js + 2 if self.grid.south_edge else self.grid.js - 1
        if je3 is None:
            je3 = self.grid.je - 1 if self.grid.north_edge else self.grid.je + 2
        if dya is None:
            self._dya = self.grid.dya
        else:
            self._dya = dya
        shape = self.grid.domain_shape_full(add=(1, 1, 1))
        self._al = utils.make_storage_from_shape(shape)
        edge_domain = (shape[0], 1, shape[2])
        self._main_al_stencil = FrozenStencil(
            main_al,
            origin=(ifirst, js1, 0),
            domain=(ilast - ifirst + 1, je3 - js1 + 1, self.grid.npz + 1),
        )
        if self.grid.south_edge:
            self._al_south_0_stencil = FrozenStencil(
                al_x_edge_0, origin=(0, self.grid.js - 1, 0), domain=edge_domain
            )
            self._al_south_1_stencil = FrozenStencil(
                al_x_edge_1, origin=(0, self.grid.js, 0), domain=edge_domain
            )
            self._al_south_2_stencil = FrozenStencil(
                al_x_edge_2, origin=(0, self.grid.js + 1, 0), domain=edge_domain
            )
        if self.grid.north_edge:
            self._al_north_0_stencil = FrozenStencil(
                al_x_edge_0, origin=(0, self.grid.je, 0), domain=edge_domain
            )
            self._al_north_1_stencil = FrozenStencil(
                al_x_edge_1, origin=(0, self.grid.je + 1, 0), domain=edge_domain
            )
            self._al_north_2_stencil = FrozenStencil(
                al_x_edge_2, origin=(0, self.grid.je + 2, 0), domain=edge_domain
            )
        self._compute_flux_stencil = FrozenStencil(
            func=compute_y_flux,
            externals={
                "mord": abs(jord),
            },
            origin=flux_origin,
            domain=flux_domain,
        )

    @computepath_method
    def compute_al(self, q):
        self._main_al_stencil(q, self._al)
        if self.grid.south_edge:
            self._al_south_0_stencil(q, self._al)
            self._al_south_1_stencil(q, self._dya, self._al)
            self._al_south_2_stencil(q, self._dya, self._al)

        if self.grid.north_edge:
            self._al_north_0_stencil(q, self._al)
            self._al_north_1_stencil(q, self._dya, self._al)
            self._al_north_2_stencil(q, self._dya, self._al)

    @computepath_method
    def __call__(self, q, c, flux):
        """
        Compute y-flux using the PPM method.

        Args:
            q (in): Transported scalar
            c (in): Courant number
            flux (out): Flux
            ifirst: Starting index of the I-dir compute domain
            ilast: Final index of the I-dir compute domain
        """
        self.compute_al(q)
        self._compute_flux_stencil(q, c, self._al, self._dya, flux)
