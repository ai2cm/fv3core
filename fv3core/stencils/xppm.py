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
def xt_dxa_edge_0_base(q, dxa):
    return 0.5 * (
        ((2.0 * dxa + dxa[-1, 0]) * q - dxa * q[-1, 0, 0]) / (dxa[-1, 0] + dxa)
        + ((2.0 * dxa[1, 0] + dxa[2, 0]) * q[1, 0, 0] - dxa[1, 0] * q[2, 0, 0])
        / (dxa[1, 0] + dxa[2, 0])
    )


@gtscript.function
def xt_dxa_edge_1_base(q, dxa):
    return 0.5 * (
        (
            (2.0 * dxa[-1, 0] + dxa[-2, 0]) * q[-1, 0, 0]
            - dxa[-1, 0] * q[-2, 0, 0]
        )
        / (dxa[-2, 0] + dxa[-1, 0])
        + ((2.0 * dxa + dxa[1, 0]) * q - dxa * q[1, 0, 0]) / (dxa + dxa[1, 0])
    )


@gtscript.function
def xt_dxa_edge_0(q, dxa, xt_minmax):
    xt = xt_dxa_edge_0_base(q, dxa)
    minq = 0.0
    maxq = 0.0
    if xt_minmax:
        minq = min(min(min(q[-1, 0, 0], q), q[1, 0, 0]), q[2, 0, 0])
        maxq = max(max(max(q[-1, 0, 0], q), q[1, 0, 0]), q[2, 0, 0])
        xt = min(max(xt, minq), maxq)
    return xt


@gtscript.function
def xt_dxa_edge_1(q, dxa, xt_minmax):
    xt = xt_dxa_edge_1_base(q, dxa)
    minq = 0.0
    maxq = 0.0
    if xt_minmax:
        minq = min(min(min(q[-2, 0, 0], q[-1, 0, 0]), q), q[1, 0, 0])
        maxq = max(max(max(q[-2, 0, 0], q[-1, 0, 0]), q), q[1, 0, 0])
        xt = min(max(xt, minq), maxq)
    return xt


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


def west_edge_iord8plus_0(q: FloatField, dxa: FloatFieldIJ, dm: FloatField, bl: FloatField, br: FloatField, xt_minmax: bool):
    with computation(PARALLEL), interval(...):
        bl = yppm.s14 * dm[-1, 0, 0] + yppm.s11 * (q[-1, 0, 0] - q)
        xt = xt_dxa_edge_0(q, dxa, xt_minmax)
        br = xt - q
        
def west_edge_iord8plus_1(q: FloatField, dxa: FloatFieldIJ, dm: FloatField, bl: FloatField, br: FloatField, xt_minmax: bool):
    with computation(PARALLEL), interval(...):
        xt = xt_dxa_edge_1(q, dxa, xt_minmax)
        bl = xt - q
        xt = yppm.s15 * q + yppm.s11 * q[1, 0, 0] - yppm.s14 * dm[1, 0, 0]
        br = xt - q
def west_edge_iord8plus_2(q: FloatField, dxa: FloatFieldIJ, dm: FloatField, al: FloatField, bl: FloatField, br: FloatField):
    with computation(PARALLEL), interval(...):
        xt = yppm.s15 * q[-1, 0, 0] + yppm.s11 * q - yppm.s14 * dm
        bl = xt - q
        br = al[1, 0, 0] - q

def east_edge_iord8plus_0(q: FloatField, dxa: FloatFieldIJ, dm: FloatField, al: FloatField, bl: FloatField, br: FloatField):
    with computation(PARALLEL), interval(...):
        bl = al - q
        xt = yppm.s15 * q[1, 0, 0] + yppm.s11 * q + yppm.s14 * dm
        br = xt - q

def east_edge_iord8plus_1(q: FloatField, dxa: FloatFieldIJ, dm: FloatField, bl: FloatField, br: FloatField, xt_minmax: bool):
    with computation(PARALLEL), interval(...):
        xt = yppm.s15 * q + yppm.s11 * q[-1, 0, 0] + yppm.s14 * dm[-1, 0, 0]
        bl = xt - q
        xt = xt_dxa_edge_0(q, dxa, xt_minmax)
        br = xt - q
def east_edge_iord8plus_2(q: FloatField, dxa: FloatFieldIJ, dm: FloatField, bl: FloatField, br: FloatField, xt_minmax: bool):
    with computation(PARALLEL), interval(...):
        xt = xt_dxa_edge_1(q, dxa, xt_minmax)
        bl = xt - q
        br = yppm.s11 * (q[1, 0, 0] - q) - yppm.s14 * dm[1, 0, 0]




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

        is1 = self.grid.is_ + 2 if self.grid.west_edge else self.grid.is_ - 1
        ie3 = self.grid.ie - 1 if self.grid.east_edge else self.grid.ie + 2
        if dxa is None:
            self._dxa = self.grid.dxa
        else:
            self._dxa = dxa
        shape = self.grid.domain_shape_full(add=(1, 1, 1))
        self._al = utils.make_storage_from_shape(shape)
        edge_domain = (1, shape[1], shape[2])
        if self._mord < 8:
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
                    "mord": self._mord,
                },
                origin=flux_origin,
                domain=flux_domain,
            )
        else:
            assert iord == 8
            ie1 = self.grid.ie - 2 if self.grid.east_edge else self.grid.ie + 1
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
            y_edge_domain = (1, dj, self.grid.npz + 1)
            self._do_xt_minmax = True
            if self.grid.west_edge:
                self._west_edge_iord8plus_0_stencil = FrozenStencil(
                    west_edge_iord8plus_0,
                    origin=(self.grid.is_ - 1, jfirst, 0),
                    domain=y_edge_domain,
                )
                self._west_edge_iord8plus_1_stencil = FrozenStencil(
                    west_edge_iord8plus_1,
                    origin=(self.grid.is_, jfirst, 0),
                    domain=y_edge_domain,
                )
                self._west_edge_iord8plus_2_stencil = FrozenStencil(
                    west_edge_iord8plus_2,
                    origin=(self.grid.is_+1, jfirst, 0),
                    domain=y_edge_domain,
                )
                self._pert_ppm_west_stencil = FrozenStencil(
                    yppm.pert_ppm_standard_constraint,
                    origin=(self.grid.is_ - 1, jfirst, 0), domain=(3, dj, self.grid.npz+1)
                )
                
            if self.grid.east_edge:
                self._east_edge_iord8plus_0_stencil = FrozenStencil(
                    east_edge_iord8plus_0,
                    origin=(self.grid.ie - 1, jfirst, 0),
                    domain=y_edge_domain,
                )
                self._east_edge_iord8plus_1_stencil = FrozenStencil(
                    east_edge_iord8plus_1,
                    origin=(self.grid.ie, jfirst, 0),
                    domain=y_edge_domain,
                )
                self._east_edge_iord8plus_2_stencil = FrozenStencil(
                    east_edge_iord8plus_2,
                    origin=(self.grid.ie + 1, jfirst, 0),
                    domain=y_edge_domain,
                )
                self._pert_ppm_east_stencil = FrozenStencil(
                    yppm.pert_ppm_standard_constraint,
                    origin=(self.grid.ie - 1, jfirst, 0), domain=(3, dj, self.grid.npz+1)
                )
                
            self._finalflux_ord8plus_stencil =  FrozenStencil(
                finalflux_ord8plus,
                origin=flux_origin,
                domain=flux_domain,
            )

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
        if self.grid.west_edge:
            self._west_edge_iord8plus_0_stencil(
                q,
                self._dxa,
                self._dm,
                self._bl,
                self._br,
                self._do_xt_minmax,
            )
            self._west_edge_iord8plus_1_stencil(
                q,
                self._dxa,
                self._dm,
                self._bl,
                self._br,
                self._do_xt_minmax,
            )
            self._west_edge_iord8plus_2_stencil(
                q,
                self._dxa,
                self._dm,
                self._al,
                self._bl,
                self._br,
            )
            self._pert_ppm_west_stencil(q, self._bl, self._br)
        if self.grid.east_edge:
            self._east_edge_iord8plus_0_stencil(
                q,
                self._dxa,
                self._dm,
                self._al,
                self._bl,
                self._br,
            )
            self._east_edge_iord8plus_1_stencil(
                q,
                self._dxa,
                self._dm,
                self._bl,
                self._br,
                self._do_xt_minmax,
            )
            self._east_edge_iord8plus_2_stencil(
                q,
                self._dxa,
                self._dm,
                self._bl,
                self._br,
                self._do_xt_minmax,
            )
            self._pert_ppm_east_stencil(q, self._bl, self._br)
            


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
            self.compute_al(q)
            self._compute_flux_stencil(q, c, self._al, self._dxa, xflux)
        else:
            self._compute_blbr_ord8plus(q)
            self._finalflux_ord8plus_stencil(q, c, self._bl, self._br, xflux)
