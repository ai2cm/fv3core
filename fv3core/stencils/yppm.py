from gt4py import gtscript
from gt4py.gtscript import PARALLEL, computation, interval

import fv3core._config as spec
import fv3core.utils.gt4py_utils as utils
from fv3core.decorators import FrozenStencil
from fv3core.utils.typing import FloatField, FloatFieldIJ
from fv3core.stencils.basic_operations import sign

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

@gtscript.function
def xt_dya_edge_0_base(q, dya):
    return 0.5 * (
        ((2.0 * dya + dya[0, -1]) * q - dya * q[0, -1, 0]) / (dya[0, -1] + dya)
        + ((2.0 * dya[0, 1] + dya[0, 2]) * q[0, 1, 0] - dya[0, 1] * q[0, 2, 0])
        / (dya[0, 1] + dya[0, 2])
    )
@gtscript.function
def xt_dya_edge_1_base(q, dya):
    return 0.5 * (
        (
            (2.0 * dya[0, -1] + dya[0, -2]) * q[0, -1, 0]
            - dya[0, -1] * q[0, -2, 0]
        )
        / (dya[0, -2] + dya[0, -1])
        + ((2.0 * dya + dya[0, 1]) * q - dya * q[0, 1, 0]) / (dya + dya[0, 1])
    )


@gtscript.function
def xt_dya_edge_0(q, dya, xt_minmax):
    xt = xt_dya_edge_0_base(q, dya)
    if xt_minmax:
        minq = min(min(min(q[0, -1, 0], q), q[0, 1, 0]), q[0, 2, 0])
        maxq = max(max(max(q[0, -1, 0], q), q[0, 1, 0]), q[0, 2, 0])
        xt = min(max(xt, minq), maxq)
    return xt


@gtscript.function
def xt_dya_edge_1(q, dya, xt_minmax):
    xt = xt_dya_edge_1_base(q, dya)
    if xt_minmax:
        minq = min(min(min(q[0, -2, 0], q[0, -1, 0]), q), q[0, 1, 0])
        maxq = max(max(max(q[0, -2, 0], q[0, -1, 0]), q), q[0, 1, 0])
        xt = min(max(xt, minq), maxq)
    return xt



@gtscript.function
def get_b0(bl, br):
    b0 = bl + br
    return b0


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

def finalflux_ord8plus(q: FloatField, c: FloatField, bl: FloatField, br: FloatField, flux: FloatField):
    with computation(PARALLEL), interval(...):
        b0 = get_b0(bl, br)
        fx1 = fx1_fn(c, br, b0, bl)
        flux = q[0, -1, 0] + fx1 if c > 0.0 else q + fx1

def dm_jord8plus(q: FloatField, al: FloatField, dm: FloatField):
    with computation(PARALLEL), interval(...):
        xt = 0.25 * (q[0, 1, 0] - q[0, -1, 0])
        dqr = max(max(q, q[0, -1, 0]), q[0, 1, 0]) - q
        dql = q - min(min(q, q[0, -1, 0]), q[0, 1, 0])
        dm = sign(min(min(abs(xt), dqr), dql), xt)

def al_jord8plus(q: FloatField, al: FloatField, dm: FloatField, r3: float):
    with computation(PARALLEL), interval(...):
        al = 0.5 * (q[0, -1, 0] + q) + r3 * (dm[0, -1, 0] - dm)

def blbr_jord8(q: FloatField, al: FloatField, bl: FloatField, br: FloatField, dm: FloatField):
    with computation(PARALLEL), interval(...):
        xt = 2.0 * dm
        aldiff = al - q
        aldiffj = al[0, 1, 0] - q
        bl = -1.0 * sign(min(abs(xt), abs(aldiff)), xt)
        br = sign(min(abs(xt), abs(aldiffj)), xt)

def south_edge_jord8plus_0(q: FloatField, dya: FloatFieldIJ, dm: FloatField, bl: FloatField, br: FloatField, xt_minmax: bool):
    with computation(PARALLEL), interval(...):
        bl = s14 * dm[0, -1, 0] + s11 * (q[0, -1, 0] - q)
        xt = xt_dya_edge_0(q, dya, xt_minmax)
        br = xt - q



def south_edge_jord8plus_1(q: FloatField, dya: FloatFieldIJ, dm: FloatField, bl: FloatField, br: FloatField, xt_minmax: bool):
    with computation(PARALLEL), interval(...):
        xt = xt_dya_edge_1(q, dya, xt_minmax)
        bl = xt - q
        xt = s15 * q + s11 * q[0, 1, 0] - s14 * dm[0, 1, 0]
        br = xt - q



def south_edge_jord8plus_2(q: FloatField, dya: FloatFieldIJ, dm: FloatField, al: FloatField, bl: FloatField, br: FloatField):
    with computation(PARALLEL), interval(...):
        xt = s15 * q[0, -1, 0] + s11 * q - s14 * dm
        bl = xt - q
        br = al[0, 1, 0] - q



def north_edge_jord8plus_0(q: FloatField, dya: FloatFieldIJ, dm: FloatField, al: FloatField, bl: FloatField, br: FloatField):
    with computation(PARALLEL), interval(...):
        bl = al - q
        xt = s15 * q[0, 1, 0] + s11 * q + s14 * dm
        br = xt - q



def north_edge_jord8plus_1(q: FloatField, dya: FloatFieldIJ, dm: FloatField, bl: FloatField, br: FloatField, xt_minmax: bool):
    with computation(PARALLEL), interval(...):
        xt = s15 * q + s11 * q[0, -1, 0] + s14 * dm[0, -1, 0]
        bl = xt - q
        xt = xt_dya_edge_0(q, dya, xt_minmax)
        br = xt - q



def north_edge_jord8plus_2(q: FloatField, dya: FloatFieldIJ, dm: FloatField, bl: FloatField, br: FloatField, xt_minmax: bool):
    with computation(PARALLEL), interval(...):
        xt = xt_dya_edge_1(q, dya, xt_minmax)
        bl = xt - q
        br = s11 * (q[0, 1, 0] - q) - s14 * dm[0, 1, 0]



def pert_ppm_standard_constraint(a0: FloatField, al: FloatField, ar: FloatField):
    with computation(PARALLEL), interval(...):
        da1 = 0.0
        da2 = 0.0
        a6da = 0.0
        if al * ar < 0.0:
            da1 = al - ar
            da2 = da1 ** 2
            a6da = 3.0 * (al + ar) * da1
            if a6da < -da2:
                ar = -2.0 * al
            elif a6da > da2:
                al = -2.0 * ar
        else:
            # effect of dm=0 included here
            al = 0.0
            ar = 0.0

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
        self._mord = abs(jord)
       
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
       
        if self._mord < 8:
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
                    "mord": self._mord,
                },
                origin=flux_origin,
                domain=flux_domain,
            )
        else:
            assert jord == 8
            je1 = self.grid.je - 2 if self.grid.north_edge else self.grid.je + 1
            self._bl = utils.make_storage_from_shape(shape)
            self._br = utils.make_storage_from_shape(shape)
            self._dm = utils.make_storage_from_shape(shape)
            di = ilast - ifirst + 1
            self._dm_jord8plus_stencil = FrozenStencil(
                dm_jord8plus,
                origin=(ifirst, self.grid.js - 2, 0), domain=(di, self.grid.njc + 4, self.grid.npz + 1)
            )
            self._al_jord8plus_stencil = FrozenStencil(
                 al_jord8plus, origin=(ifirst, js1, 0), domain=(di, je1 - js1 + 2, self.grid.npz + 1)
            )
            self._blbr_jord8_stencil = FrozenStencil(
                blbr_jord8,
                origin=(ifirst, js1, 0),
                domain=(di, je1 - js1 + 1, self.grid.npz + 1),
            )
            x_edge_domain = (di, 1, self.grid.npz + 1)
            self._do_xt_minmax = True
            if self.grid.south_edge:
                self._south_edge_jord8plus_0_stencil = FrozenStencil(
                    south_edge_jord8plus_0,
                    origin=(ifirst, self.grid.js - 1, 0),
                    domain=x_edge_domain,
                )
                self._south_edge_jord8plus_1_stencil = FrozenStencil(
                    south_edge_jord8plus_1,
                    origin=(ifirst, self.grid.js, 0),
                    domain=x_edge_domain,
                )
                self._south_edge_jord8plus_2_stencil = FrozenStencil(
                    south_edge_jord8plus_2,
                    origin=(ifirst, self.grid.js+1, 0),
                    domain=x_edge_domain,
                )
                self._pert_ppm_south_stencil = FrozenStencil(
                    pert_ppm_standard_constraint,
                    origin=(ifirst, self.grid.js - 1, 0), domain=(di, 3, self.grid.npz+1)
                )
                
            if self.grid.north_edge:
                self._north_edge_jord8plus_0_stencil = FrozenStencil(
                    north_edge_jord8plus_0,
                    origin=(ifirst, self.grid.je - 1, 0),
                    domain=x_edge_domain,
                )
                self._north_edge_jord8plus_1_stencil = FrozenStencil(
                    north_edge_jord8plus_1,
                    origin=(ifirst, self.grid.je, 0),
                    domain=x_edge_domain,
                )
                self._north_edge_jord8plus_2_stencil = FrozenStencil(
                    north_edge_jord8plus_2,
                    origin=(ifirst, self.grid.je + 1, 0),
                    domain=x_edge_domain,
                )
                self._pert_ppm_north_stencil = FrozenStencil(
                    pert_ppm_standard_constraint,
                   origin=(ifirst, self.grid.je - 1, 0), domain=(di, 3, self.grid.npz+1)
                )
                
            self._finalflux_ord8plus_stencil =  FrozenStencil(
                finalflux_ord8plus,
                origin=flux_origin,
                domain=flux_domain,
            )
              

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
    
    def compute_blbr_ord8plus(self, q):
        r3 = 1.0 / 3.0
     
        self._dm_jord8plus_stencil(q, self._al, self._dm)
        self._al_jord8plus_stencil(q, self._al, self._dm, r3)
        self._blbr_jord8_stencil(q, self._al, self._bl, self._br, self._dm)
        if self.grid.south_edge:
            self._south_edge_jord8plus_0_stencil(
                q,
                self._dya,
                self._dm,
                self._bl,
                self._br,
                self._do_xt_minmax,
            )
            self._south_edge_jord8plus_1_stencil(
                q,
                self._dya,
                self._dm,
                self._bl,
                self._br,
                self._do_xt_minmax,
            )
            self._south_edge_jord8plus_2_stencil(
                q,
                self._dya,
                self._dm,
                self._al,
                self._bl,
                self._br,
            )
            self._pert_ppm_south_stencil(q, self._bl, self._br)
        if self.grid.north_edge:
            self._north_edge_jord8plus_0_stencil(
                q,
                self._dya,
                self._dm,
                self._al,
                self._bl,
                self._br,
            )
            self._north_edge_jord8plus_1_stencil(
                q,
                self._dya,
                self._dm,
                self._bl,
                self._br,
                self._do_xt_minmax,
            )
            self._north_edge_jord8plus_2_stencil(
                q,
                self._dya,
                self._dm,
                self._bl,
                self._br,
                self._do_xt_minmax,
                )

            self._pert_ppm_north_stencil(q, self._bl, self._br)

    def __call__(self, q: FloatField, c: FloatField, flux: FloatField):
        """
        Compute y-flux using the PPM method.

        Args:
            q (in): Transported scalar
            c (in): Courant number
            flux (out): Flux
            ifirst: Starting index of the I-dir compute domain
            ilast: Final index of the I-dir compute domain
        """
        if self._mord < 8:
            self.compute_al(q)
            self._compute_flux_stencil(q, c, self._al, self._dya, flux)
        else:
            self.compute_blbr_ord8plus(q)
            self._finalflux_ord8plus_stencil(q, c, self._bl, self._br, flux)
