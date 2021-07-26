import dace
from gt4py import gtscript
from gt4py.gtscript import __INLINED, PARALLEL, computation, interval

import fv3core._config as spec
import fv3core.utils.gt4py_utils as utils
from fv3core.decorators import FrozenStencil, computepath_method
from fv3core.stencils import xtp_u, yppm
from fv3core.utils.typing import FloatField, FloatFieldIJ


@gtscript.function
def _get_flux(
    v: FloatField,
    courant: FloatField,
    rdy: FloatFieldIJ,
    bl: FloatField,
    br: FloatField,
):
    """
    Compute the y-dir flux of kinetic energy(?).

    Inputs:
        v: y-dir wind
        courant: Courant number in flux form
        rdy: 1.0 / dy
        bl: ???
        br: ???

    Returns:
        Kinetic energy flux
    """
    from __externals__ import jord

    b0 = bl + br
    cfl = courant * rdy[0, -1] if courant > 0 else courant * rdy[0, 0]
    fx0 = yppm.fx1_fn(cfl, br, b0, bl)

    if __INLINED(jord < 8):
        tmp = yppm.get_tmp(bl, b0, br)
    else:
        tmp = 1.0
    return yppm.final_flux(courant, v, fx0, tmp)


def bl_br_main(v: FloatField, al: FloatField, bl: FloatField, br: FloatField):
    with computation(PARALLEL), interval(...):
        bl = al[0, 0, 0] - v[0, 0, 0]
        br = al[0, 1, 0] - v[0, 0, 0]


def _ytp_v(
    courant: FloatField,
    v: FloatField,
    bl: FloatField,
    br: FloatField,
    flux: FloatField,
    rdy: FloatFieldIJ,
):
    with computation(PARALLEL), interval(...):

        flux = _get_flux(v, courant, rdy, bl, br)


class YTP_V:
    def __init__(self, namelist):
        jord = spec.namelist.hord_mt
        assert jord < 8
        if jord not in (5, 6, 7, 8):
            raise NotImplementedError(
                "Currently xtp_v is only supported for hord_mt == 5,6,7,8"
            )
        assert namelist.grid_type < 3

        self.grid = spec.grid

        self.dy = self.grid.dy
        self.rdy = self.grid.rdy

        assert namelist.grid_type < 3
        shape = self.grid.domain_shape_full(add=(1, 1, 1))
        self._bl = utils.make_storage_from_shape(shape)
        self._br = utils.make_storage_from_shape(shape)
        self._bl_br_stencil = FrozenStencil(
            bl_br_main,
            origin=(self.grid.is_, self.grid.js - 1, 0),
            domain=(self.grid.nic + 1, self.grid.njc + 2, self.grid.npz),
        )
        corner_domain = (1, 2, self.grid.npz)
        if self.grid.sw_corner:
            self._zero_bl_br_sw_corner_stencil = FrozenStencil(
                xtp_u.zero_br_bl,
                origin=(self.grid.is_, self.grid.js - 1, 0),
                domain=corner_domain,
            )
        if self.grid.nw_corner:
            self._zero_bl_br_nw_corner_stencil = FrozenStencil(
                xtp_u.zero_br_bl,
                origin=(self.grid.is_, self.grid.je, 0),
                domain=corner_domain,
            )
        if self.grid.se_corner:
            self._zero_bl_br_se_corner_stencil = FrozenStencil(
                xtp_u.zero_br_bl,
                origin=(self.grid.ie + 1, self.grid.js - 1, 0),
                domain=corner_domain,
            )
        if self.grid.ne_corner:
            self._zero_bl_br_ne_corner_stencil = FrozenStencil(
                xtp_u.zero_br_bl,
                origin=(self.grid.ie + 1, self.grid.je, 0),
                domain=corner_domain,
            )

        self.stencil = FrozenStencil(
            _ytp_v,
            externals={
                "jord": jord,
                "mord": jord,
            },
            origin=self.grid.compute_origin(),
            domain=self.grid.domain_shape_compute(add=(1, 1, 0)),
        )

        self._yppm = yppm.YPiecewiseParabolic(
            namelist,
            jord,
            self.grid.is_,
            self.grid.ie + 1,
            self.grid.dy,
            self.grid.js - 1,
            self.grid.je + 2,
        )

    @computepath_method
    def __call__(self, c, v, flux):
        """
        Compute flux of kinetic energy in y-dir.

        Args:
        c (in): Courant number in flux form
        v (in): y-dir wind on Arakawa D-grid
        flux (out): Flux of kinetic energy
        """
        self._yppm.compute_al(v)
        self._bl_br_stencil(v, self._yppm._al, self._bl, self._br)
        if self.grid.sw_corner:
            self._zero_bl_br_sw_corner_stencil(self._bl, self._br)
        if self.grid.nw_corner:
            self._zero_bl_br_nw_corner_stencil(self._bl, self._br)
        if self.grid.se_corner:
            self._zero_bl_br_se_corner_stencil(self._bl, self._br)
        if self.grid.ne_corner:
            self._zero_bl_br_ne_corner_stencil(self._bl, self._br)
        self.stencil(c, v, self._bl, self._br, flux, self.rdy)
