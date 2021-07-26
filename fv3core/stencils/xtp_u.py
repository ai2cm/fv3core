import dace
from gt4py import gtscript
from gt4py.gtscript import __INLINED, PARALLEL, computation, interval

import fv3core._config as spec
import fv3core.utils.gt4py_utils as utils
from fv3core.decorators import FrozenStencil, computepath_method
from fv3core.stencils import xppm
from fv3core.utils.typing import FloatField, FloatFieldIJ


@gtscript.function
def _get_flux(
    u: FloatField,
    courant: FloatField,
    rdx: FloatFieldIJ,
    bl: FloatField,
    br: FloatField,
):
    """
    Compute the x-dir flux of kinetic energy(?).

    Inputs:
        u: x-dir wind
        courant: Courant number in flux form
        rdx: 1.0 / dx
        bl: ???
        br: ???

    Returns:
        flux: Kinetic energy flux
    """
    # Could try merging this with xppm version.

    from __externals__ import iord

    b0 = bl + br
    cfl = courant * rdx[-1, 0] if courant > 0 else courant * rdx
    fx0 = xppm.fx1_fn(cfl, br, b0, bl)

    if __INLINED(iord < 8):
        tmp = xppm.get_tmp(bl, b0, br)
    else:
        tmp = 1.0
    return xppm.final_flux(courant, u, fx0, tmp)


def bl_br_main(u: FloatField, al: FloatField, bl: FloatField, br: FloatField):
    with computation(PARALLEL), interval(...):
        bl = al[0, 0, 0] - u[0, 0, 0]
        br = al[1, 0, 0] - u[0, 0, 0]


def xtp_flux(
    courant: FloatField,
    u: FloatField,
    bl: FloatField,
    br: FloatField,
    flux: FloatField,
    rdx: FloatFieldIJ,
):
    with computation(PARALLEL), interval(...):
        flux = _get_flux(u, courant, rdx, bl, br)


def zero_br_bl(br: FloatField, bl: FloatField):
    with computation(PARALLEL), interval(...):
        br = 0.0
        bl = 0.0


class XTP_U:
    def __init__(self, namelist):
        iord = spec.namelist.hord_mt
        assert iord < 8
        if iord not in (5, 6, 7, 8):
            raise NotImplementedError(
                "Currently xtp_v is only supported for hord_mt == 5,6,7,8"
            )
        assert namelist.grid_type < 3
        self.grid = spec.grid
        self.rdx = self.grid.rdx
        assert namelist.grid_type < 3
        shape = self.grid.domain_shape_full(add=(1, 1, 1))
        self._bl = utils.make_storage_from_shape(shape)
        self._br = utils.make_storage_from_shape(shape)
        self._bl_br_stencil = FrozenStencil(
            bl_br_main,
            origin=(self.grid.is_ - 1, self.grid.js, 0),
            domain=(self.grid.nic + 2, self.grid.njc + 1, self.grid.npz),
        )
        corner_domain = (2, 1, self.grid.npz)
        if self.grid.sw_corner:
            self._zero_bl_br_sw_corner_stencil = FrozenStencil(
                zero_br_bl,
                origin=(self.grid.is_ - 1, self.grid.js, 0),
                domain=corner_domain,
            )
        if self.grid.nw_corner:
            self._zero_bl_br_nw_corner_stencil = FrozenStencil(
                zero_br_bl,
                origin=(self.grid.is_ - 1, self.grid.je + 1, 0),
                domain=corner_domain,
            )
        if self.grid.se_corner:
            self._zero_bl_br_se_corner_stencil = FrozenStencil(
                zero_br_bl, origin=(self.grid.ie, self.grid.js, 0), domain=corner_domain
            )
        if self.grid.ne_corner:
            self._zero_bl_br_ne_corner_stencil = FrozenStencil(
                zero_br_bl,
                origin=(self.grid.ie, self.grid.je + 1, 0),
                domain=corner_domain,
            )
        self._xtp_flux = FrozenStencil(
            xtp_flux,
            externals={
                "iord": iord,
                "mord": iord,
            },
            origin=self.grid.compute_origin(),
            domain=self.grid.domain_shape_compute(add=(1, 1, 0)),
        )
        self._xppm = xppm.XPiecewiseParabolic(
            namelist, iord, self.grid.js, self.grid.je + 1, self.grid.dx
        )

    @computepath_method
    def __call__(self, c, u, flux):
        """
        Compute flux of kinetic energy in x-dir.

        Args:
            c (in): Courant number in flux form
            u (in): x-dir wind on D-grid
            flux (out): Flux of kinetic energy
        """

        self._xppm.compute_al(u)
        self._bl_br_stencil(u, self._xppm._al, self._bl, self._br)
        if self.grid.sw_corner:
            self._zero_bl_br_sw_corner_stencil(self._bl, self._br)
        if self.grid.nw_corner:
            self._zero_bl_br_nw_corner_stencil(self._bl, self._br)
        if self.grid.se_corner:
            self._zero_bl_br_se_corner_stencil(self._bl, self._br)
        if self.grid.ne_corner:
            self._zero_bl_br_ne_corner_stencil(self._bl, self._br)

        self._xtp_flux(
            c,
            u,
            self._bl,
            self._br,
            flux,
            self.rdx,
        )
