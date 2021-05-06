from typing import Optional

import gt4py.gtscript as gtscript
from gt4py.gtscript import PARALLEL, computation, interval

import fv3core._config as spec
import fv3core.stencils.basic_operations as basic
import fv3core.utils.corners as corners
import fv3core.utils.gt4py_utils as utils
from fv3core.decorators import gtstencil
from fv3core.stencils.a2b_ord4 import AGrid2BGridFourthOrder
from fv3core.stencils.basic_operations import copy_stencil
from fv3core.utils.typing import FloatField, FloatFieldIJ, FloatFieldK
import fv3core.stencils.d_sw as d_sw
from types import SimpleNamespace 
@gtstencil
def ptc_main(
    u: FloatField,
    va: FloatField,
    cosa_v: FloatFieldIJ,
    sina_v: FloatFieldIJ,
    dyc: FloatFieldIJ,
    ptc: FloatField,
):
    with computation(PARALLEL), interval(...):
        ptc = (u - 0.5 * (va[0, -1, 0] + va) * cosa_v) * dyc * sina_v


@gtstencil
def ptc_y_edge(
    u: FloatField,
    vc: FloatField,
    dyc: FloatFieldIJ,
    sin_sg4: FloatFieldIJ,
    sin_sg2: FloatFieldIJ,
    ptc: FloatField,
):
    with computation(PARALLEL), interval(...):
        ptc = u * dyc * sin_sg4[0, -1] if vc > 0 else u * dyc * sin_sg2


@gtstencil
def vorticity_main(
    v: FloatField,
    ua: FloatField,
    cosa_u: FloatFieldIJ,
    sina_u: FloatFieldIJ,
    dxc: FloatFieldIJ,
    vort: FloatField,
):
    with computation(PARALLEL), interval(...):
        vort = (v - 0.5 * (ua[-1, 0, 0] + ua) * cosa_u) * dxc * sina_u


@gtstencil
def vorticity_x_edge(
    v: FloatField,
    uc: FloatField,
    dxc: FloatFieldIJ,
    sin_sg3: FloatFieldIJ,
    sin_sg1: FloatFieldIJ,
    vort: FloatField,
):
    with computation(PARALLEL), interval(...):
        vort = v * dxc * sin_sg3[-1, 0] if uc > 0 else v * dxc * sin_sg1


@gtstencil
def delpc_main(vort: FloatField, ptc: FloatField, delpc: FloatField):
    with computation(PARALLEL), interval(...):
        delpc = vort[0, -1, 0] - vort + ptc[-1, 0, 0] - ptc


@gtstencil
def corner_south_remove_extra_term(vort: FloatField, delpc: FloatField):
    with computation(PARALLEL), interval(...):
        delpc -= vort[0, -1, 0]


@gtstencil
def corner_north_remove_extra_term(vort: FloatField, delpc: FloatField):
    with computation(PARALLEL), interval(...):
        delpc += vort


@gtscript.function
def damp_tmp(q, da_min_c, d2_bg, dddmp):
    tmpddd = dddmp * q
    mintmp = 0.2 if 0.2 < tmpddd else tmpddd
    maxd2 = d2_bg if d2_bg > mintmp else mintmp
    damp = da_min_c * maxd2
    return damp


@gtstencil
def damping_nord0_stencil(
    rarea_c: FloatFieldIJ,
    delpc: FloatField,
    vort: FloatField,
    ke: FloatField,
    d2_bg: FloatFieldK,
    da_min_c: float,
    dddmp: float,
    dt: float,
):
    with computation(PARALLEL), interval(...):
        delpc = rarea_c * delpc
        delpcdt = delpc * dt
        absdelpcdt = delpcdt if delpcdt >= 0 else -delpcdt
        damp = damp_tmp(absdelpcdt, da_min_c, d2_bg, dddmp)
        vort = damp * delpc
        ke += vort


@gtstencil
def damping_nord_highorder_stencil(
    vort: FloatField,
    ke: FloatField,
    delpc: FloatField,
    divg_d: FloatField,
    da_min_c: float,
    d2_bg: float,
    dddmp: float,
    dd8: float,
):
    with computation(PARALLEL), interval(...):
        damp = damp_tmp(vort, da_min_c, d2_bg, dddmp)
        vort = damp * delpc + dd8 * divg_d
        ke = ke + vort


@gtstencil
def vc_from_divg(divg_d: FloatField, divg_u: FloatFieldIJ, vc: FloatField):
    with computation(PARALLEL), interval(...):
        vc[0, 0, 0] = (divg_d[1, 0, 0] - divg_d) * divg_u


@gtstencil
def uc_from_divg(divg_d: FloatField, divg_v: FloatFieldIJ, uc: FloatField):
    with computation(PARALLEL), interval(...):
        uc[0, 0, 0] = (divg_d[0, 1, 0] - divg_d) * divg_v


@gtstencil
def redo_divg_d(uc: FloatField, vc: FloatField, divg_d: FloatField):
    with computation(PARALLEL), interval(...):
        divg_d[0, 0, 0] = uc[0, -1, 0] - uc + vc[-1, 0, 0] - vc


@gtstencil
def smagorinksy_diffusion_approx(delpc: FloatField, vort: FloatField, absdt: float):
    with computation(PARALLEL), interval(...):
        vort = absdt * (delpc ** 2.0 + vort ** 2.0) ** 0.5


def vorticity_calc( wk, vort, delpc, dt, nord, kstart, nk, dddmp, grid_type):
    if nord != 0:
        if dddmp < 1e-5:
            vort[:, :, kstart : kstart + nk] = 0
        else:
            if grid_type < 3:
                a2b = utils.cached_stencil_class(AGrid2BGridFourthOrder)(
                    grid_type,
                    kstart,
                    nk,
                    replace=False,
                    cache_key="a2bdd-" + str(kstart) + "-" + str(nk),
                )

                a2b(wk, vort)
                smagorinksy_diffusion_approx(
                    delpc,
                    vort,
                    abs(dt),
                    origin=(spec.grid.is_, spec.grid.js, kstart),
                    domain=(spec.grid.nic + 1, spec.grid.njc + 1, nk),
                )
            else:
                raise Exception("Not implemented, smag_corner")

class DivergenceDamping:
    """
     A large section in Fortran's d_sw that applies divergence damping
    """
    def __init__(self, namelist: SimpleNamespace, nord_col: FloatFieldK, d2_bg: FloatFieldK):
        self.grid = spec.grid
        assert not self.grid.nested, "nested not implemented"
        self._dddmp = namelist.dddmp
        self._d4_bg = namelist.d4_bg
        self._grid_type = namelist.grid_type
        self._nord_column = nord_col
        self._d2_bg_column = d2_bg
        self._nonzero_nord_k = 0
        self._nonzero_nord = int(namelist.nord)
        for k in range(len(self._nord_column)):
            if self._nord_column[k] > 0:
                self._nonzero_nord_k = k
                self._nonzero_nord = int(self._nord_column[k])
                break
            
    def __call__(
        self, 
        u: FloatField,
        v: FloatField,
        va: FloatField,
        ptc: FloatField,
        vort: FloatField,
        ua: FloatField,
        divg_d: FloatField,
        vc: FloatField,
        uc: FloatField,
        delpc: FloatField,
        ke: FloatField,
        wk: FloatField,
        dt: float
    ) -> None:

        is2 = self.grid.is_ + 1 if self.grid.west_edge else self.grid.is_
        ie1 = self.grid.ie if self.grid.east_edge else self.grid.ie + 1
        self.damping_zero_order(
            u, v, va, ptc, vort, ua, vc, uc, delpc, ke, self._d2_bg_column, dt, is2, ie1, 0, self._nonzero_nord_k
        )
        kstart = self._nonzero_nord_k
        nk = self.grid.npz - self._nonzero_nord_k
        copy_stencil(
            divg_d,
            delpc,
            origin=(self.grid.is_, self.grid.js, kstart),
            domain=(self.grid.nic + 1, self.grid.njc + 1, nk),
        )
        for n in range(1, self._nonzero_nord + 1):
            nt = self._nonzero_nord - n
            nint = self.grid.nic + 2 * nt + 1
            njnt = self.grid.njc + 2 * nt + 1
            js = self.grid.js - nt
            is_ = self.grid.is_ - nt
            fillc = (
                (n != self._nonzero_nord)
                and self._grid_type < 3
                and (
                    self.grid.sw_corner or self.grid.se_corner or self.grid.ne_corner or self.grid.nw_corner
                )
            )
            if fillc:
                corners.fill_corners_bgrid_x(
                    divg_d,
                    origin=(self.grid.isd, self.grid.jsd, kstart),
                    domain=(self.grid.nid + 1, self.grid.njd + 1, nk),
                )
            vc_from_divg(
                divg_d,
                self.grid.divg_u,
                vc,
                origin=(is_ - 1, js, kstart),
                domain=(nint + 1, njnt, nk),
            )
            if fillc:
                corners.fill_corners_bgrid_y(
                    divg_d,
                    origin=(self.grid.isd, self.grid.jsd, kstart),
                    domain=(self.grid.nid + 1, self.grid.njd + 1, nk),
                )
            uc_from_divg(
                divg_d,
                self.grid.divg_v,
                uc,
                origin=(is_, js - 1, kstart),
                domain=(nint, njnt + 1, nk),
            )
            if fillc:
                corners.fill_corners_dgrid(
                    vc,
                    uc,
                    -1.0,
                    origin=(self.grid.isd, self.grid.jsd, kstart),
                    domain=(self.grid.nid + 1, self.grid.njd + 1, nk),
                )

            redo_divg_d(
                uc, vc, divg_d, origin=(is_, js, kstart), domain=(nint, njnt, nk)
            )
            corner_domain = (1, 1, nk)
            if self.grid.sw_corner:
                corner_south_remove_extra_term(
                    uc, divg_d, origin=(self.grid.is_, self.grid.js, kstart), domain=corner_domain
                )
            if self.grid.se_corner:
                corner_south_remove_extra_term(
                    uc,
                    divg_d,
                    origin=(self.grid.ie + 1, self.grid.js, kstart),
                    domain=corner_domain,
                )
            if self.grid.ne_corner:
                corner_north_remove_extra_term(
                    uc,
                    divg_d,
                    origin=(self.grid.ie + 1, self.grid.je + 1, kstart),
                    domain=corner_domain,
                )
            if self.grid.nw_corner:
                corner_north_remove_extra_term(
                    uc,
                    divg_d,
                    origin=(self.grid.is_, self.grid.je + 1, kstart),
                    domain=corner_domain,
                )
            if not self.grid.stretched_grid:
                basic.adjustmentfactor_stencil(
                    self.grid.rarea_c,
                    divg_d,
                    origin=(is_, js, kstart),
                    domain=(nint, njnt, nk),
                )

        vorticity_calc(wk, vort, delpc, dt, self._nonzero_nord, kstart, nk, self._dddmp, self._grid_type)
        if self.grid.stretched_grid:
            dd8 = self.grid.da_min * self._d4_bg ** (self._nonzero_nord + 1)
        else:
            dd8 = (self.grid.da_min_c * self._d4_bg) ** (self._nonzero_nord + 1)
        damping_nord_highorder_stencil(
            vort,
            ke,
            delpc,
            divg_d,
            self.grid.da_min_c,
            self._d2_bg_column[kstart],
            self._dddmp,
            dd8,
            origin=(self.grid.is_, self.grid.js, kstart),
            domain=(self.grid.nic + 1, self.grid.njc + 1, nk),
        )


    def damping_zero_order(
        self,
        u: FloatField,
        v: FloatField,
        va: FloatField,
        ptc: FloatField,
        vort: FloatField,
        ua: FloatField,
        vc: FloatField,
        uc: FloatField,
        delpc: FloatField,
        ke: FloatField,
        d2_bg: FloatFieldK,
        dt: float,
        is2: int,
        ie1: int,
        kstart: int,
        nk: int,
    ) -> None:
        # if nested
        # TODO: ptc and vort are equivalent, but x vs y, consolidate if possible.
        ptc_main(
            u,
            va,
            self.grid.cosa_v,
            self.grid.sina_v,
            self.grid.dyc,
            ptc,
            origin=(self.grid.is_ - 1, self.grid.js, kstart),
            domain=(self.grid.nic + 2, self.grid.njc + 1, nk),
        )
        y_edge_domain = (self.grid.nic + 2, 1, nk)
        if self.grid.south_edge:
            ptc_y_edge(
                u,
                vc,
                self.grid.dyc,
                self.grid.sin_sg4,
                self.grid.sin_sg2,
                ptc,
                origin=(self.grid.is_ - 1, self.grid.js, kstart),
                domain=y_edge_domain,
            )
        if self.grid.north_edge:
            ptc_y_edge(
                u,
                vc,
                self.grid.dyc,
                self.grid.sin_sg4,
                self.grid.sin_sg2,
                ptc,
                origin=(self.grid.is_ - 1, self.grid.je + 1, kstart),
                domain=y_edge_domain,
            )

        vorticity_main(
            v,
            ua,
            self.grid.cosa_u,
            self.grid.sina_u,
            self.grid.dxc,
            vort,
            origin=(is2, self.grid.js - 1, kstart),
            domain=(ie1 - is2 + 1, self.grid.njc + 2, nk),
        )
        x_edge_domain = (1, self.grid.njc + 2, nk)
        if self.grid.west_edge:
            vorticity_x_edge(
                v,
                uc,
                self.grid.dxc,
                self.grid.sin_sg3,
                self.grid.sin_sg1,
                vort,
                origin=(self.grid.is_, self.grid.js - 1, kstart),
                domain=x_edge_domain,
            )
        if self.grid.east_edge:
            vorticity_x_edge(
                v,
                uc,
                self.grid.dxc,
                self.grid.sin_sg3,
                self.grid.sin_sg1,
                vort,
                origin=(self.grid.ie + 1, self.grid.js - 1, kstart),
                domain=x_edge_domain,
            )
        # end if nested
        compute_origin = (self.grid.is_, self.grid.js, kstart)
        compute_domain = (self.grid.nic + 1, self.grid.njc + 1, nk)
        delpc_main(vort, ptc, delpc, origin=compute_origin, domain=compute_domain)
        corner_domain = (1, 1, nk)
        if self.grid.sw_corner:
            corner_south_remove_extra_term(
                vort, delpc, origin=(self.grid.is_, self.grid.js, kstart), domain=corner_domain
            )
        if self.grid.se_corner:
            corner_south_remove_extra_term(
                vort, delpc, origin=(self.grid.ie + 1, self.grid.js, kstart), domain=corner_domain
            )
        if self.grid.ne_corner:
            corner_north_remove_extra_term(
                vort, delpc, origin=(self.grid.ie + 1, self.grid.je + 1, kstart), domain=corner_domain
            )
        if self.grid.nw_corner:
            corner_north_remove_extra_term(
                vort, delpc, origin=(self.grid.is_, self.grid.je + 1, kstart), domain=corner_domain
            )

        damping_nord0_stencil(
            self.grid.rarea_c,
            delpc,
            vort,
            ke,
            d2_bg,
            self.grid.da_min_c,
            self._dddmp,
            dt,
            origin=compute_origin,
            domain=compute_domain,
        )
