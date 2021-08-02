import gt4py.gtscript as gtscript
from gt4py.gtscript import __INLINED, PARALLEL, computation, interval

import fv3core._config as spec
import fv3core.stencils.delnflux as delnflux
import fv3core.utils.corners as corners
import fv3core.utils.global_constants as constants
import fv3core.utils.gt4py_utils as utils
from fv3core.decorators import FrozenStencil, computepath_method
from fv3core.stencils.delnflux import DelnFluxNoSG
from fv3core.stencils.divergence_damping import DivergenceDamping
from fv3core.stencils.fvtp2d import FiniteVolumeTransport
from fv3core.stencils.fxadv import FiniteVolumeFluxPrep
from fv3core.stencils.xtp_u import XTP_U
from fv3core.stencils.ytp_v import YTP_V
from fv3core.utils.grid import axis_offsets
from fv3core.utils.typing import FloatField, FloatFieldIJ, FloatFieldK


dcon_threshold = 1e-5

# NOTE leaving the refrence to spec.grid here on purpose
# k_bounds should be refactored out of existence
def k_bounds():
    # UpdatedzD needs to go one k level higher than D_SW, to the buffer point that
    # usually isn't used. To reuse the same 'column_namelist' and remove the
    # specification of 'kstart' and 'nk in many methods, we just make all of the
    # column namelist calculations go to the top of the array
    return [[0, 1], [1, 1], [2, 1], [3, spec.grid.npz - 2]]


@gtscript.function
def flux_component(gx, gy, rarea):
    return (gx - gx[1, 0, 0] + gy - gy[0, 1, 0]) * rarea


@gtscript.function
def flux_integral(w, delp, gx, gy, rarea):
    return w * delp + flux_component(gx, gy, rarea)


def flux_adjust(
    w: FloatField, delp: FloatField, gx: FloatField, gy: FloatField, rarea: FloatFieldIJ
):
    with computation(PARALLEL), interval(...):
        w = flux_integral(w, delp, gx, gy, rarea)


def flux_capacitor(
    cx: FloatField,
    cy: FloatField,
    xflux: FloatField,
    yflux: FloatField,
    crx_adv: FloatField,
    cry_adv: FloatField,
    fx: FloatField,
    fy: FloatField,
):
    """Accumulates the flux capacitor and courant number variables
    Saves the mass fluxes to the "flux capacitor" variables for tracer transport
    Also updates the accumulated courant numbers
    Args:
        cx: accumulated courant number in the x direction (inout)
        cy: accumulated courant number in the y direction (inout)
        xflux: flux capacitor in the x direction, accumlated mass flux (inout)
        yflux: flux capacitor in the y direction, accumlated mass flux (inout)
        crx_adv: local courant numver, dt*ut/dx  (in)
        cry_adv: local courant number dt*vt/dy (in)
        fx: 1-D x-direction flux (in)
        fy: 1-D y-direction flux (in)
    """
    with computation(PARALLEL), interval(...):
        cx = cx + crx_adv
        cy = cy + cry_adv
        xflux = xflux + fx
        yflux = yflux + fy


@gtscript.function
def horizontal_relative_vorticity_from_winds(u, v, ut, vt, dx, dy, rarea, vorticity):
    """
    Compute the area mean relative vorticity in the z-direction from the D-grid winds.

    Args:
        u (in): x-direction wind on D grid
        v (in): y-direction wind on D grid
        ut (out): u * dx
        vt (out): v * dy
        dx (in): gridcell width in x-direction
        dy (in): gridcell width in y-direction
        rarea (in): inverse of area
        vorticity (out): area mean horizontal relative vorticity
    """

    vt = u * dx
    ut = v * dy
    vorticity = rarea * (vt - vt[0, 1, 0] - ut + ut[1, 0, 0])

    return vt, ut, vorticity


def compute_temperature_and_pressure_delta(
    gx: FloatField,
    gy: FloatField,
    rarea: FloatFieldIJ,
    fx: FloatField,
    fy: FloatField,
    pt: FloatField,
    delp: FloatField,
):
    from __externals__ import inline_q

    with computation(PARALLEL), interval(...):
        if __INLINED(inline_q == 0):
            pt = flux_integral(pt, delp, gx, gy, rarea)
            delp = delp + flux_component(fx, fy, rarea)
            pt = pt / delp


def ub_from_vort(
    vort: FloatField,
    ub: FloatField,
    dcon: FloatFieldK,
):
    with computation(PARALLEL), interval(...):
        if dcon[0] > dcon_threshold:
            ub = vort - vort[1, 0, 0]


def vb_from_vort(
    vort: FloatField,
    vb: FloatField,
    dcon: FloatFieldK,
):
    with computation(PARALLEL), interval(...):
        if dcon[0] > dcon_threshold:
            vb = vort - vort[0, 1, 0]


def u_from_ke(
    ke: FloatField,
    vt: FloatField,
    fy: FloatField,
    u: FloatField,
):
    with computation(PARALLEL), interval(...):
        u = vt + ke - ke[1, 0, 0] + fy


def v_from_ke(
    ke: FloatField,
    ut: FloatField,
    fx: FloatField,
    v: FloatField,
):
    with computation(PARALLEL), interval(...):
        v = ut + ke - ke[0, 1, 0] - fx


# TODO: This is untested and the radius may be incorrect
@gtscript.function
def coriolis_force_correction(zh, radius):
    return 1.0 + (zh + zh[0, 0, 1]) / radius


def compute_vorticity(
    wk: FloatField,
    f0: FloatFieldIJ,
    zh: FloatField,
    vort: FloatField,
):

    from __externals__ import do_f3d, hydrostatic, radius

    with computation(PARALLEL), interval(...):
        if __INLINED(do_f3d and not hydrostatic):
            z_rat = coriolis_force_correction(zh, radius)
            vort = wk + f0 * z_rat
        else:
            vort = wk[0, 0, 0] + f0[0, 0]


def adjust_w_and_qcon(
    w: FloatField,
    delp: FloatField,
    dw: FloatField,
    q_con: FloatField,
    damp_w: FloatFieldK,
):
    with computation(PARALLEL), interval(...):
        w = w / delp
        w = w + dw if damp_w > 1e-5 else w
        # Fortran: #ifdef USE_COND
        q_con = q_con / delp


@gtscript.function
def heat_damping_term(ub, vb, gx, gy, rsin2, cosa_s, u2, v2, du2, dv2):
    return rsin2 * (
        (ub * ub + ub[0, 1, 0] * ub[0, 1, 0] + vb * vb + vb[1, 0, 0] * vb[1, 0, 0])
        + 2.0 * (gy + gy[0, 1, 0] + gx + gx[1, 0, 0])
        - cosa_s * (u2 * dv2 + v2 * du2 + du2 * dv2)
    )


def heat_diss(
    fx2: FloatField,
    fy2: FloatField,
    w: FloatField,
    rarea: FloatFieldIJ,
    heat_source: FloatField,
    diss_est: FloatField,
    dw: FloatField,
    damp_w: FloatFieldK,
    ke_bg: FloatFieldK,
    dt: float,
):
    with computation(PARALLEL), interval(...):
        diss_e = diss_est
        if damp_w > 1e-5:
            dd8 = ke_bg * abs(dt)
            dw = (fx2 - fx2[1, 0, 0] + fy2 - fy2[0, 1, 0]) * rarea
            heat_source = dd8 - dw * (w + 0.5 * dw)
            diss_est = diss_e + heat_source


def heat_source_from_vorticity_damping(
    ub: FloatField,
    vb: FloatField,
    ut: FloatField,
    vt: FloatField,
    u: FloatField,
    v: FloatField,
    delp: FloatField,
    rsin2: FloatFieldIJ,
    cosa_s: FloatFieldIJ,
    rdx: FloatFieldIJ,
    rdy: FloatFieldIJ,
    heat_source: FloatField,
    dampterm: FloatField,
    kinetic_energy_fraction_to_damp: FloatFieldK,
):

    from __externals__ import do_skeb

    with computation(PARALLEL), interval(...):
        # if (kinetic_energy_fraction_to_damp[0] > dcon_threshold) or do_skeb:
        heat_s = heat_source
        ubt = (ub + vt) * rdx
        fy = u * rdx
        gy = fy * ubt
        vbt = (vb - ut) * rdy
        fx = v * rdy
        gx = fx * vbt
    with computation(PARALLEL), interval(...):
        if (kinetic_energy_fraction_to_damp[0] > dcon_threshold) or do_skeb:
            u2 = fy + fy[0, 1, 0]
            du2 = ubt + ubt[0, 1, 0]
            v2 = fx + fx[1, 0, 0]
            dv2 = vbt + vbt[1, 0, 0]
            dampterm = heat_damping_term(
                ubt, vbt, gx, gy, rsin2, cosa_s, u2, v2, du2, dv2
            )
            heat_source = delp * (
                heat_s - 0.25 * kinetic_energy_fraction_to_damp[0] * dampterm
            )


def heat_source_accumulate(
    heat_source: FloatField,
    heat_source_total: FloatField,
    dissipation_estimate: FloatField,
    dampterm: FloatField,
):
    from __externals__ import d_con, do_skeb

    with computation(PARALLEL), interval(...):
        diss_e = dissipation_estimate
        if __INLINED((d_con > dcon_threshold) or do_skeb):
            heat_source_total = heat_source_total + heat_source
            if __INLINED(do_skeb == 1):
                dissipation_estimate = diss_e - dampterm


def damped_u(
    vt: FloatField,
    u: FloatField,
    damp_vt: FloatFieldK,
):
    with computation(PARALLEL), interval(...):
        if damp_vt > 1e-5:
            u = u + vt


def damped_v(
    ut: FloatField,
    v: FloatField,
    damp_vt: FloatFieldK,
):
    with computation(PARALLEL), interval(...):
        if damp_vt > 1e-5:
            v = v - ut


def ke_from_bwind(
    ke: FloatField,
    ub: FloatField,
    vb: FloatField,
):
    with computation(PARALLEL), interval(...):
        ke = 0.5 * (ke + ub * vb)


def horizontal_vorticity(
    u: FloatField,
    v: FloatField,
    ut: FloatField,
    vt: FloatField,
    dx: FloatFieldIJ,
    dy: FloatFieldIJ,
    rarea: FloatFieldIJ,
    vorticity: FloatField,
):
    with computation(PARALLEL), interval(...):
        vt, ut, vorticity = horizontal_relative_vorticity_from_winds(
            u, v, ut, vt, dx, dy, rarea, vorticity
        )


# Set the unique parameters for the smallest
# k-values, e.g. k = 0, 1, 2 when generating
# the column namelist
def set_low_kvals(col, k):
    for name in ["nord", "nord_w", "d_con"]:
        col[name][k] = 0
    col["damp_w"][k] = col["d2_divg"][k]


# For the column namelist at a specific k-level
# set the vorticity parameters if do_vort_damp is true
def vorticity_damping_option(column, k, do_vort_damp):
    if do_vort_damp:
        column["nord_v"][k] = 0
        column["damp_vt"][k] = 0.5 * column["d2_divg"][k]


def lowest_kvals(column, k, do_vort_damp):
    set_low_kvals(column, k)
    vorticity_damping_option(column, k, do_vort_damp)


def get_column_namelist(namelist, npz):
    """
    Generate a dictionary of columns that specify how parameters (such as nord, damp)
    used in several functions called by D_SW vary over the k-dimension.

    In a near-future PR, the need for this will disappear as we refactor
    individual modules to apply this parameter variation explicitly in the
    stencils themselves. If it doesn't, we should compute it only in the init phase.
    The unique set of all column parameters is specified by k_bounds. For each k range
    as specified by (kstart, nk) this sets what several different parameters are.
    It previously was a dictionary with the k value as the key, the value being another
    dictionary of values, but this did not work when we removed the k loop from some
    modules and instead wanted to push the whole column ingestion down a level.
    """
    direct_namelist = ["ke_bg", "d_con", "nord"]
    all_names = direct_namelist + [
        "nord_v",
        "nord_w",
        "nord_t",
        "damp_vt",
        "damp_w",
        "damp_t",
        "d2_divg",
    ]
    col = {}
    for name in all_names:
        col[name] = utils.make_storage_from_shape((npz + 1,), (0,))
    for name in direct_namelist:
        col[name][:] = getattr(namelist, name)

    col["d2_divg"][:] = min(0.2, namelist.d2_bg)
    col["nord_v"][:] = min(2, col["nord"][0])
    col["nord_w"][:] = col["nord_v"][0]
    col["nord_t"][:] = col["nord_v"][0]
    if namelist.do_vort_damp:
        col["damp_vt"][:] = namelist.vtdm4
    else:
        col["damp_vt"][:] = 0
    col["damp_w"][:] = col["damp_vt"][0]
    col["damp_t"][:] = col["damp_vt"][0]
    if npz == 1 or namelist.n_sponge < 0:
        col["d2_divg"][0] = namelist.d2_bg
    else:
        col["d2_divg"][0] = max(0.01, namelist.d2_bg, namelist.d2_bg_k1)
        lowest_kvals(col, 0, namelist.do_vort_damp)
        if namelist.d2_bg_k2 > 0.01:
            col["d2_divg"][1] = max(namelist.d2_bg, namelist.d2_bg_k2)
            lowest_kvals(col, 1, namelist.do_vort_damp)
        if namelist.d2_bg_k2 > 0.05:
            col["d2_divg"][2] = max(namelist.d2_bg, 0.2 * namelist.d2_bg_k2)
            set_low_kvals(col, 2)
    return col


def kinetic_energy(
    ub: FloatField,
    vb: FloatField,
    ke: FloatField,
):
    with computation(PARALLEL), interval(...):
        ke = vb * ub


def main_vb(
    vc: FloatField,
    uc: FloatField,
    cosa: FloatFieldIJ,
    rsina: FloatFieldIJ,
    vb: FloatField,
    dt5: float,
):
    with computation(PARALLEL), interval(...):
        vb[0, 0, 0] = dt5 * (vc[-1, 0, 0] + vc - (uc[0, -1, 0] + uc) * cosa) * rsina


def vb_y_edge(vt: FloatField, vb: FloatField, dt5: float):
    with computation(PARALLEL), interval(...):
        vb[0, 0, 0] = dt5 * (vt[-1, 0, 0] + vt)


def vb_x_edge(vt: FloatField, vb: FloatField, dt4: float):
    with computation(PARALLEL), interval(...):
        vb[0, 0, 0] = dt4 * (-vt[-2, 0, 0] + 3.0 * (vt[-1, 0, 0] + vt) - vt[1, 0, 0])


def main_ub(
    uc: FloatField,
    vc: FloatField,
    cosa: FloatFieldIJ,
    rsina: FloatFieldIJ,
    ub: FloatField,
    dt5: float,
):
    with computation(PARALLEL), interval(...):
        ub[0, 0, 0] = dt5 * (uc[0, -1, 0] + uc - (vc[-1, 0, 0] + vc) * cosa) * rsina


def ub_x_edge(ut: FloatField, ub: FloatField, dt5: float):
    with computation(PARALLEL), interval(...):
        ub[0, 0, 0] = dt5 * (ut[0, -1, 0] + ut)


def ub_y_edge(ut: FloatField, ub: FloatField, dt4: float):
    with computation(PARALLEL), interval(...):
        ub[0, 0, 0] = dt4 * (-ut[0, -2, 0] + 3.0 * (ut[0, -1, 0] + ut) - ut[0, 1, 0])


class DGridShallowWaterLagrangianDynamics:
    """
    Fortran name is the d_sw subroutine
    """

    def __init__(self, namelist, column_namelist):
        self.grid = spec.grid
        assert (
            namelist.grid_type < 3
        ), "ubke and vbke only implemented for grid_type < 3"
        assert not namelist.inline_q, "inline_q not yet implemented"
        assert (
            namelist.d_ext <= 0
        ), "untested d_ext > 0. need to call a2b_ord2, not yet implemented"
        assert (column_namelist["damp_vt"] > dcon_threshold).all()
        # TODO: in theory, we should check if damp_vt > 1e-5 for each k-level and
        # only compute delnflux for k-levels where this is true
        assert (column_namelist["damp_w"] > dcon_threshold).all()
        # TODO: in theory, we should check if damp_w > 1e-5 for each k-level and
        # only compute delnflux for k-levels where this is true

        # only compute for k-levels where this is true
        shape = self.grid.domain_shape_full(add=(1, 1, 1))
        origin = self.grid.compute_origin()
        self.hydrostatic = namelist.hydrostatic
        self._tmp_heat_s = utils.make_storage_from_shape(shape, origin)
        self._tmp_ub = utils.make_storage_from_shape(shape, origin)
        self._tmp_vb = utils.make_storage_from_shape(shape, origin)
        self._tmp_ke = utils.make_storage_from_shape(shape, origin)
        self._tmp_vort = utils.make_storage_from_shape(shape, origin)
        self._tmp_ut = utils.make_storage_from_shape(shape, origin)
        self._tmp_vt = utils.make_storage_from_shape(shape, origin)
        self._tmp_fx = utils.make_storage_from_shape(shape, origin)
        self._tmp_fy = utils.make_storage_from_shape(shape, origin)
        self._tmp_gx = utils.make_storage_from_shape(shape, origin)
        self._tmp_gy = utils.make_storage_from_shape(shape, origin)
        self._tmp_dw = utils.make_storage_from_shape(shape, origin)
        self._tmp_wk = utils.make_storage_from_shape(shape, origin)
        self._tmp_fx2 = utils.make_storage_from_shape(shape, origin)
        self._tmp_fy2 = utils.make_storage_from_shape(shape, origin)
        self._tmp_damp_3d = utils.make_storage_from_shape((1, 1, self.grid.npz))
        self._tmp_dampterm = utils.make_storage_from_shape(shape, origin)
        self._column_namelist = column_namelist

        self.delnflux_nosg_w = DelnFluxNoSG(self._column_namelist["nord_w"])
        self.delnflux_nosg_v = DelnFluxNoSG(self._column_namelist["nord_v"])
        self.fvtp2d_dp = FiniteVolumeTransport(
            namelist,
            namelist.hord_dp,
            self._column_namelist["nord_v"],
            self._column_namelist["damp_vt"],
        )
        self.fvtp2d_dp_t = FiniteVolumeTransport(
            namelist,
            namelist.hord_dp,
            self._column_namelist["nord_t"],
            self._column_namelist["damp_t"],
        )
        self.fvtp2d_vt = FiniteVolumeTransport(
            namelist,
            namelist.hord_vt,
            self._column_namelist["nord_v"],
            self._column_namelist["damp_vt"],
        )
        self.fvtp2d_tm = FiniteVolumeTransport(
            namelist,
            namelist.hord_tm,
            self._column_namelist["nord_v"],
            self._column_namelist["damp_vt"],
        )
        self.fvtp2d_vt_nodelnflux = FiniteVolumeTransport(namelist, namelist.hord_vt)
        self.fv_prep = FiniteVolumeFluxPrep()
        self.ytp_v = YTP_V(namelist)
        self.xtp_u = XTP_U(namelist)
        self.divergence_damping = DivergenceDamping(
            namelist, column_namelist["nord"], column_namelist["d2_divg"]
        )
        full_origin = self.grid.full_origin()
        full_domain = self.grid.domain_shape_full()
        ax_offsets_full = axis_offsets(self.grid, full_origin, full_domain)
        b_origin = self.grid.compute_origin()
        b_domain = self.grid.domain_shape_compute(add=(1, 1, 0))
        ax_offsets_b = axis_offsets(self.grid, b_origin, b_domain)
        self._temperature_and_dp_stencil = FrozenStencil(
            compute_temperature_and_pressure_delta,
            externals={"inline_q": namelist.inline_q},
            origin=self.grid.compute_origin(),
            domain=self.grid.domain_shape_compute(),
        )
        js2 = self.grid.js + 1 if self.grid.south_edge else self.grid.js
        is2 = self.grid.is_ + 1 if self.grid.west_edge else self.grid.is_
        je1 = self.grid.je if self.grid.north_edge else self.grid.je + 1
        ie1 = self.grid.ie if self.grid.east_edge else self.grid.ie + 1
        jdiff = je1 - js2 + 1
        idiff = ie1 - is2 + 1
        domain_x = (1, jdiff, self.grid.npz)
        domain_y = (self.grid.nic + 1, 1, self.grid.npz)
        self._vb_stencil = FrozenStencil(
            main_vb, origin=(is2, js2, 0), domain=(idiff, jdiff, self.grid.npz)
        )
        if self.grid.south_edge:
            self._vb_south_edge_stencil = FrozenStencil(
                vb_y_edge, origin=self.grid.compute_origin(), domain=domain_y
            )
        if self.grid.west_edge:
            self._vb_west_edge_stencil = FrozenStencil(
                vb_x_edge, origin=(self.grid.is_, js2, 0), domain=domain_x
            )
        if self.grid.east_edge:
            self._vb_east_edge_stencil = FrozenStencil(
                vb_x_edge, origin=(self.grid.ie + 1, js2, 0), domain=domain_x
            )
        if self.grid.north_edge:
            self._vb_north_edge_stencil = FrozenStencil(
                vb_y_edge, origin=(self.grid.is_, self.grid.je + 1, 0), domain=domain_y
            )
        self._flux_adjust_stencil = FrozenStencil(
            flux_adjust,
            origin=self.grid.compute_origin(),
            domain=self.grid.domain_shape_compute(),
        )
        self._flux_capacitor_stencil = FrozenStencil(
            flux_capacitor, origin=full_origin, domain=full_domain
        )
        self._ub_from_vort_stencil = FrozenStencil(
            ub_from_vort,
            origin=self.grid.compute_origin(),
            domain=self.grid.domain_shape_compute(add=(0, 1, 0)),
        )
        self._vb_from_vort_stencil = FrozenStencil(
            vb_from_vort,
            origin=self.grid.compute_origin(),
            domain=self.grid.domain_shape_compute(add=(1, 0, 0)),
        )
        self._u_from_ke_stencil = FrozenStencil(
            u_from_ke,
            origin=self.grid.compute_origin(),
            domain=self.grid.domain_shape_compute(add=(0, 1, 0)),
        )
        self._v_from_ke_stencil = FrozenStencil(
            v_from_ke,
            origin=self.grid.compute_origin(),
            domain=self.grid.domain_shape_compute(add=(1, 0, 0)),
        )
        self._compute_vorticity_stencil = FrozenStencil(
            compute_vorticity,
            externals={
                "radius": constants.RADIUS,
                "do_f3d": namelist.do_f3d,
                "hydrostatic": self.hydrostatic,
            },
            origin=full_origin,
            domain=full_domain,
        )
        self._adjust_w_and_qcon_stencil = FrozenStencil(
            adjust_w_and_qcon,
            origin=self.grid.compute_origin(),
            domain=self.grid.domain_shape_compute(),
        )
        self._heat_diss_stencil = FrozenStencil(
            heat_diss,
            origin=self.grid.compute_origin(),
            domain=self.grid.domain_shape_compute(),
        )
        self._heat_source_from_vorticity_damping_stencil = FrozenStencil(
            heat_source_from_vorticity_damping,
            externals={
                "do_skeb": namelist.do_skeb,
            },
            origin=b_origin,
            domain=b_domain,
        )
        self._heat_source_accumulate = FrozenStencil(
            heat_source_accumulate,
            externals={
                "do_skeb": namelist.do_skeb,
                "d_con": namelist.d_con,
            },
            origin=self.grid.compute_origin(),
            domain=self.grid.domain_shape_compute(),
        )
        self._damp_u = FrozenStencil(
            damped_u,
            origin=self.grid.compute_origin(),
            domain=self.grid.domain_shape_compute(add=(0, 1, 0)),
        )
        self._damp_v = FrozenStencil(
            damped_v,
            origin=self.grid.compute_origin(),
            domain=self.grid.domain_shape_compute(add=(1, 0, 0)),
        )
        self._ke_from_bwind_stencil = FrozenStencil(
            ke_from_bwind,
            origin=full_origin,
            domain=full_domain,
        )
        corner_domain = (1, 1, self.grid.npz)
        if self.grid.sw_corner:
            self.ke_sw_corner_stencil = FrozenStencil(
                corners.corner_ke,
                origin=self.grid.compute_origin(),
                domain=corner_domain,
            )
        if self.grid.se_corner:
            self.ke_se_corner_stencil = FrozenStencil(
                corners.corner_ke,
                origin=(self.grid.ie + 1, self.grid.js, 0),
                domain=corner_domain,
            )
        if self.grid.ne_corner:
            self.ke_ne_corner_stencil = FrozenStencil(
                corners.corner_ke,
                origin=(self.grid.ie + 1, self.grid.je + 1, 0),
                domain=corner_domain,
            )
        if self.grid.nw_corner:
            self.ke_nw_corner_stencil = FrozenStencil(
                corners.corner_ke,
                origin=(self.grid.is_, self.grid.je + 1, 0),
                domain=corner_domain,
            )

        self._horizontal_vorticity_stencil = FrozenStencil(
            horizontal_vorticity,
            origin=full_origin,
            domain=full_domain,
        )
        self._ke_stencil = FrozenStencil(
            kinetic_energy, origin=b_origin, domain=b_domain
        )

        is2 = self.grid.is_ + 1 if self.grid.west_edge else self.grid.is_
        ie1 = self.grid.ie if self.grid.east_edge else self.grid.ie + 1
        idiff = ie1 - is2 + 1
        domain_y = (idiff, 1, self.grid.npz)
        domain_x = (1, self.grid.njc + 1, self.grid.npz)
        self._ub_stencil = FrozenStencil(
            main_ub,
            origin=(is2, self.grid.js, 0),
            domain=(idiff, self.grid.njc + 1, self.grid.npz),
        )
        if self.grid.south_edge:
            self._ub_south_edge_stencil = FrozenStencil(
                ub_y_edge, origin=(is2, self.grid.js, 0), domain=domain_y
            )
        if self.grid.west_edge:
            self._ub_west_edge_stencil = FrozenStencil(
                ub_x_edge, origin=(self.grid.is_, self.grid.js, 0), domain=domain_x
            )
        if self.grid.east_edge:
            self._ub_east_edge_stencil = FrozenStencil(
                ub_x_edge, origin=(self.grid.ie + 1, self.grid.js, 0), domain=domain_x
            )
        if self.grid.north_edge:
            self._ub_north_edge_stencil = FrozenStencil(
                ub_y_edge, origin=(is2, self.grid.je + 1, 0), domain=domain_y
            )
        self._damping_factor_calculation_stencil = FrozenStencil(
            delnflux.calc_damp, origin=(0, 0, 0), domain=(1, 1, self.grid.npz)
        )

        self._damping_factor_calculation_stencil(
            self._tmp_damp_3d,
            self._column_namelist["nord_v"],
            self._column_namelist["damp_vt"],
            self.grid.da_min_c,
        )
        self._delnflux_damp_vt = utils.make_storage_data(
            self._tmp_damp_3d[0, 0, :], (self.grid.npz,), (0,)
        )

        self._damping_factor_calculation_stencil(
            self._tmp_damp_3d,
            self._column_namelist["nord_w"],
            self._column_namelist["damp_w"],
            self.grid.da_min_c,
        )
        self._delnflux_damp_w = utils.make_storage_data(
            self._tmp_damp_3d[0, 0, :], (self.grid.npz,), (0,)
        )

    @computepath_method
    def __call__(
        self,
        delpc,
        delp,
        ptc,
        pt,
        u,
        v,
        w,
        uc,
        vc,
        ua,
        va,
        divgd,
        mfx,
        mfy,
        cx,
        cy,
        crx,
        cry,
        xfx,
        yfx,
        q_con,
        zh,
        heat_source,
        diss_est,
        dt,
    ):
        """D-Grid Shallow Water Routine
        Peforms a full-timestep advance of the D-grid winds and other
        prognostic variables using Lagrangian dynamics on the cubed-sphere.
        described by Lin 1997, Lin 2004 and Harris 2013.
        Args:
            delpc: C-grid  vertical delta in pressure (in)
            delp: D-grid vertical delta in pressure (inout),
            ptc: C-grid potential temperature (in)
            pt: D-grid potnetial teperature (inout)
            u: D-grid x-velocity (inout)
            v: D-grid y-velocity (inout)
            w: vertical velocity (inout)
            uc: C-grid x-velocity (in)
            vc: C-grid y-velocity (in)
            ua: A-grid x-velocity (in)
            va A-grid y-velocity(in)
            divgd: D-grid horizontal divergence (inout)
            mfx: accumulated x mass flux (inout)
            mfy: accumulated y mass flux (inout)
            cx: accumulated Courant number in the x direction (inout)
            cy: accumulated Courant number in the y direction (inout)
            crx: local courant number in the x direction (inout)
            cry: local courant number in the y direction (inout)
            xfx: flux of area in x-direction, in units of m^2 (in)
            yfx: flux of area in y-direction, in units of m^2 (in)
            q_con: total condensate mixing ratio (inout)
            zh: geopotential height defined on layer interfaces (in)
            heat_source:  accumulated heat source (inout)
            diss_est: dissipation estimate (inout)
            dt: acoustic timestep in seconds (in)
        """

        self.fv_prep(uc, vc, crx, cry, xfx, yfx, self._tmp_ut, self._tmp_vt, dt)

        self.fvtp2d_dp(
            delp,
            crx,
            cry,
            xfx,
            yfx,
            self._tmp_fx,
            self._tmp_fy,
        )

        self._flux_capacitor_stencil(
            cx, cy, mfx, mfy, crx, cry, self._tmp_fx, self._tmp_fy
        )

        if not self.hydrostatic:

            self.delnflux_nosg_w(
                w,
                self._tmp_fx2,
                self._tmp_fy2,
                self._delnflux_damp_w,
                self._tmp_wk,
            )

            self._heat_diss_stencil(
                self._tmp_fx2,
                self._tmp_fy2,
                w,
                self.grid.rarea,
                self._tmp_heat_s,
                diss_est,
                self._tmp_dw,
                self._column_namelist["damp_w"],
                self._column_namelist["ke_bg"],
                dt,
            )

            self.fvtp2d_vt(
                w,
                crx,
                cry,
                xfx,
                yfx,
                self._tmp_gx,
                self._tmp_gy,
                None,
                mfx=self._tmp_fx,
                mfy=self._tmp_fy,
            )

            self._flux_adjust_stencil(
                w,
                delp,
                self._tmp_gx,
                self._tmp_gy,
                self.grid.rarea,
            )
        # Fortran: #ifdef USE_COND
        self.fvtp2d_dp_t(
            q_con,
            crx,
            cry,
            xfx,
            yfx,
            self._tmp_gx,
            self._tmp_gy,
            mass=delp,
            mfx=self._tmp_fx,
            mfy=self._tmp_fy,
        )

        self._flux_adjust_stencil(
            q_con, delp, self._tmp_gx, self._tmp_gy, self.grid.rarea
        )

        # Fortran #endif //USE_COND

        self.fvtp2d_tm(
            pt,
            crx,
            cry,
            xfx,
            yfx,
            self._tmp_gx,
            self._tmp_gy,
            mass=delp,
            mfx=self._tmp_fx,
            mfy=self._tmp_fy,
        )

        dt5 = 0.5 * dt
        dt4 = 0.25 * dt

        self._temperature_and_dp_stencil(
            self._tmp_gx,
            self._tmp_gy,
            self.grid.rarea,
            self._tmp_fx,
            self._tmp_fy,
            pt,
            delp,
        )
        if self.grid.south_edge:
            self._vb_south_edge_stencil(self._tmp_vt, self._tmp_vb, dt5)
        self._vb_stencil(vc, uc, self.grid.cosa, self.grid.rsina, self._tmp_vb, dt5)
        if self.grid.west_edge:
            self._vb_west_edge_stencil(self._tmp_vt, self._tmp_vb, dt4)
        if self.grid.east_edge:
            self._vb_east_edge_stencil(self._tmp_vt, self._tmp_vb, dt4)
        if self.grid.north_edge:
            self._vb_north_edge_stencil(self._tmp_vt, self._tmp_vb, dt5)
        self.ytp_v(self._tmp_vb, v, self._tmp_ub)
        self._ke_stencil(self._tmp_ub, self._tmp_vb, self._tmp_ke)

        if self.grid.west_edge:
            self._ub_west_edge_stencil(self._tmp_ut, self._tmp_ub, dt5)
        self._ub_stencil(uc, vc, self.grid.cosa, self.grid.rsina, self._tmp_ub, dt5)
        if self.grid.south_edge:
            self._ub_south_edge_stencil(self._tmp_ut, self._tmp_ub, dt4)
        if self.grid.north_edge:
            self._ub_north_edge_stencil(self._tmp_ut, self._tmp_ub, dt4)
        if self.grid.east_edge:
            self._ub_east_edge_stencil(self._tmp_ut, self._tmp_ub, dt5)
        self.xtp_u(self._tmp_ub, u, self._tmp_vb)
        self._ke_from_bwind_stencil(self._tmp_ke, self._tmp_ub, self._tmp_vb)
        if self.grid.sw_corner:
            self.ke_sw_corner_stencil(
                self._tmp_ke, u, v, self._tmp_ut, self._tmp_vt, dt, 0, 0, -1, 1
            )
        if self.grid.se_corner:
            self.ke_se_corner_stencil(
                self._tmp_ke, u, v, self._tmp_ut, self._tmp_vt, dt, -1, 0, 0, -1
            )
        if self.grid.ne_corner:
            self.ke_ne_corner_stencil(
                self._tmp_ke, u, v, self._tmp_ut, self._tmp_vt, dt, -1, -1, 0, 1
            )
        if self.grid.nw_corner:
            self.ke_nw_corner_stencil(
                self._tmp_ke, u, v, self._tmp_ut, self._tmp_vt, dt, 0, -1, -1, -1
            )
        self._horizontal_vorticity_stencil(
            u,
            v,
            self._tmp_ut,
            self._tmp_vt,
            self.grid.dx,
            self.grid.dy,
            self.grid.rarea,
            self._tmp_wk,
        )

        # TODO if namelist.d_f3d and ROT3 unimplemeneted
        self._adjust_w_and_qcon_stencil(
            w, delp, self._tmp_dw, q_con, self._column_namelist["damp_w"]
        )
        self.divergence_damping(
            u,
            v,
            va,
            ptc,
            self._tmp_vort,
            ua,
            divgd,
            vc,
            uc,
            delpc,
            self._tmp_ke,
            self._tmp_wk,
            dt,
        )

        self._ub_from_vort_stencil(
            self._tmp_vort, self._tmp_ub, self._column_namelist["d_con"]
        )

        self._vb_from_vort_stencil(
            self._tmp_vort, self._tmp_vb, self._column_namelist["d_con"]
        )

        # Vorticity transport
        self._compute_vorticity_stencil(self._tmp_wk, self.grid.f0, zh, self._tmp_vort)

        self.fvtp2d_vt_nodelnflux(
            self._tmp_vort, crx, cry, xfx, yfx, self._tmp_fx, self._tmp_fy
        )

        self._u_from_ke_stencil(
            self._tmp_ke,
            self._tmp_vt,
            self._tmp_fy,
            u,
        )
        self._v_from_ke_stencil(self._tmp_ke, self._tmp_ut, self._tmp_fx, v)

        self.delnflux_nosg_v(
            self._tmp_wk,
            self._tmp_ut,
            self._tmp_vt,
            self._delnflux_damp_vt,
            self._tmp_vort,
        )

        self._heat_source_from_vorticity_damping_stencil(
            self._tmp_ub,
            self._tmp_vb,
            self._tmp_ut,
            self._tmp_vt,
            u,
            v,
            delp,
            self.grid.rsin2,
            self.grid.cosa_s,
            self.grid.rdx,
            self.grid.rdy,
            self._tmp_heat_s,
            self._tmp_dampterm,
            self._column_namelist["d_con"],
        )
        self._heat_source_accumulate(
            self._tmp_heat_s, heat_source, diss_est, self._tmp_dampterm
        )
        self._damp_u(self._tmp_vt, u, self._column_namelist["damp_vt"])
        self._damp_v(self._tmp_ut, v, self._column_namelist["damp_vt"])
