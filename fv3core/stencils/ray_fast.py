import dace
import gt4py.gtscript as gtscript
from gt4py.gtscript import (
    __INLINED,
    BACKWARD,
    FORWARD,
    PARALLEL,
    computation,
    interval,
    log,
    sin,
)

import fv3core.utils.global_constants as constants
import fv3core.utils.gt4py_utils as utils
from fv3core.decorators import FrozenStencil, computepath_method
from fv3core.utils.typing import FloatField, FloatFieldK


SDAY = 86400.0

# NOTE: The fortran version of this computes rf in the first timestep only. Then
# rf_initialized let's you know you can skip it. Here we calculate it every
# time.
@gtscript.function
def compute_rf_vals(pfull, bdt, rf_cutoff, tau0, ptop):
    return (
        bdt
        / tau0
        * sin(0.5 * constants.PI * log(rf_cutoff / pfull) / log(rf_cutoff / ptop)) ** 2
    )


@gtscript.function
def compute_rff_vals(pfull, dt, rf_cutoff, tau0, ptop):
    rffvals = compute_rf_vals(pfull, dt, rf_cutoff, tau0, ptop)
    rffvals = 1.0 / (1.0 + rffvals)
    return rffvals


@gtscript.function
def dm_layer(rf, dp, wind):
    return (1.0 - rf) * dp * wind


def dm_compute(
    rf: FloatField,
    dm: FloatField,
    dp: FloatFieldK,
    pfull: FloatFieldK,
    dt: float,
    ptop: float,
    rf_cutoff_nudge: float,
    ks: int,
):
    from __externals__ import rf_cutoff, tau

    # dm_stencil
    with computation(PARALLEL), interval(...):
        # TODO -- in the fortran model rf is only computed once, repeating
        # the computation every time ray_fast is run is inefficient
        if pfull < rf_cutoff:
            rf = compute_rff_vals(pfull, dt, rf_cutoff, tau * SDAY, ptop)
    with computation(FORWARD):
        with interval(0, 1):
            if pfull < rf_cutoff_nudge:  # TODO and kaxes(k) < ks:
                dm = dp
        with interval(1, None):
            dm = dm[0, 0, -1]
            if pfull < rf_cutoff_nudge:  # TODO and kaxes(k) < ks:
                dm += dp
    with computation(BACKWARD), interval(0, -1):
        if pfull < rf_cutoff_nudge:
            dm = dm[0, 0, 1]


def ray_fast_wind(
    wind: FloatField,
    rf: FloatField,
    dm: FloatField,
    dp: FloatFieldK,
    pfull: FloatFieldK,
    dt: float,
    rf_cutoff_nudge: float,
    ks: int,
):
    from __externals__ import rf_cutoff

    with computation(FORWARD):
        with interval(0, 1):
            if pfull < rf_cutoff:
                dmdir = dm_layer(rf, dp, wind)
                wind *= rf
            else:
                dm = 0
        with interval(1, None):
            dmdir = dmdir[0, 0, -1]
            if pfull < rf_cutoff:
                dmdir += dm_layer(rf, dp, wind)
                wind *= rf
    with computation(BACKWARD), interval(0, -1):
        if pfull < rf_cutoff:
            dmdir = dmdir[0, 0, 1]
    with computation(PARALLEL), interval(...):
        if pfull < rf_cutoff_nudge:  # TODO and axes(k) < ks:
            wind += dmdir / dm


def ray_fast_wind_w(
    w: FloatField,
    rf: FloatField,
    pfull: FloatFieldK,
):
    from __externals__ import hydrostatic, rf_cutoff

    with computation(PARALLEL), interval(...):
        if __INLINED(not hydrostatic):
            if pfull < rf_cutoff:
                w *= rf


class RayleighDamping:
    """
    Apply Rayleigh damping (for tau > 0).

    Namelist:
        - tau [Float]: time scale (in days) for Rayleigh friction applied to horizontal
                       and vertical winds; lost kinetic energy is converted to heat,
                       except on nested grids.
        - rf_cutoff [Float]: pressure below which no Rayleigh damping is applied
                             if tau > 0.

    Fotran name: ray_fast.
    """

    def __init__(self, grid, namelist):
        self._rf_cutoff = namelist.rf_cutoff
        origin = grid.compute_origin()
        domain = (grid.nic + 1, grid.njc + 1, grid.npz)

        shape = grid.domain_shape_full(add=(1, 1, 1))
        self._tmp_dm = utils.make_storage_from_shape(shape)

        self._tmp_rf = utils.make_storage_from_shape(shape)
        self._dm_compute = FrozenStencil(
            dm_compute,
            origin=origin,
            domain=domain,
            externals={
                "rf_cutoff": namelist.rf_cutoff,
                "tau": namelist.tau,
            },
        )
        self._ray_fast_u = FrozenStencil(
            ray_fast_wind,
            origin=origin,
            domain=grid.domain_shape_compute(add=(0, 1, 0)),
            externals={
                "rf_cutoff": namelist.rf_cutoff,
            },
        )
        self._ray_fast_v = FrozenStencil(
            ray_fast_wind,
            origin=origin,
            domain=grid.domain_shape_compute(add=(1, 0, 0)),
            externals={
                "rf_cutoff": namelist.rf_cutoff,
            },
        )
        self._ray_fast_w = FrozenStencil(
            ray_fast_wind_w,
            origin=origin,
            domain=grid.domain_shape_compute(),
            externals={
                "rf_cutoff": namelist.rf_cutoff,
                "hydrostatic": namelist.hydrostatic,
            },
        )

    @computepath_method
    def __call__(
        self,
        u,
        v,
        w,
        dp,
        pfull,
        dt: float,
        ptop: float,
        ks: int,
    ):
        rf_cutoff_nudge = self._rf_cutoff + min(100.0, 10.0 * ptop)
        self._dm_compute(
            self._tmp_rf, self._tmp_dm, dp, pfull, dt, ptop, rf_cutoff_nudge, ks
        )
        self._ray_fast_u(
            u,
            self._tmp_rf,
            self._tmp_dm,
            dp,
            pfull,
            dt,
            rf_cutoff_nudge,
            ks,
        )
        self._ray_fast_v(
            v,
            self._tmp_rf,
            self._tmp_dm,
            dp,
            pfull,
            dt,
            rf_cutoff_nudge,
            ks,
        )
        self._ray_fast_w(w, self._tmp_rf, pfull)
