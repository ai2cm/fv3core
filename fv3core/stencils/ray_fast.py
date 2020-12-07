import math

import fv3gfs.util as fv3util
import gt4py.gtscript as gtscript
import numpy as np
from gt4py.gtscript import PARALLEL, computation, interval

import fv3core._config as spec
import fv3core.stencils.c2l_ord as c2l_ord
import fv3core.utils.global_constants as constants
import fv3core.utils.gt4py_utils as utils
from fv3core.decorators import gtstencil
from fv3core.stencils.rayleigh_super import SDAY, compute_rf_vals


FloatField = utils.FloatField
FloatFieldK = utils.FloatFieldK


@gtscript.function
def compute_rff_vals(pfull, dt, rf_cutoff, tau0, ptop):
    rffvals = compute_rf_vals(pfull, dt, rf_cutoff, tau0, ptop)
    rffvals = 1.0 / (1.0 + rffvals)
    return rffvals


@gtscript.function
def dm_layer(rf, dp, wind):
    return (1.0 - rf) * dp * wind


@gtstencil()
def ray_fast_wind(
    u: FloatField,
    v: FloatField,
    w: FloatField,
    rf: FloatField,  # K,
    dp: FloatField,
    dm: FloatField,  # K,
    pfull: FloatField,
    dt: float,
    ptop: float,
    rf_cutoff_nudge: float,
    ks: int,
    hydrostatic: bool,
):
    from __externals__ import i_end, j_end, namelist

    # dm_stencil
    with computation(PARALLEL), interval(...):
        # TODO -- in the fortran model rf is only computed once, repeating
        # the computation every time ray_fast is run is inefficient
        if pfull < namelist.rf_cutoff:
            rf = compute_rff_vals(
                pfull, dt, namelist.rf_cutoff, namelist.tau * SDAY, ptop
            )
    with computation(FORWARD):
        with interval(0, 1):
            # dm = dp if pfull < rf_cutoff_nudge else 0.0  # TODO and kaxes(k) < ks:
            if pfull < rf_cutoff_nudge:  # TODO and kaxes(k) < ks:
                dm = dp
        with interval(1, None):
            dm = dm[0, 0, -1]
            if pfull < rf_cutoff_nudge:  # TODO and kaxes(k) < ks:
                dm += dp
    with computation(BACKWARD), interval(0, -1):
        if pfull < rf_cutoff_nudge:
            dm = dm[0, 0, 1]
    # ray_fast_wind(u)
    with computation(FORWARD):
        with interval(0, 1):
            with parallel(region[: i_end + 1, :]):
                if pfull < namelist.rf_cutoff:
                    dmdir = dm_layer(rf, dp, u)
                    u *= rf
                else:
                    dm = 0
        with interval(1, None):
            with parallel(region[: i_end + 1, :]):
                dmdir = dmdir[0, 0, -1]
                if pfull < namelist.rf_cutoff:
                    dmdir += dm_layer(rf, dp, u)
                    u *= rf
    with computation(BACKWARD), interval(0, -1):
        if pfull < namelist.rf_cutoff:
            dmdir = dmdir[0, 0, 1]
    with computation(PARALLEL), interval(...):
        with parallel(region[: i_end + 1, :]):
            if pfull < rf_cutoff_nudge:  # TODO and axes(k) < ks:
                u += dmdir / dm
    # ray_fast_wind(v)
    with computation(FORWARD):
        with interval(0, 1):
            with parallel(region[:, : j_end + 1]):
                if pfull < namelist.rf_cutoff:
                    dmdir = dm_layer(rf, dp, v)
                    v *= rf
                else:
                    dm = 0
        with interval(1, None):
            with parallel(region[:, : j_end + 1]):
                dmdir = dmdir[0, 0, -1]
                if pfull < namelist.rf_cutoff:
                    dmdir += dm_layer(rf, dp, v)
                    v *= rf
    with computation(BACKWARD), interval(0, -1):
        if pfull < namelist.rf_cutoff:
            dmdir = dmdir[0, 0, 1]
    with computation(PARALLEL), interval(...):
        with parallel(region[:, : j_end + 1]):
            if pfull < rf_cutoff_nudge:  # TODO and axes(k) < ks:
                v += dmdir / dm
    # ray_fast_w
    with computation(PARALLEL), interval(...):
        with parallel(region[: i_end + 1, : j_end + 1]):
            if not hydrostatic and pfull < namelist.rf_cutoff:
                w *= rf


def compute(u, v, w, dp, pfull, dt, ptop, ks):
    grid = spec.grid
    namelist = spec.namelist
    # The next 3 variables and dm_stencil could be pushed into ray_fast_wind and still work, but then recomputing it all twice
    rf_cutoff_nudge = namelist.rf_cutoff + min(100.0, 10.0 * ptop)
    # TODO 1D variable
    shape = (u.shape[0], u.shape[1], u.shape[2] - 1)
    # shape = (u.shape[2] - 1,)
    dm = utils.make_storage_from_shape(shape, grid.default_origin())
    # TODO 1D variable
    rf = utils.make_storage_from_shape(shape, grid.default_origin())

    ray_fast_wind(
        u,
        v,
        w,
        rf,
        dp,
        dm,
        pfull,
        dt,
        ptop,
        rf_cutoff_nudge,
        ks,
        hydrostatic=namelist.hydrostatic,
        origin=grid.compute_origin(),
        domain=(grid.nic + 1, grid.njc + 1, grid.npz),
    )
