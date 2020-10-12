#!/usr/bin/env python3
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


sd = utils.sd


@gtscript.function
def compute_rf_nudged_cutoff(ptop):
    return spec.namelist.rf_cutoff + min(100.0, 10.0 * ptop)


@gtscript.function
def compute_rff_vals(pfull, dt, rf_cutoff, tau0, ptop):
    rffvals = compute_rf_vals(pfull, dt, rf_cutoff, tau0, ptop)
    rffvals = 1.0 / (1.0 + rffvals)
    return rffvals


@gtscript.function
def dm_adjusted_wind(dmwind, dm, wind, pfull, rf_cutoff_nudge, kindex, ks):
    if pfull < rf_cutoff_nudge and kindex < ks:
        dmwind = dmwind / dm
        wind = wind + dmwind
    return dmwind, wind


@gtscript.function
def dm_layer(rf, dp, wind):
    return (1.0 - rf) * dp * wind


@gtstencil()
def dm_stencil(
    dp: sd,
    dm: sd,
    pfull: sd,
    rf: sd,
    kindex: sd,
    dt: float,
    ptop: float,
    rf_cutoff_nudge: float,
    ks: int,
):
    with computation(PARALLEL), interval(...):
        if pfull < spec.namelist.rf_cutoff:
            rf = compute_rff_vals(
                pfull, dt, spec.namelist.rf_cutoff, spec.namelist.tau * SDAY, ptop
            )
    with computation(FORWARD):
        with interval(0, 1):
            if pfull < rf_cutoff_nudge and kindex < ks:
                dm = dp
            else:
                dm = 0.0
        with interval(1, None):
            if pfull < rf_cutoff_nudge and kindex < ks:
                dm = dm[0, 0, -1] + dp
            else:
                dm = dm[0, 0, -1]
    with computation(BACKWARD), interval(0, -1):
        if pfull < rf_cutoff_nudge:
            dm = dm[0, 0, 1]


@gtstencil()
def ray_fast_wind(
    wind: sd,
    rf: sd,
    dp: sd,
    dm: sd,
    pfull: sd,
    kindex: sd,
    rf_cutoff_nudge: float,
    ks: int,
):
    with computation(FORWARD):
        with interval(0, 1):
            if pfull < spec.namelist.rf_cutoff:
                dmdir = dm_layer(rf, dp, wind)
                wind = rf * wind
            else:
                dm = 0
        with interval(1, None):
            if pfull < spec.namelist.rf_cutoff:
                dmdir = dmdir[0, 0, -1] + dm_layer(rf, dp, wind)
                wind = rf * wind
            else:
                dmdir = dmdir[0, 0, -1]
    with computation(BACKWARD), interval(0, -1):
        if pfull < spec.namelist.rf_cutoff:
            dmdir = dmdir[0, 0, 1]
    with computation(PARALLEL), interval(...):
        dmdir, wind = dm_adjusted_wind(
            dmdir, dm, wind, pfull, rf_cutoff_nudge, kindex, ks
        )


@gtstencil()
def ray_fast_w(w: sd, rf: sd, pfull: sd):
    with computation(PARALLEL), interval(...):
        if pfull < spec.namelist.rf_cutoff:
            w = rf * w


def compute(u, v, w, dp, pfull, dt, ptop, ks):
    grid = spec.grid

    # TODO get rid of this when we can refer to the index in the stencil
    kindex = utils.make_storage_data(np.squeeze(np.indices((u.shape[2],))), u.shape)
    # The next 3 variables and dm_stencil could be pushed into ray_fast_wind and still work, but then recomputing it all twice
    rf_cutoff_nudge = spec.namelist.rf_cutoff + min(100.0, 10.0 * ptop)
    dm = utils.make_storage_from_shape(u.shape, grid.default_origin())
    rf = utils.make_storage_from_shape(u.shape, grid.default_origin())
    dm_stencil(
        dp,
        dm,
        pfull,
        rf,
        kindex,
        dt,
        ptop,
        rf_cutoff_nudge,
        ks,
        origin=grid.compute_origin(),
        domain=(grid.nic + 1, grid.njc + 1, grid.npz),
    )
    ray_fast_wind(
        u,
        rf,
        dp,
        dm,
        pfull,
        kindex,
        rf_cutoff_nudge,
        ks,
        origin=grid.compute_origin(),
        domain=(grid.nic, grid.njc + 1, grid.npz),
    )
    ray_fast_wind(
        v,
        rf,
        dp,
        dm,
        pfull,
        kindex,
        rf_cutoff_nudge,
        ks,
        origin=grid.compute_origin(),
        domain=(grid.nic + 1, grid.njc, grid.npz),
    )

    if not spec.namelist.hydrostatic:
        ray_fast_w(
            w,
            rf,
            pfull,
            origin=grid.compute_origin(),
            domain=(grid.nic, grid.njc, grid.npz),
        )
