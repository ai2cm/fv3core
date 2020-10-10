#!/usr/bin/env python3
import fv3core.utils.gt4py_utils as utils
import gt4py.gtscript as gtscript
import fv3core._config as spec
import fv3core.stencils.c2l_ord as c2l_ord
from gt4py.gtscript import computation, interval, PARALLEL
import fv3core.utils.global_constants as constants
from fv3core.stencils.rayleigh_super import compute_rf_vals, SDAY
import numpy as np
import math
import fv3gfs.util as fv3util

sd = utils.sd

@gtscript.function
def compute_rf_nudged_cutoff(ptop):
    return spec.namelist.rf_cutoff + min(100.0, 10.0 * ptop)

@gtscript.function
def compute_rff_vals(pfull, bdt, rf_cutoff, tau0, ptop):
    rffvals = compute_rf_vals(pfull, bdt, rf_cutoff, tau0, ptop)
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

@gtscript.function
def dm_aggregate(dm, rf, dp, wind):
    return dm[0, 0, -1] + dm_layer(rf, dp, wind)

@gtscript.function
def dm_and_wind_layer(dm, wind, rf, dp, pfull):
    if pfull < spec.namelist.rf_cutoff:
        dm = dm_layer(rf, dp, wind)
        wind = rf * wind
    else:
        dm = 0.
    return dm, wind

@gtscript.function
def dm_and_wind_update(dm, wind, rf, dp, pfull):
    if pfull < spec.namelist.rf_cutoff:
        dm = dm_aggregate(dm, rf, dp, wind) 
        wind = rf * wind
    else:
        dm = dm[0, 0, -1]
    return dm, wind
  

@utils.stencil()
def ray_fast_wind(wind: sd, rf: sd, dp: sd, dmdir: sd, dm: sd, pfull: sd, kindex: sd, rf_cutoff_nudge: float, ks: int):
    with computation(FORWARD):
        with interval(0, 1):
            dmdir, wind = dm_and_wind_layer(dmdir, wind, rf, dp, pfull)
        with interval(1, None):
            # TODO why does this not validate
            #  dmdir, wind = dm_and_wind_update(dmdir, wind, rf, dp, pfull)
            if pfull < spec.namelist.rf_cutoff:
                dmdir = dm_aggregate(dmdir, rf, dp, wind)
                wind = rf * wind
            else:
                dmdir = dmdir[0, 0, -1]
    with computation(BACKWARD), interval(0, -1):
        if pfull < spec.namelist.rf_cutoff:
            dmdir = dmdir[0, 0, 1]
    with computation(PARALLEL), interval(...):
        dmdir, wind = dm_adjusted_wind(dmdir, dm, wind, pfull, rf_cutoff_nudge, kindex, ks)

@utils.stencil()
def ray_fast_w(w: sd, rf: sd, pfull: sd):
    with computation(PARALLEL), interval(...):
        if pfull < spec.namelist.rf_cutoff:
            w = rf * w

@utils.stencil()
def dm_stencil(dp: sd, dm: sd, pfull: sd, rf: sd, kindex: sd, bdt: float,ptop: float, rf_cutoff_nudge: float, ks: int):
    with computation(PARALLEL), interval(...):
        if pfull < spec.namelist.rf_cutoff:
            rf = compute_rff_vals(pfull, bdt, spec.namelist.rf_cutoff, spec.namelist.tau * SDAY, ptop)
    with computation(FORWARD):
        with interval(0, 1):
            if pfull < rf_cutoff_nudge and kindex < ks:
                dm = dp
            else:
                dm = 0.
        with interval(1, None):
            if pfull < rf_cutoff_nudge and kindex < ks:
                dm = dm[0, 0, -1] + dp
            else:
                dm = dm[0, 0, -1]
    with computation(BACKWARD), interval(0, -1):
        if pfull < rf_cutoff_nudge:
            dm = dm[0, 0, 1]
   

def compute(u, v, w, dp, pfull, dt, ptop, ks):
    grid = spec.grid
   
    rf = utils.make_storage_from_shape(u.shape, grid.default_origin())
    # TODO get rid of this when we can refer to the index in the stencil
    kindex = utils.make_storage_data(np.squeeze(np.indices((u.shape[2],))), u.shape)
    rf_cutoff_nudge = spec.namelist.rf_cutoff + min(100.0, 10.0 * ptop)
    dm = utils.make_storage_from_shape(u.shape, grid.default_origin())
    dmu = utils.make_storage_from_shape(u.shape, grid.default_origin())
    dmv = utils.make_storage_from_shape(u.shape, grid.default_origin())
    # This could be pushed into ray_fast_wind and still work, but then computing dm and rf twice
    dm_stencil(
        dp, dm, pfull, rf, kindex, dt, ptop, rf_cutoff_nudge, ks, origin=grid.compute_origin(), domain=(grid.nic + 1, grid.njc + 1, grid.npz)
    )
    ray_fast_wind(
        u,
        rf,
        dp,
        dmu,dm,
        pfull, kindex, rf_cutoff_nudge, ks,
        origin=grid.compute_origin(),
        domain=(grid.nic, grid.njc + 1, grid.npz),
    )
    ray_fast_wind(
        v,
        rf,
        dp,
        dmv, dm,
        pfull, kindex, rf_cutoff_nudge, ks,
        origin=grid.compute_origin(),
        domain=(grid.nic + 1, grid.njc, grid.npz),
    )
   
    if not spec.namelist.hydrostatic:
        ray_fast_w(
            w, rf, pfull, origin=grid.compute_origin(), domain=(grid.nic, grid.njc, grid.npz),
        )

