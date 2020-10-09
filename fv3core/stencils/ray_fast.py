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
def dm_adjusted_wind(wind, dmwind, dm, pfull, rf_cutoff_nudge, kindex, ks):
    if pfull < rf_cutoff_nudge and kindex < ks:
        dmwind = dmwind / dm
        wind = wind + dmwind
    return wind, dmwind

@utils.stencil()
def ray_fast_u(u: sd, rf: sd, dp: sd, dmu: sd, pfull: sd):
    with computation(FORWARD):
        with interval(0, 1):
            if pfull < spec.namelist.rf_cutoff:
                dmu = (1.0 - rf) * dp * u
                u = rf * u
            else:
                dmu = 0.
        with interval(1, None):
            if pfull < spec.namelist.rf_cutoff:
                dmu = dmu[0, 0, -1] + (1.0 - rf) * dp * u
                u = rf * u
            else:
                dmu = dmu[0, 0, -1]
    with computation(BACKWARD), interval(0, -1):
        if pfull < spec.namelist.rf_cutoff:
            dmu = dmu[0, 0, 1]

@utils.stencil()
def ray_fast_v(
        v: sd, rf: sd, dp: sd, dmv: sd, pfull: sd
):
    with computation(FORWARD):
        with interval(0, 1):
            if pfull < spec.namelist.rf_cutoff:
                dmv = (1.0 - rf) * dp * v
                v = rf * v
            else:
                dmv = 0.
        with interval(1, None):
            if pfull < spec.namelist.rf_cutoff:
                dmv = dmv[0, 0, -1] + (1.0 - rf) * dp * v
                v = rf * v
            else:
                dmv = dmv[0, 0, -1]
    with computation(BACKWARD), interval(0, -1):
        if pfull < spec.namelist.rf_cutoff:
            dmv = dmv[0, 0, 1]

@utils.stencil()
def ray_fast_w(w: sd, rf: sd, pfull: sd):
    with computation(PARALLEL), interval(...):
        if pfull < spec.namelist.rf_cutoff:
            w = rf * w

@utils.stencil()
def ray_fast_horizontal_dm(u: sd, v:sd,  dmwind_u: sd, dmwind_v: sd, dm: sd, pfull: sd, kindex: sd, rf_cutoff_nudge: float, ks: int):
    with computation(PARALLEL), interval(...):
        u, dmwind_u = dm_adjusted_wind(u, dmwind_u, dm, pfull, rf_cutoff_nudge, kindex, ks)
        v, dmwind_v = dm_adjusted_wind(v, dmwind_v, dm, pfull, rf_cutoff_nudge, kindex, ks)


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

    dm_stencil(
        dp, dm, pfull, rf, kindex, dt, ptop, rf_cutoff_nudge, ks, origin=grid.compute_origin(), domain=(grid.nic + 1, grid.njc + 1, grid.npz)
    )
    ray_fast_u(
        u,
        rf,
        dp,
        dmu,
        pfull,
        origin=grid.compute_origin(),
        domain=(grid.nic, grid.njc + 1, grid.npz),
    )
    ray_fast_v(
        v,
        rf,
        dp,
        dmv,
        pfull,
        origin=grid.compute_origin(),
        domain=(grid.nic + 1, grid.njc, grid.npz),
    )
    ray_fast_horizontal_dm(
        u, v, dmu, dmv, dm, pfull, kindex, rf_cutoff_nudge, ks, origin=grid.compute_origin(), domain=(grid.nic+1, grid.njc + 1, grid.npz)
    )
  
    if not spec.namelist.hydrostatic:
        ray_fast_w(
            w, rf, pfull, origin=grid.compute_origin(), domain=(grid.nic, grid.njc, grid.npz),
        )


    '''
# Doing it all in one stencil (does not yet validate, and DynCore does not validate when it is used)
@utils.stencil()
def rayfast(u: sd, v: sd, w: sd, dp: sd, pfull: sd, kindex:sd, bdt: float, ptop: float,  ks: int):
    from __splitters__ import i_start, i_end, j_end, j_start
    with computation(PARALLEL), interval(...):
        rf_cutoff_nudge = compute_rf_nudged_cutoff(ptop)
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
    with computation(FORWARD):
        with interval(0, 1):
            with parallel(region[i_start:i_end+1, j_start:j_end+2]):
                if pfull < spec.namelist.rf_cutoff:
                    dmu = (1.0 - rf) * dp * u
                    u = rf * u
                else:
                    dmu = 0.
            with parallel(region[i_start:i_end+2, j_start:j_end+1]):
                if pfull < spec.namelist.rf_cutoff:
                    dmv = (1.0 - rf) * dp * v
                    v = rf * v
                else:
                    dmv = 0.
        with interval(1, None):
            with parallel(region[i_start:i_end+1, j_start:j_end+2]):
                if pfull < spec.namelist.rf_cutoff:
                    dmu = dmu[0, 0, -1] + (1.0 - rf) * dp * u
                    u = rf * u
                else:
                    dmu = dmu[0, 0, -1]
            with parallel(region[i_start:i_end+2, j_start:j_end+1]):
                if pfull < spec.namelist.rf_cutoff:
                    dmv = dmv[0, 0, -1] + (1.0 - rf) * dp * v
                    v = rf * v
                else:
                    dmv = dmv[0, 0, -1]
    with computation(BACKWARD), interval(0, -1):
        with parallel(region[i_start:i_end+1, j_start:j_end+2]):
            if pfull < spec.namelist.rf_cutoff:
                dmu = dmu[0, 0, 1]
        with parallel(region[i_start:i_end+2, j_start:j_end+1]):
            if pfull < spec.namelist.rf_cutoff:
                dmv = dmv[0, 0, 1]
    with computation(PARALLEL), interval(...):
        u, dmu = dm_adjusted_wind(u, dmu, dm, pfull, rf_cutoff_nudge, kindex, ks)
        v, dmv = dm_adjusted_wind(v, dmv, dm, pfull, rf_cutoff_nudge, kindex, ks)
        with parallel(region[i_start:i_end+1, j_start:j_end+1]):
            if pfull < spec.namelist.rf_cutoff:
                w = rf * w

def compute(u, v, w, dp, pfull, dt, ptop, ks):
    grid = spec.grid
    # TODO get rid of this when we can refer to the index in the stencil
    kindex = utils.make_storage_data(np.squeeze(np.indices((u.shape[2],))), u.shape)
    rayfast(u, v, w, dp, pfull, kindex, dt, ptop, ks,  origin=grid.compute_origin(), domain=(grid.nic+1, grid.njc+1, grid.npz), splitters=grid.splitters)
    '''        
