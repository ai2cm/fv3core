#!/usr/bin/env python3
import fv3core.utils.gt4py_utils as utils
import gt4py.gtscript as gtscript
import fv3core._config as spec
import fv3core.stencils.c2l_ord as c2l_ord
from gt4py.gtscript import computation, interval, PARALLEL
import fv3core.utils.global_constants as constants
from fv3core.stencils.rayleigh_super import compute_rf_vals, RCV
import numpy as np
import math
import fv3gfs.util as fv3util

sd = utils.sd
SDAY = 86400.0  # seconds per day
U000 = 4900.0  # scaling velocity


@utils.stencil()
def initialize_u2f_friction(pfull: sd, ua: sd, va: sd, w: sd, u2f: sd, hydrostatic: bool):
    with computation(PARALLEL), interval(...):
        if pfull < spec.namelist.rf_cutoff:
            if hydrostatic:
                u2f = ua ** 2 + va ** 2
            else:
                u2f = ua ** 2 + va ** 2 + w ** 2


@utils.stencil()
def rayleigh_pt_friction(
    pt: sd,
    pfull: sd,
    u2f: sd,
    delz: sd,
    bdt: float,
    tau0: float,
    ptop: float,
):
    with computation(PARALLEL), interval(...):
        if pfull < spec.namelist.rf_cutoff:
            if spec.namelist.hydrostatic:
                pt = pt + 0.5 * u2f / (
                    constants.CP_AIR - constants.RDGAS * ptop / pfull
                ) * (1.0 - 1.0 / (1.0 + rk * (u2f / U000) ** 0.5))
            else:
                rf = compute_rf_vals(pfull, bdt, spec.namelist.rf_cutoff, tau0, ptop)
                delz = delz / pt
                pt = pt + 0.5 * u2f * RCV * (
                1.0 - 1.0 / (1.0 + rf * (u2f / U000) ** 0.5)
                )
            if not hydrostatic:
                w = w / (1.0 + u2f)


@utils.stencil()
def update_u2f(u2f: sd, rf: sd):
    with computation(PARALLEL), interval(...):
        if pfull < spec.namelist.rf_cutoff:
            u2f = rf * (u2f / U000) ** 0.5


@utils.stencil()
def rayleigh_u_friction(u: sd, pfull: sd, u2f: sd):
    with computation(PARALLEL), interval(...):
        if pfull < spec.namelist.rf_cutoff:
            u = u / (1.0 + 0.5 * (u2f[0, -1, 0] + u2f))


@utils.stencil()
def rayleigh_v_friction(v: sd, pfull: sd, u2f: sd):
    with computation(PARALLEL), interval(...):
        if pfull < spec.namelist.rf_cutoff:
            v = v / (1.0 + 0.5 * (u2f[-1, 0, 0] + u2f))


def compute(u, v, w, ua, va, pt, delz, phis, bdt, ptop, pfull, comm):
    grid = spec.grid
    c2l_ord.compute_ord2(u, v, ua, va)
    raise NotImplementedError('Rayleight Friction code is untested')
    # TODO this really only needs to be kmax size in the 3rd dimension...
    u2f = grid.quantity_factory.zeros(
        [fv3util.X_DIM, fv3util.Y_DIM, fv3util.Z_DIM], "m/s"
    )

    initialize_u2f_friction(
        pfull,
        ua,
        va,
        w,
        u2f.storage,
        spec.namelist.hydrostatic,
        origin=grid.compute_origin(),
        domain=grid.compute_domain(),
    )

    comm.halo_update(u2f, n_points=utils.halo)
    rf = utils.make_storage_from_shape(pt.shape, utils.origin())
    rayleigh_pt_friction(
        pt,
        rf,
        pfull,
        u2f.storage,
        delz,
        bdt,
        spec.namelist.tau * SDAY,
        ptop,
        origin=grid.compute_origin(),
        domain=grid.compute_domain(),
    )
    update_u2f(
        u2f.storage,
        rf,
        origin=(grid.is_ - 1, grid.js - 1, 0),
        domain=(grid.nic + 2, grid.njc + 2, grid.npz),
    )
    rayleigh_u_friction(
        u,
        u2f.storage,
        origin=grid.compute_origin(),
        domain=(grid.nic, grid.njc + 1, grid.npz),
    )
    rayleigh_v_friction(
        v,
        u2f.storage,
        origin=grid.compute_origin(),
        domain=(grid.nic + 1, grid.njc, grid.npz),
    )
