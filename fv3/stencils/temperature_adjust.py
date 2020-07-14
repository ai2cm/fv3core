#!/usr/bin/env python3
import fv3.utils.gt4py_utils as utils
import gt4py.gtscript as gtscript
import fv3._config as spec
import fv3.utils.global_constants as constants
from fv3.stencils.basic_operations import absolute_value, min_fn
from gt4py.gtscript import computation, interval, PARALLEL
import numpy as np
import math

sd = utils.sd


@utils.stencil()
def compute_pkz_tempadjust(
    delp: sd, delz: sd, cappa: sd, heat_source: sd, delt: sd, pt: sd, pkz: sd
):
    with computation(PARALLEL), interval(...):
        pkz = (constants.RDG * delp / delz * pt) ** (cappa / (1.0 - cappa))
        dtmp = heat_source / (constants.CV_AIR * delp)
        abs_dtmp = absolute_value(dtmp)
        deltmin = min_fn(delt, abs_dtmp) * dtmp / abs_dtmp
        pt = pt + deltmin / pkz


# TODO use stencils. limited by functions exp, log and variable that depends on k
def compute(pt, pkz, heat_source, delz, delp, cappa, n_con, bdt):
    grid = spec.grid
    delt_column = np.ones(delz.shape[2]) * abs(bdt * spec.namelist["delt_max"])
    delt_column[0] *= 0.1
    delt_column[1] *= 0.5
    delt = utils.make_storage_data_from_1d(
        delt_column, delz.shape, origin=grid.default_origin()
    )
    """
    isl = slice(grid.is_, grid.ie + 1)
    jsl = slice(grid.js, grid.je + 1)
    ksl = slice(0, n_con)
    pkz[isl, jsl, ksl] = np.exp(
        cappa[isl, jsl, ksl]
        / (1 - cappa[isl, jsl, ksl])
        * np.log(
            constants.RDG
            * delp[isl, jsl, ksl]
            / delz[isl, jsl, ksl]
            * pt[isl, jsl, ksl]
        )
    )
    """
    compute_pkz_tempadjust(
        delp,
        delz,
        cappa,
        heat_source,
        delt,
        pt,
        pkz,
        origin=grid.compute_origin(),
        domain=(grid.nic, grid.njc, n_con),
    )
