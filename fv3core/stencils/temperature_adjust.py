from gt4py.gtscript import PARALLEL, computation, exp, interval, log

import fv3core._config as spec
import fv3core.utils.global_constants as constants
import fv3core.utils.gt4py_utils as utils
from fv3core.decorators import gtstencil
from fv3core.stencils.basic_operations import sign


sd = utils.sd


@gtstencil()
def compute_pkz_tempadjust(
    delp: sd,
    delz: sd,
    cappa: sd,
    heat_source: sd,
    pt: sd,
    pkz: sd,
    delt_time_factor: float,
):
    with computation(PARALLEL):
        with interval(...):
            pkz = exp(cappa / (1.0 - cappa) * log(constants.RDG * delp / delz * pt))
            pkz = (constants.RDG * delp / delz * pt) ** (cappa / (1.0 - cappa))
            dtmp = heat_source / (constants.CV_AIR * delp)
        with interval(0, 1):
            deltmin = sign(min(delt_time_factor * 0.1, abs(dtmp)), dtmp)
            pt = pt + deltmin / pkz
        with interval(1, 2):
            deltmin = sign(min(delt_time_factor * 0.5, abs(dtmp)), dtmp)
            pt = pt + deltmin / pkz
        with interval(2, None):
            deltmin = sign(min(delt_time_factor, abs(dtmp)), dtmp)
            pt = pt + deltmin / pkz


# TODO use stencils. limited by functions exp, log and variable that depends on k
def compute(pt, pkz, heat_source, delz, delp, cappa, n_con, bdt):
    grid = spec.grid
    delt_time_factor = abs(bdt * spec.namelist.delt_max)
    compute_pkz_tempadjust(
        delp,
        delz,
        cappa,
        heat_source,
        pt,
        pkz,
        delt_time_factor,
        origin=grid.compute_origin(),
        domain=(grid.nic, grid.njc, n_con),
    )
