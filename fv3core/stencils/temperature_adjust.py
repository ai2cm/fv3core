from gt4py.gtscript import PARALLEL, computation, exp, interval, log

import fv3core._config as spec
import fv3core.utils.global_constants as constants
from fv3core.decorators import gtstencil
from fv3core.stencils.basic_operations import sign
from fv3core.utils.typing import FloatField


@gtstencil()
def compute_pkz_tempadjust(
    delp: FloatField,
    delz: FloatField,
    cappa: FloatField,
    heat_source: FloatField,
    pt: FloatField,
    pkz: FloatField,
    delt_factor: float,
):
    with computation(PARALLEL):
        with interval(0, 1):
            delt = delt_factor * 0.1
        with interval(1, 2):
            delt = delt_factor * 0.5
        with interval(2, None):
            delt = delt_factor
    with computation(PARALLEL), interval(...):
        pkz = exp(cappa / (1.0 - cappa) * log(constants.RDG * delp / delz * pt))
        pkz = (constants.RDG * delp / delz * pt) ** (cappa / (1.0 - cappa))
        dtmp = heat_source / (constants.CV_AIR * delp)
        deltmin = sign(min(delt, abs(dtmp)), dtmp)
        pt = pt + deltmin / pkz


# TODO use stencils. limited by functions exp, log and variable that depends on k
def compute(
    pt: FloatField,
    pkz: FloatField,
    heat_source: FloatField,
    delz: FloatField,
    delp: FloatField,
    cappa: FloatField,
    n_con: int,
    bdt: float,
):
    grid = spec.grid
    compute_pkz_tempadjust(
        delp,
        delz,
        cappa,
        heat_source,
        pt,
        pkz,
        abs(bdt * spec.namelist.delt_max),
        origin=grid.compute_origin(),
        domain=(grid.nic, grid.njc, n_con),
    )
