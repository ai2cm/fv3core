import fv3.utils.gt4py_utils as utils
from fv3.utils.corners import fill2_4corners, fill_4corners
import gt4py.gtscript as gtscript
import fv3._config as spec
from gt4py.gtscript import computation, interval, PARALLEL
import fv3.stencils.copy_stencil as cp
from math import log

sd = utils.sd


def grid():
    return spec.grid


def compute(q4, dp1, km, i1, i2, iv, kord):
    pass
